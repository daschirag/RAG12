"""A3 - Forecasting models for 1-hour-ahead plant AC power.

All models share one interface:

    model.fit(train, val, frame)          # SupervisedSet splits + A1 frame
    model.predict(data, frame) -> ndarray # one value per row of data.X

``frame`` is the A1 15-min frame; only the sequence models read it (to build
input windows ending at each origin). Hyperparameters are chosen on ``val``
only. Missing inputs are filled with statistics of the training split only
(``TrainOnlyImputer``), so no row is ever dropped and every model is scored
on the same rows. ``postprocess`` is applied identically to every model.
"""

from __future__ import annotations

import copy
import itertools
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .features import SupervisedSet
from .metrics import mae
from .preprocess import TARGET, PreprocessConfig

_PCFG = PreprocessConfig()
PLANT_CAPACITY_KW = _PCFG.inverter_ac_max_kw * _PCFG.n_inverters


# ------------------------------------------------------------------ shared pieces

class TrainOnlyImputer:
    """Fill NaN with the training median for the same hour (fallback: global median)."""

    def __init__(self, group_col: str = "hour") -> None:
        self.group_col = group_col
        self.by_group_: pd.DataFrame | None = None
        self.global_: pd.Series | None = None

    def fit(self, X: pd.DataFrame) -> "TrainOnlyImputer":
        self.by_group_ = X.groupby(self.group_col).median(numeric_only=True)
        self.global_ = X.median(numeric_only=True)
        if self.global_.isna().any():
            empty = list(self.global_[self.global_.isna()].index)
            raise ValueError(f"Columns with no training values: {empty}")
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.by_group_ is None or self.global_ is None:
            raise RuntimeError("TrainOnlyImputer.transform called before fit")
        group_fill = self.by_group_.reindex(X[self.group_col].to_numpy())
        group_fill.index = X.index
        # The group column itself is the groupby key (never NaN), so reindex rather than select.
        out = X.fillna(group_fill.reindex(columns=X.columns))
        return out.fillna(self.global_.reindex(X.columns))


def eval_mask(data: SupervisedSet) -> np.ndarray:
    """Rows that are scored: targets that were observed (not night-filled by A1)."""
    return ~data.meta["target_imputed"].to_numpy()


# Night-zeroing rule: predictions are set to 0 when the prior-days irradiation
# envelope at T is <= this threshold (NaN envelope counts as night). None
# disables the rule. The strict rule (0.0) was chosen in A4 on validation data
# only and is used everywhere from A5 on (A3 originally used 0.05, which equals
# ``~meta["target_daytime"]``).
DEFAULT_NIGHT_THRESHOLD: float | None = 0.0


def night_mask(data: SupervisedSet, threshold: float | None = DEFAULT_NIGHT_THRESHOLD) -> np.ndarray:
    """True where the night rule zeroes the forecast."""
    if threshold is None:
        return np.zeros(len(data.X), dtype=bool)
    return ~(data.X["irr_envelope_target"].to_numpy() > threshold)


def postprocess(
    pred: np.ndarray,
    data: SupervisedSet,
    night_threshold: float | None = DEFAULT_NIGHT_THRESHOLD,
    capacity: float = PLANT_CAPACITY_KW,
) -> np.ndarray:
    """Clip to [0, capacity]; 0 where the night rule applies (None = no night rule, e.g. wind)."""
    out = np.clip(np.asarray(pred, dtype=float), 0.0, capacity)
    out[night_mask(data, night_threshold)] = 0.0
    return out


def _val_mae(
    model_pred: np.ndarray,
    val: SupervisedSet,
    night_threshold: float | None = DEFAULT_NIGHT_THRESHOLD,
    capacity: float = PLANT_CAPACITY_KW,
) -> float:
    m = eval_mask(val)
    return mae(val.y.to_numpy()[m], postprocess(model_pred, val, night_threshold, capacity)[m])


# ---------------------------------------------------------------------- baselines

class Persistence:
    """y_hat(t + 1h) = y(t). Missing y(t) -> train-only imputed value."""

    name = "persistence"
    column = "ac_now"

    def fit(self, train: SupervisedSet, val: SupervisedSet, frame: pd.DataFrame) -> "Persistence":
        self.imputer_ = TrainOnlyImputer().fit(train.X)
        self.params_: dict = {}
        return self

    def predict(self, data: SupervisedSet, frame: pd.DataFrame) -> np.ndarray:
        return self.imputer_.transform(data.X)[self.column].to_numpy()


class DailyPersistence(Persistence):
    """y_hat(T) = y(T - 24h)."""

    name = "daily_persistence"
    column = "ac_same_slot_yesterday"


# ---------------------------------------------------------------------- tree models

class _TabularModel:
    name = "tabular"
    grid: dict[str, list] = {}

    def __init__(
        self,
        seed: int = 0,
        features: list[str] | None = None,
        night_threshold: float | None = DEFAULT_NIGHT_THRESHOLD,
        capacity: float = PLANT_CAPACITY_KW,
    ) -> None:
        self.seed = seed
        self.features = features
        self.night_threshold = night_threshold
        self.capacity = capacity

    def _cols(self, data: SupervisedSet) -> list[str]:
        return self.features or data.feature_names

    def _make(self, params: dict):
        raise NotImplementedError

    def _fit_one(self, params: dict, Xtr, ytr, Xva, yva):
        model = self._make(params)
        model.fit(Xtr, ytr)
        return model

    def fit(self, train: SupervisedSet, val: SupervisedSet, frame: pd.DataFrame):
        cols = self._cols(train)
        self.imputer_ = TrainOnlyImputer().fit(train.X)
        Xtr = self.imputer_.transform(train.X)[cols]
        Xva = self.imputer_.transform(val.X)[cols]
        best = None
        keys = sorted(self.grid)
        for values in itertools.product(*(self.grid[k] for k in keys)):
            params = dict(zip(keys, values))
            model = self._fit_one(params, Xtr, train.y, Xva, val.y)
            score = _val_mae(model.predict(Xva), val, self.night_threshold, self.capacity)
            if best is None or score < best[0]:
                best = (score, params, model)
        self.val_mae_, self.params_, self.model_ = best
        self.params_ = {**self.params_, **self._extra_params()}
        return self

    def _extra_params(self) -> dict:
        return {}

    def predict(self, data: SupervisedSet, frame: pd.DataFrame) -> np.ndarray:
        X = self.imputer_.transform(data.X)[self._cols(data)]
        return self.model_.predict(X)


class RandomForest(_TabularModel):
    name = "random_forest"
    n_estimators = 300
    grid = {"max_depth": [None, 12], "min_samples_leaf": [1, 5], "max_features": [1.0, 0.5]}

    def _make(self, params: dict):
        from sklearn.ensemble import RandomForestRegressor

        return RandomForestRegressor(n_estimators=self.n_estimators, random_state=self.seed, n_jobs=-1, **params)

    def _fit_one(self, params, Xtr, ytr, Xva, yva):
        model = super()._fit_one(params, Xtr, ytr, Xva, yva)
        # Parallel predict sums tree outputs in thread-completion order, which changes
        # the last bits between runs; predict single-threaded for exact reproducibility.
        model.set_params(n_jobs=1)
        return model


class XGBoost(_TabularModel):
    name = "xgboost"
    grid = {"max_depth": [3, 5], "learning_rate": [0.03, 0.1]}

    def _make(self, params: dict):
        from xgboost import XGBRegressor

        return XGBRegressor(
            n_estimators=2000, subsample=0.8, colsample_bytree=0.8, random_state=self.seed,
            early_stopping_rounds=50, eval_metric="mae", tree_method="hist", n_jobs=4, **params,
        )

    def _fit_one(self, params, Xtr, ytr, Xva, yva):
        model = self._make(params)
        model.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
        return model

    def _extra_params(self) -> dict:
        return {"best_iteration": int(self.model_.best_iteration)}


# ------------------------------------------------------------------ sequence models

SEQ_COLUMNS = (TARGET, "irradiation", "ambient_temp", "module_temp")   # solar defaults
CONTEXT_COLS = ("tod_sin", "tod_cos", "irr_envelope_target", "ac_same_slot_yesterday",
                "irr_same_slot_yesterday")


@dataclass
class SequenceEncoder:
    """Builds (window, context) tensors; all statistics come from the training period."""

    lookback: int = 24
    # Frame columns read per slot (first = target) and context columns from X.
    # The first two columns also get missing-value indicator channels.
    seq_columns: tuple[str, ...] = SEQ_COLUMNS
    context_cols: tuple[str, ...] = CONTEXT_COLS
    fill_: pd.DataFrame | None = None       # per-hour medians of the sequence columns
    fill_global_: pd.Series | None = None
    seq_mean_: np.ndarray | None = None
    seq_std_: np.ndarray | None = None
    ctx_imputer_: TrainOnlyImputer | None = None
    ctx_mean_: np.ndarray | None = None
    ctx_std_: np.ndarray | None = None
    y_mean_: float = 0.0
    y_std_: float = 1.0

    def _raw(self, frame: pd.DataFrame) -> pd.DataFrame:
        return frame[list(self.seq_columns)]

    def fit(self, train: SupervisedSet, frame: pd.DataFrame) -> "SequenceEncoder":
        # Training period = every slot up to the last training origin.
        raw = self._raw(frame.loc[: train.X.index.max()])
        self.fill_ = raw.groupby(raw.index.hour).median()
        self.fill_global_ = raw.median()
        filled = self._fill(raw)
        self.seq_mean_ = filled.mean().to_numpy()
        self.seq_std_ = np.where(filled.std().to_numpy() > 0, filled.std().to_numpy(), 1.0)
        self.ctx_imputer_ = TrainOnlyImputer().fit(train.X)
        ctx = self.ctx_imputer_.transform(train.X)[list(self.context_cols)].to_numpy()
        self.ctx_mean_ = ctx.mean(axis=0)
        self.ctx_std_ = np.where(ctx.std(axis=0) > 0, ctx.std(axis=0), 1.0)
        self.y_mean_ = float(train.y.mean())
        self.y_std_ = float(train.y.std()) or 1.0
        return self

    def _fill(self, raw: pd.DataFrame) -> pd.DataFrame:
        by_hour = self.fill_.reindex(raw.index.hour)
        by_hour.index = raw.index
        return raw.fillna(by_hour).fillna(self.fill_global_)

    def transform(self, data: SupervisedSet, frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """Window for origin t covers t-lookback+1 .. t (inclusive); nothing after t."""
        L = self.lookback
        step = frame.index[1] - frame.index[0]
        start = frame.index[0] - (L - 1) * step
        grid = pd.date_range(start, frame.index[-1], freq=step)
        raw = self._raw(frame).reindex(grid)
        missing = raw[list(self.seq_columns[:2])].isna().to_numpy(dtype=np.float32)
        filled = (self._fill(raw).to_numpy() - self.seq_mean_) / self.seq_std_
        minutes = grid.hour * 60 + grid.minute
        tod = np.c_[np.sin(2 * np.pi * minutes / 1440), np.cos(2 * np.pi * minutes / 1440)]
        per_slot = np.c_[filled, missing, tod].astype(np.float32)
        end = grid.get_indexer(data.X.index)
        if (end < 0).any():
            raise ValueError("origin outside the frame")
        windows = np.stack([per_slot[e - L + 1: e + 1] for e in end])
        ctx = self.ctx_imputer_.transform(data.X)[list(self.context_cols)].to_numpy()
        ctx = ((ctx - self.ctx_mean_) / self.ctx_std_).astype(np.float32)
        return windows, ctx

    @property
    def n_inputs(self) -> int:
        return len(self.seq_columns) + 2 + 2


def _torch():
    import torch

    return torch


class _LSTMNet:
    @staticmethod
    def build(n_inputs: int, n_ctx: int, hidden: int):
        torch = _torch()
        nn = torch.nn

        class Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.lstm = nn.LSTM(n_inputs, hidden, batch_first=True)
                self.head = nn.Sequential(nn.Linear(hidden + n_ctx, hidden), nn.ReLU(), nn.Linear(hidden, 1))

            def forward(self, seq, ctx):
                out, _ = self.lstm(seq)
                return self.head(torch.cat([out[:, -1], ctx], dim=1)).squeeze(1)

        return Net()


class _TCNNet:
    @staticmethod
    def build(n_inputs: int, n_ctx: int, hidden: int, dilations=(1, 2, 4, 8), kernel: int = 3):
        torch = _torch()
        nn = torch.nn

        class CausalBlock(nn.Module):
            def __init__(self, c_in, c_out, dilation):
                super().__init__()
                self.pad = (kernel - 1) * dilation
                self.conv1 = nn.Conv1d(c_in, c_out, kernel, dilation=dilation)
                self.conv2 = nn.Conv1d(c_out, c_out, kernel, dilation=dilation)
                self.skip = nn.Conv1d(c_in, c_out, 1) if c_in != c_out else nn.Identity()
                self.act = nn.ReLU()

            def forward(self, x):
                # Left padding only: output at step s sees inputs <= s.
                y = self.act(self.conv1(nn.functional.pad(x, (self.pad, 0))))
                y = self.act(self.conv2(nn.functional.pad(y, (self.pad, 0))))
                return self.act(y + self.skip(x))

        class Net(nn.Module):
            def __init__(self):
                super().__init__()
                blocks, c = [], n_inputs
                for d in dilations:
                    blocks.append(CausalBlock(c, hidden, d))
                    c = hidden
                self.tcn = nn.Sequential(*blocks)
                self.head = nn.Sequential(nn.Linear(hidden + n_ctx, hidden), nn.ReLU(), nn.Linear(hidden, 1))

            def forward(self, seq, ctx):
                h = self.tcn(seq.transpose(1, 2))[:, :, -1]
                return self.head(torch.cat([h, ctx], dim=1)).squeeze(1)

        return Net()


@dataclass
class TrainConfig:
    lookback: int = 24
    hidden: int = 32
    lr: float = 1e-3
    batch_size: int = 64
    max_epochs: int = 300
    patience: int = 25
    weight_decay: float = 1e-4


class _SequenceModel:
    name = "sequence"
    net_builder = None

    def __init__(
        self,
        seed: int = 0,
        train_cfg: TrainConfig | None = None,
        night_threshold: float | None = DEFAULT_NIGHT_THRESHOLD,
        capacity: float = PLANT_CAPACITY_KW,
        seq_columns: tuple[str, ...] = SEQ_COLUMNS,
        context_cols: tuple[str, ...] = CONTEXT_COLS,
    ) -> None:
        self.seed = seed
        self.cfg = train_cfg or TrainConfig()
        self.night_threshold = night_threshold
        self.capacity = capacity
        self.seq_columns = seq_columns
        self.context_cols = context_cols

    def fit(self, train: SupervisedSet, val: SupervisedSet, frame: pd.DataFrame):
        torch = _torch()
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        torch.use_deterministic_algorithms(True)
        self.encoder_ = SequenceEncoder(lookback=self.cfg.lookback, seq_columns=self.seq_columns,
                                        context_cols=self.context_cols).fit(train, frame)
        enc = self.encoder_
        seq_tr, ctx_tr = (torch.from_numpy(a) for a in enc.transform(train, frame))
        seq_va, ctx_va = (torch.from_numpy(a) for a in enc.transform(val, frame))
        y_tr = torch.tensor(((train.y - enc.y_mean_) / enc.y_std_).to_numpy(), dtype=torch.float32)

        net = self.net_builder.build(enc.n_inputs, len(self.context_cols), self.cfg.hidden)
        opt = torch.optim.Adam(net.parameters(), lr=self.cfg.lr, weight_decay=self.cfg.weight_decay)
        gen = torch.Generator().manual_seed(self.seed)
        best_score, best_state, best_epoch, bad = np.inf, None, 0, 0
        self.history_: list[tuple[int, float]] = []
        for epoch in range(1, self.cfg.max_epochs + 1):
            net.train()
            order = torch.randperm(len(y_tr), generator=gen)
            for i in range(0, len(order), self.cfg.batch_size):
                idx = order[i: i + self.cfg.batch_size]
                opt.zero_grad()
                loss = torch.nn.functional.mse_loss(net(seq_tr[idx], ctx_tr[idx]), y_tr[idx])
                loss.backward()
                opt.step()
            net.eval()
            with torch.no_grad():
                pred_va = net(seq_va, ctx_va).numpy() * enc.y_std_ + enc.y_mean_
            score = _val_mae(pred_va, val, self.night_threshold, self.capacity)
            self.history_.append((epoch, score))
            if score < best_score - 1e-9:
                best_score, best_state, best_epoch, bad = score, copy.deepcopy(net.state_dict()), epoch, 0
            else:
                bad += 1
                if bad >= self.cfg.patience:
                    break
        net.load_state_dict(best_state)
        self.net_ = net
        self.val_mae_ = best_score
        self.params_ = {**self.cfg.__dict__, "best_epoch": best_epoch,
                        "n_parameters": int(sum(p.numel() for p in net.parameters()))}
        return self

    def predict(self, data: SupervisedSet, frame: pd.DataFrame) -> np.ndarray:
        torch = _torch()
        seq, ctx = (torch.from_numpy(a) for a in self.encoder_.transform(data, frame))
        self.net_.eval()
        with torch.no_grad():
            out = self.net_(seq, ctx).numpy()
        return out * self.encoder_.y_std_ + self.encoder_.y_mean_


class LSTMForecaster(_SequenceModel):
    name = "lstm"
    net_builder = _LSTMNet


class TCNForecaster(_SequenceModel):
    name = "tcn"
    net_builder = _TCNNet


@dataclass
class ModelSpec:
    """One row of the A3 comparison."""

    label: str
    factory: object                      # callable(seed) -> model
    seeds: tuple[int, ...] = (0,)
    oracle: bool = False
    notes: str = ""
