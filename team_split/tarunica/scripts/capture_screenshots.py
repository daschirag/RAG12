"""Capture console screenshots with Playwright and check the browser console for errors.

    uvicorn src.backend.main:app        # in another terminal; wait until GET /ready says ready/degraded
    python scripts/capture_screenshots.py [--base http://127.0.0.1:8000] [--channel chrome]

Uses the installed Chrome/Edge (``--channel``), so no browser download is needed. Writes
docs/screenshots/<page>.png and docs/screenshots/console_check.json (console errors per page). Pages that call the
LLM wait for the real answer. No keys or secrets are shown on any page.
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "screenshots"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8000")
    ap.add_argument("--channel", default="chrome", help="chrome or msedge (installed browser)")
    ap.add_argument("--theme", default="light", choices=("light", "dark"))
    ap.add_argument("--mobile", action="store_true", help="390 px wide viewport; also checks horizontal overflow")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    report: dict[str, dict] = {}
    with sync_playwright() as p:
        browser = p.chromium.launch(channel=args.channel, headless=True)
        size = {"width": 390, "height": 844} if args.mobile else {"width": 1366, "height": 900}
        ctx = browser.new_context(viewport=size, color_scheme=args.theme,
                                  reduced_motion="reduce")
        page = ctx.new_page()
        errors: list[str] = []
        page.on("console", lambda m: errors.append(f"{m.type}: {m.text}") if m.type in ("error", "warning") else None)
        page.on("pageerror", lambda e: errors.append(f"pageerror: {e}"))
        suffix = ("" if args.theme == "light" else "_dark") + ("_mobile" if args.mobile else "")

        def shot(name: str, steps) -> None:
            errors.clear()
            t0 = time.time()
            steps()
            # full-page capture only: sticky bars would be painted mid-image
            page.add_style_tag(content=".topbar,.composer{position:static !important}")
            page.wait_for_timeout(600)
            page.screenshot(path=str(OUT / f"{name}{suffix}.png"), full_page=True)
            overflow = page.evaluate("document.documentElement.scrollWidth - window.innerWidth")
            report[name] = {"errors": list(errors), "seconds": round(time.time() - t0, 1),
                            "horizontal_overflow_px": overflow}
            print(f"{name}: {len(errors)} console errors/warnings, horizontal overflow {overflow} px "
                  f"({report[name]['seconds']} s)")

        def assistant():
            page.goto(f"{args.base}/console/#assistant")
            page.locator(".chips .chip").first.wait_for(timeout=60_000)
            page.locator(".chips .chip").nth(3).click()
            page.locator(".msg.bot .trust, .msg.bot .error-box").first.wait_for(timeout=400_000)
            page.locator(".msg.bot details").nth(1).click()          # open the sources list
            page.locator(".msg.bot details.source").first.click()     # expand one source card
            page.locator(".msg.bot details.source .body").first.wait_for()
            page.wait_for_timeout(1500)

        def forecast():
            page.goto(f"{args.base}/console/#forecast")
            page.locator("#page-forecast .chart svg").first.wait_for(timeout=120_000)
            page.locator("#page-forecast .hbar-row").first.wait_for(timeout=60_000)
            page.get_by_role("button", name="Check with the decision engine").click()
            page.locator("#page-forecast .trust").first.wait_for(timeout=60_000)

        def diagnosis():
            page.goto(f"{args.base}/console/#diagnosis")
            page.locator("#page-diagnosis tbody tr").first.wait_for(timeout=60_000)
            page.locator("#page-diagnosis tbody tr").filter(
                has=page.locator("td:first-child", has_text=re.compile(r"^73$"))).first.click()
            page.get_by_role("button", name="Diagnose").click()
            page.locator("#page-diagnosis .trust, #page-diagnosis .error-box").first.wait_for(timeout=400_000)

        def lab():
            page.goto(f"{args.base}/console/#lab")
            page.get_by_role("button", name="Run attack").wait_for(timeout=60_000)
            page.get_by_role("button", name="Run attack").click()
            page.locator("#page-lab .kpis").first.wait_for(timeout=180_000)

        def lab_heldout():
            page.goto(f"{args.base}/console/#lab")
            page.get_by_role("button", name="Held-out weakness").click()
            page.locator("#page-lab table").first.wait_for(timeout=60_000)

        def lab_injection():
            page.goto(f"{args.base}/console/#lab")
            page.get_by_role("button", name="Prompt injection").click()
            page.get_by_role("button", name="Test question").click()
            page.locator("#page-lab .trust").first.wait_for(timeout=60_000)

        def lab_poisoning():
            page.goto(f"{args.base}/console/#lab")
            page.get_by_role("button", name="KB poisoning").click()
            page.get_by_role("button", name="Check document trust").click()
            page.locator("#page-lab .grid .badge").first.wait_for(timeout=120_000)

        def results():
            page.goto(f"{args.base}/console/#results")
            page.locator("#page-results section").first.wait_for(timeout=60_000)

        def about():
            page.goto(f"{args.base}/console/#about")
            page.locator("#page-about .checks").first.wait_for(timeout=60_000)

        for name, fn in (("assistant", assistant), ("forecast", forecast), ("diagnosis", diagnosis), ("lab", lab),
                         ("lab_injection", lab_injection), ("lab_poisoning", lab_poisoning),
                         ("lab_heldout", lab_heldout), ("results", results), ("about", about)):
            # every navigation reloads the module; hash routing alone would keep the old page mounted
            page.goto("about:blank")
            shot(name, fn)
        browser.close()
    (OUT / f"console_check{suffix}.json").write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
