# Wind T1: where persistence wins (diagnostic, test rows of the 4 rolling-origin months)

MAE (kW) per group; learned models use seed-averaged predictions. 'level' = observed power at the
target as a share of rated power; 'movement' = |power(T) - power(t)|, i.e. how much the hour changed.

| split | group | rows | share | persistence | random_forest | xgboost | lstm | tcn |
|---|---|---|---|---|---|---|---|---|
| level | ~0 (<1%) | 3192 | 19.5% | 45.6 | 186.5 | 140.2 | 142.6 | 125.8 |
| level | low (1-10%) | 1823 | 11.2% | 219.5 | 271.1 | 252.4 | 270.0 | 252.0 |
| level | mid (10-90%) | 8459 | 51.8% | 444.9 | 403.1 | 405.5 | 428.8 | 419.6 |
| level | rated (>=90%) | 2856 | 17.5% | 195.6 | 385.0 | 361.8 | 457.2 | 364.2 |
| movement | flat (<1% change) | 5004 | 30.6% | 5.5 | 156.3 | 121.8 | 159.6 | 122.2 |
| movement | moderate (1-10%) | 6453 | 39.5% | 172.0 | 228.4 | 210.1 | 266.8 | 225.5 |
| movement | ramp (>10%) | 4873 | 29.8% | 765.5 | 686.0 | 698.9 | 689.5 | 694.4 |
