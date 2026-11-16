| Series | Variant | Expert memory (GB) | Read per token (GB) | Dropped slots | KL | KL, no tool-use |
|---|---|---:|---:|---:|---:|---:|
| static pruning | frontier:s48 | 17.7 | 2.21 | 0.0% | 1.746 | 1.640 |
| static pruning | frontier:s96 | 35.4 | 2.21 | 0.0% | 1.129 | 1.010 |
| static pruning | tau384:s144 | 53.1 | 2.21 | 0.0% | 0.763 | 0.653 |
| static pruning | frontier:s200 | 73.7 | 2.21 | 0.0% | 0.543 | 0.427 |
| static pruning | frontier:s250 | 92.2 | 2.21 | 0.0% | 0.452 | 0.303 |
| static pruning | tau384:q384 | 141.6 | 2.21 | 0.0% | 0.390 | 0.235 |
| routed cache | c1-residents:k32 | 11.8 | 2.21 | 0.0% | 0.942 | 0.732 |
| routed cache | c1-residents:k48 | 17.7 | 2.21 | 0.0% | 0.790 | 0.582 |
| routed cache | c1-residents:k64 | 23.6 | 2.21 | 0.0% | 0.705 | 0.496 |
| routed cache | c1-residents:k96 | 35.4 | 2.21 | 0.0% | 0.605 | 0.390 |
| routed cache | c1-residents:k144 | 53.1 | 2.21 | 0.0% | 0.454 | 0.320 |
| routed cache | c1-residents:k200 | 73.7 | 2.21 | 0.0% | 0.401 | 0.279 |
| routed cache | c1-residents:k250 | 92.2 | 2.21 | 0.0% | 0.398 | 0.254 |
| routed cache | tau384:q384 | 141.6 | 2.21 | 0.0% | 0.390 | 0.235 |
| routed + 3-bit hot | b2-nested:k234n32 | 92.2 | 2.45 | 0.0% | 0.328 | 0.223 |
| routed + 3-bit hot | b2-nested:k218n64 | 92.2 | 2.6 | 0.0% | 0.311 | 0.215 |
| routed + 3-bit hot | b2-nested:k202n96 | 92.2 | 2.71 | 0.0% | 0.308 | 0.222 |
| tier 0, 144 residents | c1-residents:k144 | 53.1 | 2.21 | 0.0% | 0.454 | 0.320 |
| tier 0, 144 residents | frontier:k144d05 | 53.1 | 2.19 | 0.9% | 0.462 | 0.331 |
| tier 0, 144 residents | frontier:k144d10 | 53.1 | 2.08 | 6.0% | 0.478 | 0.332 |
| tier 0, 144 residents | frontier:k144d20 | 53.1 | 1.25 | 43.7% | 0.615 | 0.387 |
| tier 0, 218 + 64 hot | b2-nested:k218n64 | 92.2 | 2.6 | 0.0% | 0.311 | 0.215 |
| tier 0, 218 + 64 hot | frontier:k218n64d05 | 92.2 | 2.58 | 0.7% | 0.313 | 0.219 |
| tier 0, 218 + 64 hot | frontier:k218n64d10 | 92.2 | 2.47 | 5.2% | 0.317 | 0.220 |
