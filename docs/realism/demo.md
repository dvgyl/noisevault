# Realism table demo

Each row scores a profile on counts from one run with `nv compare`. A factor multiplies
the error rates of the profile, so x2 means about twice the errors. A factor of 1 means
that the calibration explains the counts. The fit column gives the p-value of the
goodness-of-fit test. A p-value below 0.01 means that one gate factor and
one readout factor do not explain every circuit.

Every row is simulated. The counts come from a profile with known factors, not from
a device. The source column gives the factors that the simulation used.

| Device | Vendor | Technology | Date | Gate factor (95% interval) | Readout factor (95% interval) | Fit | Source |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `google_weber` | google | superconducting | 2021-11-04 | x1.14 (1.04 to 1.24) | x1.49 (1.39 to 1.58) | p = 0.21 | simulated, x1.1 gates, x1.5 readout |
| `ibm_kingston` | ibm | superconducting | 2026-04-16 | x1.95 (1.59 to 2.35) | x1.22 (1.05 to 1.39) | p = 0.51 | simulated, x1.6 gates, x1.2 readout |
| `ibm_kingston` | ibm | superconducting | 2026-04-23 | x2.27 (1.86 to 2.73) | x1.39 (1.21 to 1.58) | p = 0.2 | simulated, x2.2 gates, x1.3 readout |
| `quantinuum_h1-1` | quantinuum | trapped ion | 2025-05-03 | x1.26 (0.854 to 1.74) | x1.11 (0.874 to 1.39) | p = 0.07 | simulated, x1.3 gates, x1 readout |

The JSON file beside this page holds each value with six significant digits.
