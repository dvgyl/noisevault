# Realism table

Each row scores a profile on counts from one run with `nv compare`. A factor multiplies
the error rates of the profile, so x2 means about twice the errors. A factor of 1 means
that the calibration explains the counts. The fit column gives the p-value of the
goodness-of-fit test. A p-value below 0.01 means that one gate factor and
one readout factor do not explain every circuit.

| Device | Vendor | Technology | Date | Gate factor (95% interval) | Readout factor (95% interval) | Fit | Source |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `ibm_fez` | ibm | superconducting | 2026-11-09 | x1.75 (1.36 to 2.19) | x0.859 (0.515 to 1.3) | p = 0, poor | hardware |
| `ibm_fez` | ibm | superconducting | 2026-11-09 | x2.77 (1.78 to 3.93) | x0.793 (0.583 to 1.04) | p = 0.0025, poor | hardware |
| `ibm_fez` | ibm | superconducting | 2026-11-12 | x0.936 (0.883 to 0.994) | x0.967 (0.616 to 1.41) | p = 0.0025, poor | hardware |
| `ibm_fez` | ibm | superconducting | 2026-11-12 | x1.04 (0.981 to 1.1) | x0.902 (0.68 to 1.16) | p = 0, poor | hardware |
| `ibm_kingston` | ibm | superconducting | 2026-11-09 | x1.78 (1.32 to 2.36) | x1.44 (0.637 to 2.59) | p = 0, poor | hardware |
| `ibm_kingston` | ibm | superconducting | 2026-11-09 | x3.24 (1.23 to 6.01) | x0.83 (0.456 to 1.33) | p = 0, poor | hardware |
| `ibm_marrakesh` | ibm | superconducting | 2026-11-10 | x1.66 (1.35 to 2.02) | x1.46 (1.07 to 1.92) | p = 0, poor | hardware |
| `ibm_marrakesh` | ibm | superconducting | 2026-11-10 | x2.12 (0.949 to 3.98) | x0.898 (0.525 to 1.4) | p = 0, poor | hardware |

The JSON file beside this page holds each value with six significant digits.
