# Rigetti fixture

Synthetic Rigetti QCS instruction set architectures (ISAs) with made-up numbers. They describe
no real device. Their shape follows the QCS API models in `qcs-api-client-openapi` and the ISA
that pyquil 4.23.0 reads, as `qcs_sdk.qpu.isa.get_instruction_set_architecture(id).json()`
saves it. The benchmark names for T1 and T2 (`FreeInversionRecovery`, `FreeInductionDecay`)
come from a Cepheus-1-108Q ISA that the QCS API served in September 2026. No real ISA is stored
here, because Rigetti's terms forbid redistribution.

- `ankaa_isa.json`: an Ankaa-style ISA with `ISWAP`, isolated and simultaneous RB, and no T1 or
  T2. It has these cases:
  - Node 4 is not an architecture node, and node 6 has only `I` and `RESET` sites (dead).
  - `MEASURE` on node 3 has `fRO` of exactly 1, and on node 5 the dead-site marker (0.5, 1.0).
  - `ISWAP` on 1-2 has the dead-site marker, on 2-3 no `fISWAP`, and the site on 1-5 lists
    `[5, 1]`. The edge 3-5 has no `ISWAP` site, and the edge 2-6 has a calibrated site on
    the dead node 6.
  - Node 3 has an `fRB` of 0.3, below the physical bound.
  - The ISA also has `XY`, `RESET` with `fAR`, and one timestamp of 1970-01-01.
- `cepheus_isa.json`: a Cepheus-style ISA with `CZ`, only simultaneous RB, and T1 and T2
  benchmarks. Node 3 has a T1 of 0 s, the pair 2-3 has an `fCZ` of 0.15, below the physical
  bound, and `CPHASE` has the placeholder value 0.01 with error 0.99.
