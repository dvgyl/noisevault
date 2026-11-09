# Real Rigetti payloads

Each folder here holds one real instruction set architecture (ISA) that a user saved from
Rigetti QCS. `tests/test_real_payloads.py` finds every folder and checks it. This folder has no
payload yet.

To add a payload:

1. Save the ISA:

   ```python
   from qcs_sdk.qpu.isa import get_instruction_set_architecture

   isa = get_instruction_set_architecture("Ankaa-3")
   ```

2. Make a folder `<device>-<date>`, such as `ankaa-3-2026-11-01`, and write `isa.json(pretty=True)`
   to `isa.json` in it.
3. Optionally, add `expect.json` with values that the profile must keep. The keys are `id`,
   `num_qubits`, `calibrated_at`, `gates`, `calibrated` (the gates that have a calibrated
   record), `fingerprint` and `artifact_hash`, for example:

   ```json
   {"id": "rigetti_ankaa-3", "num_qubits": 84, "calibrated": ["iswap", "rx"]}
   ```

   A `fingerprint` pins the physics of the profile, so a change to any number fails the test. An
   `artifact_hash` also pins `provenance`, which includes the NoiseVault version, so it changes
   with each release.

Each payload must import with at least one calibrated gate record and one readout error, and
pass `nv check` when a framework is installed.

Rigetti's terms of service forbid you to display any part of the service to third parties, so do
not commit a real ISA to this public repository. Put the folder in a private folder
`<private>/rigetti/` instead, and set `NOISEVAULT_REAL_PAYLOADS=<private>` where the tests run.
