# Real IQM payloads

Each folder here holds one real calibration that a user saved from IQM Resonance.
`tests/test_real_payloads.py` finds every folder and checks it. This folder has no payload yet.

To add a payload:

1. Save the quality metric set and the dynamic quantum architecture of one calibration set.
   This example uses `iqm-client` 36:

   ```python
   from iqm.iqm_client import IQMClient

   # iqm-client 36 reads the token from the IQM_TOKEN environment variable
   client = IQMClient.from_url("https://resonance.meetiqm.com/", quantum_computer="garnet")
   metrics = client.get_quality_metric_set()
   architecture = client.get_dynamic_quantum_architecture(metrics.describes_id)
   ```

2. Make a folder `<device>-<date>`, such as `garnet-2026-11-01`. Write
   `metrics.model_dump_json()` to `metrics.json` and `architecture.model_dump_json()` to
   `architecture.json` in it. `architecture.json` is optional.
3. Optionally, add `expect.json` with values that the profile must keep. The keys are `id`,
   `num_qubits`, `calibrated_at`, `gates`, `calibrated` (the gates that have a calibrated
   record), `fingerprint` and `artifact_hash`, for example:

   ```json
   {"id": "iqm_garnet-2026-11-01", "num_qubits": 20, "calibrated": ["cz", "r"]}
   ```

   A `fingerprint` pins the physics of the profile, so a change to any number fails the test. An
   `artifact_hash` also pins `provenance`, which includes the NoiseVault version, so it changes
   with each release.

Each payload must import with at least one calibrated gate record and one readout error, and
pass `nv check` when a framework is installed.

Read IQM's terms of use before you commit a payload. If the terms do not let you publish it, put
the folder in a private folder `<private>/iqm/` instead, and set
`NOISEVAULT_REAL_PAYLOADS=<private>` where the tests run.
