# IQM fixture

Synthetic IQM Resonance documents with made-up numbers. They describe no real device. Their
shape follows the public models in `iqm-station-control-client` 14.0.0, which `iqm-client`
36.0.0 returns. No real IQM payload was available when these files were written.

- `metrics.json`: a quality metric set, as `IQMClient.get_quality_metric_set().model_dump_json()`
  saves it. The observation names follow the grammar of `iqm/station_control/client/qon.py`.
- `architecture.json`: the dynamic quantum architecture of the same calibration set, as
  `IQMClient.get_dynamic_quantum_architecture().model_dump_json()` saves it.

The set has five qubits, `QB1` to `QB5`, and holds these cases:

- `QB5` has data but is not in the architecture's `qubits`, and its T1 is marked `invalid`.
- `QB4` has only a Ramsey T2 (`t2_time`), and `QB5` has only a readout fidelity.
- `QB2__QB3` has two `cz` implementations, and the architecture overrides the default for it.
- `QB3__QB4` has a `cz` fidelity of exactly 1, a placeholder.
- `QB4__QB5` has a `cz` fidelity but no locus in the architecture.
- `QB1__QB4` has a locus in the architecture but no fidelity.
- `QB1` has two `prx` fidelities whose names differ only in the suffix.
- The set also has kinds of metrics that NoiseVault does not read.
