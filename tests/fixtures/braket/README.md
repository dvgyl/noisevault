# Braket fixture

Synthetic device properties with made-up numbers, shaped like documents saved with
`AwsDevice(arn).properties.json()` and validated against the `amazon-braket-schemas` models:

- `iqm_capabilities_v1.json`: IQM-style capabilities with standardized v1, qubits numbered from 1
- `rigetti_standardized_v1.json`: only the standardized v1 part, with directed CNOT records, a
  gap in the qubit numbers and a fidelity type NoiseVault does not know
- `ionq_capabilities_v3.json`: IonQ-style capabilities with device-level standardized v3

The run script tests use four more synthetic files, which hold the whole capabilities document
with the OpenQASM action, the verbatim pragma and, where the device runs one, the program set
action. The Braket `LocalEmulator` accepts each file:

- `iqm_garnet_run.json`: IQM-style, natives `cz` and `prx`, standardized v1, qubits 1 to 6
- `rigetti_cepheus_run.json`: Rigetti-style, natives `rx`, `rz` and `cz`, standardized v1
- `aqt_ibex_run.json`: AQT-style, natives `prx`, `xx` and `rz`, standardized v3 with gate
  durations, all-to-all
- `ionq_forte_run.json`: IonQ-style, natives `GPI`, `GPI2` and `MS`, standardized v3 with gate
  durations, all-to-all, and no program set action

They describe no real device.
