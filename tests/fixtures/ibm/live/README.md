# Live IBM calibrations

`nv pull ibm_kingston` and `nv pull ibm_fez` made these profiles on 2026-11-09 from the IBM
Quantum public calibration endpoint, with no account. On each device, one qubit has every
1-qubit gate disabled, but its readout and its 2-qubit gates stay calibrated: qubit 113 on
ibm_kingston and qubit 149 on ibm_fez. The bundled snapshots and the qiskit-ibm-runtime fakes
have no such qubit. `tests/test_ibm_campaign.py` uses these files to check that the campaign
picks only chains that `plan` can run.
