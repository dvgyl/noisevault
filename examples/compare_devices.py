"""Plot the GHZ success probability against n for the same GHZ-n circuit on five devices.

The five devices are four bundled profiles (two IBM superconducting devices, two Quantinuum
trapped-ion devices) and one hypothetical neutral-atom device. The script writes
compare_devices.svg in the current folder, or at the path given as the first argument.

Needs:
    pip install "noisevault[qiskit] @ git+https://github.com/dvgyl/noisevault"
    pip install matplotlib
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit, transpile

import noisevault as nv

SIZES = range(2, 11, 2)
SHOTS = 4000
OUT = Path(sys.argv[1] if len(sys.argv) > 1 else "compare_devices.svg")

atoms = nv.Profile.uniform(
    "toy-atoms",
    technology="neutral_atom",
    num_qubits=20,
    one_qubit_error=5e-4,
    two_qubit_error=5e-3,
    readout_error=1e-2,
    t1_us=4e6,
    t2_us=1.5e6,
    one_qubit_ns=500,
    two_qubit_ns=250,
)
# Each color is readable on white and on GitHub's dark background. Text labels also name each
# device.
DEVICES = [
    (nv.load("ibm_fez"), "#3987e5", "o", "-"),
    (nv.load("ibm_brisbane"), "#d95926", "s", "-"),
    (nv.load("quantinuum_h1-1"), "#199e70", "D", "-"),
    (nv.load("quantinuum_h2-1"), "#c98500", "^", "-"),
    (atoms, "#d55181", "v", "--"),
]


def ghz(n: int) -> QuantumCircuit:
    circuit = QuantumCircuit(n)
    circuit.h(0)
    for q in range(n - 1):
        circuit.cx(q, q + 1)
    circuit.measure_all()
    return circuit


def success(profile: nv.Profile, sim, n: int) -> float:
    layout = list(profile.suggest_layout(n).values())
    native = transpile(ghz(n), sim, initial_layout=layout, seed_transpiler=1)
    counts = sim.run(native, shots=SHOTS, seed_simulator=1).result()
    counts = counts.get_counts()
    return (counts.get("0" * n, 0) + counts.get("1" * n, 0)) / SHOTS


# The transparent figure shows on GitHub's white or #0d1117 background. No gray reaches a
# contrast of 4.5:1 on both. This gray has the highest minimum contrast (4.29:1 on white, 4.41:1
# on #0d1117).
INK, GRID = "#7a7a7a", "#7a7a7a40"
plt.rcParams.update(
    {"font.family": "sans-serif", "font.size": 10, "text.color": INK, "svg.hashsalt": "nv"}
)
fig, ax = plt.subplots(figsize=(7.2, 4.2))

print(f"{'device':<30}" + "".join(f"n={n:<6}" for n in SIZES))
for profile, color, marker, style in DEVICES:
    sim = profile.to_qiskit()
    values = [success(profile, sim, n) for n in SIZES]
    when = profile.device.calibrated_at
    label = f"{profile.id} ({when.date()})" if when else f"{profile.id} (hypothetical)"
    print(f"{label:<30}" + "".join(f"{v:<8.3f}" for v in values))
    ax.plot(SIZES, values, color=color, marker=marker, markersize=6, linestyle=style, lw=2)
    ax.annotate(
        label,
        (SIZES[-1], values[-1]),
        xytext=(8, 0),
        textcoords="offset points",
        va="center",
        fontsize=9,
        color=INK,
    )

ax.set_xlabel("GHZ size n (qubits)", color=INK)
ax.set_ylabel(f"P(all 0 or all 1), {SHOTS} shots", color=INK)
ax.set_title("GHZ success under calibrated noise (Qiskit Aer)", color=INK, loc="left")
ax.set_xticks(list(SIZES))
ax.set_ylim(None, 1.0)
ax.grid(axis="y", color=GRID, linewidth=0.8)
ax.tick_params(colors=INK)
for side in ("top", "right"):
    ax.spines[side].set_visible(False)
for side in ("left", "bottom"):
    ax.spines[side].set_color(INK)
fig.tight_layout()
fig.savefig(OUT, transparent=True, metadata={"Date": None})
print(f"wrote {OUT}")
