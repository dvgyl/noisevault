"""The website builds from the bundled profiles, and its numbers match their sources."""

import os
import subprocess
import sys
from pathlib import Path

import noisevault as nv

ROOT = Path(__file__).resolve().parents[1]


def test_site_builds_with_every_placeholder_filled(tmp_path):
    run = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "build_site.py"), "--out", str(tmp_path)],
        capture_output=True,
        text=True,
    )
    assert run.returncode == 0, run.stderr
    page = (tmp_path / "index.html").read_text()
    assert f"Version {nv.__version__}</a>" in page
    assert f"releases/tag/v{nv.__version__}" in page
    for token in ("__DATA__", "__VERSION__", "__RELEASED__"):
        assert token not in page


def test_site_shows_the_bundled_profile_when_the_vault_has_a_newer_one(tmp_path):
    bundled = nv.load("ibm_fez")
    data = bundled.to_dict()
    data["device"]["calibrated_at"] = "2099-01-01T00:00:00Z"
    data["qubits"][0]["readout"] = {"error": 0.4}
    newer = nv.Profile.from_dict(data)
    home = tmp_path / "home"
    env = {**os.environ, "NOISEVAULT_HOME": str(home)}
    code = (
        "import sys, noisevault as nv; from noisevault.catalog import vault_path;"
        "p = nv.load(sys.argv[1]); t = vault_path(p); t.parent.mkdir(parents=True); p.save(t)"
    )
    source = newer.save(tmp_path / "newer.json")
    subprocess.run([sys.executable, "-c", code, str(source)], env=env, check=True)
    run = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "build_site.py"), "--out", str(tmp_path)],
        capture_output=True,
        text=True,
        env=env,
    )
    assert run.returncode == 0, run.stderr
    page = (tmp_path / "index.html").read_text()
    assert f'"fingerprint":"{bundled.short_fingerprint}"' in page
    assert newer.short_fingerprint not in page
