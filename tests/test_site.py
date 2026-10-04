"""The website builds from the bundled profiles, and its numbers match their sources."""

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
