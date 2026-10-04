# Contributing

Bug reports, new data sources and fixes are welcome. Open an issue first for anything larger
than a small fix, so we can agree on the approach.

## Set up

NoiseVault uses [uv](https://docs.astral.sh/uv/) for development. In a checkout, run these
commands:

```bash
uv venv --python 3.13
source .venv/bin/activate
uv pip install -e '.[dev]'
```

`dev` includes every framework extra, PyMatching, pytest, ruff and build. Run the commands below
in the active environment.

## Run the tests

```bash
pytest                                 # missing frameworks skip
NOISEVAULT_REQUIRE_ALL=1 pytest        # a missing framework fails instead of skipping
NOISEVAULT_NETWORK=1 pytest -m network # tests that reach live calibration endpoints
ruff check .
ruff format --check .
```

- `NOISEVAULT_REQUIRE_ALL=1` makes sure a full environment runs every test. CI sets it on the
  full-install job.
- Tests that need the network have `@pytest.mark.network` and skip unless you set
  `NOISEVAULT_NETWORK=1`. All other tests must run offline, on saved fixtures.
- Every test gets an empty vault through the `vault` fixture in `tests/conftest.py`, so tests
  never read or write `~/.noisevault`.

`tests/test_examples.py` runs every script in `examples/` and every Python block in `docs/`.
`examples/compare_devices.py` also needs matplotlib, and `examples/mitigation_zne.py` needs
Mitiq and ply, which support Python up to 3.12:

```bash
uv run --no-project --python 3.12 --with-editable '.[dev]' --with matplotlib --with mitiq \
  --with ply python -m pytest tests/test_examples.py
```

`--no-project` stops uv from changing your `.venv`. The command runs in a temporary
environment.

## Style

- Code uses small, typed functions and prefers data structures to branching. A comment explains
  a reason that is not obvious, never what the code does.
- Framework and source modules import their framework inside the module, and nothing in
  `noisevault/__init__.py` imports those modules, so a core install keeps working.
- Errors say what went wrong and what to do next. Expected failures raise a `NoiseVaultError`
  subclass, and the CLI prints them without a traceback.
- Tests assert behavior and must be able to fail for the defect they target.
- Docs use short, concrete sentences with no em dashes, and a reader can check every claim.
  Python blocks in `docs/` run in CI. Put `<!-- not-run: reason -->` on the line before a block
  that cannot run there.
- Line length is 100. `ruff format` decides formatting.

## Add a data source

[docs/data-sources.md](docs/data-sources.md#add-a-source) lists the steps. The rule that matters
most is to set `provenance.redistributable` to `"yes"` only when the data's license allows
redistribution. Only a profile with that value can be a bundled profile.

## Add a framework export

1. Write `src/noisevault/frameworks/<name>.py` with a `to_<name>(profile, *, layout=None,
   unknown_gates="typical", ...)` function. When the framework is missing, raise an `ImportError`
   that names the extra.
2. Map circuit qubits with `noisevault.layout.normalize_layout`. Get each gate's channels from
   `noisevault.conversion.resolve_op`, and each idle qubit's relaxation from
   `noisevault.conversion.idle_channel`. These functions keep the lookup rules, errors and report
   wording identical across frameworks.
3. Start a report with `Report.start(profile, "<name>", <framework version>, **options)`.
   Call `report.record_effects(profile.effects)`. Mark what the export reproduces exactly,
   approximates, omits and does not know.
4. Return the framework's own object type with `.report` and `.profile` attached.
5. Add a `Profile.to_<name>` method in `src/noisevault/profile.py`, an extra in
   `pyproject.toml`, and a runner in `src/noisevault/check.py` so `nv check` covers the export.
6. Test the export against `noisevault.reference.probabilities`. Circuits on every bundled
   technology should agree with the reference to a total variation distance of 1e-9. Include
   circuits with reversed operands, non-contiguous layouts and asymmetric readout.
7. Document the export in [docs/frameworks.md](docs/frameworks.md) from the report that the
   export produces.

## Add or update a bundled profile

1. Make sure that the data has an open license. Make sure that the source module returns the
   data from `bundled_profiles()`.
2. Rebuild the bundle and NOTICE:

   ```bash
   python scripts/build_catalog.py
   python scripts/build_catalog.py --check   # prints "up to date"
   ```

   The build is deterministic, so the same source packages give byte-identical files.
3. Commit `src/noisevault/data/profiles/` and `NOTICE` together. The wheel must stay under
   3 MB. CI checks the wheel size.

If you change the pydantic models in `src/noisevault/profile.py`, regenerate the JSON Schema:

```bash
nv schema > docs/schema/profile-1.0.json
```

## Release

1. Update `version` in `pyproject.toml` and `__version__` in `src/noisevault/__init__.py`.
2. Add the release to `CHANGELOG.md` under a heading `## X.Y.Z (YYYY-MM-DD)`. Update `version`
   and `date-released` in `CITATION.cff`.
3. Merge to `main` when CI passes.
4. Tag the release and push the tag, as in `git tag v0.3.0 && git push origin v0.3.0`.

The tag starts `.github/workflows/release.yml`. The workflow checks that the tag matches the
version, builds the wheel and sdist, and creates a GitHub release with the notes of that version
from `CHANGELOG.md`. The wheel and sdist are attached to the release. A tag uploads nothing to
PyPI. To publish a tagged build to PyPI later, run the workflow by hand on that tag. That step
needs a trusted publisher on PyPI (owner `dvgyl`, repository `noisevault`, workflow
`release.yml`, environment `pypi`) and the `pypi` environment in the repository settings.

## Website

`site/index.html` is the source of the project website. `scripts/build_site.py` writes the
finished page to `_site/`, with data from the bundled profiles and the real output of `nv show`
and `nv cite`:

```bash
python scripts/build_site.py
```

The script stops if a number in the text does not match the data. On a push to `main`, the
`Website` workflow builds the page and publishes it to GitHub Pages.
