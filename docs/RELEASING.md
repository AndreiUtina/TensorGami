# Releasing Tensorgami

Releases are made by GitHub Actions when a version tag is pushed
([`.github/workflows/release.yml`](../.github/workflows/release.yml)). PyPI trusts the
workflow itself (Trusted Publishing), so there are no API tokens to create, store or rotate.

```
git push origin v0.2.0
  └─ build      the tag matches __version__; build the wheel and sdist; twine check;
  │             install the wheel alone in a fresh environment and draw in every style
  ├─ testpypi   upload to test.pypi.org                    (only if PUBLISH_TO_PYPI = true)
  ├─ pypi       upload to pypi.org, after your approval    (only if PUBLISH_TO_PYPI = true)
  └─ github-release   a GitHub Release with the wheel and sdist attached
```

Every push to `main` and every pull request runs [`ci.yml`](../.github/workflows/ci.yml):
lint, the tests on Python 3.11–3.13, and the built wheel drawing without PyTorch, both with
the newest dependencies and with the oldest accepted (Python 3.11, matplotlib 3.7).

## One-time setup

1. **Make the repository public**: GitHub → Settings → General → Danger Zone → Change
   visibility. (Required reviewers on environments, step 3, need a public repository on a
   free plan.)
2. **Register the publishers** for the project name `tensorgami`, which does not exist on
   PyPI yet, so it is a *pending* publisher:
   - [pypi.org](https://pypi.org/manage/account/publishing/) (an account with 2FA) →
     Publishing → *Add a new pending publisher* → GitHub:
     PyPI project name `tensorgami`, owner `AndreiUtina`, repository `TensorGami`,
     workflow name `release.yml`, environment name `pypi`.
   - [test.pypi.org](https://test.pypi.org/manage/account/publishing/) (a separate account):
     the same, with environment name `testpypi`.

   The first successful upload creates the project and turns the pending publisher into a
   normal one.
3. **Create the environments**: GitHub → Settings → Environments → *New environment*:
   - `testpypi`: no rules needed.
   - `pypi`: *Required reviewers*: yourself, so every upload to PyPI waits for your click.
     Under *Deployment branches and tags*, allow only tags matching `v*`.
4. **Switch publishing on**: Settings → Secrets and variables → Actions → Variables →
   *New repository variable* `PUBLISH_TO_PYPI` = `true`. Without it a tag only builds,
   checks and makes a GitHub Release, which is a useful dry run.

## Each release

1. Choose the version (0.x: a minor bump for new features or changed API, a patch bump for
   fixes) and set it in:
   - `src/tensorgami/__init__.py`: `__version__ = "X.Y.Z"`, the single source the build reads;
   - `CITATION.cff`: `version: "X.Y.Z"` and `date-released: "YYYY-MM-DD"`;
   - `CHANGELOG.md`: move what is under *Unreleased* into `## [X.Y.Z] — YYYY-MM-DD`.

   `tests/test_release.py` checks that the three agree.
2. Check locally, commit, push, and wait for CI to pass:
   ```sh
   uv run pytest && uv run ruff check . && uv build
   git commit -am "release X.Y.Z" && git push origin main
   ```
3. Tag and push the tag:
   ```sh
   git tag -a vX.Y.Z -m "Tensorgami X.Y.Z"
   git push origin vX.Y.Z
   ```
4. Follow the *Release* workflow under Actions. After the TestPyPI upload, check the page
   at <https://test.pypi.org/p/tensorgami>, then approve the `pypi` job.
5. Check the release from a clean environment:
   ```sh
   uv run --isolated --no-project --with tensorgami==X.Y.Z python -c "import tensorgami as tg; print(tg.__version__)"
   ```

## When something goes wrong

- **The build job fails** (the tag doesn't match `__version__`, a test or check fails):
  nothing was uploaded. Fix it on `main`, then move the tag:
  `git tag -d vX.Y.Z && git push --delete origin vX.Y.Z`, and tag again.
- **An upload fails** (e.g. the publisher's settings don't match): fix the settings, then
  *Re-run failed jobs* on the workflow run. TestPyPI skips files it already has.
- **A broken release reached PyPI**: a version can never be uploaded twice, not even after
  deleting it. *Yank* it on PyPI (pip then skips it unless pinned exactly) and release a
  patch version.
