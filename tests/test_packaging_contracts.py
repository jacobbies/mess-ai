"""Packaging/CI contract tests for extras and install guidance."""

from __future__ import annotations

import builtins
import re
import tomllib
from pathlib import Path

import pytest

from mess.probing import discovery

pytestmark = pytest.mark.unit


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _ci_extras(ci_yaml: str) -> set[str]:
    return set(re.findall(r"--extra\s+([a-zA-Z0-9_-]+)", ci_yaml))


def test_ci_extras_exist_in_pyproject_optional_dependencies():
    root = _repo_root()
    pyproject_data = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    workflows_dir = root / ".github" / "workflows"
    ci_candidates = [
        workflows_dir / "ci.yml",
        workflows_dir / "ci.yml.disabled",
    ]
    ci_yaml = next(
        (p.read_text(encoding="utf-8") for p in ci_candidates if p.exists()),
        None,
    )
    assert ci_yaml is not None, "CI workflow file not found (ci.yml or ci.yml.disabled)"

    optional_deps = pyproject_data["project"]["optional-dependencies"]
    extras_used_by_ci = _ci_extras(ci_yaml)

    assert extras_used_by_ci, "CI should declare at least one --extra dependency."
    missing = sorted(extra for extra in extras_used_by_ci if extra not in optional_deps)
    assert not missing, f"CI references undefined extras: {missing}"


def test_supported_optional_dependency_profiles_are_search_and_ml():
    root = _repo_root()
    pyproject_data = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))

    optional_deps = pyproject_data["project"]["optional-dependencies"]

    assert set(optional_deps) == {"search", "ml"}


def test_discovery_sklearn_error_hint_matches_supported_install_path(monkeypatch):
    real_import = builtins.__import__

    def fake_import(name, globals_=None, locals_=None, fromlist=(), level=0):
        if name.startswith("sklearn"):
            raise ModuleNotFoundError("No module named 'sklearn'")
        return real_import(name, globals_, locals_, fromlist, level)

    discovery._require_sklearn.cache_clear()
    monkeypatch.setattr(builtins, "__import__", fake_import)

    with pytest.raises(ModuleNotFoundError, match=r"mess-ai\[ml\]"):
        discovery._require_sklearn()

    discovery._require_sklearn.cache_clear()
