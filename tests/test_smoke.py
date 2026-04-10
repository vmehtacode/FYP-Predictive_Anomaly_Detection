"""Smoke tests to ensure basic functionality and CI pipeline."""

import sys
from pathlib import Path


def test_basic_arithmetic():
    """Basic test to ensure pytest is working."""
    assert 2 + 2 == 4
    assert 10 * 5 == 50
    assert 100 / 4 == 25.0


def test_python_version():
    """Ensure we're running Python 3.11+."""
    assert sys.version_info >= (3, 11), f"Python 3.11+ required, got {sys.version_info}"


def test_project_structure():
    """Verify basic project structure exists."""
    project_root = Path(__file__).parent.parent

    # Check main directories exist
    assert (project_root / "src").exists(), "src/ directory missing"
    assert (project_root / "docs").exists(), "docs/ directory missing"
    assert (project_root / "data").exists(), "data/ directory missing"
    assert (project_root / "tests").exists(), "tests/ directory missing"

    # Check key files exist
    assert (project_root / "README.md").exists(), "README.md missing"
    assert (project_root / "pyproject.toml").exists(), "pyproject.toml missing"
    assert (project_root / "dvc.yaml").exists(), "dvc.yaml missing"
    assert (project_root / ".gitignore").exists(), ".gitignore missing"


def test_stdlib_imports():
    """Test importing standard library modules."""
    import datetime
    import json
    import math
    import os
    import pathlib

    # Basic usage to ensure imports work
    assert json.loads('{"test": true}')["test"] is True
    assert os.path.exists(".")
    assert isinstance(datetime.datetime.now(), datetime.datetime)
    assert pathlib.Path(".").exists()
    assert math.pi > 3.14


def test_data_directories():
    """Verify data directory structure."""
    project_root = Path(__file__).parent.parent
    data_dir = project_root / "data"

    assert (data_dir / "raw").exists(), "data/raw/ directory missing"
    assert (data_dir / "processed").exists(), "data/processed/ directory missing"
    assert (data_dir / "derived").exists(), "data/derived/ directory missing"


def test_docs_structure():
    """Verify documentation structure."""
    project_root = Path(__file__).parent.parent
    docs_dir = project_root / "docs"

    expected_docs = [
        "datasets.md",
        "data_governance.md",
        "experiments.md",
        "selfplay_design.md",
        "feeder_eval.md",
    ]

    for doc in expected_docs:
        assert (docs_dir / doc).exists(), f"Documentation file {doc} missing"


def test_github_workflows():
    """Verify GitHub workflows exist."""
    project_root = Path(__file__).parent.parent
    workflows_dir = project_root / ".github" / "workflows"

    assert workflows_dir.exists(), ".github/workflows/ directory missing"
    assert (workflows_dir / "ci.yml").exists(), "CI workflow missing"


class TestStringOperations:
    """Group of tests for string operations (example test class)."""

    def test_string_concatenation(self):
        """Test string concatenation."""
        assert "hello" + " " + "world" == "hello world"

    def test_string_formatting(self):
        """Test string formatting."""
        name = "Energy Forecasting"
        assert f"Project: {name}" == "Project: Energy Forecasting"
        assert f"Project: {name}" == "Project: Energy Forecasting"

    def test_string_methods(self):
        """Test various string methods."""
        text = "Self-Play Energy Forecasting"
        assert text.lower().startswith("self")
        assert "Energy" in text
        assert len(text.split()) == 3
