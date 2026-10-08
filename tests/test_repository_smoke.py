"""Fast, credential-free repository smoke tests."""

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def test_default_config_is_valid_yaml():
    """The committed default configuration should parse as YAML."""
    with (ROOT / "config" / "default_config.yaml").open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    assert isinstance(config, dict)
    assert config


def test_ticker_universe_is_valid_yaml():
    """The committed ticker universe should parse as YAML."""
    with (ROOT / "config" / "symbol_universe.yaml").open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    assert isinstance(config, dict)
    assert config


def test_src_package_exists():
    """The application source package must exist."""
    assert (ROOT / "src" / "__init__.py").is_file()


def test_core_packages_exist():
    """All core sub-packages must exist with __init__.py files."""
    packages = [
        "src/data",
        "src/features",
        "src/models",
        "src/optimizer",
        "src/evaluation",
        "src/backtesting",
        "src/utils",
    ]
    for pkg in packages:
        assert (ROOT / pkg).is_dir(), f"Missing package directory: {pkg}"
        assert (ROOT / pkg / "__init__.py").is_file(), f"Missing __init__.py in {pkg}"


def test_license_file_exists():
    """The repository root must contain a valid MIT license file."""
    license_file = ROOT / "LICENSE"
    assert license_file.is_file(), "Root LICENSE file missing"
    text = license_file.read_text(encoding="utf-8")
    assert "MIT License" in text
    assert "Andrew C. Photinakis" in text


def test_pre_commit_config_is_valid_yaml():
    """The pre-commit configuration file must exist and be valid YAML."""
    config_file = ROOT / ".pre-commit-config.yaml"
    assert config_file.is_file(), "Missing .pre-commit-config.yaml"
    with config_file.open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    assert isinstance(config, dict)
    assert "repos" in config


def test_github_workflows_are_valid_yaml():
    """GitHub Actions workflows must exist and parse as valid YAML."""
    workflows = [
        ROOT / ".github" / "workflows" / "gitleaks.yml",
        ROOT / ".github" / "workflows" / "python-tests.yml",
        ROOT / ".github" / "dependabot.yml",
    ]
    for wf in workflows:
        assert wf.is_file(), f"Missing workflow or config file: {wf}"
        with wf.open(encoding="utf-8") as stream:
            data = yaml.safe_load(stream)
        assert isinstance(data, dict), f"Failed to parse {wf} as dict"
