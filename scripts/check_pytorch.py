"""
Check if PyTorch and required dependencies are available.

This script verifies the environment is ready for full self-play training.
"""

import sys


def check_dependency(name: str, import_path: str) -> tuple[bool, str]:
    """
    Check if a dependency is available.

    Args:
        name: Human-readable name
        import_path: Python import path

    Returns:
        (success, message): Status and version/error message
    """
    try:
        module = __import__(import_path)
        version = getattr(module, "__version__", "unknown")
        return True, f"v{version}"
    except ImportError as e:
        return False, str(e)


def main() -> bool:
    """
    Check all required dependencies.

    Returns:
        True if all dependencies available, False otherwise
    """
    print("=" * 70)
    print("PyTorch Environment Check")
    print("=" * 70)

    dependencies = {
        "PyTorch": "torch",
        "NumPy": "numpy",
        "Pandas": "pandas",
        "Scikit-learn": "sklearn",
        "Matplotlib": "matplotlib",
        "Transformers": "transformers",
        "tqdm": "tqdm",
        "loguru": "loguru",
    }

    results: dict[str, tuple[bool, str]] = {}
    all_available = True

    for name, import_path in dependencies.items():
        success, message = check_dependency(name, import_path)
        results[name] = (success, message)
        all_available = all_available and success

        status = "✅" if success else "❌"
        print(f"{status} {name:20s}: {message}")

    print("\n" + "=" * 70)

    if all_available:
        print("✅ ALL DEPENDENCIES AVAILABLE")
        print("\nYou can now run full self-play training with real metrics.")
        print("Expected MAE: 0.5-1.5 kWh (not NaN)")
        return True
    else:
        print("❌ MISSING DEPENDENCIES")
        print("\nTo install missing packages:")
        print("  Option 1 (poetry): poetry install")
        print("  Option 2 (pip):    pip install torch transformers")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
