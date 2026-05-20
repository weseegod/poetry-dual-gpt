#!/usr/bin/env python
"""
Test runner — discovers and runs all tests in the tests/ directory.

Usage:
    python tests/tests.py           # run all
    python tests/tests.py -v        # verbose output
    python tests/tests.py TestCleanText  # run specific test class
"""

import sys
import unittest
from pathlib import Path

if __name__ == "__main__":
    # Ensure src/ is importable for all test modules
    src_path = Path(__file__).parent.parent / "src"
    sys.path.insert(0, str(src_path))

    # Discover all tests in the tests/ directory
    loader = unittest.TestLoader()
    start_dir = Path(__file__).parent
    suite = loader.discover(start_dir=str(start_dir), pattern="test_*.py")

    # Run
    verbosity = 2 if "-v" in sys.argv else 1
    runner = unittest.TextTestRunner(verbosity=verbosity)
    result = runner.run(suite)

    # Exit with non-zero if failures
    sys.exit(0 if result.wasSuccessful() else 1)
