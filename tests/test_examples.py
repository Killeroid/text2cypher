"""The runnable examples keep working: the documentation includes them."""

import subprocess
import sys
from pathlib import Path

import pytest

EXAMPLES = Path(__file__).parent.parent / "examples"


@pytest.mark.db  # it starts its own embedded database
def test_the_quickstart_example_runs_and_shows_the_correction():
    done = subprocess.run(
        [sys.executable, str(EXAMPLES / "quickstart.py")],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )

    assert done.returncode == 0, done.stderr
    assert "status: valid" in done.stdout
    assert "opposite direction" in done.stdout  # the validator's finding on attempt 1
    assert "attempt 2: MATCH (p:Person)-[:ACTED_IN]->" in done.stdout
    assert "Keanu Reeves" in done.stdout
