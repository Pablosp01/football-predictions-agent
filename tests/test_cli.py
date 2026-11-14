import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _run(*args: str) -> subprocess.CompletedProcess:
    env = {"PYTHONPATH": str(ROOT / "src"), "PATH": "/usr/bin:/bin:/usr/local/bin"}
    import os
    env.update({k: v for k, v in os.environ.items() if k not in env})
    env["FOOTBALL_DATA_API_KEY"] = ""  # force offline sample data
    return subprocess.run([sys.executable, "-m", "football_agent.cli", *args],
                          capture_output=True, text=True, cwd=ROOT, env=env)


def test_cli_predict_smoke():
    r = _run("predict", "--home", "Arsenal", "--away", "Chelsea")
    assert r.returncode == 0, r.stderr
    assert "Arsenal vs Chelsea" in r.stdout
    assert "Home win" in r.stdout
    assert "Expected goals" in r.stdout
    assert "BTTS" in r.stdout


def test_cli_backtest_smoke():
    r = _run("backtest", "--league", "PL")
    assert r.returncode == 0, r.stderr
    assert "Accuracy" in r.stdout and "Brier score" in r.stdout


def test_cli_agent_writes_files(tmp_path):
    r = _run("agent", "--home", "Arsenal", "Liverpool", "--away", "Chelsea", "Man City",
             "--out-dir", str(tmp_path / "out"))
    assert r.returncode == 0, r.stderr
    assert (tmp_path / "out" / "predictions.csv").exists()
    assert (tmp_path / "out" / "predictions.md").exists()
    assert "Arsenal" in (tmp_path / "out" / "predictions.csv").read_text()
