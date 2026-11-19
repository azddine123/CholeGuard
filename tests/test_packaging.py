import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def test_no_source_file_is_gitignored():
    sources = [p.relative_to(REPO).as_posix() for p in (REPO / "src").rglob("*.py")]
    sources += [p.relative_to(REPO).as_posix() for p in (REPO / "scripts").glob("*") if p.is_file()]
    sources += [p.relative_to(REPO).as_posix() for p in (REPO / "tests").rglob("*.py")]
    r = subprocess.run(["git", "check-ignore", *sources], cwd=REPO, capture_output=True, text=True)
    assert r.stdout.strip() == "", f"ignored source files:\n{r.stdout}"
