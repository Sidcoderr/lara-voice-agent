"""Check that nothing secret is about to be published.

    uv run python check_secrets.py

It scans every file in the project (not .venv, .git or caches) for:
  1. the exact values of the secrets in your .env (found by name: KEY, TOKEN, SECRET, PASSWORD), and
  2. text shaped like an API key from the services this project uses.

It NEVER prints a secret. It shows only the file, the line number, and a masked hint such
as "AIza... (39 characters)". A finding inside a file that .gitignore keeps out of the
repository is reported as safe. A finding anywhere else is a problem. It also checks that
.env, log files and the demo database are ignored.

The last line says OK TO PUBLISH or DO NOT PUBLISH YET. The exit code is 1 for the second.
"""

import fnmatch
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

SKIP_DIRS = {".venv", "venv", ".git", "__pycache__", "node_modules", ".pytest_cache", ".mypy_cache", ".ruff_cache"}
SKIP_FILES = {"uv.lock"}  # full of hashes that look like keys
MAX_BYTES = 5_000_000
SECRET_NAME = re.compile(r"(KEY|TOKEN|SECRET|PASSWORD)", re.I)
PLACEHOLDER = re.compile(r"(your_|xxxx|changeme|example|placeholder|<)", re.I)

# Shapes of keys from the services this project uses. Deliberately a little broad.
KEY_SHAPES = [
    ("a Google key", re.compile(r"AIza[0-9A-Za-z_\-]{30,}")),
    ("a Google key", re.compile(r"\bAQ\.[A-Za-z0-9_\-]{30,}")),
    ("an ElevenLabs or OpenAI-style key", re.compile(r"\bsk[_\-][A-Za-z0-9_\-]{20,}")),
    ("a Groq key", re.compile(r"\bgsk_[A-Za-z0-9]{30,}")),
    ("a Deepgram-style key (40 hex characters)", re.compile(r"\b[a-f0-9]{40}\b")),
]

# These must never be published, so they must be covered by .gitignore.
MUST_BE_IGNORED = [".env", "run.log", "server.log", "data/surgery.db"]


@dataclass
class Finding:
    path: str
    line: int
    what: str
    hint: str  # masked: never the secret itself


@dataclass
class Report:
    scanned: int = 0
    safe: list = field(default_factory=list)  # found, but the file is kept out of the repository
    problems: list = field(default_factory=list)  # found in a file that would be published
    not_ignored: list = field(default_factory=list)  # files that should be ignored but are not
    gitignore_missing: bool = False

    @property
    def ok(self) -> bool:
        return not self.problems and not self.not_ignored and not self.gitignore_missing


def _mask(text: str) -> str:
    return f"{text[:4]}... ({len(text)} characters)"


def _read_patterns(root: Path) -> list[str]:
    path = root / ".gitignore"
    if not path.exists():
        return []
    lines = [line.strip() for line in path.read_text(encoding="utf-8", errors="replace").splitlines()]
    return [line for line in lines if line and not line.startswith("#") and not line.startswith("!")]


def is_ignored(rel_path: str, patterns: list[str]) -> bool:
    """A small version of .gitignore matching: enough for the patterns this project uses."""
    rel = rel_path.replace("\\", "/")
    if rel.startswith("./"):
        rel = rel[2:]  # (not lstrip: that would also strip the leading dot from ".env")
    parts = rel.split("/")
    for pattern in patterns:
        pat = pattern.replace("\\", "/")
        if pat.endswith("/"):  # a directory: ignore anything inside it
            name = pat.strip("/").lstrip("/")
            if any(fnmatch.fnmatch(part, name) for part in parts[:-1]):
                return True
        elif "/" in pat.strip("/"):  # a path such as data/surgery.db
            if fnmatch.fnmatch(rel, pat.lstrip("/")):
                return True
        else:  # a name or wildcard such as .env or *.log, matched against every part of the path
            if any(fnmatch.fnmatch(part, pat.lstrip("/")) for part in parts):
                return True
    return False


def _parse_env(text: str) -> dict:
    """Read KEY=VALUE lines. Handles comments, quotes, 'export' and Windows line endings."""
    values = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[len("export "):].lstrip()
        name, _, value = line.partition("=")
        value = value.strip()
        if value[:1] in ("'", '"') and value[-1:] == value[:1] and len(value) >= 2:
            value = value[1:-1]
        else:
            value = re.split(r"\s+#", value, maxsplit=1)[0].strip()  # an unquoted trailing comment
        values[name.strip()] = value
    return values


def _env_secrets(root: Path) -> list[str]:
    path = root / ".env"
    if not path.exists():
        return []
    values = []
    for name, value in _parse_env(path.read_text(encoding="utf-8", errors="replace")).items():
        if value and SECRET_NAME.search(name) and len(value) >= 8 and not PLACEHOLDER.search(value):
            values.append(value)
    return values


def _files(root: Path):
    for path in sorted(root.rglob("*")):
        if not path.is_file() or any(part in SKIP_DIRS for part in path.relative_to(root).parts):
            continue
        if path.name in SKIP_FILES or path.name == ".env":
            continue  # .env is where the secrets are meant to live
        if path.stat().st_size > MAX_BYTES:
            continue
        yield path


def scan(root: Path) -> Report:
    root = Path(root)
    patterns = _read_patterns(root)
    report = Report(gitignore_missing=not (root / ".gitignore").exists())
    secrets = _env_secrets(root)
    for example in MUST_BE_IGNORED:
        if not is_ignored(example, patterns):
            report.not_ignored.append(example)
    for path in _files(root):
        data = path.read_bytes()
        if b"\x00" in data[:1024]:
            continue  # a binary file
        report.scanned += 1
        rel = path.relative_to(root).as_posix()
        target = report.safe if is_ignored(rel, patterns) else report.problems
        for number, line in enumerate(data.decode("utf-8", errors="replace").splitlines(), 1):
            hit = next((secret for secret in secrets if secret in line), None)
            if hit:
                target.append(Finding(rel, number, "the exact value of a secret from your .env", _mask(hit)))
                continue
            for label, shape in KEY_SHAPES:
                match = shape.search(line)
                if match:
                    target.append(Finding(rel, number, f"text shaped like {label}", _mask(match.group(0))))
                    break
    return report


def render(report: Report) -> str:
    lines = [f"Scanned {report.scanned} files.", ""]
    if report.gitignore_missing:
        lines += ["There is no .gitignore file, so nothing is being kept out of the repository.", ""]
    if report.not_ignored:
        lines.append("These files must be ignored by .gitignore but are NOT:")
        lines += [f"  - {name}" for name in report.not_ignored]
        lines.append("")
    if report.problems:
        lines.append("PROBLEMS: these would be published:")
        lines += [f"  - {f.path}, line {f.line}: {f.what} [{f.hint}]" for f in report.problems]
        lines.append("")
    if report.safe:
        lines.append("Safe: found in files that .gitignore keeps out of the repository:")
        lines += [f"  - {f.path}, line {f.line}: {f.what}" for f in report.safe]
        lines.append("")
    lines.append("RESULT: OK TO PUBLISH." if report.ok else "RESULT: DO NOT PUBLISH YET. Fix the items above, then run this again.")
    return "\n".join(lines)


if __name__ == "__main__":
    result = scan(Path(__file__).parent)
    print(render(result))
    sys.exit(0 if result.ok else 1)