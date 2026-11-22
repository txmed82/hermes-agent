"""search_files must never read a leading-hyphen pattern or path as an engine flag.

Regression: a content search for ``--(image|mmproj|prompt)`` came back as
``rg: unrecognized flag --(image|mmproj|prompt)`` because the pattern was the
first positional word after the options. The same shape broke the grep
fallbacks and the zero-match probes, and a relative search root such as
``-logs`` reached ``rg``/``grep``/``find`` as an option. Every engine call must
pass the pattern through ``-e`` and keep paths behind ``--`` (or an equivalent
``./`` anchor for ``find``), on both the native and the shell transport.
"""

import pytest

from tools.environments.local import LocalEnvironment
from tools.file_operations import ShellFileOperations

HELP_LOG = (
    "usage: llama-cli [options]\n"
    "  --image FILE        path to an image file\n"
    "  --mmproj FILE       multimodal projector\n"
    "  --prompt TEXT       prompt to start generation with\n"
    "  --no-conversation   disable conversation mode\n"
    "  --help              print usage and exit\n"
    "  -v                  verbose\n"
    "plain line without flags\n"
)


@pytest.fixture
def tree(tmp_path):
    root = tmp_path / "work"
    root.mkdir()
    (root / "cli_help.log").write_text(HELP_LOG)
    (root / "notes.txt").write_text("ordinary words only\n")
    dashed = root / "-logs"
    dashed.mkdir()
    (dashed / "run.log").write_text("  --image FILE\n")
    (dashed / "--nested-dir").mkdir()
    return root


def _ops(root, monkeypatch, engine="rg", native=True):
    ops = ShellFileOperations(LocalEnvironment(cwd=str(root)), cwd=str(root))
    if engine == "grep":
        monkeypatch.setattr(ops, "_has_command", lambda command: command in ("grep", "find"))
    if not native:
        monkeypatch.setattr(ops, "_native_read_enabled", lambda: False)
    return ops


ENGINES = [
    pytest.param("rg", True, id="rg-native"),
    pytest.param("rg", False, id="rg-shell"),
    pytest.param("grep", False, id="grep"),
]

LEADING_HYPHEN_REGEXES = [
    "--(no-conversation|conversation|display-prompt|no-display-prompt|simple-io|log-disable"
    "|special|file|ctx-size|temp|no-escape|single-turn|prompt)",
    "--(image|mmproj|image-min|image-max|single-turn|prompt|file|no-conversation)",
]


@pytest.mark.parametrize("engine,native", ENGINES)
@pytest.mark.parametrize("pattern", LEADING_HYPHEN_REGEXES)
def test_leading_hyphen_regex_matches(tree, monkeypatch, engine, native, pattern):
    result = _ops(tree, monkeypatch, engine, native).search(pattern, path="cli_help.log")
    assert result.error is None, result.error
    lines = sorted(m.content.strip() for m in result.matches)
    assert any(line.startswith("--prompt") for line in lines), lines
    assert all("unrecognized" not in line for line in lines)


@pytest.mark.parametrize("engine,native", ENGINES)
def test_flag_like_pattern_is_searched_not_executed(tree, monkeypatch, engine, native):
    result = _ops(tree, monkeypatch, engine, native).search("--help", path="cli_help.log")
    assert result.error is None, result.error
    assert result.total_count == 1
    assert result.matches[0].content.strip().startswith("--help")
    # ripgrep/grep usage text would mention options never present in the fixture.
    assert "--max-columns" not in str(result.to_dict())


@pytest.mark.parametrize("engine,native", ENGINES)
def test_short_flag_literal_pattern(tree, monkeypatch, engine, native):
    result = _ops(tree, monkeypatch, engine, native).search("-v ", path="cli_help.log")
    assert result.error is None, result.error
    assert [m.content.strip() for m in result.matches] == ["-v                  verbose"]


@pytest.mark.parametrize("engine,native", ENGINES)
def test_leading_hyphen_search_root(tree, monkeypatch, engine, native):
    result = _ops(tree, monkeypatch, engine, native).search("image", path="-logs")
    assert result.error is None, result.error
    assert result.total_count == 1
    assert result.matches[0].path.endswith("run.log")


@pytest.mark.parametrize("engine,native", ENGINES)
def test_leading_hyphen_pattern_and_root(tree, monkeypatch, engine, native):
    result = _ops(tree, monkeypatch, engine, native).search("--image", path="-logs")
    assert result.error is None, result.error
    assert result.total_count == 1


@pytest.mark.parametrize("engine,native", ENGINES)
def test_ordinary_pattern_files_only_and_count(tree, monkeypatch, engine, native):
    ops = _ops(tree, monkeypatch, engine, native)
    content = ops.search("ordinary", path=".")
    assert content.error is None and content.total_count == 1
    files = ops.search("--", path=".", output_mode="files_only")
    assert files.error is None, files.error
    assert sorted(f.rsplit("/", 1)[-1] for f in files.files) == ["cli_help.log", "run.log"]
    counts = ops.search("--", path="cli_help.log", output_mode="count")
    assert counts.error is None, counts.error
    assert sum(counts.counts.values()) == 5
    globbed = ops.search("--image", path=".", file_glob="*.log")
    assert globbed.error is None, globbed.error
    assert sorted(m.path.rsplit("/", 1)[-1] for m in globbed.matches) == ["cli_help.log", "run.log"]
    assert ops.search("--image", path=".", file_glob="*.txt").total_count == 0


@pytest.mark.parametrize("engine,native", ENGINES)
def test_existing_limit_and_offset_kept(tree, monkeypatch, engine, native):
    ops = _ops(tree, monkeypatch, engine, native)
    first = ops.search("--[a-z]", path="cli_help.log", limit=2)
    assert first.error is None, first.error
    assert [m.line_number for m in first.matches] == [2, 3]
    second = ops.search("--[a-z]", path="cli_help.log", limit=2, offset=2)
    assert [m.line_number for m in second.matches] == [4, 5]


def test_grep_pruned_lane_leading_hyphen(tmp_path, monkeypatch):
    hidden = tmp_path / ".cache" / "proj"
    hidden.mkdir(parents=True)
    (hidden / "cli_help.log").write_text(HELP_LOG)
    ops = _ops(hidden, monkeypatch, "grep")
    assert ops._root_under_hidden_dir(str(hidden))
    result = ops.search("--(image|mmproj)", path=str(hidden))
    assert result.error is None, result.error
    assert result.total_count == 2


@pytest.mark.parametrize("native", [True, False])
def test_zero_match_probe_handles_leading_hyphen(tree, monkeypatch, native):
    # Exact-case search misses; the case-insensitive probe must still run (not
    # die on argv parsing) and steer toward the real hit.
    result = _ops(tree, monkeypatch, "rg", native).search("--IMAGE", path="cli_help.log")
    assert result.error is None, result.error
    assert result.total_count == 0
    assert result.warning and "case" in result.warning.lower(), result.warning


@pytest.mark.parametrize("native", [True, False])
def test_file_search_glob_and_leading_hyphen_root(tree, monkeypatch, native):
    ops = _ops(tree, monkeypatch, "rg", native)
    globbed = ops.search("*.log", path=".", target="files")
    assert globbed.error is None, globbed.error
    assert sorted(f.rsplit("/", 1)[-1] for f in globbed.files) == ["cli_help.log", "run.log"]
    # Directory supplement (find) must accept a dash-prefixed root as well.
    dirs = ops.search("--nested*", path="-logs", target="files")
    assert dirs.error is None, dirs.error
    assert dirs.files == ["-logs/--nested-dir"]
    limited = ops.search("*", path=".", target="files", limit=1)
    assert limited.error is None and len(limited.files) == 1 and limited.truncated


def test_find_fallback_file_search_leading_hyphen(tree, monkeypatch):
    ops = _ops(tree, monkeypatch, "grep")
    result = ops.search("--nested*", path="-logs", target="files")
    assert result.error is None, result.error
    assert [f.rsplit("/", 1)[-1] for f in result.files] == ["--nested-dir"]
