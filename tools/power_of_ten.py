"""Check Python code against the parts of NASA/JPL's "Power of Ten" rules that need a custom tool.

The Power of Ten (Gerard Holzmann, NASA/JPL, 2006) are ten rules for writing
safety-critical code that is simple enough to verify. Most of them map onto
standard tools (see LEARNING.md); this script checks the rest:

  R1  No recursion: a function may not call itself (direct recursion).
  R2  Loops need a visible bound: no `while True` / `while 1`.
  R4  Functions fit on one printed page: at most 60 lines.
  R6  Data in the smallest scope: no `global`/`nonlocal`, and no mutable
      containers (list, dict, set) at module level. Use tuples, frozensets
      or constants instead.

Usage:  python tools/power_of_ten.py src tools [more files or folders]
Exit code 0 when clean, 1 when there are violations.
"""

from __future__ import annotations

import ast
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import NamedTuple

MAX_FUNCTION_LINES = 60
MAX_FILES = 10_000  # R2: even this loop has a bound
MUTABLE_NODES = (ast.List, ast.Dict, ast.Set, ast.ListComp, ast.DictComp, ast.SetComp)
FunctionNode = ast.FunctionDef | ast.AsyncFunctionDef


class Violation(NamedTuple):
    path: Path
    line: int
    rule: str
    message: str

    def __str__(self) -> str:
        return f"{self.path}:{self.line}: {self.rule} {self.message}"


def calls_itself(func: FunctionNode) -> int | None:
    """Line of the first call `func` makes to itself, or None."""
    for node in ast.walk(func):
        if not isinstance(node, ast.Call):
            continue
        target = node.func
        if isinstance(target, ast.Name) and target.id == func.name:
            return node.lineno
        if (
            isinstance(target, ast.Attribute)
            and target.attr == func.name
            and isinstance(target.value, ast.Name)
            and target.value.id in {"self", "cls"}
        ):
            return node.lineno
    return None


def check_function(path: Path, func: FunctionNode) -> Iterator[Violation]:
    length = (func.end_lineno or func.lineno) - func.lineno + 1
    if length > MAX_FUNCTION_LINES:
        yield Violation(
            path, func.lineno, "R4", f"'{func.name}' is {length} lines (max {MAX_FUNCTION_LINES})"
        )
    line = calls_itself(func)
    if line is not None:
        yield Violation(path, line, "R1", f"'{func.name}' calls itself (recursion)")


def is_unbounded_while(node: ast.While) -> bool:
    test = node.test
    return isinstance(test, ast.Constant) and bool(test.value)


def check_module_level(path: Path, tree: ast.Module) -> Iterator[Violation]:
    for stmt in tree.body:
        value = stmt.value if isinstance(stmt, ast.Assign | ast.AnnAssign) else None
        if isinstance(value, MUTABLE_NODES):
            yield Violation(
                path,
                stmt.lineno,
                "R6",
                "mutable container at module level (use a tuple, frozenset or constant)",
            )


def check_tree(path: Path, tree: ast.Module) -> Iterator[Violation]:
    yield from check_module_level(path, tree)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            yield from check_function(path, node)
        elif isinstance(node, ast.While) and is_unbounded_while(node):
            yield Violation(path, node.lineno, "R2", "unbounded `while True` loop")
        elif isinstance(node, ast.Global | ast.Nonlocal):
            keyword = "global" if isinstance(node, ast.Global) else "nonlocal"
            yield Violation(path, node.lineno, "R6", f"`{keyword}` statement")


def python_files(targets: list[str]) -> list[Path]:
    files: list[Path] = []
    for target in map(Path, targets):
        found = sorted(target.rglob("*.py")) if target.is_dir() else [target]
        files.extend(f for f in found if f.suffix == ".py")
    if len(files) > MAX_FILES:
        raise SystemExit(f"Refusing to check more than {MAX_FILES} files.")
    return files


def main(argv: list[str]) -> int:
    if not argv:
        sys.stderr.write("usage: python tools/power_of_ten.py PATH [PATH ...]\n")
        return 2
    violations: list[Violation] = []
    for path in python_files(argv):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        violations.extend(check_tree(path, tree))
    for violation in violations:
        sys.stdout.write(f"{violation}\n")
    if violations:
        sys.stdout.write(f"\n{len(violations)} Power of Ten violation(s).\n")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
