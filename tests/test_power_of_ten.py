import ast
from pathlib import Path

from power_of_ten import check_tree, main


def rules(code: str) -> list[str]:
    return [v.rule for v in check_tree(Path("x.py"), ast.parse(code))]


def test_clean_code_passes():
    assert rules("X = (1, 2)\n\ndef f(n):\n    for i in range(n):\n        print(i)\n") == []


def test_recursion_is_flagged():
    assert rules("def f(n):\n    return f(n - 1)\n") == ["R1"]


def test_method_recursion_is_flagged():
    assert rules("class A:\n    def f(self):\n        return self.f()\n") == ["R1"]


def test_unbounded_while_is_flagged():
    assert rules("def f():\n    while True:\n        pass\n") == ["R2"]


def test_long_function_is_flagged():
    body = "\n".join("    x = 1" for _ in range(60))
    assert rules(f"def f():\n{body}\n") == ["R4"]


def test_mutable_module_level_data_is_flagged():
    assert rules("CONFIG = {'a': 1}\nNAMES = [1]\n") == ["R6", "R6"]


def test_global_statement_is_flagged():
    assert rules("X = 1\ndef f():\n    global X\n    X = 2\n") == ["R6"]


def test_main_reports_and_fails(tmp_path, capsys):
    bad = tmp_path / "bad.py"
    bad.write_text("def f():\n    return f()\n")
    assert main([str(tmp_path)]) == 1
    assert "R1" in capsys.readouterr().out
    assert main([str(Path(__file__).parent.parent / "src")]) == 0


def test_main_needs_arguments():
    assert main([]) == 2
