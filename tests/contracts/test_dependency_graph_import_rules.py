from __future__ import annotations

import ast
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:  # pragma: no cover - import guard
    sys.path.insert(0, str(PROJECT_ROOT))


RULES: dict[str, set[str]] = {
    "infra": {"research", "execution", "training"},
    "execution": {"research", "training"},
    "research": {"execution"},
    "scripts": {"execution"},
    "orchestration": {"infra"},
}


def test_dependency_graph_import_rules() -> None:
    violations: list[str] = []
    for source, forbidden in RULES.items():
        source_root = PROJECT_ROOT / source
        if not source_root.exists():
            continue
        for path in source_root.rglob("*.py"):
            violations.extend(_scan_file(path, source, forbidden))
    assert not violations, "Import boundary violations:\n" + "\n".join(sorted(violations))


def _scan_file(path: Path, source: str, forbidden: set[str]) -> list[str]:
    text = path.read_text(encoding="utf-8")
    tree = ast.parse(text, filename=str(path))
    visitor = _ImportVisitor()
    visitor.visit(tree)
    violations: list[str] = []
    rel_path = path.relative_to(PROJECT_ROOT)
    for module in visitor.modules:
        root = module.split(".")[0]
        if root in forbidden:
            violations.append(f"{rel_path}: imports {module} (forbidden in {source})")
    return violations


class _ImportVisitor(ast.NodeVisitor):
    def __init__(self) -> None:
        self.modules: list[str] = []
        self._type_check_depth = 0

    def visit_If(self, node: ast.If) -> None:
        if _is_type_checking_guard(node.test):
            self._type_check_depth += 1
            for child in node.body:
                self.visit(child)
            self._type_check_depth -= 1
            for child in node.orelse:
                self.visit(child)
        else:
            self.generic_visit(node)

    def visit_Import(self, node: ast.Import) -> None:
        if self._type_check_depth:
            return
        for alias in node.names:
            self.modules.append(alias.name)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        if self._type_check_depth or node.level:
            return
        if node.module:
            self.modules.append(node.module)


def _is_type_checking_guard(node: ast.AST) -> bool:
    if isinstance(node, ast.Name) and node.id == "TYPE_CHECKING":
        return True
    if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
        return node.value.id == "typing" and node.attr == "TYPE_CHECKING"
    return False
