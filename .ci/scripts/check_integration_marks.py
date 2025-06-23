import ast
import sys
from pathlib import Path

missing_marks = []


def has_integration_marker(decorator_list):
    for decorator in decorator_list:
        if isinstance(decorator, ast.Call):
            func = decorator.func
        else:
            func = decorator

        if isinstance(func, ast.Attribute):
            if (
                func.attr == "integration"
                and isinstance(func.value, ast.Name)
                and func.value.id == "mark"
            ):
                return True
            if (
                func.attr == "integration"
                and isinstance(func.value, ast.Attribute)
                and func.value.attr == "mark"
            ):
                return True
    return False


for file in Path("pm25ml").rglob("*__it.py"):
    tree = ast.parse(file.read_text(), filename=str(file))
    has_pytestmark = any(
        isinstance(node, ast.Assign)
        and any(
            target.id == "pytestmark" for target in node.targets if isinstance(target, ast.Name)
        )
        for node in tree.body
    )

    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name.startswith("test"):
            if not has_integration_marker(node.decorator_list) and not has_pytestmark:
                missing_marks.append(f"{file}:{node.lineno} {node.name}")

if missing_marks:
    print("❌ Missing @pytest.mark.integration or pytestmark:")
    for line in missing_marks:
        print("  -", line)
    sys.exit(1)
