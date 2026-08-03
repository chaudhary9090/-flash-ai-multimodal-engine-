"""
Phase E: Sandboxed Python Code Execution Service
------------------------------------------------
Provides safe, isolated Python code execution with AST pre-inspection verification,
restricted imports, isolated scratch directory, and 5-second process timeout.
"""

import ast
import os
import sys
import subprocess
import tempfile
from typing import Tuple
from app.core.logging import logger
from app.core.config import settings


class SecurityASTValidator(ast.NodeVisitor):
    """AST Inspector enforcing strict sandboxing rules before execution."""

    BLOCKED_IMPORTS = {"os", "subprocess", "sys", "socket", "shutil", "urllib", "requests", "http", "ftplib", "builtins", "importlib"}
    BLOCKED_FUNCTIONS = {"eval", "exec", "__import__", "compile"}

    def __init__(self):
        self.errors = []

    def visit_Import(self, node: ast.Import):
        for alias in node.names:
            name = alias.name.split(".")[0]
            if name in self.BLOCKED_IMPORTS:
                self.errors.append(f"Import of restricted module '{name}' is forbidden for security safety.")
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom):
        if node.module:
            name = node.module.split(".")[0]
            if name in self.BLOCKED_IMPORTS:
                self.errors.append(f"Import from restricted module '{name}' is forbidden for security safety.")
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call):
        if isinstance(node.func, ast.Name):
            func_name = node.func.id
            if func_name in self.BLOCKED_FUNCTIONS:
                self.errors.append(f"Call to restricted function '{func_name}()' is forbidden.")
            elif func_name == "open":
                # Check for write/append modes in open()
                if len(node.args) > 1 and isinstance(node.args[1], ast.Constant):
                    mode = str(node.args[1].value)
                    if any(m in mode for m in ["w", "a", "+"]):
                        self.errors.append("Writing to filesystem using open() is forbidden.")
        self.generic_visit(node)


class CodeExecutorService:
    """Runs validated Python code in an isolated subprocess."""

    def __init__(self):
        self.scratch_dir = os.path.join(settings.BASE_DIR, "scratch")
        os.makedirs(self.scratch_dir, exist_ok=True)

    def validate_code_ast(self, code: str) -> Tuple[bool, str]:
        """Parses Python syntax tree and verifies security constraints."""
        try:
            tree = ast.parse(code)
            validator = SecurityASTValidator()
            validator.visit(tree)
            if validator.errors:
                return False, "Security Error: " + " ".join(validator.errors)
            return True, ""
        except SyntaxError as e:
            return False, f"Syntax Error in Python code: {str(e)}"

    def execute_python(self, code: str, timeout_seconds: float = 5.0) -> str:
        """Executes AST-validated code inside an isolated subprocess."""
        is_valid, sec_error = self.validate_code_ast(code)
        if not is_valid:
            logger.warning(f"AST Sandbox Rejected Code: {sec_error}")
            return sec_error

        temp_script_path = os.path.join(self.scratch_dir, "temp_exec.py")
        try:
            with open(temp_script_path, "w", encoding="utf-8") as f:
                f.write(code)

            # Execute with python -I (isolated mode)
            cmd = [sys.executable, "-I", temp_script_path]
            process = subprocess.run(
                cmd,
                cwd=self.scratch_dir,
                capture_output=True,
                text=True,
                timeout=timeout_seconds
            )

            stdout = process.stdout.strip()
            stderr = process.stderr.strip()

            if process.returncode != 0:
                return f"[PYTHON CODE EXECUTION ERROR]\n{stderr if stderr else 'Execution failed with error code ' + str(process.returncode)}"

            return f"[SANDBOXED PYTHON RESULT]\n{stdout if stdout else 'Code executed successfully (no output produced).'}"

        except subprocess.TimeoutExpired:
            return f"[EXECUTION TIMEOUT]\nExecution aborted: Process exceeded the maximum safety limit of {timeout_seconds} seconds."
        except Exception as e:
            logger.error(f"Error in sandboxed code execution: {e}")
            return f"[EXECUTION ERROR]\nFailed to execute code: {str(e)}"
        finally:
            if os.path.exists(temp_script_path):
                try:
                    os.remove(temp_script_path)
                except Exception:
                    pass


code_executor_service = CodeExecutorService()
