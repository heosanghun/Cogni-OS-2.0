import ast
import unittest
from pathlib import Path


class TestAirGapStaticContract(unittest.TestCase):
    def test_runtime_sources_do_not_import_network_clients(self):
        forbidden = {
            "requests",
            "httpx",
            "aiohttp",
            "urllib",
            "socket",
            "ftplib",
            "telnetlib",
        }
        violations = []
        for package in ("cogni_core", "cogni_flow", "cogni_os"):
            for path in Path(package).rglob("*.py"):
                tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
                for node in ast.walk(tree):
                    if isinstance(node, ast.Import):
                        roots = {alias.name.split(".")[0] for alias in node.names}
                    elif isinstance(node, ast.ImportFrom):
                        roots = {(node.module or "").split(".")[0]}
                    else:
                        continue
                    if roots & forbidden:
                        violations.append(
                            f"{path}:{node.lineno}:{sorted(roots & forbidden)}"
                        )
        self.assertEqual(violations, [])


if __name__ == "__main__":
    unittest.main()
