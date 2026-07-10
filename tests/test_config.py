import unittest
from importlib import resources
from pathlib import Path
import tempfile

from cogni_os.config import MAX_VRAM_GIB, load_config


class TestConfig(unittest.TestCase):
    def test_default_config_is_offline_and_complete(self):
        config = load_config()
        self.assertTrue(config.offline)
        self.assertEqual(config.section("cts")["width"], 3)
        packaged = resources.files("cogni_os").joinpath("default.toml").read_bytes()
        source = (
            Path(__file__).resolve().parents[1] / "config" / "default.toml"
        ).read_bytes()
        self.assertEqual(packaged, source)

    def test_vram_limit_cannot_exceed_absolute_ceiling(self):
        default_path = Path(__file__).resolve().parents[1] / "config" / "default.toml"
        source = default_path.read_text(encoding="utf-8")
        source = source.replace("vram_limit_gib = 16.7", "vram_limit_gib = 16.8")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "unsafe.toml"
            path.write_text(source, encoding="utf-8")
            with self.assertRaisesRegex(ValueError, str(MAX_VRAM_GIB)):
                load_config(path)

    def test_safety_critical_bounds_cannot_be_disabled(self):
        default_path = Path(__file__).resolve().parents[1] / "config" / "default.toml"
        original = default_path.read_text(encoding="utf-8")
        unsafe_variants = (
            original.replace(
                "require_kernel_sandbox_for_production = true",
                "require_kernel_sandbox_for_production = false",
            ),
            original.replace("latent_capacity = 256", "latent_capacity = 999"),
            original.replace("max_operator_norm = 0.10", "max_operator_norm = 0.99"),
        )
        with tempfile.TemporaryDirectory() as directory:
            for index, source in enumerate(unsafe_variants):
                with self.subTest(index=index):
                    path = Path(directory) / f"unsafe-{index}.toml"
                    path.write_text(source, encoding="utf-8")
                    with self.assertRaises(ValueError):
                        load_config(path)


if __name__ == "__main__":
    unittest.main()
