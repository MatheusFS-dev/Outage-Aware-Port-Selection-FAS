"""Regression checks for the standalone dissertation companion layout."""
import importlib.util
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]

class ExtractionTests(unittest.TestCase):
    def test_nas_inputs_resolve_inside_companion(self):
        path = ROOT / "studies/chapter06_mobilenet_nas/scripts/analysis/nas/analyze_all_regime_nas_dissertation.py"
        self.assertTrue(path.is_file(), str(path))
        spec = importlib.util.spec_from_file_location("standalone_nas", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self.assertEqual(module.REPOSITORY_ROOT, ROOT)
        self.assertTrue(module.RESULTS_ROOT.is_relative_to(ROOT / "studies/chapter06_mobilenet_nas/results"))
        self.assertTrue(module.RESULTS_ROOT.is_dir())

    def test_no_executable_paths_point_at_original_checkout(self):
        import ast
        bad = []
        for path in (ROOT / "studies").rglob("*.py"):
            for node in ast.walk(ast.parse(path.read_text())):
                if isinstance(node, ast.Constant) and isinstance(node.value, str):
                    if node.value.startswith(("/home/", "/media/", "D:/")):
                        bad.append((str(path.relative_to(ROOT)), node.value))
        self.assertEqual(bad, [])

    def test_firmware_discovers_all_seventy_selected_models(self):
        import sys
        directory = ROOT / "studies/chapter06_mobilenet_nas/firmware"
        sys.path.insert(0, str(directory))
        spec = importlib.util.spec_from_file_location("companion_firmware", directory / "generate_per_regime_uf2.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self.assertEqual(module.TEMPLATE_DIR, directory)
        self.assertEqual(len(module.list_selected_models()), 70)
        spec = importlib.util.spec_from_file_location("companion_benchmark", directory / "run_uf2_benchmarks.py")
        benchmark = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(benchmark)
        self.assertTrue(benchmark.UF2_ROOT.is_relative_to(directory))
        self.assertTrue(benchmark.RESULT_ROOT.is_relative_to(directory.parent / "results"))

    def test_channel_data_is_external(self):
        import sys
        sys.path.insert(0, str(ROOT / "src"))
        from utils.data import DEFAULT_DATASET_ROOT_PATH, DEFAULT_KMS_RUNTIME_CACHE_PATH
        self.assertEqual(Path(DEFAULT_DATASET_ROOT_PATH), ROOT / "data/kappa-mu_shadowed")
        self.assertEqual(Path(DEFAULT_KMS_RUNTIME_CACHE_PATH), ROOT / "data/kms_runtime_cache_float32")
        self.assertFalse(list((ROOT / "data").glob("**/*.mat")))

if __name__ == "__main__":
    unittest.main()
