"""Tests for the common-CPU architecture benchmark."""

# Standalone companion paths, independent of the original checkout.
import sys as _companion_sys
from pathlib import Path as _CompanionPath
_COMPANION_ROOT = next(p for p in _CompanionPath(__file__).resolve().parents if (p / "pyproject.toml").is_file())
for _companion_path in (_COMPANION_ROOT, _COMPANION_ROOT / "src"):
    if str(_companion_path) not in _companion_sys.path:
        _companion_sys.path.insert(0, str(_companion_path))

import unittest
from unittest.mock import patch

import tensorflow as tf

from studies.chapter08_policy_model_comparison.scripts.train import benchmark_cpu_architectures
from studies.chapter08_policy_model_comparison.scripts.train.benchmark_cpu_architectures import (
    OBSERVATION_BUDGETS,
    WCNPS_2025_CONFIGS,
    build_wcnps_2025_cnn,
    parse_proc_status_rss_kib,
    summarize_peak_rss,
)


EXPECTED_CONFIGS = {
    3: {
        "filters": (192, 384),
        "kernels": (3, 3),
        "pools": (1, 1),
        "activations": ("relu", "relu"),
    },
    4: {
        "filters": (160, 256),
        "kernels": (4, 5),
        "pools": (1, 2),
        "activations": ("relu", "relu"),
    },
    5: {
        "filters": (512,),
        "kernels": (5,),
        "pools": (4,),
        "activations": ("swish",),
    },
    6: {
        "filters": (192, 160, 96),
        "kernels": (5, 1, 2),
        "pools": (1, 1, 1),
        "activations": ("leaky_relu", "relu", "elu"),
    },
    7: {
        "filters": (416, 160, 224),
        "kernels": (5, 4, 5),
        "pools": (1, 1, 1),
        "activations": ("relu", "swish", "swish"),
    },
}


class BenchmarkCpuArchitecturesTest(unittest.TestCase):
    """Verify published CNN configurations and model construction."""

    def test_wcnps_configs_match_published_table(self) -> None:
        """Keep all shared budgets aligned with WCNPS 2025 Table I."""
        self.assertEqual(OBSERVATION_BUDGETS, (3, 4, 5, 6, 7))
        self.assertEqual(WCNPS_2025_CONFIGS, EXPECTED_CONFIGS)

    def test_wcnps_models_use_budget_specific_shapes(self) -> None:
        """Build each published CNN with its matching compact input width."""
        for observed_ports in OBSERVATION_BUDGETS:
            with self.subTest(observed_ports=observed_ports):
                model = build_wcnps_2025_cnn(observed_ports)
                self.assertEqual(model.input_shape, (None, observed_ports))
                self.assertEqual(model.output_shape, (None, 100))
                self.assertEqual(
                    model.name,
                    f"wcnps_2025_cnn_{observed_ports}",
                )

    def test_wcnps_model_rejects_unpublished_budget(self) -> None:
        """Reject budgets without a configuration in the published table."""
        with self.assertRaisesRegex(
            ValueError,
            "No WCNPS 2025 CNN configuration",
        ):
            build_wcnps_2025_cnn(8)

    def test_reference_interpreter_disables_default_delegates(self) -> None:
        """Keep hardware-specific delegates out of the architecture screen."""
        factory = getattr(
            benchmark_cpu_architectures,
            "create_reference_tflite_interpreter",
            None,
        )
        self.assertTrue(callable(factory))
        with patch.object(tf.lite, "Interpreter") as interpreter_class:
            interpreter = interpreter_class.return_value

            result = factory(b"tflite-model")

        self.assertIs(result, interpreter)
        interpreter_class.assert_called_once_with(
            model_content=b"tflite-model",
            num_threads=1,
            experimental_op_resolver_type=(
                tf.lite.experimental.OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES
            ),
        )
        interpreter.allocate_tensors.assert_called_once_with()

    def test_parse_proc_status_rss_kib_reads_current_resident_memory(self) -> None:
        """Extract VmRSS without confusing it with virtual or peak memory."""
        status = """Name:\tpython3
VmPeak:\t 999999 kB
VmHWM:\t  456789 kB
VmRSS:\t  123456 kB
"""

        self.assertEqual(parse_proc_status_rss_kib(status), 123456)

    def test_parse_proc_status_rss_kib_requires_linux_rss_field(self) -> None:
        """Reject status text that cannot support an RSS measurement."""
        with self.assertRaisesRegex(ValueError, "VmRSS"):
            parse_proc_status_rss_kib("Name:\tpython3\n")

    def test_summarize_peak_rss_reports_absolute_and_incremental_peak(self) -> None:
        """Report the sampled peak and its increase over the clean baseline."""
        self.assertEqual(
            summarize_peak_rss(120000, [120000, 124000, 123000]),
            {
                "baseline_rss_kib": 120000,
                "peak_rss_kib": 124000,
                "peak_incremental_rss_kib": 4000,
            },
        )


if __name__ == "__main__":
    unittest.main()
