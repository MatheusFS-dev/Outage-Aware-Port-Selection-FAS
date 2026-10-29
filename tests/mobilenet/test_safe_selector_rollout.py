"""Non-training contract tests for the MobileNet safe-selector rollout."""

# Standalone companion paths, independent of the original checkout.
import sys as _companion_sys
from pathlib import Path as _CompanionPath
_COMPANION_ROOT = next(p for p in _CompanionPath(__file__).resolve().parents if (p / "pyproject.toml").is_file())
for _companion_path in (_COMPANION_ROOT, _COMPANION_ROOT / "src"):
    if str(_companion_path) not in _companion_sys.path:
        _companion_sys.path.insert(0, str(_companion_path))

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

from studies.chapter06_mobilenet_nas.scripts.analysis.nas.load_study_data import (
    normalize_trial_dataframe,
)
from studies.chapter06_mobilenet_nas.scripts.nas.safe_selector_nas import (
    build_nas_selector_models,
)
from studies.chapter06_mobilenet_nas.scripts.train import safe_selector as selector_module
from studies.chapter06_mobilenet_nas.scripts.train.multi_seed_runner import (
    TRAIN_SEEDS,
    build_script_case_specs,
)
from studies.chapter06_mobilenet_nas.scripts.train.safe_selector import (
    OUTAGE_THRESHOLD,
    SafeDecisionScores,
    SelectionCheckpoint,
    build_non_identity_permutations,
    build_recovery_training_data,
    compute_reload_diagnostics,
)
from utils.qttf import (
    convert_ptq_model_to_strict_int8_tflite,
    make_tflite_interpreter,
)


def test_permutations_are_non_identity_for_every_campaign_seed() -> None:
    """Reject the identity-permutation diagnostic defect for all budgets."""
    for seed in TRAIN_SEEDS:
        for observed_ports in range(3, 8):
            identity = np.arange(observed_ports)
            permutations = build_non_identity_permutations(
                num_observed_ports=observed_ports,
                seed=seed,
            )
            assert len(permutations) == 3
            assert len({tuple(values) for values in permutations}) == 3
            assert all(
                not np.array_equal(values, identity)
                for values in permutations
            )


def test_five_seed_case_totals_cover_both_campaigns() -> None:
    """Enumerate all expected run-all cases without launching child scripts."""
    run_root = Path("runs")
    variants = ("mobv1_alpha02", "mobv2_alpha02", "mobv3_alpha02")
    all_regime_cases = [
        case
        for seed in TRAIN_SEEDS
        for variant in variants
        for case in build_script_case_specs(
            f"train_all_regimes_{variant}.py",
            run_root,
            seed,
        )
    ]
    per_regime_cases = [
        case
        for seed in TRAIN_SEEDS
        for variant in variants
        for case in build_script_case_specs(
            f"train_per_regime_{variant}.py",
            run_root,
            seed,
        )
    ]
    assert len(all_regime_cases) == 75
    assert len(per_regime_cases) == 1_050
    assert len({case.run_dir for case in all_regime_cases}) == 75
    assert len({case.run_dir for case in per_regime_cases}) == 1_050


def test_selection_checkpoint_never_requests_training_stop(tmp_path: Path) -> None:
    """Keep selecting checkpoints without using early-stopping patience."""

    class FakeModel:
        """Minimal callback model that records checkpoint operations."""

        def __init__(self) -> None:
            self.stop_training = False
            self.saved = 0
            self.loaded = 0

        def save_weights(self, path: Path) -> None:
            """Record one simulated checkpoint save."""
            del path
            self.saved += 1

        def load_weights(self, path: Path) -> None:
            """Record one simulated checkpoint restore."""
            del path
            self.loaded += 1

    checkpoint = SelectionCheckpoint(
        decision_model=object(),
        validation_values=np.ones((1, 100), dtype=np.float32),
        num_observed_ports=3,
        batch_size=1,
        checkpoint_path=tmp_path / "best.weights.h5",
        patience=1,
    )
    fake_model = FakeModel()
    checkpoint.set_model(fake_model)
    checkpoint._consider_checkpoint(
        {
            "epoch": 1.0,
            "model_op": 0.4,
            "observed_op": 0.5,
            "distinct_selected_ports": 12.0,
            "permutation_change_rate": 0.5,
        }
    )
    for epoch in range(2, 102):
        checkpoint._consider_checkpoint(
            {
                "epoch": float(epoch),
                "model_op": 0.5,
                "observed_op": 0.5,
                "distinct_selected_ports": 12.0,
                "permutation_change_rate": 0.5,
            }
        )
    checkpoint.on_train_end()
    assert fake_model.stop_training is False
    assert checkpoint.best_epoch == 1
    assert fake_model.saved == 1
    assert fake_model.loaded == 1


def test_reload_diagnostics_compare_matching_predict_paths() -> None:
    """Exclude eager-versus-graph drift from saved-model validation."""

    class FakeModel:
        """Return fixed graph predictions and reject eager execution."""

        def __init__(self, predictions: np.ndarray) -> None:
            self.predictions = predictions

        def __call__(self, inputs: np.ndarray, training: bool) -> np.ndarray:
            """Reject the mismatched execution path that caused run crashes."""
            del inputs
            del training
            raise AssertionError("Reload diagnostics must use predict().")

        def predict(
            self,
            inputs: np.ndarray,
            batch_size: int,
            verbose: int,
        ) -> np.ndarray:
            """Return deterministic graph-mode scores."""
            del inputs
            del batch_size
            del verbose
            return self.predictions

    predictions = np.asarray(
        [[0.1, 0.9], [0.8, 0.2]],
        dtype=np.float32,
    )
    max_abs, decision_change_rate = compute_reload_diagnostics(
        reference_model=FakeModel(predictions),
        reloaded_model=FakeModel(predictions.copy()),
        inputs=np.zeros((2, 1), dtype=np.float32),
        batch_size=2,
    )
    assert max_abs == 0.0
    assert decision_change_rate == 0.0


def test_nas_selector_builds_sparse_protocol_model() -> None:
    """Construct and execute one NAS candidate without fitting it."""
    observed_indices = np.array([0, 49, 99], dtype=np.int64)
    explorer, compact, protocol = build_nas_selector_models(
        observed_indices=observed_indices,
        log_scale=5.0,
        alpha=0.1,
        truncation_block_name="conv_pw_1_relu",
        head_depth=1,
        head_width=16,
        train_seed=0,
    )
    assert explorer.output_shape == (None, 100)
    assert compact.output_shape == (None, 100)
    assert protocol.input_shape == (None, 32, 32, 3)
    assert protocol(np.zeros((1, 32, 32, 3), dtype=np.float32)).shape == (
        1,
        100,
    )


def test_safe_decision_scores_preserve_observed_first_policy() -> None:
    """Select the strongest safe observation or any unobserved fallback."""
    layer = SafeDecisionScores(
        observed_indices=[0, 49, 99],
        threshold=OUTAGE_THRESHOLD,
    )
    observed_values = tf.constant(
        [
            [2.0, 3.0, 0.5],
            [0.1, 0.2, 0.3],
        ],
        dtype=tf.float32,
    )
    exploration_logits = np.full((2, 100), -20.0, dtype=np.float32)
    exploration_logits[:, 10] = -2.0
    exploration_logits[:, 11] = 2.0
    exploration_logits[:, 98] = 4.0

    scores = layer(
        (observed_values, tf.constant(exploration_logits))
    ).numpy()

    assert np.argmax(scores, axis=1).tolist() == [49, 98]


def test_recovery_targets_cover_every_successful_unobserved_port() -> None:
    """Build normalized targets over the complete unobserved set.

    Args:
        None.

    Returns:
        None.

    Raises:
        AssertionError: If recoverability or target support excludes an
            unobserved threshold-satisfying port.
    """
    values = np.zeros((4, 100), dtype=np.float32)
    values[0, 0] = OUTAGE_THRESHOLD + 1.0
    values[0, 12] = OUTAGE_THRESHOLD + 1.0
    values[1, 12] = OUTAGE_THRESHOLD + 1.0
    values[1, 48] = OUTAGE_THRESHOLD + 2.0
    values[2, :] = OUTAGE_THRESHOLD
    values[3, 0] = OUTAGE_THRESHOLD
    values[3, 73] = OUTAGE_THRESHOLD + 1.0

    inputs, targets, observed_indices = build_recovery_training_data(
        y_values=values,
        num_observed_ports=3,
    )

    assert observed_indices.tolist() == [0, 49, 99]
    assert inputs.shape == (2, 3)
    np.testing.assert_allclose(np.sum(targets, axis=1), [1.0, 1.0])
    assert np.flatnonzero(targets[0]).tolist() == [12, 48]
    np.testing.assert_allclose(targets[0, [12, 48]], [0.5, 0.5])
    assert np.flatnonzero(targets[1]).tolist() == [73]
    assert np.all(targets[:, observed_indices] == 0.0)

    for port_index in np.setdiff1d(np.arange(100), observed_indices):
        single_success = np.zeros((1, 100), dtype=np.float32)
        single_success[0, port_index] = OUTAGE_THRESHOLD + 1.0
        _, single_target, _ = build_recovery_training_data(
            y_values=single_success,
            num_observed_ports=3,
        )
        assert single_target.shape == (1, 100)
        assert np.flatnonzero(single_target[0]).tolist() == [port_index]


def test_observed_logit_mask_normalizes_over_unobserved_ports_only() -> None:
    """Exclude observed logits from the learned categorical distribution.

    Args:
        None.

    Returns:
        None.

    Raises:
        AssertionError: If the masking layer is absent, leaves observed
            probability mass, or changes the unobserved normalization total.
    """
    layer_class = getattr(selector_module, "ObservedPortLogitMask", None)
    assert layer_class is not None
    layer = layer_class(observed_indices=[0, 49, 99])
    logits = np.zeros((1, 100), dtype=np.float32)
    logits[0, [0, 49, 99]] = 100.0

    masked_logits = layer(tf.constant(logits))
    probabilities = tf.nn.softmax(masked_logits, axis=1).numpy()[0]

    assert np.all(probabilities[[0, 49, 99]] == 0.0)
    np.testing.assert_allclose(np.sum(np.delete(probabilities, [0, 49, 99])), 1.0)


def test_observed_logit_mask_has_finite_mixed_float16_loss() -> None:
    """Keep masked categorical loss and gradients finite in mixed precision.

    Args:
        None.

    Returns:
        None.

    Raises:
        AssertionError: If the observed-port mask overflows float16 or causes
            non-finite categorical loss or gradients.
    """
    original_policy = tf.keras.mixed_precision.global_policy()
    try:
        tf.keras.mixed_precision.set_global_policy("mixed_float16")
        layer = selector_module.ObservedPortLogitMask(
            observed_indices=[0, 49, 99]
        )
        logits = tf.Variable(np.zeros((2, 100), dtype=np.float32))
        targets = np.zeros((2, 100), dtype=np.float32)
        targets[0, 12] = 1.0
        targets[1, 73] = 1.0
        loss_function = tf.keras.losses.CategoricalCrossentropy(
            from_logits=True
        )

        with tf.GradientTape() as tape:
            masked_logits = layer(logits)
            loss = loss_function(targets, masked_logits)
        gradients = tape.gradient(loss, logits)

        assert bool(tf.reduce_all(tf.math.is_finite(masked_logits)))
        assert bool(tf.math.is_finite(loss))
        assert gradients is not None
        assert bool(tf.reduce_all(tf.math.is_finite(gradients)))
    finally:
        tf.keras.mixed_precision.set_global_policy(original_policy)


def test_every_unobserved_port_remains_eligible_for_selection() -> None:
    """Allow each unobserved physical port to win the learned branch.

    Args:
        None.

    Returns:
        None.

    Raises:
        AssertionError: If any unobserved port is suppressed or learned-score
            ties do not follow deterministic lowest-index argmax behavior.
    """
    observed_indices = [0, 49, 99]
    layer = SafeDecisionScores(
        observed_indices=observed_indices,
        threshold=OUTAGE_THRESHOLD,
    )
    observed_values = tf.zeros((1, 3), dtype=tf.float32)

    for port_index in range(100):
        if port_index in observed_indices:
            continue
        logits = np.full((1, 100), -20.0, dtype=np.float32)
        logits[0, port_index] = 20.0
        scores = layer((observed_values, tf.constant(logits))).numpy()
        assert int(np.argmax(scores[0])) == port_index

    tied_logits = np.full((1, 100), -20.0, dtype=np.float32)
    tied_logits[0, [12, 73]] = 20.0
    tied_scores = layer((observed_values, tf.constant(tied_logits))).numpy()
    assert int(np.argmax(tied_scores[0])) == 12


def test_strict_int8_safe_scores_invoke_for_every_observed_input_code() -> None:
    """Keep valid asymmetric INT8 codes from being mistaken for division by zero."""
    observed_values = tf.keras.layers.Input(
        shape=(3,),
        name="observed_values",
    )
    exploration_logits = tf.keras.layers.Dense(
        100,
        kernel_initializer="zeros",
        bias_initializer="zeros",
        name="fixed_exploration_logits",
    )(observed_values)
    decision_scores = SafeDecisionScores(
        observed_indices=[0, 49, 99],
        threshold=OUTAGE_THRESHOLD,
    )((observed_values, exploration_logits))
    model = tf.keras.Model(observed_values, decision_scores)
    calibration_values = np.linspace(
        0.0,
        36.87769,
        256,
        dtype=np.float32,
    )
    calibration_data = np.repeat(
        calibration_values[:, np.newaxis],
        3,
        axis=1,
    )
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            category=UserWarning,
            message="Statistics for quantized inputs were expected.*",
        )
        warnings.filterwarnings(
            "ignore",
            category=UserWarning,
            message=".*tf.lite.Interpreter is deprecated.*",
        )
        tflite_model = convert_ptq_model_to_strict_int8_tflite(
            model,
            calibration_data,
        )
        interpreter = make_tflite_interpreter(tflite_model)
    interpreter.allocate_tensors()
    input_detail = interpreter.get_input_details()[0]

    for input_code in range(-128, 128):
        interpreter.set_tensor(
            input_detail["index"],
            np.full(
                input_detail["shape"],
                input_code,
                dtype=input_detail["dtype"],
            ),
        )
        interpreter.invoke()


def test_fp32_and_int8_guarded_decisions_match() -> None:
    """Preserve guarded selected-port decisions through strict INT8 PTQ.

    Args:
        None.

    Returns:
        None.

    Raises:
        AssertionError: If strict INT8 conversion changes the selected port
            for the fixed observed-success and learned-exploration cases.
    """
    observed_values = tf.keras.layers.Input(shape=(3,), name="observed_values")
    bias_values = np.full(100, -4.0, dtype=np.float32)
    bias_values[12] = 2.0
    bias_values[73] = 4.0
    exploration_logits = tf.keras.layers.Dense(
        100,
        kernel_initializer="zeros",
        bias_initializer=tf.keras.initializers.Constant(bias_values),
        name="fixed_exploration_logits",
    )(observed_values)
    decision_scores = SafeDecisionScores(
        observed_indices=[0, 49, 99],
        threshold=OUTAGE_THRESHOLD,
    )((observed_values, exploration_logits))
    model = tf.keras.Model(observed_values, decision_scores)
    calibration_data = np.linspace(
        0.0,
        5.0,
        256 * 3,
        dtype=np.float32,
    ).reshape(256, 3)
    evaluation_inputs = np.asarray(
        [
            [0.1, 0.2, 0.3],
            [2.0, 3.0, 0.5],
            [4.0, 2.0, 3.0],
        ],
        dtype=np.float32,
    )
    expected_ports = np.argmax(model(evaluation_inputs).numpy(), axis=1)

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            category=UserWarning,
            message="Statistics for quantized inputs were expected.*",
        )
        warnings.filterwarnings(
            "ignore",
            category=UserWarning,
            message=".*tf.lite.Interpreter is deprecated.*",
        )
        tflite_model = convert_ptq_model_to_strict_int8_tflite(
            model,
            calibration_data,
        )
        interpreter = make_tflite_interpreter(tflite_model)
    interpreter.resize_tensor_input(
        interpreter.get_input_details()[0]["index"],
        [len(evaluation_inputs), 3],
        strict=True,
    )
    interpreter.allocate_tensors()
    input_detail = interpreter.get_input_details()[0]
    output_detail = interpreter.get_output_details()[0]
    input_scale, input_zero_point = input_detail["quantization"]
    quantized_inputs = np.clip(
        np.rint(evaluation_inputs / input_scale + input_zero_point),
        -128,
        127,
    ).astype(input_detail["dtype"])
    interpreter.set_tensor(input_detail["index"], quantized_inputs)
    interpreter.invoke()
    actual_ports = np.argmax(
        interpreter.get_tensor(output_detail["index"]),
        axis=1,
    )

    assert expected_ports.tolist() == [73, 49, 0]
    assert actual_ports.tolist() == expected_ports.tolist()


def test_nas_reader_distinguishes_new_op_and_legacy_mse() -> None:
    """Normalize both schemas without changing their declared semantics."""
    ranked = pd.DataFrame({"rank": [1], "trial_id": [0]})
    common = {
        "trial_id": [0],
        "ram_budget_violated": [False],
    }
    new = normalize_trial_dataframe(
        pd.DataFrame(
            {
                **common,
                "best_not_qt_val_model_op": [0.3],
                "best_qt_val_model_op": [0.31],
            }
        ),
        ranked,
        objective_semantics="model_only_op",
    )
    legacy = normalize_trial_dataframe(
        pd.DataFrame(
            {
                **common,
                "best_not_qt_val_mse": [0.001],
                "best_qt_val_mse": [0.002],
            }
        ),
        ranked,
        objective_semantics="validation_mse",
    )
    assert new.loc[0, "best_qt_val_metric"] == 0.31
    assert new.loc[0, "objective_semantics"] == "model_only_op"
    assert legacy.loc[0, "best_qt_val_metric"] == 0.002
    assert legacy.loc[0, "objective_semantics"] == "validation_mse"
