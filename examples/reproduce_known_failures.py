"""Reproduce outstanding calibration defects without private data or service calls.

This is a diagnostic, not a passing preference-quality evaluation. Update it when
the corresponding defects are repaired; assertions detect stale findings.
"""
from types import SimpleNamespace
import warnings

import numpy as np

from residualized_tone_space.core.tone_axis_learning import ToneAxisLearner
from residualized_tone_space.utils.validation import TopicToneValidator
from scripts.utilities.tone_calibration_system_fixed import ToneCalibrationSystem


def main():
    rng = np.random.default_rng(42)
    # Bypass only artifact-loading initialization; execute the actual learner.
    system = ToneCalibrationSystem.__new__(ToneCalibrationSystem)
    system.calculator = SimpleNamespace(get_residual_embedding=lambda value: value)
    system.user_choices = [
        SimpleNamespace(
            chosen_embedding=rng.normal(size=1536),
            rejected_embedding=rng.normal(size=1536),
            confidence=1.0,
        )
        for _ in range(5)
    ]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        system._update_preference_model()
    assert system.is_calibrated() and np.isnan(system.user_preference_vector).all()
    assert np.isnan(system.score_content(np.ones(1536)))
    print("KNOWN FAILURE: calibrated=True with NaN preference vector and content score")
    print("Warnings:", [str(warning.message) for warning in caught])

    try:
        ToneAxisLearner().learn_tone_axis_residual_vectors(
            "synthetic",
            [{"chosen_idx": 0, "rejected_idx": 1}, {"chosen_idx": 2, "rejected_idx": 3}],
            rng.normal(size=(4, 1536)),
        )
    except ValueError as error:
        assert "at least 2 classes" in str(error)
        print("KNOWN FAILURE:", error)
    else:
        raise AssertionError("One-class failure no longer reproduces; update handoff")

    validator = TopicToneValidator()
    first = validator.validate_topic_tone_separation(
        np.eye(2), np.eye(2), {"topic": ["A", "B"], "tone": ["calm", "angry"]}
    )
    second = validator.validate_topic_tone_separation(
        np.zeros((2, 2)), np.full((2, 2), 123.),
        {"topic": ["B", "A"], "tone": ["angry", "calm"]},
    )
    assert first == second and all(value == 0 for value in first.values())
    print("KNOWN LIMITATION: semantic validation returns constant zero metrics")


if __name__ == "__main__":
    main()
