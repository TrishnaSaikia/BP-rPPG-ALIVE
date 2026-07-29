import numpy as np

from alive.metrics import quality_weighted_fusion


def test_quality_weighted_fusion():
    rows = [
        {
            "video_key": "v1",
            "dataset": "d",
            "subject_id": "s",
            "video_id": "v",
            "prediction": np.array([100.0, 70.0]),
            "target": np.array([110.0, 75.0]),
            "quality": 1.0,
        },
        {
            "video_key": "v1",
            "dataset": "d",
            "subject_id": "s",
            "video_id": "v",
            "prediction": np.array([120.0, 80.0]),
            "target": np.array([110.0, 75.0]),
            "quality": 3.0,
        },
    ]
    fused = quality_weighted_fusion(rows)[0]
    assert np.isclose(fused["SBP_pred"], 115.0)
    assert np.isclose(fused["DBP_pred"], 77.5)
