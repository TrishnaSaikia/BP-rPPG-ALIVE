import torch

from alive.models import StudentRPPGNetwork, TeacherPPGNetwork


def test_paper_model_shapes():
    teacher = TeacherPPGNetwork(clip_samples=120, dropout=0.0)
    student = StudentRPPGNetwork(k_signals=15, clip_samples=120, dropout=0.0)
    teacher_features, teacher_bp = teacher(torch.randn(2, 1, 120))
    student_features, student_bp = student(torch.randn(2, 15, 120))
    assert teacher_features.shape == (2, 120)
    assert student_features.shape == (2, 120)
    assert teacher_bp.shape == (2, 2)
    assert student_bp.shape == (2, 2)
    assert teacher.backbone.dilation_powers == (0, 1, 2, 3, 4, 5, 6)
