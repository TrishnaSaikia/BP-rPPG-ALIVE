"""Official training and evaluation code for the BP-rPPG ALIVE baseline."""

from .models import TeacherPPGNetwork, StudentRPPGNetwork

__all__ = ["TeacherPPGNetwork", "StudentRPPGNetwork"]
