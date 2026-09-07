"""Identity: enrollment + sequential frontal register + IoU/appearance tracking."""

from src.identity.enrollment import EnrollmentGallery
from src.identity.global_registry import (
    GlobalFaceRegistry,
    create_face_lazy_gallery,
    link_session_enrollment_to_global,
)
from src.identity.sequential_enroll import enroll_sequential_from_video
from src.identity.tracker import FaceBodyTracker

__all__ = [
    "EnrollmentGallery",
    "FaceBodyTracker",
    "GlobalFaceRegistry",
    "create_face_lazy_gallery",
    "enroll_sequential_from_video",
    "link_session_enrollment_to_global",
]
