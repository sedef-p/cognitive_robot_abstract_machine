"""
The failures of finding objects on the floor with the HSR's head camera.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta

from coraplex.exceptions import PerceptionException

# %% finding nothing


@dataclass(kw_only=True)
class NoObjectOnFloorFound(PerceptionException):
    """
    Raised when the camera sees no object of the size looked for in a search region,
    from any of the spots the robot looked at.
    """

    region_description: str
    """
    The region the object was looked for in, as a person reads it.
    """

    def error_message(self) -> str:
        return (
            f"No object of the size looked for was found on the floor in "
            f"{self.region_description}."
        )

    def suggest_correction(self) -> str:
        return (
            "put the object inside the search region, widen the region, or loosen the "
            "size limits of the query."
        )


# %% receiving no pictures


@dataclass(kw_only=True)
class CameraImagesMissing(PerceptionException):
    """
    Raised when the robot's head camera sends no picture taken after the robot stopped
    moving.
    """

    topic: str
    """
    The topic the color images were expected on.
    """

    waited: timedelta
    """
    How long a new picture was waited for.
    """

    def error_message(self) -> str:
        return (
            f"No new picture arrived on {self.topic} within "
            f"{self.waited.total_seconds():.1f} s."
        )

    def suggest_correction(self) -> str:
        return (
            "check with 'ros2 topic hz' that the head camera publishes, and that this "
            "computer receives the topic and the tf from the robot."
        )
