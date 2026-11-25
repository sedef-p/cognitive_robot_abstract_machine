"""
The perception trash can demo on the real HSR: the plans are sent to Giskard, and
RoboKudo looks through the robot's real head camera, whose pose comes from tf in the map
frame.

First check the camera without moving the robot: point the head at the floor by hand,
for instance with the joystick, start the robot's bringup, and run from this folder, with
ROS 2 and the ROS workspace sourced::

    python throw_trash_perception_robot.py --camera-only

Then start Giskard as well, make sure that the regions of
:class:`~throw_trash_perception_rviz_demo.PerceivedFloorToTrashCanDemonstration` and its
:attr:`passage_spot` fit the lab, and run the whole demo::

    python throw_trash_perception_robot.py
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass, field
from datetime import timedelta
from enum import StrEnum

from coraplex.demonstrations import RobotDemonstrationRosSession
from coraplex.plans.executors import RealRobotPlanExecutor
from floor_perception import (
    CameraViewTopic,
    FloorObjectQuery,
    FloorRegion,
    RobotHeadCamera,
)
from semantic_digital_twin.semantic_annotations.semantic_annotations import (
    Milk,
    TrashCan,
)
from semantic_digital_twin.world_description.world_entity import SemanticAnnotation
from throw_trash_perception_rviz_demo import PerceivedFloorToTrashCanDemonstration

# %% looking without moving


@dataclass(kw_only=True)
class FloorObjectSearch:
    """
    One object that is looked for: what it is, how it is looked for and where.
    """

    semantic_annotation: type[SemanticAnnotation]
    """
    What the object is.
    """

    query: FloorObjectQuery
    """
    How the object is looked for.
    """

    region: FloorRegion
    """
    Where the object is known to stand.
    """


@dataclass(kw_only=True)
class FloorObjectLookout:
    """
    Looks with the real head camera for what :attr:`demonstration` searches, where it
    searches it, logs every object found and publishes its points on
    :attr:`~floor_perception.CameraViewTopic.OBJECTS` in the camera's world frame.
    """

    demonstration: PerceivedFloorToTrashCanDemonstration = field(
        default_factory=PerceivedFloorToTrashCanDemonstration
    )
    """
    The demonstration whose queries and regions are used.
    """

    rounds: int = 20
    """
    How often the camera looks for both objects.
    """

    pause: timedelta = timedelta(seconds=2)
    """
    How long to wait between two rounds.
    """

    ros_node_name: str = "floor_object_lookout"
    """
    Name of the ROS node this check registers.
    """

    def run(self) -> None:
        """
        Looks :attr:`rounds` times for the milk and the trash can.
        """
        scene = self.demonstration.build_scene()
        searches = [
            FloorObjectSearch(
                semantic_annotation=Milk,
                query=self.demonstration.milk_query,
                region=self.demonstration.create_milk_region(world=scene),
            ),
            FloorObjectSearch(
                semantic_annotation=TrashCan,
                query=self.demonstration.trash_can_query,
                region=self.demonstration.create_trash_can_region(world=scene),
            ),
        ]
        with RobotDemonstrationRosSession.start(self.ros_node_name) as session:
            camera = RobotHeadCamera(ros_node=session.node)
            publisher = CameraViewTopic.OBJECTS.create_publisher(node=session.node)
            logger = session.node.get_logger()
            for _ in range(self.rounds):
                for search in searches:
                    found = camera.find_floor_objects(
                        query=search.query,
                        region=search.region,
                        nearest_to=search.region.center_xy,
                    )
                    logger.info(
                        f"{search.semantic_annotation.__name__}: {len(found)} found in "
                        f"{search.region.describe()}"
                        + "".join(
                            f"\n  at {detection.describe()}, lowest point "
                            f"{detection.bottom_height:.2f} m above the floor"
                            for detection in found
                        )
                    )
                    publisher.publish(
                        camera.found_points.to_message(
                            frame_id=camera.world_frame,
                            stamp=session.node.get_clock().now().to_msg(),
                        )
                    )
                time.sleep(self.pause.total_seconds())


# %% running on the robot


class CommandLineFlag(StrEnum):
    """
    The flags this script takes.
    """

    CAMERA_ONLY = "--camera-only"
    """
    Only look for the milk and the trash can with the head camera, without moving the
    robot.
    """


def main() -> None:
    """
    Check the head camera with :class:`FloorObjectLookout` if
    :attr:`CommandLineFlag.CAMERA_ONLY` is given, and run the demonstration on the real
    HSR otherwise.
    """
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0].strip())
    parser.add_argument(
        CommandLineFlag.CAMERA_ONLY,
        action="store_true",
        help="only look for the milk and the trash can, without moving the robot",
    )
    if parser.parse_args().camera_only:
        FloorObjectLookout().run()
        return
    PerceivedFloorToTrashCanDemonstration(
        executor_type=RealRobotPlanExecutor, real_time_factor=None
    ).run()


if __name__ == "__main__":
    main()
