#!/usr/bin/env python
"""
Runs the HSR perception trash can demo as a regression test, as fast as the computer
allows, from three start spots of the milk. Fails when the size RoboKudo estimates is
off, or when the milk does not end up inside the trash can. Exits non-zero on failure.

Run it from this folder, with ROS 2 and the ROS workspace sourced::

    python throw_trash_perception_rviz_demo_test.py
"""

from __future__ import annotations

import numpy as np
import throw_trash_perception_rviz_demo as demo
from semantic_digital_twin.datastructures.definitions import (
    JointStateType,
    StaticJointState,
    TorsoState,
)
from semantic_digital_twin.robots.robot_parts import AbstractRobotPart
from semantic_digital_twin.semantic_annotations.semantic_annotations import (
    Milk,
    TrashCan,
)

# %% checking what was found and where it ends up


def check_milk_is_found_and_thrown_away(
    milk_start: demo.FloorSpot, size_tolerance: float = 0.015
) -> None:
    """
    Run the demo with the milk starting at `milk_start`, and check that the milk and
    the trash can the robot added match the real ones in size and place, that the milk
    ends upright on the trash can's bottom, that the points last shown as found are the
    trash can's, and that the robot ends parked.

    :param milk_start: Where the milk stands on the floor before the plan begins.
    :param size_tolerance: How far each estimated extent and position may be off, in
        meters.
    """
    demonstration = demo.PerceivedFloorToTrashCanDemonstration(
        milk_start=milk_start,
        search_spot=demo.FloorSpot(x=milk_start.x, y=milk_start.y),
        real_time_factor=None,
    )

    world = demonstration.run()

    [milk] = world.get_semantic_annotations_by_type(Milk)
    camera = demonstration.camera
    [true_milk] = camera.ground_truth.get_semantic_annotations_by_type(Milk)
    [trash_can] = world.get_semantic_annotations_by_type(TrashCan)
    [true_trash_can] = camera.ground_truth.get_semantic_annotations_by_type(TrashCan)
    milk_box = milk.root.collision.as_bounding_box_collection_in_frame(
        milk.root
    ).bounding_box()
    true_milk_box = true_milk.root.collision.as_bounding_box_collection_in_frame(
        true_milk.root
    ).bounding_box()
    trash_can_box = trash_can.root.collision.as_bounding_box_collection_in_frame(
        world.root
    ).bounding_box()
    true_trash_can_box = (
        true_trash_can.root.collision.as_bounding_box_collection_in_frame(
            camera.ground_truth.root
        ).bounding_box()
    )
    true_milk_position = camera.ground_truth.compute_forward_kinematics_np(
        camera.ground_truth.root, true_milk.root
    )[:3, 3]
    expected_position = [
        (trash_can_box.min_x + trash_can_box.max_x) / 2,
        (trash_can_box.min_y + trash_can_box.max_y) / 2,
        trash_can_box.min_z + demonstration.trash_can_wall_thickness - milk_box.min_z,
    ]
    position = world.compute_forward_kinematics_np(world.root, milk.root)[:3, 3]
    assert np.allclose(
        sorted(milk_box.dimensions[:2]) + [milk_box.dimensions[2]],
        sorted(true_milk_box.dimensions[:2]) + [true_milk_box.dimensions[2]],
        atol=size_tolerance,
    )
    assert np.allclose(
        trash_can_box.center.to_np()[:3],
        true_trash_can_box.center.to_np()[:3],
        atol=size_tolerance,
    )
    assert np.allclose(
        sorted(trash_can_box.dimensions[:2]) + [trash_can_box.dimensions[2]],
        sorted(true_trash_can_box.dimensions[:2]) + [true_trash_can_box.dimensions[2]],
        atol=size_tolerance,
    )
    assert milk.root.parent_connection.parent is trash_can.root
    assert np.allclose(position, expected_position, atol=0.01)
    assert true_trash_can_box.min_x < true_milk_position[0] < true_trash_can_box.max_x
    assert true_trash_can_box.min_y < true_milk_position[1] < true_trash_can_box.max_y
    chosen_points = camera.found_points.positions[
        np.all(camera.found_points.colors == camera.chosen_object_color, axis=1)
    ]
    assert len(chosen_points) > 0
    assert np.all(
        (chosen_points[:, 0] > true_trash_can_box.min_x - size_tolerance)
        & (chosen_points[:, 0] < true_trash_can_box.max_x + size_tolerance)
        & (chosen_points[:, 1] > true_trash_can_box.min_y - size_tolerance)
        & (chosen_points[:, 1] < true_trash_can_box.max_y + size_tolerance)
    )
    [robot] = world.get_semantic_annotations_by_type(demonstration.used_robot)
    for arm in robot.all_arms:
        check_joints_reached(arm, StaticJointState.PARK)
    check_joints_reached(robot.get_torso(), TorsoState.LOW)


def check_joints_reached(
    robot_part: AbstractRobotPart,
    joint_state_type: JointStateType,
    tolerance: float = 0.05,
) -> None:
    """
    Check that every joint of `robot_part` stands where its joint state of
    `joint_state_type` puts it.

    :param robot_part: The arm or torso whose joints are checked.
    :param joint_state_type: The joint state the joints are to stand in.
    :param tolerance: How far each joint may be off, in rad or meters.
    """
    joint_state = robot_part.get_joint_state_by_type(joint_state_type)
    positions = [float(connection.position) for connection in joint_state.connections]
    assert np.allclose(positions, joint_state.target_values, atol=tolerance)


# %% running the checks


def main() -> None:
    """
    Check three start spots of the milk on the floor, one after the other.
    """
    for milk_start in [
        demo.FloorSpot(x=1.7, y=4.6, yaw=0.3),
        demo.FloorSpot(x=1.2, y=4.0, yaw=1.0),
        demo.FloorSpot(x=0.4, y=4.3, yaw=2.5),
    ]:
        check_milk_is_found_and_thrown_away(milk_start)


if __name__ == "__main__":
    main()
