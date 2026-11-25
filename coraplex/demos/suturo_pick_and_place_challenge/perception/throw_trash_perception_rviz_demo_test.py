"""
Tests of the HSR perception trash can demo: where the robot looks, and full simulated
runs from several spots of the milk and the trash can, run as fast as the computer
allows. Run them from this folder, with ROS 2 and the ROS workspace sourced::

    python -m pytest throw_trash_perception_rviz_demo_test.py
"""

from __future__ import annotations

import numpy as np
import pytest
import throw_trash_perception_rviz_demo as demo
from floor_perception import FloorRegion
from krrood_spatial_math.spatial_types import HomogeneousTransformationMatrix, Pose2D
from semantic_digital_twin.datastructures.definitions import (
    JointStateType,
    StaticJointState,
    TorsoState,
)
from semantic_digital_twin.robots.robot_parts import AbstractRobotPart
from semantic_digital_twin.semantic_annotations.semantic_annotations import (
    Milk,
    Sofa,
    TrashCan,
)
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.geometry import PlanarBoundingBox

# %% fixtures


@pytest.fixture(scope="module")
def scene() -> World:
    """
    The kitchen the demo acts in, without the robot.
    """
    return demo.PerceivedFloorToTrashCanDemonstration().build_scene()


# %% where the trash can is looked for


def test_trash_can_region_lies_beside_the_sofa_end_on_its_left(scene: World) -> None:
    demonstration = demo.PerceivedFloorToTrashCanDemonstration()
    region = demonstration.create_trash_can_region(world=scene)
    [sofa] = scene.get_semantic_annotations_by_type(Sofa)
    sofa_box = sofa.root.collision.as_bounding_box_collection_in_frame(
        scene.root
    ).bounding_box()

    assert region.contains(
        xy=np.array(
            [
                float(demonstration.trash_can_spot.x),
                float(demonstration.trash_can_spot.y),
            ]
        )
    )
    assert not region.contains(
        xy=np.array(
            [
                (sofa_box.min_x + sofa_box.max_x) / 2,
                (sofa_box.min_y + sofa_box.max_y) / 2,
            ]
        )
    )
    assert region.world_bounds.upper[0] < sofa_box.min_x
    sofa_frame_bounds = region.area.to_array_bounds()
    assert sofa_frame_bounds.upper[1] - sofa_frame_bounds.lower[1] == pytest.approx(
        demonstration.trash_can_reach
    )


# %% where the robot looks


def create_region_in_world_frame(
    *, world: World, lower: np.ndarray, upper: np.ndarray
) -> FloorRegion:
    """
    :return: The region between the corners `lower` and `upper`, in the root frame of
        `world`.
    """
    return FloorRegion(
        area=PlanarBoundingBox.from_array_bounds(
            lower, upper, HomogeneousTransformationMatrix(reference_frame=world.root)
        )
    )


def test_look_targets_stay_out_of_reach_of_the_robot_body(scene: World) -> None:
    region = create_region_in_world_frame(
        world=scene, lower=np.array([-1.0, -0.5]), upper=np.array([1.0, 0.5])
    )
    viewer = Pose2D.from_xy_yaw(x=0.1, y=0.0, yaw=np.pi / 2)
    minimum_distance = 0.9

    targets = region.compute_look_targets(
        viewer=viewer, minimum_distance=minimum_distance, spacing=0.5
    )

    viewer_xy = np.array([0.1, 0.0])
    assert len(targets) > 0
    assert all(
        np.linalg.norm(target - viewer_xy) >= minimum_distance - 1e-9
        for target in targets
    )


def test_look_targets_begin_where_the_robot_faces(scene: World) -> None:
    region = create_region_in_world_frame(
        world=scene, lower=np.array([1.5, -1.5]), upper=np.array([2.5, 1.5])
    )
    viewer = Pose2D.from_xy_yaw(x=0.0, y=1.0, yaw=0.0)

    [first, *others] = region.compute_look_targets(
        viewer=viewer, minimum_distance=0.5, spacing=1.0
    )

    def turn(target: np.ndarray) -> float:
        return abs(float(np.arctan2(target[1] - 1.0, target[0])))

    assert all(turn(first) <= turn(other) for other in others)


# %% full runs


def check_joints_reached(
    *, robot_part: AbstractRobotPart, joint_state_type: JointStateType
) -> None:
    """
    Check that every joint of `robot_part` stands where its joint state of
    `joint_state_type` puts it.
    """
    joint_state = robot_part.get_joint_state_by_type(joint_state_type)
    positions = [float(connection.position) for connection in joint_state.connections]
    assert np.allclose(positions, joint_state.target_values, atol=0.05)


@pytest.mark.parametrize(
    ("milk_start", "trash_can_spot"),
    [
        (Pose2D.from_xy_yaw(x=1.7, y=4.6, yaw=0.3), Pose2D.from_xy_yaw(x=2.45, y=1.6)),
        (Pose2D.from_xy_yaw(x=1.5, y=3.95, yaw=1.0), Pose2D.from_xy_yaw(x=2.1, y=1.1)),
        (Pose2D.from_xy_yaw(x=0.4, y=4.3, yaw=2.5), Pose2D.from_xy_yaw(x=2.45, y=1.6)),
    ],
)
def test_milk_is_searched_found_and_thrown_away(
    milk_start: Pose2D, trash_can_spot: Pose2D
) -> None:
    size_tolerance = 0.015
    demonstration = demo.PerceivedFloorToTrashCanDemonstration(
        milk_start=milk_start,
        trash_can_spot=trash_can_spot,
        real_time_factor=None,
        camera_view_period=None,
    )

    world = demonstration.run()

    demonstration.camera.synchronize_with_robot()
    ground_truth = demonstration.camera.ground_truth
    [milk] = world.get_semantic_annotations_by_type(Milk)
    [true_milk] = ground_truth.get_semantic_annotations_by_type(Milk)
    [trash_can] = world.get_semantic_annotations_by_type(TrashCan)
    [true_trash_can] = ground_truth.get_semantic_annotations_by_type(TrashCan)
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
            ground_truth.root
        ).bounding_box()
    )
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

    expected_position = [
        (trash_can_box.min_x + trash_can_box.max_x) / 2,
        (trash_can_box.min_y + trash_can_box.max_y) / 2,
        trash_can_box.max_z + demonstration.drop_clearance - milk_box.min_z,
    ]
    position = world.compute_forward_kinematics_np(world.root, milk.root)[:3, 3]
    assert milk.root.parent_connection.parent is world.root
    assert np.allclose(position, expected_position, atol=0.01)
    true_milk_position = ground_truth.compute_forward_kinematics_np(
        ground_truth.root, true_milk.root
    )[:3, 3]
    assert true_trash_can_box.min_x < true_milk_position[0] < true_trash_can_box.max_x
    assert true_trash_can_box.min_y < true_milk_position[1] < true_trash_can_box.max_y

    [robot] = world.get_semantic_annotations_by_type(demonstration.used_robot)
    for arm in robot.all_arms:
        check_joints_reached(robot_part=arm, joint_state_type=StaticJointState.PARK)
    check_joints_reached(robot_part=robot.get_torso(), joint_state_type=TorsoState.LOW)
