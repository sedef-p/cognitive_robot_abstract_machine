"""
The HSR finds a milk carton on the floor and the trash can with RoboKudo, through its
simulated head camera, and throws the milk away; only what RoboKudo finds is added to the
robot's world. Run it from this folder, with ROS 2 and the ROS workspace sourced::

    python throw_trash_perception_rviz_demo.py

The camera's points are published on :class:`CameraViewTopic`; RViz shows them with::

    rviz2 -d throw_trash_perception.rviz
"""

from __future__ import annotations

import array
import sys
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import StrEnum
from functools import partial
from pathlib import Path

import numpy as np
import open3d
import rclpy
from builtin_interfaces.msg import Time
from coraplex.exceptions import PerceptionException
from coraplex.plans.context_extensions import RobotAccess
from coraplex.plans.executors import KinematicallySimulatedPlanExecutor
from coraplex.robot_plans.actions.composite.carrying import CarryAction
from coraplex.robot_plans.actions.composite.facing import FaceAndLookAtAction
from coraplex.robot_plans.actions.composite.transporting import MoveAndPickUpAction
from coraplex.robot_plans.actions.core.navigation import (
    FaceAtAction,
    LookAtAction,
    NavigateAction,
)
from coraplex.robot_plans.actions.core.pick_up import PickUpAction
from coraplex.robot_plans.actions.core.placing import PlaceAction
from coraplex.robot_plans.actions.core.robot_body import (
    MoveManipulatorAction,
    MoveTorsoAction,
    ParkArmsAction,
)
from cramph.composites import Parallel, Sequence
from cramph.context import StatechartContext
from cramph.node import StatechartNode
from cramph.statechart import Statechart
from krrood.entity_query_language.factories import a
from krrood_spatial_math.spatial_types import HomogeneousTransformationMatrix, Pose
from py_trees.common import Status
from rclpy.node import Node
from rclpy.publisher import Publisher
from rclpy.qos import DurabilityPolicy, QoSProfile
from robokudo import world as robokudo_world
from robokudo.annotators.cluster_pose_bb import ClusterPoseBBAnnotator
from robokudo.annotators.collection_reader import CollectionReaderAnnotator
from robokudo.annotators.core import BaseAnnotator
from robokudo.annotators.image_preprocessor import ImagePreprocessorAnnotator
from robokudo.annotators.plane import PlaneAnnotator
from robokudo.annotators.pointcloud_cluster_extractor import PointCloudClusterExtractor
from robokudo.annotators.pointcloud_crop import PointcloudCropAnnotator
from robokudo.behaviours.clear_annotator_outputs import ClearAnnotatorOutputs
from robokudo.cas import CAS, CASViews
from robokudo.descriptors.camera_configs.config_ray_tracing import (
    RayTracingCameraConfiguration,
    RuntimeRobotWorldSource,
)
from robokudo.descriptors.factories.cr_descriptor_factory import (
    CollectionReaderDescriptorFactory,
)
from robokudo.pipeline import Pipeline
from robokudo.types.annotation import BoundingBox3DAnnotation
from robokudo.types.scene import ObjectHypothesis
from robokudo.utils.annotator_helper import (
    get_camera_to_world_transform_matrix,
    transform_cloud_from_camera_to_world,
)
from robokudo.utils.transform import get_transform_matrix_from_q
from robokudo.utils.tree_execution import run_tree_once
from semantic_digital_twin.callbacks.callback import StateChangeCallback
from semantic_digital_twin.datastructures.camera_model import (
    CameraRange,
    CameraResolution,
    FieldOfView,
    PinholeCameraModel,
)
from semantic_digital_twin.datastructures.camera_rendering import (
    ColorRenderingMode,
    TexturedColorRenderingMode,
)
from semantic_digital_twin.datastructures.definitions import TorsoState
from semantic_digital_twin.grasping.grasp_candidates import GraspCandidate
from semantic_digital_twin.robots.robot_parts import AbstractRobot, Arm
from semantic_digital_twin.semantic_annotations.semantic_annotations import (
    Milk,
    TrashCan,
)
from semantic_digital_twin.specifications.connections import (
    Connection6DoFSpecification,
)
from semantic_digital_twin.specifications.kinematic_structure_entities import (
    BodySpecification,
)
from semantic_digital_twin.specifications.semantic_annotations import (
    SemanticAnnotationWithRootSpecification,
)
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.geometry import Scale
from semantic_digital_twin.world_description.world_entity import (
    Body,
    SemanticAnnotation,
)
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Header

# the demo this one builds on lies in the folder above, which is no package
sys.path.append(str(Path(__file__).resolve().parents[1]))
from throw_trash_rviz_demo import (
    FloorSpot,
    FloorToTrashCanDemonstration,
    SceneBody,
    SettleOnContainerBottom,
)

# %% what RoboKudo finds


class PerceivedSceneBody(StrEnum):
    """
    The names of the bodies added to the robot's world once RoboKudo found them, besides
    those of :class:`SceneBody`.
    """

    TRASH_CAN = "trash_can"


@dataclass
class FloorObjectDetection:
    """
    An object RoboKudo found standing on the floor: where it stands and how big it is.
    """

    spot: FloorSpot
    """
    Where on the floor the object stands and which way it is turned.
    """

    size: Scale
    """
    The extents of the object along its own axes, in meters, from the floor up.
    """

    points: np.ndarray
    """
    The points the camera saw of the object, one row of x, y and z per point, in the
    world frame.
    """

    @property
    def largest_extent(self) -> float:
        """
        The largest of the object's extents, in meters.
        """
        return max(self.size.x, self.size.y, self.size.z)

    def distance_to(self, spot: FloorSpot) -> float:
        """
        :return: How far the object stands from `spot`, along the floor, in meters.
        """
        return float(np.hypot(self.spot.x - spot.x, self.spot.y - spot.y))

    @property
    def bottom_height(self) -> float:
        """
        How high above the floor the lowest point seen of the object is, in meters.
        """
        return float(self.points[:, 2].min())


@dataclass
class FloorObjectQuery:
    """
    What is looked for on the floor around a search spot: how far from it, and how big.
    """

    maximum_extent: float
    """
    The largest extent a found object may have, in meters.
    """

    minimum_extent: float = 0.0
    """
    How much a found object's largest extent must exceed, in meters.
    """

    search_radius: float = 0.8
    """
    How far from the search spot, along the floor, an object may stand, in meters.
    """

    maximum_floor_gap: float = 0.05
    """
    How high above the floor the lowest point seen of an object may be, in meters, for
    it to count as standing on the floor rather than being held.
    """

    crop_height: float = 0.4
    """
    How high above the floor the point cloud is kept, in meters.
    """

    floor_depth: float = 0.05
    """
    How far below the floor the point cloud is kept, in meters, so the floor itself is
    found as the plane the objects stand on.
    """

    def create_crop_annotator(self, search_spot: FloorSpot) -> PointcloudCropAnnotator:
        """
        :param search_spot: Where on the floor the objects are looked for.
        :return: An annotator keeping the points just above the floor around
            `search_spot`.
        """
        crop = PointcloudCropAnnotator.Descriptor()
        crop.parameters.relative_to_world = True
        crop.parameters.min_x = search_spot.x - self.search_radius
        crop.parameters.max_x = search_spot.x + self.search_radius
        crop.parameters.min_y = search_spot.y - self.search_radius
        crop.parameters.max_y = search_spot.y + self.search_radius
        crop.parameters.min_z = -self.floor_depth
        crop.parameters.max_z = self.crop_height
        return PointcloudCropAnnotator(descriptor=crop)

    def is_matched_by(
        self, detection: FloorObjectDetection, search_spot: FloorSpot
    ) -> bool:
        """
        :return: Whether `detection` is as big as looked for and stands on the floor
            within :attr:`search_radius` of `search_spot`.
        """
        return (
            self.minimum_extent < detection.largest_extent <= self.maximum_extent
            and detection.bottom_height <= self.maximum_floor_gap
            and detection.distance_to(search_spot) <= self.search_radius
        )


# %% what the camera saw


class CameraViewTopic(StrEnum):
    """
    The topics what the head camera sees is published on.
    """

    SCENE = "throw_trash_perception/scene_points"
    """
    Every point the camera sees, in the colors it sees them.
    """

    OBJECTS = "throw_trash_perception/object_points"
    """
    The points of the objects RoboKudo found the last time it looked for some, the one
    the robot acts on in another color than the others.
    """


@dataclass
class ColoredPoints:
    """
    Points in the world frame, each with a color.
    """

    positions: np.ndarray
    """
    One row of x, y and z per point, in meters.
    """

    colors: np.ndarray
    """
    One row of red, green and blue per point, each in [0, 1].
    """

    @classmethod
    def from_cloud(cls, cloud: open3d.geometry.PointCloud) -> ColoredPoints:
        """
        :param cloud: Points in the world frame.
        :return: The points of `cloud`, gray where it has no colors.
        """
        positions = np.asarray(cloud.points).reshape(-1, 3)
        if cloud.has_colors():
            return cls(positions=positions, colors=np.asarray(cloud.colors))
        return cls(positions=positions, colors=np.full_like(positions, 0.5))

    @classmethod
    def from_uniformly_colored(
        cls, positions: list[np.ndarray], color: tuple[float, float, float]
    ) -> ColoredPoints:
        """
        :param positions: Groups of points, one row of x, y and z per point.
        :param color: The red, green and blue in [0, 1] every point gets.
        :return: The points of all groups together, each in `color`.
        """
        joined = np.concatenate([np.empty((0, 3)), *positions])
        return cls(positions=joined, colors=np.tile(color, (len(joined), 1)))

    def without_points_near(
        self, positions: np.ndarray, distance: float
    ) -> ColoredPoints:
        """
        :param positions: One row of x, y and z per point, in the world frame.
        :param distance: How near to a point of `positions` a point may be to be left
            out, in meters.
        :return: These points, without those within `distance` of `positions`.
        """
        if len(self.positions) == 0 or len(positions) == 0:
            return self
        own = open3d.geometry.PointCloud(open3d.utility.Vector3dVector(self.positions))
        other = open3d.geometry.PointCloud(open3d.utility.Vector3dVector(positions))
        kept = np.asarray(own.compute_point_cloud_distance(other)) > distance
        return ColoredPoints(positions=self.positions[kept], colors=self.colors[kept])

    def concatenate(self, other: ColoredPoints) -> ColoredPoints:
        """
        :return: The points of this and of `other` together, each keeping its color.
        """
        return ColoredPoints(
            positions=np.concatenate([self.positions, other.positions]),
            colors=np.concatenate([self.colors, other.colors]),
        )

    def to_message(self, frame_id: str, stamp: Time) -> PointCloud2:
        """
        :param frame_id: The tf frame of the world frame.
        :param stamp: When the points were seen.
        :return: The points as a message RViz shows.
        """
        point_type = np.dtype(
            [("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("rgb", "<u4")]
        )
        points = np.empty(len(self.positions), dtype=point_type)
        points["x"], points["y"], points["z"] = self.positions.T
        red, green, blue = (
            np.round(np.clip(self.colors, 0.0, 1.0) * 255).astype(np.uint32).T
        )
        points["rgb"] = (red << 16) | (green << 8) | blue
        return PointCloud2(
            header=Header(frame_id=frame_id, stamp=stamp),
            height=1,
            width=len(points),
            fields=[
                PointField(
                    name=name,
                    offset=point_type.fields[name][1],
                    datatype=PointField.FLOAT32,
                    count=1,
                )
                for name in point_type.names
            ],
            is_bigendian=False,
            point_step=point_type.itemsize,
            row_step=point_type.itemsize * len(points),
            data=array.array("B", points.tobytes()),
            is_dense=True,
        )


# %% the simulated head camera


@dataclass
class FollowedBody:
    """
    A body of the world as it really is that moves along with the body the robot
    believes it to be, the way an object moves along with the hand that holds it.
    """

    true_body: Body
    """
    The body in the world as it really is.
    """

    believed_body: Body
    """
    The body in the robot's world.
    """

    believed_T_true: np.ndarray
    """
    Where :attr:`true_body` is relative to :attr:`believed_body`.
    """

    @classmethod
    def from_bodies(cls, true_body: Body, believed_body: Body) -> FollowedBody:
        """
        :return: `true_body` following `believed_body`, keeping where it is relative to
            it now.
        """
        true_world = true_body._world
        believed_world = believed_body._world
        world_T_true = true_world.compute_forward_kinematics_np(
            true_world.root, true_body
        )
        world_T_believed = believed_world.compute_forward_kinematics_np(
            believed_world.root, believed_body
        )
        return cls(
            true_body=true_body,
            believed_body=believed_body,
            believed_T_true=np.linalg.inv(world_T_believed) @ world_T_true,
        )

    def follow(self) -> None:
        """
        Moves :attr:`true_body` to where it is relative to :attr:`believed_body`.
        """
        believed_world = self.believed_body._world
        world_T_believed = believed_world.compute_forward_kinematics_np(
            believed_world.root, self.believed_body
        )
        self.true_body.parent_connection.origin = HomogeneousTransformationMatrix(
            data=world_T_believed @ self.believed_T_true,
            reference_frame=self.true_body._world.root,
        )


@dataclass
class SimulatedHeadCamera:
    """
    The robot's head camera, rendered in :attr:`ground_truth`, the world as it really
    is, with the robot standing as in :attr:`believed_world`. RoboKudo looks through it.
    """

    ground_truth: World
    """
    The world as it really is, holding objects the robot does not know about yet.
    """

    believed_world: World
    """
    The robot's world, whose robot moves as the plan moves it.
    """

    rgb_mode: ColorRenderingMode = field(default_factory=TexturedColorRenderingMode)
    """
    How the colors of the pictures are rendered.
    """

    image_resolution: CameraResolution = field(
        default_factory=lambda: CameraResolution(width=640, height=480)
    )
    """
    The size of the pictures, that of the HSR's head RGB-D camera.
    """

    camera_minimum_distance: float = 0.3
    """
    The nearest distance the head camera sees, in meters, so the rendering does not hit
    the robot's own head.
    """

    maximum_ticks: int = 100
    """
    How often a pipeline is ticked at most before it counts as having seen nothing.
    """

    tick_rate: int = 20
    """
    How often per second a pipeline is ticked.
    """

    ros_node_name: str = "simulated_head_camera"
    """
    Name of the ROS node a pipeline runs on.
    """

    chosen_object_color: tuple[float, float, float] = (0.1, 0.9, 0.2)
    """
    The red, green and blue in [0, 1] the points of the nearest found object, the one
    the robot acts on, are shown in.
    """

    other_object_color: tuple[float, float, float] = (1.0, 0.5, 0.0)
    """
    The red, green and blue in [0, 1] the points of every other found object are shown
    in.
    """

    found_points: ColoredPoints = field(
        init=False,
        default_factory=lambda: ColoredPoints.from_uniformly_colored([], (0, 0, 0)),
    )
    """
    The points of the objects found the last time objects were looked for.
    """

    followed_bodies: list[FollowedBody] = field(init=False, default_factory=list)
    """
    The bodies of :attr:`ground_truth` that move along with bodies of
    :attr:`believed_world`.
    """

    _lock: threading.Lock = field(init=False, default_factory=threading.Lock)
    """
    Held while :attr:`ground_truth` is changed or rendered.
    """

    def __post_init__(self) -> None:
        """
        Gives the head camera of the robot in :attr:`ground_truth` the range and model
        it is rendered with, and lets RoboKudo look into :attr:`ground_truth`.
        """
        [true_robot] = self.ground_truth.get_semantic_annotations_by_type(AbstractRobot)
        camera = true_robot.get_default_camera()
        camera.camera_range = CameraRange(minimum_distance=self.camera_minimum_distance)
        camera.camera_model = self._create_camera_model(
            camera.camera_model.field_of_view
        )
        robokudo_world.set_world(self.ground_truth)

    def _create_camera_model(self, field_of_view: FieldOfView) -> PinholeCameraModel:
        """
        :param field_of_view: The field of view of the robot's head camera.
        :return: A centered camera of :attr:`image_resolution` with the horizontal
            extent of `field_of_view` and square pixels, as textured rendering needs.
        """
        focal_length = self.image_resolution.width / (
            2.0 * np.tan(field_of_view.horizontal_angle / 2.0)
        )
        return PinholeCameraModel.from_field_of_view(
            self.image_resolution,
            FieldOfView(
                horizontal_angle=field_of_view.horizontal_angle,
                vertical_angle=2.0
                * np.arctan(self.image_resolution.height / (2.0 * focal_length)),
            ),
        )

    def follow(self, true_body: Body, believed_body: Body) -> None:
        """
        Lets `true_body` move along with `believed_body` from now on.
        """
        self.followed_bodies.append(FollowedBody.from_bodies(true_body, believed_body))

    def take_on_believed_state(self, wait: bool) -> bool:
        """
        Moves every degree of freedom in :attr:`ground_truth` to where it is in
        :attr:`believed_world`, and every followed body along.

        :param wait: Whether to wait while :attr:`ground_truth` is rendered, instead of
            doing nothing.
        :return: Whether :attr:`ground_truth` was changed.
        """
        if not self._lock.acquire(blocking=wait):
            return False
        try:
            self._copy_believed_state()
        finally:
            self._lock.release()
        return True

    def _copy_believed_state(self) -> None:
        """
        Does what :meth:`take_on_believed_state` does, while :attr:`_lock` is held.
        """
        followed_names = {
            degree_of_freedom.name
            for followed in self.followed_bodies
            for degree_of_freedom in followed.true_body.parent_connection.dofs
        }
        true_degrees_of_freedom = {
            degree_of_freedom.name: degree_of_freedom
            for degree_of_freedom in self.ground_truth.degrees_of_freedom
            if degree_of_freedom.name not in followed_names
        }
        for degree_of_freedom in self.believed_world.degrees_of_freedom:
            if degree_of_freedom.name not in true_degrees_of_freedom:
                continue
            true_id = true_degrees_of_freedom[degree_of_freedom.name].id
            self.ground_truth.state[true_id].position = self.believed_world.state[
                degree_of_freedom.id
            ].position
        for followed in self.followed_bodies:
            followed.follow()
        self.ground_truth.notify_state_change()

    def capture(self, annotators: list[BaseAnnotator]) -> CAS | None:
        """
        :param annotators: What RoboKudo does with the picture, once it is a point cloud.
        :return: What RoboKudo knows about a picture taken now, or ``None`` if the
            pipeline did not succeed.
        """
        pipeline = Pipeline("SimulatedHeadCameraPipeline")
        pipeline.add_children(
            [
                ClearAnnotatorOutputs(),
                CollectionReaderAnnotator(
                    descriptor=CollectionReaderDescriptorFactory.create_descriptor(
                        RayTracingCameraConfiguration.registry_name,
                        source=RuntimeRobotWorldSource(),
                        rgb_mode=self.rgb_mode,
                    )
                ),
                ImagePreprocessorAnnotator("ImagePreprocessor"),
                *annotators,
            ]
        )
        with self._lock:
            node = rclpy.create_node(self.ros_node_name)
            status = run_tree_once(
                pipeline,
                node,
                max_iterations=self.maximum_ticks,
                tick_rate=self.tick_rate,
            )
            node.destroy_node()
        if status is not Status.SUCCESS:
            return None
        return pipeline.cas

    def find_floor_objects(
        self, query: FloorObjectQuery, search_spot: FloorSpot
    ) -> list[FloorObjectDetection]:
        """
        Looks for the objects `query` describes, and keeps their points in
        :attr:`found_points`.

        :param query: What is looked for.
        :param search_spot: Where on the floor the objects are looked for.
        :return: The objects found near `search_spot`, the nearest first.
        """
        self.take_on_believed_state(wait=True)
        cas = self.capture(
            [
                query.create_crop_annotator(search_spot),
                PlaneAnnotator(),
                PointCloudClusterExtractor(),
                ClusterPoseBBAnnotator(),
            ]
        )
        if cas is None:
            return []
        found = sorted(
            (
                detection
                for hypothesis in cas.filter_annotations_by_type(ObjectHypothesis)
                for bounding_box in cas.filter_by_type(
                    BoundingBox3DAnnotation, hypothesis.annotations
                )
                if query.is_matched_by(
                    detection := self._detection_in_world(
                        hypothesis, bounding_box, cas
                    ),
                    search_spot,
                )
            ),
            key=lambda detection: detection.distance_to(search_spot),
        )
        chosen, others = found[:1], found[1:]
        self.found_points = ColoredPoints.from_uniformly_colored(
            [detection.points for detection in chosen], self.chosen_object_color
        ).concatenate(
            ColoredPoints.from_uniformly_colored(
                [detection.points for detection in others], self.other_object_color
            )
        )
        return found

    @staticmethod
    def _detection_in_world(
        hypothesis: ObjectHypothesis,
        bounding_box: BoundingBox3DAnnotation,
        cas: CAS,
    ) -> FloorObjectDetection:
        """
        :param hypothesis: An object RoboKudo found, with its points in the camera frame.
        :param bounding_box: The bounding box RoboKudo estimated for `hypothesis`, in the
            camera frame.
        :param cas: What the pipeline knows about the picture `hypothesis` was found in.
        :return: The object the box describes, extended down to the floor, since the
            camera only sees an object's upper and front faces.
        """
        world_T_box = get_camera_to_world_transform_matrix(
            cas
        ) @ get_transform_matrix_from_q(
            bounding_box.pose.rotation, bounding_box.pose.translation
        )
        top_height = world_T_box[2, 3] + bounding_box.z_length / 2
        return FloorObjectDetection(
            spot=FloorSpot(
                x=float(world_T_box[0, 3]),
                y=float(world_T_box[1, 3]),
                yaw=float(np.arctan2(world_T_box[1, 0], world_T_box[0, 0])),
            ),
            size=Scale(bounding_box.x_length, bounding_box.y_length, top_height),
            points=ColoredPoints.from_cloud(
                transform_cloud_from_camera_to_world(cas, hypothesis.points)
            ).positions,
        )


# %% seeing live what the camera sees


@dataclass(eq=False, kw_only=True)
class LiveCameraView(StateChangeCallback):
    """
    Publishes again and again what :attr:`camera` sees, and what it found last, while
    the robot moves in the world this callback watches.
    """

    camera: SimulatedHeadCamera
    """
    The camera whose pictures are published.
    """

    publishers: dict[CameraViewTopic, Publisher]
    """
    The publisher of every topic, keeping the last points for RViz displays added
    later.
    """

    ros_node: Node
    """
    The node whose clock stamps the points.
    """

    period: float = 1.0
    """
    How long to wait between two pictures, in seconds.
    """

    voxel_size: float = 0.01
    """
    The edge of the cubes a picture's points are thinned out to one point each, in
    meters, so it stays small enough to show.
    """

    _state_requested: threading.Event = field(
        init=False, default_factory=threading.Event
    )
    """
    Set while the next picture waits for the camera to take on the robot's state.
    """

    _state_taken: threading.Event = field(init=False, default_factory=threading.Event)
    """
    Set once the camera took on the robot's state for the next picture.
    """

    _stopped: threading.Event = field(init=False, default_factory=threading.Event)
    """
    Set once no more pictures are to be published.
    """

    _thread: threading.Thread = field(init=False)
    """
    Takes and publishes the pictures.
    """

    def __post_init__(self) -> None:
        """
        Unlike :meth:`StateChangeCallback.__post_init__`, also starts publishing the
        pictures.
        """
        super().__post_init__()
        self._thread = threading.Thread(
            target=self._publish_continuously, daemon=True, name="live_camera_view"
        )
        self._thread.start()

    def on_state_change(self, **kwargs) -> None:
        if not self._state_requested.is_set() or self._world.world_is_being_modified:
            return
        if not self.camera.take_on_believed_state(wait=False):
            return
        self._state_requested.clear()
        self._state_taken.set()

    def stop(self) -> None:
        """
        Unlike :meth:`StateChangeCallback.stop`, also waits for the last picture to be
        published.
        """
        self._stopped.set()
        self._thread.join()
        super().stop()

    def _publish_continuously(self) -> None:
        """
        Publishes a picture every :attr:`period` until :meth:`stop`. Without a change
        of the robot's state within :attr:`period`, the picture shows the last one.
        """
        while not self._stopped.is_set():
            self._state_taken.clear()
            self._state_requested.set()
            self._state_taken.wait(timeout=self.period)
            self._state_requested.clear()
            self._publish_picture()
            self._stopped.wait(timeout=self.period)

    def _publish_picture(self) -> None:
        """
        Takes a picture and publishes its points and :attr:`SimulatedHeadCamera.found_points`.
        """
        cas = self.camera.capture([])
        if cas is None:
            return
        scene = ColoredPoints.from_cloud(
            transform_cloud_from_camera_to_world(
                cas, cas.get(CASViews.CLOUD).voxel_down_sample(self.voxel_size)
            )
        )
        stamp = self.ros_node.get_clock().now().to_msg()
        frame_id = str(self._world.root.name)
        for topic, points in [
            (CameraViewTopic.SCENE, scene),
            (CameraViewTopic.OBJECTS, self.camera.found_points),
        ]:
            self.publishers[topic].publish(
                points.to_message(frame_id=frame_id, stamp=stamp)
            )


# %% forgetting what was taken away


@dataclass(eq=False, kw_only=True)
class TakenObjectEraser(StateChangeCallback):
    """
    Takes a found object's points out of :attr:`SimulatedHeadCamera.found_points` as
    soon as the object has left the spot it was seen at.
    """

    camera: SimulatedHeadCamera
    """
    The camera that found the object.
    """

    detection: FloorObjectDetection
    """
    What the camera saw of the object.
    """

    body: Body
    """
    The object in the robot's world.
    """

    displacement_tolerance: float = 0.02
    """
    How far the object may move from where it was seen, in meters, before it counts as
    taken away.
    """

    erase_distance: float = 0.02
    """
    How near to a point of the object a found point may be to be taken out with it, in
    meters.
    """

    seen_position: np.ndarray = field(init=False)
    """
    Where the object was when it was seen, in the world frame.
    """

    def __post_init__(self) -> None:
        """
        Unlike :meth:`StateChangeCallback.__post_init__`, also takes where :attr:`body`
        is now as where it was seen.
        """
        self.seen_position = self._body_position()
        super().__post_init__()

    def on_state_change(self, **kwargs) -> None:
        if (
            np.linalg.norm(self._body_position() - self.seen_position)
            <= self.displacement_tolerance
        ):
            return
        self.camera.found_points = self.camera.found_points.without_points_near(
            self.detection.points, self.erase_distance
        )
        self.stop()

    def _body_position(self) -> np.ndarray:
        """
        :return: Where :attr:`body` is now, in the world frame.
        """
        return self._world.compute_forward_kinematics_np(self._world.root, self.body)[
            :3, 3
        ]


# %% finding nothing


@dataclass
class NoObjectOnFloorFound(PerceptionException):
    """
    Raised when the camera sees no object of the size looked for near the search spot.
    """

    search_spot: FloorSpot
    """
    Where on the floor the object was looked for.
    """

    def error_message(self) -> str:
        return (
            f"No object of the size looked for was found on the floor near "
            f"({self.search_spot.x}, {self.search_spot.y})."
        )

    def suggest_correction(self) -> str:
        return "place the object closer to the search spot, or move the search spot."


# %% the demonstration


@dataclass
class PerceivedFloorToTrashCanDemonstration(FloorToTrashCanDemonstration):
    """
    The HSR finds the milk and the trash can with RoboKudo and throws the milk away.

    The robot's world starts without both; each is added when found. Step by step::

        first plan: park the arm, turn to and look at search_spot
        find the milk on the floor, add it as a box of the estimated size
        second plan: pick the milk up, bring it upright in front of the base, carry it
            past passage_spot to approach_distance before trash_can_search_spot,
            raise the torso and look there
        find the trash can, add it as a trash can of the estimated size
        third plan: place the milk above the trash can's opening, let go of it, park
            the arm and lower the torso
    """

    trash_can_search_spot: FloorSpot = field(
        default_factory=lambda: FloorSpot(x=2.3, y=2.0)
    )
    """
    The spot on the floor the robot looks at to find the trash can: in the living room,
    next to the sofa.
    """

    milk_query: FloorObjectQuery = field(
        default_factory=lambda: FloorObjectQuery(maximum_extent=0.3)
    )
    """
    How the milk is looked for: anything smaller than furniture.
    """

    trash_can_query: FloorObjectQuery = field(
        default_factory=lambda: FloorObjectQuery(
            minimum_extent=0.3, maximum_extent=0.6, search_radius=1.0, crop_height=0.6
        )
    )
    """
    How the trash can is looked for: bigger than what is picked up, smaller than a sofa.
    """

    carrying_height: float = 0.6
    """
    How high above the floor the milk's origin is carried, in meters, since the trash
    can, and so the height the milk is let go of at, is not found until later.
    """

    camera_view_period: float | None = 1.0
    """
    How long to wait between two pictures of the head camera published to RViz, in
    seconds; ``None`` publishes none.
    """

    camera_view_display_duration: float = 4.0
    """
    How long the robot waits after finding something, in seconds, so what the camera
    found can be watched before it moves on; it does not wait without
    :attr:`real_time_factor`.
    """

    camera: SimulatedHeadCamera | None = field(init=False, default=None)
    """
    The head camera in the world as it really is; set up when the demonstration runs.
    """

    live_camera_view: LiveCameraView | None = field(init=False, default=None)
    """
    Publishes what the head camera sees while the demonstration runs.
    """

    def build_statechart(self, context: StatechartContext) -> Statechart:
        """
        Unlike :meth:`FloorToTrashCanDemonstration.build_statechart`, only turns to and
        looks at :attr:`search_spot`; :meth:`run` does the rest.
        """
        hsrb = context.require_extension(RobotAccess).robot
        search_pose = self.search_spot.create_transform(context.world.root).pose
        self._let_robot_touch_the_floor(hsrb)
        statechart = Statechart(context=context)
        statechart.add_node(
            Sequence(
                nodes=[
                    ParkArmsAction(arms=hsrb.all_arms),
                    FaceAndLookAtAction(
                        face_at=FaceAtAction(target=search_pose),
                        look_at=LookAtAction(target=search_pose),
                    ),
                ]
            )
        )
        return statechart

    def build_trash_can_search(
        self, context: StatechartContext, milk_spot: FloorSpot
    ) -> Statechart:
        """
        :param context: The context of the plan.
        :param milk_spot: Where the milk was found.
        :return: Picking the milk up, carrying it past :attr:`passage_spot` to
            :attr:`approach_distance` before :attr:`trash_can_search_spot`, raising the
            torso and looking there from above, so the camera sees into the trash can.
        """
        hsrb = context.require_extension(RobotAccess).robot
        [arm] = hsrb.all_arms
        [milk] = context.world.get_semantic_annotations_by_type(Milk)
        search_pose = self.trash_can_search_spot.create_transform(
            context.world.root
        ).pose
        lookout_pose = self._compute_standing_pose_facing(
            search_pose, self.approach_distance, milk_spot
        )
        passage_pose = self._compute_floor_pose_facing(
            np.array([self.passage_spot.x, self.passage_spot.y]), lookout_pose
        )
        milk_pose = Pose(reference_frame=milk.root)
        grasp = self._create_grasp_from_above(milk)
        self._let_wrist_touch_the_object(arm, milk.root)
        statechart = Statechart(context=context)
        statechart.add_node(
            Sequence(
                nodes=[
                    a(MoveAndPickUpAction)(
                        navigate=a(NavigateAction)(
                            target_location=self._leave_standing_pose_open(
                                milk_pose, arm, context
                            )
                        ),
                        face_and_look_at=self._face_and_look_at(milk_pose),
                        pick_up=a(PickUpAction)(grasp=grasp, arm=arm),
                    ),
                    self._create_move_to_carrying_pose(hsrb, arm, grasp),
                    FaceAtAction(target=passage_pose),
                    CarryAction(
                        navigate=NavigateAction(target_location=passage_pose),
                        holding_arms=[arm],
                    ),
                    CarryAction(
                        navigate=NavigateAction(target_location=lookout_pose),
                        holding_arms=[arm],
                    ),
                    MoveTorsoAction(torso_state=TorsoState.HIGH),
                    LookAtAction(target=search_pose),
                ]
            )
        )
        return statechart

    def build_throw_away(self, context: StatechartContext) -> Statechart:
        """
        :param context: The context of the plan, whose robot holds the milk.
        :return: Driving to :attr:`drop_distance` before the trash can, on the side the
            robot stands, letting go of the milk above the opening, parking the arm and
            lowering the torso, and, in a kinematic simulation, putting the milk onto
            the trash can's bottom.
        """
        world = context.world
        hsrb = context.require_extension(RobotAccess).robot
        [arm] = hsrb.all_arms
        [milk] = world.get_semantic_annotations_by_type(Milk)
        [trash_can] = world.get_semantic_annotations_by_type(TrashCan)
        robot_x, robot_y = world.compute_forward_kinematics_np(world.root, hsrb.root)[
            :2, 3
        ]
        drop_pose = self._compute_drop_pose(trash_can.root, milk.root)
        steps: list[StatechartNode] = [
            CarryAction(
                navigate=NavigateAction(
                    target_location=self._compute_standing_pose_facing(
                        drop_pose,
                        self.drop_distance,
                        FloorSpot(x=float(robot_x), y=float(robot_y)),
                    )
                ),
                holding_arms=[arm],
            ),
            FaceAtAction(target=drop_pose),
            PlaceAction(object_designator=milk, target_location=drop_pose),
            Parallel(
                nodes=[
                    ParkArmsAction(arms=hsrb.all_arms),
                    MoveTorsoAction(torso_state=TorsoState.LOW),
                ]
            ),
        ]
        if issubclass(self.executor_type, KinematicallySimulatedPlanExecutor):
            steps.append(
                SettleOnContainerBottom(
                    item=milk.root,
                    container=trash_can.root,
                    wall_thickness=self.trash_can_wall_thickness,
                )
            )
        statechart = Statechart(context=context)
        statechart.add_node(Sequence(nodes=steps))
        return statechart

    @classmethod
    def _compute_standing_pose_facing(
        cls, target_pose: Pose, distance: float, approach_side: FloorSpot
    ) -> Pose:
        """
        :param target_pose: What the robot is to stand before.
        :param distance: How far from `target_pose` the base stands, in meters.
        :param approach_side: A spot on the side of `target_pose` the robot comes from.
        :return: A pose on the floor `distance` from `target_pose`, on the line towards
            `approach_side`, facing `target_pose`.
        """
        target_xy = np.array([float(target_pose.x), float(target_pose.y)])
        towards_side = np.array([approach_side.x, approach_side.y]) - target_xy
        standing_xy = target_xy + distance * towards_side / np.linalg.norm(towards_side)
        return cls._compute_floor_pose_facing(standing_xy, target_pose)

    def _create_move_to_carrying_pose(
        self, robot: AbstractRobot, arm: Arm, grasp: GraspCandidate
    ) -> MoveManipulatorAction:
        """
        :param robot: The robot carrying the milk.
        :param arm: The arm holding the milk.
        :param grasp: The grasp the milk is held by.
        :return: Bringing the milk upright to :attr:`carrying_reach` in front of the base
            and :attr:`carrying_height` above the floor, where it stays while the robot
            drives.
        """
        world = robot._world
        base_height = world.compute_forward_kinematics_np(world.root, robot.root)[2, 3]
        carrying_pose = Pose.from_xyz_rpy(
            x=self.carrying_reach,
            z=self.carrying_height - base_height,
            reference_frame=robot.root,
        )
        return MoveManipulatorAction(
            target_pose=arm.end_effector.tool_frame_goal_for_grasp(
                grasp.moved_to(carrying_pose)
            ),
            end_effector=arm.end_effector,
            allow_gripper_collision=False,
        )

    def run(self) -> World:
        """
        Unlike :meth:`FloorToTrashCanDemonstration.run`, takes the milk and the trash can
        out of the robot's world, builds a second world as it really is for the camera,
        and performs three plans, finding the milk and the trash can between them.

        :raises NoObjectOnFloorFound: If the camera sees no milk or no trash can.
        """
        try:
            world = self.acquire_world()
            self._forget(world, Milk)
            self._forget(world, TrashCan)
            self.camera = SimulatedHeadCamera(
                ground_truth=self.build_simulated_world(), believed_world=world
            )
            self._start_live_camera_view(world)
            self.start_simulation(world)
            self._perform(self.build_statechart, world)
            milk_found = self._add_milk_found(world)
            self._perform(
                partial(self.build_trash_can_search, milk_spot=milk_found.spot), world
            )
            self._add_trash_can_found(world)
            self._perform(self.build_throw_away, world)
        finally:
            if self.live_camera_view is not None:
                self.live_camera_view.stop()
            self.tear_down()
        return world

    def _perform(
        self, build: Callable[[StatechartContext], Statechart], world: World
    ) -> None:
        """
        Performs a plan, then lets :attr:`camera` take on the state it ended in.

        :param build: Builds the plan in the context of a new executor.
        :param world: The world the plan is performed in.
        """
        executor = self.create_executor(world)
        executor.compile(build(executor.context))
        executor.execute()
        executor.close()
        self.camera.take_on_believed_state(wait=True)

    @staticmethod
    def _forget(world: World, annotation_type: type[SemanticAnnotation]) -> None:
        """
        Takes the one semantic annotation of `annotation_type` and its bodies out of
        `world`, so the robot has to find it.

        :param world: The robot's world, populated by
            :meth:`FloorToTrashCanDemonstration.populate_scene`.
        :param annotation_type: What the robot is to find.
        """
        [annotation] = world.get_semantic_annotations_by_type(annotation_type)
        with world.modify_world():
            world.remove_semantic_annotation(annotation)
            world.remove_branch_from_world(annotation.root)
            world.delete_orphaned_dofs()

    def _start_live_camera_view(self, world: World) -> None:
        """
        Publishes what the head camera sees every :attr:`camera_view_period`. Does
        nothing without a ROS node or a period.

        :param world: The robot's world, whose root frame the points are published in.
        """
        if self.ros_node is None or self.camera_view_period is None:
            return
        self.live_camera_view = LiveCameraView(
            _world=world,
            camera=self.camera,
            publishers={
                topic: self.ros_node.create_publisher(
                    PointCloud2,
                    topic,
                    QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL),
                )
                for topic in CameraViewTopic
            },
            ros_node=self.ros_node,
            period=self.camera_view_period,
        )

    def _find(
        self, world: World, query: FloorObjectQuery, search_spot: FloorSpot
    ) -> FloorObjectDetection:
        """
        Looks for the objects `query` describes near `search_spot`, then looks straight
        at the nearest and measures it again, and waits
        :attr:`camera_view_display_duration` unless :attr:`real_time_factor` is
        ``None``, so what was found can be watched.

        :param world: The robot's world.
        :return: The object nearest to `search_spot`, as measured the second time.
        :raises NoObjectOnFloorFound: If nothing is found.
        """
        glance = self._find_nearest(query, search_spot)
        target = glance.spot.create_transform(world.root, height=glance.size.z / 2).pose
        self._perform(partial(self._build_look_at, target=target), world)
        found = self._find_nearest(query, glance.spot)
        if self.real_time_factor is not None:
            time.sleep(self.camera_view_display_duration)
        return found

    def _find_nearest(
        self, query: FloorObjectQuery, search_spot: FloorSpot
    ) -> FloorObjectDetection:
        """
        :return: The object `query` describes nearest to `search_spot`.
        :raises NoObjectOnFloorFound: If nothing is found.
        """
        detections = self.camera.find_floor_objects(query, search_spot)
        if not detections:
            raise NoObjectOnFloorFound(search_spot=search_spot)
        return detections[0]

    @staticmethod
    def _build_look_at(context: StatechartContext, target: Pose) -> Statechart:
        """
        :param context: The context of the plan.
        :param target: What the robot looks at.
        :return: Pointing the head at `target`.
        """
        statechart = Statechart(context=context)
        statechart.add_node(LookAtAction(target=target))
        return statechart

    def _add_milk_found(self, world: World) -> FloorObjectDetection:
        """
        Looks for the milk near :attr:`search_spot` and adds it to `world` as a box of
        the estimated size. From then on, the real milk moves along with the box, and its
        points are no longer shown as found once the robot takes it away.

        :param world: The robot's world.
        :return: The milk found.
        :raises NoObjectOnFloorFound: If nothing is found.
        """
        found = self._find(world, self.milk_query, self.search_spot)
        with world.modify_world():
            milk = SemanticAnnotationWithRootSpecification(
                name=SceneBody.MILK,
                semantic_annotation_type=Milk,
                root_specification=BodySpecification.box(
                    SceneBody.MILK,
                    found.size,
                    connection_specification=Connection6DoFSpecification(),
                ),
            ).spawn(world)
        milk.root.parent_connection.origin = found.spot.create_transform(
            world.root, height=found.size.z / 2
        )
        [true_milk] = self.camera.ground_truth.get_semantic_annotations_by_type(Milk)
        self.camera.follow(true_milk.root, milk.root)
        TakenObjectEraser(
            _world=world, camera=self.camera, detection=found, body=milk.root
        )
        return found

    def _add_trash_can_found(self, world: World) -> None:
        """
        Looks for the trash can near :attr:`trash_can_search_spot` and adds it to `world`
        as a trash can of the estimated size, with walls
        :attr:`trash_can_wall_thickness` thick.

        :param world: The robot's world.
        :raises NoObjectOnFloorFound: If nothing is found.
        """
        found = self._find(world, self.trash_can_query, self.trash_can_search_spot)
        with world.modify_world():
            TrashCan.get_annotation_specification(
                PerceivedSceneBody.TRASH_CAN,
                TrashCan.get_default_root_kinematic_structure_entity_specification(
                    scale=found.size, wall_thickness=self.trash_can_wall_thickness
                ),
            ).spawn(
                world,
                parent_T_self=found.spot.create_transform(
                    world.root, height=found.size.z / 2
                ),
            )


def main() -> World:
    """
    Run the demonstration.

    :return: The robot's world after the milk was thrown away.
    """
    return PerceivedFloorToTrashCanDemonstration().run()


if __name__ == "__main__":
    main()
