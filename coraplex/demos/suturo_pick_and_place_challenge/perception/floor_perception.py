"""
Finding objects that stand on the floor with RoboKudo, through the HSR's head camera:

rendered in simulation, or the real one on the robot. Both run the same annotators, so
what is found in simulation is found the same way on the robot.
"""

from __future__ import annotations

import array
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import timedelta
from enum import Enum, IntEnum, StrEnum

import numpy as np
import open3d
import rclpy
from builtin_interfaces.msg import Time
from exceptions import CameraImagesMissing
from krrood_spatial_math.spatial_types import (
    HomogeneousTransformationMatrix,
    Point2,
    Pose2D,
)
from py_trees.common import Status
from rclpy.node import Node
from rclpy.publisher import Publisher
from rclpy.qos import DurabilityPolicy, QoSProfile
from robokudo.annotators.cluster_pose_bb import ClusterPoseBBAnnotator
from robokudo.annotators.collection_reader import CollectionReaderAnnotator
from robokudo.annotators.core import BaseAnnotator
from robokudo.annotators.image_preprocessor import ImagePreprocessorAnnotator
from robokudo.annotators.plane import PlaneAnnotator
from robokudo.annotators.pointcloud_cluster_extractor import PointCloudClusterExtractor
from robokudo.annotators.pointcloud_crop import PointcloudCropAnnotator
from robokudo.behaviours.clear_annotator_outputs import ClearAnnotatorOutputs
from robokudo.behaviours.clear_errors import ClearErrors
from robokudo.cas import CAS, CASViews
from robokudo.descriptors.camera_configs.config_hsr_ros2 import HsrRos2CameraConfig
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
from semantic_digital_twin.adapters.ros.node_registry import ROSNodeRegistry
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
from semantic_digital_twin.robots.robot_parts import AbstractRobot
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.geometry import (
    CoordinateBounds,
    PlanarBoundingBox,
    Scale,
)
from semantic_digital_twin.world_description.world_entity import (
    Body,
    KinematicStructureEntity,
)
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Header

from robokudo import world as robokudo_world

# %% where an object is looked for


class Side(IntEnum):
    """
    A side of a piece of furniture, as someone who faces out of its front sees it; the
    value is the sign of the turn from the front towards that side.
    """

    LEFT = 1
    RIGHT = -1


class FurnitureFront(Enum):
    """
    The axis of a piece of furniture's own frame that its front faces along, as x and y
    of that axis.
    """

    POSITIVE_X = (1.0, 0.0)
    NEGATIVE_X = (-1.0, 0.0)
    POSITIVE_Y = (0.0, 1.0)
    NEGATIVE_Y = (0.0, -1.0)


@dataclass(kw_only=True)
class FloorRegion:
    """
    A rectangle on the floor that an object is known to stand in, without knowing where
    in it.
    """

    area: PlanarBoundingBox
    """
    The rectangle, axis-aligned in the frame of its origin, which is a body of a world.
    """

    @classmethod
    def from_furniture_side(
        cls,
        *,
        furniture: Body,
        side: Side,
        front: FurnitureFront,
        reach: float,
        gap: float = 0.05,
        margin: float = 0.3,
    ) -> FloorRegion:
        """
        :param furniture: The furniture the object stands next to.
        :param side: The side of `furniture` the object stands at.
        :param front: The axis the front of `furniture` faces along.
        :param reach: How far beyond the side of `furniture` the object may stand, in
            meters.
        :param gap: How far the region keeps away from `furniture`, in meters.
        :param margin: How far the region reaches beyond the front and the back of
            `furniture`, in meters.
        :return: The region at `side` of `furniture`, as deep as `furniture` plus
            `margin` towards its front and its back, in the frame of `furniture`.
        """
        box = furniture.collision.as_bounding_box_collection_in_frame(
            furniture
        ).bounding_box()
        box_bounds = PlanarBoundingBox(
            min_x=box.min_x,
            min_y=box.min_y,
            max_x=box.max_x,
            max_y=box.max_y,
            origin=HomogeneousTransformationMatrix(reference_frame=furniture),
        ).to_array_bounds()
        frontwards = np.array(front.value)
        outwards = side.value * np.array([-frontwards[1], frontwards[0]])
        center = (box_bounds.lower + box_bounds.upper) / 2
        half_extents = (box_bounds.upper - box_bounds.lower) / 2
        inner_edge = center + outwards * (np.abs(outwards) @ half_extents + gap)
        across = frontwards * (np.abs(frontwards) @ half_extents + margin)
        corners = np.array(
            [inner_edge - across, inner_edge + outwards * reach + across]
        )
        return cls(
            area=PlanarBoundingBox.from_array_bounds(
                corners.min(axis=0),
                corners.max(axis=0),
                HomogeneousTransformationMatrix(reference_frame=furniture),
            )
        )

    @property
    def world(self) -> World:
        """
        The world the region's origin belongs to.
        """
        return self.area.origin.reference_frame._world

    @property
    def world_T_area(self) -> np.ndarray:
        """
        Where the frame the rectangle is axis-aligned in is, in the world frame.
        """
        return (
            self.world.compute_forward_kinematics_np(
                self.world.root, self.area.origin.reference_frame
            )
            @ self.area.origin.to_np()
        )

    def compute_world_xy(self, *, area_xy: np.ndarray) -> np.ndarray:
        """
        :param area_xy: Points on the floor, one row of x and y per point, in the frame
            the rectangle is axis-aligned in.
        :return: The same points, in the world frame.
        """
        area_xy = np.atleast_2d(area_xy)
        homogeneous = np.column_stack(
            [area_xy, np.zeros(len(area_xy)), np.ones(len(area_xy))]
        )
        return (homogeneous @ self.world_T_area.T)[:, :2]

    @property
    def center_xy(self) -> np.ndarray:
        """
        The middle of the region, as x and y in the world frame.
        """
        bounds = self.area.to_array_bounds()
        return self.compute_world_xy(area_xy=(bounds.lower + bounds.upper) / 2)[0]

    @property
    def world_bounds(self) -> CoordinateBounds:
        """
        The smallest rectangle that is axis-aligned in the world frame and holds the
        region.
        """
        return self.area.transform_to_origin(
            HomogeneousTransformationMatrix(reference_frame=self.world.root)
        ).to_array_bounds()

    def contains(self, *, xy: np.ndarray) -> bool:
        """
        :param xy: A point on the floor, as x and y in the world frame.
        :return: Whether `xy` lies inside the region.
        """
        return self.area.contains(
            Point2(float(xy[0]), float(xy[1]), reference_frame=self.world.root)
        )

    def compute_look_targets(
        self, *, viewer: Pose2D, minimum_distance: float, spacing: float
    ) -> list[np.ndarray]:
        """
        :param viewer: Where the robot stands and which way it faces, in the world frame.
        :param minimum_distance: How far from `viewer` a spot must be for the head camera
            to see the floor around it, in meters.
        :param spacing: How far apart the spots are, in meters, at most as far as the
            camera sees across the floor.
        :return: The spots on the floor, as x and y in the world frame, that bring the
            whole region into view one after the other: the middles of cells `spacing`
            wide, each moved out to `minimum_distance` from `viewer`, those `viewer`
            turns to least first.
        """
        viewer_xy = viewer.position.to_np()
        bounds = self.area.to_array_bounds()
        extents = bounds.upper - bounds.lower
        cell_counts = np.maximum(1, np.ceil(extents / spacing)).astype(int)
        cell_middles = np.array(
            [
                bounds.lower + (np.array([column, row]) + 0.5) / cell_counts * extents
                for column in range(cell_counts[0])
                for row in range(cell_counts[1])
            ]
        )
        targets: list[np.ndarray] = []
        for cell_xy in self.compute_world_xy(area_xy=cell_middles):
            offset = cell_xy - viewer_xy
            distance = float(np.linalg.norm(offset))
            if distance == 0.0:
                continue
            target = viewer_xy + offset * max(1.0, minimum_distance / distance)
            if all(np.linalg.norm(target - other) >= spacing / 2 for other in targets):
                targets.append(target)

        def turn_towards(target: np.ndarray) -> float:
            bearing = np.arctan2(target[1] - viewer_xy[1], target[0] - viewer_xy[0])
            return abs((bearing - float(viewer.yaw) + np.pi) % (2 * np.pi) - np.pi)

        return sorted(targets, key=turn_towards)

    def describe(self) -> str:
        """
        :return: The region as a person reads it, in a log line or an error message.
        """
        extents = self.area.to_array_bounds()
        length, width = extents.upper - extents.lower
        return (
            f"the {length:.2f} m x {width:.2f} m region around "
            f"({self.center_xy[0]:.2f}, {self.center_xy[1]:.2f})"
        )


# %% what RoboKudo finds


@dataclass(kw_only=True)
class FloorObjectDetection:
    """
    An object RoboKudo found standing on the floor: where it stands and how big it is.
    """

    spot: Pose2D
    """
    Where on the floor the object stands and which way it is turned, in the world frame.
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
    def xy(self) -> np.ndarray:
        """
        Where on the floor the object stands, as x and y in the world frame.
        """
        return self.spot.position.to_np()

    @property
    def largest_extent(self) -> float:
        """
        The largest of the object's extents, in meters.
        """
        return max(self.size.x, self.size.y, self.size.z)

    @property
    def bottom_height(self) -> float:
        """
        How high above the floor the lowest point seen of the object is, in meters.
        """
        return float(self.points[:, 2].min())

    def describe(self) -> str:
        """
        :return: Where the object stands and how big it is, as a person reads it in a log
            line.
        """
        return (
            f"({self.xy[0]:.2f}, {self.xy[1]:.2f}) of size {self.size.x:.2f} x "
            f"{self.size.y:.2f} x {self.size.z:.2f}"
        )

    def create_center_transform(
        self, *, world_root: KinematicStructureEntity
    ) -> HomogeneousTransformationMatrix:
        """
        :param world_root: The root of the world the object is put into.
        :return: The middle of the object's box, turned the way the object is.
        """
        return HomogeneousTransformationMatrix.from_xyz_rpy(
            x=float(self.spot.x),
            y=float(self.spot.y),
            z=self.size.z / 2,
            yaw=float(self.spot.yaw),
            reference_frame=world_root,
        )


@dataclass(kw_only=True)
class FloorObjectQuery:
    """
    What is looked for on the floor in a region: how big it is.
    """

    maximum_extent: float
    """
    The largest extent a found object may have, in meters.
    """

    minimum_extent: float = 0.0
    """
    How much a found object's largest extent must exceed, in meters.
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

    region_margin: float = 0.2
    """
    How far beyond the search region the point cloud is kept, in meters, so an object
    standing at its edge is seen whole.
    """

    def create_crop_annotator(self, *, region: FloorRegion) -> PointcloudCropAnnotator:
        """
        :param region: Where on the floor the objects are looked for.
        :return: An annotator keeping the points just above the floor around `region`.
        """
        bounds = region.world_bounds
        crop = PointcloudCropAnnotator.Descriptor()
        crop.parameters.relative_to_world = True
        crop.parameters.min_x = float(bounds.lower[0]) - self.region_margin
        crop.parameters.max_x = float(bounds.upper[0]) + self.region_margin
        crop.parameters.min_y = float(bounds.lower[1]) - self.region_margin
        crop.parameters.max_y = float(bounds.upper[1]) + self.region_margin
        crop.parameters.min_z = -self.floor_depth
        crop.parameters.max_z = self.crop_height
        return PointcloudCropAnnotator(descriptor=crop)

    def is_matched_by(
        self, *, detection: FloorObjectDetection, region: FloorRegion
    ) -> bool:
        """
        :return: Whether `detection` is as big as looked for and stands on the floor
            inside `region`.
        """
        return (
            self.minimum_extent < detection.largest_extent <= self.maximum_extent
            and detection.bottom_height <= self.maximum_floor_gap
            and region.contains(xy=detection.xy)
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

    def create_publisher(self, *, node: Node) -> Publisher:
        """
        :param node: The node that publishes.
        :return: A publisher of points on this topic that keeps the last points for RViz
            displays added later.
        """
        return node.create_publisher(
            PointCloud2,
            self,
            QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL),
        )


@dataclass(kw_only=True)
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
    def from_cloud(cls, *, cloud: open3d.geometry.PointCloud) -> ColoredPoints:
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
        cls, *, positions: list[np.ndarray], color: tuple[float, float, float]
    ) -> ColoredPoints:
        """
        :param positions: Groups of points, one row of x, y and z per point.
        :param color: The red, green and blue in [0, 1] every point gets.
        :return: The points of all groups together, each in `color`.
        """
        joined = np.concatenate([np.empty((0, 3)), *positions])
        return cls(positions=joined, colors=np.tile(color, (len(joined), 1)))

    def without_points_near(
        self, *, positions: np.ndarray, distance: float
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

    def concatenate(self, *, other: ColoredPoints) -> ColoredPoints:
        """
        :return: The points of this and of `other` together, each keeping its color.
        """
        return ColoredPoints(
            positions=np.concatenate([self.positions, other.positions]),
            colors=np.concatenate([self.colors, other.colors]),
        )

    def to_message(self, *, frame_id: str, stamp: Time) -> PointCloud2:
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


# %% the head camera


@dataclass(kw_only=True)
class HeadCamera(ABC):
    """
    The robot's head camera, which RoboKudo looks through to find objects on the floor.
    """

    maximum_ticks: int = 100
    """
    How often a pipeline is ticked at most before it counts as having seen nothing.
    """

    tick_rate: int = 20
    """
    How often per second a pipeline is ticked.
    """

    pipeline_node_name: str = "head_camera_pipeline"
    """
    Name of the ROS node a pipeline runs on.
    """

    chosen_object_color: tuple[float, float, float] = (0.1, 0.9, 0.2)
    """
    The red, green and blue in [0, 1] the points of the found object the robot acts on
    are shown in.
    """

    other_object_color: tuple[float, float, float] = (1.0, 0.5, 0.0)
    """
    The red, green and blue in [0, 1] the points of every other found object are shown
    in.
    """

    found_points: ColoredPoints = field(
        init=False,
        default_factory=lambda: ColoredPoints.from_uniformly_colored(
            positions=[], color=(0, 0, 0)
        ),
    )
    """
    The points of the objects found the last time objects were looked for.
    """

    @abstractmethod
    def create_collection_reader(self) -> CollectionReaderAnnotator:
        """
        :return: The annotator that turns the camera's next picture into a point cloud.
        """

    @abstractmethod
    def synchronize_with_robot(self) -> None:
        """
        Make sure the next picture shows what the robot sees from where it stands now.
        """

    def capture(self, *, annotators: list[BaseAnnotator]) -> CAS | None:
        """
        :param annotators: What RoboKudo does with the picture, once it is a point cloud.
        :return: What RoboKudo knows about a picture taken now, or ``None`` if the
            pipeline did not succeed.
        """
        pipeline = Pipeline("HeadCameraPipeline")
        pipeline.add_children(
            [
                ClearErrors(),
                ClearAnnotatorOutputs(),
                self.create_collection_reader(),
                ImagePreprocessorAnnotator("ImagePreprocessor"),
                *annotators,
            ]
        )
        node = rclpy.create_node(self.pipeline_node_name)
        status = run_tree_once(
            pipeline, node, max_iterations=self.maximum_ticks, tick_rate=self.tick_rate
        )
        node.destroy_node()
        if status is not Status.SUCCESS:
            return None
        return pipeline.cas

    def find_floor_objects(
        self, *, query: FloorObjectQuery, region: FloorRegion, nearest_to: np.ndarray
    ) -> list[FloorObjectDetection]:
        """
        Looks for the objects `query` describes, and keeps their points in
        :attr:`found_points`.

        :param query: What is looked for.
        :param region: Where on the floor the objects are looked for.
        :param nearest_to: A spot on the floor, as x and y in the world frame.
        :return: The objects found in `region`, the one nearest to `nearest_to` first.
        """
        self.synchronize_with_robot()
        cas = self.capture(
            annotators=[
                query.create_crop_annotator(region=region),
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
                    detection=(
                        detection := self._detection_in_world(
                            hypothesis=hypothesis, bounding_box=bounding_box, cas=cas
                        )
                    ),
                    region=region,
                )
            ),
            key=lambda detection: float(np.linalg.norm(detection.xy - nearest_to)),
        )
        chosen, others = found[:1], found[1:]
        self.found_points = ColoredPoints.from_uniformly_colored(
            positions=[detection.points for detection in chosen],
            color=self.chosen_object_color,
        ).concatenate(
            other=ColoredPoints.from_uniformly_colored(
                positions=[detection.points for detection in others],
                color=self.other_object_color,
            )
        )
        return found

    @staticmethod
    def _detection_in_world(
        *, hypothesis: ObjectHypothesis, bounding_box: BoundingBox3DAnnotation, cas: CAS
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
            spot=Pose2D.from_pose(
                HomogeneousTransformationMatrix(data=world_T_box).pose
            ),
            size=Scale(bounding_box.x_length, bounding_box.y_length, top_height),
            points=ColoredPoints.from_cloud(
                cloud=transform_cloud_from_camera_to_world(cas, hypothesis.points)
            ).positions,
        )


# %% the simulated head camera


@dataclass(kw_only=True)
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
    def from_bodies(cls, *, true_body: Body, believed_body: Body) -> FollowedBody:
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


@dataclass(kw_only=True)
class SimulatedHeadCamera(HeadCamera):
    """
    The robot's head camera, rendered in :attr:`ground_truth`, the world as it really
    is, with the robot standing as in :attr:`believed_world`.
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
            field_of_view=camera.camera_model.field_of_view
        )
        robokudo_world.set_world(self.ground_truth)

    def _create_camera_model(self, *, field_of_view: FieldOfView) -> PinholeCameraModel:
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

    def follow(self, *, true_body: Body, believed_body: Body) -> None:
        """
        Lets `true_body` move along with `believed_body` from now on.
        """
        self.followed_bodies.append(
            FollowedBody.from_bodies(true_body=true_body, believed_body=believed_body)
        )

    def take_on_believed_state(self, *, wait: bool) -> bool:
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

    def create_collection_reader(self) -> CollectionReaderAnnotator:
        return CollectionReaderAnnotator(
            descriptor=CollectionReaderDescriptorFactory.create_descriptor(
                RayTracingCameraConfiguration.registry_name,
                source=RuntimeRobotWorldSource(),
                rgb_mode=self.rgb_mode,
            )
        )

    def synchronize_with_robot(self) -> None:
        self.take_on_believed_state(wait=True)

    def capture(self, *, annotators: list[BaseAnnotator]) -> CAS | None:
        """
        Does what :meth:`HeadCamera.capture` does, while nothing changes
        :attr:`ground_truth`.
        """
        with self._lock:
            return super().capture(annotators=annotators)


# %% the real head camera


@dataclass(kw_only=True)
class RobotHeadCamera(HeadCamera):
    """
    The real HSR's head RGB-D camera, read over ROS, with the camera's pose taken from
    tf in the map frame, so what it finds is in the frame of the robot's world.
    """

    ros_node: Node
    """
    The node the pictures and tf arrive on; something else has to spin it the whole
    time, so that the newest picture is always at hand.
    """

    camera_configuration: str = HsrRos2CameraConfig.registry_name
    """
    The RoboKudo camera configuration naming the topics and frames of the camera.
    """

    settling_time: timedelta = timedelta(seconds=0.5)
    """
    How much later than the request a picture must have been taken, so the head has
    stopped moving and the picture is sharp.
    """

    image_timeout: timedelta = timedelta(seconds=10)
    """
    How long to wait for such a picture before
    :class:`~exceptions.CameraImagesMissing` is raised.
    """

    polling_period: timedelta = timedelta(seconds=0.05)
    """
    How long to wait between two checks whether such a picture arrived.
    """

    _collection_reader_descriptor: CollectionReaderAnnotator.Descriptor = field(
        init=False
    )
    """
    The camera interface, subscribed to the camera for the camera's whole life, so a
    picture never has to be waited for from scratch.
    """

    def __post_init__(self) -> None:
        """
        Subscribes to the camera through :attr:`ros_node`, and gives RoboKudo a world of
        its own to put the camera's pose into, so the robot's world stays untouched.
        """
        registry = ROSNodeRegistry()
        previous_node = registry.get() if registry.has_node() else None
        registry.register(self.ros_node)
        self._collection_reader_descriptor = (
            CollectionReaderDescriptorFactory.create_descriptor(
                self.camera_configuration
            )
        )
        if previous_node is None:
            registry.clear(expected_node=self.ros_node)
        else:
            registry.register(previous_node)
        robokudo_world.clear_world()

    @property
    def world_frame(self) -> str:
        """
        The tf frame the camera's pose is looked up in, which what it finds is given in.
        """
        return self._collection_reader_descriptor.parameters.camera_config.tf_to

    def create_collection_reader(self) -> CollectionReaderAnnotator:
        return CollectionReaderAnnotator(descriptor=self._collection_reader_descriptor)

    def synchronize_with_robot(self) -> None:
        camera = self._collection_reader_descriptor.parameters.camera_interface
        requested_at = self._read_newest_image_time()
        deadline = time.monotonic() + self.image_timeout.total_seconds()
        while time.monotonic() < deadline:
            taken_at = self._read_newest_image_time()
            if taken_at is not None and (
                requested_at is None
                or taken_at - requested_at >= self.settling_time.total_seconds()
            ):
                return
            time.sleep(self.polling_period.total_seconds())
        raise CameraImagesMissing(
            topic=camera.camera_config.topic_color, waited=self.image_timeout
        )

    def _read_newest_image_time(self) -> float | None:
        """
        :return: When the newest picture at hand was taken, in seconds of the robot's
            clock, ``None`` before the first one arrived.
        """
        stamp = self._collection_reader_descriptor.parameters.camera_interface.timestamp
        if stamp is None:
            return None
        return stamp.sec + stamp.nanosec * 1e-9


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
    The publisher of every topic, keeping the last points for RViz displays added later.
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
        Publishes a picture every :attr:`period` until :meth:`stop`.

        Without a change of the robot's state within :attr:`period`, the picture shows
        the last one.
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
        Takes a picture and publishes its points and :attr:`HeadCamera.found_points`.
        """
        cas = self.camera.capture(annotators=[])
        if cas is None:
            return
        scene = ColoredPoints.from_cloud(
            cloud=transform_cloud_from_camera_to_world(
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
    Takes a found object's points out of :attr:`HeadCamera.found_points` as soon as the
    object has left the spot it was seen at.
    """

    camera: HeadCamera
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
            positions=self.detection.points, distance=self.erase_distance
        )
        self.stop()

    def _body_position(self) -> np.ndarray:
        """
        :return: Where :attr:`body` is now, in the world frame.
        """
        return self._world.compute_forward_kinematics_np(self._world.root, self.body)[
            :3, 3
        ]
