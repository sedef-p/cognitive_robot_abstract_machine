"""
Tests of :class:`floor_perception.RobotHeadCamera` against a stand-in for the HSR's head
camera, which publishes pictures of a box on the floor on the topics and tf frames of the
real one. Run them from this folder, with ROS 2 sourced::

    python -m pytest robot_head_camera_test.py
"""

from __future__ import annotations

import threading
from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import timedelta
from enum import StrEnum
from types import TracebackType

import cv2
import numpy as np
import pytest
import rclpy
from coraplex.demonstrations import RobotDemonstrationRosSession
from exceptions import CameraImagesMissing
from floor_perception import FloorObjectQuery, FloorRegion, RobotHeadCamera
from geometry_msgs.msg import TransformStamped
from krrood_spatial_math.spatial_types import HomogeneousTransformationMatrix
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from rclpy.publisher import Publisher
from robokudo.descriptors.camera_configs.config_hsr_ros2 import HsrRos2CameraConfig
from scipy.spatial.transform import Rotation
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.geometry import PlanarBoundingBox, Scale
from semantic_digital_twin.world_description.world_entity import Body
from sensor_msgs.msg import CameraInfo, CompressedImage
from std_msgs.msg import Header
from tf2_ros import StaticTransformBroadcaster
from typing_extensions import Self

# %% a stand-in for the head camera


class RosNodeName(StrEnum):
    """
    The names of the ROS nodes these tests start.
    """

    CAMERA = "box_on_floor_camera"
    """
    The node publishing the pictures of the stand-in camera.
    """

    SESSION = "robot_head_camera_test"
    """
    The node the pictures are received on, as a demonstration's node.
    """


class CompressedImageFormat(StrEnum):
    """
    The formats the HSR's head camera compresses its pictures in.
    """

    COLOR = "bgr8; jpeg"
    """
    A color picture, as JPEG.
    """

    DEPTH = "16UC1; compressedDepth png"
    """
    A depth picture in millimeters, as PNG behind a header.
    """


@dataclass(kw_only=True)
class RenderedPictures:
    """
    A color and a depth picture of the same view.
    """

    color: np.ndarray
    """
    The color picture, in BGR order: gray floor, red box.
    """

    depth_in_millimeters: np.ndarray
    """
    The depth picture, 0 where nothing is hit.
    """


@dataclass(kw_only=True)
class BoxOnFloorCamera:
    """
    Publishes, like the HSR's head camera, pictures of a box standing on the floor,
    rendered by casting one ray per pixel.
    """

    camera_position: np.ndarray
    """
    Where the camera is, in the map frame.
    """

    looked_at: np.ndarray
    """
    The point the camera looks at, in the map frame.
    """

    box_center: np.ndarray
    """
    Where on the floor the middle of the box stands, as x and y in the map frame.
    """

    box_size: Scale
    """
    The extents of the box along the map's axes, in meters.
    """

    configuration: HsrRos2CameraConfig = field(default_factory=HsrRos2CameraConfig)
    """
    The topics and frames of the real camera, which the pictures are published on.
    """

    width: int = 640
    """
    How many pixels wide a picture is.
    """

    height: int = 480
    """
    How many pixels high a picture is.
    """

    focal_length: float = 525.0
    """
    The focal length of the camera, in pixels.
    """

    period: float = 1.0 / 15.0
    """
    How long to wait between two pictures, in seconds.
    """

    depth_header_size: int = 12
    """
    How many bytes of header come before the PNG of a compressed depth picture.
    """

    node: Node = field(init=False)
    """
    The node publishing the pictures.
    """

    _color_data: bytes = field(init=False)
    """
    The color picture, compressed as the camera sends it.
    """

    _depth_data: bytes = field(init=False)
    """
    The depth picture, compressed as the camera sends it.
    """

    _broadcaster: StaticTransformBroadcaster = field(init=False)
    """
    Publishes where the camera is.
    """

    _executor: SingleThreadedExecutor = field(init=False)
    """
    Spins :attr:`node`.
    """

    _thread: threading.Thread = field(init=False)
    """
    Runs :attr:`_executor`.
    """

    def __enter__(self) -> Self:
        self.node = rclpy.create_node(RosNodeName.CAMERA)
        pictures = self._render()
        self._color_data = cv2.imencode(".jpg", pictures.color)[1].tobytes()
        self._depth_data = (
            bytes(self.depth_header_size)
            + cv2.imencode(".png", pictures.depth_in_millimeters)[1].tobytes()
        )
        self._broadcaster = StaticTransformBroadcaster(self.node)
        self._broadcaster.sendTransform(self._create_camera_transform())
        color_publisher, depth_publisher = (
            self.node.create_publisher(CompressedImage, topic, 10)
            for topic in (
                self.configuration.topic_color,
                self.configuration.topic_depth,
            )
        )
        info_publisher = self.node.create_publisher(
            CameraInfo, self.configuration.topic_camera_info, 10
        )
        self.node.create_timer(
            self.period,
            lambda: self._publish(
                color_publisher=color_publisher,
                depth_publisher=depth_publisher,
                info_publisher=info_publisher,
            ),
        )
        self._executor = SingleThreadedExecutor()
        self._executor.add_node(self.node)
        self._thread = threading.Thread(target=self._executor.spin, daemon=True)
        self._thread.start()
        return self

    def __exit__(
        self,
        exception_type: type[BaseException] | None,
        exception: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self._executor.shutdown()
        self._thread.join()
        self.node.destroy_node()

    @property
    def map_R_camera(self) -> np.ndarray:
        """
        The orientation of the camera's optical frame, x right, y down and z forwards.
        """
        forwards = self.looked_at - self.camera_position
        forwards /= np.linalg.norm(forwards)
        right = np.cross(forwards, [0.0, 0.0, 1.0])
        right /= np.linalg.norm(right)
        return np.column_stack([right, np.cross(forwards, right), forwards])

    def _render(self) -> RenderedPictures:
        """
        :return: What the camera sees of the floor and the box.
        """
        columns, rows = np.meshgrid(np.arange(self.width), np.arange(self.height))
        camera_rays = np.stack(
            [
                (columns - self.width / 2) / self.focal_length,
                (rows - self.height / 2) / self.focal_length,
                np.ones_like(columns, dtype=float),
            ],
            axis=-1,
        )
        rays = camera_rays @ self.map_R_camera.T
        with np.errstate(divide="ignore", invalid="ignore"):
            floor_depth = np.where(
                rays[..., 2] < 0, -self.camera_position[2] / rays[..., 2], np.inf
            )
            half = np.array([self.box_size.x, self.box_size.y, self.box_size.z]) / 2
            box_middle = np.array([*self.box_center, self.box_size.z / 2])
            low = (box_middle - half - self.camera_position) / rays
            high = (box_middle + half - self.camera_position) / rays
            entry = np.minimum(low, high).max(axis=-1)
            leave = np.maximum(low, high).min(axis=-1)
        box_depth = np.where((entry <= leave) & (entry > 0), entry, np.inf)
        depth = np.minimum(floor_depth, box_depth)
        color = np.full((self.height, self.width, 3), 128, dtype=np.uint8)
        color[box_depth < floor_depth] = (0, 0, 200)
        depth_in_millimeters = np.where(np.isfinite(depth), depth * 1000, 0)
        return RenderedPictures(
            color=color, depth_in_millimeters=depth_in_millimeters.astype(np.uint16)
        )

    def _create_camera_transform(self) -> TransformStamped:
        """
        :return: Where the camera's optical frame is in the map frame.
        """
        transform = TransformStamped()
        transform.header.frame_id = self.configuration.tf_to
        transform.child_frame_id = self.configuration.tf_from
        (
            transform.transform.translation.x,
            transform.transform.translation.y,
            transform.transform.translation.z,
        ) = map(float, self.camera_position)
        (
            transform.transform.rotation.x,
            transform.transform.rotation.y,
            transform.transform.rotation.z,
            transform.transform.rotation.w,
        ) = map(float, Rotation.from_matrix(self.map_R_camera).as_quat())
        return transform

    def _publish(
        self,
        *,
        color_publisher: Publisher,
        depth_publisher: Publisher,
        info_publisher: Publisher,
    ) -> None:
        """
        Publishes the pictures and the calibration, stamped with the time now.
        """
        header = Header(
            frame_id=self.configuration.tf_from,
            stamp=self.node.get_clock().now().to_msg(),
        )
        color_publisher.publish(
            CompressedImage(
                header=header, format=CompressedImageFormat.COLOR, data=self._color_data
            )
        )
        depth_publisher.publish(
            CompressedImage(
                header=header,
                format=CompressedImageFormat.DEPTH,
                data=self._depth_data,
            )
        )
        info_publisher.publish(
            CameraInfo(
                header=header,
                width=self.width,
                height=self.height,
                k=[
                    self.focal_length,
                    0.0,
                    self.width / 2,
                    0.0,
                    self.focal_length,
                    self.height / 2,
                    0.0,
                    0.0,
                    1.0,
                ],
                p=[
                    self.focal_length,
                    0.0,
                    self.width / 2,
                    0.0,
                    0.0,
                    self.focal_length,
                    self.height / 2,
                    0.0,
                    0.0,
                    0.0,
                    1.0,
                    0.0,
                ],
            )
        )


# %% finding what the camera shows


def create_map_world() -> World:
    """
    :return: A world holding only the map frame, which what the camera finds is given in.
    """
    world = World()
    with world.modify_world():
        world.add_kinematic_structure_entity(
            Body(name=PrefixedName(name=HsrRos2CameraConfig().tf_to))
        )
    return world


@pytest.fixture
def ros_session() -> Iterator[RobotDemonstrationRosSession]:
    """
    A node spun in the background, as a demonstration has one.
    """
    with RobotDemonstrationRosSession.start(RosNodeName.SESSION) as session:
        yield session


def test_robot_head_camera_finds_the_box_on_the_floor(
    ros_session: RobotDemonstrationRosSession,
) -> None:
    box_size = Scale(0.07, 0.07, 0.2)
    box_center = np.array([1.0, 0.05])
    with BoxOnFloorCamera(
        camera_position=np.array([0.0, 0.0, 1.1]),
        looked_at=np.array([1.0, 0.0, 0.0]),
        box_center=box_center,
        box_size=box_size,
    ):
        camera = RobotHeadCamera(ros_node=ros_session.node)
        found = camera.find_floor_objects(
            query=FloorObjectQuery(maximum_extent=0.3),
            region=FloorRegion(
                area=PlanarBoundingBox(
                    min_x=0.5,
                    min_y=-0.5,
                    max_x=1.5,
                    max_y=0.5,
                    origin=HomogeneousTransformationMatrix(
                        reference_frame=create_map_world().root
                    ),
                )
            ),
            nearest_to=box_center,
        )

    [box] = found
    assert np.allclose(box.xy, box_center, atol=0.02)
    assert box.size.z == pytest.approx(box_size.z, abs=0.015)


def test_robot_head_camera_reports_that_no_pictures_arrive(
    ros_session: RobotDemonstrationRosSession,
) -> None:
    camera = RobotHeadCamera(
        ros_node=ros_session.node, image_timeout=timedelta(seconds=1)
    )

    with pytest.raises(CameraImagesMissing):
        camera.synchronize_with_robot()


def test_robot_head_camera_finds_objects_in_the_frame_its_pose_is_looked_up_in(
    ros_session: RobotDemonstrationRosSession,
) -> None:
    camera = RobotHeadCamera(ros_node=ros_session.node)

    assert camera.world_frame == HsrRos2CameraConfig().tf_to
