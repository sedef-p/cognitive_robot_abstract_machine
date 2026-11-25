"""
The HSR searches the milk carton on the floor and the trash can with RoboKudo, through
its head camera, and throws the milk away. The robot only knows roughly where each one
is: the milk somewhere in :attr:`PerceivedFloorToTrashCanDemonstration.milk_region`, the
trash can somewhere beside the sofa. Only what RoboKudo finds is added to the robot's
world.

In simulation, the head camera is rendered from a second world as it really is. Run it
from this folder, with ROS 2 and the ROS workspace sourced::

    python throw_trash_perception_rviz_demo.py

The camera's points are published on :class:`CameraViewTopic`; RViz shows them with::

    rviz2 -d throw_trash_perception.rviz

On the real HSR, run ``throw_trash_perception_robot.py`` instead, which reads the real
head camera.
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path

import numpy as np
from coraplex.plans.context_extensions import RobotAccess
from coraplex.plans.executors import (
    SimulatedPlanExecutor,
)
from coraplex.querying.grasp_closeness import GraspsByCloseness
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
    MoveTorsoAction,
    ParkArmsAction,
)
from cramph.composites import Parallel, Sequence
from cramph.context import StatechartContext
from cramph.statechart import Statechart
from exceptions import NoObjectOnFloorFound
from floor_perception import (
    CameraViewTopic,
    FloorObjectDetection,
    FloorObjectQuery,
    FloorRegion,
    FurnitureFront,
    HeadCamera,
    LiveCameraView,
    RobotHeadCamera,
    Side,
    SimulatedHeadCamera,
    TakenObjectEraser,
)
from krrood.entity_query_language.factories import a
from krrood_spatial_math.spatial_types import (
    HomogeneousTransformationMatrix,
    Pose,
    Pose2D,
)
from rclpy.publisher import Publisher
from semantic_digital_twin.datastructures.definitions import TorsoState
from semantic_digital_twin.semantic_annotations.semantic_annotations import (
    Milk,
    Sofa,
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
from semantic_digital_twin.world_description.geometry import (
    CoordinateBounds,
    PlanarBoundingBox,
)
from semantic_digital_twin.world_description.world_entity import (
    SemanticAnnotation,
)

# the demo this one builds on lies in the folder above, which is no package
sys.path.append(str(Path(__file__).resolve().parents[1]))
from throw_trash_rviz_demo import FloorToTrashCanDemonstration, SceneBody

# %% what RoboKudo finds


class PerceivedSceneBody(StrEnum):
    """
    The names of the bodies added to the robot's world once RoboKudo found them, besides
    those of :class:`SceneBody`.
    """

    TRASH_CAN = "trash_can"


# %% the demonstration


@dataclass(kw_only=True)
class PerceivedFloorToTrashCanDemonstration(FloorToTrashCanDemonstration):
    """
    The HSR searches the milk and the trash can with RoboKudo and throws the milk away.

    The robot's world starts without both; each is added when found. Step by step::

        park the arm
        look at the spots of milk_bounds one after the other until the milk is found,
            look straight at it, measure it again and add it as a box of that size
        drive past passage_spot to trash_can_lookout_distance before the region beside
            the sofa and raise the torso
        look at the spots of that region until the trash can is found, look straight
            at it, measure it again and add it as a trash can of that size
        lower the torso, drive back past passage_spot, take the milk from the side
            facing the passage and park the arm
        carry it past passage_spot to approach_distance and drop_distance before the
            trash can, let go of it above the opening, park the arm and lower the torso
    """

    milk_bounds: CoordinateBounds = field(
        default_factory=lambda: CoordinateBounds(
            lower=np.array([0.2, 3.8]), upper=np.array([2.1, 5.1])
        )
    )
    """
    The corners of the rectangle on the floor the milk is known to stand in, as x and y
    in the world frame: in the bedroom, around where the robot starts.
    """

    trash_can_side: Side = Side.LEFT
    """
    The side of the sofa the trash can is known to stand at.
    """

    sofa_front: FurnitureFront = FurnitureFront.NEGATIVE_X
    """
    The axis of the sofa's own frame its front faces along: -x, since
    :class:`~semantic_digital_twin.predefined_maps.kitchen_environment.KitchenSofa` puts
    the back cushions at its +x end.
    """

    trash_can_reach: float = 1.0
    """
    How far beyond the sofa's side the trash can may stand, in meters.
    """

    trash_can_lookout_distance: float = 1.2
    """
    How far from the middle of the trash can's region the robot stands to look for it,
    on the side it comes from, in meters.
    """

    minimum_look_distance: float = 0.9
    """
    How far from the robot's base a spot on the floor the robot looks at must be, in
    meters, for the head camera to see the floor around it rather than the robot.
    """

    look_spacing: float = 0.6
    """
    How far apart the spots of a search region the robot looks at are, in meters.
    """

    robot_footprint_radius: float = 0.3
    """
    How far from the middle of the robot's base what the camera finds is the robot
    itself, in meters.
    """

    milk_query: FloorObjectQuery = field(
        default_factory=lambda: FloorObjectQuery(maximum_extent=0.3)
    )
    """
    How the milk is looked for: anything smaller than furniture.
    """

    trash_can_query: FloorObjectQuery = field(
        default_factory=lambda: FloorObjectQuery(
            minimum_extent=0.3, maximum_extent=0.6, crop_height=0.6
        )
    )
    """
    How the trash can is looked for: bigger than what is picked up, smaller than a sofa.
    """

    trash_can_wall_thickness: float = 0.02
    """
    How thick the walls and the bottom of the trash can added to the robot's world are,
    in meters.
    """

    camera_view_period: float | None = 1.0
    """
    How long to wait between two pictures of the simulated head camera published to
    RViz, in seconds; ``None`` publishes none.
    """

    camera_view_display_duration: float = 4.0
    """
    How long the robot waits after finding something, in seconds, so what the camera
    found can be watched before it moves on; it does not wait without
    :attr:`real_time_factor`.
    """

    camera: HeadCamera | None = field(init=False, default=None)
    """
    The head camera RoboKudo looks through; set up when the demonstration runs.
    """

    live_camera_view: LiveCameraView | None = field(init=False, default=None)
    """
    Publishes what the simulated head camera sees while the demonstration runs.
    """

    found_points_publisher: Publisher | None = field(init=False, default=None)
    """
    Publishes the points of what was found, each time something was looked for.
    """

    carrying_start: np.ndarray | None = field(init=False, default=None)
    """
    Where on the floor the robot set off from with the milk towards the trash can, as x
    and y in the world frame, once it did.
    """

    # %% running the plans

    def run(self) -> World:
        """
        Unlike :meth:`FloorToTrashCanDemonstration.run`, takes the milk and the trash
        can out of the robot's world, searches both, and then picks the milk up and
        throws it away in one plan.

        :raises NoObjectOnFloorFound: If the camera sees no milk or no trash can.
        """
        try:
            world = self.acquire_world()
            self._forget(world=world, annotation_type=Milk)
            self._forget(world=world, annotation_type=TrashCan)
            self.camera = self._create_camera(world=world)
            self._start_camera_view(world=world)
            self.start_simulation(world)
            self.perform_plan(world)
            self._add_milk_found(world=world)
            trash_can_region = self.create_trash_can_region(world=world)
            self.execute_statechart(
                world,
                lambda context: self.build_drive_to_trash_can_lookout(
                    context=context, trash_can_region=trash_can_region
                ),
            )
            self._add_trash_can_found(world=world, region=trash_can_region)
            self.carrying_start = self._compute_robot_floor_pose(
                world=world
            ).position.to_np()
            self.execute_statechart(
                world, lambda context: self.build_throw_away(context=context)
            )
        finally:
            if self.live_camera_view is not None:
                self.live_camera_view.stop()
            self.tear_down()
        return world

    # %% the plans

    def build_statechart(self, context: StatechartContext) -> Statechart:
        """
        Unlike :meth:`FloorToTrashCanDemonstration.build_statechart`, only parks the
        arm; :meth:`run` does the rest.
        """
        robot = context.require_extension(extension_type=RobotAccess).robot
        self._let_robot_touch_the_floor(robot)
        statechart = Statechart(context=context)
        statechart.add_node(node=ParkArmsAction(arms=robot.all_arms))
        return statechart

    def build_drive_to_trash_can_lookout(
        self, *, context: StatechartContext, trash_can_region: FloorRegion
    ) -> Statechart:
        """
        :param context: The context of the plan.
        :param trash_can_region: Where the trash can is known to stand.
        :return: Driving past :attr:`passage_spot` to :attr:`trash_can_lookout_distance`
            before `trash_can_region` and raising the torso, so the camera sees into the
            trash can.
        """
        passage_xy = self.passage_spot.position.to_np()
        region_pose = Pose.from_xyz_rpy(
            x=float(trash_can_region.center_xy[0]),
            y=float(trash_can_region.center_xy[1]),
            reference_frame=context.world.root,
        )
        lookout_pose = self._compute_standing_pose_facing(
            target=region_pose,
            distance=self.trash_can_lookout_distance,
            approach_xy=passage_xy,
        )
        passage_pose = self._compute_floor_pose_facing(passage_xy, lookout_pose)
        statechart = Statechart(context=context)
        statechart.add_node(
            node=Sequence(
                nodes=[
                    FaceAtAction(look_target=passage_pose),
                    NavigateAction(planar_goal_pose=Pose2D.from_pose(passage_pose)),
                    NavigateAction(planar_goal_pose=Pose2D.from_pose(lookout_pose)),
                    MoveTorsoAction(torso_state=TorsoState.HIGH),
                ]
            )
        )
        return statechart

    def build_throw_away(self, *, context: StatechartContext) -> Statechart:
        """
        :param context: The context of the plan, whose world holds the milk and the trash
            can found.
        :return: Lowering the torso, driving back past :attr:`passage_spot`, picking the
            milk up from the side facing the passage and parking the arm, turning
            towards the trash can, carrying the milk past :attr:`passage_spot` to
            :attr:`approach_distance` and then :attr:`drop_distance` before the trash
            can, on the side of :attr:`carrying_start`, letting go of it above the
            opening, parking the arm and lowering the torso.
        """
        world = context.world
        robot = context.require_extension(extension_type=RobotAccess).robot
        [arm] = robot.all_arms
        [milk] = world.get_semantic_annotations_by_type(Milk)
        [trash_can] = world.get_semantic_annotations_by_type(TrashCan)
        passage_xy = self.passage_spot.position.to_np()
        milk_pose = Pose(reference_frame=milk.root)
        passage_pose_facing_milk = self._compute_floor_pose_facing(
            passage_xy, milk.root.global_pose
        )
        [grasp] = GraspsByCloseness(
            graspable=milk,
            standing_position=Pose2D.from_pose(passage_pose_facing_milk),
            number_of_grasps=1,
        )()
        drop_pose = self._compute_drop_pose(trash_can=trash_can.root, grasp=grasp)
        approach_pose = self._compute_standing_pose_before(
            drop_pose, self.approach_distance
        )
        passage_pose_facing_approach = self._compute_floor_pose_facing(
            passage_xy, approach_pose
        )
        statechart = Statechart(context=context)
        statechart.add_node(
            node=Sequence(
                nodes=[
                    Parallel(
                        nodes=[
                            ParkArmsAction(arms=robot.all_arms),
                            MoveTorsoAction(torso_state=TorsoState.LOW),
                        ]
                    ),
                    FaceAtAction(look_target=passage_pose_facing_milk),
                    NavigateAction(
                        planar_goal_pose=Pose2D.from_pose(passage_pose_facing_milk)
                    ),
                    a(MoveAndPickUpAction)(
                        navigate=a(NavigateAction)(
                            planar_goal_pose=self._leave_standing_pose_open(
                                milk_pose, arm, context
                            )
                        ),
                        face_and_look_at=self._face_and_look_at(milk_pose),
                        pick_up=a(PickUpAction)(grasp_candidate=grasp, arm=arm),
                    ),
                    ParkArmsAction(arms=robot.all_arms),
                    FaceAtAction(look_target=approach_pose),
                    CarryAction(
                        navigate=NavigateAction(
                            planar_goal_pose=Pose2D.from_pose(
                                passage_pose_facing_approach
                            )
                        ),
                        holding_arms=[arm],
                    ),
                    CarryAction(
                        navigate=NavigateAction(
                            planar_goal_pose=Pose2D.from_pose(approach_pose)
                        ),
                        holding_arms=[arm],
                    ),
                    CarryAction(
                        navigate=NavigateAction(
                            planar_goal_pose=Pose2D.from_pose(
                                self._compute_standing_pose_before(
                                    drop_pose, self.drop_distance
                                )
                            )
                        ),
                        holding_arms=[arm],
                    ),
                    FaceAtAction(look_target=drop_pose),
                    a(PlaceAction)(graspable=milk, goal_pose=drop_pose, arm=arm),
                    Parallel(
                        nodes=[
                            ParkArmsAction(arms=robot.all_arms),
                            MoveTorsoAction(torso_state=TorsoState.LOW),
                        ]
                    ),
                ]
            )
        )
        return statechart

    @staticmethod
    def _build_face_and_look_at(
        *, context: StatechartContext, target: Pose
    ) -> Statechart:
        """
        :param context: The context of the plan.
        :param target: What the robot turns to and looks at.
        :return: Turning the base towards `target` and pointing the head at it.
        """
        statechart = Statechart(context=context)
        statechart.add_node(
            node=FaceAndLookAtAction(
                face_at=FaceAtAction(look_target=target),
                look_at=LookAtAction(look_target=target),
            )
        )
        return statechart

    @staticmethod
    def _build_look_at(*, context: StatechartContext, target: Pose) -> Statechart:
        """
        :param context: The context of the plan.
        :param target: What the robot looks at.
        :return: Pointing the head at `target`.
        """
        statechart = Statechart(context=context)
        statechart.add_node(node=LookAtAction(look_target=target))
        return statechart

    # %% where to stand and where to let go

    @classmethod
    def _compute_standing_pose_facing(
        cls, *, target: Pose, distance: float, approach_xy: np.ndarray
    ) -> Pose:
        """
        :param target: What the robot is to stand before.
        :param distance: How far from `target` the base stands, in meters.
        :param approach_xy: A spot on the floor on the side of `target` the robot comes
            from, as x and y in the frame of `target`.
        :return: A pose on the floor `distance` from `target`, on the line towards
            `approach_xy`, facing `target`.
        """
        target_xy = Pose2D.from_pose(target).position.to_np()
        towards_side = np.asarray(approach_xy) - target_xy
        standing_xy = target_xy + distance * towards_side / np.linalg.norm(towards_side)
        return cls._compute_floor_pose_facing(standing_xy, target)

    def _compute_carrying_direction(self, drop_xy: np.ndarray) -> np.ndarray:
        """
        Unlike :meth:`FloorToTrashCanDemonstration._compute_carrying_direction`, points
        from :attr:`carrying_start`, since where the milk stood is only known once it
        was found.
        """
        towards_drop = drop_xy - self.carrying_start
        return towards_drop / np.linalg.norm(towards_drop)

    # %% what the robot knows of the scene

    def create_milk_region(self, *, world: World) -> FloorRegion:
        """
        :param world: The world whose root frame :attr:`milk_bounds` is given in.
        :return: Where the milk is known to stand.
        """
        return FloorRegion(
            area=PlanarBoundingBox.from_array_bounds(
                self.milk_bounds.lower,
                self.milk_bounds.upper,
                HomogeneousTransformationMatrix(reference_frame=world.root),
            )
        )

    def create_trash_can_region(self, *, world: World) -> FloorRegion:
        """
        :param world: A world holding the sofa.
        :return: Where the trash can is known to stand: at :attr:`trash_can_side` of the
            sofa, up to :attr:`trash_can_reach` beyond it.
        """
        [sofa] = world.get_semantic_annotations_by_type(Sofa)
        return FloorRegion.from_furniture_side(
            furniture=sofa.root,
            side=self.trash_can_side,
            front=self.sofa_front,
            reach=self.trash_can_reach,
        )

    @staticmethod
    def _forget(*, world: World, annotation_type: type[SemanticAnnotation]) -> None:
        """
        Takes every semantic annotation of `annotation_type` and its bodies out of
        `world`, so the robot has to find it.

        :param world: The robot's world, which may also hold what an earlier run found.
        :param annotation_type: What the robot is to find.
        """
        for annotation in world.get_semantic_annotations_by_type(annotation_type):
            with world.modify_world():
                world.remove_semantic_annotation(annotation)
                world.remove_branch_from_world(annotation.root)
                world.delete_orphaned_dofs()

    # %% the head camera

    def _create_camera(self, *, world: World) -> HeadCamera:
        """
        :param world: The robot's world.
        :return: The simulated head camera, looking into a world as it really is, in a
            simulated run, and the robot's real head camera otherwise.
        """
        if issubclass(self.executor_type, SimulatedPlanExecutor):
            return SimulatedHeadCamera(
                ground_truth=self.build_simulated_world(), believed_world=world
            )
        return RobotHeadCamera(ros_node=self.ros_node)

    def _start_camera_view(self, *, world: World) -> None:
        """
        Publishes the points of what is found, and, for a simulated camera, what it sees
        every :attr:`camera_view_period`. Does nothing without a ROS node.

        :param world: The robot's world, whose root frame the points are published in.
        """
        if self.ros_node is None:
            return
        publishers = {
            topic: topic.create_publisher(node=self.ros_node)
            for topic in CameraViewTopic
        }
        self.found_points_publisher = publishers[CameraViewTopic.OBJECTS]
        if self.camera_view_period is None or not isinstance(
            self.camera, SimulatedHeadCamera
        ):
            return
        self.live_camera_view = LiveCameraView(
            _world=world,
            camera=self.camera,
            publishers=publishers,
            ros_node=self.ros_node,
            period=self.camera_view_period,
        )

    def _publish_found_points(self, *, world: World) -> None:
        """
        Publishes :attr:`HeadCamera.found_points`, if there is a publisher.

        :param world: The robot's world, whose root frame the points are published in.
        """
        if self.found_points_publisher is None:
            return
        self.found_points_publisher.publish(
            self.camera.found_points.to_message(
                frame_id=str(world.root.name),
                stamp=self.ros_node.get_clock().now().to_msg(),
            )
        )

    # %% searching

    def _search(
        self, *, world: World, query: FloorObjectQuery, region: FloorRegion
    ) -> FloorObjectDetection:
        """
        Turns to and looks at the spots of `region` one after the other until an object
        `query` describes is found, then looks straight at it and measures it again, and
        waits :attr:`camera_view_display_duration` unless :attr:`real_time_factor` is
        ``None``, so what was found can be watched.

        :param world: The robot's world.
        :param query: What is looked for.
        :param region: Where it is known to stand.
        :return: The object found nearest to the middle of `region`, as measured the
            second time.
        :raises NoObjectOnFloorFound: If nothing is found from any of the spots.
        """
        glance = self._glance_over(world=world, query=query, region=region)
        glance_pose = glance.create_center_transform(world_root=world.root).pose
        self.execute_statechart(
            world,
            lambda context: self._build_look_at(context=context, target=glance_pose),
        )
        measured = self._find_beside_robot(
            world=world, query=query, region=region, nearest_to=glance.xy
        )
        if self.real_time_factor is not None:
            time.sleep(self.camera_view_display_duration)
        if not measured:
            return glance
        return measured[0]

    def _glance_over(
        self, *, world: World, query: FloorObjectQuery, region: FloorRegion
    ) -> FloorObjectDetection:
        """
        :return: The object `query` describes found first, looking at the spots of
            `region` one after the other.
        :raises NoObjectOnFloorFound: If nothing is found from any of the spots.
        """
        look_targets = region.compute_look_targets(
            viewer=self._compute_robot_floor_pose(world=world),
            minimum_distance=self.minimum_look_distance,
            spacing=self.look_spacing,
        )
        for look_target in look_targets:
            look_pose = Pose.from_xyz_rpy(
                x=float(look_target[0]),
                y=float(look_target[1]),
                reference_frame=world.root,
            )
            self.execute_statechart(
                world,
                lambda context: self._build_face_and_look_at(
                    context=context, target=look_pose
                ),
            )
            found = self._find_beside_robot(
                world=world, query=query, region=region, nearest_to=region.center_xy
            )
            if found:
                return found[0]
        raise NoObjectOnFloorFound(region_description=region.describe())

    def _find_beside_robot(
        self,
        *,
        world: World,
        query: FloorObjectQuery,
        region: FloorRegion,
        nearest_to: np.ndarray,
    ) -> list[FloorObjectDetection]:
        """
        Does what :meth:`HeadCamera.find_floor_objects` does, leaves out what stands
        within :attr:`robot_footprint_radius` of the robot, which is the robot itself,
        publishes the points found and logs how many objects were found.

        :param world: The robot's world.
        :return: The objects found in `region` beside the robot, the one nearest to
            `nearest_to` first.
        """
        robot_xy = self._compute_robot_floor_pose(world=world).position.to_np()
        found = [
            detection
            for detection in self.camera.find_floor_objects(
                query=query, region=region, nearest_to=nearest_to
            )
            if np.linalg.norm(detection.xy - robot_xy) > self.robot_footprint_radius
        ]
        self._publish_found_points(world=world)
        if self.ros_node is not None:
            self.ros_node.get_logger().info(
                f"standing at ({robot_xy[0]:.2f}, {robot_xy[1]:.2f}), found "
                f"{len(found)} objects in {region.describe()}: "
                + ", ".join(detection.describe() for detection in found)
            )
        return found

    def _compute_robot_floor_pose(self, *, world: World) -> Pose2D:
        """
        :param world: The robot's world.
        :return: Where the robot's base stands and which way it faces, in the world frame.
        """
        [robot] = world.get_semantic_annotations_by_type(self.used_robot)
        return Pose2D.from_pose(robot.root.global_pose)

    def _add_milk_found(self, *, world: World) -> None:
        """
        Searches the milk in :attr:`milk_bounds` and adds it to `world` as a box of the
        estimated size. From then on, the wrist the hand is mounted on may touch it, a
        simulated real milk moves along with the box, and its points are no longer shown
        as found once the robot takes it away.

        :param world: The robot's world.
        :raises NoObjectOnFloorFound: If nothing is found.
        """
        found = self._search(
            world=world,
            query=self.milk_query,
            region=self.create_milk_region(world=world),
        )
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
        milk.root.parent_connection.origin = found.create_center_transform(
            world_root=world.root
        )
        [robot] = world.get_semantic_annotations_by_type(self.used_robot)
        [arm] = robot.all_arms
        self._let_wrist_touch_the_object(arm, milk.root)
        if isinstance(self.camera, SimulatedHeadCamera):
            [true_milk] = self.camera.ground_truth.get_semantic_annotations_by_type(
                Milk
            )
            self.camera.follow(true_body=true_milk.root, believed_body=milk.root)
        TakenObjectEraser(
            _world=world, camera=self.camera, detection=found, body=milk.root
        )

    def _add_trash_can_found(self, *, world: World, region: FloorRegion) -> None:
        """
        Searches the trash can in `region` and adds it to `world` as a trash can of the
        estimated size, with walls :attr:`trash_can_wall_thickness` thick.

        :param world: The robot's world.
        :param region: Where the trash can is known to stand.
        :raises NoObjectOnFloorFound: If nothing is found.
        """
        found = self._search(world=world, query=self.trash_can_query, region=region)
        with world.modify_world():
            TrashCan.get_annotation_specification(
                PerceivedSceneBody.TRASH_CAN,
                TrashCan.get_default_root_kinematic_structure_entity_specification(
                    scale=found.size, wall_thickness=self.trash_can_wall_thickness
                ),
            ).spawn(
                world,
                parent_T_self=found.create_center_transform(world_root=world.root),
            )


def main() -> World:
    """
    Run the demonstration in a kinematic simulation.

    :return: The robot's world after the milk was thrown away.
    """
    return PerceivedFloorToTrashCanDemonstration().run()


if __name__ == "__main__":
    main()
