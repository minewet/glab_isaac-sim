# Copyright (c) 2020-2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

from omni.isaac.core.controllers import BaseController
from omni.isaac.motion_generation import PathPlannerVisualizer, PathPlanner
from omni.isaac.motion_generation.lula import RRT
from omni.isaac.core.utils.types import ArticulationAction
from omni.isaac.core.articulations import Articulation
from omni.isaac.motion_generation import interface_config_loader
import omni.isaac.core.objects

import carb
from typing import Optional
import numpy as np


class PathPlannerController(BaseController):
    def __init__(
        self,
        name: str,
        path_planner_visualizer: PathPlannerVisualizer,
        cspace_interpolation_max_dist: float = 0.5,
        frames_per_waypoint: int = 30,
    ):
        BaseController.__init__(self, name)

        self._path_planner_visualizer = path_planner_visualizer
        self._path_planner = path_planner_visualizer.get_path_planner()

        self._cspace_interpolation_max_dist = cspace_interpolation_max_dist
        self._frames_per_waypoint = frames_per_waypoint

        self._plan = None

        self._frame_counter = 1

    def _make_new_plan(
        self, target_end_effector_position: np.ndarray, target_end_effector_orientation: Optional[np.ndarray] = None
    ) -> None:
        self._path_planner.set_end_effector_target(target_end_effector_position, target_end_effector_orientation)
        self._path_planner.update_world()
        self._plan = self._path_planner_visualizer.compute_plan_as_articulation_actions(
            max_cspace_dist=self._cspace_interpolation_max_dist
        )
        if self._plan is None or self._plan == []:
            carb.log_warn("No plan could be generated to target pose: " + str(target_end_effector_position))

    def forward(
        self, target_end_effector_position: np.ndarray, target_end_effector_orientation: Optional[np.ndarray] = None
    ) -> ArticulationAction:
        if self._plan is None:
            # This will only happen the first time the forward function is used
            self._make_new_plan(target_end_effector_position, target_end_effector_orientation)

        if len(self._plan) == 0:
            # The plan is completed; return null action to remain in place
            self._frame_counter = 1
            return ArticulationAction()

        if self._frame_counter % self._frames_per_waypoint != 0:
            # Stop at each waypoint in the plan for self._frames_per_waypoint frames
            self._frame_counter += 1
            return self._plan[0]
        else:
            self._frame_counter += 1
            return self._plan.pop(0)

    def add_obstacle(self, obstacle: omni.isaac.core.objects, static: bool = False) -> None:
        self._path_planner.add_obstacle(obstacle, static)

    def remove_obstacle(self, obstacle: omni.isaac.core.objects) -> None:
        self._path_planner.remove_obstacle(obstacle)

    def reset(self) -> None:
        # PathPlannerController will make one plan per reset
        self._path_planner.reset()
        self._plan = None
        self._frame_counter = 1

    def get_path_planner_visualizer(self) -> PathPlannerVisualizer:
        return self._path_planner_visualizer

    def get_path_planner(self) -> PathPlanner:
        return self._path_planner


class FrankaRrtController(PathPlannerController):
    def __init__(
        self,
        name,
        robot_articulation: Articulation,
        cspace_interpolation_max_dist: float = 0.5,
        frames_per_waypoint: int = 30,
    ):
        rrt_config = interface_config_loader.load_supported_path_planner_config("Franka", "RRT")
        rrt = RRT(**rrt_config)

        visualizer = PathPlannerVisualizer(robot_articulation, rrt)

        PathPlannerController.__init__(self, name, visualizer, cspace_interpolation_max_dist, frames_per_waypoint)
