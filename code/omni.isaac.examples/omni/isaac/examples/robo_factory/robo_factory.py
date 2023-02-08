# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#

from omni.isaac.examples.base_sample import BaseSample
from omni.isaac.franka.tasks import Stacking
from omni.isaac.franka.controllers import StackingController
import numpy as np


class RoboFactory(BaseSample):
    def __init__(self) -> None:
        super().__init__()
        self._tasks = []
        self._controllers = []
        self._articulation_controllers = []
        self._robots = []
        self._num_of_tasks = 4
        return

    def setup_scene(self):
        world = self.get_world()
        for i in range(self._num_of_tasks):
            task = Stacking(name="task" + str(i), offset=np.array([0, (i * 2) - 3, 0]))
            world.add_task(task)
        return

    async def setup_post_load(self):
        for i in range(self._num_of_tasks):
            self._tasks.append(self._world.get_task(name="task" + str(i)))
        for i in range(self._num_of_tasks):
            self._robots.append(self._world.scene.get_object(self._tasks[i].get_params()["robot_name"]["value"]))
            self._controllers.append(
                StackingController(
                    name="stacking_controller",
                    gripper=self._robots[i].gripper,
                    robot_articulation=self._robots[i],
                    picking_order_cube_names=self._tasks[i].get_cube_names(),
                    robot_observation_name=self._robots[i].name,
                )
            )
        for i in range(self._num_of_tasks):
            self._articulation_controllers.append(self._robots[i].get_articulation_controller())
        return

    def _on_start_factory_physics_step(self, step_size):
        observations = self._world.get_observations()
        for i in range(self._num_of_tasks):
            actions = self._controllers[i].forward(
                observations=observations, end_effector_offset=np.array([0, 0, -0.015])
            )
            self._articulation_controllers[i].apply_action(actions)
        return

    async def _on_start_stacking_event_async(self):
        world = self.get_world()
        world.add_physics_callback("sim_step", self._on_start_factory_physics_step)
        await world.play_async()
        return

    async def setup_pre_reset(self):
        world = self.get_world()
        if world.physics_callback_exists("sim_step"):
            world.remove_physics_callback("sim_step")
            for i in range(len(self._controllers)):
                self._controllers[i].reset()
        return

    def world_cleanup(self):
        self._tasks = []
        self._controllers = []
        self._articulation_controllers = []
        self._robots = []
        return
