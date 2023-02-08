# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#

from omni.isaac.examples.base_sample import BaseSample
from omni.isaac.franka.tasks import FollowTarget as FollowTargetTask
from omni.isaac.franka.controllers import RMPFlowController
from .tasks.follow_target import FollowTargett
from omni.isaac.core.utils.prims import create_prim, delete_prim
from .ik_solver import KinematicsSolver
import carb
import numpy as np

class FollowTarget(BaseSample):
    def __init__(self) -> None:
        super().__init__()
        self._controller = None
        self._articulation_controller = None

    def setup_scene(self):
        world = self.get_world()
        
        world.scene.clear()
        create_prim(
            prim_path="/sk",
            #usd_path=self._assets_root_path + "/Isaac/Environments/Grid/gridroom_curved.usd",]
            usd_path = "C:/Users/kimhj/Downloads/sk/demo.usd",
            position=np.array([2, 1, 0]),
            #scale=np.array([20,20,20]),
        )
        world.add_task(FollowTargett(target_position=np.array([0.48232911772461917, 0.39553378102036146, 1.3290173692434306])))            
        return

    async def setup_pre_reset(self):
        world = self.get_world()
        if world.physics_callback_exists("sim_step"):
            world.remove_physics_callback("sim_step")
       # self._controller.reset()
        return

    def world_cleanup(self):
        #self._controller = None
        return

    async def setup_post_load(self):
        my_world = self.get_world()
        my_world.reset()
        await self._world.play_async()
        task_params = my_world.get_task("denso_follow_target").get_params()

        self.target_name = task_params["target_name"]["value"]
        denso_name = task_params["robot_name"]["value"]
        my_denso = my_world.scene.get_object(denso_name)
        #initialize the controller
        self.my_controller = KinematicsSolver(my_denso)
        self.articulation_controller = my_denso.get_articulation_controller()
        return
        

    async def _on_follow_target_event_async(self, val):
        world = self.get_world()
        if val: #Start
            await world.play_async()
            world.add_physics_callback("sim_step", self._on_follow_target_simulation_step)
        else: #Stop
            world.remove_physics_callback("sim_step")
        return

    def _on_follow_target_simulation_step(self, step_size):
        observations = self._world.get_observations()
        actions, succ = self.my_controller.compute_inverse_kinematics(
            target_position=observations[self.target_name]["position"],
            target_orientation=observations[self.target_name]["orientation"],
        )
        if succ:
            self.articulation_controller.apply_action(actions)

        return

    def _on_add_obstacle_event(self):
        world = self.get_world()
        current_task = list(world.get_current_tasks().values())[0]
        cube = current_task.add_obstacle()
        #self._controller.add_obstacle(cube)
        return

    def _on_remove_obstacle_event(self):
        world = self.get_world()
        current_task = list(world.get_current_tasks().values())[0]
        obstacle_to_delete = current_task.get_obstacle_to_delete()
        #self._controller.remove_obstacle(obstacle_to_delete)
        current_task.remove_obstacle()
        return

    def _on_logging_event(self, val):
        world = self.get_world()
        data_logger = world.get_data_logger()
        if not world.get_data_logger().is_started():
            robot_name = self._task_params["robot_name"]["value"]
            target_name = self._task_params["target_name"]["value"]

            def frame_logging_func(tasks, scene):
                return {
                    "joint_positions": scene.get_object(robot_name).get_joint_positions().tolist(),
                    "applied_joint_positions": scene.get_object(robot_name)
                    .get_applied_action()
                    .joint_positions.tolist(),
                    "target_position": scene.get_object(target_name).get_world_pose()[0].tolist(),
                }

            data_logger.add_data_frame_logging_func(frame_logging_func)
        if val:
            data_logger.start()
        else:
            data_logger.pause()
        return

    def _on_save_data_event(self, log_path):
        world = self.get_world()
        data_logger = world.get_data_logger()
        data_logger.save(log_path=log_path)
        data_logger.reset()
        return
