from omni.isaac.manipulators import SingleManipulator
from omni.isaac.manipulators.grippers import ParallelGripper
from omni.isaac.core.utils.stage import add_reference_to_stage
import omni.isaac.core.tasks as tasks
from typing import Optional
import numpy as np


# Inheriting from the base class Follow Target
class FollowTarget(tasks.FollowTarget):
    def __init__(
        self,
        name: str = "denso_follow_target",
        target_prim_path: Optional[str] = None,
        target_name: Optional[str] = None,
        target_position: Optional[np.ndarray] = None,
        target_orientation: Optional[np.ndarray] = None,
        offset: Optional[np.ndarray] = None,
    ) -> None:
        tasks.FollowTarget.__init__(
            self,
            name=name,
            target_prim_path=target_prim_path,
            target_name=target_name,
            target_position=target_position,
            target_orientation=target_orientation,
            offset=offset,
        )
        return

    def set_robot(self) -> SingleManipulator:
        #TODO: change this to the robot usd file.
        asset_path = "C:/Users/kimhj/Downloads/sk/ridgeback_iiwa/ridgeback_iiwa.usd"
        add_reference_to_stage(usd_path=asset_path, prim_path="/ridgeback")
        gripper = ParallelGripper(
            end_effector_prim_path="/ridgeback/iiwa_link_6",
            joint_prim_names=["iiwa_joint_7", "iiwa_joint_7"],
            joint_opened_positions=np.array([0, 0]),
            joint_closed_positions=np.array([0, 0]),
            action_deltas=np.array([0, 0]))
        manipulator = SingleManipulator(prim_path="/ridgeback",
                                        name="cobotta_robot",
                                        end_effector_prim_name="iiwa_link_7",
                                        gripper=gripper,
                                        position = [0,0,0]
                                        )
        joints_default_positions = np.zeros(12)
        manipulator.set_joints_default_state(positions=joints_default_positions)
        return manipulator