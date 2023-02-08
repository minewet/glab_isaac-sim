from omni.isaac.motion_generation import ArticulationKinematicsSolver, LulaKinematicsSolver
from omni.isaac.core.articulations import Articulation
from typing import Optional


class KinematicsSolver(ArticulationKinematicsSolver):
    def __init__(self, robot_articulation: Articulation, end_effector_frame_name: Optional[str] = None) -> None:
        #TODO: change the config path
        self._kinematics = LulaKinematicsSolver(robot_description_path="C:/Users/kimhj/AppData/Local/ov/pkg/isaac_sim-2022.1.1/standalone_examples/api/omni.isaac.franka/rmpflow/robot_descriptor.yaml",
                                                urdf_path="C:/Users/kimhj/Downloads/glab_robots-master/glab_robots-master/ridgeback_iiwa_description/urdf/ridgeback_iiwa.urdf")
        if end_effector_frame_name is None:
            end_effector_frame_name = "iiwa_link_6"
        ArticulationKinematicsSolver.__init__(self, robot_articulation, self._kinematics, end_effector_frame_name)
        return