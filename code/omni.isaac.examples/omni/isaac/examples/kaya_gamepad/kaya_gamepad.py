# Copyright (c) 2021-2022, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.


from omni.isaac.examples.base_sample import BaseSample
from omni.isaac.core.utils.nucleus import get_assets_root_path
import omni.usd
import carb
import omni.graph.core as og


class KayaGamepad(BaseSample):
    def __init__(self) -> None:
        super().__init__()

    def setup_scene(self):
        assets_root_path = get_assets_root_path()
        if assets_root_path is None:
            carb.log_error("Could not find Isaac Sim assets folder")
            return
        kaya_usd = assets_root_path + "/Isaac/Robots/Kaya/kaya.usd"
        kaya_ogn_usd = assets_root_path + "/Isaac/Robots/Kaya/kaya_ogn_gamepad.usd"
        stage = omni.usd.get_context().get_stage()
        graph_prim = stage.DefinePrim("/World", "Xform")
        graph_prim.GetReferences().AddReference(kaya_ogn_usd)

    def world_cleanup(self):
        pass
