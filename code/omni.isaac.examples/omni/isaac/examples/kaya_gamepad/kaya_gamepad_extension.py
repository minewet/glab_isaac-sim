# Copyright (c) 2020-2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import os
from omni.isaac.examples.base_sample import BaseSampleExtension
from omni.isaac.examples.kaya_gamepad import KayaGamepad


class KayaGamepadExtension(BaseSampleExtension):
    def on_startup(self, ext_id: str):
        super().on_startup(ext_id)
        overview = "This Example shows how to drive a NVIDIA Kaya robot using a Gamepad in Isaac Sim."
        overview += "\n\nConnect a gamepad to the robot, and the press PLAY to begin simulating."
        overview += "\n\nPress the 'Open in IDE' button to view the source code."

        super().start_extension(
            menu_name="Input Devices",
            submenu_name="",
            name="Kaya Gamepad",
            title="NVIDIA Kaya Gamepad Example",
            doc_link="https://docs.omniverse.nvidia.com/app_isaacsim/app_isaacsim/tutorial_advanced_gamepad_control.html",
            overview=overview,
            file_path=os.path.abspath(__file__),
            sample=KayaGamepad(),
        )
        return
