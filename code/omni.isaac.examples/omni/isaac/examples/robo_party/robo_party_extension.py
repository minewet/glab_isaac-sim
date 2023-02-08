# Copyright (c) 2020-2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import os
from omni.isaac.examples.base_sample import BaseSampleExtension
from omni.isaac.examples.robo_party import RoboParty
import asyncio
import omni.ui as ui
from omni.isaac.ui.ui_utils import btn_builder


class RoboPartyExtension(BaseSampleExtension):
    def on_startup(self, ext_id: str):
        super().on_startup(ext_id)
        super().start_extension(
            menu_name="Multi-Robot",
            submenu_name="",
            name="RoboParty",
            title="RoboParty",
            doc_link="https://docs.omniverse.nvidia.com/app_isaacsim/app_isaacsim/overview.html",
            overview="This Example shows how to run multiple tasks in the same scene.\n\nPress the 'Open in IDE' button to view the source code.",
            sample=RoboParty(),
            file_path=os.path.abspath(__file__),
            number_of_extra_frames=1,
        )
        self.task_ui_elements = {}
        frame = self.get_frame(index=0)
        self.build_task_controls_ui(frame)
        return

    def _on_start_party_button_event(self):
        asyncio.ensure_future(self.sample._on_start_party_event_async())
        self.task_ui_elements["Start Party"].enabled = False
        return

    def post_reset_button_event(self):
        self.task_ui_elements["Start Party"].enabled = True
        return

    def post_load_button_event(self):
        self.task_ui_elements["Start Party"].enabled = True
        return

    def post_clear_button_event(self):
        self.task_ui_elements["Start Party"].enabled = False
        return

    def build_task_controls_ui(self, frame):
        with frame:
            with ui.VStack(spacing=5):
                # Update the Frame Title
                frame.title = "Task Controls"
                frame.visible = True
                dict = {
                    "label": "Start Party",
                    "type": "button",
                    "text": "Start Party",
                    "tooltip": "Start Party",
                    "on_clicked_fn": self._on_start_party_button_event,
                }

                self.task_ui_elements["Start Party"] = btn_builder(**dict)
                self.task_ui_elements["Start Party"].enabled = False
