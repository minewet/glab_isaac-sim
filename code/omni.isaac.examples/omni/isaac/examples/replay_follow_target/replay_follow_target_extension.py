# Copyright (c) 2020-2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import os
from omni.isaac.examples.base_sample import BaseSampleExtension
from omni.isaac.examples.replay_follow_target import ReplayFollowTarget
import asyncio
import omni.ui as ui
from omni.isaac.ui.ui_utils import btn_builder, str_builder


class ReplayFollowTargetExtension(BaseSampleExtension):
    def on_startup(self, ext_id: str):
        super().on_startup(ext_id)
        super().start_extension(
            menu_name="Manipulation",
            submenu_name="",
            name="Replay Follow Target",
            title="Replay Follow Target Task",
            doc_link="https://docs.omniverse.nvidia.com/app_isaacsim/app_isaacsim/overview.html",
            overview="This Example shows how to use data logging to replay data collected\n\n from the follow target extension example.\n\n Press the 'Open in IDE' button to view the source code.",
            sample=ReplayFollowTarget(),
            file_path=os.path.abspath(__file__),
            number_of_extra_frames=2,
            window_width=700,
        )
        self.task_ui_elements = {}
        frame = self.get_frame(index=0)
        self.build_data_logging_ui(frame)
        return

    def _on_replay_trajectory_button_event(self):
        asyncio.ensure_future(
            self.sample._on_replay_trajectory_event_async(self.task_ui_elements["Data File"].get_value_as_string())
        )
        self.task_ui_elements["Replay Trajectory"].enabled = False
        self.task_ui_elements["Replay Scene"].enabled = False
        return

    def _on_replay_scene_button_event(self):
        asyncio.ensure_future(
            self.sample._on_replay_scene_event_async(self.task_ui_elements["Data File"].get_value_as_string())
        )
        self.task_ui_elements["Replay Trajectory"].enabled = False
        self.task_ui_elements["Replay Scene"].enabled = False
        return

    def post_reset_button_event(self):
        self.task_ui_elements["Replay Trajectory"].enabled = True
        self.task_ui_elements["Replay Scene"].enabled = True
        return

    def post_load_button_event(self):
        self.task_ui_elements["Replay Trajectory"].enabled = True
        self.task_ui_elements["Replay Scene"].enabled = True
        return

    def post_clear_button_event(self):
        self.task_ui_elements["Replay Trajectory"].enabled = False
        self.task_ui_elements["Replay Scene"].enabled = False
        return

    def build_data_logging_ui(self, frame):
        with frame:
            with ui.VStack(spacing=5):
                frame.title = "Data Replay"
                frame.visible = True
                example_data_file = os.path.abspath(
                    os.path.join(os.path.abspath(__file__), "../../../../../data/example_data_file.json")
                )
                dict = {
                    "label": "Data File",
                    "type": "stringfield",
                    "default_val": example_data_file,
                    "tooltip": "Data File",
                    "on_clicked_fn": None,
                    "use_folder_picker": False,
                    "read_only": False,
                }
                self.task_ui_elements["Data File"] = str_builder(**dict)
                dict = {
                    "label": "Replay Trajectory",
                    "type": "button",
                    "text": "Replay Trajectory",
                    "tooltip": "Replay Trajectory",
                    "on_clicked_fn": self._on_replay_trajectory_button_event,
                }

                self.task_ui_elements["Replay Trajectory"] = btn_builder(**dict)
                self.task_ui_elements["Replay Trajectory"].enabled = False
                dict = {
                    "label": "Replay Scene",
                    "type": "button",
                    "text": "Replay Scene",
                    "tooltip": "Replay Scene",
                    "on_clicked_fn": self._on_replay_scene_button_event,
                }

                self.task_ui_elements["Replay Scene"] = btn_builder(**dict)
                self.task_ui_elements["Replay Scene"].enabled = False
        return
