# Copyright (c) 2018-2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#

import os
import asyncio
import weakref
import carb
import omni
import omni.ui as ui
import omni.kit.test
from omni.kit.menu.utils import add_menu_items, remove_menu_items, MenuItemDescription

from omni.isaac.ui.ui_utils import setup_ui_headers, get_style, btn_builder, scrolling_frame_builder
from omni.isaac.core.utils.stage import open_stage_async
from omni.isaac.dynamic_control import _dynamic_control
from pxr import Usd
import omni.physx as _physx
import numpy as np

# joint animation states
ANIM_SEEK_LOWER = 1
ANIM_SEEK_UPPER = 2
ANIM_SEEK_DEFAULT = 3
ANIM_FINISHED = 4

EXTENSION_NAME = "Joint Controller"


def clamp(x, min_value, max_value):
    return max(min(x, max_value), min_value)


def _print_body_rec(dc, body, indent_level=0):
    indent = " " * indent_level

    body_name = dc.get_rigid_body_name(body)
    str_output = "%sBody: %s\n" % (indent, body_name)

    for i in range(dc.get_rigid_body_child_joint_count(body)):
        joint = dc.get_rigid_body_child_joint(body, i)
        joint_name = dc.get_joint_name(joint)
        child = dc.get_joint_child_body(joint)
        child_name = dc.get_rigid_body_name(child)
        str_output = str_output + "%s  Joint: %s -> %s\n" % (indent, joint_name, child_name)
        str_output = str_output + _print_body_rec(dc, child, indent_level + 4)
    return str_output


class Extension(omni.ext.IExt):
    def on_startup(self, ext_id: str):
        """Initialize extension and UI elements"""
        self._dc = _dynamic_control.acquire_dynamic_control_interface()
        self._window = None
        self._physxIFace = _physx.acquire_physx_interface()
        self._physx_subscription = None
        self._timeline = omni.timeline.get_timeline_interface()

        self.ar = _dynamic_control.INVALID_HANDLE

        ext_manager = omni.kit.app.get_app().get_extension_manager()
        dc_ext_id = ext_manager.get_enabled_extension_id("omni.isaac.dynamic_control")
        self._asset_path = ext_manager.get_extension_path(dc_ext_id)
        self._ext_id = ext_id
        menu_items = [
            MenuItemDescription(name=EXTENSION_NAME, onclick_fn=lambda a=weakref.proxy(self): a._menu_callback())
        ]
        self._menu_items = [MenuItemDescription(name="Dynamic Control", sub_menu=menu_items)]
        add_menu_items(self._menu_items, "Isaac Examples")

    def _menu_callback(self):
        self._build_ui()

    def _build_ui(self):
        if not self._window:
            self._window = ui.Window(
                title=EXTENSION_NAME, width=500, height=500, visible=True, dockPreference=ui.DockPreference.LEFT_BOTTOM
            )
            with self._window.frame:
                with ui.VStack(spacing=5, height=0):
                    title = "Robot Joint Controller Example"
                    doc_link = "https://docs.omniverse.nvidia.com/app_isaacsim/app_isaacsim/overview.html"

                    overview = "This example shows how move a robot arm by driving its joints."
                    overview += "First press the 'Load Robot' button and then press `Move Joints` to simulate."
                    overview += "\n\nPress the 'Open in IDE' button to view the source code."

                    setup_ui_headers(self._ext_id, __file__, title, doc_link, overview)

                    frame = ui.CollapsableFrame(
                        title="Command Panel",
                        height=0,
                        collapsed=False,
                        style=get_style(),
                        style_type_name_override="CollapsableFrame",
                        horizontal_scrollbar_policy=ui.ScrollBarPolicy.SCROLLBAR_AS_NEEDED,
                        vertical_scrollbar_policy=ui.ScrollBarPolicy.SCROLLBAR_ALWAYS_ON,
                    )
                    with frame:
                        with ui.VStack(style=get_style(), spacing=5, height=0):
                            kwargs = {
                                "label": "Load Robot",
                                "type": "button",
                                "text": "Load",
                                "tooltip": "Loads a Robot Arm and sets its properties",
                                "on_clicked_fn": self._on_load_robot,
                            }
                            btn_builder(**kwargs)

                            kwargs = {
                                "label": "Move Joints",
                                "type": "button",
                                "text": "Move",
                                "tooltip": "Moves the Robot's Joints",
                                "on_clicked_fn": self._on_move_joints,
                            }
                            btn_builder(**kwargs)

                            self.dof_states_label = scrolling_frame_builder()
                            self.dof_props_label = scrolling_frame_builder()

    def on_shutdown(self):
        self._sub_stage_event = None
        self._physx_subscription = None
        remove_menu_items(self._menu_items, "Isaac Examples")
        self._window = None

    async def _setup_scene(self):
        await open_stage_async(self._asset_path + "/data/usd/robots/franka/franka.usd")
        await omni.kit.app.get_app().next_update_async()
        self._viewport = omni.kit.viewport_legacy.get_default_viewport_window()
        self._viewport.set_camera_position("/OmniverseKit_Persp", 150, -150, 150, True)
        self._viewport.set_camera_target("/OmniverseKit_Persp", -96, 108, 0, True)
        self._timeline.play()
        await omni.kit.app.get_app().next_update_async()

    def _on_load_robot(self):
        asyncio.ensure_future(self._setup_scene())

    def _on_move_joints(self):
        self._physx_subscription = self._physxIFace.subscribe_physics_step_events(self._on_physics_step)
        self._timeline.play()
        self._sub_stage_event = (
            omni.usd.get_context().get_stage_event_stream().create_subscription_to_pop(self._on_stage_event)
        )

    def _on_stage_event(self, event):
        if event.type == int(omni.usd.StageEventType.OPENED) or event.type == int(omni.usd.StageEventType.CLOSED):
            # stage was opened or closed, cleanup
            self._physx_subscription = None
            self.ar = _dynamic_control.INVALID_HANDLE

    def _on_first_step(self):
        self.ar = self._dc.get_articulation("/panda")
        if self.ar == _dynamic_control.INVALID_HANDLE:
            carb.log_warn("'%s' is not an articulation, please click load button first" % "/panda")
            return False

        num_dofs = self._dc.get_articulation_dof_count(self.ar)
        dof_props = self._dc.get_articulation_dof_properties(self.ar)

        dof_types = dof_props["type"]
        has_limits = dof_props["hasLimits"]
        lower_limits = dof_props["lower"]
        upper_limits = dof_props["upper"]

        self.dof_props_label.text = str("--- Degree of freedom (DOF) properties:\n") + str(dof_props) + "\n"

        # allocate dof state buffer
        dof_states = np.zeros(num_dofs, dtype=_dynamic_control.DofState.dtype)
        dof_positions = dof_states["pos"]
        speed_scale = 1.0

        # initialize default positions, limits, and speeds (make sure they are in reasonable ranges)
        defaults = np.array([0.0, -0.0, 0.0, -0.0, 0.0, 3.037, 0.741, 4.0, 4.0], dtype=np.float32)
        speeds = np.zeros(num_dofs)

        for i in range(num_dofs):
            if has_limits[i]:
                if dof_types[i] == _dynamic_control.DOF_ROTATION:
                    lower_limits[i] = clamp(lower_limits[i], -np.pi, np.pi)
                    upper_limits[i] = clamp(upper_limits[i], -np.pi, np.pi)
                # make sure our default position is in range
                if lower_limits[i] > 0.0:
                    defaults[i] = lower_limits[i]
                elif upper_limits[i] < 0.0:
                    defaults[i] = upper_limits[i]
            else:
                # set reasonable animation limits for unlimited joints
                if dof_types[i] == _dynamic_control.DOF_ROTATION:
                    # unlimited revolute joint
                    lower_limits[i] = -np.pi
                    upper_limits[i] = np.pi
                elif dof_types[i] == _dynamic_control.DOF_TRANSLATION:
                    # unlimited prismatic joint
                    lower_limits[i] = -1.0
                    upper_limits[i] = 1.0
            # set DOF position to default
            dof_positions[i] = defaults[i]
            # set speed depending on DOF type and range of motion
            if dof_types[i] == _dynamic_control.DOF_ROTATION:
                speeds[i] = speed_scale * clamp(2 * (upper_limits[i] - lower_limits[i]), 0.25 * np.pi, 3.0 * np.pi)
            else:
                speeds[i] = speed_scale * clamp(2 * (upper_limits[i] - lower_limits[i]), 0.1, 7.0)

        self.num_dofs = num_dofs
        self.defaults = defaults
        self.speeds = speeds
        self.lower_limits = lower_limits
        self.upper_limits = upper_limits

        self.dof_states = dof_states
        self.dof_positions = dof_positions

        self.anim_state = ANIM_SEEK_LOWER
        self.current_dof = 0
        return True

    def _on_physics_step(self, step):
        if self._dc.is_simulating():
            if self.ar == _dynamic_control.INVALID_HANDLE:
                if not self._on_first_step():
                    return
            dof = self.current_dof
            speed = self.speeds[dof]
            # animate the dofs
            if self.anim_state == ANIM_SEEK_LOWER:
                self.dof_positions[dof] -= speed * step
                if self.dof_positions[dof] <= self.lower_limits[dof]:
                    self.dof_positions[dof] = self.lower_limits[dof]
                    self.anim_state = ANIM_SEEK_UPPER
            elif self.anim_state == ANIM_SEEK_UPPER:
                self.dof_positions[dof] += speed * step
                if self.dof_positions[dof] >= self.upper_limits[dof]:
                    self.dof_positions[dof] = self.upper_limits[dof]
                    self.anim_state = ANIM_SEEK_DEFAULT
            if self.anim_state == ANIM_SEEK_DEFAULT:
                self.dof_positions[dof] -= speed * step
                if self.dof_positions[dof] <= self.defaults[dof]:
                    self.dof_positions[dof] = self.defaults[dof]
                    self.anim_state = ANIM_FINISHED
            elif self.anim_state == ANIM_FINISHED:
                self.dof_positions[dof] = self.defaults[dof]
                self.current_dof = (dof + 1) % self.num_dofs
                self.anim_state = ANIM_SEEK_LOWER
                # print("Animating DOF %d" % (self.current_dof,))
            self._dc.wake_up_articulation(self.ar)
            self._dc.set_articulation_dof_position_targets(self.ar, self.dof_positions)
            dof_states = self._dc.get_articulation_dof_states(self.ar, _dynamic_control.STATE_ALL)
            self.dof_states_label.text = str("--- Degree of freedom (DOF) states:\n") + str(dof_states) + "\n"

        return
