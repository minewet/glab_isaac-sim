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
import textwrap
import carb
import omni
import omni.ui as ui
from omni.kit.menu.utils import add_menu_items, remove_menu_items, MenuItemDescription
from omni.isaac.ui.ui_utils import setup_ui_headers, get_style, btn_builder, scrolling_frame_builder
from omni.isaac.core.utils.stage import open_stage_async
from omni.isaac.dynamic_control import _dynamic_control
from pxr import Usd

import omni.physx as _physx
import omni.kit.menu

EXTENSION_NAME = "Read Articulations"


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
        self._dc = _dynamic_control.acquire_dynamic_control_interface()
        self._window = None
        self._physxIFace = _physx.acquire_physx_interface()

        self._timeline = omni.timeline.get_timeline_interface()

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
                title=EXTENSION_NAME, width=700, height=700, visible=True, dockPreference=ui.DockPreference.LEFT_BOTTOM
            )
        with self._window.frame:
            with ui.VStack(spacing=5, height=0):

                title = "Read Articulation Information"
                doc_link = "https://docs.omniverse.nvidia.com/app_isaacsim/app_isaacsim/overview.html"

                overview = "This sample loads a Franka robot and enables simulation. Various information for the robot articulation degrees of freedom is retrieved and shown on screen."
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
                        dict = {
                            "label": "Load Franka USD",
                            "type": "button",
                            "text": "Load",
                            "tooltip": "Press to load the Franka USD file and start simulation",
                            "on_clicked_fn": self._on_load_robot,
                        }
                        btn_builder(**dict)

                        dict = {
                            "label": "Get Articulation Info",
                            "type": "button",
                            "text": "Get Info",
                            "tooltip": "Pressing this button will print information below",
                            "on_clicked_fn": self._on_print_info,
                        }
                        btn_builder(**dict)
                        self.hierarchy_label = scrolling_frame_builder("Hierarchy", "scrolling_frame")
                        self.body_states_label = scrolling_frame_builder("Body States", "scrolling_frame")
                        self.dof_states_label = scrolling_frame_builder("Joint States", "scrolling_frame")
                        self.dof_props_label = scrolling_frame_builder("Joint Properties", "scrolling_frame")

        self._window.visible = True

    def on_shutdown(self):
        remove_menu_items(self._menu_items, "Isaac Examples")
        self._window = None

    async def _setup_scene(self):
        # wait for the stage load task to finish before setting camera and starting simulation
        await open_stage_async(self._asset_path + "/data/usd/robots/franka/franka.usd")
        await omni.kit.app.get_app().next_update_async()
        self._viewport = omni.kit.viewport_legacy.get_default_viewport_window()
        self._viewport.set_camera_position("/OmniverseKit_Persp", 150, 150, 50, True)
        self._viewport.set_camera_target("/OmniverseKit_Persp", 0, 0, 50, True)
        self._timeline.play()
        await omni.kit.app.get_app().next_update_async()

    def _on_load_robot(self):
        asyncio.ensure_future(self._setup_scene())

    def _on_print_info(self):
        ar = self._dc.get_articulation("/panda")
        if ar == _dynamic_control.INVALID_HANDLE:
            carb.log_warn("'%s' is not an articulation, please click load button first" % "/panda")
            return

        root = self._dc.get_articulation_root_body(ar)
        self.hierarchy_label.text = str("Articulation handle %d \n" % ar) + _print_body_rec(self._dc, root)

        body_states = self._dc.get_articulation_body_states(ar, _dynamic_control.STATE_ALL)
        self.body_states_label.text = str(body_states) + "\n"

        dof_states = self._dc.get_articulation_dof_states(ar, _dynamic_control.STATE_ALL)
        self.dof_states_label.text = str(dof_states) + "\n"

        dof_props = self._dc.get_articulation_dof_properties(ar)
        self.dof_props_label.text = str(dof_props) + "\n"

        return
