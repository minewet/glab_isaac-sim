# Copyright (c) 2018-2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#
from pxr import Gf
import carb
import omni.usd
import omni.ext
import omni.ui as ui
import omni.physx as _physx
from omni.kit.menu.utils import add_menu_items, remove_menu_items, MenuItemDescription
from pxr import UsdGeom
import math
from omni.isaac.ui.ui_utils import (
    setup_ui_headers,
    get_style,
    btn_builder,
    xyz_builder,
    add_separator,
    dropdown_builder,
    combo_floatfield_slider_builder,
    state_btn_builder,
)

import asyncio
import gc
import weakref
import numpy as np
from omni.isaac.dynamic_control import _dynamic_control
from .utils.simple_robot_controller import RobotController
from omni.isaac.core.utils.stage import set_stage_up_axis
from omni.isaac.core.utils.prims import create_prim
from omni.isaac.core import PhysicsContext
from omni.isaac.core.utils.nucleus import get_assets_root_path

EXTENSION_NAME = "SK: Navigation"


def create_xyz(init={"X": 100, "Y": 100, "Z": 0}):
    all_axis = ["X", "Y", "Z"]
    colors = {"X": 0xFF5555AA, "Y": 0xFF76A371, "Z": 0xFFA07D4F}
    float_drags = {}
    for axis in all_axis:
        with ui.HStack():
            with ui.ZStack(width=15):
                ui.Rectangle(
                    width=15,
                    height=20,
                    style={"background_color": colors[axis], "border_radius": 3, "corner_flag": ui.CornerFlag.LEFT},
                )
                ui.Label(axis, name="transform_label", alignment=ui.Alignment.CENTER)
            float_drags[axis] = ui.FloatDrag(name="transform", min=-1000000, max=1000000, step=1, width=100)
            float_drags[axis].model.set_value(init[axis])
    return float_drags


class Extension(omni.ext.IExt):
    def on_startup(self, ext_id: str):
        """Initialize extension and UI elements"""

        self._ext_id = ext_id

        self._timeline = omni.timeline.get_timeline_interface()
        self._viewport = omni.kit.viewport_legacy.get_default_viewport_window()
        self._usd_context = omni.usd.get_context()
        self._stage = self._usd_context.get_stage()

        self._window = None
        # self._window = ui.Window(EXTENSION_NAME, width=500, height=175, visible=False)
        # self._window.set_visibility_changed_fn(self._on_window)

        menu_items = [
            MenuItemDescription(name=EXTENSION_NAME, onclick_fn=lambda a=weakref.proxy(self): a._menu_callback())
        ]
        self._menu_items = [MenuItemDescription(name="Ewha GLAB", sub_menu=menu_items)]
        add_menu_items(self._menu_items, "Isaac Examples")
        self._dc = _dynamic_control.acquire_dynamic_control_interface()
        self._build_ui()

        self._setup_done = False
        self._rc = None

        

    def _menu_callback(self):
        self._window.visible = not self._window.visible

    def _on_window(self, visible):
        if self._window.visible:
            self._sub_stage_event = self._usd_context.get_stage_event_stream().create_subscription_to_pop(
                self._on_stage_event
            )
        else:
            self._sub_stage_event = None

    def _build_ui(self):
        if not self._window:
            self._window = ui.Window(
                title=EXTENSION_NAME, width=0, height=0, visible=False, dockPreference=ui.DockPreference.LEFT_BOTTOM
            )
            self._window.set_visibility_changed_fn(self._on_window)
            with self._window.frame:
                with ui.VStack(spacing=5, height=0):
                    title = "Mobile Robot Navigation Example"
                    doc_link = (
                        "https://docs.omniverse.nvidia.com/app_isaacsim/app_isaacsim/manual_omni_isaac_demos.html"
                    )

                    overview = "SK building."
                    overview += "\n\nPress the button below to operate simple simulation to the robots."
                    overview += "\n\nPress the 'Open in IDE' button to view the source code."

                    setup_ui_headers(self._ext_id, __file__, title, doc_link, overview)

                    frame0 = ui.CollapsableFrame(
                        title="Sk building",
                        height=0,
                        collapsed=False,
                        style=get_style(),
                        style_type_name_override="CollapsableFrame",
                        horizontal_scrollbar_policy=ui.ScrollBarPolicy.SCROLLBAR_AS_NEEDED,
                        vertical_scrollbar_policy=ui.ScrollBarPolicy.SCROLLBAR_ALWAYS_ON,
                    )

                    frame1 = ui.CollapsableFrame(
                        title="Navigation",
                        height=0,
                        collapsed=False,
                        style=get_style(),
                        style_type_name_override="CollapsableFrame",
                        horizontal_scrollbar_policy=ui.ScrollBarPolicy.SCROLLBAR_AS_NEEDED,
                        vertical_scrollbar_policy=ui.ScrollBarPolicy.SCROLLBAR_ALWAYS_ON,
                    )

                    frame2 = ui.CollapsableFrame(
                        title="Follow Target",
                        height=0,
                        collapsed=True,
                        style=get_style(),
                        style_type_name_override="CollapsableFrame",
                        horizontal_scrollbar_policy=ui.ScrollBarPolicy.SCROLLBAR_AS_NEEDED,
                        vertical_scrollbar_policy=ui.ScrollBarPolicy.SCROLLBAR_ALWAYS_ON,
                    )
                    with frame0:
                        with ui.VStack(style=get_style(), spacing=5):

                            args = {
                                "label": "SK building",
                                "type": "button",
                                "text": "Load",
                                "tooltip": "Load a SK building (including robots) into the Scene",
                                "on_clicked_fn": self._on_environment_setup,
                            }
                            self._load_btn = btn_builder(**args)

                            ui.Spacer()

                    with frame1:
                        with ui.VStack(style=get_style(), spacing=5):

                            args = {
                                "label": "Robot Type",
                                "default_val": 0,
                                "tooltip": "Select which type of Robot to simulate",
                                "items": ["ridgeback_iiwa"],
                            }
                            self._robot_option = dropdown_builder(**args)

                            args = {
                                "label": "Move Robot",
                                "type": "button",
                                "text": "Move",
                                "tooltip": "Move the robot Forward",
                                "on_clicked_fn": self._on_move_fn,
                            }
                            self._move_btn = btn_builder(**args)

                            args = {
                                "label": "Spin Robot",
                                "type": "button",
                                "text": "Rotate CCW",
                                "tooltip": "Rotate the robot",
                                "on_clicked_fn": self._on_rotate_ccw_fn,
                            }
                            self._rotate_btn_1 = btn_builder(**args)

                            args = {
                                "label": "Spin Robot",
                                "type": "button",
                                "text": "Rotate CW",
                                "tooltip": "Rotate the robot",
                                "on_clicked_fn": self._on_rotate_cw_fn,
                            }
                            self._rotate_btn_2 = btn_builder(**args)

                            add_separator()

                            args = {
                                "label": "Target Pose",
                                "axis_count": 3,
                                "min": -1000,
                                "max": 1000,
                                "step": 1,
                                "tooltip": "Pose is specified as (X, Y, theta)",
                            }
                            self.goal_coord = xyz_builder(**args)
                            args = {"label": "Speed Coefficient", "default_val": 0.5, "min": 0, "max": 1, "step": 0.05}
                            self._start_vel, _ = combo_floatfield_slider_builder(**args)
                            args = {"label": "Accel Coefficient", "default_val": 0.02, "min": 0, "max": 1, "step": 0.01}
                            self._start_acc, _ = combo_floatfield_slider_builder(**args)
                            args = {"label": "max speed", "default_val": 1.50, "min": 0, "max": 3.000, "step": 1}
                            self._max_speed, _ = combo_floatfield_slider_builder(**args)
                            args = {
                                "label": "Move to Target",
                                "type": "button",
                                "text": "Move",
                                "tooltip": "Move robot to target pose",
                                "on_clicked_fn": self._on_navigate_fn,
                            }
                            self._navigate_btn = btn_builder(**args)

                            args = {
                                "label": "Stop",
                                "type": "button",
                                "text": "Stop",
                                "tooltip": "Pause the robot when navigating",
                                "on_clicked_fn": self._on_navigate_stop_fn,
                            }
                            self._stop_btn = btn_builder(**args)
                            self._stop_btn.enabled = False

                            ui.Spacer()

                    with frame2:
                        with ui.VStack(style=get_style(), spacing=5):
                            args = {
                                "label": "Robot Type",
                                "default_val": 0,
                                "tooltip": "Select which type of Robot to simulate",
                                "items": ["ridgeback_iiwa", "fetch"],
                            }
                            self._robot_option_ik = dropdown_builder(**args)

                            args = {
                                "label": "Follow Target",
                                "type": "button",
                                "a_text": "START",
                                "b_text": "STOP",
                                "tooltip": "Follow Target",
                                "on_clicked_fn": self._on_follow_target_button_event,
                            }
                            self._follow_target_btn = state_btn_builder(**args)
                            self._follow_target_btn.enabled = False

                            ui.Spacer()


    async def _create_robot(self, task):
        done, pending = await asyncio.wait({task})
        if task in done:
            print("Loading Robot Environment")
            self._viewport.set_camera_position("/OmniverseKit_Persp", -5.7, -7.5, 4.8, True)
            self._viewport.set_camera_target("/OmniverseKit_Persp", -1.8, 4.1, 0.1, True)
            self._stage = self._usd_context.get_stage()
            self._assets_root_path = get_assets_root_path()
            if self._assets_root_path is None:
                carb.log_error("Could not find Isaac Sim assets folder")
                return

            current_robot_index = self._robot_option.get_item_value_model().as_int
            
            if current_robot_index == 0:
                #asset_path = self._assets_root_path + "/Isaac/Robots/Transporter"
                #robot_usd = "C:/Users/kimhj/Downloads/sk/ridgeback.usd"
                self._robot_prim_path = "/sk/ridgeback_iiwa"
                self._robot_chassis =  self._robot_prim_path + "/chassis_link"
                self._robot_wheels = ["front_left_wheel", "front_right_wheel","rear_left_wheel", "rear_right_wheel"]
                self._robot_wheels_speed = [50, 50]
                self._wheelbase_Length = 0.56
                self._wheel_radius = 0.082


            set_stage_up_axis("z")
            PhysicsContext(physics_dt=1.0 / 60.0)
            create_prim(
                prim_path="/sk",
                #usd_path=self._assets_root_path + "/Isaac/Environments/Grid/gridroom_curved.usd",]
                usd_path = "C:/Users/kimhj/Downloads/sk/demo.usd"
                #position=np.array([0, 0, -20]),
                #scale=np.array([20,20,20]),
            )

            # robot usd 파일을 reference를 통해 prim을 씬에 추가
            # setup high-level robot prim
            # self.prim = self._stage.DefinePrim(self._robot_prim_path, "Xform")
            # self.prim.GetReferences().AddReference(robot_usd)

            #그러나 여기서는 robot usd 파일이 배경에 포함되어있기 때문에 위에서 미리 지정하였다.
            #self._robot_prim_path = "/robot"

    def _on_stage_event(self, event):
        self._stage = self._usd_context.get_stage()
        if event.type == int(omni.usd.StageEventType.OPENED):
            self._move_btn.enabled = self._setup_done
            self._rotate_btn_1.enabled = self._setup_done
            self._rotate_btn_2.enabled = self._setup_done
            self._navigate_btn.enabled = self._setup_done
            self._stop_btn.enabled = self._setup_done
            self._follow_target_btn.enabled = self._setup_done
            self._stage_unit = UsdGeom.GetStageMetersPerUnit(self._stage)
            if self._rc:
                self._rc.enable_navigation(False)
            self._setup_done = False

    async def _play(self, task):
        done, pending = await asyncio.wait({task})
        if task in done:
            self._timeline.play()
            await asyncio.sleep(1)

    async def _on_setup_fn(self, task):
        done, pending = await asyncio.wait({task})
        if task in done:
            self._stage = self._usd_context.get_stage()
            # setup robot controller
            self._rc = RobotController(
                self._stage,
                # self._timeline,
                self._dc,
                self._robot_prim_path,
                self._robot_chassis,
                self._robot_wheels,
                self._robot_wheels_speed,
                [0.2, 0.5],
                self._wheelbase_Length,
                self._wheel_radius,
            )
            self._rc.control_setup()
            # start stepping
            self._editor_event_subscription = _physx.get_physx_interface().subscribe_physics_step_events(
                self._rc.update
            )
            self._debug_draw_subs = (
                omni.kit.app.get_app().get_update_event_stream().create_subscription_to_pop(self._rc.draw_path)
            )


    def _on_environment_setup(self):
        # wait for new stage before creating robot
        task = asyncio.ensure_future(omni.usd.get_context().new_stage_async())
        task1 = asyncio.ensure_future(self._create_robot(task))
        # set editor to play before setting up robot controller
        task2 = asyncio.ensure_future(self._play(task1))
        asyncio.ensure_future(self._on_setup_fn(task2))

        # self._load_btn.enabled=False
        self._move_btn.enabled = True
        self._rotate_btn_1.enabled = True
        self._rotate_btn_2.enabled = True
        self._navigate_btn.enabled = True
        self._stop_btn.enabled = True
        self._follow_target_btn.enabled = True
        self._setup_done = True

        

    def _on_move_fn(self):
        print("Moving forward")
        self._rc.control_command(30, 30)

    def _on_rotate_ccw_fn(self):
        print("Rotating in-place")
        self._rc.control_command(-30, 30)

    def _on_rotate_cw_fn(self):
        print("Rotating in-place")
        self._rc.control_command(30, -30)

    def _on_navigate_fn(self):
        goal_x = self.goal_coord[0].get_value_as_float() * self._stage_unit
        goal_y = self.goal_coord[1].get_value_as_float() * self._stage_unit
        goal_z = self.goal_coord[2].get_value_as_float()
        max_speed = self._max_speed.get_value_as_float() * self._stage_unit
        print("Navigating to goal ({}, {}, {})".format(goal_x, goal_y, goal_z))
        sv = self._start_vel.get_value_as_float()
        sa = self._start_acc.get_value_as_float()

        gv = sv  # self._goal_speed.get_value_as_float()
        ga = sa  # self._goal_acc.get_value_as_float()
        self._rc.set_goal(goal_x, goal_y, math.radians(goal_z), sv, sa, gv, ga, max_speed)
        self._rc.enable_navigation(True)

    def _on_navigate_stop_fn(self):
        print("Navigation Stopped")
        self._rc.enable_navigation(False)
        self._rc.control_command(0, 0)

    def _on_follow_target_button_event(self, val): 
        # state_btn의 이벤트 함수는 val까지 input을 받아야 한다
        print(self._robot_option_ik.get_item_value_model().as_int)

        

    def on_shutdown(self):
        self._rc = None
        self._timeline.stop()
        self._editor_event_subscription = None
        remove_menu_items(self._menu_items, "Isaac Examples")
        self._window = None
        gc.collect()
