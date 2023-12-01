import numpy as np
import taichi as ti

from .cube_object import DIY_MATERIAL, SOLID_CUBE


class Controller:
    def __init__(self, width, height, presets, materials):
        res = (width, height)
        self.window = ti.ui.Window("MPM Material Dreamworks", res, vsync=True)
        self.gui = self.window.get_gui()
        self.presets = presets
        self.materials = materials

        self.dt = [2e-4]
        self.auto_restart = True
        self.GRAVITY = [0, -9.8, 0]  # 3d gravity

        self.E = 1000  # Young's modulus
        nu = 0.2  # Poisson's ratio
        self.Lame = [self.E / (2 * (1 + nu)), self.E * nu / ((1 + nu) * (1 - 2 * nu))]  # Lame parameters
        print("mu_0: ", self.Lame[0], "    lambda:", self.Lame[1])

        self.V_parameter = 0.5
        self.is_elastic_object = False
        self.H = [0.3]
        self.plasticity_boundary = [-250, 45]
        self.curr_preset_id = 0
        self.use_random_colors = False
        self.particles_radius = 0.01

    def init(self, mpm):
        mpm.init_objs(self.presets[self.curr_preset_id])

    def viscosity_to_lame_parameter(self, V: float):
        if V < 0:
            V = 0
        elif V > 0.7:
            V = 0.7

        mu_0 = self.E * V
        lambda_0 = self.E * (1 - V)
        return mu_0, lambda_0

    # options
    def show_options(self, mpm, max_timestep, max_hard):

        with self.gui.sub_window("Controller", 0., 0.0, 0.25, 0.1) as w:
            self.dt[0] = w.slider_float("Time Step", self.dt[0] * 10000, 0, max_timestep) / 10000
            if w.button("restart"):
                self.init(mpm)

        with self.gui.sub_window("Material Setting", 0., 0.8, 1.0, 0.2) as w:
            self.auto_restart = w.checkbox("Auto Restart", self.auto_restart)

            old_is_elastic_object = self.is_elastic_object
            self.is_elastic_object = w.checkbox("Elastic Object", self.is_elastic_object)

            low = 0
            old_H = self.H[0]
            if self.is_elastic_object:
                self.H[0] = w.slider_float("Hard", self.H[0], 0.25, max_hard)
                low = 0.05

            if self.is_elastic_object:
                self.Lame[0], self.Lame[1] = self.viscosity_to_lame_parameter(0.05)
            else:
                self.Lame[0], self.Lame[1] = self.viscosity_to_lame_parameter(self.V_parameter)

            old_Viscosity = self.V_parameter
            if not self.is_elastic_object:
                self.V_parameter = w.slider_float("low: Viscosity; high: Stiffness", self.V_parameter, low, 0.7)
                if old_Viscosity != self.V_parameter:
                    self.Lame[0], self.Lame[1] = self.viscosity_to_lame_parameter(self.V_parameter)

            # old_mu = Lame[0]
            # old_lambda = Lame[1]
            # Lame[0] = w.slider_float("Mu", Lame[0], low, 1000)
            # Lame[1] = w.slider_float("Lambda", Lame[1], 0, 1000)

            old_plasticity_boundary_0 = self.plasticity_boundary[0]
            old_plasticity_boundary_1 = self.plasticity_boundary[1]
            if not self.is_elastic_object:
                self.plasticity_boundary[0] = w.slider_float("plasticity lower bound", self.plasticity_boundary[0],
                                                             -400, -100)
                self.plasticity_boundary[1] = w.slider_float("plasticity upper bound", self.plasticity_boundary[1], 0,
                                                             100)

            if self.is_elastic_object != old_is_elastic_object \
                    or old_H != self.H[0] \
                    or old_Viscosity != self.V_parameter \
                    or old_plasticity_boundary_0 != self.plasticity_boundary[0] or old_plasticity_boundary_1 != \
                    self.plasticity_boundary[1]:
                # or Lame[0] != old_mu or Lame[1] != old_lambda \

                if self.auto_restart:
                    self.init(mpm)
                else:
                    mpm.reset()

        with self.gui.sub_window("Color", 0., 0.7, 0.2, 0.1) as w:
            self.use_random_colors = w.checkbox("use_random_colors", self.use_random_colors)
            if not self.use_random_colors:
                self.materials[DIY_MATERIAL].color = w.color_edit_3("material color",
                                                                    self.materials[
                                                                        DIY_MATERIAL].color)
                material_colors = []
                material_colors.append(self.materials[DIY_MATERIAL].color)
                material_colors.append(self.materials[SOLID_CUBE].color)
                mpm.set_color_by_material(np.array(material_colors, dtype=np.float32))

        with self.gui.sub_window("Gravity", 0., 0.55, 0.2, 0.15) as w:
            self.GRAVITY[0] = w.slider_float("x", self.GRAVITY[0], -10, 10)
            self.GRAVITY[1] = w.slider_float("y", self.GRAVITY[1], -10, 10)
            self.GRAVITY[2] = w.slider_float("z", self.GRAVITY[2], -10, 10)

        with self.gui.sub_window("Presets", 0., 0.45, 0.2, 0.1) as w:
            old_preset = self.curr_preset_id
            for i in range(len(self.presets)):
                if w.checkbox(self.presets[i].name, self.curr_preset_id == i):
                    self.curr_preset_id = i
            if self.curr_preset_id != old_preset:
                self.init(mpm)
