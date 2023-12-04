from .config_loader import ConfigLoader
from .controller import Controller
from .mpm import MPM
from .render import Render
from .objects import *
from .materials import *

"""
Material dreamworks based on MPM
"""
class Dragon:
    def __init__(self, config_path):
        config = ConfigLoader(config_path)
        self.max_timestep = config.max_timestep
        self.max_hard = config.max_hard
        self.steps = 25  # time step
        self.mpm = MPM(config.G_number, config.max_hard)
        self.controller = Controller(config.width, config.height, config.presets, config.materials)
        self.controller.init_from_ply(self.mpm)
        self.render = Render(self.controller.window)

    def run(self):
        start = PLY_NUM
        end = PLY_NUM+500
        while self.controller.window.running:
            if end < FLUID_NUM+PLY_NUM:
                self.mpm.spawn_fluid(start, end, 0.35, 0.6, 0.45, 0.05, 0.05, 0.05, WATER)
                start = end
                end += 1

            for _ in range(self.steps):
                self.mpm.substep(*self.controller.dt, *self.controller.GRAVITY, *self.controller.Lame,
                                 *self.controller.plasticity_boundary,
                                 *self.controller.H, self.controller.is_elastic_object)

            self.render.render(self.mpm, self.controller.use_random_colors, self.controller.particles_radius)
            self.controller.show_options(self.mpm, self.max_timestep, self.max_hard)
            self.controller.window.show()
