from .config_loader import ConfigLoader
from .controller import Controller
from .mpm import MPM
from .render import Render

"""
Material dreamworks based on MPM
"""
class DMPM:
    def __init__(self, width, height, G_number, max_timestep, max_hard, config_path):
        self.max_timestep = max_timestep
        self.max_hard = max_hard
        self.steps = 25  # time step
        self.mpm = MPM(G_number, max_hard)
        config = ConfigLoader(config_path)
        self.controller = Controller(width, height, config.presets, config.materials)
        self.controller.init(self.mpm)
        self.render = Render(self.controller.window)

    def run(self):
        while self.controller.window.running:
            for _ in range(self.steps):
                self.mpm.substep(*self.controller.dt, *self.controller.GRAVITY, *self.controller.Lame,
                                 *self.controller.plasticity_boundary,
                                 *self.controller.H, self.controller.is_elastic_object)

            self.render.render(self.mpm, self.controller.use_random_colors, self.controller.particles_radius)
            self.controller.show_options(self.mpm, self.max_timestep, self.max_hard)
            self.controller.window.show()
