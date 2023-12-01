import taichi as ti

from config_loader import ConfigLoader
from controller import Controller
from mpm import MPM
from render import Render

ti.init(arch=ti.gpu)

# Hyperparameters
width, height = 1920, 1080
G_number, max_timestep, max_hard = 64, 2, 2.5  # more particles, num: 65536
# G_number, max_timestep, max_hard = 32, 4, 4  # less particles, num: 8192
steps = 25  # time step
config_path = 'config.json'

if __name__ == "__main__":
    mpm = MPM(G_number, max_hard)
    config = ConfigLoader(config_path)
    controller = Controller(width, height, config.presets, config.materials)
    controller.init(mpm)
    render = Render(controller.window)
    while controller.window.running:
        for _ in range(steps):
            mpm.substep(*controller.dt, *controller.GRAVITY, *controller.Lame, *controller.plasticity_boundary,
                        *controller.H, controller.is_elastic_object)

        render.render(mpm, controller.use_random_colors, controller.particles_radius)
        controller.show_options(mpm, max_timestep, max_hard)
        controller.window.show()
