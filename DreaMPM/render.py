import taichi as ti


class Render:
    def __init__(self, window):
        self.window = window
        self.canvas = window.get_canvas()
        self.scene = ti.ui.Scene()

        self.camera = ti.ui.Camera()
        self.camera.position(0.5, 1.0, 1.95)
        self.camera.lookat(0.5, 0.3, 0.5)
        self.camera.fov(55)

    def render(self, mpm, use_random_colors, particles_radius):
        self.camera.track_user_inputs(self.window, movement_speed=0.03, hold_key=ti.ui.RMB)
        self.scene.set_camera(self.camera)

        self.scene.ambient_light((0, 0, 0))

        colors_used = mpm.F_colors_random if use_random_colors else mpm.F_colors
        self.scene.particles(mpm.F_x, per_vertex_color=colors_used, radius=particles_radius)

        self.scene.point_light(pos=(0.5, 1.5, 0.5), color=(0.5, 0.5, 0.5))
        self.scene.point_light(pos=(0.5, 1.5, 1.5), color=(0.5, 0.5, 0.5))

        self.canvas.scene(self.scene)
