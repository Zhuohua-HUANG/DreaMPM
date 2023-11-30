import numpy as np

import taichi as ti

# Hyperparameters
width, height = 1920, 1080
# G_number, max_timestep, max_hard = 64, 2.25, 2.75  # more particles, num: 65536
G_number, max_timestep, max_hard = 32, 4, 4 # less particles, num: 8192


ti.init(arch=ti.gpu)

dim = 3
P_number = G_number ** dim // 2 ** (dim - 1)  # 65536 or 8192
print("particle number:", P_number)  # you can check the particle number right here

# time step
steps = 25
dt = [2e-4]

auto_restart = True

dx = 1 / G_number
p_rho = 1  # 1
p_vol = (dx * 0.5) ** 2
p_mass = p_vol * p_rho
GRAVITY = [0, -9.8, 0]  # 3d gravity
bound = 3
E = 1000  # Young's modulus
nu = 0.2  # Poisson's ratio
Lame = [E / (2 * (1 + nu)), E * nu / ((1 + nu) * (1 - 2 * nu))]  # Lame parameters
print("mu_0: ", Lame[0], "    lambda:", Lame[1])

V_parameter = 0.5

is_elastic_object = False
H = [0.3]

plasticity_boundary = [-250, 45]

F_x = ti.Vector.field(dim, float, P_number)
F_v = ti.Vector.field(dim, float, P_number)
C = ti.Matrix.field(dim, dim, float, P_number)
def_grad = ti.Matrix.field(3, 3, dtype=float, shape=P_number)  # deformation gradient
F_Jp = ti.field(float,
                P_number)  # the volumetric rate of Plasticity change which will change the Hardening coefficient

F_colors = ti.Vector.field(4, float, P_number)
F_colors_random = ti.Vector.field(4, float, P_number)
F_materials = ti.field(int, P_number)
F_grid_v = ti.Vector.field(dim, float, (G_number,) * dim)
F_grid_m = ti.field(float, (G_number,) * dim)
F_used = ti.field(int, P_number)

neighbour = (3,) * dim

DIY_MATERIAL = 0
SOLID_CUBE = 1


@ti.kernel
def substep(dt: float, g_x: float, g_y: float, g_z: float,
            mu_0: float, lambda_0: float,
            plasticity_lower_bound: float, plasticity_upper_bound: float,
            H: float, is_elastic_object: bool):
    lower = 1 + plasticity_lower_bound / 10000
    upper = 1 + plasticity_upper_bound / 10000

    for I in ti.grouped(F_grid_m):
        F_grid_v[I] = ti.zero(F_grid_v[I])
        F_grid_m[I] = 0
    ti.loop_config(block_dim=G_number)
    # P2G
    for p in F_x:
        if F_used[p] == 0:
            continue
        Xp = F_x[p] / dx
        base = int(Xp - 0.5)
        fx = Xp - base
        w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1) ** 2, 0.5 * (fx - 0.5) ** 2]

        # Deformation gradient update moved to the front (elastic deformation) (hack!)
        def_grad[p] = (ti.Matrix.identity(float, 3) + dt * C[p]) @ def_grad[p]  # deformation gradient update

        # Hardening coefficient: material harder when compressed
        h = ti.exp(10 * (1.0 - F_Jp[p]))  # Plastic change effect the h, or the h is always 1

        if is_elastic_object:  # elastic object, make it softer
            h = H  # 0.1 ~ 1.0

        mu, la = mu_0 * h, lambda_0 * h

        if F_materials[p] == SOLID_CUBE:
            h = 0.3
            mu, la = 416.6 * h, 277.7 * h

        U, sig, V = ti.svd(def_grad[p])
        # the deformation rate (1dim)
        J = 1.0
        # calculate the J and update sig
        for d in ti.static(range(3)):

            orgin_sig = sig[d, d]  # singular values
            new_sig = orgin_sig
            if F_materials[p] == DIY_MATERIAL and not is_elastic_object:  # DIY_MATERIAL with Plasticity
                new_sig = ti.min(
                    ti.max(
                        orgin_sig,
                        lower  # default: 1 - 2.5e-2
                    ),
                    upper  # default: 1 + 4.5e-3
                )  # Plasticity: Forget too much deformation (hack)

            F_Jp[p] *= orgin_sig / new_sig  # volumetric rate of Plasticity change
            sig[d, d] = new_sig
            J *= new_sig

        if mu < 1.0:
            # if mu is too small, reset deformation gradient to avoid numerical instability
            new_F = ti.Matrix.identity(float, 3)
            new_F[0, 0] = J
            def_grad[p] = new_F
        else:
            # Reconstruct remain elastic deformation gradient after plasticity
            def_grad[p] = U @ sig @ V.transpose()

        # Corotated used elastic F_dg
        # stress_part1
        # = P(F) * F.transposed()
        # = (2µ(F − R) + λ(J − 1)J * F.transposed().inverse()) * F.transposed()
        # = 2µ(F − R)* F.transposed() + λ(J − 1)J
        stress_part1 = (
                2 * mu * (def_grad[p] - U @ V.transpose()) @ def_grad[p].transpose() +
                ti.Matrix.identity(
                    float, 3
                ) * la * J * (J - 1)
        )
        # stress
        # = 4*∆t / ∆x**2 * V * P(F) * F.transposed()
        # = 4*∆t / ∆x**2 * V * stress_part1
        stress = (-dt * p_vol * 4) * stress_part1 / dx ** 2
        affine = p_mass * C[p] + stress

        # 3d Grid momentum
        for offset in ti.static(ti.grouped(ti.ndrange(*neighbour))):
            dpos = (offset - fx) * dx
            weight = 1.0
            for i in ti.static(range(dim)):
                weight *= w[offset[i]][i]
            F_grid_v[base + offset] += weight * (p_mass * F_v[p] + affine @ dpos)
            F_grid_m[base + offset] += weight * p_mass

    for I in ti.grouped(F_grid_m):
        if F_grid_m[I] > 0:
            F_grid_v[I] /= F_grid_m[I]

        F_grid_v[I] += dt * ti.Vector([g_x, g_y, g_z])  # 3d gravity

        cond = (I < bound) & (F_grid_v[I] < 0) | (I > G_number - bound) & (F_grid_v[I] > 0)  # boundary condition

        F_grid_v[I] = ti.select(cond, 0, F_grid_v[I])
    ti.loop_config(block_dim=G_number)
    # G2P
    for p in F_x:
        if F_used[p] == 0:
            continue
        Xp = F_x[p] / dx
        base = int(Xp - 0.5)
        fx = Xp - base
        w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1) ** 2, 0.5 * (fx - 0.5) ** 2]
        new_v = ti.zero(F_v[p])
        new_C = ti.zero(C[p])
        for offset in ti.static(ti.grouped(ti.ndrange(*neighbour))):
            dpos = (offset - fx) * dx
            weight = 1.0
            for i in ti.static(range(dim)):
                weight *= w[offset[i]][i]
            g_v = F_grid_v[base + offset]
            new_v += weight * g_v
            new_C += 4 * weight * g_v.outer_product(dpos) / dx ** 2  # velocity gradient (APIC)
        # semi-implicit advection on particle
        F_v[p] = new_v
        F_x[p] += dt * F_v[p]
        C[p] = new_C


class CubeVolume:
    def __init__(self, minimum, size, material):
        self.minimum = minimum
        self.size = size
        self.volume = self.size.x * self.size.y * self.size.z
        self.material = material


@ti.kernel
def init_cube_vol(
        first_par: int,
        last_par: int,
        x_begin: float,
        y_begin: float,
        z_begin: float,
        x_size: float,
        y_size: float,
        z_size: float,
        material: int,
):
    for i in range(first_par, last_par):
        F_x[i] = ti.Vector([ti.random() for i in range(dim)]) * ti.Vector([x_size, y_size, z_size]) + ti.Vector(
            [x_begin, y_begin, z_begin]
        )
        F_Jp[i] = 1
        def_grad[i] = ti.Matrix([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        F_v[i] = ti.Vector([0.0, 0.0, 0.0])
        F_materials[i] = material
        F_colors_random[i] = ti.Vector([ti.random(), ti.random(), ti.random(), ti.random()])
        F_used[i] = 1


@ti.kernel
def set_all_unused():
    for p in F_used:
        F_used[p] = 0
        # placing in a very far place
        F_x[p] = ti.Vector([533799.0, 533799.0, 533799.0])
        F_Jp[p] = 1
        def_grad[p] = ti.Matrix([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        C[p] = ti.Matrix([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
        F_v[p] = ti.Vector([0.0, 0.0, 0.0])


@ti.kernel
def reset():
    for p in F_used:
        F_used[p] = 1
        F_Jp[p] = 1
        def_grad[p] = ti.Matrix([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        C[p] = ti.Matrix([[0, 0, 0], [0, 0, 0], [0, 0, 0]])


def init_vols(vols):
    set_all_unused()
    total_vol = 0
    for v in vols:
        total_vol += v.volume

    next_p = 0
    for i, v in enumerate(vols):
        v = vols[i]
        if isinstance(v, CubeVolume):
            par_count = int(v.volume / total_vol * P_number)
            if i == len(vols) - 1:  # this is the last volume, so use all remaining particles
                par_count = P_number - next_p
            init_cube_vol(next_p, next_p + par_count, *v.minimum, *v.size, v.material)
            next_p += par_count
        else:
            raise Exception("???")


@ti.kernel
def set_color_by_material(mat_color: ti.types.ndarray()):
    for i in range(P_number):
        mat = F_materials[i]
        F_colors[i] = ti.Vector([mat_color[mat, 0], mat_color[mat, 1], mat_color[mat, 2], 1.0])


presets = [
    [
        CubeVolume(ti.Vector([0.35, 0.35, 0.35]), ti.Vector([0.25, 0.25, 0.25]), DIY_MATERIAL),
    ],
    [
        CubeVolume(ti.Vector([0.40, 0.25, 0.40]), ti.Vector([0.25, 0.25, 0.25]), DIY_MATERIAL),
        CubeVolume(ti.Vector([0.30, 0.6, 0.30]), ti.Vector([0.25, 0.25, 0.25]), SOLID_CUBE),
    ],
]
preset_names = [
    "DIY_MATERIAL",
    "DIY_MATERIAL and SOLID_CUBE",
]

curr_preset_id = 0

paused = False

use_random_colors = False
particles_radius = 0.01

material_color = (1.0, 1.0, 1.0)

material_colors = [material_color, (0.93, 0.33, 0.23)]


def init():
    global paused
    init_vols(presets[curr_preset_id])


init()

res = (width, height)
window = ti.ui.Window("MPM Material Dreamworks", res, vsync=True)

canvas = window.get_canvas()
gui = window.get_gui()
scene = ti.ui.Scene()

camera = ti.ui.Camera()
camera.position(0.5, 1.0, 1.95)
camera.lookat(0.5, 0.3, 0.5)
camera.fov(55)


def viscosity_to_lame_parameter(V: float):
    if V < 0:
        V = 0
    elif V > 0.7:
        V = 0.7

    mu_0 = E * V
    lambda_0 = E * (1 - V)
    return mu_0, lambda_0


def show_options():
    global use_random_colors
    global paused
    global particles_radius
    global curr_preset_id
    global is_elastic_object
    global auto_restart
    global V_parameter

    with gui.sub_window("Controller", 0., 0.0, 0.25, 0.1) as w:

        dt[0] = w.slider_float("Time Step", dt[0] * 10000, 0, max_timestep) / 10000
        if w.button("restart"):
            init()

    with gui.sub_window("Material Setting", 0., 0.8, 1.0, 0.2) as w:
        auto_restart = w.checkbox("Auto Restart", auto_restart)

        old_is_elastic_object = is_elastic_object
        is_elastic_object = w.checkbox("Elastic Object", is_elastic_object)

        low = 0
        old_H = H[0]
        if is_elastic_object:
            H[0] = w.slider_float("Hard", H[0], 0.25, max_hard)
            low = 0.05

        if is_elastic_object:
            Lame[0], Lame[1] = viscosity_to_lame_parameter(0.05)
        else:
            Lame[0], Lame[1] = viscosity_to_lame_parameter(V_parameter)

        old_Viscosity = V_parameter
        if not is_elastic_object:
            V_parameter = w.slider_float("low: Viscosity; high: Stiffness", V_parameter, low, 0.7)
            if old_Viscosity != V_parameter:
                Lame[0], Lame[1] = viscosity_to_lame_parameter(V_parameter)

        # old_mu = Lame[0]
        # old_lambda = Lame[1]
        # Lame[0] = w.slider_float("Mu", Lame[0], low, 1000)
        # Lame[1] = w.slider_float("Lambda", Lame[1], 0, 1000)

        old_plasticity_boundary_0 = plasticity_boundary[0]
        old_plasticity_boundary_1 = plasticity_boundary[1]
        if not is_elastic_object:
            plasticity_boundary[0] = w.slider_float("plasticity lower bound", plasticity_boundary[0], -400, -100)
            plasticity_boundary[1] = w.slider_float("plasticity upper bound", plasticity_boundary[1], 0, 100)

        if is_elastic_object != old_is_elastic_object \
                or old_H != H[0] \
                or old_Viscosity != V_parameter \
                or old_plasticity_boundary_0 != plasticity_boundary[0] or old_plasticity_boundary_1 != \
                plasticity_boundary[1]:
            # or Lame[0] != old_mu or Lame[1] != old_lambda \

            if auto_restart:
                init()
            else:
                reset()

    with gui.sub_window("Color", 0., 0.7, 0.2, 0.1) as w:
        use_random_colors = w.checkbox("use_random_colors", use_random_colors)
        if not use_random_colors:
            material_colors[DIY_MATERIAL] = w.color_edit_3("material color", material_colors[DIY_MATERIAL])
            set_color_by_material(np.array(material_colors, dtype=np.float32))

    with gui.sub_window("Gravity", 0., 0.55, 0.2, 0.15) as w:
        GRAVITY[0] = w.slider_float("x", GRAVITY[0], -10, 10)
        GRAVITY[1] = w.slider_float("y", GRAVITY[1], -10, 10)
        GRAVITY[2] = w.slider_float("z", GRAVITY[2], -10, 10)

    with gui.sub_window("Presets", 0., 0.45, 0.2, 0.1) as w:
        old_preset = curr_preset_id
        for i in range(len(presets)):
            if w.checkbox(preset_names[i], curr_preset_id == i):
                curr_preset_id = i
        if curr_preset_id != old_preset:
            init()


def render():
    camera.track_user_inputs(window, movement_speed=0.03, hold_key=ti.ui.RMB)
    scene.set_camera(camera)

    scene.ambient_light((0, 0, 0))

    colors_used = F_colors_random if use_random_colors else F_colors
    scene.particles(F_x, per_vertex_color=colors_used, radius=particles_radius)

    scene.point_light(pos=(0.5, 1.5, 0.5), color=(0.5, 0.5, 0.5))
    scene.point_light(pos=(0.5, 1.5, 1.5), color=(0.5, 0.5, 0.5))

    canvas.scene(scene)


def main():
    frame_id = 0

    while window.running:
        frame_id += 1
        frame_id = frame_id % 256

        if not paused:
            for _ in range(steps):
                substep(*dt, *GRAVITY, *Lame, *plasticity_boundary, *H, is_elastic_object)

        render()
        show_options()
        window.show()


if __name__ == "__main__":
    main()
