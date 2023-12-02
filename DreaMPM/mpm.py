import taichi as ti

from .objects import CubeObject
from .materials import DIY_MATERIAL, SOLID_CUBE

ti.init(arch=ti.gpu)

"""
An implementation of MPM using Corotated model, APIC and semi-implicit advection
"""


@ti.data_oriented
class MPM:
    def __init__(self, G_number, max_hard):
        self.G_number = G_number
        self.max_hard = max_hard
        self.dim = 3
        self.P_number = self.G_number ** self.dim // 2 ** (self.dim - 1)  # 65536 or 8192
        print("particle number:", self.P_number)  # you can check the particle number right here

        # time step
        self.dt = [2e-4]

        self.dx = 1 / G_number
        p_rho = 1  # 1
        self.p_vol = (self.dx * 0.5) ** 2
        self.p_mass = self.p_vol * p_rho
        self.bound = 3
        self.V_parameter = 0.5
        self.F_x = ti.Vector.field(self.dim, float, self.P_number)
        self.F_v = ti.Vector.field(self.dim, float, self.P_number)
        self.C = ti.Matrix.field(self.dim, self.dim, float, self.P_number)
        self.def_grad = ti.Matrix.field(3, 3, dtype=float, shape=self.P_number)  # deformation gradient
        self.F_Jp = ti.field(float,
                             self.P_number)  # the volumetric rate of Plasticity change which will change the Hardening coefficient

        self.F_colors = ti.Vector.field(4, float, self.P_number)
        self.F_colors_random = ti.Vector.field(4, float, self.P_number)
        self.F_materials = ti.field(int, self.P_number)
        self.F_grid_v = ti.Vector.field(self.dim, float, (G_number,) * self.dim)
        self.F_grid_m = ti.field(float, (G_number,) * self.dim)
        self.F_used = ti.field(int, self.P_number)

        self.neighbour = (3,) * self.dim

    @ti.kernel
    def init_cube_object(
            self,
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
            self.F_x[i] = ti.Vector([ti.random() for i in range(self.dim)]) * ti.Vector(
                [x_size, y_size, z_size]) + ti.Vector(
                [x_begin, y_begin, z_begin]
            )
            self.F_Jp[i] = 1
            self.def_grad[i] = ti.Matrix([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
            self.F_v[i] = ti.Vector([0.0, 0.0, 0.0])
            self.F_materials[i] = material
            self.F_colors_random[i] = ti.Vector([ti.random(), ti.random(), ti.random(), ti.random()])
            self.F_used[i] = 1

    @ti.kernel
    def set_all_unused(self):
        for p in self.F_used:
            self.F_used[p] = 0
            # placing in a very far place
            self.F_x[p] = ti.Vector([90000.0, 90000.0, 90000.0])
            self.F_Jp[p] = 1
            self.def_grad[p] = ti.Matrix([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
            self.C[p] = ti.Matrix([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
            self.F_v[p] = ti.Vector([0.0, 0.0, 0.0])

    @ti.kernel
    def reset(self):
        for p in self.F_used:
            self.F_used[p] = 1
            self.F_Jp[p] = 1
            self.def_grad[p] = ti.Matrix([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
            self.C[p] = ti.Matrix([[0, 0, 0], [0, 0, 0], [0, 0, 0]])

    def init_objs(self, presets):
        objects = presets.objects
        self.set_all_unused()
        total_vol = 0
        for obj in objects:
            total_vol += obj.volume

        next_p = 0
        for i, obj in enumerate(objects):
            obj = objects[i]
            if isinstance(obj, CubeObject):
                par_count = int(obj.volume / total_vol * self.P_number)
                if i == len(objects) - 1:  # this is the last volume, so use all remaining particles
                    par_count = self.P_number - next_p
                self.init_cube_object(next_p, next_p + par_count, *obj.minimum, *obj.size, obj.material.id)
                next_p += par_count
            else:
                raise Exception("???")

    @ti.kernel
    def set_color_by_material(self, mat_color: ti.types.ndarray()):
        for i in range(self.P_number):
            m_id = self.F_materials[i]
            self.F_colors[i] = ti.Vector(
                [mat_color[m_id, 0], mat_color[m_id, 1], mat_color[m_id, 2], mat_color[m_id, 3]])

    @ti.kernel
    def substep(self, dt: float, g_x: float, g_y: float, g_z: float,
                mu_0: float, lambda_0: float,
                plasticity_lower_bound: float, plasticity_upper_bound: float,
                H: float, is_elastic_object: bool):
        lower = 1 + plasticity_lower_bound / 10000
        upper = 1 + plasticity_upper_bound / 10000

        for I in ti.grouped(self.F_grid_m):
            self.F_grid_v[I] = ti.zero(self.F_grid_v[I])
            self.F_grid_m[I] = 0
        ti.loop_config(block_dim=self.G_number)
        # P2G
        for p in self.F_x:
            if self.F_used[p] == 0:
                continue
            Xp = self.F_x[p] / self.dx
            base = int(Xp - 0.5)
            fx = Xp - base
            w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1) ** 2, 0.5 * (fx - 0.5) ** 2]

            # Deformation gradient update moved to the front (elastic deformation) (hack!)
            self.def_grad[p] = (ti.Matrix.identity(float, 3) + dt * self.C[p]) @ self.def_grad[
                p]  # deformation gradient update

            # Hardening coefficient: material harder when compressed
            h = ti.exp(10 * (1.0 - self.F_Jp[p]))  # Plastic change effect the h, or the h is always 1

            if is_elastic_object:  # elastic object, make it softer
                h = H  # 0.1 ~ 1.0

            mu, la = mu_0 * h, lambda_0 * h

            if self.F_materials[p] == SOLID_CUBE:
                h = self.max_hard
                mu, la = 416.6 * h, 277.7 * h

            U, sig, V = ti.svd(self.def_grad[p])
            # the deformation rate (1dim)
            J = 1.0
            # calculate the J and update sig
            for d in ti.static(range(3)):

                orgin_sig = sig[d, d]  # singular values
                new_sig = orgin_sig
                if self.F_materials[p] == DIY_MATERIAL and not is_elastic_object:  # DIY_MATERIAL with Plasticity
                    new_sig = ti.min(
                        ti.max(
                            orgin_sig,
                            lower  # default: 1 - 2.5e-2
                        ),
                        upper  # default: 1 + 4.5e-3
                    )  # Plasticity: Forget too much deformation (hack)

                self.F_Jp[p] *= orgin_sig / new_sig  # volumetric rate of Plasticity change
                sig[d, d] = new_sig
                J *= new_sig

            if mu < 1.0:
                # if mu is too small, reset deformation gradient to avoid numerical instability
                new_F = ti.Matrix.identity(float, 3)
                new_F[0, 0] = J
                self.def_grad[p] = new_F
            else:
                # Reconstruct remain elastic deformation gradient after plasticity
                self.def_grad[p] = U @ sig @ V.transpose()

            # Corotated used elastic F_dg
            # stress_part1
            # = P(F) * F.transposed()
            # = (2µ(F − R) + λ(J − 1)J * F.transposed().inverse()) * F.transposed()
            # = 2µ(F − R)* F.transposed() + λ(J − 1)J
            stress_part1 = (
                    2 * mu * (self.def_grad[p] - U @ V.transpose()) @ self.def_grad[p].transpose() +
                    ti.Matrix.identity(
                        float, 3
                    ) * la * J * (J - 1)
            )
            # stress
            # = 4*∆t / ∆x**2 * V * P(F) * F.transposed()
            # = 4*∆t / ∆x**2 * V * stress_part1
            stress = (-dt * self.p_vol * 4) * stress_part1 / self.dx ** 2
            affine = self.p_mass * self.C[p] + stress

            # 3d Grid momentum
            for offset in ti.static(ti.grouped(ti.ndrange(*self.neighbour))):
                dpos = (offset - fx) * self.dx
                weight = 1.0
                for i in ti.static(range(self.dim)):
                    weight *= w[offset[i]][i]
                self.F_grid_v[base + offset] += weight * (self.p_mass * self.F_v[p] + affine @ dpos)
                self.F_grid_m[base + offset] += weight * self.p_mass

        for I in ti.grouped(self.F_grid_m):
            if self.F_grid_m[I] > 0:
                self.F_grid_v[I] /= self.F_grid_m[I]

            self.F_grid_v[I] += dt * ti.Vector([g_x, g_y, g_z])  # 3d gravity

            cond = (I < self.bound) & (self.F_grid_v[I] < 0) | (I > self.G_number - self.bound) & (
                    self.F_grid_v[I] > 0)  # boundary condition

            self.F_grid_v[I] = ti.select(cond, 0, self.F_grid_v[I])
        ti.loop_config(block_dim=self.G_number)
        # G2P
        for p in self.F_x:
            if self.F_used[p] == 0:
                continue
            Xp = self.F_x[p] / self.dx
            base = int(Xp - 0.5)
            fx = Xp - base
            w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1) ** 2, 0.5 * (fx - 0.5) ** 2]
            new_v = ti.zero(self.F_v[p])
            new_C = ti.zero(self.C[p])
            for offset in ti.static(ti.grouped(ti.ndrange(*self.neighbour))):
                dpos = (offset - fx) * self.dx
                weight = 1.0
                for i in ti.static(range(self.dim)):
                    weight *= w[offset[i]][i]
                g_v = self.F_grid_v[base + offset]
                new_v += weight * g_v
                new_C += 4 * weight * g_v.outer_product(dpos) / self.dx ** 2  # velocity gradient (APIC)
            # semi-implicit advection on particle
            self.F_v[p] = new_v
            self.F_x[p] += dt * self.F_v[p]
            self.C[p] = new_C
