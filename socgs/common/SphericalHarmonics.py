# %%
import taichi as ti
import taichi.math as tm

# %%
vec3f = ti.types.vector(3, float)
vec5f = ti.types.vector(5, float)
vec7f = ti.types.vector(7, float)
vec16f = ti.types.vector(16, float)

# Spherical Harmonic
C0 = 0.28209479177387814

C1 = 0.4886025119029199

C21 = 1.0925484305920792
C22 = -1.0925484305920792
C23 = 0.31539156525252005
C24 = -1.0925484305920792
C25 = 0.54627421529603959

C31 = -0.59004358992664352
C32 = 2.8906114426405538
C33 = -0.45704579946446572
C34 = 0.3731763325901154
C35 = -0.45704579946446572
C36 = 1.4453057213202769
C37 = -0.59004358992664352


@ti.func
def get_spherical_harmonic_from_xyz(
        xyz: tm.vec3
):
    xyz = tm.normalize(xyz)
    x, y, z = xyz.x, xyz.y, xyz.z
    l0m0 = C0
    l1m1 = -C1 * y
    l1m0 = C1 * z
    l1p1 = -C1 * x
    l2m2 = C21 * x * y
    l2m1 = C22 * y * z
    l2m0 = 3.0 * C23 * z * z - C23
    l2p1 = C24 * x * z
    l2p2 = C25 * x * x - C25 * y * y
    l3m3 = -C31 * y * (-3.0 * x * x + y * y)
    l3m2 = C32 * x * y * z
    l3m1 = -C33 * y * (1.0 - 5.0 * z * z)
    l3m0 = C34 * z * (5.0 * z * z - 3.0)
    l3p1 = -C35 * x * (1.0 - 5.0 * z * z)
    l3p2 = C36 * z * (x * x - y * y)
    l3p3 = -C37 * x * (-x * x + 3.0 * y * y)
    return vec16f(l0m0, l1m1, l1m0, l1p1, l2m2, l2m1, l2m0, l2p1, l2p2, l3m3, l3m2, l3m1, l3m0, l3p1, l3p2, l3p3)


@ti.dataclass
class SphericalHarmonics:
    factor: vec16f

    @ti.func
    def evaluate(
            self,
            xyz: tm.vec3
    ) -> ti.float32:
        spherical_harmonic = get_spherical_harmonic_from_xyz(xyz)
        return tm.dot(self.factor, spherical_harmonic)

    @ti.func
    def evaluate_with_jacobian(
            self,
            xyz: tm.vec3
    ):
        spherical_harmonic = get_spherical_harmonic_from_xyz(xyz)
        return tm.dot(self.factor, spherical_harmonic), spherical_harmonic

    @ti.func
    def evaluate_with_direction_jacobian(
            self,
            xyz: tm.vec3
    ):
        rx, ry, rz = xyz.x, xyz.y, xyz.z
        spherical_harmonic = get_spherical_harmonic_from_xyz(xyz)
        norm_xyz = tm.normalize(xyz)
        nx, ny, nz = norm_xyz.s, norm_xyz.y, norm_xyz.z
        d_ci_d_normx = -C1 * self.factor[3] + C21 * ny * self.factor[4] + C24 * nz * self.factor[7] + \
                    2.0 * C25 * nx * self.factor[8] + 3.0 * C31 * 2.0 * nx * ny * self.factor[9] + \
                    C32 * ny * nz * self.factor[10] - C35 * self.factor[13] + \
                    C36 * 2.0 * nx * nz * self.factor[14] + C37 * 3.0 * nx * nx * self.factor[15] - \
                    3.0 * C37 * ny * ny * self.factor[15]
        d_ci_d_normy = -C1 * self.factor[1] + C21 * nx * self.factor[4] + C22 * nz * self.factor[5] - \
                    2.0 * C25 * ny * self.factor[8] + (3.0 * C31 * nx * nx - 3.0 * C31 * ny * ny) * self.factor[9] + \
                    C32 * nx * nz * self.factor[10] - C33 * self.factor[11] - 2.0 * C36 * ny * nz * self.factor[14] - \
                    3.0 * C37 * 2.0 * nx * ny * self.factor[15]
        d_ci_d_normz = C1 * self.factor[2] + C22 * ny * self.factor[5] + C23 * 3.0 * 2.0 * nz * self.factor[6] + \
                    C24 * nx * self.factor[7] + C32 * nx * ny * self.factor[10] + C33 * 5.0 * 2.0 * ny * nz * self.factor[11] + \
                    (C34 * 5.0 * 3.0 * nz * nz - C34 * 3.0) * self.factor[12] + \
                    C35 * 5.0 * 2.0 * nx * nz * self.factor[13] + C36 * nx * nx * self.factor[14]

        d_ci_d_norm_xyz = vec3f(d_ci_d_normx, d_ci_d_normy, d_ci_d_normz)
        d_norm_xyz_d_xyz = tm.mat3([
            [(ry**2+rz**2)/((rx**2+ry**2+rz**2)**1.5), -(rx*ry)/((rx**2+ry**2+rz**2)**1.5), -(rx*rz)/((rx**2+ry**2+rz**2)**1.5)],
            [-(rx*ry)/((rx**2+ry**2+rz**2)**1.5), (rx**2+rz**2)/((rx**2+ry**2+rz**2)**1.5), -(ry*rz)/((rx**2+ry**2+rz**2)**1.5)],
            [-(rx*rz)/((rx**2+ry**2+rz**2)**1.5), -(ry*rz)/((rx**2+ry**2+rz**2)**1.5), (rx**2+ry**2)/((rx**2+ry**2+rz**2)**1.5)],
        ])
        d_ci_d_xyz = d_ci_d_norm_xyz @ d_norm_xyz_d_xyz

        return tm.dot(self.factor, spherical_harmonic), d_ci_d_xyz
