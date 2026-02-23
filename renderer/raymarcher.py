import taichi as ti
from renderer.sdf import scene_sdf

@ti.data_oriented
class RayMarcher:
    def __init__(self, width, height, max_steps=128, max_dist=20.0, eps=1e-3):
        self.width = width
        self.height = height
        self.max_steps = max_steps
        self.max_dist = max_dist
        self.eps = eps

        self.pixels = ti.Vector.field(3, dtype=ti.f32, shape=(width, height))

    @ti.func
    def estimate_normal(self, p, centers, radii, n_stars, black_hole_radius):
        """ Estime la normale en un point p en utilisant un gradient numérique. """
        eps = 1e-3
        dx = ti.Vector([eps, 0.0, 0.0])
        dy = ti.Vector([0.0, eps, 0.0])
        dz = ti.Vector([0.0, 0.0, eps])

        d0, _, _ = scene_sdf(p, centers, radii, n_stars, black_hole_radius)
        nx, _, _ = scene_sdf(p + dx, centers, radii, n_stars, black_hole_radius)
        ny, _, _ = scene_sdf(p + dy, centers, radii, n_stars, black_hole_radius)
        nz, _, _ = scene_sdf(p + dz, centers, radii, n_stars, black_hole_radius)

        normal = ti.Vector([nx - d0, ny - d0, nz - d0])
        return normal.normalized()

    @ti.kernel
    def render(self, centers: ti.template(), radii: ti.template(), # type: ignore
               colors: ti.template(), n_stars: int, black_hole_radius: float): # type: ignore
        """ Lance le ray marching pour chaque pixel. """
        for i, j in self.pixels:
            # Coordonnées normalisées
            u = (i / self.width) * 2.0 - 1.0
            v = (j / self.height) * 2.0 - 1.0

            # Caméra simple
            ro = ti.Vector([-0.7, 0.0, -6.0])  # position caméra
            rd = ti.Vector([u+0.15, v, 1.5]).normalized()

            t = 0.0
            hit = False
            obj_type = 0
            obj_index = -1
            hit_pos = ti.Vector([0.0, 0.0, 0.0])

            # Ray marching
            for _ in range(self.max_steps):
                p = ro + rd * t
                d, typ, idx = scene_sdf(p, centers, radii, n_stars, black_hole_radius)

                if d < self.eps:
                    hit = True
                    obj_type = typ
                    obj_index = idx
                    hit_pos = p
                    break

                if t > self.max_dist:
                    break

                t += d

            # Couleur selon l'objet touché
            if hit:
                if obj_type == 1:
                    # Trou noir : noir absolu
                    self.pixels[i, j] = ti.Vector([0.0, 0.0, 0.0])

                elif obj_type == 2:
                    # Étoile : couleur stockée
                    self.pixels[i, j] = colors[obj_index]
                
                elif obj_type == 3:
                    # Disque d'accrétion dans YZ
                    r = ti.sqrt(hit_pos.y * hit_pos.y + hit_pos.z * hit_pos.z)
                    heat = 1.0 - (r / (black_hole_radius * 4.0))
                    heat = ti.max(0.0, heat)

                    color = ti.Vector([
                        1.0,
                        0.5 + 0.5 * heat,
                        0.2 * heat
                    ])

                    self.pixels[i, j] = color


            else:
                # Fond étoilé 2D (simple bruit de Perlin)
                uv = ti.Vector([u, v])
                noise = ti.sin(123.4 * uv.dot(ti.Vector([12.9898, 78.233]))) * 43758.5453
                star = noise - ti.floor(noise)

                if star > 0.997:
                    temp = ti.sin(noise * 10.0) * 0.5 + 0.5
                    color = ti.Vector([1.0, 0.9 + 0.1 * temp, 0.8 + 0.2 * temp])
                    self.pixels[i, j] = color
                else:
                    self.pixels[i, j] = ti.Vector([0.1, 0.1, 0.1])
