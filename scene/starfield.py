import taichi as ti
import math
import random

@ti.data_oriented
class StarField:
    def __init__(self, n_stars=50, min_dist=0.5, max_dist=5.0, max_radius=0.1):
        self.n_stars = n_stars
        self.min_dist = min_dist      # distance minimale au trou noir situé en (0,0,0)
        self.max_dist = max_dist      # distance maximale
        self.max_radius = max_radius

        # Champs Taichi
        self.centers = ti.Vector.field(3, dtype=ti.f32, shape=n_stars)
        self.radii = ti.field(dtype=ti.f32, shape=n_stars)
        self.colors = ti.Vector.field(3, dtype=ti.f32, shape=n_stars)

        self.generate_stars()

    def generate_stars(self):
        """ Génère des étoiles 3D.

        - position aléatoire dans une sphère
        - rayon variable
        - couleur basée sur la température (blanc froid -> blanc chaud)
        - évite la zone autour du trou noir
        """

        for i in range(self.n_stars):
            # Trouver une position valide (pas trop proche du trou noir)
            while True:
                # Position aléatoire dans une sphère
                x = random.uniform(-1, 1)
                y = random.uniform(-1, 1)
                z = random.uniform(-1, 1)
                p = ti.Vector([x, y, z])

                dist = math.sqrt(x*x + y*y + z*z)

                # Normaliser et étendre dans [min_dist, max_dist]
                if dist > 0.001:  # éviter division par zéro
                    p = p / dist
                    d = random.uniform(self.min_dist, self.max_dist)
                    p = p * d

                    # Zone d'exclusion autour du trou noir
                    if d > self.min_dist:
                        break

            # Rayon variable 
            r = random.random() ** 5 * self.max_radius

            # Température couleur (blanc froid -> blanc chaud)
            temp = random.random()
            color = (
                1.0,
                0.8 + 0.2 * temp,
                0.7 + 0.3 * temp
            )

            # Stockage dans les champs Taichi
            self.centers[i] = p
            self.radii[i] = r
            self.colors[i] = ti.Vector(color)