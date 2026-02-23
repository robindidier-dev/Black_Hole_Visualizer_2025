import taichi as ti
import imageio
import numpy as np

from scene.starfield import StarField
from renderer.raymarcher import RayMarcher

ti.init(arch=ti.vulkan)

# Résolution
width, height = 800, 800

# Création du champ d'étoiles
starfield = StarField(
    n_stars=300,
    min_dist=1,   # zone d'exclusion autour du trou noir
    max_dist=6.0,
    max_radius=0.04
)

# Création du ray marcher
rm = RayMarcher(width, height)

# Paramètres de la scène
black_hole_radius = 1.3

# Rendu
rm.render(
    starfield.centers,
    starfield.radii,
    starfield.colors,
    starfield.n_stars,
    black_hole_radius
)

# Export image
img = rm.pixels.to_numpy()
img_uint8 = (img * 255).astype(np.uint8)
imageio.imwrite("images/scene_starfield.png", img_uint8)

print("Ray marcher OK.")