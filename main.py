import taichi as ti
import imageio
import numpy as np
from renderer.raymarcher import render_raymarch


ti.init(arch=ti.vulkan)

width, height = 800, 800
pixels = ti.Vector.field(3, dtype=ti.f32, shape=(width, height))

render_raymarch(pixels, width, height)

img = pixels.to_numpy()
img_uint8 = (img * 255).astype(np.uint8)
imageio.imwrite("images/sphere_raymarch.png", img_uint8)

print("Ray marcher OK.")
