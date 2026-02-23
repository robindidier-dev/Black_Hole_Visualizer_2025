import taichi as ti

@ti.func
def sdf_sphere(p, center, radius):
    """ Calcule la distance signée d'un point p à la surface d'une sphère. """
    return (p - center).norm() - radius


@ti.func
def estimate_normal(p):
    """ Estime la normale à la surface d'une sphère de rayon 0.7 centrée en (0, 0, 0). """
    eps = 1e-3
    dx = sdf_sphere(p + ti.Vector([eps, 0, 0]), ti.Vector([0, 0, 0]), 0.7) - sdf_sphere(p - ti.Vector([eps, 0, 0]), ti.Vector([0, 0, 0]), 0.7)
    dy = sdf_sphere(p + ti.Vector([0, eps, 0]), ti.Vector([0, 0, 0]), 0.7) - sdf_sphere(p - ti.Vector([0, eps, 0]), ti.Vector([0, 0, 0]), 0.7)
    dz = sdf_sphere(p + ti.Vector([0, 0, eps]), ti.Vector([0, 0, 0]), 0.7) - sdf_sphere(p - ti.Vector([0, 0, eps]), ti.Vector([0, 0, 0]), 0.7)
    return ti.Vector([dx, dy, dz]).normalized()

@ti.kernel
def render_raymarch(pixels: ti.template(), width: int, height: int): # type: ignore
    """ Effectue le ray marching pour chaque pixel de l'image. """
    for i, j in pixels:
        # Coordonnées normalisées
        u = (i / width) * 2 - 1
        v = (j / height) * 2 - 1

        # Caméra définie par sa position et sa direction
        ro = ti.Vector([0.0, 0.0, 2.5])  
        rd = ti.Vector([u, v, -1.0]).normalized()

        # Ray marching
        t = 0.0
        hit = False
        for _ in range(100):  # nombre max de steps
            p = ro + rd * t
            d = sdf_sphere(p, ti.Vector([0, 0, 0]), 0.7)
            if d < 0.001:
                hit = True
                break
            t += d
            if t > 10.0:
                break

        if hit:
            p = ro + rd * t
            normal = estimate_normal(p)
            color = (normal + 1) * 0.5 # normalisation pour obtenir des couleurs entre 0 et 1
            pixels[i, j] = color
            # Bleu -> face à la caméra 
            # Vert -> à droite de la sphère 
            # Rouge -> en bas de la sphère

        else:
            pixels[i, j] = ti.Vector([0.0, 0.0, 0.0])