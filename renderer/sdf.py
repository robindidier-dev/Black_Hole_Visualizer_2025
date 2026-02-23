import taichi as ti

@ti.func
def sdf_sphere(p, center, radius):
    """ Calcule la distance signée d'un point p à la surface d'une sphère. """
    return (p - center).norm() - radius


@ti.func
def sdf_black_hole(p, radius):
    """ SDF du trou noir centré en (0,0,0). 
        On utilise une sphère simple pour l'horizon des événements. """
    return sdf_sphere(p, ti.Vector([0.0, 0.0, 0.0]), radius)


@ti.func
def sdf_starfield(p, centers, radii, n_stars):
    """ Calcule la distance signée au champ d'étoiles.
        Retourne la distance minimale et l'index de l'étoile la plus proche. """
    d_min = 1e9
    idx_min = -1

    for i in range(n_stars):
        d = sdf_sphere(p, centers[i], radii[i])
        if d < d_min:
            d_min = d
            idx_min = i

    return d_min, idx_min


@ti.func
def sdf_accretion_torus_yz(p, R, r):
    """ SDF d'un anneau (torus) dans le plan YZ, autour de l'axe X. """
    q = ti.Vector([ti.sqrt(p.y * p.y + p.z * p.z) - R, p.x])
    return q.norm() - r


@ti.func
def scene_sdf(p, centers, radii, n_stars, black_hole_radius):
    """ Combine les SDF de la scène :
        - trou noir
        - champ d'étoiles
        Retourne :
        - distance minimale
        - type d'objet (0 = rien, 1 = trou noir, 2 = étoile)
        - index de l'étoile si applicable
    """
    # Trou noir
    d_bh = sdf_black_hole(p, black_hole_radius)

    # Champ d'étoiles
    d_star, idx_star = sdf_starfield(p, centers, radii, n_stars)

    

    # Variables de sortie
    d = d_bh
    typ = 1
    idx = -1

    d_disk = sdf_accretion_torus_yz(
    p,
    R = black_hole_radius * 1.2,
    r = black_hole_radius * 0.05
)

    if d_disk < d:
        d = d_disk
        typ = 3
        idx = -1


    # Si une étoile est plus proche
    if d_star < d_bh:
        d = d_star
        typ = 2
        idx = idx_star

    return d, typ, idx