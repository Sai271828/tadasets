import numpy as np
from .dimension import embed
from .rotate import rotate_2D
from typing import Optional

__all__ = ["torus", "dsphere", "sphere", "genusg_surface", "swiss_roll", "infty_sign", "eyeglasses"]


# TODO: Make a base class that controls `ambient` and `noise`.
# class Shape:
# def __init__(self):
# pass


def dsphere(
    n: int = 100,
    d: int = 2,
    r: float = 1,
    noise: Optional[float] = None,
    ambient: Optional[int] = None,
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Sample ``n`` data points on a ``d``-sphere.

    Parameters
    -----------
    n : int, default=100
        Number of data points in shape.
    d : int, default=2
        Intrinsic dimension of ``d``-sphere.
    r : float, default=1
        Radius of sphere.
    noise : float, optional
        Standard deviation of normally distributed noise added to data.
    ambient : int, optional
        Embed the sphere into a space with ambient dimension equal to `ambient`. The sphere is randomly rotated in this high dimensional space.
    seed : int, optional
        Seed for random state.

    Returns
    -------
    data : np.ndarray
        An ``(n,ambient)`` np.ndarray if ``ambient`` is specifed or
        a ``(n,d+1)`` np.ndarray otherwise.
    """
    rng = np.random.default_rng(seed)
    data = rng.standard_normal((n, d + 1))

    # Normalize points to the sphere
    data = r * data / np.sqrt(np.sum(data**2, 1)[:, None])

    if noise:
        data += noise * rng.standard_normal(data.shape)

    if ambient:
        assert ambient > d, "Must embed in higher dimensions"
        data = embed(data, ambient)

    return data


def sphere(
    n: int = 100,
    r: float = 1.0,
    noise: Optional[float] = None,
    ambient: Optional[int] = None,
    seed: Optional[int] = None,
    uniform: bool = False,
) -> np.ndarray:
    """
        Sample ``n`` data points on a sphere.

    Parameters
    -----------
    n : int, default=100
        Number of data points in shape.
    r : float, default=1
        Radius of sphere.
    noise : float, optional
        Standard deviation of normally distributed noise added to data.
    ambient : int, optional
        Embed the sphere into a space with ambient dimension equal to `ambient`. The sphere is randomly rotated in this high dimensional space.
    seed : int, optional
        Seed for random state.
    uniform : bool, default=False
        If True, sample points uniformly on the sphere. If False, sample points by choosing spherical coordinates uniformly.

    Returns
    -------
    data : np.ndarray
        An ``(n,3)`` np.ndarray.
    """

    rng = np.random.default_rng(seed)

    if uniform:
        data = rng.standard_normal((n, 3))
        data = r * data / np.sqrt(np.sum(data**2, 1)[:, None])

    else:
        theta = rng.random(n) * 2.0 * np.pi
        phi = rng.random(n) * np.pi
        rad = np.ones((n,)) * r

        data = np.zeros((n, 3))

        data[:, 0] = rad * np.cos(theta) * np.cos(phi)
        data[:, 1] = rad * np.cos(theta) * np.sin(phi)
        data[:, 2] = rad * np.sin(theta)

    if noise:
        data += noise * rng.standard_normal(data.shape)

    if ambient:
        data = embed(data, ambient)

    return data


def torus(
    n: int = 100,
    c: float = 2.0,
    a: float = 1.0,
    noise: Optional[float] = None,
    ambient: Optional[int] = None,
    seed: Optional[int] = None,
    uniform: bool = False,
) -> np.ndarray:
    """
    Sample ``n`` data points on a torus.

    Parameters
    -----------
    n : int, default=100
        Number of data points in shape.
    c : float, default=2.0
        Distance from center to center of tube.
    a : float, default=1.0
        Radius of tube.
    noise: float, optional
        Standard deviation of normally distributed noise added to data.
    ambient : int, optional
        Embed the torus into a space with ambient dimension equal to `ambient`. The torus is randomly rotated in this high dimensional space.
    seed : int, optional
        Seed for random state.
    uniform : bool, default=False
        If True, sample points uniformly on the torus. If False, sample points by choosing angles uniformly.

    Returns
    -------
    data : np.ndarray
        An ``(n,3)`` np.ndarray.
    """

    assert a <= c, "That's not a torus"

    rng = np.random.default_rng(seed)
    if uniform:
        samples = []
        while len(samples) < n:
            u = np.random.uniform(0, 2 * np.pi)
            v = np.random.uniform(0, 2 * np.pi)
            x = (c + a * np.cos(v)) * np.cos(u)
            y = (c + a * np.cos(v)) * np.sin(u)
            z = a * np.sin(v)

            # REJECTION SAMPLING TO ENSURE UNIFORMITY
            # The map from u,v to x,y,z is not area-preserving, so we use rejection sampling to ensure uniformity
            jacobian = a * (c + a * np.cos(v))
            acceptance_prob = jacobian / (a * (c + a))
            if np.random.uniform(0, 1) < acceptance_prob:
                samples.append((x, y, z))
        data = np.array(samples)

    else:
        theta = rng.random(n) * 2.0 * np.pi
        phi = rng.random(n) * 2.0 * np.pi

        data = np.zeros((n, 3))
        data[:, 0] = (c + a * np.cos(theta)) * np.cos(phi)
        data[:, 1] = (c + a * np.cos(theta)) * np.sin(phi)
        data[:, 2] = a * np.sin(theta)

    if noise:
        data += noise * rng.standard_normal(data.shape)

    if ambient:
        data = embed(data, ambient)

    return data


def genusg_surface(
    n: int = 100,
    genus: int = 2,
    c: float = 2.0,
    a: float = 1.0,
    noise: Optional[float] = None,
    ambient: Optional[int] = None,
    seed: Optional[int] = None,
    uniform: bool = False,
) -> np.ndarray:
    """
    Sample `n` data points from a genus-g surface. The genus is assumed to be at least 2.

    Parameters
    ----------
    n : int, default=100
        Number of data points in shape.
    genus : int, default=2
        Genus of the surface.
    c : float, default=2.0
        Distance from center to center of tube.
    a : float, default=1.0
        Radius of tube.
    noise: float, optional
        Standard deviation of normally distributed noise added to data.
    ambient : int, optional
        Embed the surface into a space with ambient dimension equal to `ambient`. The surface is randomly rotated in this high dimensional space.
    seed : int, optional
        Seed for random state.
    uniform : bool, default=False
        If True, sample points uniformly on the surface.

    Returns
    -------
    data : np.ndarray
        An ``(n,3)`` np.ndarray.
    """
    rng = np.random.default_rng(seed)
    # Sample points on the surface

    assert genus >= 2, "Genus should be at least 2"
    assert a <= c, "The major radius should be larger than the minor radius"
    # Generate the left most donut
    # For uniform sampling, we can use rejection sampling as done for genus 1

    n_samples_per_genus = n // genus

    left_donut = []
    while len(left_donut) < n_samples_per_genus + n - n_samples_per_genus * genus:
        u = np.random.uniform(0, 2 * np.pi)
        v = np.random.uniform(0, 2 * np.pi)
        x = (c + a * np.cos(v)) * np.cos(u)
        y = (c + a * np.cos(v)) * np.sin(u)
        z = a * np.sin(v)
        if x < c + 3 * a / 4:
            if uniform:
                # REJECTION SAMPLING TO ENSURE UNIFORMITY
                # The map from u,v to x,y,z is not area-preserving, so we use rejection sampling to ensure uniformity
                jacobian = a * (c + a * np.cos(v))
                acceptance_prob = jacobian / (a * (c + a))
                if np.random.uniform(0, 1) < acceptance_prob:
                    left_donut.append((x, y, z))

            else:
                left_donut.append((x, y, z))
    left_donut = np.array(left_donut)

    # Generate the right most donut
    right_donut = []
    while len(right_donut) < n_samples_per_genus:
        u = np.random.uniform(0, 2 * np.pi)
        v = np.random.uniform(0, 2 * np.pi)
        x = (c + a * np.cos(v)) * np.cos(u)
        y = (c + a * np.cos(v)) * np.sin(u)
        z = a * np.sin(v)
        if x > a / 4 - c - a:
            if uniform:
                # REJECTION SAMPLING TO ENSURE UNIFORMITY
                # The map from u,v to x,y,z is not area-preserving, so we use rejection sampling to ensure uniformity
                jacobian = a * (c + a * np.cos(v))
                acceptance_prob = jacobian / (a * (c + a))
                if np.random.uniform(0, 1) < acceptance_prob:
                    right_donut.append((x, y, z))

            else:
                right_donut.append((x, y, z))
    right_donut = np.array(right_donut)

    # Generate middle donuts
    middle_donuts = []
    while len(middle_donuts) < n_samples_per_genus:
        u = np.random.uniform(0, 2 * np.pi)
        v = np.random.uniform(0, 2 * np.pi)
        x = (c + a * np.cos(v)) * np.cos(u)
        y = (c + a * np.cos(v)) * np.sin(u)
        z = a * np.sin(v)
        if (x > a / 4 - c - a) & (x < c + 3 * a / 4):
            if uniform:
                # REJECTION SAMPLING TO ENSURE UNIFORMITY
                # The map from u,v to x,y,z is not area-preserving, so we use rejection sampling to ensure uniformity
                jacobian = a * (c + a * np.cos(v))
                acceptance_prob = jacobian / (a * (c + a))
                if np.random.uniform(0, 1) < acceptance_prob:
                    middle_donuts.append((x, y, z))

            else:
                middle_donuts.append((x, y, z))
    middle_donuts = np.array(middle_donuts)

    # Combine all points
    # If genus is 2, we have only left and right donuts
    if genus == 2:
        # MOVE RIGHT DONUT TO THE RIGHT
        right_donut[:, 0] += 2 * c + 3 * a / 2
        data = np.vstack((left_donut, right_donut))
    else:
        # Move middle donuts to the right positions
        move_distances = [(2 * c + 3 * a / 2) * (i + 1) for i in range(genus - 1)]
        for i in range(0, genus - 2):
            temp_donut = middle_donuts.copy()
            temp_donut[:, 0] += move_distances[i]
            left_donut = np.vstack((left_donut, temp_donut))
        # Move right donut to the rightmost position
        right_donut[:, 0] += move_distances[-1]
        data = np.vstack((left_donut, right_donut))

    if noise:
        data += noise * rng.standard_normal(data.shape)

    if ambient:
        data = embed(data, ambient)

    return data


def swiss_roll(
    n: int = 100,
    r: float = 10.0,
    noise: Optional[float] = None,
    ambient: Optional[int] = None,
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Sample `n` data points from a Swiss roll.

    Parameters
    ----------
    n : int, default=100
        Number of data points in shape.
    r : float, default=10.0
        Length of roll.
    noise: float, optional
        Standard deviation of normally distributed noise added to data.
    ambient : int, optional
        Embed the swiss roll into a space with ambient dimension equal to `ambient`. The swiss roll is randomly rotated in this high dimensional space.
    seed : int, optional
        Seed for random state.

    References
    ----------
    Equations mimic `Swiss Roll and SNE by jlmelville https://jlmelville.github.io/smallvis/swisssne.html`_.

    Returns
    -------
    data : np.ndarray
        An ``(n,3)`` np.ndarray.
    """

    rng = np.random.default_rng(seed)
    phi = (rng.random(n) * 3 + 1.5) * np.pi
    psi = rng.random(n) * r

    data = np.zeros((n, 3))
    data[:, 0] = phi * np.cos(phi)
    data[:, 1] = phi * np.sin(phi)
    data[:, 2] = psi

    if noise:
        data += noise * rng.standard_normal(data.shape)

    if ambient:
        data = embed(data, ambient)

    return data


def infty_sign(
    n: int = 100,
    noise: Optional[float] = None,
    angle: Optional[float] = None,
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Construct a figure 8 or infinity sign with ``n`` points and noise level with ``noise`` standard deviation.

    Parameters
    ----------

    n: int, default=100
        Number of data points in shape.
    noise: float, optional
        Standard deviation of normally distributed noise added to data.
    angle: float, optional
        Angle in radians to rotate the infinity sign.
    seed : int, optional
        Seed for random state.

    Returns
    -------
    data : np.ndarray
        An ``(n,2)`` np.ndarray.
    """
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 2 * np.pi, n + 1)[0:n]
    X = np.zeros((n, 2))
    X[:, 0] = np.cos(t)
    X[:, 1] = np.sin(2 * t)

    if noise:
        X += noise * rng.standard_normal((n, 2))
    if angle is not None:
        assert angle >= -np.pi and angle <= 2 * np.pi, (
            "Angle {angle} not in range. Angle should be in the range {min_angle} <= angle <= {max_angle}".format(
                angle=angle, min_angle="-pi", max_angle="2*pi"
            )
        )

        X = rotate_2D(X, angle=angle)
    return X


def eyeglasses(
    n: int = 100,
    r1: float = 1.0,
    r2: Optional[float] = None,
    neck_size: Optional[float] = None,
    noise: Optional[float] = None,
    ambient: Optional[int] = None,
    seed: Optional[int] = None,
) -> np.ndarray:
    """Sample ``n`` points on an eyeglasses shape.

    Parameters
    ----------
    n : int, default=100
        Number of points in shape
    r1 : float, default=1.0
        The radius of the left half of the eyeglasses shape
    r2 : float, optional
        The radius of the right half. If None, it is equal to `r1`.
    neck_size : float, optional
        The width of the neck.
    noise : float, optional
        Standard deviation of normally distributed noise added to data.
    ambient : int, optional
        Embed the eyeglasses shape into a space with ambient dimension equal to `ambient`.
        The eyeglasses shape is randomly rotated in this high dimensional space.
    seed : int, optional
        Seed for random state.

    Returns
    -------
    data : np.ndarray
        An ``(n,ambient)`` np.ndarray if ``ambient`` is specified or
        an ``(n,2)`` np.ndarray otherwise.
    """
    rng = np.random.default_rng(seed)

    if r2 is None:
        r2 = r1
    if neck_size is None:
        neck_size = min(r1, r2)

    assert neck_size < 2 * min(r1, r2), "Neck should be smaller than 2*min(r1,r2)."
    half_neck = neck_size / 2

    r_neck = min(r1, r2)

    x_left = ((r_neck + r1) ** 2 - (r_neck + half_neck) ** 2) ** (
        1 / 2
    ) - r1  # x distance from left circle to neck
    x_right = ((r_neck + r2) ** 2 - (r_neck + half_neck) ** 2) ** (1 / 2) - r2

    alpha_1 = np.arcsin((r_neck + half_neck) / (r1 + r_neck))
    alpha_2 = np.arcsin((r_neck + half_neck) / (r2 + r_neck))

    centers = {
        "l": [-r1 - x_left, 0],
        "r": [x_right + r2, 0],
        "t": [0, half_neck + r_neck],
        "b": [0, -half_neck - r_neck],
    }

    radii = {"l": r1, "r": r2, "t": r_neck, "b": r_neck}

    angle_ranges = {
        "l": [alpha_1, 2 * np.pi - alpha_1],
        "r": [-np.pi + alpha_2, np.pi - alpha_2],
        "t": [np.pi + alpha_1, 2 * np.pi - alpha_2],
        "b": [alpha_2, np.pi - alpha_1],
    }

    # compute how many points each circle will contain.
    arc_lens = {
        x: (angle_ranges[x][1] - angle_ranges[x][0]) * radii[x]
        for x in ["l", "r", "t", "b"]
    }
    total_arc_len = sum(arc_lens.values())
    probabilities = {x: arc_lens[x] / total_arc_len for x in ["l", "r", "t", "b"]}
    positions = np.random.choice(
        ["l", "r", "t", "b"], p=list(probabilities.values()), size=n
    )
    positions, counts = np.unique(positions, return_counts=True)
    counts = {x: c for x, c in zip(positions, counts)}

    data = []
    for x in ["l", "r", "t", "b"]:
        angles = rng.uniform(angle_ranges[x][0], angle_ranges[x][1], counts[x])
        points = (
            np.vstack((radii[x] * np.cos(angles), radii[x] * np.sin(angles))).T
            + centers[x]
        )
        data.append(points)

    data = np.concatenate(data)

    if noise:
        data += noise * rng.standard_normal((n, 2))

    if ambient:
        data = embed(data, ambient)

    return data
