import random

import numpy as np

from .singularity import singularity


def random_polar():
    theta = random() * 2 * np.pi
    r = random()
    return r * np.cos(theta), r * np.sin(theta)


def xyz(radius: float, show: bool = False):
    theta = random.random() * 2 * np.pi
    r = random.random() * radius
    new_xyz = np.array([r * np.cos(theta), r * np.sin(theta), 0])

    if show:
        print(f"Случайная точка: {new_xyz} мм")
    return new_xyz


def angle(limit: list, show: bool = False):
    angle = np.zeros(3)
    angle[-1] = random.uniform(*limit)
    return angle


def target(radius, limit=[0, 180]):
    return xyz(radius=radius), angle(limit=limit)


def is_unique(array_list, new_array):
    return not any(np.array_equal(existing, new_array) for existing in array_list)


def points(radius=500, limit=[0, 180], R=100, r=25, n=100):
    unique_coords = []
    unique_angles = []

    while len(unique_coords) < n or len(unique_angles) < n:
        coord, angle = target(radius=radius, limit=limit)
        if is_unique(unique_coords, coord):
            unique_coords.append(coord)
        if is_unique(unique_angles, angle):
            unique_angles.append(angle)

        if singularity(*coord[:2], fi=angle[2], R=R, r=r):
            yield coord, angle


if __name__ == "__main__":
    for _ in range(1000):
        xyz(radius=56, show=True)
        angle(limit=[-25, 25], show=True)

    print(target(radius=56, limit=[0, 180]))
