import os

import numpy as np
import pandas as pd

from rpr.simulator.logic.singularity import singularity
from rpr.simulator.logic.target import is_unique, target


def L(A, r, j, x, y, fi):
    return np.sqrt(
        np.power((x + r * np.cos(np.deg2rad(fi) + np.deg2rad(j)) - A[0]), 2)
        + np.power((y + r * np.sin(np.deg2rad(fi) + np.deg2rad(j)) - A[1]), 2)
    )

R = 1000
r = 250
L_lim = [R / 10, R + (R - R / 10)]

radius = int(R/2)
fi_lim = [10, 170]

n = 1000


xq = np.sqrt(3) * R / 2  # При R = 100: xq = 86.60254037844386
yq = R / 2  # При R = 100: yq = 50.0

A_1 = {
    "A1_x": 0,
    "A1_y": R,
    "A1_z": 0,
}
A_2 = {
    "A2_x": xq,
    "A2_y": -yq,
    "A2_z": 0,
}
A_3 = {
    "A3_x": -xq,
    "A3_y": -yq,
    "A3_z": 0,
}

J = {
    "j_1": np.rad2deg(np.pi / 2),
    "j_2": np.rad2deg(-np.pi / 6),
    "j_3": np.rad2deg(-5 * np.pi / 6),
}

rd = {"r_1": r, "r_2": r, "r_3": r}

A = [A_1, A_2, A_3]


def points(
    A,
    radius=50,
    limit=[0, 180],
    R=100,
    r=25,
    n=100,
    Llim=[10, 190],
):
    unique_coord = []
    unique_angle = []
    while len(unique_coord) < n or len(unique_angle) < n:
        coord, angle = target(radius=radius, limit=limit)
        if is_unique(unique_coord, coord[:2]) and is_unique(unique_angle, angle[2]):
            Ls = {}
            Ls.update(
                {
                    f"Ld_{idx+1}": L(A=a, r=r, j=j, x=coord[0], y=coord[1], fi=angle[2])
                    for idx, (a, r, j) in enumerate(
                        zip([list(a.values()) for a in A], rd.values(), J.values())
                    )
                }
            )
            if all((i >= Llim[0] and i <= Llim[1]) for i in list(Ls.values())):
                if singularity(x=coord[0], y=coord[1], fi=angle[2], R=R, r=r):
                    unique_coord.append(coord[:2])
                    unique_angle.append(angle[2])

                    data = {}
                    for a in [A_1, A_2, A_3]:
                        data.update(a)
                    data.update(J)
                    data.update(rd)
                    data.update({"x": coord[0], "y": coord[1], "fi": angle[2]})
                    data.update(Ls)

                    yield data


def write(df, filename: str = "test.csv"):
    if not os.path.exists(filename):
        keys_df = pd.DataFrame([df.keys()], columns=data.keys())
        keys_df.to_csv(filename, header=False, index=False)
    pd.DataFrame([df]).to_csv(filename, header=False, index=False, mode="a")

for idx, data in enumerate(
    points(
        radius=radius,
        limit=fi_lim,
        R=R,
        r=r,
        n=n,
        Llim=L_lim,
        A=A,
    )
):
    write(
        data,
        filename=rf"C:\Users\nebolsinvasili\Documents\rpr\rpr\сalculate\data\test_{R}_{r}_{radius}_{fi_lim[0]}_{fi_lim[1]}_{n}.csv",
    )
