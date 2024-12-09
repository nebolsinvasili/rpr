import random
import time

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from utils import size

from .logic.logging import log, write
from .logic.target import points
from .modules import RPR

RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


R = 1000
r = 250
L_lim = [R / 10, R + (R - R / 10)]

radius = int(R/2)
fi_lim = [10, 170]

n = 1000

main_rpr = RPR(
    ground_joints=R,
    platform_joints=r,
    Lmin=L_lim[0],
    Lmax=L_lim[1],
    name="RPR_main",
)

rprs: list[RPR] = [main_rpr]

# for i in range(10):
#     for joint in range(2):
#         ground_joints=main_rpr.ground.joints
#         for axis in ['x', 'y', 'xy']:
#             ground_joints[joint]=modify_vector(ground_joints[joint], offset_point, axis)
#             rpr = RPR(
#                     ground_joints=ground_joints,
#                     platform_joints=main_rpr.platform.joints,
#                     Lmin=main_rpr.Lmin, Lmax=main_rpr.Lmax,
#                     name=f"RPR_{i}_ground_{axis}",
#                 )
#             rprs.append(rpr)

# for i in range(10):
#     for joint in range(2):
#         platform_joints=main_rpr.platform.joints
#         for axis in ['x', 'y', 'xy']:
#             platform_joints[joint]=modify_vector(platform_joints[joint], offset_point, axis)
#             rpr = RPR(
#                     ground_joints=main_rpr.ground.joints,
#                     platform_joints=platform_joints,
#                     Lmin=main_rpr.Lmin, Lmax=main_rpr.Lmax,
#                     name=f"RPR_{i}_platform_{axis}",
#                 )
#             rprs.append(rpr)

xyz = [
    (coord, angle) for coord, angle in points(radius=radius, limit=fi_lim, R=R, r=r, n=n)
]


for i, rpr in tqdm(enumerate(rprs), total=len(rprs)):
    for idx, (coord, angle) in enumerate(xyz):
        rpr.move(
            coord=coord,
            angle=angle,
        )
        data = log(rpr)  # , ref=main_rpr)
        write(data, filename=f"rpr/simulator/data/test_{R}_{r}_{radius}_{n}.csv")

        if idx % 100 == 0:
            fig, ax = plt.subplots(ncols=1, nrows=1, figsize=size(np.array([10, 10])))
            rpr.plot(axis=ax)

            ax.margins(0.25)
            plt.show(block=False)
            plt.pause(0.1)
            time.sleep(1)
            plt.close()
