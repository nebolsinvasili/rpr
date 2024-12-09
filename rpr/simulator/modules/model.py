import math
import os
from typing import Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from loguru import logger
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from utils.plot import size

from . import Ground, Leg, Platform


class RPR:
    def __init__(
        self,
        ground_joints: Union[np.ndarray, list, float],
        platform_coord=np.array([0, 0, 0]),
        platform_angle=np.array([0, 0, 0]),
        platform_joints: Union[np.ndarray, list, float] = None,
        Lmin=10,
        Lmax=190,
        scale: float = 1.0,
        fig: Figure = None,
        axis: Axes = None,
        name: str = "RPR",
    ):
        self.ground: Ground = Ground(
            coord=np.array([0, 0, 0]),
            angle=np.array([0, 0, 90]),
            joints=ground_joints,
            scale=scale,
        )
        self.platform = Platform(
            coord=platform_coord,
            angle=platform_angle,
            joints=platform_joints,
            scale=scale,
        )
        self.leg = Leg(self.ground, self.platform)
        self.Lmin = Lmin
        self.Lmax = Lmax

        self.fig, self.axis = fig, axis

        self.name = name

    def plot(self, axis: Axes):
        # axis.plot(*self.platform.coord[:2], "ro")  # Plot center
        axis.plot(
            *zip(
                *np.concatenate(
                    (self.platform.joints, self.platform.joints[0][None, :]), axis=0
                )[:, :2].tolist()
            ),
            color="blue",
        )

        r = np.stack((self.ground.joints, self.platform.joints), axis=1)[
            :, :, :2
        ].tolist()
        for i in r:
            axis.plot(*zip(*i), color="blue")  # Plot legs

    def move(
        self,
        coord: Union[np.ndarray, list] = None,
        angle: Union[np.ndarray, list] = None,
    ):
        if self.platform.move(coord, angle):
            # print(Leg.get_distance_legs(self.ground.joints, self.platform.joints), Leg.get_angle_leg(self.ground.joints, self.platform.joints))

            # if all(Leg.get_distance_legs(self.ground.joints, self.platform.joints)) >= self.Lmin and all(Leg.get_distance_legs(self.ground.joints, self.platform.joints)) >=  self.Lmax:
            logger.info(
                f"{self.name} | MOVE "
                f"| Coord: {self.platform.coord} "
                f"| Angle: {self.platform.angle} "
                f"| Offsets: ({self.platform.offset[0]}, {self.platform.offset[1]})"
                f"| RS: {self.platform.rs}"
                f"| JS: {self.platform.js}"
                f"| L1: {Leg.get_distance_legs(self.ground.joints, self.platform.joints), Leg.get_angle_leg(self.ground.joints, self.platform.joints)}"
            )

    def data(self, filename: str = "test.csv"):
        data = {}
        for i, xyz in enumerate(self.ground.joints):
            for axis, value in zip("xyz", xyz):
                data.update({f"A{i+1}_{axis}": value})

        for axis, value in zip("xyz", self.platform.coord):
            data.update({f"O_{axis}": value})
        data.update({"O_fi": self.platform.angle[2]})

        for i, value in enumerate(
            Platform.distance_attact_joints(self.platform.coord, self.platform.joints)
        ):
            data.update({f"Bdj{i+1}": value})

        for i, value in enumerate(
            Platform.angle_attact_joints(self.platform.coord, self.platform.joints)
        ):
            data.update({f"Baj{i+1}": value})

        for i, xyz in enumerate(self.platform.joints):
            for axis, value in zip("xyz", xyz):
                data.update({f"B{i+1}_{axis}": value})

        for i, distance in enumerate(
            Leg.get_distance_legs(
                ground_joints=self.ground.joints, platform_joints=self.platform.joints
            )
        ):
            data.update({f"L{i+1}d": distance})

        for i, angle in enumerate(
            Leg.get_angle_leg(
                ground_joints=self.ground.joints, platform_joints=self.platform.joints
            )
        ):
            data.update({f"L{i+1}a": angle})
        # print(data)

        if not os.path.exists(filename):
            keys_df = pd.DataFrame([data.keys()], columns=data.keys())
            keys_df.to_csv(filename, header=False, index=False)
        pd.DataFrame([data]).to_csv(filename, header=False, index=False, mode="a")


if __name__ == "__main__":
    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=size(np.array([10, 10])))

    rpr = RPR(
        ground_joints=200 * math.sqrt(3) / 6,
        platform_joints=100 * math.sqrt(3) / 6,
        Lmin=100,
        Lmax=160,
        name="RPR_1",
    )

    print(rpr.ground.joints)
    print(rpr.platform.joints)

    rpr.move(
        coord=np.array([0, 0, 10]),
        angle=np.array([0, 0, 200]),
    )
    rpr.plot(axis=ax)

    ax.margins(0.25)
    plt.show()
