import math

import numpy as np

from . import Ground, Platform


class Leg:
    def __init__(
        self,
        ground: Ground,
        platform: Platform,
    ) -> None:
        self.ground = ground
        self.platform = platform

    @classmethod
    def get_distance_legs(cls, ground_joints, platform_joints, ndigits: int = 8):
        return np.array(
            [
                round(np.linalg.norm(diff), ndigits=ndigits)
                for diff in ground_joints - platform_joints
            ]
        )

    @classmethod
    def get_angle_leg(cls, ground_joints, platform_joints):
        return np.array(
            [
                np.rad2deg(np.arctan2(*np.flip(diff[:2], axis=0)))
                for diff in platform_joints - ground_joints
            ]
        )

    @classmethod
    def get_coords_leg(cls, coord, angle, length):
        x2 = coord[0] + length * math.cos(np.deg2rad(angle))
        y2 = coord[1] + length * math.sin(np.deg2rad(angle))
        return (coord[0], coord[1]), (x2, y2)

    @classmethod
    def get_theta(cls, point, center):
        vector = point - center
        return np.arctan2(vector[1], vector[0])

    @classmethod
    def angle(cls, point1, center, point2):
        theta1 = Leg.get_theta(point1, center)
        theta2 = Leg.get_theta(point2, center)
        # np.rad2deg((theta2 - theta1) % 360)
        angle_span = (theta2 - theta1) % 360
        angle = np.deg2rad(theta1 + angle_span / 2)
        return angle


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    from ...utils.plot import size

    ground = Ground(
        coord=np.array([0, 0, 0]),
        angle=np.array([0, 0, 90]),
        joints=100,
    )

    platform = Platform(
        coord=np.array([0, 0, 0]),
        angle=np.array([0, 0, 0]),
        joints=25,
        scale=1,
    )

    leg = Leg(ground, platform)

    print(
        Leg.get_distance_legs(
            ground_joints=ground.joints, platform_joints=platform.joints
        )
    )
    print(
        Leg.get_angle_leg(ground_joints=ground.joints, platform_joints=platform.joints)
    )

    dict = {}
    for i, distance in enumerate(
        Leg.get_distance_legs(
            ground_joints=ground.joints, platform_joints=platform.joints
        )
    ):
        dict.update({f"L{i+1}d": distance})
    print(dict)

    for i, angle in enumerate(
        Leg.get_angle_leg(ground_joints=ground.joints, platform_joints=platform.joints)
    ):
        dict.update({f"L{i+1}a": angle})
    print(dict)

    fig, ax = plt.subplots(figsize=size(np.array([12, 12])))

    ax.plot(*platform.coord[:2], "ro")
    ax.plot(
        *zip(
            *np.concatenate((platform.joints, platform.joints[0][None, :]), axis=0)[
                :, :2
            ].tolist()
        ),
        color="blue",
    )  # Plot platform

    r = np.stack((ground.joints, platform.joints), axis=1)[:, :, :2].tolist()
    for i in r:
        ax.plot(*zip(*i), color="blue")

    plt.show()

    platform.move(
        coord=np.array([10, 0, 0]),
        angle=np.array([0, 0, 15]),
    )

    print(leg.get_distance_legs())
    print(leg.get_angle_leg())

    fig, ax = plt.subplots(figsize=size(np.array([12, 12])))

    ax.plot(*platform.coord[:2], "ro")
    ax.plot(
        *zip(
            *np.concatenate((platform.joints, platform.joints[0][None, :]), axis=0)[
                :, :2
            ].tolist()
        ),
        color="blue",
    )

    r = np.stack((ground.joints, platform.joints), axis=1)[:, :, :2].tolist()
    for i in r:
        ax.plot(*zip(*i), color="blue")

    # ax.set_xlim((-100, 100))
    # ax.set_ylim((-100, 100))
    ax.relim()  # update axes limits

    plt.show()

    platform.move(
        coord=np.array([10, 0, 0]),
        angle=np.array([0, 0, 15]),
    )
