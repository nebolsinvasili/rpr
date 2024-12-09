from typing import Union

import numpy as np
from numpy.linalg import multi_dot

from .detail import Detail

np.set_printoptions(precision=3, floatmode="fixed")


class Rotate:
    def __init__(self):
        pass

    @staticmethod
    def Rx(theta):
        return np.array(
            [
                [1, 0, 0],
                [0, np.cos(theta), np.sin(theta)],
                [0, -np.sin(theta), np.cos(theta)],
            ]
        )

    @staticmethod
    def Ry(phi):
        return np.array(
            [
                [np.cos(phi), 0, -np.sin(phi)],
                [0, 1, 0],
                [np.sin(phi), 0, np.cos(phi)],
            ]
        )

    @staticmethod
    def Rz(psi):
        return np.array(
            [
                [np.cos(psi), np.sin(psi), 0],
                [-np.sin(psi), np.cos(psi), 0],
                [0, 0, 1],
            ]
        )


class Platform(Detail):
    def __init__(
        self,
        coord: Union[np.ndarray, list] = np.array([0, 0, 0]),
        angle: Union[np.ndarray, list] = np.array([0, 0, 0]),
        joints: Union[np.ndarray, list, float] = 50,
        scale: float = 1.0,
        speed: float = 1.0,
        name: str = "platform",
    ):
        super().__init__(coord, angle, joints, scale, speed, name)

        self.rs = self.distance_attact_joints(self.coord, self.joints)
        self.js = self.angle_attact_joints(self.coord, self.joints)

        self.moving = False
        self.offset = None

    def move(
        self,
        coord: Union[np.ndarray, list] = None,
        angle: Union[np.ndarray, list] = None,
    ):
        coord = coord if coord is not None else self.coord
        angle = angle if angle is not None else self.angle

        self.offset = np.array(
            [np.subtract(coord, self.coord), np.subtract(angle, self.angle)]
        )
        if self.offset.any() != 0:
            if self.moving:
                self.queue.append((coord, angle))
                return False

            self.moving = True
            self.coord = coord
            self.angle = angle

            joints = self.joints - self.coord_old

            joints = self.rotate_joints(offset_angle=self.offset[1], joints=joints)

            self.joints = joints + self.coord

            self.rs = Platform.distance_attact_joints(self.coord, self.joints)
            self.js = Platform.angle_attact_joints(self.coord, self.joints)

            self.moving = False
            return True

        else:
            return False

    @staticmethod
    def Rx(theta):
        return np.array(
            [
                [1, 0, 0],
                [0, np.cos(theta), np.sin(theta)],
                [0, -np.sin(theta), np.cos(theta)],
            ]
        )

    @staticmethod
    def Ry(phi):
        return np.array(
            [
                [np.cos(phi), 0, -np.sin(phi)],
                [0, 1, 0],
                [np.sin(phi), 0, np.cos(phi)],
            ]
        )

    @staticmethod
    def Rz(psi):
        return np.array(
            [
                [np.cos(psi), np.sin(psi), 0],
                [-np.sin(psi), np.cos(psi), 0],
                [0, 0, 1],
            ]
        )

    def rotate_joints(
        self,
        offset_angle: np.ndarray,
        joints: np.ndarray,
    ) -> np.ndarray:
        off_x, off_y, off_z = np.deg2rad(offset_angle)
        return multi_dot(
            [
                joints,
                Platform.Rx(theta=off_x),
                Platform.Ry(phi=off_y),
                Platform.Rz(psi=off_z),
            ]
        ).astype(np.float32)

    @classmethod
    def distance_joint(cls, a, b):
        """
        distances from coord to joint
        """
        return np.linalg.norm(a - b)

    @classmethod
    def angle_joint(cls, a, b):
        """
        angle from coord to joint
        """
        return np.rad2deg(np.arctan2(*np.flip(a - b, axis=0)))

    @classmethod
    def distance_attact_joints(cls, coord: np.ndarray, joints: np.ndarray):
        return np.apply_along_axis(
            lambda joint: Platform.distance_joint(coord, joint), axis=1, arr=joints
        )

    @classmethod
    def angle_attact_joints(cls, coord: np.ndarray, joints: np.ndarray):
        return np.apply_along_axis(
            lambda joint: Platform.angle_joint(joint[:2], coord[:2]), axis=1, arr=joints
        )


if __name__ == "__main__":
    platform = Platform(
        coord=np.array([0, 0, 0]),
        angle=np.array([0, 0, 0]),
        joints=25,
        scale=1,
    )
    print(platform.coord, platform.angle)
    print(platform.joints)

    dict = {}
    for axis, value in zip("xyz", platform.coord):
        dict.update({f"O_{axis}": value})
    print(dict)
    dict.update({"O_fi": platform.angle[2]})

    for i, value in enumerate(
        Platform.distance_attact_joints(platform.coord, platform.joints)
    ):
        dict.update({f"Bdj{i+1}": value})
    print(dict)

    for i, value in enumerate(
        Platform.angle_attact_joints(platform.coord, platform.joints)
    ):
        dict.update({f"Baj{i+1}": value})
    print(dict)

    for i, xyz in enumerate(platform.joints):
        for axis, value in zip("xyz", xyz):
            dict.update({f"B{i+1}_{axis}": value})
    print(dict)

    platform.move(
        coord=np.array([10, 0, 0]),
        angle=np.array([0, 0, 15]),
    )

    platform.move(
        coord=np.array([10, 0, 0]),
        angle=np.array([0, 0, 15]),
    )
