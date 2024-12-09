import os

import pandas as pd

from ..modules import Leg, Platform


def data_A(data, rpr, ref=None):
    if ref is not None:
        for i, xyz in enumerate(ref.ground.joints):
            for axis, value in zip("xyz", xyz):
                data.update({f"Aref{i+1}_{axis}": value})

    for i, xyz in enumerate(rpr.ground.joints):
        for axis, value in zip("xyz", xyz):
            data.update({f"A{i+1}_{axis}": value})

    return data


def data_B(data, rpr, ref=None):
    if ref is not None:
        for i, value in enumerate(
            Platform.distance_attact_joints(ref.platform.coord, ref.platform.joints)
        ):
            data.update({f"Bref{i+1}_dj": value})

        for i, value in enumerate(
            Platform.angle_attact_joints(ref.platform.coord, ref.platform.joints)
        ):
            data.update({f"Bref{i+1}_aj": value})

        for i, xyz in enumerate(ref.platform.joints):
            for axis, value in zip("xyz", xyz):
                data.update({f"Bref{i+1}_{axis}": value})

    for i, value in enumerate(
        Platform.distance_attact_joints(rpr.platform.coord, rpr.platform.joints)
    ):
        data.update({f"B{i+1}_dj": value})

    for i, value in enumerate(
        Platform.angle_attact_joints(rpr.platform.coord, rpr.platform.joints)
    ):
        data.update({f"B{i+1}_aj": value})

    for i, xyz in enumerate(rpr.platform.joints):
        for axis, value in zip("xyz", xyz):
            data.update({f"B{i+1}_{axis}": value})

    return data


def data_leg(data, rpr, ref=None):
    if ref is not None:
        for i, distance in enumerate(
            Leg.get_distance_legs(
                ground_joints=ref.ground.joints, platform_joints=ref.platform.joints
            )
        ):
            data.update({f"Lref{i+1}_d": distance})

        for i, angle in enumerate(
            Leg.get_angle_leg(
                ground_joints=ref.ground.joints, platform_joints=ref.platform.joints
            )
        ):
            data.update({f"Lref{i+1}_a": angle})

    for i, distance in enumerate(
        Leg.get_distance_legs(
            ground_joints=rpr.ground.joints, platform_joints=rpr.platform.joints
        )
    ):
        data.update({f"L{i+1}d": distance})

    for i, angle in enumerate(
        Leg.get_angle_leg(
            ground_joints=rpr.ground.joints, platform_joints=rpr.platform.joints
        )
    ):
        data.update({f"L{i+1}a": angle})

    return data


def data_out(data, rpr):
    for axis, value in zip("xyz", rpr.platform.coord):
        data.update({f"out_{axis}": value})
    data.update({"out_fi": rpr.platform.angle[2]})

    return data


def log(rpr, ref=None):
    data = {}

    data = data_A(data, rpr, ref=ref)

    data = data_B(data, rpr, ref=ref)

    data = data_out(data, rpr)

    data = data_leg(data, rpr, ref=ref)

    return data


def write(data, filename: str = "test.csv"):
    if not os.path.exists(filename):
        keys_df = pd.DataFrame([data.keys()], columns=data.keys())
        keys_df.to_csv(filename, header=False, index=False)
    pd.DataFrame([data]).to_csv(filename, header=False, index=False, mode="a")
