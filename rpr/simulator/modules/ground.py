from typing import Union

import numpy as np
import pandas as pd

from . import Detail


class Ground(Detail):
    def __init__(
        self,
        coord: Union[np.ndarray, list] = np.array([0, 0, 0]),
        angle: Union[np.ndarray, list, float] = np.array([0, 0, 0]),
        joints: Union[np.ndarray, list, float] = 50,
        scale: float = 1.0,
    ):
        super().__init__(coord, angle, joints, scale)


if __name__ == "__main__":
    ground = Ground(
        coord=np.array([0, 0, 0]),
        angle=np.array([0, 0, 0]),
        joints=100,
    )

    dict_info = {}
    for i, xyz in enumerate(ground.joints):
        for axis, value in zip("xyz", xyz):
            dict_info.update({f"A{i+1}_{axis}": value})
    print(dict_info)

    print(pd.DataFrame([dict_info.values()], columns=dict_info.keys()))
