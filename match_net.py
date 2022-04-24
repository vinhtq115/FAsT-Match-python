import math
from typing import Union
import copy
import numpy as np


def _get_steps(bound_min,
               bound_max,
               step,
               factor: Union[float, int] = 1,
               ):
    steps = np.arange(bound_min, bound_max, step)

    if steps[-1] + step - bound_max < 1e-10:
        steps = np.concatenate([steps, [bound_max]])

    if bound_max - steps[-1] > factor * step:
        steps = np.concatenate([steps, [steps[-1] + step]])

    return steps


class MatchNet:
    def __init__(self,
                 width: int,
                 height: int,
                 delta: float,
                 min_tx: float,
                 max_tx: float,
                 min_ty: float,
                 max_ty: float,
                 min_r: float,
                 max_r: float,
                 min_s: float,
                 max_s: float,
                 ):
        self.bounds_trans_x = (min_tx, max_tx)
        self.bounds_trans_y = (min_ty, max_ty)
        self.bounds_rotate = (min_r, max_r)
        self.bounds_scale = (min_s, max_s)

        self.steps_trans_X = delta * width / math.sqrt(2.0)
        self.steps_trans_Y = delta * height / math.sqrt(2.0)
        self.steps_rotate = delta * math.sqrt(2.0)
        self.steps_scale = delta / math.sqrt(2.0)

    def get_x_translation_steps(self):
        return _get_steps(self.bounds_trans_x[0], self.bounds_trans_x[1], self.steps_trans_X, 0.5)

    def get_y_translation_steps(self):
        return _get_steps(self.bounds_trans_x[0], self.bounds_trans_x[1], self.steps_trans_Y, 0.5)

    def get_rotation_steps(self):
        return _get_steps(self.bounds_rotate[0], self.bounds_rotate[1], self.steps_rotate)

    def get_scale_steps(self):
        s_steps = np.arange(self.bounds_scale[0], self.bounds_scale[1], self.steps_scale)
        if s_steps[-1] + self.steps_scale - self.bounds_scale[1] < 1e-10:
            s_steps = np.concatenate([s_steps, [self.bounds_scale[1]]])

        if self.steps_scale - 0.0 < 1e-10:
            s_steps = np.array([self.bounds_scale[0]])

        if self.bounds_scale[1] - s_steps[-1] > 0.5 * self.steps_scale:
            s_steps = np.concatenate([s_steps, [s_steps[-1] + self.steps_scale]])

        return s_steps.tolist()

    def __mul__(self, factor: float):
        result = copy.deepcopy(self)
        result.steps_trans_X *= factor
        result.steps_trans_Y *= factor
        result.steps_rotate *= factor
        result.steps_scale *= factor
        return result
