from math import sin, cos

import numpy as np


class MatchConfig:
    def __init__(self,
                 trans_x: float,
                 trans_y: float,
                 rotate_1: float,
                 rotate_2: float,
                 scale_x: float,
                 scale_y: float,
                 ):
        self.translateX = trans_x
        self.translateY = trans_y
        self.rotate1 = rotate_1
        self.rotate2 = rotate_2
        self.scaleX = scale_x
        self.scaleY = scale_y

        # Create Affine matrix
        try:
            cos_r1 = cos(self.rotate1)
            sin_r1 = sin(self.rotate1)
            cos_r2 = cos(self.rotate2)
            sin_r2 = sin(self.rotate2)
        except (Exception,) as e:
            print(self.rotate1, self.rotate2)
            raise e

        a11 = self.scaleX * cos_r1 * cos_r2 - self.scaleY * sin_r1 * sin_r2
        a12 = -self.scaleX * cos_r1 * sin_r2 - self.scaleY * cos_r2 * sin_r1
        a21 = self.scaleX * cos_r2 * sin_r1 + self.scaleY * cos_r1 * sin_r2
        a22 = self.scaleY * cos_r1 * cos_r2 - self.scaleX * sin_r1 * sin_r2

        self.affine = np.array([
            [a11, a12, self.translateX],
            [a21, a22, self.translateY],
        ])

    def as_matrix(self):
        return np.array([[self.translateX, self.translateY, self.rotate1, self.rotate2, self.scaleX, self.scaleY]])

    @staticmethod
    def from_matrix(configs):
        result = []
        for config in configs:
            # try:
            result.append(MatchConfig(
                config[0],
                config[1],
                config[2],
                config[3],
                config[4],
                config[5],
            ))

        return result
