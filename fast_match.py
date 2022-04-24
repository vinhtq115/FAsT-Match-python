import math
import itertools

import cv2
import numpy as np

from match_net import MatchNet
from match_config import MatchConfig

DELTA_FACT = 1.511


def preprocess_image(image: np.ndarray):
    """
    Convert image to gray and normalize to 0-1 range.
    :param image: Input image
    :return: Normalized image
    """
    # Convert to gray
    temp = image.copy()
    if len(temp.shape) != 2:
        temp = cv2.cvtColor(temp, cv2.COLOR_BGR2GRAY)

    # Normalize
    temp = temp.astype(float)
    temp /= 255.0
    return temp


def is_within(val: tuple,
              top_left: tuple,
              bottom_right: tuple,
              ):
    """
    Check if a point is inside a rectangle.
    :param val: Point
    :param top_left: Top left of rectangle
    :param bottom_right: Bottom right of rectangle
    :return: True if inside, else False
    """
    return top_left[0] < val[0] < bottom_right[0] and top_left[1] < val[1] < bottom_right[1]


class FAsTMatch:
    def __init__(self,
                 epsilon: float,
                 delta: float,
                 photometric_invariance: bool,
                 min_scale: float,
                 max_scale: float,
                 ):
        self.epsilon = epsilon
        self.delta = delta
        self.photometric_invariance = photometric_invariance
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.image = None
        self.template = None

    def apply(self, original_image, original_template):
        """
        Apply Fast Template Matching algorithm
        :param original_image: Image
        :param original_template: Template
        :return: Rectangle corners of best affine transformation
        """
        image = preprocess_image(original_image)
        template = preprocess_image(original_template)

        r1x = 0.5 * (template.shape[1] - 1)
        r1y = 0.5 * (template.shape[0] - 1)
        r2x = 0.5 * (image.shape[1] - 1)
        r2y = 0.5 * (image.shape[0] - 1)

        min_trans_x = -(r2x - r1x + self.min_scale)
        max_trans_x = -min_trans_x
        min_trans_y = -(r2y - r1y * self.min_scale)
        max_trans_y = -min_trans_y

        min_rotation = -3.14159265358979323846
        max_rotation = 3.14159265358979323846

        # Create the matching grid/net
        net = MatchNet(
            template.shape[1],
            template.shape[0],
            self.delta,
            min_trans_x,
            max_trans_x,
            min_trans_y,
            max_trans_y,
            min_rotation,
            max_rotation,
            self.min_scale,
            self.max_scale
        )

        # Smooth images
        image = cv2.GaussianBlur(image, (0, 0), 2.0, 2.0)
        template = cv2.GaussianBlur(template, (0, 0), 2.0, 2.0)

        no_of_points = round(10 / (self.epsilon ** 2))

        # Randomly sample points
        xs = np.random.randint(1, template.shape[1], size=(1, no_of_points), dtype=np.int32)
        ys = np.random.randint(1, template.shape[0], size=(1, no_of_points), dtype=np.int32)

        level = 0

        new_delta = self.delta

        best_distances = [0.0] * 20

        while True:
            level += 1

            # First create configurations based on our net
            configs = self.create_list_of_configs(net)

            configs_count = len(configs)

            affines, insiders = self.configs_to_affine(configs, image, template)

            temp_configs = []
            for i in range(len(insiders)):
                if insiders[i]:
                    temp_configs.append(configs[i])
            configs = temp_configs

            distances = self.evaluate_configs(image,
                                              template,
                                              affines,
                                              xs,
                                              ys,
                                              self.photometric_invariance)

            best_distance = np.min(distances)
            best_distances[level] = best_distance
            best_distance_idx = np.argmin(distances)
            best_config = configs[best_distance_idx]
            best_trans = best_config.affine

            # Conditions to exit the loop
            if best_distance < 0.005 or (level > 2 and best_distance < 0.015) or level >= 20:
                break

            if level > 3:
                mean_value = sum(best_distances[level - 3:]) * 1.0 / len(distances)

                if best_distance > mean_value * 0.97:
                    break

            good_configs, thresh, too_high_percentage = self.get_good_configs_by_distance(
                configs,
                best_distance,
                new_delta,
                distances
            )

            if (too_high_percentage and best_distance > 0.05 and level == 1 and configs_count < 7.5e6) \
                    or (best_distance > 0.1 and level == 1 and configs_count < 5e6):
                factor = 0.9
                new_delta = new_delta * factor
                level = 0
                net = net * factor
                configs = self.create_list_of_configs(net)
            else:
                new_delta = new_delta / DELTA_FACT

                expanded_configs = self.random_expand_configs(good_configs, net, level, 80, DELTA_FACT)

                configs.clear()
                configs.extend(good_configs)
                configs.extend(expanded_configs)

            xs = np.random.randint(1, template.shape[1], size=(1, no_of_points), dtype=np.int32)
            ys = np.random.randint(1, template.shape[0], size=(1, no_of_points), dtype=np.int32)

        return self.calc_corners(image.shape, template.shape, best_trans), best_distance

    def create_list_of_configs(self, net: MatchNet):
        """
        Given our grid/net, create a list of matching configurations.
        :param net: Net
        :return: List of configs
        """
        tx_steps = net.get_x_translation_steps()
        ty_steps = net.get_y_translation_steps()
        r_steps = net.get_rotation_steps()
        s_steps = net.get_scale_steps()

        if math.fabs((net.bounds_rotate[1] - net.bounds_rotate[0]) - (2 * math.pi)) < 0.1:
            # Refine the number of steps for the 2nd rotation parameter
            nr2_steps = len([i for i in r_steps if i < (-math.pi / 2 + net.steps_rotate / 2)])
        else:
            nr2_steps = len(r_steps)

        configs = itertools.product(tx_steps, ty_steps, r_steps, r_steps[:nr2_steps], s_steps, s_steps)

        configs = [MatchConfig(*config) for config in configs]
        return configs

    def random_expand_configs(self,
                              configs: [MatchConfig],
                              net: MatchNet,
                              level: int,
                              no_of_points: int,
                              delta_factor: float
                              ):
        """
        Randomly expands the configuration.
        :param configs:
        :param net:
        :param level:
        :param no_of_points:
        :param delta_factor:
        :return:
        """
        factor = delta_factor ** level

        half_step_tx = net.steps_trans_X / factor
        half_step_ty = net.steps_trans_Y / factor
        half_step_r = net.steps_rotate / factor
        half_step_s = net.steps_scale / factor

        no_of_configs = len(configs)

        random_vec = np.random.choice([-1, 0, 1], (no_of_configs * no_of_points, 6)).astype(float)

        configs_mat = [config.as_matrix() for config in configs]

        expanded = np.vstack(configs_mat)
        expanded = np.tile(expanded, (no_of_points, 1))

        ranges_vec = np.array([half_step_tx, half_step_ty, half_step_r, half_step_r, half_step_s, half_step_s])

        ranges = np.tile(np.transpose(ranges_vec), (no_of_configs * no_of_points, 1))

        # The expanded configs is the original configs plus some random changes
        np.multiply(random_vec, ranges)
        expanded_configs = expanded + np.multiply(random_vec, ranges)

        return MatchConfig.from_matrix(expanded_configs)

    def get_good_configs_by_distance(self,
                                     configs: [MatchConfig],
                                     best_dist: float,
                                     new_delta: float,
                                     distances: [float]):
        """
        Given the previously calcuated distances for each configurations,
        filter out all distances that fall within a certain threshold
        :param configs: List of MatchConfig
        :param best_dist:
        :param new_delta:
        :param distances:
        :return:
        """
        thresh = best_dist + self.get_threshold_per_delta(new_delta)

        # Only those configs that have distances below the given threshold are
        # categorized as good configurations
        good_configs = []
        for i in range(len(distances)):
            if distances[i] <= thresh:
                good_configs.append(configs[i])

        no_of_configs = len(good_configs)

        # Well if there's still too many configurations, keep shrinking the threshold
        while no_of_configs > 27000:
            thresh *= 0.99
            good_configs.clear()

            for i in range(len(distances)):
                if distances[i] <= thresh:
                    good_configs.append(configs[i])

            no_of_configs = len(good_configs)

        assert no_of_configs > 0

        percentage = 1.0 * no_of_configs / len(configs)

        # If it's above 97.8% it's too high percentage
        too_high_percentage = percentage > 0.022

        return good_configs, thresh, too_high_percentage

    def get_threshold_per_delta(self, delta: float):
        p = (0.1341, 0.0278)
        safety = 0.02

        return p[0] * delta + p[1] - safety

    def configs_to_affine(self,
                          configs: [MatchConfig],
                          image: np.ndarray,
                          template: np.ndarray,
                          ):
        """
        From given list of configurations, convert them into affine matrices.
        But filter out all the rectangles that are out of the given boundaries.
        :param configs: List of MatchConfig
        :param image: Input image
        :param template: Template image
        :return: Config matrix fitting boundaries and a boolean array of configs satisfied (same length with `configs`).
        """
        no_of_configs = len(configs)

        insiders = [False] * no_of_configs
        result = []

        top_left = (-10.0, -10.0)
        bottom_right = (image.shape[1] + 10, image.shape[0] + 10)

        r1x = 0.5 * (template.shape[1] - 1)
        r1y = 0.5 * (template.shape[0] - 1)
        r2x = 0.5 * (image.shape[1] - 1)
        r2y = 0.5 * (image.shape[0] - 1)

        corners = np.array([
            [1 - (r1x + 1), template.shape[1] - (r1x + 1), template.shape[1] - (r1x + 1), 1 - (r1x + 1)],
            [1 - (r1y + 1), 1 - (r1y + 1), template.shape[0] - (r1y + 1), template.shape[0] - (r1y + 1)],
            [1.0, 1.0, 1.0, 1.0]
        ])

        transl = np.array([
            [r2x + 1, r2y + 1],
            [r2x + 1, r2y + 1],
            [r2x + 1, r2y + 1],
            [r2x + 1, r2y + 1],
        ])

        for i in range(no_of_configs):
            affine = configs[i].affine

            # Check if our affine transformed rectangle still fits within our boundary
            affine_corners = np.transpose(np.matmul(affine, corners)) + transl

            if is_within(affine_corners[0], top_left, bottom_right) \
                    and is_within(affine_corners[1], top_left, bottom_right) \
                    and is_within(affine_corners[2], top_left, bottom_right) \
                    and is_within(affine_corners[3], top_left, bottom_right):
                result.append(affine)
                insiders[i] = True

        return result, insiders

    def evaluate_configs(self,
                         image: np.ndarray,
                         template: np.ndarray,
                         affine_matrices: list,
                         xs: np.ndarray,
                         ys: np.ndarray,
                         photometric_invariance: bool):
        """
        Evaluate the score of the given configurations
        :param image: Image
        :param template: Template
        :param affine_matrices: List of Affine matrices
        :param xs: x value of random points
        :param ys: y value of random points
        :param photometric_invariance: True if photometric invariance
        :return: List of score
        """
        r1x = 0.5 * (template.shape[1] - 1)
        r1y = 0.5 * (template.shape[0] - 1)
        r2x = 0.5 * (image.shape[1] - 1)
        r2y = 0.5 * (image.shape[0] - 1)

        no_of_configs = len(affine_matrices)
        no_of_points = xs.shape[1]

        # Use a padded image, to avoid boundary checking
        padded = np.vstack([
            np.zeros_like(image, dtype=image.dtype),
            image,
            np.zeros_like(image, dtype=image.dtype),
        ])

        # Create a lookup array for our template values based on the given random x and y points
        xs_ptr = xs[0]
        ys_ptr = ys[0]

        vals_i1 = [template[ys_ptr[i] - 1, xs_ptr[i] - 1] for i in range(no_of_points)]

        # Recenter our indices
        xs_centered = xs.copy() - (r1x + 1)
        ys_centered = ys.copy() - (r1y + 1)

        xs_ptr_cent = xs_centered[0]
        ys_ptr_cent = ys_centered[0]

        distances = [0.0] * no_of_configs

        # Calculate the score for each configurations on each of our randomly sampled points
        for i in range(no_of_configs):
            a11 = affine_matrices[i][0][0]
            a12 = affine_matrices[i][0][1]
            a13 = affine_matrices[i][0][2]
            a21 = affine_matrices[i][1][0]
            a22 = affine_matrices[i][1][1]
            a23 = affine_matrices[i][1][2]

            tmp_1 = (r2x + 1) + a13 + 0.5
            tmp_2 = (r2y + 1) + a23 + 0.5 + 1 * image.shape[0]
            score = 0.0

            if not photometric_invariance:
                for j in range(no_of_points):
                    target_x = int(a11 * xs_ptr_cent[j] + a12 * ys_ptr_cent[j] + tmp_1)
                    target_y = int(a21 * xs_ptr_cent[j] + a22 * ys_ptr_cent[j] + tmp_2)

                    if 0 <= target_x < padded.shape[1]:
                        score += abs(vals_i1[j] - padded[target_y - 1, target_x - 1])
            else:
                xs_target = []
                ys_target = []

                sum_x = 0.0
                sum_y = 0.0
                sum_x_squared = 0.0
                sum_y_squared = 0.0

                for j in range(no_of_points):
                    target_x = int(a11 * xs_ptr_cent[j] + a12 * ys_ptr_cent[j] + tmp_1)
                    target_y = int(a21 * xs_ptr_cent[j] + a22 * ys_ptr_cent[j] + tmp_2)

                    xi = vals_i1[j]
                    yi = padded[target_y - 1, target_x - 1]

                    xs_target.append(xi)
                    ys_target.append(yi)

                    sum_x += xi
                    sum_y += yi

                    sum_x_squared += (xi * xi)
                    sum_y_squared += (yi * yi)

                epsilon = 1e-7
                mean_x = sum_x / no_of_points
                mean_y = sum_y / no_of_points
                sigma_x = math.sqrt((sum_x_squared - (sum_x * sum_x) / no_of_points) / no_of_points) + epsilon
                sigma_y = math.sqrt((sum_y_squared - (sum_y * sum_y) / no_of_points) / no_of_points) + epsilon

                sigma_div = sigma_x / sigma_y
                temp = -mean_x + sigma_div * mean_y

                for j in range(no_of_points):
                    score += math.fabs(xs_target[j] - sigma_div * ys_target[j] + temp)

            distances[i] = score / no_of_points

        return distances

    def calc_corners(self,
                     image_size: tuple,
                     template_size: tuple,
                     affine: np.ndarray
                     ):
        """
        From the given affine matrix, calculate the four corners of the affine transformed rectangle
        :param image_size: Tuple of image size
        :param template_size: Tuple of template size
        :param affine: Affine matrix
        :return:
        """
        r1x = 0.5 * (template_size[1] - 1)
        r1y = 0.5 * (template_size[0] - 1)
        r2x = 0.5 * (image_size[1] - 1)
        r2y = 0.5 * (image_size[0] - 1)

        a11 = affine[0, 0]
        a12 = affine[0, 1]
        a13 = affine[0, 2]
        a21 = affine[1, 0]
        a22 = affine[1, 1]
        a23 = affine[1, 2]

        templ_w = template_size[1]
        templ_h = template_size[0]

        # The four corners of affine transformed template
        c1x = a11 * (1 - (r1x + 1)) + a12 * (1 - (r1y + 1)) + (r2x + 1) + a13
        c1y = a21 * (1 - (r1x + 1)) + a22 * (1 - (r1y + 1)) + (r2y + 1) + a23

        c2x = a11 * (templ_w - (r1x + 1)) + a12 * (1 - (r1y + 1)) + (r2x + 1) + a13
        c2y = a21 * (templ_w - (r1x + 1)) + a22 * (1 - (r1y + 1)) + (r2y + 1) + a23

        c3x = a11 * (templ_w - (r1x + 1)) + a12 * (templ_h - (r1y + 1)) + (r2x + 1) + a13
        c3y = a21 * (templ_w - (r1x + 1)) + a22 * (templ_h - (r1y + 1)) + (r2y + 1) + a23

        c4x = a11 * (1 - (r1x + 1)) + a12 * (templ_h - (r1y + 1)) + (r2x + 1) + a13
        c4y = a21 * (1 - (r1x + 1)) + a22 * (templ_h - (r1y + 1)) + (r2y + 1) + a23

        return np.array([
            [c1x, c1y],
            [c2x, c2y],
            [c3x, c3y],
            [c4x, c4y],
        ])
