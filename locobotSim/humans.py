import numpy as np
from numpy.linalg import norm


def obq_line_distance(p3):
    p1 = np.array([3.6, 10.55])
    p2 = np.array([-0.1, -10.55])

    return np.cross(p2 - p1, p3 - p1) / norm(p2 - p1)


class Humans:
    MAX_DISP = 0.02
    MAX_ANGLE = np.pi / 2
    DISP_GRAN = 0.01
    ANGLE_GRAN = np.pi / 18

    def __init__(self, num_humans, model, data, sites, get_robot_position):
        self.human_indices = [i + 4 for i in range(num_humans)]

        self.model = model
        self.data = data
        self.sites = sites

        self.get_robot_position = get_robot_position

    def is_valid_position(self, idx, position):
        x, y = position
        oblique_cond = 0.5 - obq_line_distance(position)

        if x < -2.8 or y < -10.05 or y > 10.05 or oblique_cond <= 0:  # wall conditions
            return False, "Wall"

        if np.linalg.norm(position - np.array([-1.9, 3.175])) < 0.5:
            return False, "Stairs"

        # if -2.8 <= x <= -0.06 and -0.225 <= y <= 6.825:  # stairs conditions
        #     return False, "Stairs"

        robot_pos = self.get_robot_position()

        if np.linalg.norm(np.array(robot_pos) - np.array(position)) < 0.5:
            return False, "Robot"

        is_colliding = (
            np.linalg.norm(
                self.data.site_xpos[self.human_indices, :2] - position, axis=1
            )
            < 0.8
        )

        is_colliding[idx] = False

        if np.any(is_colliding):
            return False, "Human"

        return True, "Valid"

    def rvo(self, idx, position):
        goal = self.goals[idx]

        # Get the vector from the position to the goal
        goal_vector = self.sites[goal] - position

        # print(goal_vector)

        # Get unit vector of the goal vector
        movement_vector = goal_vector / np.linalg.norm(goal_vector)

        # return position + self.MAX_DISP * movement_vector

        # breakpoint()

        for disp in np.arange(self.MAX_DISP, 0, -self.DISP_GRAN):
            for angle in np.arange(0, self.MAX_ANGLE, self.ANGLE_GRAN):
                for angle_direction in [-1, 1]:

                    # Determine the angle change ased on the direction
                    angle_change = angle * angle_direction

                    if movement_vector[0] == 0:
                        new_theta = (
                            np.sign(movement_vector[1]) * np.pi / 2 - angle_change
                        )
                    else:
                        new_theta = (
                            np.arctan2(movement_vector[1], movement_vector[0])
                            - angle_change
                        )

                    # Determine the new movement vector based on the angle change
                    new_movement_vector = np.array(
                        [np.cos(new_theta), np.sin(new_theta)]
                    )

                    # Correct the sign of the new movement vector based on the goal vector

                    # Determine the new position based on the displacement and angle and the goal vector
                    new_position = position + disp * new_movement_vector

                    # breakpoint()

                    valid, reason = self.is_valid_position(idx, new_position)

                    # print(reason)

                    if valid:
                        return new_position

        print("No valid position found")
        return position

    def reset(self, starts=None, goals=None):
        if starts is not None:
            assert len(starts) == len(
                self.human_indices
            ), f"Invalid human start data, expected {len(self.human_indices)} got {len(starts)}"

        else:
            starts = []

            for _ in range(len(self.human_indices)):
                starts.append(np.random.choice(list(self.sites.keys())))

        if goals is not None:
            assert len(goals) == len(
                self.human_indices
            ), f"Invalid human goal data, expected {len(self.human_indices)} got {len(goals)}"

        else:
            goals = []

            for i in range(len(self.human_indices)):
                goal = np.random.choice(list(self.sites.keys()))
                while goal == starts[i]:
                    goal = np.random.choice(list(self.sites.keys()))
                goals.append(goal)

        for i, human_idx in enumerate(self.human_indices):
            self.model.site_pos[human_idx, :2] = self.sites[starts[i]]

        self.goals = goals

    def apply_human_action(self, i, position):
        self.model.site_pos[self.human_indices[i], :2] = position

    def step(self):
        human_positions = self.data.site_xpos[self.human_indices, :2]

        for i, position in enumerate(human_positions):
            new_position = self.rvo(i, position)

            print(new_position)

            self.apply_human_action(i, new_position)
