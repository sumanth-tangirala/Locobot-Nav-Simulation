from numpy.linalg import norm

from locobotSim.env.agentMotionPlanner import *


def obq_line_distance(p3):
    p1 = np.array([3.0, 10.0])
    p2 = np.array([-0.4, -10.0])

    return np.cross(p2 - p1, p3 - p1) / norm(p2 - p1)


class Humans:
    def __init__(
        self,
        num_humans,
        model,
        data,
        sites,
        get_robot_position,
        motion_planner="RVO",
    ):
        self.human_indices = [i + 5 for i in range(num_humans)]

        self.model = model
        self.data = data
        self.sites = sites
        self.goals = None

        self.HUMAN_RADIUS = 0.2

        self.get_robot_position = get_robot_position

        AgntMotionPlanner = eval(motion_planner)

        self.agent_motion_planner = AgntMotionPlanner(
            model,
            self.human_indices,
            self.in_env_collision,
            self.in_agent_collision,
            self.get_human_positions,
            self.reset_goal,
        ),
        
    def set_human_positions(self, positions):
        for i, position in enumerate(positions):
            self.model.site_pos[self.human_indices[i], :2] = position

    def get_human_positions(self, exclude=None):
        indices = self.human_indices.copy()

        if exclude is not None:
            if isinstance(exclude, int):
                exclude = [exclude]

            for excluded_idx in exclude:
                indices.remove(excluded_idx)

        human_positions = self.data.site_xpos[indices, :2]

        return human_positions

    def in_env_collision(self, positions):
        is_single_point = len(positions.shape) == 1

        if is_single_point:
            positions = positions[None, :]

        x = positions[:, 0]
        y = positions[:, 1]

        oblique_cond = 0.1 - obq_line_distance(positions)

        exceeds_env_bounds = np.array([False for _ in range(positions.shape[0])])

        exceeds_env_bounds[x < -2.8] = True
        exceeds_env_bounds[y < -10.05] = True
        exceeds_env_bounds[y > 10.05] = True
        exceeds_env_bounds[oblique_cond <= 0] = True

        # if x < -2.8 or y < -10.05 or y > 10.05 or oblique_cond <= 0:  # wall conditions
        #     return True

        # if -2.8 <= x <= -0.06 and -0.225 <= y <= 6.825:  # stairs conditions
        #     return True

        in_collision_with_stairs = np.all(
            [-2.8 <= x, x <= -0.06, -0.225 <= y, y <= 6.825], axis=0
        )

        in_collision = np.any([exceeds_env_bounds, in_collision_with_stairs], axis=0)

        if is_single_point:
            return in_collision[0]

        return in_collision

    def in_agent_collision(self, position, exclude=tuple()):
        robot_pos = self.get_robot_position()
        human_positions = self.get_human_positions(
            exclude=[self.human_indices[idx] for idx in exclude]
        )

        if np.linalg.norm(np.array(robot_pos) - np.array(position)) < 2 * self.HUMAN_RADIUS:
            return True

        is_colliding = np.linalg.norm(human_positions - position, axis=1) < 3 * self.HUMAN_RADIUS

        if np.any(is_colliding):
            return True

        return False

    def reset(self):
        shuffled_starts = list(self.sites.keys())
        np.random.shuffle(shuffled_starts)

        goals = []

        for i, human_idx in enumerate(self.human_indices):
            start = shuffled_starts[i]
            self.model.site_pos[human_idx, :2] = self.sites[start]

            goal = np.random.choice(shuffled_starts)

            while goal == start:
                goal = np.random.choice(shuffled_starts)

            goals.append(goal)

        goal_coords = np.array([self.sites[goal] for goal in goals])

        self.goals = goals
        self.agent_motion_planner.update_goals(goal_coords)

    def reset_goal(self, indices):
        if isinstance(indices, int):
            indices = [indices]

        current_site = self.goals[indices[0]]
        goals = list(self.sites.keys())
        goals.remove(current_site)
        np.random.shuffle(goals)

        for idx, i in enumerate(indices):
            self.goals[i] = np.array(goals[idx])
            self.agent_motion_planner.update_goal(i, self.sites[goals[idx]])

    def apply_human_action(self, i, position):
        self.model.site_pos[self.human_indices[i], :2] = position

    def step(self):
        self.agent_motion_planner.step()
