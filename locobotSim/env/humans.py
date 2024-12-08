from locobotSim.env.humanMotionPlanners import *

class Humans:
    def __init__(
        self,
        num_humans,
        model,
        data,
        sites,
        get_robot_position,
        motion_planner="CostMapPlanner",
    ):
        self.human_indices = [i + 5 for i in range(num_humans)]

        self.model = model
        self.data = data
        self.sites = sites
        self.goals = None

        self.HUMAN_RADIUS = 0.2
        self.ROBOT_RADIUS = 0.3

        self.get_robot_position = get_robot_position

        AgntMotionPlanner = eval(motion_planner)

        self.agent_motion_planner = AgntMotionPlanner(
            model,
            self.human_indices,
            self.in_agent_collision,
            self.get_human_positions,
            self.get_robot_position,
            self.reset_goal,
            sites,
            num_humans=num_humans,
        )

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

    def in_agent_collision(self, position, exclude=tuple(), exclude_robot=False):
        if not exclude_robot:
            robot_pos = self.get_robot_position()
            if (
                np.linalg.norm(np.array(robot_pos) - np.array(position))
                < self.ROBOT_RADIUS + self.HUMAN_RADIUS
            ):
                return True

        if len(self.human_indices) == 0:
            return False

        human_positions = self.get_human_positions(
            exclude=[self.human_indices[idx] for idx in exclude]
        )

        is_colliding = (
            np.linalg.norm(human_positions - position, axis=1) < (3* self.HUMAN_RADIUS)
        )

        if np.any(is_colliding):
            return True

        return False

    def reset(self, starts=None, goals=None):
        shuffled_starts = list(self.sites.keys())
        np.random.shuffle(shuffled_starts)
        start_sites = [None for _ in range(len(self.human_indices))]

        for i, human_idx in enumerate(self.human_indices):
            if starts is not None:
                start = starts[i]
            else:
                start_site = shuffled_starts[i]
                start_sites[i] = start_site
                start = self.sites[start_site]
            self.model.site_pos[human_idx, :2] = start

        if goals is None:
            goals = []
            for i, human_idx in enumerate(self.human_indices):
                goal = np.random.choice(shuffled_starts)

                while goal == start_sites[i]:
                    goal = np.random.choice(shuffled_starts)

                goals.append(goal)

        for goal in goals:
            if not isinstance(goal, str):
                breakpoint()

        goal_coords = np.array([self.sites[goal] for goal in goals])

        self.goals = goals
        self.agent_motion_planner.update_goals(goal_coords, goals)

    def reset_goal(self, indices):
        if isinstance(indices, int):
            indices = [indices]

        current_site = self.goals[indices[0]]
        goals = list(self.sites.keys())
        goals.remove(current_site)
        np.random.shuffle(goals)

        for idx, i in enumerate(indices):
            self.goals[i] = goals[idx]
            self.agent_motion_planner.update_goal(i, self.sites[goals[idx]], goals[idx])

    def apply_human_action(self, i, position):
        self.model.site_pos[self.human_indices[i], :2] = position

    def step(self):
        if len(self.human_indices) > 0:
            self.agent_motion_planner.step()
