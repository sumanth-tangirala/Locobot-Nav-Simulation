import numpy as np
import heapq
from scipy.ndimage import convolve
from scipy.spatial import distance_matrix as get_distance_matrix


class AgentMotionPlanner:
    DISP_GRAN = 0.01

    def __init__(
        self,
        model,
        human_indices,
        in_env_collision,
        in_agent_collision,
        get_human_positions,
        reset_goal,
    ):
        self.model = model
        self.human_indices = human_indices
        self.in_env_collision = in_env_collision
        self.in_agent_collision = in_agent_collision
        self.get_human_positions = get_human_positions
        self.reset_goal = reset_goal

        self.goals = None

    def update_goals(self, goals):
        self.goals = goals

    def update_goal(self, i, goal):
        self.goals[i] = goal

    def apply_human_action(self, i, position):
        self.model.site_pos[self.human_indices[i], :2] = position

    def compute_distance_matrix(self, human_positions):
        human_positions = np.array(human_positions)

        return get_distance_matrix(human_positions, human_positions)

    def get_close_human_indices(self, agent_idx, distance_matrix, threshold):
        distances = distance_matrix[agent_idx]
        close_human_indices = np.where(distances < threshold)
        # Remove the agent itself
        close_human_indices = close_human_indices[0][
            close_human_indices[0] != agent_idx
        ]

        return close_human_indices

    def step(self):
        raise NotImplementedError()


class RVO(AgentMotionPlanner):
    MAX_DISP = 0.01
    DISP_GRAN = 0.0025
    LOOKAHEAD_DIST = 7
    COLL_AVOID_MAX_ANGLE = np.pi / 1.8
    MOVEMENT_MAX_ANGLE = np.pi / 2
    ANGLE_GRAN = np.pi / 18
    IS_SIMPLE = False
    MAX_STUCK_COUNTER = 70
    GOAL_WAIT_TIMESTEPS = 50

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.stuck_counter = [0] * len(self.human_indices)
        self.reset_counter = [None] * len(self.human_indices)

    def step(self):
        human_positions = self.get_human_positions()

        self.distance_matrix = self.compute_distance_matrix(human_positions)

        for i, position in enumerate(human_positions):
            old_position = position.copy()
            new_position = self.get_new_position(i, position)
            human_positions[i] = new_position
            self.apply_human_action(i, new_position)
            if np.linalg.norm(new_position - old_position) < 0.005:
                self.stuck_counter[i] += 1

                if self.stuck_counter[i] > self.MAX_STUCK_COUNTER:
                    self.reset_goal([i])
                    self.stuck_counter[i] = 0
            else:
                self.stuck_counter[i] = 0

        goal_distances = np.linalg.norm(human_positions - self.goals, axis=1)
        success_indices = np.where(goal_distances < 0.05)[0]

        for i in success_indices:
            if self.reset_counter[i] is None:
                self.reset_counter[i] = self.GOAL_WAIT_TIMESTEPS

        for i, counter in enumerate(self.reset_counter):
            if counter is not None:
                self.reset_counter[i] -= 1
                if self.reset_counter[i] == 0:
                    self.reset_goal(i)
                    self.reset_counter[i] = None

    def get_movement_vector(self, position, goal):
        goal_vector = goal - position

        goal_dist = np.linalg.norm(goal_vector)

        if goal_dist == 0:
            return np.array([0, 0])

        movement_vector = goal_vector / goal_dist

        if self.IS_SIMPLE:
            return movement_vector

        for disp in np.arange(min(self.LOOKAHEAD_DIST, goal_dist), 0, -1):
            for angle in np.arange(0, self.MOVEMENT_MAX_ANGLE, self.ANGLE_GRAN):
                for angle_direction in [-1, 1]:
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

                    new_movement_vector = np.array(
                        [np.cos(new_theta), np.sin(new_theta)]
                    )

                    if not self.in_env_collision(position + disp * new_movement_vector):
                        return new_movement_vector
        return None

    def get_new_position(self, agent_idx, position):
        goal = self.goals[agent_idx]
        movement_vector = self.get_movement_vector(position, goal)

        if movement_vector is None:
            return position

        for disp in np.arange(self.MAX_DISP, 0, -self.DISP_GRAN):
            for angle in np.arange(
                0, self.COLL_AVOID_MAX_ANGLE + self.ANGLE_GRAN, self.ANGLE_GRAN
            ):
                for angle_direction in [-1, 1]:

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

                    new_movement_vector = np.array(
                        [np.cos(new_theta), np.sin(new_theta)]
                    )

                    new_position = position + disp * new_movement_vector

                    is_reaching_goal = (
                        np.linalg.norm(new_position - goal) <= self.MAX_DISP
                    )

                    if not is_reaching_goal and (
                        self.in_env_collision(new_position)
                        or self.in_agent_collision(new_position, [agent_idx])
                    ):
                        continue

                    return new_position
        return position


class CostMapPlanner(AgentMotionPlanner):
    X_LIMITS = [-3, 4.5]
    Y_LIMITS = [-10.2, 10.2]

    # X_LIMITS = [-2.8, 0]
    # Y_LIMITS = [-1, 0]

    ENV_COLLISION_ACT_DIST = 1
    ENV_COLLISION_COST = 5

    AGENT_COLLISION_ACT_DIST = 2
    AGENT_COLLISION_COST = 5
    AGENT_AVOIDANCE_DIST = 50

    GOAL_REWARD = 0
    COLLISION_COST = 1e10

    IS_FIRST_STEP = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.x_coords = np.arange(
            self.X_LIMITS[0], self.X_LIMITS[1] + self.DISP_GRAN, self.DISP_GRAN
        )
        self.y_coords = np.arange(
            self.Y_LIMITS[0], self.Y_LIMITS[1] + self.DISP_GRAN, self.DISP_GRAN
        )

        self.m = len(self.x_coords)
        self.n = len(self.y_coords)

        self.generate_cost_kernels()
        self.cost_map = self.generate_cost_map(self.in_env_collision)

    def generate_cost_kernels(self):
        self.map_buffer = max(
            self.AGENT_COLLISION_ACT_DIST, self.ENV_COLLISION_ACT_DIST
        )
        self.env_collision_cost_kernel = self.generate_collision_kernel(
            self.ENV_COLLISION_ACT_DIST, self.ENV_COLLISION_COST
        )
        self.agent_collision_cost_kernel = self.generate_collision_kernel(
            self.AGENT_COLLISION_ACT_DIST, self.AGENT_COLLISION_COST
        )

    def generate_collision_kernel(self, act_dist, cost):
        kernel = np.zeros((1 + 2 * act_dist, 1 + 2 * act_dist))

        for i in range(kernel.shape[0]):
            for j in range(kernel.shape[1]):
                x = abs(i - act_dist)
                y = abs(j - act_dist)
                dist = np.linalg.norm([x, y])

                if dist == 0:
                    kernel[i][j] = self.COLLISION_COST
                else:
                    kernel[i][j] = cost / dist

        return kernel

    def visualize_map(self, map):
        import matplotlib.pyplot as plt

        plt.imshow(map)
        plt.show()

    def generate_cost_map(self, collision_checker):
        occupancy_grid = np.zeros(
            (
                self.m + 2 * self.map_buffer,
                self.n + 2 * self.map_buffer,
            ),
            dtype=np.float32,
        )

        positions_to_query = []
        for x in self.x_coords:
            for y in self.y_coords:
                positions_to_query.append([x, y])

        positions_to_query = np.array(positions_to_query)

        collision_points = (
            collision_checker(positions_to_query)
            .reshape((len(self.x_coords), len(self.y_coords)))
            .astype(np.float32)
        )

        occupancy_grid[
            self.map_buffer : self.m + self.map_buffer,
            self.map_buffer : self.n + self.map_buffer,
        ] = collision_points

        cost_map = convolve(occupancy_grid, self.env_collision_cost_kernel)

        cost_map[occupancy_grid == 1] = np.inf

        return cost_map

    def init_human_coordinates(self):
        human_positions = self.get_human_positions()
        self.human_coords = [None for _ in range(len(human_positions))]

        # Check for the closest point in the self.x_coords and self.y_coords
        for i, position in enumerate(human_positions):
            x_idx = np.argmin(np.abs(self.x_coords - position[0])) + self.map_buffer
            y_idx = np.argmin(np.abs(self.y_coords - position[1])) + self.map_buffer

            self.human_coords[i] = (x_idx, y_idx)

    def update_cost_map(self):
        latest_cost_map = self.cost_map.copy()

        for position in self.human_coords:
            latest_cost_map[
                position[0] - self.map_buffer : position[0] + self.map_buffer + 1,
                position[1] - self.map_buffer : position[1] + self.map_buffer + 1,
            ] += self.agent_collision_cost_kernel

            latest_cost_map[position[0]][position[1]] = self.COLLISION_COST

        return latest_cost_map

    def get_personal_cost_map(self, cost_map, curr_pos, goal_pos):
        cost_map = cost_map.copy()

        cost_map[
            curr_pos[0] - self.map_buffer : curr_pos[0] + self.map_buffer + 1,
            curr_pos[1] - self.map_buffer : curr_pos[1] + self.map_buffer + 1,
        ] -= self.agent_collision_cost_kernel

        cost_map[curr_pos[0]][curr_pos[1]] = 0

        cost_map[goal_pos[0]][goal_pos[1]] = self.GOAL_REWARD

        return cost_map

    def convert_coordinates_to_position(self, coords):
        i, j = coords
        x_index = i - self.map_buffer
        y_index = j - self.map_buffer
        x = self.x_coords[x_index]
        y = self.y_coords[y_index]
        return np.array([x, y])

    def perform_astar(self, cost_map, start, goal):
        open_set = []
        closed_set = set()

        heapq.heappush(open_set, (0, start))

        # Dictionaries to keep track of the path and costs
        came_from = {}
        g_score = {start: 0}

        # Heuristic function (Euclidean distance)
        def heuristic(a, b):
            return np.linalg.norm(np.array(a) - np.array(b))

        while open_set:
            # Pop the node with the lowest f_score
            f_current, current = heapq.heappop(open_set)

            # Check if the goal has been reached
            if current == goal:
                # Reconstruct the path from start to goal
                coords = current
                path = [coords]
                while coords in came_from:
                    coords = came_from[coords]
                    path.append(coords)
                path.reverse()

                return path

            closed_set.add(current)

            i, j = current

            # Explore neighboring cells (up, down, left, right and diagonals)
            neighbors = []
            for di, dj in [
                (-1, 0),
                (1, 0),
                (0, -1),
                (0, 1),
                (-1, -1),
                (-1, 1),
                (1, -1),
                (1, 1),
            ]:
                neighbor = (i + di, j + dj)

                if (
                    neighbor in closed_set
                    or neighbor[0] < 0
                    or neighbor[0] >= self.m
                    or neighbor[1] < 0
                    or neighbor[1] >= self.n
                    or cost_map[neighbor[0], neighbor[1]] == np.inf
                ):
                    continue

                # Calculate the tentative g_score
                tentative_g_score = (
                    g_score[current] + cost_map[neighbor[0], neighbor[1]]
                )

                # If this path to neighbor is better, record it
                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score = tentative_g_score + heuristic(neighbor, goal)
                    heapq.heappush(open_set, (f_score, neighbor))
        return None

    def get_new_coordinates(self, cost_map, agent_idx):

        start = self.human_coords[agent_idx]

        goal_pos = self.goals[agent_idx]

        goal_i = np.searchsorted(self.x_coords, goal_pos[0]) + self.map_buffer
        goal_j = np.searchsorted(self.y_coords, goal_pos[1]) + self.map_buffer
        goal = (goal_i, goal_j)

        cost_map = self.get_personal_cost_map(cost_map, start, goal)

        path = self.perform_astar(cost_map, start, goal)

        if path is not None:
            if len(path) > 1:
                next_pos = path[1]
            else:
                next_pos = path[0]
            return next_pos

        return start

    def move_human(self, agent_idx):
        new_coordinates = self.get_new_coordinates(self.latest_cost_map, agent_idx)
        self.human_coords[agent_idx] = new_coordinates
        new_position = self.convert_coordinates_to_position(new_coordinates)
        return new_position, new_coordinates

    def recompute_all_paths(self):
        if self.IS_FIRST_STEP:
            self.IS_FIRST_STEP = False
            self.init_human_coordinates()

        self.latest_cost_map = self.update_cost_map()

        for i, _ in enumerate(self.human_coords):
            new_position, new_coordinates = self.move_human(i)
            self.human_coords[i] = new_coordinates
            self.apply_human_action(i, new_position)

        # num_process = min(len(self.human_coords), cpu_count())

        # with Pool(num_process) as p:
        #     new_positions_coords = p.map(self.move_human, range(len(self.human_coords)))

        # for i, position_coords in enumerate(new_positions_coords):
        #     position, coords = position_coords
        #     self.human_coords[i] = coords
        #     self.apply_human_action(i, position)

    def plan_human_path(self, agent_idx, cost_map):
        start = self.human_coords[agent_idx]

        goal_pos = self.goals[agent_idx]

        goal_i = np.searchsorted(self.x_coords, goal_pos[0]) + self.map_buffer
        goal_j = np.searchsorted(self.y_coords, goal_pos[1]) + self.map_buffer
        goal = (goal_i, goal_j)

        cost_map = self.get_personal_cost_map(cost_map, start, goal)

        path = self.perform_astar(cost_map, start, goal)

        return path

    def init_human_paths(self, cost_map, distance_matrix=None):
        self.human_paths = []
        for agent_idx, _ in enumerate(self.human_coords):
            if distance_matrix is None:
                self.human_paths.append(
                    self.plan_human_path(
                        agent_idx,
                        cost_map,
                    )
                )
            else:
                close_human_indices = self.get_close_human_indices(
                    agent_idx, distance_matrix, self.AGENT_AVOIDANCE_DIST
                )
                # self.

    def reset_goal(self, agent_idx):
        goal_pos = self.goals[agent_idx]

        goal_i = np.searchsorted(self.x_coords, goal_pos[0]) + self.map_buffer
        goal_j = np.searchsorted(self.y_coords, goal_pos[1]) + self.map_buffer
        goal = (goal_i, goal_j)

        self.human_paths[agent_idx] = self.perform_astar(
            self.latest_cost_map, self.human_coords[agent_idx], goal
        )

    def execute_path(self, agent_idx, path):
        if path is None:
            return
        if len(path) > 1:
            new_coordinates = path[1]
        else:
            new_coordinates = path[0]

        new_position = self.convert_coordinates_to_position(new_coordinates)

        self.human_paths[agent_idx] = path[1:]
        self.human_coords[agent_idx] = new_coordinates
        self.apply_human_action(agent_idx, new_position)

        if np.linalg.norm(new_position - self.goals[agent_idx]) < 0.1:
            return True
        return False

    def compute_distance_matrix(self, human_coords):
        human_coords = np.array(human_coords)

        return get_distance_matrix(human_coords, human_coords)

    def economic_path_planning(self):
        cost_map = self.cost_map

        if self.IS_FIRST_STEP:
            self.IS_FIRST_STEP = False
            self.init_human_coordinates()
            self.init_human_paths(cost_map)

        is_cost_map_updated = False
        for agent_idx, path in enumerate(self.human_paths):
            agent_coordinates = self.human_coords[agent_idx]

            is_other_agent_close = False

            for other_agent_idx, other_agent_coords in enumerate(self.human_coords):
                if agent_idx == other_agent_idx:
                    continue

                if (
                    np.linalg.norm(
                        np.array(agent_coordinates) - np.array(other_agent_coords)
                    )
                    < self.AGENT_COLLISION_ACT_DIST
                ):
                    is_other_agent_close = True
                    break

            if path is not None and not is_other_agent_close and len(path) > 1:
                if self.execute_path(agent_idx, path):
                    self.reset_goal(agent_idx)
                continue

            if not is_cost_map_updated:
                cost_map = self.update_cost_map()
                is_cost_map_updated = True

            print("Recomputing paths for agent", agent_idx)

            self.human_paths[agent_idx] = self.plan_human_path(agent_idx, cost_map)

            self.execute_path(agent_idx, self.human_paths[agent_idx])

    def avoidance_path_planning(self):
        cost_map = self.cost_map

        if self.IS_FIRST_STEP:
            self.IS_FIRST_STEP = False
            self.init_human_coordinates()
            distance_matrix = self.compute_distance_matrix(self.human_coords)
            self.init_human_paths(cost_map, distance_matrix)

    def step(self):
        # self.recompute_all_paths()
        # self.economic_path_planning()
        self.avoidance_path_planning()
