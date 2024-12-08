import numpy as np
import heapq
from scipy.ndimage import convolve
from scipy.spatial import distance_matrix as get_distance_matrix
from collections import deque

from locobotSim.env import in_env_collision


class AgentMotionPlanner:
    def __init__(
        self,
        model,
        human_indices,
        in_agent_collision,
        get_human_positions,
        get_robot_position,
        reset_goal,
        sites,
        num_humans,
    ):
        self.model = model
        self.human_indices = human_indices
        self.in_agent_collision = in_agent_collision
        self.get_human_positions = get_human_positions
        self.get_robot_position = get_robot_position
        self.reset_goal = reset_goal
        self.sites = sites
        self.num_humans = num_humans

        self.goals = None
        self.goal_sites = None

    def update_goals(self, goals, goal_sites):
        self.goals = goals
        self.goal_sites = goal_sites

    def update_goal(self, i, goal, goal_site):
        self.goals[i] = goal
        self.goal_sites[i] = goal_site

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
    MAX_DISP = 0.007
    DISP_GRAN = 0.0035
    COLL_AVOID_MAX_ANGLE = np.pi / 1.2
    LOOKAHEAD_DIST = 7
    MOVEMENT_MAX_ANGLE = np.pi / 1.5
    ANGLE_GRAN = np.pi / 12
    IS_SIMPLE = False
    MAX_STUCK_COUNTER = 80
    STUCK_THRESHOLD = DISP_GRAN
    GOAL_WAIT_TIMESTEPS = 50
    ROBOT_ACT_DIST = 0.2
    ROBOT_ANGLE_DIST = np.pi/20
    GOAL_RADIUS=0.1

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.reset_counter = [None] * len(self.human_indices)
        self.displacements = np.zeros((len(self.human_indices), self.MAX_STUCK_COUNTER, 2))
        self.displacements.fill(np.inf)

    def update_displacements(self, i, old_position, new_position):
        displacement = new_position - old_position
        self.displacements[i] = np.roll(self.displacements[i], 1, axis=0)
        self.displacements[i][0] = displacement

    def reset_human_goal(self, i):
        self.reset_goal(i)
        self.reset_counter[i] = None
        self.displacements[i] = np.zeros((self.MAX_STUCK_COUNTER, 2))
        self.displacements[i].fill(np.inf)

    def is_human_stuck(self, i):
        if np.linalg.norm(self.displacements[i].sum(axis=0)) <= self.STUCK_THRESHOLD:
            return True

        return False

    def step(self):
        human_positions = self.get_human_positions()

        self.distance_matrix = self.compute_distance_matrix(human_positions)

        for i, position in enumerate(human_positions):
            if self.reset_counter[i] is not None:
                self.reset_counter[i] -= 1
                if self.reset_counter[i] == 0:
                    self.reset_human_goal(i)
                    self.reset_counter[i] = None
                continue
            old_position = position.copy()
            new_position = self.get_new_position(i, position)
            human_positions[i] = new_position
            self.apply_human_action(i, new_position)
            self.update_displacements(i, old_position, new_position)
            if self.is_human_stuck(i) and self.reset_counter[i] is None:
                self.reset_human_goal(i)

        goal_distances = np.linalg.norm(human_positions - self.goals, axis=1)
        success_indices = np.where(goal_distances < self.GOAL_RADIUS)[0]

        for i in success_indices:
            if self.reset_counter[i] is None:
                self.reset_counter[i] = self.GOAL_WAIT_TIMESTEPS

    def get_movement_vector(self, position, goal):
        goal_vector = goal - position
        goal_dist = np.linalg.norm(goal_vector)

        if goal_dist == 0:
            return 0

        movement_vector = goal_vector / goal_dist

        if movement_vector[0] == 0:
            movement_theta = np.sign(movement_vector[1]) * np.pi / 2
        else:
            movement_theta = np.arctan2(movement_vector[1], movement_vector[0])

        if self.IS_SIMPLE:
            return movement_theta

        robot_vector = self.get_robot_position() - position
        robot_dist = np.linalg.norm(robot_vector)
        check_robot_vec = robot_dist < self.ROBOT_ACT_DIST
        # check_robot_vec = False

        if check_robot_vec:
            if robot_vector[0] == 0:
                robot_theta = np.sign(robot_vector[1]) * np.pi / 2
            else:
                robot_theta = np.arctan2(robot_vector[1], robot_vector[0])
        else:
            robot_theta = 0

        for disp in np.arange(min(self.LOOKAHEAD_DIST, goal_dist), 0, -3):
            for angle in np.arange(0, self.MOVEMENT_MAX_ANGLE, self.ANGLE_GRAN):
                for angle_direction in [-1, 1]:
                    angle_change = angle * angle_direction
                    new_theta = movement_theta - angle_change
                    new_movement_vector = np.array(
                        [np.cos(new_theta), np.sin(new_theta)]
                    )
                    is_movement_toward_robot = check_robot_vec and np.abs(new_theta - robot_theta) < self.ROBOT_ANGLE_DIST

                    if is_movement_toward_robot:
                        continue

                    is_in_env_collision = in_env_collision(position + disp * new_movement_vector)
                    if not is_in_env_collision:
                        return new_theta
        return None

    def get_new_position(self, agent_idx, position):
        goal = self.goals[agent_idx]
        movement_theta = self.get_movement_vector(position, goal)

        if movement_theta is None:
            return position

        for angle in np.arange(
            0, self.COLL_AVOID_MAX_ANGLE + self.ANGLE_GRAN, self.ANGLE_GRAN
        ):
            for angle_direction in [-1, 1]:
                angle_change = angle * angle_direction
                new_theta = movement_theta - angle_change

                new_movement_vector = np.array(
                    [np.cos(new_theta), np.sin(new_theta)]
                )
                for disp in np.arange(self.MAX_DISP, 0, -self.DISP_GRAN):
                    new_position = position + disp * new_movement_vector

                    is_reaching_goal = np.linalg.norm(new_position - goal) < self.GOAL_RADIUS

                    is_in_env_collision = in_env_collision(new_position)
                    in_agent_collision = self.in_agent_collision(new_position, [agent_idx])

                    if is_reaching_goal and in_agent_collision:
                        return position

                    if not is_in_env_collision and not in_agent_collision:
                        return new_position
        return position


class CostMapPlanner(AgentMotionPlanner):
    DISP_GRAN = 0.01
    X_LIMITS = [-3, 4.5]
    Y_LIMITS = [-10.2, 10.2]

    ENV_COLLISION_ACT_DIST = int(.5/DISP_GRAN)
    ENV_COLLISION_COST = int(.25/DISP_GRAN)

    AGENT_COLLISION_ACT_DIST = int(0.4/DISP_GRAN)
    AGENT_COLLISION_COST = int(0/DISP_GRAN)

    COLLISION_COST = 1e10

    IS_FIRST = True

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
        self.cost_maps = self.load_cost_maps()

        self.cost_map_shape = (self.m + 2 * self.map_buffer, self.n + 2 * self.map_buffer)
        self.human_coords = [None for _ in range(self.num_humans)]

    def generate_cost_kernels(self):
        self.map_buffer = int(1/self.DISP_GRAN)
        # self.env_collision_cost_kernel = self.generate_collision_kernel(
        #     self.ENV_COLLISION_ACT_DIST, self.ENV_COLLISION_COST
        # )
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

    def load_cost_maps(self):
        goal_cost_maps = dict()

        for site in self.sites.keys():
            goal_site_cost_map = np.load(f'./data/costmaps/{site}.npy')
            goal_cost_maps[site] = goal_site_cost_map

        return goal_cost_maps

    def generate_cost_maps(self, collision_checker):
        print('Generating cost maps')
        occupancy_grid = np.zeros(
            (
                self.m + 2 * self.map_buffer,
                self.n + 2 * self.map_buffer,
            ),
            dtype=np.float32,
        )

        print('Generated occupancy grid')
        positions_to_query = []
        for x in self.x_coords:
            for y in self.y_coords:
                positions_to_query.append([x, y])

        positions_to_query = np.array(positions_to_query)

        print('Generated positions to query')

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

        print('size of cost map:', cost_map.size)

        print('Generated env cost_map')

        goal_cost_maps = dict()

        sites = {"werblin": self.sites["werblin"]}
        for site, site_coords in sites.items():
            goal_site_cost_map = cost_map.copy()
            x_idx = np.argmin(np.abs(self.x_coords - site_coords[0])) + self.map_buffer
            y_idx = np.argmin(np.abs(self.y_coords - site_coords[1])) + self.map_buffer

            visited = set()
            queue = deque([(x_idx, y_idx, 0)])

            while queue:
                x, y, cost = queue.popleft()
                if (x, y) in visited:
                    continue
                visited.add((x, y))

                goal_site_cost_map[x, y] += cost
                for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                    new_x, new_y = x + dx, y + dy

                    if new_x < 0 or new_x >= self.m + 2 * self.map_buffer or new_y < 0 or new_y >= self.n + 2 * self.map_buffer:
                        continue
                    if goal_site_cost_map[new_x, new_y] == np.inf or (new_x, new_y) in visited:
                        continue
                    queue.append((new_x, new_y, cost + 1))
                for dx, dy in [(1, 1), (-1, 1), (1, -1), (-1, -1)]:
                    new_x, new_y = x + dx, y + dy
                    if new_x < 0 or new_x >= self.m + 2 * self.map_buffer or new_y < 0 or new_y >= self.n + 2 * self.map_buffer:
                        continue
                    if goal_site_cost_map[new_x, new_y] == np.inf or (new_x, new_y) in visited:
                        continue
                    queue.append((new_x, new_y, cost + 1.2))

            goal_cost_maps[site] = goal_site_cost_map
            print('Generated cost map for site', site)

            # Save cost map as a numpy file
            np.save(f'./data/costmaps/{site}.npy', goal_site_cost_map)
        raise NotImplementedError()
        return goal_cost_maps

    def init_human_coordinates(self):
        human_positions = self.get_human_positions()
        # Check for the closest point in the self.x_coords and self.y_coords
        for i, position in enumerate(human_positions):
            x_idx = np.argmin(np.abs(self.x_coords - position[0])) + self.map_buffer
            y_idx = np.argmin(np.abs(self.y_coords - position[1])) + self.map_buffer

            self.human_coords[i] = (x_idx, y_idx)

    def get_robot_coords(self):
        robot_position = self.get_robot_position()
        x_idx = np.argmin(np.abs(self.x_coords - robot_position[0])) + self.map_buffer
        y_idx = np.argmin(np.abs(self.y_coords - robot_position[1])) + self.map_buffer
        return x_idx, y_idx

    def get_agents_cost_map(self):
        robot_coords = self.get_robot_coords()

        agents_cost_map = np.zeros(self.cost_map_shape)

        agents_cost_map[
            robot_coords[0] - self.AGENT_COLLISION_ACT_DIST : robot_coords[0] + self.AGENT_COLLISION_ACT_DIST + 1,
            robot_coords[1] - self.AGENT_COLLISION_ACT_DIST : robot_coords[1] + self.AGENT_COLLISION_ACT_DIST + 1,
        ] += self.agent_collision_cost_kernel

        agents_cost_map[robot_coords[0], robot_coords[1]] = self.COLLISION_COST

        for human_coord in self.human_coords:
            agents_cost_map[
                human_coord[0] - self.AGENT_COLLISION_ACT_DIST : human_coord[0] + self.AGENT_COLLISION_ACT_DIST + 1,
                human_coord[1] - self.AGENT_COLLISION_ACT_DIST : human_coord[1] + self.AGENT_COLLISION_ACT_DIST + 1,
            ] += self.agent_collision_cost_kernel

            agents_cost_map[human_coord[0], human_coord[1]] = self.COLLISION_COST


        return agents_cost_map

    def convert_coordinates_to_position(self, coords):
        i, j = coords
        x_index = i - self.map_buffer
        y_index = j - self.map_buffer
        x = self.x_coords[x_index]
        y = self.y_coords[y_index]
        return np.array([x, y])

    def move_human(self, agent_idx, agents_cost_map):
        goal = self.goal_sites[agent_idx]

        cost_map = self.cost_maps[goal] + agents_cost_map

        # Remove the agent from the cost map

        curr_pos = self.human_coords[agent_idx]

        cost_map[
            curr_pos[0] - self.AGENT_COLLISION_ACT_DIST : curr_pos[0] + self.AGENT_COLLISION_ACT_DIST + 1,
            curr_pos[1] - self.AGENT_COLLISION_ACT_DIST : curr_pos[1] + self.AGENT_COLLISION_ACT_DIST + 1,
        ] -= self.agent_collision_cost_kernel

        cost_map[curr_pos[0], curr_pos[1]] = 0

        min_cost = np.inf
        next_position = None
        for i, j in [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, 1), (1, -1), (-1, -1)]:
            new_x, new_y = curr_pos[0] + i, curr_pos[1] + j
            if new_x < 0 or new_x >= (self.m + self.map_buffer) or new_y < 0 or new_y >= (self.n + self.map_buffer):
                continue

            if cost_map[new_x][new_y] < min_cost:
                min_cost = cost_map[new_x][new_y]
                next_position = (new_x, new_y)

        if next_position is None:
            next_position = curr_pos

        return self.convert_coordinates_to_position(next_position), next_position



    def step(self):
        if self.IS_FIRST:
            self.init_human_coordinates()
            self.IS_FIRST = False


        agents_cost_map = self.get_agents_cost_map()

        for i in range(self.num_humans):
            new_position, new_coordinates = self.move_human(i, agents_cost_map)
            self.apply_human_action(i, new_position)
            self.human_coords[i] = new_coordinates
