import numpy as np
import random
from enum import Enum, IntEnum
    
import itertools
from collections import defaultdict

arrows = {0:8680, 1:11008, 2:11009, 3:8678, 4:11011, 5:11010}
arrows_agent = {0:11157, 1:11016, 2:11017, 3: 11013, 4:11019, 5:11018 }


class MovingObject:

    def __init__(self, initial_r, initial_c):
        self.appearance = random.normalvariate(Appearance.MOVING.value, 0.1)
        self.direction = random.randint(0, 5)
        self.position = initial_r, initial_c

        # When colliding with something not movable, what is the change of direction?
        self.stir = random.choice( [-1,1] )

    @property
    def position(self):
        return self._r, self._c

    @position.setter
    def position(self, pos):
        self._r, self._c = pos

    @property
    def direction(self):
        return self._angle

    @direction.setter
    def direction(self, angle):
        self._angle = angle%6

    def __repr__(self):
        return chr(arrows[self.direction])


class Agent(MovingObject):

    appearance = 4

    def __repr__(self):
        return chr(arrows_agent[self.direction])


class Appearance(Enum):

    EMPTY = 0
    OBSTACLE = 0.5
    MOVABLE = 1
    MOVING = 1.5


class Physics(Enum):

    EMPTY = 0
    OBSTACLE = 1
    MOVABLE = 2
    MOVING = 3
    AGENT = 4


class HexaWorld:

    """ Creates a single agent in a Hexagonal grid, composed of obstacles, movable obstacles, and moving objects. """
    def __init__(self, size, ratio_moving_objects = 0.05, ratio_obstacles = 0.05, ratio_movable = 0.05,
                 range_observation = 3, collision_free = False):

        # Account for borders
        self.size = size + 1
        self.range_observation = range_observation
        
        # Build Empty environment
        self.physical_envir = np.ones((self.size, self.size), dtype=int)*Physics.EMPTY.value
        self.visual_envir = np.ones((self.size, self.size), dtype=float)*Appearance.EMPTY.value

        # Fill in borders
        self.physical_envir[:, 0] = Physics.OBSTACLE.value
        self.physical_envir[:, -1] = Physics.OBSTACLE.value
        self.physical_envir[0, :] = Physics.OBSTACLE.value
        self.physical_envir[-1, :] = Physics.OBSTACLE.value

        self.visual_envir[:, 0] = np.random.normal(Appearance.OBSTACLE.value, 0.1, self.size)
        self.visual_envir[:, -1] = np.random.normal(Appearance.OBSTACLE.value, 0.1, self.size)
        self.visual_envir[0, :] = np.random.normal(Appearance.OBSTACLE.value, 0.1, self.size)
        self.visual_envir[-1, :] = np.random.normal(Appearance.OBSTACLE.value, 0.1, self.size)

        # Add obstacles
        number_obstacles = int(ratio_obstacles * (size ** 2))
        empty_cells_coordinates = np.where( self.physical_envir == Physics.EMPTY.value )
        selected_indices = np.random.choice( np.arange(len(empty_cells_coordinates[0])), number_obstacles )

        empty_cells_coordinates = empty_cells_coordinates[0][selected_indices], empty_cells_coordinates[1][selected_indices]
        self.visual_envir[empty_cells_coordinates[0],
                          empty_cells_coordinates[1]]\
                            = np.random.normal(Appearance.OBSTACLE.value, 0.1, len(selected_indices))

        self.physical_envir[empty_cells_coordinates[0], empty_cells_coordinates[1]] = Physics.OBSTACLE.value

        # Add movable objects
        number_mov_objects = int(ratio_movable * (size ** 2))
        empty_cells_coordinates = np.where(self.physical_envir == Physics.EMPTY.value)
        selected_indices = np.random.choice(np.arange(len(empty_cells_coordinates[0])), number_mov_objects)

        empty_cells_coordinates = empty_cells_coordinates[0][selected_indices], empty_cells_coordinates[1][
            selected_indices]
        self.visual_envir[empty_cells_coordinates[0],
                          empty_cells_coordinates[1]] \
            = np.random.normal(Appearance.MOVABLE.value, 0.1, len(selected_indices))

        self.physical_envir[empty_cells_coordinates[0], empty_cells_coordinates[1]] = Physics.MOVABLE.value

        # Add moving objects
        number_mov_objects = int(ratio_moving_objects * (size ** 2))
        self.moving_objects = []
        empty_cells_coordinates = np.where(self.physical_envir == Physics.EMPTY.value)
        selected_indices = np.random.choice(np.arange(len(empty_cells_coordinates[0])), number_mov_objects)

        for index in selected_indices:

            r, c = empty_cells_coordinates[0][index], empty_cells_coordinates[1][index]
            moving_object = MovingObject(r, c)
            self.visual_envir[r, c] = moving_object.appearance
            self.physical_envir[r, c] = Physics.MOVING.value

            self.moving_objects.append(moving_object)

        # Agent
        empty_cells_coordinates = np.where(self.physical_envir == Physics.EMPTY.value)
        index = np.random.choice(np.arange(len(empty_cells_coordinates[0])), 1)
        r, c = empty_cells_coordinates[0][index][0], empty_cells_coordinates[1][index][0]


        self.agent = Agent(r, c)
        self.visual_envir[r, c] = self.agent.appearance
        self.physical_envir[r, c] = Physics.AGENT.value

    def step(self, angle, forward):

        # Move agent
        self.agent.direction += angle

        if forward == 1:
            r, c = self.agent.position
            next_r, next_c, next_cell = self.get_next_cell(r, c, self.agent.direction)

            if next_cell == Physics.OBSTACLE.value or next_cell == Physics.MOVING.value:
                pass

            elif next_cell == Physics.MOVABLE.value:
                next_next_r, next_next_c, next_next_cell = self.get_next_cell(next_r, next_c, self.agent.direction)

                if next_next_cell == Physics.EMPTY.value:
                    self.move((next_r, next_c), (next_next_r, next_next_c))
                    self.move(self.agent.position, (next_r, next_c), self.agent)

            else:
                self.move( self.agent.position, (next_r, next_c), self.agent )


        # Move moving objects
        for mov_obj in self.moving_objects:

            r, c = mov_obj.position
            next_r, next_c, next_cell = self.get_next_cell(r, c, mov_obj.direction)

            if next_cell == Physics.OBSTACLE.value or next_cell == Physics.MOVING.value or next_cell == Physics.AGENT.value:

                mov_obj.direction += mov_obj.stir

            elif next_cell == Physics.MOVABLE.value:
                next_next_r, next_next_c, next_next_cell = self.get_next_cell(next_r, next_c, mov_obj.direction)

                if next_next_cell == Physics.EMPTY.value:

                    self.move((next_r, next_c), (next_next_r, next_next_c))
                    self.move(mov_obj.position, (next_r, next_c), mov_obj)

                else:
                    mov_obj.direction += mov_obj.stir


            else:
                self.move(mov_obj.position, (next_r, next_c), mov_obj)

        # get observation
        observation = self.observe()

        # get rewards
        reward = 0

        # test if the game terminated
        done = False

        return observation, reward, done

    def move(self, pos_start, pos_end , obj = None):

        r, c = pos_start
        next_r, next_c = pos_end

        self.physical_envir[r, c], self.physical_envir[next_r, next_c] = self.physical_envir[next_r, next_c], self.physical_envir[r, c]
        self.visual_envir[r, c], self.visual_envir[next_r, next_c] = self.visual_envir[next_r, next_c], self.visual_envir[r, c]

        if self.physical_envir[next_r, next_c] in [ Physics.AGENT.value, Physics.MOVING.value ] :
            obj.position = next_r, next_c


    def observe(self):

        obs = []

        # iterate over range
        for obs_range in range(1, self.range_observation + 1):

            row, col = self.agent.position
            angle = self.agent.direction

            # go to start
            for i in range(obs_range):
                row, col, _ = self.get_next_cell(row, col, (angle - 1) % 6)


            if 0 <= row < self.size and 0 <= col < self.size:
                obs.append(self.visual_envir[row, col])
            else:
                obs.append(0)

            # move first segment
            for i in range(obs_range):
                row, col, _ = self.get_next_cell(row, col, (angle + 1) % 6)

                if 0 <= row < self.size and 0 <= col < self.size:
                    obs.append(self.visual_envir[row, col])
                else:
                    obs.append(0)

            # move second segment
            for i in range(obs_range):
                row, col, _ = self.get_next_cell(row, col, (angle + 2) % 6)

                if 0 <= row < self.size and 0 <= col < self.size:
                    obs.append(self.visual_envir[row, col])
                else:
                    obs.append(0)

        return np.asarray(obs)

    def get_next_cell(self, row, col, angle):

        row_new, col_new = row, col

        if angle == 0:
            col_new += 1
        elif angle == 1:
            row_new -= 1
            col_new += row%2
        elif angle == 2:
            row_new -= 1
            col_new += row%2 - 1
        elif angle == 3:
            col_new -= 1
        elif angle == 4:
            row_new += 1
            col_new += row%2 - 1
        else:
            row_new += 1
            col_new += row%2

        if 0 <= row_new < self.size and 0 <= col_new < self.size:
            next_cell = self.physical_envir[row_new, col_new]
        else:
            next_cell = 0

        return row_new, col_new, next_cell

    def __repr__(self):


        full_repr = ""

        for r in range(self.size):
            if r%2 == 0:
                line = ""
            else:
                line = "{0:1}".format("")

            for c in range(self.size):

                if self.physical_envir[r,c] == Physics.EMPTY.value:
                    repr = chr(9900)

                elif self.physical_envir[r,c] == Physics.OBSTACLE.value:
                    repr = 'X'

                elif self.physical_envir[r,c] == Physics.MOVABLE.value:
                    repr = chr(9632)

                elif self.physical_envir[r,c] == Physics.MOVING.value:
                    repr  = [ str(elem) for elem in self.moving_objects if elem.position == (r,c) ][0]

                else:
                    repr = str(self.agent)

                line += "{0:2}".format(repr)

            full_repr += line + "\n"

        return full_repr

class ActionSampler:

    def __init__(self, number_actions):

        forward = [0,1]
        rotation = [-1, 0, 1]

        self.available_action_sequences = defaultdict(list)

        all_actions = itertools.product(rotation, forward)
        all_trajectories = itertools.product(all_actions, repeat=number_actions)

        for action_sequence in all_trajectories:

            row, col, angle = 0, 0, 0

            for rot, forw in action_sequence:

                angle = (angle + rot)%6

                if forw != 0:

                    row, col = self.get_proximal_coordinate(row, col, angle)

            self.available_action_sequences[(row, col)].append(action_sequence)

        self.number_actions = number_actions
        self.counter_actions = 0

        self.current_trajectory = []

    def get_proximal_coordinate(self, row, col, angle):

        row_new, col_new = row, col

        if angle == 0:
            col_new += 1
        elif angle == 1:
            row_new -= 1
            col_new += row%2
        elif angle == 2:
            row_new -= 1
            col_new += row%2 - 1
        elif angle == 3:
            col_new -= 1
        elif angle == 4:
            row_new += 1
            col_new += row%2 - 1
        else:
            row_new += 1
            col_new += row%2

        return row_new, col_new

    def sample(self):

        if len(self.current_trajectory) == 0:

            #Pick an end-point, and the different trajectories leading to it
            possible_trajectories = random.choice(list(self.available_action_sequences.values()))

            #Pick trajectory
            traj = random.choice(possible_trajectories)

            self.current_trajectory = list(traj)

        action = self.current_trajectory.pop(0)

        return action



