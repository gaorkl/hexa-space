import numpy as np
import random
import math
    
import itertools
from collections import defaultdict


class HexaWorld:
    
    def __init__(self, size, mode='random_float', ratio_obstacles = 0.1, range_observation = 3):
        
        self.size = size 
        self.range_observation = range_observation
        
        # Build Physical Environment
        self.envir = np.zeros((self.size, self.size), dtype = int)
        self.envir[:int(ratio_obstacles*size)] = 1
        
        self.envir = self.envir.reshape(-1, 1)
        
        np.random.shuffle(self.envir)
        self.envir = self.envir.reshape(self.size, self.size)
        
        # Place obstacle outside
        self.envir[:,0] = 1
        self.envir[:,-1] = 1
        self.envir[0,:] = 1
        self.envir[-1,:] = 1
        
        # Agent Initial Position
        self.position_agent = [ int(self.size/2), int(self.size/2), 0]
        self.envir[self.position_agent[0], self.position_agent[1]] = 3
        
        # Visual
        if mode == 'random_float':
            self.envir_visual = np.random.uniform(0, 1, self.envir.shape)
        elif mode == 'random_int':
            self.envir_visual = np.random.randint(1, 9, self.envir.shape)
#         self.envir_visual[self.envir == 0] = 0
#         self.envir_visual[self.position_agent[0], self.position_agent[1]] = 42
        
        # Display stuff
        self.display_dict={0:'.', 1:'X', 2:'O', 3: 'A'}
    
    def __repr__(self):
        repr = ""
        for r in range(self.size):
            if r%2 == 0:
                line = ""
            else:
                line = " "
            
            for c in range(self.size):
                line += self.display_dict[self.envir[r,c]]+ " "
                
            repr += line + "\n"
            
        repr += "-"*(self.size)
        repr += "\n"
        
        
        for r in range(self.size):
            if r%2 == 0:
                line = ""
            else:
                line = " "
            
            for c in range(self.size):
                line += str(self.envir_visual[r,c])+ " "
                
            repr += line + "\n"
            
        return repr
        
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
        
    def move(self, row, col, angle):
                
        row_new, col_new = self.get_proximal_coordinate(row, col, angle)
        
        if self.envir[row_new, col_new] == 0:
            
            self.envir[row_new, col_new] = self.envir[row, col]
            self.envir[row, col] = 0
            
            return row_new, col_new
        
        else:
            return row, col
        
    def observe(self):
        
        
        obs = []
        
        row, col, angle = self.position_agent
        obs.append(self.envir_visual[row, col])
        
        
        # iterate over range
        for obs_range in range(1, self.range_observation + 1):
        
            row, col, angle = self.position_agent
            
            # go to start
            for i in range(obs_range):
                row, col = self.get_proximal_coordinate(row, col, (angle - 1)%6 )
                
        
            if 0 < row < self.size and 0 < col < self.size:
                obs.append(self.envir_visual[row, col])
            else:
                obs.append(0)
            
            # move first segment
            for i in range(obs_range):
                row, col = self.get_proximal_coordinate(row, col, (angle + 1)%6 )

                if 0 < row < self.size and 0 < col < self.size:
                    obs.append(self.envir_visual[row, col])
                else:
                    obs.append(0)
                    
            # move second segment
            for i in range(obs_range):
                row, col = self.get_proximal_coordinate(row, col, (angle + 2)%6 )

                if 0 < row < self.size and 0 < col < self.size:
                    obs.append(self.envir_visual[row, col])
                else:
                    obs.append(0)
            
            
        return obs
                
        
            
    
    def step(self, angle, forward):
        
        # change angle
        self.position_agent[2] = (self.position_agent[2] + angle) % 6
        
        while forward != 0:
            
#             self.envir_visual[self.position_agent[0], self.position_agent[1]] = 0
            row, col = self.move( *self.position_agent )
            
            self.position_agent = [row, col, self.position_agent[2]]
#             self.envir_visual[self.position_agent[0], self.position_agent[1]] = 42
            forward -= 1

        observation = self.observe()
        
        return observation

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