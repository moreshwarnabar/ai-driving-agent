#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
# Licensing Information:  
# Please DO NOT DISTRIBUTE OR PUBLISH this project or the solutions for this project.
# We reserve the right to publish and provide access to this code.
# This project is built for the use of CSE 471 Introduction to AI class 
# instructed by Yu Zhang (yzhan442@asu.edu).
# 
# Attribution Information: The Autonomous Driving AI project was developed at Arizona State University Fall 2023
# The core project and autograder was primarily created by Akku Hanni
# (ahanni@asu.edu) and contributed by Kevin Vora (kvora1@asu.edu)

"""


import copy

"""
A state must define methods such as get_percept, generate_successor, get_legal_actions, etc. as required
depending on the type of agent for which an interface is being built.
States are used by the AutonomousDriving object to capture the actual state of the environment/road and
it can be used by agents to reason about driving.
Every agent type has a different state information available to it.
This is a super class for all types of state information.
    
"""

class ManualState():
    """
    The full state or percept is displayed to a user engaging in manual control. 
    You may modify this at your will but it is unnecessary.
    
    """    
    
    def __init__(self, env):
        self.__env = env
        
    def display_state(self):
        self.__env.display_environment() # Comment this if you do not wish to display the road at every step
        # self.__env.display_percept() # Comment this if you do not wish to display the percept at every step
        pass

class RandomState():
    """
    The full state or percept is displayed when random agent control is chosen. 
    You may modify this at your will but it is unnecessary.
    
    """   
    
    def __init__(self, env):
        self.__env = env
        
    def display_state(self):
        # self.__env.display_environment() # Comment this if you do not wish to display the road at every step
        self.__env.display_percept() # Comment this if you do not wish to display the percept at every step
        pass
    
class ReflexState():
    """
    A reflex agent chooses an action at each choice point by examining its alternatives via a percept of the state available to it.
    A reflex agent's percept is a (3 x 3) grid sized partial view of the road environment.

    """
    
    def __init__(self, env):
        self.__env = env
        
    def get_percept(self):
        """
        Description
        -------
        This function returns the percept - a (3 x 3) grid sized partial view of the road environment,
        where teh current location of 'A' is at the center of the grid. It has the following markers:
            0: empty cell
            1: other car
            2: autonomous agent ('A')
           -1: edge of the road (left, right and bottom edge of the road)
           10: end of the road (top edge of the road)

        """
        percept = [[0 for _ in range(3)] for _ in range(3)]
        
        AA_loc = self.__env.cars[0]
        for x, r in enumerate(range(AA_loc[0]+1, AA_loc[0]-2, -1)):
            for y, c in enumerate(range(AA_loc[1]-1, AA_loc[1]+2, 1)):
                if r < 0:
                    percept[x][y] = -1
                elif r == self.__env.height:
                    percept[x][y] = 10
                elif c < 0 or c == self.__env.width:
                    percept[x][y] = -1
                else:
                    percept[x][y] = self.__env.road[r][c]     
                    
        return percept
    
class ExpectimaxState():
    """
    An expectimax agent chooses an action at each choice point by evaluating the expected return for choosing from its available actions.
    Helper functions are defined to access the attributes of the environment. 
    We strongly recommend that you use the following functions to write your agent function and not access the environment variable directly.

    """
    
    def __init__(self, env):
        self.__env = env
        
    def get_legal_actions(self, car_index):
        """
        Description
        -------
        Returns all the legal actions for the given car.
        
        """
        if car_index == 0: # 'A' 
            return self.__env.actions # All actions are legal for 'A'
        elif car_index in self.__env.cars.keys():
            car_loc = self.__env.cars[car_index]
            if car_loc == None:
                return []
            else:
                return self.__env.get_legal_actions(car_index)
        else:
            raise Exception(f"Invalid car index specified: {car_index}")
    
    def generate_successor(self, car_index, action):
        """
        Description
        -------
        Returns the successor state after the specified car takes the action.
        
        """
        # Check that successors exist
        if self.is_done() or self.is_crash():
            raise Exception('Cannot generate a successor of a terminal state.')
        elif self.__env.cars[car_index] == None:
            raise Exception('Cannot generate a successor of a state with car index that is no longer on the road.')

        # Copy current state
        state = copy.deepcopy(ExpectimaxState(self.__env))

        # Let car's logic deal with its action's effects on the road
        if car_index == 0:  # 'A' is moving
            state.apply_AA_action(action)
        else:                # Other car is moving
            state.apply_action(car_index, action)

        return state
    
    def apply_AA_action(self, action):
        """
        Description
        -------
        Simulates the action that is performed by 'A'. 
        
        """
        self.__env.perform_AA_action(action)
        
    def apply_action(self, car_index, action):
        """
        Description
        -------
        Simulates the action that is performed by the given car. 
        
        """
        self.__env.perform_car_action(car_index, action)
        
    def get_num_cars(self):
        """
        Description
        -------
        Returns the number of cars initialized on the road.
        Note that this also includes count of cars that may have already crossed the road.
        
        """
        return len(self.__env.cars.items())
    
    def get_car_position(self, car_index):
        """
        Description
        -------
        Returns the location of the given car in the form of a tuple (row, col)
        
        """
        return self.__env.cars[car_index]
    
    def get_min_distance_to_goal(self, AA_loc):
        """
        Description
        -------
        Returns the shortest distance to cross the road by the 'A'.
        
        """
        return self.__env.height - AA_loc[0]
        
    def get_score(self):
        """
        Description
        -------
        Returns the score of a given state.
        
        """
        return self.__env.score
        
    def is_car_on_road(self, car_index):
        """
        Description
        -------
        Returns a boolean indicating if the given car is still on the road.
        
        """
        if car_index in self.__env.cars.keys():
            if self.__env.cars[car_index] == None:
                return False
            else:
                return True
        else:
            raise Exception(f"Invalid car index specified: {car_index}")
        
    def is_done(self):
        """
        Description
        -------
        Returns a boolean indicating if 'A' has successfully crossed the road.
        
        """
        return self.__env.done
    
    def is_crash(self):
        """
        Description
        -------
        Returns a boolean indicating if there is a crash on the road.
        
        """
        return self.__env.crash
    

class LearningState():
    """
    A learning agent chooses an action at each choice point based on the Q values approximated.
    In this project your learning agent is essentiually an ApproximateQLearningAgent
    Helper functions are defined to access the attributes of the environment. 
    We strongly recommend that you use the following functions to write your 
    agent function and not access the environment variable directly.

    """
    
    def __init__(self, env):
        self.__env = env
        
    def is_terminal(self):
        """
        Description
        -------
        Returns a boolean indicating if there is a crash on the road or 
        if the agent has successfully navigated to the end of the road.
        
        """
        if self.__env.done or self.__env.crash:
            return True
        else:
            return False
        
    def step(self, action):
        """
        Description
        -------
        Steps through the environment by having 'A' perform the given action. 
        
        """
        next_state = copy.deepcopy(LearningState(self.__env))
        reward, result = next_state.__env.step_RL(action)

        return next_state, reward, result
    
    def get_road_data(self):
        """
        Description
        -------
        Returns access to the road data with the following markers.
            0: empty cell
            1: obstacle car
            2: autonomous agent ('A')
           -1: crash
        
        """
        if self.is_terminal():
            return None
        else:
            return self.__env.road
    
    def get_car_locations(self):
        """
        Description
        -------
        Returns access to all the car locations.
        Location of a car is given by (r, c)
        where r is the index of rows and c is the index of columns in the road array.
        The 'A''s location is stored at index 0.
        If a car is no longer on the road, its location is None.
        
        
        """
        if self.is_terminal():
            return None
        else:
            return self.__env.cars
        
    def get_height(self):
        """
        Description
        -------
        Returns access to the height of the road.
        
        """
        return self.__env.height
    
    def get_width(self):
        """
        Description
        -------
        Returns access to the width of the road.
        
        """
        return self.__env.width