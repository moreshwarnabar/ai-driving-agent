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

import random
import sys

MAX_VIEW = 10 # DO NOT CHANGE THIS
INIT_LOCATION_WEIGHT = 1/5 # DO NOT CHANGE THIS 

class Environment:
    def __init__(self, height=10, width=5, occupancy=0.1, init=0.2, seed=5):
        self.height = height
        self.width = width
        self.occupancy = occupancy
        self.init = init
        self.cars = {}
        self.road = [[0 for _ in range(width)] for _ in range(height)]
        self.crash = None
        self.done = None
        
        self.actions = ['F', 'L', 'R', 'W']
        self.score = 0
        self.__generate_environment(seed)
        
    def __generate_environment(self, seed):
        """            
        Description
        -----------
        This function is used to populate self.road and self.cars
        self.cars is a dict that holds the location of all cars on the road including the autonomous agent's (idx 0)
        self.road is a 2D list (grid) representing the discretized road environment with the following markers:
            0: empty cell
            1: obstacle car
            2: autonomous agent ('A')
           -1: crash
        
        The convention for positions, like an array, is that (r,c) = (0,0) is the lower left corner, r increases
        vertically and c increases horizontally.
        
        """
        random.seed(seed)
        
        # Number of obstacles should not exceed OCCUPANCY cells in the environment
        n_obstacles = int(self.height * self.width * self.occupancy)
        
        coordinate_list = [(r, c) for r in range(self.height) for c in range(self.width)]
        obstacle_list = random.sample(coordinate_list, n_obstacles)
        
        # Mark the 'A'
        r = int(self.height * self.init)
        c = int(self.width * self.init)
        self.road[r][c] = 2 
        self.cars[0] = (r, c) #'A' location is stored in the 0th index of cars dict
        
        # Mark obstacle cars
        idx = 1
        for r, c in obstacle_list:
            if (r, c) != self.cars[0]:
                self.road[r][c] = 1 
                self.cars[idx] = (r, c)
                idx += 1 
    
    def get_next_cell(self, idx, action):
        """
        Description
        -----------
        This function finds a car's next location given a car's index and action it may perform.
        If the action performed leads a car outside the road, its next location is updated to None
        
        """
        loc = self.cars[idx]
        
        if action == 'F':
            if loc[0]+1 < self.height:
                return (loc[0]+1, loc[1])
            elif loc[0]+1 == self.height: # moving forward leads out of the road
                return None
        elif action == 'L':
            return (loc[0], loc[1]-1)
        elif action == 'R':
            return (loc[0], loc[1]+1)
        elif action == 'W':
            return loc
        
    def get_possible_next_cells(self, idx):
        """
        Description
        -----------
        This function finds all possible next locations given a car's index (except 'A''s).
        
        """
        next_cells = []
        for action in self.get_legal_actions(idx):
            next_loc = self.get_next_cell(idx, action)
            next_cells.append(next_loc)
        
        return next_cells

    def get_legal_actions(self, idx):
        """
        Description
        -----------
        This function finds all possible legal actions given a car's index by considering its current location.
        Actions leading to locations where a car (except 'A') is already present is considered illegal for other cars.
        Wait action is considered legal by default for other cars.
        
        """
        loc = self.cars[idx]
        legal_actions = []
        # Check all three directions a car can move to
        if loc[0]+1 < self.height and self.road[loc[0]+1][loc[1]] != 1: # Forward action
            legal_actions.append('F')
        elif loc[0]+1 == self.height: # moving forward leads out of the road
            legal_actions.append('F')
        
        if loc[1]-1 >= 0 and self.road[loc[0]][loc[1]-1] != 1: # Left action
            legal_actions.append('L')
            
        if loc[1]+1 < self.width and self.road[loc[0]][loc[1]+1] != 1: # Right action
            legal_actions.append('R')
        
        legal_actions.append('W') # Wait action
        
        return legal_actions
    
    
    def __update_environment(self):
        """
        Description
        -----------
        This function updates the position of all cars (excluding 'A''s) in the environment everytime the step function is called.
        A radom action is chosen among the available legal actions
        Cars that move forwad beyond the scope of the road are no longer considered part of the environment and their location updates to None.
        
        """
        
        for idx, loc in self.cars.items():
            if idx == 0 or loc == None:
                continue
            legal_actions = self.get_legal_actions(idx)
            action = random.choice(legal_actions)
            self.perform_car_action(idx, action)
            
    def perform_car_action(self, idx, action):
        """
        Description
        -----------
        This function updates the position of a car in self.cars (except 'A') as a result of the action performed in the environment.
        
        """
        loc = self.cars[idx]
        next_cell = self.get_next_cell(idx, action)
        if next_cell == None: # car has crossed the road
            self.road[loc[0]][loc[1]] = 0 # Erase current marker
        else:
            self.road[loc[0]][loc[1]] = 0 # Erase current marker
            self.road[next_cell[0]][next_cell[1]] = 1 # Add new marker
            
        self.cars[idx] = next_cell # update location of car in self.cars
            
    def perform_AA_action(self, action):
        """
        Description
        -----------
        This function updates the position of 'A' as a result of the action performed in the environment everytime the step function is called.
        
        """
        loc = self.cars[0]
        if action == 'F':
            if loc[0]+1 < self.height: # Next cell is within the scope of the road
                if self.road[loc[0]+1][loc[1]] == 1: # 'A' is crashing into a car
                    self.crash = True
                    if self.road[loc[0]][loc[1]] != 1:
                        self.road[loc[0]][loc[1]] = 0 # Erase current marker
                    self.road[loc[0]+1][loc[1]] = -1 # Add marker for crash
                    self.cars[0] = (-1, -1)
                else:
                    if self.road[loc[0]][loc[1]] != 1:
                        self.road[loc[0]][loc[1]] = 0 # Erase current marker
                    self.road[loc[0]+1][loc[1]] = 2 # Add new marker 
                    self.cars[0] = (loc[0]+1, loc[1])
                    self.score += 1
            else: # 'A' has successfully navigated till the end of the road
                self.done = True
                self.cars[0] = None
        elif action == 'L':
            if loc[1]-1 >= 0: # Next cell is within the scope of the road
                if self.road[loc[0]][loc[1]-1] == 1: # A is crashing into a car
                    self.crash = True
                    if self.road[loc[0]][loc[1]] != 1:
                        self.road[loc[0]][loc[1]] = 0 # Erase current marker
                    self.road[loc[0]][loc[1]-1] = -1 # Add marker for crash
                    self.cars[0] == (-1, -1)
                else:
                    if self.road[loc[0]][loc[1]] != 1:
                        self.road[loc[0]][loc[1]] = 0 # Erase current marker
                    self.road[loc[0]][loc[1]-1] = 2 # Add new marker 
                    self.cars[0] = (loc[0], loc[1]-1)
            else: # 'A' is moving out of the scope of the road (i.e. driving beyond the left shoulder of the road)
                self.crash = True
                if self.road[loc[0]][loc[1]] != 1:
                    self.road[loc[0]][loc[1]] = 0 # Erase current marker
                self.cars[0] == (-1, -1)     
        elif action == 'R':
            if loc[1]+1 < self.width: # Next cell is within the scope of the road
                if self.road[loc[0]][loc[1]+1] == 1: # A is crashing into a car
                    self.crash = True
                    if self.road[loc[0]][loc[1]] != 1:
                        self.road[loc[0]][loc[1]] = 0 # Erase current marker
                    self.road[loc[0]][loc[1]+1] = -1 # Add marker for crash
                    self.cars[0] == (-1, -1)
                else:
                    if self.road[loc[0]][loc[1]] != 1:
                        self.road[loc[0]][loc[1]] = 0 # Erase current marker
                    self.road[loc[0]][loc[1]+1] = 2 # Add new marker 
                    self.cars[0] = (loc[0], loc[1]+1)
            else: # 'A' is moving out of the scope of the road (i.e. driving beyond the right shoulder of the road)
                self.crash = True
                if self.road[loc[0]][loc[1]] != 1:
                    self.road[loc[0]][loc[1]] = 0 # Erase current marker
                self.cars[0] == (-1, -1)
        elif action == 'W':
            if self.road[loc[0]][loc[1]] == 1: # A is crashing into a car
                self.crash = True
                self.road[loc[0]][loc[1]] = 0 # Erase current marker
                self.road[loc[0]][loc[1]] = -1 # Add marker for crash
                self.cars[0] == (-1, -1)
        elif action == 'S':
            self.done = False
        else:
            raise ValueError(f"Incorrect action specified: {action}")
            
    def get_percept(self):
        """
        Description
        -------
        This function returns the percept - a (3 x 3) grid sized partial view of the road environment,
        where current location of 'A' is at the center of the grid. It has the following markers:
            0: empty cell
            1: obstacle car
            2: autonomous agent ('A')
            -1: edge of the road (left, right and bottom edge of the road)
            10: end of the road (top edge of the road)

        """
        partial_state = [[0 for _ in range(3)] for _ in range(3)]
        
        AA_loc = self.cars[0]
        for x, r in enumerate(range(AA_loc[0]+1, AA_loc[0]-2, -1)):
            for y, c in enumerate(range(AA_loc[1]-1, AA_loc[1]+2, 1)):
                if r < 0:
                    partial_state[x][y] = -1
                elif r == self.height:
                    partial_state[x][y] = 10
                elif c < 0 or c == self.width:
                    partial_state[x][y] = -1
                else:
                    partial_state[x][y] = self.road[r][c]     
                    
        return partial_state
        
    def step(self, action):
        """
        Description
        -----------
        Execute the action taken by 'A'. 
        Agents's actions are deterministic. However, environment's stochasticity lies in the uncerternity of location of other cars. 
        
        """
        self.__update_environment()
        self.perform_AA_action(action)
        if self.crash == True:
            print("There is a crash on the road!")
            self.display_environment()
            sys.exit(0)
        elif self.done == True:
            print("\'A\' has successfully navigated the road!")
            sys.exit(0)
        elif self.done == False:
            print("Stopping simulation!")
            sys.exit(0)
        # self.display_environment()
        # self.display_percept()

    def step_RL(self, action):
        """
        Description
        -----------
        Execute the action taken by 'A'. 
        Agents's actions are deterministic. However, environment's stochasticity lies in the uncerternity of location of other cars. 
        
        """
        self.__update_environment()
        self.perform_AA_action(action)
        if self.crash == True:
            #print("There is a crash on the road!")
            #self.display_environment()
            return -100, True
            sys.exit(0)
        elif self.done == True:
            #print("'A' has successfully navigated the road!")
            return 100, True
            sys.exit(0)
        elif self.done == False:
            #print("Stopping simulation!")
            sys.exit(0)
        # self.display_environment()
        return -1, False
        self.display_percept()
        
    def display_environment(self):
        """
        Description
        -----------
        Display the environment with static view of the cars including the autonomous agent 
        Example: Change in autonomous driving agent's location.
        
        You may use this for debugging only. The 'A' may not have access to the actual state as displayed in this function.
        
        """
        
        display = ""
        for r in range(self.height-1, -1, -1):
            # Display borders
            display += chr(43)
            for c in range(self.width):
                display += chr(45) + chr(45) + chr(45) + chr(43)
            display += "\n"
               
            # Display road 
            display += chr(124)
            for c in range(self.width):
                if self.road[r][c] == 1:
                    # ASCII 94 = ^
                    display += chr(32) + chr(94) + chr(32) + chr(124)
                elif self.road[r][c] == 2:
                    # ASCII 65 = A
                    display += chr(32) + chr(65) + chr(32) + chr(124)
                elif self.road[r][c] == -1:
                    # ASCII 42 = *
                    display += chr(32) + chr(42) + chr(32) + chr(124)
                else:
                    # ASCII 32 = \space
                    display += chr(32) + chr(32) + chr(32) + chr(124)
            display += "\n" 
        
        # Display border
        display += chr(43)
        for c in range(self.width):
            display += chr(45) + chr(45) + chr(45) + chr(43)
        display += "\n"
        
        print(display)
        return
    
    def display_percept(self):
        """
        Description
        -----------
        Display the (3 x 3) grid sized percept of the 'A' based on its current location.
        
        """
        percept = self.get_percept()
        
        height, width = len(percept), len(percept[0])
        display = ""
        for r in range(height):
            # Display borders
            display += chr(43)
            for c in range(width):
                display += chr(45) + chr(45) + chr(45) + chr(43)
            display += "\n"
            
            # Display road 
            display += chr(124)
            for c in range(width):
                if percept[r][c] == 1:
                    # ASCII 94 = ^
                    display += chr(32) + chr(94) + chr(32) + chr(124)
                elif percept[r][c] == 2:
                    # ASCII 65 = A
                    display += chr(32) + chr(65) + chr(32) + chr(124)
                elif percept[r][c] == -1:
                    # ASCII 120 = x
                    display += chr(32) + chr(120) + chr(32) + chr(124)
                elif percept[r][c] == 10:
                    # ASCII 71 = G
                    display += chr(32) + chr(71) + chr(32) + chr(124)
                else:
                    # ASCII 32 = \space
                    display += chr(32) + chr(32) + chr(32) + chr(124)
            display += "\n" 
        
        # Display border
        display += chr(43)
        for c in range(width):
            display += chr(45) + chr(45) + chr(45) + chr(43)
        display += "\n"
        
        print(display)
        return