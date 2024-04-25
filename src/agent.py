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
from collections import Counter

import numpy as np


class Agent:
    """
    An agent must define a get_action method, but may also define
    other methods which will be called if they exist. 
    This is a super class for any agent type.
        
    """

    def __init__(self):
        """
        Description
        -----------
        The list of available actions for all cars are:
        Forward - 'F'
        Left - 'L'
        Right - 'R'
        Wait - 'W'

        """
        self.available_actions = ['F', 'L', 'R', 'W']

    def get_action(self, state):
        """
        The Agent will receive a State of the environment (based on the agent type) and
        must return an action from the available actions {Forward - 'F', Left - 'L', Right - 'R' and Wait - 'W'}.
        """
        pass


class ManualAgent(Agent):
    """
    A manual agent is used to control the Autonomous Agent ('A') manually by the user.

    """

    def get_action(self, percept):
        "*** YOUR CODE HERE ***"
        print("Enter Action (Forward - 'F', Left - 'L', Right - 'R', Wait - 'W', Stop - 'S'):\n")
        action = input()
        return action


class RandomAgent(Agent):
    """
    A random agent chooses an action randomly at each choice point from the list of available actions.

    """

    def get_action(self, percept):
        "*** YOUR CODE HERE ***"
        action = random.choice(self.available_actions)  # should choose an action among the legal actions available
        print(f"Random action chosen: {action}")
        input("Press enter to step through.")
        return action


class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by following simple rules

    """

    def get_action(self, percept):
        """
        Description
        ----------
        This function returns the reflex action given the current percept. 
        The percept is essentially a (3 x 3) grid sized partial view of the road environment, with 'A' at the center i.e. at index (1, 1).
        
        """
        "*** YOUR CODE HERE ***"
        # Check if any moving action is legal
        action = 'W'
        if self.move_fwd(percept):
            action = 'F'
        elif self.move_left(percept):
            action = 'L'
        elif self.move_right(percept):
            action = 'R'

        print(f'{percept} {action}')
        return action

    def move_fwd(self, percept):
        # If nxt cell is the goal
        if percept[0][1] == 10:
            return True
        # if a car could possibly move into the cell
        if percept[0][0] == 1 or percept[0][1] == 1 or percept[0][2] == 1:
            return False
        else:
            return True

    def move_left(self, percept):
        # if left cell is out of bounds or a car could move into it
        return not (percept[1][0] == -1 or percept[1][0] == 1 or percept[2][0] == 1)

    def move_right(self, percept):
        # if right cell is out of bounds or a car could move into it
        return not (percept[1][2] == -1 or percept[1][2] == 1 or percept[2][2] == 1)


class ExpectimaxAgent(Agent):
    """
    An expectimax agent chooses an action at each choice point based on the expectimax algorithm.
    The choice is dependent on the self.evaluationFunction.
    
    All other cars should be modeled as choosing uniformly at random from their legal actions.

    """

    def __init__(self, depth=3):
        self.index = 0  # 'A' is always agent index 0
        self.depth = int(depth)
        super().__init__()

    def evaluation_function(self, road_state):
        """
        Description
        ----------
        This function returns a score (float) given a state of the road.

        """
        "*** YOUR CODE HERE ***"
        if road_state.is_done():
            return 10000
        if road_state.is_crash():
            return 0

        agent_loc = road_state.get_car_position(self.index)
        dist_to_goal = road_state.get_min_distance_to_goal(agent_loc)
        current_score = road_state.get_score()

        exp_score = current_score + (1000 / dist_to_goal)
        return exp_score

    def get_action(self, road_state):
        """
        Description
        ----------
        This function returns the expectimax action using self.depth and self.evaluationFunction.
        All other cars should be modeled as choosing uniformly at random from their
        legal moves.

        """
        "*** YOUR CODE HERE ***"
        num_of_agents = road_state.get_num_cars()

        def value(state, agent, depth):
            # If current state is a terminal state: return the expected value and no action
            if state.is_done() or state.is_crash() or depth == self.depth:
                return self.evaluation_function(state), None

            # If it is the turn of our Agent: find the max expected value
            if agent == self.index:
                return max_value(state, depth)
            # Else, it is the turn of one of the cars: find the total expected value
            else:
                return exp_value(state, depth, agent)

        def max_value(state, depth):
            utility = float('-inf')
            action = None

            # for every possible action: find the maximum expected value
            for cur_action in state.get_legal_actions(self.index):
                # max agent will choose after chance agent has calculated the expected value
                exp_val = value(state.generate_successor(self.index, cur_action), 1, depth)[0]
                if exp_val > utility:
                    utility, action = exp_val, cur_action

            return utility, action

        def exp_value(state, depth, car_index):
            exp_val = 0
            action = None

            if not state.is_car_on_road(car_index):
                return exp_val, action

            actions_list = state.get_legal_actions(car_index)
            num_of_actions = len(actions_list)
            # for every possible action: find the total expected value of all the paths
            for cur_action in actions_list:
                next_state = state.generate_successor(car_index, cur_action)
                # If the car_index is the last agent: next agent will be the driving agent
                # Min agent will try to minimize the expected value for the max agent
                if car_index == num_of_agents - 1:
                    max_val = value(next_state, self.index, depth + 1)[0]
                else:
                    max_val = value(next_state, car_index + 1, depth)[0]
                exp_val += (max_val / num_of_actions)

            return exp_val, action

        return value(road_state, self.index, 0)[1]

    def cars_next_to_agent(self, car_locations, agent_loc):
        if not car_locations:
            return 0
        x, y = agent_loc
        adj_locations = [(x - 1, y - 1), (x - 1, y), (x - 1, y + 1), (x, y - 1),
                         (x, y + 1), (x + 1, y - 1), (x + 1, y), (x + 1, y + 1)]
        return sum(1 for a, b in car_locations if (a, b) in adj_locations)


class LearningAgent(Agent):
    """
    A learning agent chooses an action at each choice point based on the Q values approximated.
    In this project your learning agent is essentiually an ApproximateQLearningAgent
    
    """

    def __init__(self, num_features=4, custom_weights=False, weights=None, alpha=0.1, gamma=0.99):
        self.alpha = alpha
        self.gamma = gamma
        self.num_features = num_features
        self.decay_rate = 0.99
        self.epsilon = 1
        self.episode_num = 0

        if custom_weights:
            self.weights = weights
        else:
            self.weights = np.random.rand(num_features)

        super().__init__()

    def get_weights(self):
        return self.weights

    def get_features(self, state, action):
        """
        Description
        ----------
        This function returns a vector of features for the given state action pair
        
        Compute: f_1(s, a), f_2(s, a), ... , f_n(s, a)

        """
        "*** YOUR CODE HERE ***"
        # helpers
        road_data = state.get_road_data()
        car_locations = state.get_car_locations()
        height = state.get_height()
        width = state.get_width()
        cur_loc = car_locations[0]

        next_loc = self.calc_nxt_loc(cur_loc, action)
        distance_to_goal = 1.0 / (height - float(next_loc[0]))
        distance_to_closest_car = self.find_closest_car_distance(car_locations, next_loc, cur_loc)
        possible_collision = self.check_for_collision(cur_loc, action, road_data)
        x, y = next_loc
        wall_on_left = 1.0 if (0 == y) else 0
        wall_on_right = 1.0 if y == width - 1 else 0
        num_of_adj_cars = self.find_num_of_adj_cars(cur_loc, road_data, height, width)
        cars_above_agent = sum(row.count(1) for row in road_data[:x]) / 10.0
        cars_below_agent = sum(row.count(1) for row in road_data[x:]) / 10.0
        num_of_lanes_to_left = next_loc[1]
        num_of_lanes_to_right = width - next_loc[1] - 1
        empty_cells_adj_to_agent = self.find_empty_cells_adj_to_agent(next_loc, road_data, height, width)
        crash_into_wall = self.check_for_wall_crash(cur_loc, action, width)
        distance_to_mid_lane = width / 2 - next_loc[1]
        cars_left = sum(1 for car in car_locations if car is not None) - 1

        features = [1.0, distance_to_goal, distance_to_closest_car, possible_collision, crash_into_wall, wall_on_left,
                    num_of_adj_cars, cars_above_agent, cars_below_agent, num_of_lanes_to_left, distance_to_mid_lane]

        return [feat / 10.0 for feat in features]

    def check_for_wall_crash(self, location, action, width):
        x, y = location
        if action == 'L' and y == 0:
            return 1.0
        elif action == 'R' and y == width - 1:
            return 1.0
        return 0

    def find_empty_cells_adj_to_agent(self, location, road, height, width):
        x, y = location
        adj_locations = [(x - 1, y), (x, y - 1), (x, y + 1)]
        num_of_empty_cells = sum(1 for a, b in adj_locations
                                 if (0 <= a < height and 0 <= b < width) and road[a][b] == 0)
        return num_of_empty_cells

    def find_num_of_adj_cars(self, agent_loc, road, height, width):
        x, y = agent_loc
        adj_locations = [(x - 1, y - 1), (x - 1, y), (x - 1, y + 1), (x, y - 1),
                         (x, y + 1), (x + 1, y - 1), (x + 1, y), (x + 1, y + 1),
                         (x, y - 2), (x, y + 2)]
        num_of_cars = len([(a, b) for a, b in adj_locations
                           if (0 <= a < height and 0 <= b < width) and road[a][b] == 1])
        return num_of_cars

    def find_closest_car_distance(self, car_locations, next_loc, cur_loc):
        car_distances = [manhattan_distance(car, next_loc)
                         for car in car_locations.values() if car is not None and not car == cur_loc]
        car_distances = [distance for distance in car_distances if not distance == 0]
        return 1.0 / min(car_distances) if car_distances else 0

    def calc_nxt_loc(self, agent_loc, action):
        x, y = agent_loc
        if action == 'F':
            return x - 1, y
        elif action == 'L':
            return x, y - 1
        elif action == 'R':
            return x, y + 1
        else:
            return x, y

    def check_for_collision(self, agent_loc, action, road_data):
        x, y = agent_loc
        adj_loc_to_check = []
        if action == 'F':
            adj_loc_to_check = [(x - 1, y - 1), (x - 1, y), (x - 1, y + 1)]
        elif action == 'L':
            adj_loc_to_check = [(x, y - 2), (x, y - 1), (x + 1, y - 1)]
        elif action == 'R':
            adj_loc_to_check = [(x, y + 1), (x, y + 2), (x + 1, y + 1)]
        else:
            adj_loc_to_check = [(x, y - 1), (x + 1, y), (x, y + 1)]

        for a, b in adj_loc_to_check:
            if 0 <= a < len(road_data) and 0 <= b < len(road_data[0]) and road_data[a][b] == 1:
                return 1.0

        return 0

    def get_Q_value(self, state, action):
        """
        Description
        ----------
        This function returns the Q value; Q(state,action) = w . featureVector
        where . is the dotProduct operator
        
        Compute: Q(s, a) = w_1 * f_1(s, a) + w_2 * f_2(s, a) + ... + w_n * f_n(s, a)
        
        """
        "*** YOUR CODE HERE ***"
        features = self.get_features(state, action)
        # print(f'Weights: {self.weights}')
        return np.dot(self.weights, features)

    def compute_max_Q_value(self, state):
        """
        Description
        ----------
        This function returns the max over all Q(state, action) 
        for all legal/available actions for the given state
        Note that if there are no legal actions, which is the case at the
        terminal state, you should return a value of 0.0.
        
        Compute: max_a' Q(s', a')
        
        """
        "*** YOUR CODE HERE ***"
        if state.is_terminal():
            return 0.0
        q_values = [self.get_Q_value(state, action) for action in self.available_actions]
        return max(q_values)

    def update(self, state, action, next_state, reward):
        """
        Description
        ----------
        This function updates the weights based on the given transition
        
        """
        "*** YOUR CODE HERE ***"
        exp_reward = reward + self.gamma * self.compute_max_Q_value(next_state)
        diff_in_reward = exp_reward - self.get_Q_value(state, action)
        features = self.get_features(state, action)
        for i, feat in enumerate(features):
            self.weights[i] += self.alpha * diff_in_reward * features[i]

    def get_action(self, state):
        """
        Description
        ----------
        This function returns the best action based on the self.weights it has learned.

        """
        max_qvalue = float('-inf')
        best_action = None
        for action in self.available_actions:
            qvalue = self.get_Q_value(state, action)
            if qvalue > max_qvalue:
                max_qvalue = qvalue
                best_action = action
        return best_action

    def train(self, state):
        """
        Description
        ----------
        This function learns the weights for your approximate Q learning agent 
        by training for a single episode given the initialization of the road.
        
        """

        cum_reward = 0
        done = False
        while not (done):
            # Choose an action epsilon greedily 
            if np.random.rand() < self.epsilon:
                action = np.random.choice(self.available_actions)
            else:
                action = self.get_action(state)

            # Take the action and observe the next state and reward
            next_state, reward, done = state.step(action)
            # Update weights by approximating Q value
            self.update(state, action, next_state, reward)

            # Move to the next state
            state = next_state
            cum_reward += reward

        return cum_reward


def normalize(value, min_value, max_value):
    """
    Description
    ----------
    Normalizes a given value between 0-1
    
    """
    return (value - min_value) / (max_value - min_value)


def manhattan_distance(loc1, loc2):
    """
    Description
    ----------
    Returns the Manhattan distance between points loc1 and loc2
    
    """
    return abs(loc1[0] - loc2[0]) + abs(loc1[1] - loc2[1])
