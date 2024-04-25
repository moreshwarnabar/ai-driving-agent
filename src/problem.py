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

from environment import Environment
from agent import *
from state import *  

import copy
import random
    
    
class AutonomousDriving:
    """
    This class creates a problem instance by initializing a specific agent type and simulates the autonomous driving environment.
        
    """
    def __init__(self, agent_type, height, width, occupancy, init, seed, episodes, features, custom_weights=False, weights=None):
        self.agent = None
        self.env = Environment(height, width, occupancy, init, seed)
        
        if agent_type == 'manual':
            self.agent = ManualAgent()
        elif agent_type == 'random':
            self.agent = RandomAgent()
        elif agent_type == 'reflex':
            self.agent = ReflexAgent()
        elif agent_type == 'expectimax':
            self.agent = ExpectimaxAgent()
        elif agent_type == 'learning':
            self.agent = LearningAgent(features, custom_weights, weights)
        else:
            raise ValueError(f"Incorrect agent specified: {agent}")
            
    def run(self, agent_type, *args):
        """
        Description
        -----------
        This function calls the simulation function based on the agent initialized.
        and passes it to the step function.
        Thus, simulating the autonomous driving.
        
        """
        
        if agent_type == 'manual':
            self.run_manual_control()
        elif agent_type == 'random':
            self.run_random_control()
        if agent_type == 'reflex':
            self.run_reflex_control()
        elif agent_type == 'expectimax':
            self.run_expectimax_control()
        elif agent_type == 'learning':
            self.run_learning_control(args[0])
    
    def run_manual_control(self):
        self.env.display_environment()
        while True:
            state_obj = ManualState(copy.deepcopy(self.env))
            state_obj.display_state()
            action = self.agent.get_action(None)
            self.env.step(action)
            
    def run_random_control(self):
        self.env.display_environment()
        while True:
            state_obj = RandomState(copy.deepcopy(self.env))
            state_obj.display_state()
            action = self.agent.get_action(None)
            self.env.step(action)
            
    def run_reflex_control(self):
        self.env.display_environment()
        while True:
            state_obj = ReflexState(copy.deepcopy(self.env))
            percept = state_obj.get_percept()
            action = self.agent.get_action(percept)
            self.env.step(action)

    def run_expectimax_control(self):
        self.env.display_environment()
        while True:
            state_obj = ExpectimaxState(copy.deepcopy(self.env))
            action = self.agent.get_action(state_obj)
            print(action)
            self.env.step(action)
            
    def run_learning_control(self, num_episodes):
        W = self.train(num_episodes)
        self.test()
        
    def train(self, num_episodes):
        self.env.display_environment()
        total_reward = 0
        # Training
        print("training...")
        for eps in range(num_episodes):
            seed = random.randrange(100)
            #Create a new environment configuration
            env = Environment(self.env.height, self.env.width, self.env.occupancy, self.env.init, seed)
            obj = LearningState(env)
            total_reward += self.agent.train(obj)
            if(eps%1000==0):
                print(f"Episode {eps + 1}, Total Reward: {total_reward}")
                self.agent.epsilon = self.agent.epsilon * self.agent.decay_rate
            
        return self.agent.get_weights()
                
    def test(self):
        print("testing...")
        while True:
            state_obj = LearningState(copy.deepcopy(self.env))
            action = self.agent.get_action(state_obj)
            print(action)
            self.env.step(action)