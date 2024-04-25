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

import argparse

from problem import AutonomousDriving

parser = argparse.ArgumentParser(description='This script runs the autonomous driving environment with a given agent.')
parser.add_argument("--agent", default='random', help="Agent type (manual, random, reflex, expectimax, learning) - default: random")
parser.add_argument("--height", default=10, help="Road height (int) - default: 10")
parser.add_argument("--width", default=4, help="Road width (int) - default: 4")
parser.add_argument("--occupancy", default=0.1, help="Occupancy of the road with other cars. \nEnter a value between 0 and 0.3 - default: 0.1" )
parser.add_argument("--init", default=0.2, help="Initial location of 'A' w.r.t. the height of the road. \nENter a value between 0 and 0.9 - default: 0.2")
parser.add_argument("--seed", default=3, help="To initialize cars on the road (int) - default: 3")
parser.add_argument("--episodes", default=50000, help="Number of training episodes (int) - default: 50000")
parser.add_argument("--features", default=4, help="Number of features that will be extracted (int) - default: 4")
args = parser.parse_args()

if __name__ == "__main__":
    # Assert validity of arguments
    assert args.agent == 'manual' or 'random' or 'reflex' or 'expectimax' or 'learning', "agent should be one of the defined types (manual, random, reflex, expectimax, learning)"
    assert int(args.height) <= 100, "height of the road is large"
    assert int(args.width) <= 10, "width of the road is large"
    assert float(args.occupancy) >= 0 and float(args.occupancy) <= 0.3, "occupancy is out of bounds"
    assert float(args.init) >= 0 and float(args.init) <= 0.9, "init is out of bounds"
    assert type(int(args.seed)) is int, "seed should be an integer"
    assert int(args.episodes) > 0 and int(args.episodes) <= 100000, "number of training episodes is out of bounds"
    assert type(int(args.features)) is int, "features should be an integer"
    assert int(args.features) > 0, "features should be greater than 1"
    
    if args.agent == 'expectimax':
        assert int(args.height) <= 5, "height of the road is too large for expectimax agent"
        assert int(args.width) <= 5, "width of the road is too large for expectimax agent"
        assert float(args.occupancy) <= 0.1, "occupancy of the road is too large for expectimax agent"
    
    driver = AutonomousDriving(agent_type=args.agent, height=int(args.height), width=int(args.width), \
                               occupancy=float(args.occupancy), init=float(args.init), seed=int(args.seed), \
                               episodes=int(args.episodes), features=int(args.features), custom_weights=False, weights=None)
    driver.run(args.agent, int(args.episodes)) 