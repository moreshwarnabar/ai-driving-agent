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
from problem import AutonomousDriving

import argparse
import json
import subprocess

parser = argparse.ArgumentParser(description='This script grades the autonomous driving environment.')
parser.add_argument("--q", default=None, help="Enter the question you wish to be graded \nYour options are q1, q2, and q3 - default: None")
parser.add_argument("--verbose", default=False, help="Enter True if you wish to see detailed output - default: None")
parser.add_argument("--features", default=4, help="Number of features that will be extracted (int) - default: 4")
args = parser.parse_args()

success = '\'A\' has successfully navigated the road!'
failure = 'There is a crash on the road!'

with open('test.json', 'r') as tf:
    test_file = json.load(tf)

score_q1 = 0
score_q2 = 0
score_q3 = 0

if args.q == 'q1' or args.q == None:
    print("----- Testing Q1 -----")
    num_test_cases = len(test_file['q1'])
    for test_case in range(num_test_cases):
        height = test_file['q1'][str(test_case)]['height']
        width = test_file['q1'][str(test_case)]['width']
        occupancy = test_file['q1'][str(test_case)]['occupancy']
        init = test_file['q1'][str(test_case)]['init']
        seed = test_file['q1'][str(test_case)]['seed']
        
        cmd = f'python driving.py --agent reflex --height {height} --width {width} --occupancy {occupancy} --init {init} --seed {seed}'
        
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr = subprocess.STDOUT, encoding='utf8', shell=True)
        out, err = p.communicate() 
        
        if args.verbose == 'True':
            print(out)
        if success in out:
            score_q1 += 1
            print(f"Testcase {test_case}: PASS")
        elif failure in out:
            print(f"Testcase {test_case}: FAIL")
        else:
            print(f"Testcase {test_case}: FAIL")
            print("There was an error in running test case: ", test_case)
            
    if (score_q1 - 0) / (7 - 0) * 10 > 10:
        score_q1 = 10
    else:
        score_q1 = round((score_q1 - 0) / (7 - 0) * 10, 1)
            
if args.q == 'q2' or args.q == None:
    print("----- Testing Q2 -----")
    num_test_cases = len(test_file['q2'])
    for test_case in range(num_test_cases):
        height = test_file['q2'][str(test_case)]['height']
        width = test_file['q2'][str(test_case)]['width']
        occupancy = test_file['q2'][str(test_case)]['occupancy']
        init = test_file['q2'][str(test_case)]['init']
        seed = test_file['q2'][str(test_case)]['seed']
        
        cmd = f'python driving.py --agent expectimax --height {height} --width {width} --occupancy {occupancy} --init {init} --seed {seed}'

        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr = subprocess.STDOUT, encoding='utf8', shell=True)
        out, err = p.communicate() 

        if args.verbose == 'True':
            print(out)
            
        if success in out:
            score_q2 += 1
            print(f"Testcase {test_case}: PASS")
        elif failure in out:
            print(f"Testcase {test_case}: FAIL")
        else:
            print(f"Testcase {test_case}: FAIL")
            print("There was an error in running test case: ", test_case)
            
    if (score_q2 - 0) / (7 - 0) * 10 > 10:
        score_q2 = 10
    else:
        score_q2 = round((score_q2 - 0) / (7 - 0) * 10, 1)

if args.q == 'q3' or args.q == None:
    print("----- Testing Q3 -----")
    num_test_cases = len(test_file['q3']['test'])
    
    # Training code
    height = test_file['q3']['train'][str(0)]['height']
    width = test_file['q3']['train'][str(0)]['width']
    occupancy = test_file['q3']['train'][str(0)]['occupancy']
    init = test_file['q3']['train'][str(0)]['init']
    seed = test_file['q3']['train'][str(0)]['seed']
    episodes = test_file['q3']['train'][str(0)]['episodes']
    features = int(args.features)
    
    driver = AutonomousDriving(agent_type='learning', height=height, width=width, \
                               occupancy=occupancy, init=init, seed=seed, \
                               episodes=episodes, features=features)
    
    learned_weights = driver.train(episodes)
    learned_weights = learned_weights.tolist()
    for test_case in range(num_test_cases):
        height = test_file['q3']['test'][str(test_case)]['height']
        width = test_file['q3']['test'][str(test_case)]['width']
        occupancy = test_file['q3']['test'][str(test_case)]['occupancy']
        init = test_file['q3']['test'][str(test_case)]['init']
        seed = test_file['q3']['test'][str(test_case)]['seed']
        
        cmd = f"from problem import AutonomousDriving; \
                test_driver = AutonomousDriving(agent_type='learning', height={height}, width={width}, \
                               occupancy={occupancy}, init={init}, seed={seed}, \
                               episodes={episodes}, features={features}, custom_weights=True, weights={learned_weights}); \
                test_driver.test()"
        
        result = subprocess.run(['python', '-c', cmd], stdout=subprocess.PIPE, text=True)
        out = result.stdout

        if args.verbose == 'True':
            print(out)
            
        if success in out:
            score_q3 += 1
            print(f"Testcase {test_case}: PASS")
        elif failure in out:
            print(f"Testcase {test_case}: FAIL")
        else:
            print(f"Testcase {test_case}: FAIL")
            print("There was an error in running test case: ", test_case)
            
    if (score_q3 - 0) / (7 - 0) * 10 > 10:
        score_q3 = 10
    else:
        score_q3 = round((score_q3 - 0) / (7 - 0) * 10, 1)
            
total = score_q1/2+score_q2+score_q3

print("\n===========================")
print("Q1 score: \t", score_q1/2)
print("Q2 score: \t", score_q2)
print("Q3 score: \t", score_q3)
print("---------------------------")
print(f"Total: \t\t{total} / 25.0")
print("===========================")