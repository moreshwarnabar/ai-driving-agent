# Autonomous Agents in Simulated Driving Environment

## Project Overview

This project was developed as part of the CSE-471 Introduction to Artificial Intelligence coursework at Arizona State University. It involves the implementation of various autonomous agents that navigate a simulated driving environment. The agents are designed to make decisions based on the environment and execute actions within a grid-based simulation, highlighting techniques in reflex agents, decision-making under uncertainty, and reinforcement learning.

## Technologies Used

- Python
- NumPy

## Agents Description

### 1. Simple Reflex Agent

- **Purpose**: Navigates the simulation based on immediate perceptions without considering the outcomes of its actions.
- **Implementation Details**: Uses basic if-else logic to avoid obstacles and navigate towards the goal.

### 2. Expectimax Agent

- **Purpose**: Makes decisions considering both the chances of various possible immediate outcomes and their impact on future states.
- **Implementation Details**: Utilizes the Expectimax algorithm to choose actions that maximize the expected utility, accounting for various probabilities of occurrence.

### 3. Approximate Q-Learning Agent

- **Purpose**: Learns from the environment to make decisions with incomplete knowledge, improving performance over time through reinforcement.
- **Implementation Details**: Implements Q-learning with a function approximator to estimate the return from a particular state and action over time.

## Evaluation

The agents are evaluated based on functionality and the effectiveness of decision-making in various simulated scenarios, tested rigorously to ensure robust performance.

## Installation

To set up the project environment, follow these steps:

```bash
# Clone the repository
git clone https://github.com/moreshwarnabar/ai-driving-agent.git
cd ai-driving-agent

# Install required Python packages
pip install numpy
```
