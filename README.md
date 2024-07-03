# RL Policy Ensemble for Stock Trading

## Overview

This repository contains the implementation of an ensemble of deep Reinforcement Learning (RL) models to optimize stock trading strategies. My composite model integrates distinct RL algorithms—DQN, DDPG, A2C, and PPO—leveraging their unique strengths in a unified trading strategy. I employ a novel voting system to adjust the influence of each model based on its performance and cumulative return, aiming to outperform traditional approaches and navigate the highly volatile financial market robustly and adaptively.

## Problem Motivation

The financial market's non-linear nature and the need for robustness in trading algorithms necessitate sophisticated models. My ensemble strategy combines multiple models to enhance decision-making robustness, capitalizing on individual strengths while mitigating weaknesses.

## Methodology

### Ensemble Strategy
I explored two ensemble approaches:
1. **Max Voting:** Actions are selected based on a majority vote among the agents.
2. **Converted Majority Thresholding:** Each agent converts its action into a standardized form (-1, 0, +1 for sell, hold, buy). The sum of the converted actions determines the final action:
    - Buy if sum > 1
    - Hold if -1 ≤ sum ≤ 1
    - Sell if sum < -1

### Setup
I employ a custom-built stock trading simulator using OpenAI Gym, focusing on Apple stock (AAPL). The state space includes closing price information and the number of holdings. The action space consists of buy, hold, and sell actions.

### Data and Reward Design
I used Adjusted Closing Prices from Yahoo Finance, adjusted for inflation using CPIAUCNS data. The reward structure includes:
- Profit-based rewards
- Holding cost penalties
- Risk penalties calculated from total assets

### Baseline Method
My baseline method uses a momentum-based strategy, buying stocks if prices are decreasing over four days and selling if prices are increasing. This simple strategy serves as a comparison to my ensemble method.

## Experiments and Results

I conducted a series of experiments with different reward designs and ensemble strategies, training agents on data from 2010 to 2020 and testing on data from 2021 to 2024.

### Hyperparameters
| Agent | Gamma (γ) | Learning Rate (α) | Batch Size | Tau (τ) | Memory Size | Episodes |
|-------|-----------|-------------------|------------|---------|-------------|----------|
| DQN   | 0.99      | 0.005             | 32         | 0.005   | 2x10^5      | 100      |
| PPO   | 0.99      | 0.002             | -          | -       | -           | 25       |
| A2C   | 0.99      | 0.001             | 2000       | -       | -           | 50       |
| DDPG  | 0.98      | 0.005             | 64         | 0.005   | 2x10^5      | 100      |

### Initial Experiments
Initial experiments with basic reward functions and risk penalties showed the need for refined reward designs and better ensemble strategies.

### Updated Reward Design
I introduced a more penalizing reward function and refined risk penalties. The final reward function penalizes negative total assets heavily and includes a drawdown-based risk penalty.

### Final Results
The updated ensemble method with Converted Majority Thresholding outperformed the baseline and individual models, demonstrating robustness and adaptability in volatile market conditions.

## Conclusions

My ensemble method effectively outperformed the baseline, with DQN consistently performing well. PPO showed volatility, and A2C had limited learning. Future work could explore dynamic weighting for agent actions and incorporate more detailed market information in the observation space.

## Contributions
- Developed the DQN and DDPG agents, preprocessed data, and set up the custom environment.
- Developed the A2C and PPO agents, baseline algorithm, and experimented with reward designs.
- Contributed to ensemble strategy development, testing, and documentation.

## Acknowledgements
I acknowledge the 6.8200 Staff and Professor Agarwal for their guidance. Libraries used include `matplotlib`, `pandas`, `numpy`, `torch`, and `random`. 

