# Training a PPO-Based Rocket League Bot to Outperform Baseline AI Opponents

## Table of Contents
1. [Introduction and Motivation](#introduction-and-motivation)
2. [Related Work](#related-work)
3. [Problem Definition and Data Collection](#problem-definition-and-data-collection)
4. [Theory, Metrics, and Background](#theory-metrics-and-background)
5. [Approach and Implementation Details](#approach-and-implementation-details)
6. [Experimental Setup](#experimental-setup)
7. [Results and Analysis](#results-and-analysis)
8. [Discussion and Future Work](#discussion-and-future-work)
9. [Conclusion](#conclusion)
10. [References](#references)

---

## Introduction and Motivation
- **What problem are you addressing?**  
  Introduce the problem of training a Rocket League bot (using PPO) that can defeat AI opponents of varying skill levels.
  The stock bots that are included in Rocket League are bad. Most players quickly outpreform these bots after achieveing a basic understanding of the controls and strategy.
  Rocket league is a relatively simple game, that has an infinitely high skill ceiling. 
  
- **Why is this problem important or interesting?**  
  Discuss the real-world relevance (e.g., general reinforcement learning challenges, complexity of learning in a 3D environment, improving AI in competitive video games).
  Reinforcement learning is the key to making better bots that can exceed human capabilities. Its potential for Team based Cooperative-competative breakthroughs is great. It has a very large state space and dynamic environment that makes hardcodin not feasible. Cooperative-Competative MARL is still an open challenge in the robotics and ML field. 

- **Who is this tutorial for?**  
  Explain that the tutorial targets readers who want to reproduce these results or apply PPO to a similar task.
  THis tutorial is for those who want to train a bot using open source frameworks to be better then the stock Rocket League bots. This specific implementation is for 1v1, but 2v2 and 3v3 is supported.

- **What can the reader expect to learn?**  
  This will detail setting up the environments and packages needed, and approaches to creating a working bot.

## Related Work
- **Existing Work and References (≥5 papers)**  
  Discuss existing research on training bots in game environments, PPO-based methods, and any prior work on Rocket League bots.
  
- **Positioning Your Work**  
  How your method differs or expands upon existing literature.

## Problem Definition and Data Collection
- **Defining the Problem**  
  Being "better" than a bot is defined as the probability of scoring the next goal being greater than 50% with a 95% confidence level.
  Number of trials (n=69) and p=0.5 (null hypothesis).
  Stopping rule: achieve 42 goals before the opponent scores 28 to reject the null hypothesis with p<0.05.

- **Environment and Datasets**  
  RLbot runs in Rocket League, and has training available, but we wont be using that.
  RLgym_sim simulates the rocket league environment and allows for MMany more instance to be run much faster than real life.

- **Data Collection Process**  
  How you gathered training and evaluation data (e.g., match logs).
  Metrics are compiled in Weights and Biases (Wandb). This tracks things like policy reward, Mean_kl divergence etc.

- **Environment Setup**  
  Provide instructions for installing and running the Rocket League training environment, PPO code, and dependencies.

## Theory, Metrics, and Background
- **Background on PPO**  
  Briefly describe PPO and how it differs from standard policy gradients.

- **Metric Definition**  
  Introduce the binomial model with n=69, p=0.5, and the significance threshold (42 vs 28 goals for p<0.05).

- **Why This Metric?**  
  Justify using this statistical approach to ensure your bot’s performance is truly better than baseline.
  
- **Datasets of simulated environments**  
  Maybe the meshes for the visualizer idk

  
## Approach and Implementation Details
- **Step-by-Step Instructions**  
  Instructions on installing dependencies, cloning the repo, and configuring the environment.


- **Code Organization**  
  Explain the directory structure (e.g., `train.py`, `eval.py`, `utils/`).

- **Modifications from Base Implementations**  
  Detail what you changed if you adapted code from others.

- **Hyperparameters and Tuning**  
  Provide default hyperparameters and guidance on tuning.

## Experimental Setup
- **Training Procedure**  
  Duration, hardware, and evaluation intervals.

- **Opponent Setup**  
  Facing off against bots 1 to 5 and how matches are orchestrated.

- **Reproducibility**  
  Exact commands or scripts to run for reproducing experiments.

## Results and Analysis
- **Quantitative Results**  
  Present tables/graphs of success rates and show when your bot achieves the 42 vs 28 margin.

- **Statistical Significance**  
  Show computations of p-values to confirm significance.

- **Qualitative Observations**  
  Screenshots, video links, or behavioral notes on the bot’s improvement.

- **Failure Cases**  
  Mention where the bot fails, especially against the strongest opponents.

## Discussion and Future Work
- **Limitations**  
  Where the approach might fail or what remains challenging.

- **Ideas for Future Improvements**  
  Suggest better architectures, different RL algorithms, or improved evaluation strategies.

## Conclusion
- **Summary**  
  Recap the main achievements: PPO setup, beating certain bot levels, and statistical validation.

- **Final Remarks**  
  Encourage experimentation and adaptation of the provided code.

## References
- **Cited Works**  
  List all referenced papers and resources in a standard citation format.
- **Links to Code and Dependencies**  
  Include URLs to repositories, documentation, or installation guides.
  https://github.com/Rolv-Arild/Necto
  https://github.com/redd-rl/apollo-bot
  https://github.com/ZealanL/RLGym-PPO-Guide
  https://github.com/ZealanL/RLGym-PPO-RLBot-Example
  https://github.com/ZealanL/RocketSimVis
  https://github.com/AechPro/rocket-league-gym-sim

---
