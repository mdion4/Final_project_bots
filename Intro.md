## Setting up Environments
- **Things you need**
  - Rocket League on Epic or Steam
  - Bakkesmod from https://bakkesplugins.com/
  - RLbot from https://rlbot.org/
  - A decent computer
  - A venv with python 3.9
- **Setup Guides**
  - Follow the steps from this Wonderful PPO guide https://github.com/ZealanL/RLGym-PPO-Guide/blob/main/intro.md
  This details Setting up Python, Git, RLGym-PPO, RocketSim, rlgym_sim, getting the Rocket League collision maps, PyTorch install for CUDA, and having an example bot set up

  - You will need to be able to visualize your bots behavior to see its progress. Follow this guide to set up an easy to use visualizer https://github.com/ZealanL/RocketSimVis
  - More details on using this will be provided in the learner section.

  - This allows your bot to train in a simulated environment, but how do you actually use it in a match? RLBot loads the bot into Rocket League.
  - The steps to convert a PPO sim trained bot to one that runs in RLBot will covered later once you are more familiar with the components that make up the bot.

## Example Bot
After following the first PPO guide, you'll have an example bot thats ready to train. You run it by navigating to the folder where example.py is via the command line, then type example.py. This starts the training, and will output each iteration report at regular intervals which include the following metrics (and more advanced ML stuff):
  - **Policy Reward**:
  - **Value Function Loss**:
  - **SB3 Clip Fraction**:
  - **Collected Steps per second**:
  - **Overall Steps per second**:
  - **Timestep collection Time**:
  - **Timestep Consumption Time**:
  - **Cumulative Timesteps**:
The Training Environment is made up of a few components:
  - Rewards
  - Obs Builder
  - Action Parser
  - State Setter
  - Terminal conditions
  - Number of players
  **
