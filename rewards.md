# Rewards and Weights
### **What are rewards?**

Rewards are how your bot learns what is good and what is bad. The bot seeks to maximize its cumulative rewards. In PPO (Proximal Policy Optimization) the rewards are your way of communicating the desired behaviors to your bot. This is really the core of training a bot via PPO, as you must do a lot of trial and error on different reward functions, different weights, how to avoid getting stuck at a local maxima, properly adjusting rewards without breaking the bot etc. 


### **Types of rewards**

  Reward functions return rewards. It is structured like this (rewardFunc(), weight).
  
  There are continuous rewards, like VelocityBallToGoal() that ar calculated each step, and there are discrete awards that are awarded once, once some condition is met. Really all rewards are technically continuous, as all reward functions are calculated every step, so it can be better to think of rewards as sparse vs dense. Dense rewards will nearly always return some value, usually for things like velocities, direction, distance, boost amount, etc, while sparse rewards usually just return 0, as all the conditions are not met. 
  
  Sparse rewards should be weighted higher due to their lower occurrence rate. The exact value of the weight is not important as the weights are all relative to each other. (Foo(), 1), (Foo2(), 10) is the same as (Foo(), 0.01), (Foo2(), 0.1). 
  
  There is a built in EventReward() with different events like touch, team_goal, concede, demo etc. I recommend starting out with (EventReward(touch=1),50)



### **Structure of a reward function**

  Reward functions are actually a class that extends RewardFunction, and have 3 methods they need to implement in our case since we are using the default obs. Other more advanced obs can include an additional pre_step() but we won't worry about that.
    - \_\_init_\_ (self, ...) Add any parameters you need.
    - reset(self, initial_state: GameState)  _values at the beginning of an episode._
    - get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float: _where most of your logic is implemented, returns the reward_
        It is usually a good practice to keep your reward between (0,1) for consistency with other rewards, so that balancing the weights of reward functions makes more sense. Of course there are exceptions, just something to keep in mind.

## Supporting Functions, meta reward functions
### **UH-OH, rlgym_sim only takes 1 reward function**
We just use CombinedReward from zip to take care of that, and put the weights with its function for better readability. Following code snippet ripped from Zealan's PPO guide
```py

reward_fn = CombinedReward.from_zipped(
    # Format is (func, weight)
    (InAirReward(), 0.002),
    (SpeedTowardBallReward(), 0.01),
    (VelocityBallToGoalReward(), 0.1),
    (EventReward(team_goal=1, concede=-1, demo=0.1), 10.0)
)
```


### **ZeroSumReward**
As the name implies, a reward for one bot is a penalty for the opposing bot. Zero sum reward wraps a reward function, and uses team_spirit and opp_penalty to scale the rewards. team_spirit distributes rewards amongst teammates, a value of 0 means individuals do not share with teammates and keep the entire reward for themselves; a value of 1 means the reward is shared completely. Opp_penalty is a scalar usually between 0 and 1 dictating how much penalty to incur. It is only truly ZeroSum if opp penalty is 1, where the average reward for some reward function is 0. This is good to wrap around functions that deal with things the bot should be trying to prevent the opponent from doing, like touching the ball, having speed, having boost. It doesn't make as much sense for things like being in the air, facing the ball. I would suggest starting the opp_penalty low to start, then slowly increase as training goes on. Too much noise is bad for the bot in the early stages while it is still confused.


### **AnnealingRewards**
I believe this to be the most crucial part of training a bot beyond the initial ballchasing stage. AnnealRewards is also a wrapper function that linearly transitions between two or more reward functions over some number of steps. The pattern goes (rew1, num_steps_1_to_2, rew2, num_steps_2_to_3, etc). Unfortunately the intermediate reward functions aren't saved anywhere, so if training ends before completing all the steps, you either have to revert to the checkpoint before calling AnnealRewards, or try calculating what the current reward function would be. Calculating wrong is highly likely to just break your bot. Even freezing the policy(setting policy_lr = 0) and allowing the critic to learn the new rewards is unlikely to keep your bot from breaking upon unfreezing the policy. 

I suggest having many iterations of your combined reward function, and only use AnnealRewards() to go from 1 version to another, as you only know the current reward functions used by the bot at either end of AnnealRewards(). Once the goal reward function is reached, training will continue using that reward function. So now its safe to quit training and visualize your bot, making sure to change your reward function from AnnealReward to just the reward function you ended on.

Here is an example of what it might look like after multiple iterations. Some were longer than others as the changes were bigger and you don't want to shock the bot.
```py
    reward_fn = AnnealRewards(rew1, 150_000_000, rew2)
    reward_fn = AnnealRewards(rew2, 300_000_000, rew3)
    reward_fn = AnnealRewards(rew3, 100_000_000, rew4)
    reward_fn = AnnealRewards(rew4, 100_000_000, rew5)
    reward_fn = AnnealRewards(rew5, 100_000_000, rew6)
    reward_fn = AnnealRewards(rew6, 100_000_000, rew7)
    reward_fn = rew7
```

# Progression Strategy That Worked For Me
## Ballchasing and car control
You typically want to have all the reward functions you will be using introduced at the beginning, with the more advanced stuff having a low weight. The initial reward function I relied on to get to ball chasing stage is:
```py
rew = CombinedReward.from_zipped(
        # Format is (func, weight)
        (ZeroSumReward(TouchVelocityReward(),team_spirit, opp_penalty_scale), 10),  # Reward strong touches
        (SaveBoostReward(), 0.0001), 
        (AirTouchReward(), 0.01), # any touch in air
        (JumpTouchReward(), 1), # scaled reward based on height, min height threshold, encourage aerials.
        (ZeroSumReward(EventReward(touch=1), team_spirit, opp_penalty_scale), 50), # Giant reward for actually hitting the ball
        (LiuDistanceBallToGoalReward(), 0.1), #get ball closer to goal
        (SpeedTowardBallReward(), 5), # Move towards the ball!
        (FaceBallReward(), 0.4), # Make sure we don't start driving backward at the ball
        (InAirReward(), 0.05), # Make sure we don't forget how to jump
        (VelocityBallToGoalReward(), 0.1),
    )
```
Went to 23 million steps. Chasing good, not jumping.
I manually tweaked rewards before discovering annealing. Tried increasing amount of jumps and getting the ball toward the goal. Team_spirit is irrelevant in 1v1, opp_penalty_scale was 0.01 in the beginning.

```py
rew = CombinedReward.from_zipped(
        # Format is (func, weight)
        (ZeroSumReward(TouchVelocityReward(),team_spirit, opp_penalty_scale), 10),  # Reward strong touches
        (SaveBoostReward(), 0.0001), 
        (AirTouchReward(), 0.01), # any touch in air
        (JumpTouchReward(), 1), # scaled reward based on height, min height threshold, encourage aerials.
        (ZeroSumReward(EventReward(touch=1), team_spirit, opp_penalty_scale), 50), # Giant reward for actually hitting the ball
        (LiuDistanceBallToGoalReward(), 0.2), #get ball closer to goal
        (SpeedTowardBallReward(), 4), # Move towards the ball!
        (FaceBallReward(), 0.4), # Make sure we don't start driving backward at the ball
        (InAirReward(), 0.1), # Make sure we don't forget how to jump
        (VelocityBallToGoalReward(), 0.2),
    )
```

Tried a light tweak without freezing policy at 48 million steps

```py
rew = CombinedReward.from_zipped(
        # Format is (func, weight)
        (ZeroSumReward(TouchVelocityReward(),team_spirit, opp_penalty_scale), 10),  # Reward strong touches
        (SaveBoostReward(), 0.0001), 
        (AirTouchReward(), 0.01), # any touch in air
        (JumpTouchReward(), 1), # scaled reward based on height, min height threshold, encourage aerials.
        (ZeroSumReward(EventReward(touch=1), team_spirit, opp_penalty_scale), 48), # Giant reward for actually hitting the ball
        (LiuDistanceBallToGoalReward(), 0.25), #get ball closer to goal
        (SpeedTowardBallReward(), 3.5), # Move towards the ball!
        (FaceBallReward(), 0.35), # Make sure we don't start driving backward at the ball
        (InAirReward(), 0.15), # Make sure we don't forget how to jump
        (VelocityBallToGoalReward(), 0.2),
    )
```

Introduced Annealing rewards at 79 Million steps. Added 50% penalty to opponent on ball touch and ball distance to goal.
```py
rew1 = CombinedReward.from_zipped(
        # Format is (func, weight)
        (ZeroSumReward(TouchVelocityReward(),team_spirit, opp_penalty_scale), 10),  # Reward strong touches
        (SaveBoostReward(), 0.001),
        (AirTouchReward(), 0.01),
        (JumpTouchReward(), 1),
        (ZeroSumReward(EventReward(touch=1), team_spirit, opp_penalty_scale), 48), # Giant reward for actually hitting the ball
        (LiuDistanceBallToGoalReward(), 0.25),
        (SpeedTowardBallReward(), 3.5), # Move towards the ball!
        (FaceBallReward(), 0.35), # Make sure we don't start driving backward at the ball
        (InAirReward(), 0.15), # Make sure we don't forget how to jump
        (VelocityBallToGoalReward(), 0.2), 
    )
    rew2 = CombinedReward.from_zipped(
        (ZeroSumReward(TouchVelocityReward(),team_spirit, 0.5), 5),  # Reward strong touches
        (SaveBoostReward(), 0.8),
        (AirTouchReward(), 1),
        (JumpTouchReward(), 3),
        (EventReward(touch=0.02, team_goal=goal_reward, concede=concede_reward), 30), # Giant reward for actually hitting the ball
        (ZeroSumReward(LiuDistanceBallToGoalReward(),team_spirit, 0.5), 0.5),
        (SpeedTowardBallReward(), 1), # Move towards the ball!
        (FaceBallReward(), 0.2), # Make sure we don't start driving backward at the ball
        (InAirReward(), 0.15), # Make sure we don't forget how to jump
        (VelocityBallToGoalReward(), 0.2),
    )

reward_fn = AnnealRewards(rew1, 150_000_000, rew2)
```

Transition worked well. Next transition from rew2 to rew3 over 300k steps

```py
rew3 = CombinedReward.from_zipped(
        (ZeroSumReward(TouchVelocityReward(),team_spirit, 1), 3),  # Reward strong touches
        (SaveBoostReward(), 0.8),
        (AirTouchReward(), 3),
        (JumpTouchReward(), 5),
        (EventReward(team_goal=goal_reward, concede=concede_reward), 100), # Giant reward for actually hitting the ball
        (ZeroSumReward(LiuDistanceBallToGoalReward(),team_spirit, 1), 0.5),
        (SpeedTowardBallReward(), 1), # Move towards the ball!
        (FaceBallReward(), 0.2), # Make sure we don't start driving backward at the ball
        (InAirReward(), 0.15), # Make sure we don't forget how to jump
        (VelocityBallToGoalReward(), 0.2), #discourage corners
    )
```

Working great. Starting next transition at 945 million over 100k steps

```py
rew4 = CombinedReward.from_zipped(
        (ZeroSumReward(TouchVelocityReward(),team_spirit, 1), 3),  # Reward strong touches
        (SaveBoostReward(), 1),
        (AirTouchReward(), 5),
        (JumpTouchReward(), 5),
        (EventReward(team_goal=goal_reward, concede=concede_reward), 200), # Giant reward for scoring
        (ZeroSumReward(LiuDistanceBallToGoalReward(),team_spirit, 1), 0.5),
        (SpeedTowardBallReward(), 0.8), # Move towards the ball!
        (FaceBallReward(), 0.2), # Make sure we don't start driving backward at the ball
        (InAirReward(), 0.15), # Make sure we don't forget how to jump
        (VelocityBallToGoalReward(), 0.2), #discourage corners
    )
```

## Finding the net
Bot was still preferring to push the ball in the corner than score, I tried adding an alignment factor to only reward the velocity toward the goal where the balls direction would actually put it in the goal, not rewarding on misses. But the alignment factor was not working right, so removed that and increased the reward ratio between scoring and everything else. Really make the goal, the goal! It will use what it knows to make that happen. Done over 100k steps again

```py
rew5 = CombinedReward.from_zipped(
        (ZeroSumReward(TouchVelocityReward(),team_spirit, 1), 3),  # Reward strong touches
        (SaveBoostReward(), 1),
        (AirTouchReward(), 5),
        (JumpTouchReward(), 5),
        (EventReward(team_goal=goal_reward, concede=concede_reward), 300), # Giant reward for scoring
        (ZeroSumReward(LiuDistanceBallToGoalReward(),team_spirit, 1), 0.1),
        (SpeedTowardBallReward(), 0.5), # Move towards the ball!
        (FaceBallReward(), 0.05), # Make sure we don't start driving backward at the ball
        (InAirReward(), 0.015), # Make sure we don't forget how to jump
        (VelocityBallToGoalReward(), 0.1), #discourage corners
        # (ModifiedVelocityBallToGoalReward(), 0.1),
    )
```

Bot is now consistently scoring on net, jumping a lot, raising min height of jump touch to encourage it to jump higher. Lower reward for hard touches to encourage more possession and dribbling. Added a very light demo reward of 1. Can only have 1 EventReward(), thats why demo is scaled down so low instead of being in its own function. Switching the state setter to do kickoffs instead of random now that the bots can play well. Starting from 
1.14 billion, transitioning over 100k.
```py
rew6 = CombinedReward.from_zipped(
        (ZeroSumReward(TouchVelocityReward(),team_spirit, 1), 1),  # Reward strong touches
        (SaveBoostReward(), 1), #sqrt of boost level (0,1) boost more important the less you have 
        (AirTouchReward(), 5),
        (JumpTouchReward(350), 5), # increase min ball height needed to get reward 
        (EventReward(team_goal=goal_reward, concede=concede_reward, demo=0.003), 300), # Giant reward for scoring
        (ZeroSumReward(LiuDistanceBallToGoalReward(),team_spirit, 1), 0.1),
        (SpeedTowardBallReward(), 0.2), # Move towards the ball!
        (FaceBallReward(), 0.05), # Make sure we don't start driving backward at the ball
        (InAirReward(), 0.005), # Make sure we don't forget how to jump
        (VelocityBallToGoalReward(), 0.1), #discourage corners
    )
```

Still working good. Was trying to encourage more aerials and boost conservation. In retrospect, those are at odds with each other. A new reward relating boost amount and aerials may be needed. Made changes to hyperparameters here.
Increasing ts_per_iteration and ppo_batch_size to 300_000, exp_buffer to 900_000. (all tripled)
```py
rew7 = CombinedReward.from_zipped(
        (ZeroSumReward(TouchVelocityReward(),team_spirit, 1), 1),  # Reward strong touches
        (SaveBoostReward(), 1.2),
        (AirTouchReward(), 5),
        (JumpTouchReward(350), 5),
        (EventReward(team_goal=goal_reward, concede=concede_reward, demo=0.003), 300), # Giant reward for actually hitting the ball
        (ZeroSumReward(LiuDistanceBallToGoalReward(),team_spirit, 1), 0.10),
        (SpeedTowardBallReward(), 0.2), # Move towards the ball!
        (FaceBallReward(), 0.05), # Make sure we don't start driving backward at the ball
        (InAirReward(), 0.010), # Make sure we don't forget how to jump
        (VelocityBallToGoalReward(), 0.1), #discourage corners
    )
```


## Results and ranking
I stopped messing with rewards and hyperparemeters at 1.35 Billion. Let it cook until 2 Billion.
Bot is now provably better than all Psyonix bots, and the level 2 and 3 community designated benchmark bots, tensor_bot (2), and self_driving_car (3).


