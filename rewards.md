# Rewards and Weights
### **What are rewards?**

Rewards are how your bot learns what is good and what is bad. The bot seeks to maximize its cumulative rewards. In PPO (Proximal Policy Optimization) the rewards are your way of communicating the desired behaviors to your bot. This is really the core of training a bot via PPO, as you must do a lot of trial and error on different reward functions, different weights, how to avoid getting stuck at a local maxima, properly adjusting rewards without breaking the bot etc. 


### **Types of rewards**

  Reward functions return rewards. It is structured like this (rewardFunc(), weight).
  
  There are continuous rewards, like VelocityBallToGoal() that ar caclulated each step, and there are discrete awards that are awarded once, once some condition is met. Really all rewards are technically continuous, as all reward functions are calculated every step, so it can be better to think of rewards as sparse vs dense. Dense rewards will nearly always return some value, usually for things like velocities, direction, distance, boost amount, etc, while sparse rewards usually just return 0, as all the conditions are not met. 
  
  Sparse rewards should be weighted higher due to their lower occurrence rate. The exact value of the weight is not important as the weights are all relative to eachother. (Foo(), 1), (Foo2(), 10) is the same as (Foo(), 0.01), (Foo2(), 0.1). 
  
  There is a built in EventReward() with different events like touch, team_goal, concede, demo etc. I recommend starting out with (EventReward(touch=1),50)



### **Structure of a reward function**

  Reward functions are actually a class that extends RewardFunction, and have 3 methods they need to implement in our case since we are using the default obs. Other more advanced obs can include an additional pre_step() but we won't worry about that.
    - \_\_init_\_ (self, ...) Add any parameters you need.
    - reset(self, initial_state: GameState)  _values at the beginning of an episode._
    - get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float: _where most of your lojic is implemented, returns the reward_
        It is usually a good practive to keep your reward between (0,1) for consistency with other rewards, so that balancing the weights of reward functions makes more sense. Of course there are exceptions, just something to keep in mind.

## Supporting Functions, meta reward functions
### **UH-OH, rlgym_sim only takes 1 reward function**
We just use combinedreward from zip to take care of that, and put the weights with its function for better readability. Following code snippet ripped from Zealan's PPO guide
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
As the name implies, a reward for one bot is a penalty for the opposing bot. Zero sum reward wraps a reward function, and uses team_spirit and opp_penalty to scale the rewards. team_spirit distributes rewards amongst teammates, a value of 0 means individuals do not share with teammates and keep the entire reward for themselves; a value of 1 means the reward is shared comepletly. Opp_penalty is a scalar usually between 0 and 1 dictating how much penalty to incur. It is only truly ZeroSum if opp penalty is 1, where the average reward for some reward function is 0. This is good to wrap around functions that deal with things the bot should be trying to prevent the opponent from doing, like touching the ball, having speed, having boost. It doesnt make as much sense for things like being in the air, facing the ball. I would suggest starting the opp_penalty low to start, then slowly increase as training goes on. Too much noise is bad for the bot in the early stages while it is still confused.


###**AnnealingRewards**
I believe this to be the most crucial part of training a bot beyond the inital ballchading stage. AnnealRewards is also a wrapper function that linearly transitions between two or more reward functions over some number of steps. The pattern goes (rew1, num_steps_1_to_2, rew2, num_steps_2_to_3, etc). Unfortunately the intermediate reward functions arent saved anywhere, so if training ends before completing all the steps, you either have to revert to the checkpoint before calling AnnealRewards, or try calculating what the current reward function would be. Calculating wrong is highly likely to just break your bot. Even freezing the policy(setting policy_lr = 0) and allowing the critic to learn the new rewards is unlikely to keep your bot from breaking upon unfreezing the policy. 

I suggest having many iterations of your combined reward function, and only use AnnealRewards() to go from 1 version to another, as you only know the current reward functions used by the bot at either end of AnnealRewards(). Once the goal reward function is reached, training will continue using that reward function.


