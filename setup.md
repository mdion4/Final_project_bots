# Setting up Environments
- **Things you need**
  - Rocket League on Epic or Steam
  - Bakkesmod from https://bakkesplugins.com/
  - RLbot from https://rlbot.org/
  - A decent computer
  - A venv with python 3.9
- ## **Setup Guides**
  - Follow the steps from this Wonderful PPO guide https://github.com/ZealanL/RLGym-PPO-Guide/blob/main/intro.md.
  This details Setting up Python, Git, RLGym-PPO, RocketSim, rlgym_sim, getting the Rocket League collision maps, PyTorch install for CUDA, and having an example bot set up.

  **Visualizer**
  - You will need to be able to visualize your bots behavior to see its progress. Follow this guide to set up an easy to use visualizer https://github.com/ZealanL/RocketSimVis
  - More details on using this will be provided in the learner section.

  **RLBot**
  - This allows your bot to train in a simulated environment, but how do you actually use it in a match? RLBot loads the bot into Rocket League.
  - Follow the steps in this guide to translate the bot to work in RLBot https://github.com/ZealanL/RLGym-PPO-RLBot-Example

  - Make sure to use the same obs_builder values as we did in training
    ```py
    #in bot.py
    #in __init__
    self.obs_builder = obs_builder = YourOBS(
        pos_coef=np.asarray([1 / 4096, 1 / 6000, 1 / 2044]),
        ang_coef=1 / np.pi,
        lin_vel_coef=1 / 2300,
        ang_vel_coef=1 / 5.5)
    ```
  - This all assumes you are using LookupAct or NectoAct
  - In RLBot, I had to click on the menu and select "install missing packages" and type gym.
  - Add the following line to bot.cfg under [Locations] `looks_config = ./appearance.cfg` 
  - In agent.py replace the act function with this 
    ```py
    def act(self, state):
      with torch.no_grad():
        action_idx, probs = self.policy.get_action(state, True)
      
      action = np.array(self.action_parser.parse_actions([action_idx], None))
      if len(action.shape) == 2:
        if action.shape[0] == 1:
          action = action[0]
      
      if len(action.shape) != 1:
        raise Exception("Invalid action:", action)
      
      return action
    ```      
  - In your_act.py add (not relplace) this:
      ```py
      def parse_actionss(self, action: int, state) -> np.ndarray:
        return self._lookup_table[action]
      ```  

## Example Bot
After following the first PPO guide, you'll have an example bot thats ready to train. You run it by navigating to the folder where example.py is via the command line, then type example.py. This starts the training, and will output each iteration report at regular intervals which include the following metrics (and more advanced ML stuff):
  - **Policy Reward**: Average reward among all episodes in that iteration. Don't freak out when this starts going down as you pull back form dense rewards and increase sparse, ZeroSumRewards.
  - **Value Function Loss**: This should be trending down to indicate improvement in the critic. The specific value is less important. This is like a measure of accuracy for the critic to predict the value function.
  - **SB3 Clip Fraction**: This should typically be kept around 0.1 to 0.08 by adjusting the learning rate.
  - **Collected Steps per second**: How many steps were collected (not yet processed) per second.
  - **Overall Steps per second**: Number of steps processed per second. This is what is being referred to by SPS.
  - **Timestep collection Time**: How long it took to collect all the steps in the iteration. A higher ts_per_iteration means this will take longer, as the goal is farther away.
  - **Timestep Consumption Time**: How long it took do finish analyzing/ learning from the data set. More data, more epochs, more timesteps, all increase this.
  - **Cumulative Timesteps**: Total number of time steps, can be used to calculated how much time the bot has experienced training. This is the number of steps the gets referred to when talking about some number of steps, like how my bot trained for 2 billion steps.

**Logging**
The ExampleLogger() extends MetricsLogger which sends data to wandb which handles the graphing. Look at https://github.com/ZealanL/RLGym-PPO-Guide/blob/main/graphs.md for more information on wandb (Weights and Biases) and the graphs there.

**Environment**
The Training Environment is made up of a few components:
  - **Rewards**:
    - Will be covered more in rewards.md.
  - **Obs Builder**:
    - The brain of the bot, we will stick with DefaultObs(), but note that we are using different values for some of the parameters, this will come up later when porting this to RLBot.
    ```
    obs_builder = DefaultObs(
        pos_coef=np.asarray([1 / common_values.SIDE_WALL_X, 1 / common_values.BACK_NET_Y, 1 / common_values.CEILING_Z]),
        ang_coef=1 / np.pi,
        lin_vel_coef=1 / common_values.CAR_MAX_SPEED,
        ang_vel_coef=1 / common_values.CAR_MAX_ANG_VEL)
    ```  
  - **Action Parser**
      - This is how a bot organizes available actions and chooses the next one. We will change this to LookupAction. it can be found at https://github.com/RLGym/rlgym-tools/blob/main/rlgym_tools/extra_action_parsers/lookup_act.py. Copy the entire file into the same folder as example.py, and import LookupAction into example.py, replacing ContinuousAction(). Changing this usually necessitates a reset of the bot.
  - **State Setter**
      - Controls the state each episode starts in. I recommend using RandomState(True, True, False) which randomizes the starting locations and velocitys of players and the ball. DefaultState() is kickoffs, which is apparently not as useful for training bots in the beginning. I swapped to DefaultState() after ~1 billion steps, once the bots could move well and score intentionally. There is a way to have both options be available using WeightedStateSetter() picks one of the StateSetters each episode, with adjustable weight to its odds of being chosen. You could start with RandomState() having a higher weight than DefaultState(), adjusting them as training goes on. 
  - **Terminal conditions**
      - What determines the end of an episode. A ball being untouched for 10 seconds or a goal being scored will end the episode and start the next. I needed to double the timeout condition once I switched to kickoffs at first since the ball could end up very far away and the bots too slow to reach it. Looking back, that may not have been a good idea since my bot shows no sense of urgency after a bad kickoff that heading straight to its net, it takes its time getting mid boost instead of rushing back to defend. We will keep these terminal conditions for this tutorial.
  - **Number of players**
      -team_size and spawn_opponents determine the number of players. We will keep them at 1 and True. This means we are training out bots in 1v1 mode.

After env = rlgym_sim.make(), and before returning env, insert the line that connects to the visualizer:
```
import rocketsimvis_rlgym_sim_client as rsv
    type(env).render = lambda self: rsv.send_state_to_rocketsimvis(self._prev_state)
```
**Learner**

The Learner() class is what controls the learning loop and is home to most of our hyperparemters, as well as other useful settings. The hyperparameters will be covered in learner.py

**Recommended QoL additions/changes**

  - A function that grabs the most recent checkpoint. This requires `add_unix_timestamp=True` to be included in the Learner contructor.
    ```
      def get_most_recent_checkpoint() -> str:
          checkpoint_load_dir = "data/checkpoints/"
          checkpoint_load_dir += str(
              max(os.listdir(checkpoint_load_dir), key=lambda d: int(d.split("-")[-1])))
          checkpoint_load_dir += "/"
          checkpoint_load_dir += str(
              max(os.listdir(checkpoint_load_dir), key=lambda d: int(d)))
          return checkpoint_load_dir
      
      ## in __main__ ##
      try:
              checkpoint_load_dir = get_most_recent_checkpoint()
              print(f"Loading checkpoint: {checkpoint_load_dir}")
          except:
              print("checkpoint load dir not found.")
              checkpoint_load_dir = None
    ```
  - Adding a rendering flag to the command line
    ```
    #use cmd line to render or not
    render = False
    render_delay = False
    if len(sys.argv) > 1 and sys.argv[1].lower() == "true":
        render = True
        render_delay = gamespeed/1.2
    ```
  - Defining a constant for gamespeed based off the constant STEP_TIME `STEP_TIME = (TICK_SKIP/GAME_TICK_RATE) #1/15`
      This is used to set render_delay, which is really a scalar on the speed of the visualizer. The smaller this is, the slower the visualizer plays the game. Mine was set to 1/36, which was still faster than real time. There is not a nice easy conversion between sim_speed and real time that I have found, and may depend on processing power. The visualizer only slows down 1 game, so just be aware that it is old data, and becomes less indicitive of current behavior as time goes on. Just use it to check in for a few minutes to observe the bot behaviors and think about what needs to change.

  - Specifying some wandb data in the Learner() constructor. This will let you organize runs better in wandb
    ```
    wandb_run_name="bot", # Name of your Weights And Biases run.
    wandb_project_name="1v1", # Name of your Weights And Biases project.
    wandb_group_name="final bots", # Name of the Weights And Biases project group.
    ```

  - Adjust n_checkpoints to keep and save_every_ts to a level acceptable to you based on how many steps per second you get. I get around 13,000 sps, and save every 50 million steps which is roughly every hour. It can become quite a lot of data.
    

## Now check out learner.md or rewards.md
# [learner.md](learner.md)
# [rewards.md](rewards.md)
