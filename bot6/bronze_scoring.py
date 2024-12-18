import numpy as np
from rlgym_sim.utils.gamestates import GameState, PlayerData
from rlgym_ppo.util import MetricsLogger
import os
from rlgym_sim.utils import RewardFunction
from rlgym_sim.utils.common_values import CAR_MAX_SPEED, CEILING_Z
import rlgym_sim.utils.common_values as CommonValues
from ModifiedVelocityBallToGoalReward import ModifiedVelocityBallToGoalReward
from VelocityBallToGoalAlignedReward import VelocityBallToGoalAlignedReward
from custom_combinedreward import LogCombinedReward
from curriculum_reward import AnnealRewards



KPH_TO_VEL = 250/9
class ExampleLogger(MetricsLogger):
    def _collect_metrics(self, game_state: GameState) -> list:
        return [game_state.players[0].car_data.linear_velocity,
                game_state.players[0].car_data.rotation_mtx(),
                game_state.orange_score]

    def _report_metrics(self, collected_metrics, wandb_run, cumulative_timesteps):
        avg_linvel = np.zeros(3)
        for metric_array in collected_metrics:
            p0_linear_velocity = metric_array[0]
            avg_linvel += p0_linear_velocity
        avg_linvel /= len(collected_metrics)
        report = {"x_vel":avg_linvel[0],
                  "y_vel":avg_linvel[1],
                  "z_vel":avg_linvel[2],
                  "Cumulative Timesteps":cumulative_timesteps}
        wandb_run.log(report)

def get_most_recent_checkpoint() -> str:
    checkpoint_load_dir = "data/checkpoints/"
    checkpoint_load_dir += str(
        max(os.listdir(checkpoint_load_dir), key=lambda d: int(d.split("-")[-1])))
    checkpoint_load_dir += "/"
    checkpoint_load_dir += str(
        max(os.listdir(checkpoint_load_dir), key=lambda d: int(d)))
    return checkpoint_load_dir        

class InAirReward(RewardFunction): # We extend the class "RewardFunction"
    # Empty default constructor (required)
    def __init__(self):
        super().__init__()

    # Called when the game resets (i.e. after a goal is scored)
    def reset(self, initial_state: GameState):
        pass # Don't do anything when the game resets

    # Get the reward for a specific player, at the current state
    def get_reward(self, player: PlayerData, state: GameState, previous_action) -> float:
        
        # "player" is the current player we are getting the reward of
        # "state" is the current state of the game (ball, all players, etc.)
        # "previous_action" is the previous inputs of the player (throttle, steer, jump, boost, etc.) as an array
        
        if not player.on_ground:
            # We are in the air! Return full reward
            return 1
        else:
            # We are on ground, don't give any reward
            return 0

#from apollo-bot on RL-bot discord
class TouchBallRewardScaledByHitForce(RewardFunction):
    
    def __init__(self):
        super().__init__()
        self.max_hit_speed = 130 * KPH_TO_VEL
        self.last_ball_vel = None
        self.cur_ball_vel = None

    # game reset, after terminal condition
    def reset(self, initial_state: GameState):
        self.last_ball_vel = initial_state.ball.linear_velocity
        self.cur_ball_vel = initial_state.ball.linear_velocity

    # happens 
    def pre_step(self, state: GameState):
        self.last_ball_vel = self.cur_ball_vel
        self.cur_ball_vel = state.ball.linear_velocity

    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        if player.ball_touched:
            # print(f'touchscaledhit: last vel:{self.last_ball_vel} cur_vel:{self.cur_ball_vel}')
            reward = np.linalg.norm(self.cur_ball_vel - self.last_ball_vel) / self.max_hit_speed
            # print(f'TouchBallRewardScaledByHitForce: {reward}')
            return reward
        return 0
    

# Initially written by chatGPT o1-preview
class TouchVelocityReward(RewardFunction):
    def __init__(self, min_velocity_change=300, cooldown_steps=5):
        super().__init__()
        self.prev_ball_vel = np.zeros(3)
        # chatGPT v2
        self.min_velocity_change = min_velocity_change  # Minimum speed change to reward
        self.cooldown_steps = cooldown_steps
        self.cooldown_counter = 0

    def reset(self, initial_state: GameState):
        self.prev_ball_vel = initial_state.ball.linear_velocity
        self.cooldown_counter = 0

    def get_reward(self, player: PlayerData, state: GameState, previous_action) -> float:
        reward = 0.0

        # Check if the player touched the ball since the last step
        # originally if "player.ball_touched and not self.last_touch" 
        # this did not promote dribbling or setting up hook shots.
        if self.cooldown_counter > 0:
            self.cooldown_counter -= 1

        elif player.ball_touched:
            ball_vel = state.ball.linear_velocity
            delta_vel = ball_vel - self.prev_ball_vel
            delta_speed = np.linalg.norm(delta_vel)
            # print(f'touchVelocityReward ball:{ball_vel} delta_vel:{delta_vel}, del_spd{delta_speed}')

            if delta_speed > self.min_velocity_change:
                reward = delta_speed / CAR_MAX_SPEED
                reward = min(reward, 1.0)
                self.cooldown_counter = self.cooldown_steps  # Activate cooldown

        self.prev_ball_vel = state.ball.linear_velocity
        # print(f'touchVelocityReward:{reward}')
        return reward

class AirTouchReward(RewardFunction):
    MAX_TIME_IN_AIR = 1.75  # Maximum reasonable aerial time in seconds

    def __init__(self):
        super().__init__()
        self.last_player_on_ground = True
        self.air_time = 0.0

    def reset(self, initial_state: GameState):
        self.last_player_on_ground = True
        self.air_time = 0.0

    def get_reward(self, player: PlayerData, state: GameState, previous_action):
        reward = 0.0

        # Update air time
        if not player.on_ground:
            if self.last_player_on_ground:
                # Player just left the ground
                self.air_time = 0.0
            self.air_time += (1/15)  # Increment air time
            # print(f'airtime: {self.air_time}')
        else:
            self.air_time = 0.0  # Reset air time when on the ground

        self.last_player_on_ground = player.on_ground

        # Calculate fractions
        air_time_frac = min(self.air_time, self.MAX_TIME_IN_AIR) / self.MAX_TIME_IN_AIR
        height_frac = state.ball.position[2] / CommonValues.CEILING_Z

        # Optional: Multiply by 1 if player touched the ball in the air
        if player.ball_touched and not player.on_ground:
            # Reward is the minimum of air time fraction and ball height fraction
            reward = min(air_time_frac, height_frac)
            # print(f'AirTouchReward: {reward}')
        return reward

class SaveBoostReward(RewardFunction):
    def __init__(self):
        super().__init__()

    # Do nothing on game reset
    def reset(self, initial_state: GameState):
        pass

    # Get the reward for a specific player, at the current state
    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        reward = np.sqrt(player.boost_amount)
        return reward
    
class SpeedTowardBallReward(RewardFunction):
    # Default constructor
    def __init__(self):
        super().__init__()

    # Do nothing on game reset
    def reset(self, initial_state: GameState):
        pass

    # Get the reward for a specific player, at the current state
    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        # Velocity of our player
        player_vel = player.car_data.linear_velocity
        
        # Difference in position between our player and the ball
        # When getting the change needed to reach B from A, we can use the formula: (B - A)
        pos_diff = (state.ball.position - player.car_data.position)
        
        # Determine the distance to the ball
        # The distance is just the length of pos_diff
        dist_to_ball = np.linalg.norm(pos_diff)
        
        # We will now normalize our pos_diff vector, so that it has a length/magnitude of 1
        # This will give us the direction to the ball, instead of the difference in position
        # Normalizing a vector can be done by dividing the vector by its length
        dir_to_ball = pos_diff / dist_to_ball

        # Use a dot product to determine how much of our velocity is in this direction
        # Note that this will go negative when we are going away from the ball
        speed_toward_ball = np.dot(player_vel, dir_to_ball)
        
        if speed_toward_ball > 0:
            # We are moving toward the ball at a speed of "speed_toward_ball"
            # The maximum speed we can move toward the ball is the maximum car speed
            # We want to return a reward from 0 to 1, so we need to divide our "speed_toward_ball" by the max player speed
            reward = speed_toward_ball / CAR_MAX_SPEED
            return reward
        else:
            # We are not moving toward the ball
            # Many good behaviors require moving away from the ball, so I highly recommend you don't punish moving away
            # We'll just not give any reward
            return 0
class JumpTouchReward(RewardFunction):
    """
    a ball touch reward that only triggers when the agent's wheels aren't in contact with the floor
    adjust minimum ball height required for reward with 'min_height' as well as reward scaling with 'exp'
    """

    def __init__(self, min_height=92, exp=0.2):
        self.min_height = min_height
        self.exp = exp

    def reset(self, initial_state: GameState):
        pass

    def get_reward(
            self, player: PlayerData, state: GameState, previous_action: np.ndarray
            ) -> float:
        if player.ball_touched and not player.on_ground and state.ball.position[2] >= self.min_height:
            return ((state.ball.position[2] - 92) ** self.exp)-1

        return 0
    
def build_rocketsim_env():
    import rlgym_sim
    from lookup_act import LookupAction
    from rlgym_sim.utils.reward_functions import CombinedReward
    from zero_sum_reward import ZeroSumReward
    from rlgym_sim.utils.reward_functions.common_rewards import VelocityPlayerToBallReward,  \
        EventReward, FaceBallReward
    from rlgym_sim.utils.obs_builders import DefaultObs
    from rlgym_sim.utils.terminal_conditions.common_conditions import NoTouchTimeoutCondition, GoalScoredCondition
    from rlgym_sim.utils import common_values
    from necto_act import NectoAction
    from rlgym_sim.utils.state_setters import RandomState, DefaultState
    from ball_goal_rewards import VelocityBallToGoalReward, LiuDistanceBallToGoalReward
    spawn_opponents = True
    team_size = 1
    game_tick_rate = 120
    tick_skip = 8
    step_time = 1/15
    timeout_seconds = 20
    timeout_ticks = int(round(timeout_seconds * game_tick_rate / tick_skip))

    action_parser = LookupAction()
    terminal_conditions = [NoTouchTimeoutCondition(timeout_ticks), GoalScoredCondition()]

    aggression_bias = 0.2
    goal_reward = 1
    concede_reward = - goal_reward * (1 - aggression_bias)

    team_spirit = 0.000
    opp_penalty_scale = 0.00

    # rewards_to_combine = (SpeedTowardBallReward(),
    #                       InAirReward(),
    #                       EventReward(touch=1),
    #                       FaceBallReward(),
    #                       VelocityBallToGoalReward())
    # reward_weights = (2, 0.15, 0.5, 0.1, 4)

    # reward_fn = CombinedReward(reward_functions=rewards_to_combine,
    #                            reward_weights=reward_weights)
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
    rew4 = CombinedReward.from_zipped(
        (ZeroSumReward(TouchVelocityReward(),team_spirit, 1), 3),  # Reward strong touches
        (SaveBoostReward(), 1),
        (AirTouchReward(), 5),
        (JumpTouchReward(), 5),
        (EventReward(team_goal=goal_reward, concede=concede_reward), 200), # Giant reward for actually hitting the ball
        (ZeroSumReward(LiuDistanceBallToGoalReward(),team_spirit, 1), 0.5),
        (SpeedTowardBallReward(), 0.8), # Move towards the ball!
        (FaceBallReward(), 0.2), # Make sure we don't start driving backward at the ball
        (InAirReward(), 0.15), # Make sure we don't forget how to jump
        (VelocityBallToGoalReward(), 0.2), #discourage corners
        
    )
    rew5 = CombinedReward.from_zipped(
        (ZeroSumReward(TouchVelocityReward(),team_spirit, 1), 3),  # Reward strong touches
        (SaveBoostReward(), 1),
        (AirTouchReward(), 5),
        (JumpTouchReward(), 5),
        (EventReward(team_goal=goal_reward, concede=concede_reward), 300), # Giant reward for actually hitting the ball
        (ZeroSumReward(LiuDistanceBallToGoalReward(),team_spirit, 1), 0.1),
        (SpeedTowardBallReward(), 0.5), # Move towards the ball!
        (FaceBallReward(), 0.05), # Make sure we don't start driving backward at the ball
        (InAirReward(), 0.015), # Make sure we don't forget how to jump
        (VelocityBallToGoalReward(), 0.1), #discourage corners
        # (ModifiedVelocityBallToGoalReward(), 0.1),
    )
    rew6 = CombinedReward.from_zipped(
        (ZeroSumReward(TouchVelocityReward(),team_spirit, 1), 1),  # Reward strong touches
        (SaveBoostReward(), 1),
        (AirTouchReward(), 5),
        (JumpTouchReward(350), 5),
        (EventReward(team_goal=goal_reward, concede=concede_reward, demo=0.003), 300), # Giant reward for actually hitting the ball
        (ZeroSumReward(LiuDistanceBallToGoalReward(),team_spirit, 1), 0.1),
        (SpeedTowardBallReward(), 0.2), # Move towards the ball!
        (FaceBallReward(), 0.05), # Make sure we don't start driving backward at the ball
        (InAirReward(), 0.005), # Make sure we don't forget how to jump
        (VelocityBallToGoalReward(), 0.1), #discourage corners
    )
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
    reward_fn = AnnealRewards(rew1, 150_000_000, rew2)
    reward_fn = AnnealRewards(rew2, 300_000_000, rew3)
    reward_fn = AnnealRewards(rew3, 100_000_000, rew4)
    reward_fn = AnnealRewards(rew4, 100_000_000, rew5)
    reward_fn = AnnealRewards(rew5, 100_000_000, rew6)
    reward_fn = AnnealRewards(rew6, 100_000_000, rew7)
    reward_fn = rew7

    obs_builder = DefaultObs(
        pos_coef=np.asarray([1 / common_values.SIDE_WALL_X, 1 / common_values.BACK_NET_Y, 1 / common_values.CEILING_Z]),
        ang_coef=1 / np.pi,
        lin_vel_coef=1 / common_values.CAR_MAX_SPEED,
        ang_vel_coef=1 / common_values.CAR_MAX_ANG_VEL)
    state_setter = RandomState(True, True, False)
    state_setter = DefaultState()
    env = rlgym_sim.make(tick_skip=tick_skip,
                         team_size=team_size,
                         spawn_opponents=spawn_opponents,
                         terminal_conditions=terminal_conditions,
                         reward_fn=reward_fn,
                         obs_builder=obs_builder,
                         action_parser=action_parser,
                         state_setter=state_setter)
    

    import rocketsimvis_rlgym_sim_client as rsv
    type(env).render = lambda self: rsv.send_state_to_rocketsimvis(self._prev_state)

    return env

if __name__ == "__main__":
    import sys
    from rlgym_ppo import Learner
    metrics_logger = ExampleLogger()

    GAME_TICK_RATE = 120
    TICK_SKIP = 8
    STEP_TIME = (TICK_SKIP/GAME_TICK_RATE) #1/15

    gamespeed = STEP_TIME/2
    # 32 processes
    n_proc = 70
    try:
        checkpoint_load_dir = get_most_recent_checkpoint()
        print(f"Loading checkpoint: {checkpoint_load_dir}")
    except:
        print("checkpoint load dir not found.")
        checkpoint_load_dir = None
    #use cmd line to render or not
    render = False
    render_delay = False
    if len(sys.argv) > 1 and sys.argv[1].lower() == "true":
        render = True
        render_delay = gamespeed/1.2
    
    # educated guess - could be slightly higher or lower
    min_inference_size = max(1, int(round(n_proc * 0.9)))

    learner = Learner(build_rocketsim_env,
                      n_proc=n_proc,
                      render=render,
                      render_delay=render_delay,
                      min_inference_size=min_inference_size,
                      metrics_logger=metrics_logger,
                      ppo_batch_size=300000,
                      ts_per_iteration=300000,
                      exp_buffer_size=900000,
                      ppo_minibatch_size=50000,
                      ppo_ent_coef=0.01,
                      ppo_epochs=3,
                      standardize_returns=True,
                      standardize_obs=False,
                      policy_layer_sizes=(2048, 1024, 1024, 1024),
                      critic_layer_sizes=(2048, 1024, 1024, 1024),
                      policy_lr=0.8e-4,
                      critic_lr=0.8e-4,
                      save_every_ts=50_000_000,
                      n_checkpoints_to_keep= 100,
                      timestep_limit= 1_000_000_000_000,
                      log_to_wandb=True,
                      wandb_run_name="bot8.3.1", # Name of your Weights And Biases run.
                      wandb_project_name="1v1", # Name of your Weights And Biases project.
                      wandb_group_name="final bots", # Name of the Weights And Biases project group.
                    #   checkpoint_load_folder=None,
                      checkpoint_load_folder=checkpoint_load_dir,
                      add_unix_timestamp=True
                      )
    learner.learn()