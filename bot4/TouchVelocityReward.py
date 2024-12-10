# Initially written by chatGPT o1-preview
import numpy as np
from rlgym_sim.utils import RewardFunction
from rlgym_sim.utils.gamestates import GameState, PlayerData

class TouchVelocityReward(RewardFunction):
    def __init__(self):
        super().__init__()
        self.prev_ball_vel = np.zeros(3)
        self.last_touch = False

    def reset(self, initial_state: GameState):
        self.prev_ball_vel = initial_state.ball.linear_velocity
        self.last_touch = False

    def get_reward(self, player: PlayerData, state: GameState, previous_action) -> float:
        reward = 0.0

        # Check if the player touched the ball since the last step
        if player.ball_touched and not self.last_touch:
            # Calculate change in ball velocity
            ball_vel = state.ball.linear_velocity
            delta_vel = ball_vel - self.prev_ball_vel
            delta_speed = np.linalg.norm(delta_vel)
            # Scale the reward (adjust the scaling factor as needed)
            reward = delta_speed / 2300  # Normalize by max car speed (approximate max ball speed)
            reward = min(reward, 1.0)  # Ensure reward does not exceed 1
            self.last_touch = True
        else:
            self.last_touch = player.ball_touched
            reward = 0.0

        self.prev_ball_vel = state.ball.linear_velocity
        return reward
