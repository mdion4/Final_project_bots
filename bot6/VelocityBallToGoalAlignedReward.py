#
#Original cpp code from Discord user "SA | GamingShorts"
# Translated to python with ChatGPT-o1-preview
#
import numpy as np
from rlgym_sim.utils import RewardFunction
from rlgym_sim.utils.gamestates import GameState, PlayerData
from rlgym_sim.utils.common_values import (
    BLUE_TEAM,
    ORANGE_TEAM,
    BLUE_GOAL_BACK,
    ORANGE_GOAL_BACK,
    BALL_MAX_SPEED,
)

class VelocityBallToGoalAlignedReward(RewardFunction):
    def __init__(self, own_goal=False):
        super().__init__()
        self.own_goal = own_goal

    def reset(self, initial_state: GameState):
        pass  # No state to reset

    def get_reward(self, player: PlayerData, state: GameState, previous_action):
        # Determine the target goal
        target_orange_goal = player.team_num == BLUE_TEAM
        if self.own_goal:
            target_orange_goal = not target_orange_goal

        target_pos = (
            np.array(ORANGE_GOAL_BACK) if target_orange_goal else np.array(BLUE_GOAL_BACK)
        )

        # Calculate the direction vectors and normalize them
        ball_pos = state.ball.position
        ball_vel = state.ball.linear_velocity

        # Direction from the ball to the target goal
        ball_dir_to_goal = target_pos - ball_pos
        ball_dir_to_goal_norm = np.linalg.norm(ball_dir_to_goal)
        if ball_dir_to_goal_norm > 0:
            ball_dir_to_goal_normalized = ball_dir_to_goal / ball_dir_to_goal_norm
        else:
            ball_dir_to_goal_normalized = np.zeros(3)

        # Ball's velocity direction
        ball_vel_norm = np.linalg.norm(ball_vel)
        if ball_vel_norm > 0:
            ball_vel_normalized = ball_vel / ball_vel_norm
        else:
            ball_vel_normalized = np.zeros(3)

        # Calculate alignment factor
        diff = ball_dir_to_goal_normalized - ball_vel_normalized
        unalignment = np.linalg.norm(diff)
        MAX_UNALIGNMENT = 0.7
        alignment_factor = 1.0 - min(unalignment, MAX_UNALIGNMENT) / MAX_UNALIGNMENT
        # print(f'Alignment factor: {alignment_factor}')

        # Calculate the velocity alignment reward
        velocity_alignment_reward = np.dot(
            ball_dir_to_goal_normalized, ball_vel
        ) / BALL_MAX_SPEED
        # print(f'velocity_alignment_reward: {velocity_alignment_reward}')

        # Combine and clamp to ensure non-negative reward
        reward = alignment_factor * velocity_alignment_reward
        print(f'VelocityAligned reward: {reward}')
        reward = max(0.0, reward)
        # print(f'VelocityAligned reward: {reward}')
        return reward
