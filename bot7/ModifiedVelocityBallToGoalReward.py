import numpy as np
from rlgym_sim.utils import RewardFunction
from rlgym_sim.utils.gamestates import GameState, PlayerData
from rlgym_sim.utils.common_values import BLUE_TEAM, ORANGE_TEAM, BLUE_GOAL_BACK, ORANGE_GOAL_BACK, BALL_MAX_SPEED

def normalize(v):
    return v/(np.linalg.norm(v)+1e-10)

class ModifiedVelocityBallToGoalReward(RewardFunction):
    def __init__(self, own_goal=False, min_alignment=0.7):
        super().__init__()
        self.own_goal = own_goal
        self.min_alignment = min_alignment
    def reset(self, initial_state: GameState):
        pass

    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        # Determine which goal is the objective
        if (player.team_num == BLUE_TEAM and not self.own_goal) or (player.team_num == ORANGE_TEAM and self.own_goal):
            objective = np.array(ORANGE_GOAL_BACK)
        else:
            objective = np.array(BLUE_GOAL_BACK)

        ball_pos = state.ball.position
        ball_vel = state.ball.linear_velocity

        ball_vel_dir = normalize(ball_vel)

        # Calculate the direction from the ball to the goal
        ball_to_goal = objective - ball_pos
        ball_to_goal_dir = normalize(ball_to_goal)

        # Compute alignment
        alignment = np.dot(ball_vel_dir, ball_to_goal_dir)
        print(f'Modified velocity alignment: {alignment}')
        
        # Calculate the reward using minimum alignment
        # Change min_alignment (0 to 1) to adjust strictness
        # 1 = perfectly aligned
        # 0 = not aligned at all
        alignment_clipped = max(0.0, alignment)
        reward = min(alignment_clipped, self.min_alignment) / self.min_alignment
        print(f'Modified velocity alignment reward: {reward}')

        vel_towards_goal = np.dot(ball_vel, ball_to_goal_dir)
        print(f'Modified velocity vel to goal: {vel_towards_goal}')

        # Compute the reward for velocity towards goal times the angle towards goal
        vel_reward = (vel_towards_goal / BALL_MAX_SPEED) ** 2
        print(f'Modified velocity vel_reward: {vel_reward}')
        reward *= vel_reward

        # Ensure the reward is between 0 and 1
        reward = np.clip(reward, 0.0, 1.0)
        print(f'Modified velocity reward: {reward}')

        return reward
    
    