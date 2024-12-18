# Learner settings/hyperparameters

I touched on the learner settings that are more book keeping and functionality settings, but they dont really affect how the bot learns. The remaining settings are all hyperparameters. The full list of parameters can be found at https://github.com/AechPro/rlgym-ppo/blob/main/rlgym_ppo/learner.py

This all assumes using a GPU.

### n_proc
The number of training sessions running at once. You want this as high as possible until something crashes due to lack of memory. I was able to do 70 processes with a 12900k, 3080ti with 12GB VRAM, and 32GB RAM.   

### ppo_batch_size
The amount of data trained on per iteration. Keep this equal to ts_per_iteration, so you train on all the data.
Start out at 50,000 or 100,000. Smaller batch means more frequent updates as the bot discovers the environment.
Increase this as learning rate goes down. More refined data benefits from an increase in data points to learn from. 

### ppo_minibatch_size 
Should be a subset of ppo_batch_size. I started at 50,000. If you get errors your GPU VRAM is lacking and you should lower it.

### ts_per_iteration
How many steps an iteration is. See ppo_batch_size.

### exp_buffer_size
Holds the most recent batches until this is full, Learner trains on these batches as well as the most recent one. Should be triple ppo_batch_size

### ppo_ent_coef 
Basically the exploration factor to try new things instead of sticking with what it knows. Zealan claims 0.01 is ideal, so use that. I never modified this after setting it, though lowering it is suggested when polishing off a bot's training so it can hone what it knows.

### ppo_epochs
How many times the same batch of data is processed by the learner. Typically 1-4. I have used 2, 3, 4. My successful bot used 3.  Inverse relationship between steps per second and actual learning speed. More epochs means less steps per second, but more learning per step, while less epochs translates to more steps per second but a less learning per step. Like everything else, play around with it, find the sweet spot.

### layer_sizes
PPO uses 2 neural networks, one for the policy(actor) who plays the game and tries to maximize accumulated reward, and the critic, which learns to predict the reward the actor will receive. More layers and wider layers will increase overall skill ceiling, due to a larger brain, but may improve more slowly. It also is more computationally expensive resulting in lower SPS. I was able to run [2048, 2048, 1024, 1024] for both, but my final bot used [2048, 1024, 1024, 1024]. I think that allowed me to increase the number of processes from 40 to 70. First version had   
7,344,128 parameters, while the second version had 4,197,376, which adds up to why those were my max number of processes I could run. 293765120 - 293816320 = 51200,

### policy_lr, critic_lr
learning rate of the networks. Needs to be small enough to make progress, but not too small that it hardly learns. Too big and the networks fall apart, stuck in a chaotic state it cant make sense of. Start out "high" and reduce as bot improves. Ive seen starting values between 4e-4 and 2e-4. My winning bot started at 2e-4, lowered to 1e-4 when it could score, and lowered again to 0.8e-4 as it no longer had to deal with reward changes and focused on improving what it knew. This is something to play around with. Setting policy to 0 is referred to as freezing the policy, so it maintains its behavior while the reward function changes, and the critic learns the new value function. I have not had good experience with freezing policy and just use Annealing. 
