import gymnasium as gym
import torch
import numpy as np
from copy import deepcopy


from config import args
from model import Actor , Critic
from replay_buffer import ReplayBuffer
from learning_curve import LearningCurve
from MAMC import MAMC
from evaluate import evaluate_actors
from select_survey_corps import select_survey_corps



###### 建立環境 ######
env = gym.make(args.env_name)


###### 設定隨機種子 ######
np.random.seed(args.seed)
torch.manual_seed(args.seed)
env.reset(seed=args.seed)
env.action_space.seed(args.seed)


###### 確定維度 ######
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
max_action = env.action_space.high


###### 初始化 actors, critics, critic_targets ######
actors: list[Actor] = []
for i in range(args.actor_size):
    actor = Actor(state_dim, action_dim, max_action).to(args.device)
    actors.append(actor)

critics: list[Critic] = []
for i in range(args.critic_size):
    critic = Critic(state_dim, action_dim).to(args.device)
    critics.append(critic)

critic_targets = deepcopy(critics)


###### 初始化 actor_optimizers, critic_optimizers ######
actor_optimizers: list[torch.optim.Adam] = []
for actor in actors:
    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=args.actor_learning_rate)
    actor_optimizers.append(actor_optimizer)

critic_optimizers: list[torch.optim.Adam] = []
for critic in critics:
    critic_optimizer = torch.optim.Adam(critic.parameters(), lr=args.critic_learning_rate)
    critic_optimizers.append(critic_optimizer)


###### 初始化 replay buffer ######
replay_buffer = ReplayBuffer()


###### 初始化 learning curve ######
learning_curve = LearningCurve(actors)


###### 初始化部分參數 ######
survey_corps_size = int(np.ceil(np.sqrt(args.actor_size)))


###### 初始化 MAMC ######
mamc = MAMC(max_action)


###### 初始化環境 ######
state , _ = env.reset()


###### 開始訓練 ######
while learning_curve.steps < args.max_steps:

    if learning_curve.steps < args.start_steps:

        action = env.action_space.sample()  # 純隨機動作

    else:

        mamc.train_critics(actors, critics, critic_targets, critic_optimizers, replay_buffer)
        mamc.train_actors(actors, critics, actor_optimizers, replay_buffer)
        evaluate_actors(actors, critics, replay_buffer)
        survey_corps = select_survey_corps(actors, survey_corps_size)
        actor = survey_corps[np.random.randint(survey_corps_size)]

        with torch.no_grad():
            state_ = torch.tensor(state, dtype=torch.float32, device=args.device)
            action = actor(state_)
            action = action.cpu().numpy()

        noise = np.random.normal(0, max_action * args.exploration_noise, size=action_dim)
        action = (action + noise).clip(-max_action , max_action)


    next_state , reward , done , reach_step_limit , _ = env.step(action)

    replay_buffer.push(state, action, next_state, reward, not done)


    if done or reach_step_limit:
        state , _ = env.reset()
    else:
        state = next_state


    learning_curve.add_step()


    if learning_curve.steps >= args.start_steps:
        if learning_curve.steps % args.test_performance_freq == 0:
            print(f"steps={learning_curve.steps}  score={learning_curve.LC_scores[-1]:.3f}")
        elif learning_curve.steps % 100 == 0:
            print(f"steps={learning_curve.steps}")



###### 儲存結果 ######
if args.save_result == True:
    learning_curve.save()


print("Finish")


