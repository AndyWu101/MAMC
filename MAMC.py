import torch
import torch.nn.functional as F
import numpy as np


from config import args
from model import Actor , Critic
from replay_buffer import ReplayBuffer



class MAMC:

    def __init__(self, max_action):

        self.gamma = torch.tensor(args.gamma, dtype=torch.float32, device=args.device)
        self.tau = torch.tensor(args.tau, dtype=torch.float32, device=args.device)
        self.batch_size = torch.tensor(args.batch_size, dtype=torch.float32, device=args.device)

        self.exploration_noise = torch.tensor(args.exploration_noise, dtype=torch.float32, device=args.device)
        self.max_action = torch.tensor(max_action, dtype=torch.float32, device=args.device)

        self.critic_order = 0



    def train_actors(
            self,
            actors: list[Actor],
            critics: list[Critic],
            actor_optimizers: list[torch.optim.Adam],
            replay_buffer: ReplayBuffer
    ):


        indices = np.random.randint(replay_buffer.size, size=(args.actor_size, args.batch_size))

        for i in range(args.smr_ratio):

            multi_states = []
            multi_actions = []
            for j in range(args.actor_size):

                replays = replay_buffer.sample(indices=indices[j])
                states = torch.stack([replay.state for replay in replays])  # shape = (batch_size , state_dim)

                actions = actors[j](states)  # shape = (batch_size , action_dim)

                multi_states.append(states)
                multi_actions.append(actions)

            multi_states = torch.cat(multi_states, dim=0)    # shape = (actor_size * batch_size , state_dim)
            multi_actions = torch.cat(multi_actions, dim=0)  # shape = (actor_size * batch_size , action_dim)

            multi_Qs = critics[self.critic_order](multi_states, multi_actions)  # shape = (actor_size * batch_size , 1)

            actors_loss = -multi_Qs.sum() / self.batch_size

            for actor_optimizer in actor_optimizers:
                actor_optimizer.zero_grad()

            actors_loss.backward()

            for actor_optimizer in actor_optimizers:
                actor_optimizer.step()


        self.critic_order = (self.critic_order + 1) % args.critic_size



    def train_critics(
            self,
            actors: list[Actor],
            critics: list[Critic],
            critic_targets: list[Critic],
            critic_optimizers: list[torch.optim.Adam],
            replay_buffer: ReplayBuffer
    ):

        indices = np.random.randint(replay_buffer.size, size=(args.critic_size, args.batch_size))

        for i in range(args.smr_ratio):

            self.calculate_target_Q(critic_targets, actors, replay_buffer, indices)

            for j in range(args.critic_size):

                replays = replay_buffer.sample(indices=indices[j])

                states = torch.stack([replay.state for replay in replays])
                actions = torch.stack([replay.action for replay in replays])
                target_Qs = torch.stack([replay.target_Q for replay in replays]).unsqueeze_(dim=1)

                Qs = critics[j](states , actions)

                critic_loss = F.mse_loss(Qs , target_Qs)

                critic_optimizers[j].zero_grad()
                critic_loss.backward()
                critic_optimizers[j].step()

                with torch.no_grad():
                    for param, target_param in zip(critics[j].parameters(), critic_targets[j].parameters()):
                        target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)



    def calculate_target_Q(self, critic_targets: list[Critic], actors: list[Actor], replay_buffer: ReplayBuffer, indices: np.ndarray):

        unique_indices = np.unique(indices)
        replays = replay_buffer.sample(indices=unique_indices)

        with torch.no_grad():

            next_states = torch.stack([replay.next_state for replay in replays])  # shape = (replay_size , state_dim)

            multi_next_states = next_states.repeat(args.actor_size, 1)                       # shape = (actor_size * replay_size , state_dim)
            multi_next_actions = torch.cat([actor(next_states) for actor in actors], dim=0)  # shape = (actor_size * replay_size , action_dim)

            noise = torch.randn_like(multi_next_actions) * self.exploration_noise
            multi_next_actions = (multi_next_actions + noise).clamp(-self.max_action , self.max_action)

            all_next_Qs = []
            for critic_target in critic_targets:

                multi_next_Qs = critic_target(multi_next_states , multi_next_actions)  # shape = (actor_size * replay_size , 1)
                multi_next_Qs = multi_next_Qs.view(args.actor_size, -1)                # shape = (actor_size , replay_size)

                all_next_Qs.append(multi_next_Qs)

            all_next_Qs = torch.stack(all_next_Qs)                            # shape = (critic_size , actor_size , replay_size)
            multi_next_Qs = all_next_Qs.quantile(q=args.quantile_q, dim=0)    # shape = (actor_size , replay_size)
            next_Qs = multi_next_Qs.quantile(q=0.5, dim=0)                    # shape = (replay_size)

            rewards = torch.stack([replay.reward for replay in replays])      # shape = (replay_size)
            not_dones = torch.stack([replay.not_done for replay in replays])  # shape = (replay_size)

            target_Qs = rewards + not_dones * self.gamma * next_Qs            # shape = (replay_size)

            for i in range(len(replays)):
                replays[i].target_Q = target_Qs[i]


