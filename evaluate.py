import torch


from config import args
from model import Actor , Critic
from replay_buffer import ReplayBuffer



def evaluate_actors(actors: list[Actor], critics: list[Critic], replay_buffer: ReplayBuffer):

    with torch.no_grad():

        replays = replay_buffer.sample()
        states = torch.stack([replay.state for replay in replays])

        multi_actions = []
        for actor in actors:

            action = actor(states)

            multi_actions.append(action)

        multi_states = states.repeat(args.actor_size, 1)  # shape = (actor_size * batch_size , state_dim)
        multi_actions = torch.cat(multi_actions, dim=0)   # shape = (actor_size * batch_size , action_dim)


        evaluation_table = []
        for critic in critics:

            multi_Qs = critic(multi_states , multi_actions)  # shape = (actor_size * batch_size , 1)
            multi_Qs = multi_Qs.view(args.actor_size , -1)   # shape = (actor_size , batch_size)

            evaluation_table.append(multi_Qs)

        evaluation_table = torch.stack(evaluation_table)  # shape = (critic_size , actor_size , batch_size)

        evaluation_table_quantile = evaluation_table.quantile(q=args.quantile_q, dim=0)  # shape = (actor_size , batch_size)

        actors_skill = evaluation_table_quantile.mean(dim=1)  # shape = (actor_size)
        actors_creativity = (evaluation_table - evaluation_table_quantile).abs().mean(dim=0).mean(dim=1)  # shape = (actor_size)

        for i in range(args.actor_size):
            actors[i].skill = actors_skill[i]
            actors[i].creativity = actors_creativity[i]


