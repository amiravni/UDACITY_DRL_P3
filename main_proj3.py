from unityagents import UnityEnvironment
import numpy as np
import time
from ddpg_agent import Agents
from collections import deque
import torch
import matplotlib.pyplot as plt
import progressbar


VIS = False

TRAIN = True
NUM_EPISODES = 6000
MAX_T = 200

TEST = True
LOADNET = True
ACTORNET_PATH = './checkpoint_actor.pth'
NUM_EPISODES_TEST = 100
MAX_T_TEST = 200


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


if not VIS:
    path_prefix = '_NoVis'
else:
    path_prefix =''

env = UnityEnvironment(file_name='Tennis_Linux'+path_prefix+'/Tennis.x86_64')

# get the default brain
brain_name = env.brain_names[0]
brain = env.brains[brain_name]

# reset the environment
env_info = env.reset(train_mode=True)[brain_name]

# number of agents
num_agents = len(env_info.agents)
print('Number of agents:', num_agents)

# size of each action
action_size = brain.vector_action_space_size
print('Size of each action:', action_size)

# examine the state space
states = env_info.vector_observations
state_size = states.shape[1]
print('There are {} agents. Each observes a state with length: {}'.format(states.shape[0], state_size))
print('The state for the first agent looks like:', states[0])
print('The state for the second agent looks like:', states[1])

agents = Agents(state_size=state_size, action_size=action_size,
               num_agents = num_agents, seed = 0)  ## choosing num of agent 1 for compitibility
if TRAIN:
    print('training...')
    scores = []  # list containing scores from each episode
    scores_window = deque(maxlen=100)  # last 100 scores
    scores_window_25 = deque(maxlen=25)  # last 25 scores
    last_mean_score = 0.001
    #actor_scheduler = torch.optim.lr_scheduler.StepLR(agents.actor_optimizer, step_size=75, gamma=0.25)
    #critic_scheduler = torch.optim.lr_scheduler.StepLR(agents.critic_optimizer, step_size=75, gamma=0.25)
    #last_actor_lr,last_critic_lr = 0,0

    for i_episode in progressbar.progressbar(range(1,NUM_EPISODES+1)):
            env_info = env.reset(train_mode=True)[brain_name]
            #agent.reset()
            states = env_info.vector_observations
            score = np.zeros(num_agents)
            for t in range(MAX_T):
                action = agents.act(states)
                env_info = env.step(action)[brain_name]
                next_states = env_info.vector_observations
                reward = env_info.rewards
                done = env_info.local_done
                agents.step(states, action, reward, next_states, done)
                score += np.amax(reward) ### ?
                states = next_states


                if np.any(done):
                    break
            '''
            actor_scheduler.step()
            actor_lr = get_lr(agents.actor_optimizer)
            if actor_lr != last_actor_lr:
                print('\nChanging actor lr from ', last_actor_lr, ' to: ', actor_lr)
                last_actor_lr = actor_lr
            critic_scheduler.step()
            critic_lr = get_lr(agents.critic_optimizer)
            if critic_lr != last_critic_lr:
                print('\nChanging critic lr from ', last_critic_lr, ' to: ', critic_lr)
                last_critic_lr = critic_lr
            '''

            scores_window.append(score)  # save most recent score
            scores_window_25.append(score)  # save most recent score
            scores.append(np.mean(score))  # save most recent score
            if i_episode % 25 == 0:
                print('\rEpisode {}\tAverage Score (25): {:.2f}'.format(i_episode, np.mean(scores_window_25)))
            if i_episode % 100 == 0:
                print('\rEpisode {}\tAverage Score (100): {:.2f}'.format(i_episode, np.mean(scores_window)))
                if np.mean(scores_window) >= last_mean_score:
                    last_mean_score = np.mean(scores_window)
                    # print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode-100, np.mean(scores_window)))
                    print('Saving...')
                    torch.save(agents.actor_local.state_dict(), 'checkpoint_actor.pth')
                    torch.save(agents.critic_local.state_dict(), 'checkpoint_critic.pth')
                    # break
    torch.save(agents.actor_local.state_dict(), 'last_checkpoint_actor.pth')
    torch.save(agents.critic_local.state_dict(), 'last_checkpoint_critic.pth')
    # plot the scores
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(np.arange(len(scores)), scores)
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    #plt.show()
    plt.savefig('trainRes.png')


if TEST:
    print('testing...')
    if LOADNET:
        print('loading net...')
        agents.actor_local.load_state_dict(torch.load(ACTORNET_PATH))
    scores = []  # list containing scores from each episode
    scores_window = deque(maxlen=NUM_EPISODES_TEST)  # last 100 scores

    for i_episode in progressbar.progressbar(range(1,NUM_EPISODES_TEST+1)):
        env_info = env.reset(train_mode=False)[brain_name]
        states = env_info.vector_observations
        score = np.zeros(num_agents)
        for t in range(MAX_T_TEST):
            actions = agents.act(states)
            env_info = env.step(actions)[brain_name]
            next_states = env_info.vector_observations
            rewards = env_info.rewards
            dones = env_info.local_done
            #score += rewards
            score += np.amax(rewards)
            states = next_states
            time.sleep(0.01)
            if np.any(dones):  # exit loop if episode finished
                break
        scores_window.append(score)  # save most recent score
        scores.append(np.mean(score))  # save most recent score
    # plot the scores
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(np.arange(len(scores)), scores)
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    #plt.show()
    plt.savefig('testRes.png')
    print('Final Score: ==> ' + np.mean(scores))

env.close()

''' # random 
for i in range(5):                                         # play game for 5 episodes
    env_info = env.reset(train_mode=False)[brain_name]     # reset the environment
    states = env_info.vector_observations                  # get the current state (for each agent)
    scores = np.zeros(num_agents)                          # initialize the score (for each agent)
    while True:
        actions = np.random.randn(num_agents, action_size) # select an action (for each agent)
        actions = np.clip(actions, -1, 1)                  # all actions between -1 and 1
        env_info = env.step(actions)[brain_name]           # send all actions to tne environment
        next_states = env_info.vector_observations         # get next state (for each agent)
        rewards = env_info.rewards                         # get reward (for each agent)
        dones = env_info.local_done                        # see if episode finished
        scores += env_info.rewards                         # update the score (for each agent)
        states = next_states                               # roll over states to next time step
        if np.any(dones):                                  # exit loop if episode finished
            break
    print('Total score (averaged over agents) this episode: {}'.format(np.mean(scores)))

env.close()
'''