
# Udacity Deep Reinforcement Learning Nanodegree Program
## Project 3, Collaboration and Competition

# Vidoes
## [Before Training](https://www.youtube.com/watch?v=QhVeoJaOLnY)
[![BEFORE](https://img.youtube.com/vi/QhVeoJaOLnY/0.jpg)](https://youtu.be/QhVeoJaOLnY)

## [After Training](https://www.youtube.com/watch?v=tuU1EVdII6Q)
[![AFTER](https://img.youtube.com/vi/tuU1EVdII6Q/0.jpg)](https://youtu.be/tuU1EVdII6Q)


# The Environment

In this environment, two agents control rackets to bounce a ball over a net. If an agent hits the ball over the net, it receives a reward of +0.1. If an agent lets a ball hit the ground or hits the ball out of bounds, it receives a reward of -0.01. Thus, the goal of each agent is to keep the ball in play.

The observation space consists of 8 variables corresponding to the position and velocity of the ball and racket. Each agent receives its own, local observation. Two continuous actions are available, corresponding to movement toward (or away from) the net, and jumping.

The task is episodic, and in order to solve the environment, **your agents must get an average score of +0.5 (over 100 consecutive episodes, after taking the maximum over both agents)**

# How to run the code
* Download or clone this repository to your home folder.
* Get the simulation (NoVIS)
```
wget https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Linux_NoVis.zip
unzip Tennis_Linux_NoVis.zip
```
* Build with `docker build . --rm -t udrl_docker`
* Run with `run_docker.sh`
* Inside docker - `cd ./UDACITY_DRL_P3/`
* Inside docker - `python Collaboration_Competition.py` 


* High-levels parameters (found at the first cell):
```

VIS = False  ## Show/Hide Visual simulation

TRAIN = True  ## perform training
NUM_EPISODES = 6000  ## number of epsiodes to train
MAX_T = 200 ## max time step for each episode
REDUCE_LR = False ## reduce lr while training

TEST = True ## perform testing
LOADNET = False ## load weights from file
ACTORNET_PATH = './checkpoint_actor.pth' ## filename to load weights from 
NUM_EPISODES_TEST = 100 ## number of episodes to test
MAX_T_TEST = 200  ## max time step for each episode
```


# Report
## Definitions
### State (48 = 8*3*2) and action (2,continues) space:
```
Unity brain name: TennisBrain
        Number of Visual Observations (per agent): 0
        Vector Observation space type: continuous
        Vector Observation space size (per agent): 8
        Number of stacked Vector Observation: 3
        Vector Action space type: continuous
        Vector Action space size (per agent): 2
        Vector Action descriptions: , 
```

### Solution:
The solution includes an agent trained with Deep Deterministic Policy Gradient (DDPG) approach. In order to do so, I've taken the [DDPG model and agent from my second project](https://github.com/amiravni/UDACITY_DRL_P2) and made the needed modification to transform the agent and model to this one. The changes were made using refrences from github and the [algorithm from the paper](https://arxiv.org/pdf/1509.02971.pdf)

![algorithm](./Images/DDPG.png)

The changes, in short are as followed:

#### Changes from project2 agent to project3 agent

* Init:
	* Number of agents is 2
	* Noise = 0.1
	* Batch Size = 256
* Step:
	* Updated replay buffer for 2 agents
	* Tried: learning procedure is executed several times every step (was back to single learning phase at the end)

#### Changes on main file
* The score function is calculated as followed: 'score += np.amax(reward)' 
* Peripherials changes to run the tennis simulation (and change from jupyter-notebook to py file)



### Agent:

```
BUFFER_SIZE = int(1e6)  # replay buffer size
BATCH_SIZE = 256         # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
UPDATE_EVERY = 1        # how often to update the network

NOISE_SIGMA = 0.1	# Noise sigma
LR_ACTOR = 1e-4		# **INITIAL** learning rate
LR_CRITIC = 1e-3	# **INITIAL** learning rate

```

### Network (ACTOR):
```
    def forward(self, state):
        """Build a network that maps state -> action values."""
        x = F.relu(self.fc1(state))
        x = F.dropout(x,p=0.1)
        x = F.relu(self.fc2(x))
        x = F.dropout(x,p=0.1)
        x = F.relu(self.fc3(x))
        return F.tanh(self.fc4(x))   ## added tanh
```

* fc1: Linear with 33 inputs to 100 outputs
* fc2: Linear with 100 inputs to 100 outputs
* fc3: Linear with 100 inputs to 50 outputs
* fc4: Linear with 50 inputs to 4 outputs
* Dropout with probability of 0.1 after fc1 and fc2

### Network (CRITIC):
```
    def forward(self, state,action):
        """Build a network that maps state and action -> one value"""
        x = torch.cat((state, action), dim=1)
        x = F.relu(self.fc1(x))
        x = F.dropout(x,p=0.1)
        x = F.relu(self.fc2(x))
        x = F.dropout(x,p=0.1)
        x = F.relu(self.fc3(x))
        return self.fc4(x)
```

* fc1: Linear with 37 inputs to 100 outputs
* fc2: Linear with 100 inputs to 100 outputs
* fc3: Linear with 100 inputs to 50 outputs
* fc4: Linear with 50 inputs to 1 outputs
* Dropout with probability of 0.1 after fc1 and fc2


## Learning phase:
```
n_episodes=6000
max_t=200
#actor_scheduler = torch.optim.lr_scheduler.StepLR(agents.actor_optimizer, step_size=75, gamma=0.1)
#critic_scheduler = torch.optim.lr_scheduler.StepLR(agents.critic_optimizer, step_size=75, gamma=0.1)
```

* Last project, I tried to play with the learning rate, but couldn't achieve better results. This time I chose the LR to be constant and will play it in future attemps.

### Netowrk:
The network final architecture was as described in the definitions. 

### Agent:
DDPG was implemented as described.

### Learning Phase:

* The phase consist of 6000 episodes finally. Initially I begun with 2000 episodes, which wasn't enough.
* I started with 'max_t = 1000' and got to [a very high average score](./Results/BS_256_EP_6000_TMAX_1000_NOISE_0.01/log.txt) but at the same time the learning became very slow  (Since the games took longer) and these scores where far more than needed. So I reduced max_t to 200, achieving approximally average score of ~1.03. 
* At some point, after 5600 episodes, the score begun to decrease.
* This phase with the selected parameters yield [the next output](./Results/BS_256_EP_6000_TMAX_200_NOISE_0.1/log.txt), and as shown in the graph:

![training phase](./Images/Train.png)

### Test Phase:
At this phase the network only evaluate the predicted action at a given state.
For this phase network with the weights after 5600 episodes were used. (After 3700 episodes the score was already over 0.5) 
This phase yields and average score of ~1.03 - **Meaning the agents were able to receive an average reward (over 100 episodes) of at least +0.5**. 

![test phase](./Images/Test.png)


## Ideas for future work

* Testing the dynamic learning rate approach.
* Testing multy agents approachs.
* Testing more complex network architecture
* Make more hyperparameter testing, in particular changing some parameters (GAMMA, TAU) to dynamic parameters
