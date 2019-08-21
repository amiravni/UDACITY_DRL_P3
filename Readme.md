
# Udacity Deep Reinforcement Learning Nanodegree Program
## Project 3, Collaboration and Competition

# Vidoes
## [Before Training](https://www.youtube.com/watch?v=QhVeoJaOLnY)
[![BEFORE](https://img.youtube.com/vi/QhVeoJaOLnY/0.jpg)](https://youtu.be/QhVeoJaOLnY)

## [After Training](https://www.youtube.com/watch?v=0b5em4mF24Y)
[![AFTER](https://img.youtube.com/vi/0b5em4mF24Y/0.jpg)](https://youtu.be/0b5em4mF24Y)


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


# [Report](./Report.md)
