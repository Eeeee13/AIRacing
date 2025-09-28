# AIRacing
`AI realtime learn how to race`
# Idea
This repository was developed as part of an educational and research project. The solution implemented within the framework of the project is not optimal and effective.
The main goal of the project is to create a simulation of a racing car and a track, as well as train an agent to control a racing car through reinforcement learning. Below we will look at which technologies were used and how the development took place.

# Usage
For now, while project in unpredictability state I do not recommend to clone this repository. Therefore I do not provide guide

# DEV log
1) During the development process, a simulation was created, including graphics and physics, in `Python using the NumPy` library. The agent architecture was implemented using a `Q-table`. However, it is possible that the Q-table was not effective enough for this environment or was implemented sub-optimally .

2) Within this framework, a `DQN` model with a fully connected neural network was implemented. The agent showed some signs of learning, however, due to environmental constraints (inefficient machine control, inability to turn the hull, generation of a route with certain topologies), the agent could not navigate in some areas of the map.
	*In addition, a problem with movement rewards was identified at this stage, as this led to the agent spinning in place.*
3) Attempts were made to change the logic of route generation and machine movement, but it was decided to switch to `Pygames` and `Pymunk`.
	Gradually, the physics of the car's movement was implemented in a way that approximated reality: rotation is carried out only when moving, force is applied to the rear of the car, simulating rear-wheel drive, and the rotation point is shifted forward. Friction, increased lateral friction, and skidding have also been added.
4) The track restrictions were formed based on the route image. To do this, you need to transfer the color of the roadway to the function and create a mask separating the road from the bumps.
5) A system of interaction between the car and the highway was implemented through 6 beams emanating from the car and measuring the distance to the fences.
6) Rewards are configured.
7) An attempt was made to implement `PPO`. The opportunity to control the car was provided. An array of 6 distances was supplied to the input, and an array of two values (-1, 1) for steering and (-1, 1) for gas was obtained at the output.
8) Further actions were aimed at setting up hyperparameters. Initially, the award was given for lap speed, best time, and a collision penalty. By the end of the training, the agent began to behave very conservatively, moving slowly and making little progress. Then the adjustment of the remuneration system began.
9) After several unsuccessful attempts to change the logic of the reward system, adjust hyperparameters and the number of layers in the neural network, it was decided to study the PPO architecture more deeply.


