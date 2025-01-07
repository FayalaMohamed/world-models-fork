## Model Based RL playground

This repository contains a collection of model based reinforcement learning algorithms implemented in PyTorch. The goal is to provide a simple and easy to understand implementation of the algorithms, so that they can be used as a reference for future projects. The algorithms are implemented in a modular way, so that they can be easily extended and modified.
<br><br>
The repository is still under development, so some features may not be implemented yet. If you find any bugs or have any suggestions, please let me know.
<br><br>
The repository is mainly for educational purposes and does not provide an extension of the research in the respective paper.
### Papers
For now, I am implementing the following papers:
- [Learning Latent Dynamics for Planning from Pixels](https://arxiv.org/abs/1811.04551)
- [World Models](https://arxiv.org/abs/1803.10122)
- [Dream to Control: Learning Behaviors by Latent Imagination](https://arxiv.org/abs/1912.01603)

### Structure
The repository is structured as follows:
- `models/`: Contains the neural network models used in the algorithms.
- `envs/`: Contains the environment wrappers used in the algorithms.
- `train`: Contains the training scripts for the algorithms.

### Blog post
Corresponding blog post where all the math is derived can be found here:
https://medium.com/@lukasbierling/training-agents-to-plan-in-latent-space-a-technical-overview-f4380a94ec88

### Example algorithm of MBRL in the dreamer paper
![img.png](img.jpg) source: https://arxiv.org/pdf/1912.01603
