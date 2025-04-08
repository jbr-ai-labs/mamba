# MAMBA
This code accompanies the paper "[Scalable Multi-Agent Model-Based Reinforcement Learning](https://arxiv.org/abs/2205.15023)".

The repository contains MAMBA implementation as well as fine-tuned hyperparameters in ```configs/dreamer/optimal``` folder.

## Installation

`python3.7` is required

```
pip install wheel
pip install flatland-2.2.2/
pip install -r requirements.txt 
```

Installing Starcraft:

https://github.com/oxwhirl/smac#installing-starcraft-ii


## Usage

```
python3 train.py --n_workers 2 --env flatland --env_name 5_agents
```

Two environments are supported for env flag: flatland and starcraft.

### Optimal parameters
To train agents with optimal parameters from the paper they should be copied from `configs/dreamer/optimal/` folder to [DreamerAgentConfig.py](https://github.com/jbr-ai-labs/mamba/blob/main/configs/dreamer/DreamerAgentConfig.py) and [DreamerLearnerConfig.py](https://github.com/jbr-ai-labs/mamba/blob/main/configs/dreamer/DreamerLearnerConfig.py)

## SMAC

<img height="300" alt="starcraft" src="https://user-images.githubusercontent.com/22059171/152656435-1634c15b-ca6d-4b23-9383-72fe3759b9e3.png">

The code for the environment can be found at 
[https://github.com/oxwhirl/smac](https://github.com/oxwhirl/smac)

## Flatland

<img height="300" alt="flatland" src="https://user-images.githubusercontent.com/22059171/152656405-b4ab7e6c-d691-4300-a419-a3d4288513e8.png">

The original code for the environment can be found at 
[https://github.com/jbr-ai-labs/NeurIPS2020-Flatland-Competition-Solution](https://github.com/jbr-ai-labs/NeurIPS2020-Flatland-Competition-Solution)

## Code Structure

- ```agent``` contains implementation of MAMBA 
  - ```controllers``` contains logic for inference
  - ```learners``` contains logic for learning the agent
  - ```memory``` contains buffer implementation
  - ```models``` contains architecture of MAMBA
  - ```optim``` contains logic for optimizing loss functions
  - ```runners``` contains logic for running multiple workers
  - ```utils``` contains helper functions
  - ```workers``` contains logic for interacting with environment
- ```env``` contains environment logic
- ```networks``` contains neural network architectures


## Citation

    @inproceedings{10.5555/3535850.3535894,
          author = {Egorov, Vladimir and Shpilman, Alexei},
          title = {Scalable Multi-Agent Model-Based Reinforcement Learning},
          year = {2022},
          isbn = {9781450392136},
          publisher = {International Foundation for Autonomous Agents and Multiagent Systems},
          address = {Richland, SC},
          booktitle = {Proceedings of the 21st International Conference on Autonomous Agents and Multiagent Systems},
          pages = {381–390},
          numpages = {10},
          keywords = {communication, multi-agent reinforcement learning, model-based reinforcement learning},
          location = {Virtual Event, New Zealand},
          series = {AAMAS '22}
    }
