# MAMBA
This code accompanies the paper "Scalable Multi-Agent Model-Based Reinforcement Learning".

The repository contains MAMBA implementation as well as fine-tuned hyperparameters in ```configs/dreamer/optimal``` folder.

## Usage

```
python3 train.py --n_workers 2 --env flatland --env_type 5_agents
```

Two environments are supported for env flag: flatland and starcraft.


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
