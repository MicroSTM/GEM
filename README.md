# Discovering Generalizable Spatial Goal Representations via Graph-based Active Reward Learning

Code for the ICML 2022 paper: [*Discovering Generalizable Spatial Goal Representations via Graph-based Active Reward Learning*](https://www.tshu.io/GEM/GEM.pdf).  

The code was written by the lead authors of the paper, Aviv Netanyahu and Tianmin Shu. For more details of the dataset, please visit our [*project website*](https://www.tshu.io/GEM).

The implementation is built on top of the framework in [*HumanCompatibleAI/imitation*](http://github.com/HumanCompatibleAI/imitation). 

## Installation

Create a conda environment using `environment.yml`, activate that environment, and run installation as follows:

```
conda env create -f environment.yml
source activate GEM
cd imitation
pip install -e .
```

 
## Instruction

First, get into the `imitation/src/imitation/experiments` folder:
```
cd imitation/src/imitation/experiments
```

Then get into the generate task definitions for creating gym environments for Watch&Move tasks:
```
bash generate_tasks.sh
```

Download the expert demos from [*here*](https://www.tshu.io/expert_demos) and unzip in the `imitation/src/imitation/output` folder.

To run training and evaluation for each task, you may use the bash scripts in the `imitation/src/experiment` folder. For example, to run training for task 5, you may run the following commands.
```
cd imitation/src/imitation/experiment
bash task5.sh
```

The results will be saved in tbe `imitation/src/imitation/output/GEM` folder.

## Cite
If you use this code in your research, please cite the following papers.

```
@inproceedings{netanyahu2022discoverying,
  title={Discovering Generalizable Spatial Goal Representations via Graph-based Active Reward Learning},
  author={Netanyahu, Aviv and Shu, Tianmin and Tenenbaum, Joshua B and Agrawal, Pulkit},
  booktitle={39th International Conference on Machine Learning (ICML)},
  year={2022}
}
```
```
@misc{wang2020imitation,
  author = {Wang, Steven and Toyer, Sam and Gleave, Adam and Emmons, Scott},
  title = {The {\tt imitation} Library for Imitation Learning and Inverse Reinforcement Learning},
  year = {2020},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/HumanCompatibleAI/imitation}},
}
```

