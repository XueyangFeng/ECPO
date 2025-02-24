# **ECPO**
This repository is based on our paper: Expectation Confirmation Preference Optimization for Multi-Turn Conversational Recommendation Agent.

<div  align="center">    
<img src="./pic/ecpo.png" width = "1000" height = "600" alt="pic" align=center />
</div>

## Overview
- The Code for different datasets is in `hotpotqa/`, `strategyqa/`, and `intercode/`.
  - start training by `scripts/run.sh`
  - local test environment is in `test_data/`
- Human-Agent Collaboration Dataset in `dataset/`

## Usage
### Getting Start
You can use following scripts to install related python package through pip:
```
git clone https://github.com/XueyangFeng/ECPO.git
cd ECPO
pip install -r requirements.txt
```

### AILO Environment Construction

We provide detailed AILO's pipeline code and additional [readme files](./user_simulator/readme.md). For a quick start, you can download the [index file](https://drive.google.com/file/d/1P6QkUrikHnwxNov0fUY3SxWQkl1qve0O/view?usp=drive_link) and unzip it in the ```user_simulator/embedding/``` folder. 

### API Config Settings

In this article, all LLM calls are made through OpenAI-like interfaces. Please set your API information in config/api_config.json. For closed-source models, please set it directly. For open-source models, please use vllm for local deployment, we provide a example script in ```model/```. 

### Run

For existing prompt-based CRAs, you can set the relevant config directly in main.sh and run it.

Our CRA alignment consist of 4 stages: SGPT (Stage 1), ECPO (Stage 2-4)
- Simulator-Guided Planning Tuning:
- Forward Expectation Confirmation
- Backward Expectation Derivation
- Preference Optimization


## Results
We random sample 100 questions for test for each dataset.
The evaluation result of HotpotQA dataset is under the following figure:
<div  align="center">    
<img src="./pic/main_result.png" width = "100%" alt="pic" align=center />
</div>


(a) Human-agent collaboration evaluation. (b) GPT-4-agent collaboration evaluation. The bars below the 0-axis represent the human intervention cost $\lambda C$, the entire columns, composed of the bars above and below the 0-axis, represent the task reward $T$, and the bars above the 0-axis represent the reward $R$ ($R=T - \lambda C$). Numbers within the bars means the human intervention rate. $ReHAC\_{GPT-4}$ and $ReHAC\_{Human}$ represent the policy model trained on GPT-4-agent and human-agent collaboration datasets, respectively. ReHAC outperforms other baselines in human-agent collaboration scenarios.
 

<div  align="center">    
<img src="./pic/curve.png" width = "100%" alt="pic" align=center />
</div>

We provide original evaluation outputs of ReHAC
under `results/book`, `results/game`, and `results/yelp`.

