# **ECPO**
This repository is based on our paper: *Expectation Confirmation Preference Optimization for Multi-Turn Conversational Recommendation Agent*.

<div  align="center">    
<img src="./pic/ecpo.png" width = "1000" height = "615" alt="pic" align=center />
</div>


## Usage
### Getting Start
You can use following scripts to install related python package through pip:
```
git clone https://github.com/XueyangFeng/ECPO.git
cd ECPO
pip install -r requirements.txt
```


### **AILO Environment Setup**

We provide a detailed pipeline for the AILO environment, including additional [README files](./user_simulator/readme.md). For a quick setup, follow these steps:

1. Download the [index file](https://drive.google.com/file/d/1P6QkUrikHnwxNov0fUY3SxWQkl1qve0O/view?usp=drive_link).
2. Unzip the downloaded file into the `user_simulator/embedding/` folder.

### **API Configuration**
All LLM (Large Language Model) calls in this repository are made using OpenAI-like interfaces. To configure the APIs:

1. Set your API information in the `config/api_config.json` file.
2. For closed-source models, set the information directly in the config.
3. For open-source models, use `vllm` for local deployment. We have provided an example script in the `model/` directory.


### **Running ECPO**

To run the existing prompt-based Conversational Recommendation Agent (CRA) or an aligned CRA, you can set the relevant configuration in the `main.sh` file and execute it.

Our CRA alignment process consists of four main stages:
1. **SGPT (Stage 1)**: Simulator-Guided Planning Tuning
2. **ECPO (Stages 2-4)**: Expectation Confirmation Preference Optimization

### **Stages Overview:**
- **SGPT (Stage 1)**: [Simulator-Guided Planning Tuning](backward/Book/sft)
- **ECPO Stages (2-4)**:
  - [Forward Expectation Confirmation](forward/)
  - [Backward Expectation Derivation](backward/Book/ecpo)
  - [Preference Optimization](LLaMA-Factory/ecpo)

You can download the [training data](https://drive.google.com/file/d/16GfbEscRzd_OZOoR6vRSysPISnIJ0O2a/view?usp=sharing) and unzip it into the `backward/` directory.

### Evaluation

First, test recommendation metric using simulator environment:
```
# test the existing prompt-based CRA baseline
bash main.sh
# test the aligned CRA
bash main_lora.sh
```

Then, test dialogue metric using gpt-4o evaluator:
```
cd pair_eval
#Set up your evaluation files (`model2.log` for the targeted log file, and `model1.log` for the expert trajectory), then run:
python eval.py
```


## Results

### **Comparison with Existing Prompt-Based CRAs**
<div align="center">    
  <img src="./pic/exp1.png" width="100%" alt="Comparison with Prompt-Based CRAs" />
</div>

### **Comparison of Aligned CRAs Fine-Tuned with Different Methods in Terms of Interactivity**
<div align="center">    
  <img src="./pic/exp2.png" width="100%" alt="Comparison of Aligned CRAs" />
</div>

We also provide a series of [test raw log](test_log) data for reference.

## References
1. Our evaluation method is based on [RUCAIBox/iEvaLM-CRS](https://github.com/RUCAIBox/iEvaLM-CRS).
2. Our training code is based on [hiyouga/LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory).


