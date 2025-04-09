<div align="center">
  <!-- <h1><b> Time-LLM </b></h1> -->
  <!-- <h2><b> Time-LLM </b></h2> -->
  <h2><b> (DASFAA2025) Enhancing Time Series Forecasting via Multi-Level Text Alignment with LLMs </b></h2>
</div>



</div>

<p align="center">

<img src="./figures/overview.png">

</p>

---
>
> 🙋 Please let us know if you find out a mistake or have any suggestions!
> 
> 🌟 If you find this resource helpful, please consider to star this repository and cite our research:



## Introduction
MLTA is a multi-level reprogramming framework to repurpose LLMs for general time series forecasting with the backbone language models kept intact.
Notably, we show that time series analysis (e.g., forecasting) can be cast as yet another "language task" that can be effectively tackled by an off-the-shelf LLM.

<p align="center">
<img src="./figures/framework.png" height = "360" alt="" align=center />
</p>

- MLTA comprises two key components: (1) We propose an interpretable multi-level text alignment framework for time series forecasting using LLMs while keeping the backbone model unchanged. (2) Our method leverages this multi-level alignment to map decomposed time series components—trend, seasonality, and residuals—into distinctive, informative joint representations.

<p align="center">
<img src="./figures/method-detailed-illustration.png" height = "190" alt="" align=center />
</p>

## Requirements
Use python 3.9

- torch==2.3.0
- accelerate==0.28.0
- einops==0.7.0
- matplotlib==3.7.0
- numpy==1.26.4
- pandas==1.5.3
- scikit_learn==1.2.2
- scipy==1.12.0
- tqdm==4.65.0
- peft==0.4.0
- transformers==4.31.0
- deepspeed==0.14.0
- sentencepiece==0.2.0
- omegaconf==2.3.0
- seaborn==0.13.2
- statsmodels==0.14.2

To install all dependencies:
```
pip install -r requirements.txt
```

## Datasets
You can access the well pre-processed datasets from [[Google Drive]](https://drive.google.com/file/d/1NF7VEefXCmXuWNbnNe858WvQAkJ_7wuP/view?usp=sharing), then place the downloaded contents under `./dataset`

## Quick Demos
1. Download datasets and place them under `./dataset`
2. Tune the model. We provide one experiment script for demonstration purpose under the folder `./scripts`. For example, you can evaluate on ETT datasets by:

```bash
bash ./scripts/MLTA_ETTh1.sh 
```


## Detailed usage

Please refer to ```run_main.py```, ```run_m4.py``` and ```run_pretrain.py``` for the detailed description of each hyperparameter.


## Acknowledgement
Our implementation adapts [Time-Series-Library](https://github.com/thuml/Time-Series-Library) and [TimeLLM](https://github.com/KimMeen/Time-LLM) as the code base and have extensively modified it to our purposes. We thank the authors for sharing their implementations and related resources.
