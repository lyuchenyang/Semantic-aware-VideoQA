<div align="center">

# Semantic-aware Dynamic Retrospective-Prospective Reasoning for Event-level Video Question Answering &#x1F4BB;


**[Chenyang Lyu](https://lyuchenyang.github.io), [Tianbo Ji](mailto:jitianbo@ntu.edu.cn), [Yvette Graham](mailto:ygraham@tcd.ie), [Jennifer Foster](mailto:jennifer.foster@dcu.ie)**

School of Computing, Dublin City University, Dublin, Ireland &#x1F3E0;

This repository contains the code for the Semantic-aware Dynamic Retrospective-Prospective Reasoning system for Event-level Video Question Answering (EVQA) &#x1F4BB;. The system utilizes explicit semantic connections between questions and visual information at the event level to improve the reasoning process and provide optimal answers &#x1F50D;.
</div>

## Table of Contents

- [1. Introduction](#1-introduction) &#x1F4D8;
- [2. Dataset](#2-dataset) &#x1F4D3;
- [3. Pre-processing](#3-pre-processing) &#x1F527;
- [4. Training](#4-training) &#x1F3EB;
- [5. Usage](#5-usage) &#x1F4E6;
- [6. Dependencies](#6-dependencies) &#x1F6E0;

## 1. Introduction &#x1F4D8;

Event-Level Video Question Answering (EVQA) requires complex reasoning across video events to obtain the visual information needed for optimal answers. However, few studies have focused on utilizing explicit semantic connections between questions and visual information, especially at the event level. In this paper, we propose a semantic-aware dynamic retrospective-prospective reasoning approach for video-based question answering. We explicitly incorporate the Semantic Role Labeling (SRL) structure of the question in the dynamic reasoning process, determining which frame to move to based on the focused part of the SRL structure (agent, verb, patient, etc.) &#x1F9D0;. We evaluate our approach on the TrafficQA benchmark EVQA dataset and demonstrate superior performance compared to previous state-of-the-art models &#x1F4AA;.

## 2. Dataset &#x1F4D3;

Please download the TrafficQA dataset from this link: [https://sutdcv.github.io/SUTD-TrafficQA/#/download](https://sutdcv.github.io/SUTD-TrafficQA/#/download) including videos and corresponding annotations and then move them under `data/` directory.

## 3. Pre-processing &#x1F527;

Please use `data_preprocess.py` to extract frames from videos in the TrafficQA dataset and then tokenize the annotations data to tensor dataset.

## 4. Training &#x1F3EB;

Please use the following script to train and evaluate the VideoQA system: 
```bash
python run_traffic_qa.py --do_train --do_eval --num_train_epochs 2 --n_frames 10 --eval_n_frames 10 --learning_rate 5e-6 --train_batch_size 8 --eval_batch_size 16 --attention_heads 8 --eval_steps 5000
```

## 5. Usage &#x1F4E6;

Once the model is trained, you can use it for VideoQA tasks. Provide a video, and the system will give the most probable answer based on the video. 🔎

## 6. Dependencies 🛠️

- Python (>=3.8) 🐍
- Pytorch (>=2.0) 🔥
- MoviePy 🧮
- ffmpeg 🐼

Please make sure to install the required dependencies before running the code. ⚙️


## Citation 📄

Please cite our paper using the bibtex below if you found that our paper is useful to you:

```bibtex
@article{lyu2023semantic,
  title={Semantic-aware Dynamic Retrospective-Prospective Reasoning for Event-level Video Question Answering},
  author={Lyu, Chenyang and Ji, Tianbo and Graham, Yvette and Foster, Jennifer},
  journal={arXiv preprint arXiv:2305.08059},
  year={2023}
}
```
