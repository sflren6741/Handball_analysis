# [Score prediction using multiple object tracking for analyzing movements in 2-vs-2 Handball]

Ren Kobayashi, Rikuhei Umemoto, Kazuya Takeda, Keisuke Fujii, Score prediction using multiple object tracking for analyzing movements in 2-vs-2 Handball, IEEE 12th Global Conference on Consumer Electronics (GCCE 2023), 2023.10.10. 

## Introduction
You can identify the important features in handball 2-vs-2 by using this code. 

If you have any questions or errors, please contact the author.

## Sample Video
![2on2_1_left_0 mp4_animation](https://github.com/sflren6741/Handball_analysis/assets/103619748/d4b20954-99f8-4724-8f4b-4619427ea1b9)

## Sample result
This is an example of feature importance obtained by analysis.
![selected_features](https://github.com/sflren6741/Handball_analysis/assets/103619748/b8dc71ca-5d7c-4037-8f32-1f05b60692c7)
## Author
Ren Kobayashi - kobayashi.ren@g.sp.m.is.nagoya-u.ac.jp

## Environment
Python 3.8.10

## Evaluation from scratch
### Step 1: Downloading the required data
Please download the data that you need from [Google Drive](https://drive.google.com/drive/folders/1-7ZCkElkJSG0fVM_Edj_IsQSM-3bxzqe).
- `Handball_2on2_2022_NU.xlsx`: The annotation data.
- `positions_data.json`: The coordinates of the players data.
- `videos`: There are raw videos we captured. If you want to get the coordinates of the players from videos by yourself, please use them.
### Step 2:  Running the code and check the result
1. `python3 analyze_handball_2-vs-2.py`
2. Please see `fig` to check the results.
