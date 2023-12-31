# -*- coding: UTF-8 -*-  
import glob
import os
import sys
import pickle
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patches as patches
from scipy import signal
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn import model_selection
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

plt.rcParams["font.size"] = 18


def get_animation(filtered_positions_data):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    r1 = patches.Rectangle(xy=(6.0, 7.0), width=10.0, height=6.0, ec='#000000', fill=False)
    a1 = patches.Arc(xy=(0.0, 8.5), width=12.0, height=12.0, angle=0.0, theta1=270.0, theta2=360.0)
    a2 = patches.Arc(xy=(0.0, 11.5), width=12.0, height=12.0, angle=0.0, theta1=0.0, theta2=90.0)

    for externalID, filtered_position_data_ in filtered_positions_data.items(): # 動画ごとに処理
        ax.cla()
        ax.set_aspect('equal')
        print('export: '+ externalID)
        fig_title = f"{externalID}_animation"
        ax.set_title(fig_title)
        ax.set_xlim(0.0,20.0)
        ax.set_ylim(0.0,20.0)
        ax.add_patch(r1)
        ax.add_patch(a1)
        ax.add_patch(a2)
        ax.plot([6.0, 6.0],[8.5, 11.5],color="black")

        ims = []
        texts = []
        frame_text = []
        ball_carrier = '0_0'
        for n in range(filtered_position_data_["len_frame"]): # 1フレームごとに処理
            im = []
            text = []
            frame_text = ax.text(15.0,18.0,'frame = '+str(n),color='blue')
            for objectID, filtered_position_data in filtered_position_data_.items(): # 選手ごとに処理
                #print(filtered_position_data)
                if objectID != "len_frame":
                    if objectID == "0_0" or objectID == "0_1":
                        color = "black"
                    else:
                        color = "red"

                    im.append(ax.plot(filtered_position_data[n,0],filtered_position_data[n,1],marker='o',color=color))
                    text.append(ax.text(filtered_position_data[n,0],filtered_position_data[n,1],objectID,color='blue'))
            tmp = im[0] + [text[0]]
            for i in range(1, len(im)):
                tmp += im[i] + [text[i]]
            ims.append(tmp + [frame_text])
        anime = animation.ArtistAnimation(fig, ims, interval=100)
        anime.save(f"./ani_position/{fig_title}.gif", writer="pillow")

def get_analysis_data(positions_data):
    input_file_name = 'Handball_2on2_2022_NU.xlsx'
    frame_error = 0
    analysis_data_ = dict()
    for externalID, position_data_ in positions_data.items(): # 動画ごとに処理
        analysis_data = dict()
        print('export: '+ externalID)
        # print(filtered_position_data_)
        video_title = os.path.splitext(os.path.basename(externalID))[0]
        tmp = video_title.split('_')
        sheet_name = tmp[0] + '_' + tmp[1] + '_' + tmp[2]
        attack_num = int(tmp[3])
        #sheet_name = os.path.splitext(os.path.basename(externalID))[0]
        data_frame = pd.read_excel(input_file_name, engine='openpyxl', sheet_name=sheet_name)
        #print(data_frame)
        # 範囲外に出たものを分析の対象外とする
        if data_frame['other results'][attack_num] == 'out of area':
            continue
        analysis_data['result'] = data_frame["result (attackers' win:1)"][attack_num]
        analysis_data['block_success'] = data_frame['block success'][attack_num]
        analysis_data['shot_type'] = data_frame['shot type'][attack_num]
        analysis_data['OF1_dominant_hand']= data_frame["OF1's dominant hand(left:0, right:1)"][attack_num]
        analysis_data['other_results'] = data_frame['other results'][attack_num]

        action = []
        action_t = []
        action_p = []
        for i in range(1, 5):
            action.append(data_frame[f'action{i}'])
            action_t.append(data_frame[f'action{i}_t'])
            action_p.append(data_frame[f'action{i}_p'])
        if analysis_data['OF1_dominant_hand'] == 1:
            # print('OF1_dominant_hand==1')
            analysis_data['OF1_velocity_x_catch'] = (position_data_['0_0'][3, :][0] - position_data_['0_0'][0, :][0])
            analysis_data['OF1_velocity_y_catch'] = (position_data_['0_0'][3, :][1] - position_data_['0_0'][0, :][1])
            analysis_data['OF2_velocity_x_catch'] = (position_data_['0_1'][3, :][0] - position_data_['0_1'][0, :][0])
            analysis_data['OF2_velocity_y_catch'] = (position_data_['0_1'][3, :][1] - position_data_['0_1'][0, :][1])
            analysis_data['DF1_velocity_x_catch'] = (position_data_['1_2'][3, :][0] - position_data_['1_2'][0, :][0])
            analysis_data['DF1_velocity_y_catch'] = (position_data_['1_2'][3, :][1] - position_data_['1_2'][0, :][1])
            analysis_data['DF2_velocity_x_catch'] = (position_data_['1_3'][3, :][0] - position_data_['1_3'][0, :][0])
            analysis_data['DF2_velocity_y_catch'] = (position_data_['1_3'][3, :][1] - position_data_['1_3'][0, :][1])
            analysis_data['OF1_position_x_catch'] = position_data_['0_0'][0, :][0]
            analysis_data['OF1_position_y_catch'] = position_data_['0_0'][0, :][1]
            analysis_data['OF2_position_x_catch'] = position_data_['0_1'][0, :][0]
            analysis_data['OF2_position_y_catch'] = position_data_['0_1'][0, :][1]
            analysis_data['DF1_position_x_catch'] = position_data_['1_2'][0, :][0]
            analysis_data['DF1_position_y_catch'] = position_data_['1_2'][0, :][1]
            analysis_data['DF2_position_x_catch'] = position_data_['1_3'][0, :][0]
            analysis_data['DF2_position_y_catch'] = position_data_['1_3'][0, :][1]
            analysis_data['OF1_OF2_distance_catch'] = np.sqrt(np.sum(np.square(position_data_['0_0'][0]-position_data_['0_1'][0])))
            analysis_data['OF1_DF1_distance_catch'] = np.sqrt(np.sum(np.square(position_data_['0_0'][0]-position_data_['1_2'][0])))
            analysis_data['OF1_DF2_distance_catch'] = np.sqrt(np.sum(np.square(position_data_['0_0'][0]-position_data_['1_3'][0])))
            analysis_data['OF2_DF1_distance_catch'] = np.sqrt(np.sum(np.square(position_data_['0_1'][0]-position_data_['1_2'][0])))
            analysis_data['OF2_DF2_distance_catch'] = np.sqrt(np.sum(np.square(position_data_['0_1'][0]-position_data_['1_3'][0])))
            analysis_data['DF1_DF2_distance_catch'] = np.sqrt(np.sum(np.square(position_data_['1_2'][0]-position_data_['1_3'][0])))
        else:
            analysis_data['OF1_velocity_x_catch'] = (position_data_['0_0'][3, :][0] - position_data_['0_0'][0, :][0])
            analysis_data['OF1_velocity_y_catch'] = -((position_data_['0_0'][3, :][1] - position_data_['0_0'][0, :][1]))
            analysis_data['OF2_velocity_x_catch'] = (position_data_['0_1'][3, :][0] - position_data_['0_1'][0, :][0])
            analysis_data['OF2_velocity_y_catch'] = -((position_data_['0_1'][3, :][1] - position_data_['0_1'][0, :][1]))
            analysis_data['DF1_velocity_x_catch'] = (position_data_['1_2'][3, :][0] - position_data_['1_2'][0, :][0])
            analysis_data['DF1_velocity_y_catch'] = -((position_data_['1_2'][3, :][1] - position_data_['1_2'][0, :][1]))
            analysis_data['DF2_velocity_x_catch'] = (position_data_['1_3'][3, :][0] - position_data_['1_3'][0, :][0])
            analysis_data['DF2_velocity_y_catch'] = -((position_data_['1_3'][3, :][1] - position_data_['1_3'][0, :][1]))
            analysis_data['OF1_position_x_catch'] = position_data_['0_0'][0, :][0]
            analysis_data['OF1_position_y_catch'] = -(position_data_['0_0'][0, :][1])
            analysis_data['OF2_position_x_catch'] = position_data_['0_1'][0, :][0]
            analysis_data['OF2_position_y_catch'] = -(position_data_['0_1'][0, :][1])
            analysis_data['DF1_position_x_catch'] = position_data_['1_2'][0, :][0]
            analysis_data['DF1_position_y_catch'] = -(position_data_['1_2'][0, :][1])
            analysis_data['DF2_position_x_catch'] = position_data_['1_3'][0, :][0]
            analysis_data['DF2_position_y_catch'] = -(position_data_['1_3'][0, :][1])
            analysis_data['OF1_OF2_distance_catch'] = np.sqrt(np.sum(np.square(position_data_['0_0'][0]-position_data_['0_1'][0])))
            analysis_data['OF1_DF1_distance_catch'] = np.sqrt(np.sum(np.square(position_data_['0_0'][0]-position_data_['1_3'][0])))
            analysis_data['OF1_DF2_distance_catch'] = np.sqrt(np.sum(np.square(position_data_['0_0'][0]-position_data_['1_2'][0])))
            analysis_data['OF2_DF1_distance_catch'] = np.sqrt(np.sum(np.square(position_data_['0_1'][0]-position_data_['1_3'][0])))
            analysis_data['OF2_DF2_distance_catch'] = np.sqrt(np.sum(np.square(position_data_['0_1'][0]-position_data_['1_2'][0])))
            analysis_data['DF1_DF2_distance_catch'] = np.sqrt(np.sum(np.square(position_data_['1_2'][0]-position_data_['1_3'][0])))

        shot = False
        for i in range(4):
            if action[i][attack_num] == 'shot':
                shot = True
                #print(f'attack_num:{attack_num}')
                #print(f'action[{i}]:')
                #print(action[i])
                analysis_data['shot_t'] = action_t[i][attack_num] - action_t[0][attack_num]
                if action_p[i][attack_num] == 0:
                    analysis_data['shot_person'] = '0_0'
                else:
                    analysis_data['shot_person'] = '0_1'
                #print(f'shot_t:{analysis_data["shot_t"]}')
                #print(f'shot_person:{analysis_data["shot_person"]}')
                shot_frame = int(analysis_data['shot_t'] * 30.0 / 100.0)
                if position_data_['len_frame'] <= shot_frame:
                    shot_frame = position_data_['len_frame'] - 1
                    frame_error += 1
                if analysis_data['OF1_dominant_hand'] == 1:
                    analysis_data['OF1_velocity_x_shot'] = (position_data_['0_0'][shot_frame, :][0] - position_data_['0_0'][shot_frame-3, :][0])
                    analysis_data['OF1_velocity_y_shot'] = (position_data_['0_0'][shot_frame, :][1] - position_data_['0_0'][shot_frame-3, :][1])
                    analysis_data['OF2_velocity_x_shot'] = (position_data_['0_1'][shot_frame, :][0] - position_data_['0_1'][shot_frame-3, :][0])
                    analysis_data['OF2_velocity_y_shot'] = (position_data_['0_1'][shot_frame, :][1] - position_data_['0_1'][shot_frame-3, :][1])
                    analysis_data['DF1_velocity_x_shot'] = (position_data_['1_2'][shot_frame, :][0] - position_data_['1_2'][shot_frame-3, :][0])
                    analysis_data['DF1_velocity_y_shot'] = (position_data_['1_2'][shot_frame, :][1] - position_data_['1_2'][shot_frame-3, :][1])
                    analysis_data['DF2_velocity_x_shot'] = (position_data_['1_3'][shot_frame, :][0] - position_data_['1_3'][shot_frame-3, :][0])
                    analysis_data['DF2_velocity_y_shot'] = (position_data_['1_3'][shot_frame, :][1] - position_data_['1_3'][shot_frame-3, :][1])
                    analysis_data['OF1_position_x_shot'] = position_data_['0_0'][shot_frame, :][0]
                    analysis_data['OF1_position_y_shot'] = position_data_['0_0'][shot_frame, :][1]
                    analysis_data['OF2_position_x_shot'] = position_data_['0_1'][shot_frame, :][0]
                    analysis_data['OF2_position_y_shot'] = position_data_['0_1'][shot_frame, :][1]
                    analysis_data['DF1_position_x_shot'] = position_data_['1_2'][shot_frame, :][0]
                    analysis_data['DF1_position_y_shot'] = position_data_['1_2'][shot_frame, :][1]
                    analysis_data['DF2_position_x_shot'] = position_data_['1_3'][shot_frame, :][0]
                    analysis_data['DF2_position_y_shot'] = position_data_['1_3'][shot_frame, :][1]
                    analysis_data['OF1_OF2_distance_shot'] = np.sqrt(np.sum(np.square(position_data_['0_0'][shot_frame]-position_data_['0_1'][shot_frame])))
                    analysis_data['OF1_DF1_distance_shot'] = np.sqrt(np.sum(np.square(position_data_['0_0'][shot_frame]-position_data_['1_2'][shot_frame])))
                    analysis_data['OF1_DF2_distance_shot'] = np.sqrt(np.sum(np.square(position_data_['0_0'][shot_frame]-position_data_['1_3'][shot_frame])))
                    analysis_data['OF2_DF1_distance_shot'] = np.sqrt(np.sum(np.square(position_data_['0_1'][shot_frame]-position_data_['1_2'][shot_frame])))
                    analysis_data['OF2_DF2_distance_shot'] = np.sqrt(np.sum(np.square(position_data_['0_1'][shot_frame]-position_data_['1_3'][shot_frame])))
                    analysis_data['DF1_DF2_distance_shot'] = np.sqrt(np.sum(np.square(position_data_['1_2'][shot_frame]-position_data_['1_3'][shot_frame])))
                else:
                    analysis_data['OF1_velocity_x_shot'] = (position_data_['0_0'][shot_frame, :][0] - position_data_['0_0'][shot_frame-3, :][0])
                    analysis_data['OF1_velocity_y_shot'] = -((position_data_['0_0'][shot_frame, :][1] - position_data_['0_0'][shot_frame-3, :][1]))
                    analysis_data['OF2_velocity_x_shot'] = (position_data_['0_1'][shot_frame, :][0] - position_data_['0_1'][shot_frame-3, :][0])
                    analysis_data['OF2_velocity_y_shot'] = -((position_data_['0_1'][shot_frame, :][1] - position_data_['0_1'][shot_frame-3, :][1]))
                    analysis_data['DF1_velocity_x_shot'] = (position_data_['1_2'][shot_frame, :][0] - position_data_['1_2'][shot_frame-3, :][0])
                    analysis_data['DF1_velocity_y_shot'] = -((position_data_['1_2'][shot_frame, :][1] - position_data_['1_2'][shot_frame-3, :][1]))
                    analysis_data['DF2_velocity_x_shot'] = (position_data_['1_3'][shot_frame, :][0] - position_data_['1_3'][shot_frame-3, :][0])
                    analysis_data['DF2_velocity_y_shot'] = -((position_data_['1_3'][shot_frame, :][1] - position_data_['1_3'][shot_frame-3, :][1]))
                    analysis_data['OF1_position_x_shot'] = position_data_['0_0'][shot_frame, :][0]
                    analysis_data['OF1_position_y_shot'] = -(position_data_['0_0'][shot_frame, :][1])
                    analysis_data['OF2_position_x_shot'] = position_data_['0_1'][shot_frame, :][0]
                    analysis_data['OF2_position_y_shot'] = -(position_data_['0_1'][shot_frame, :][1])
                    analysis_data['DF1_position_x_shot'] = position_data_['1_2'][shot_frame, :][0]
                    analysis_data['DF1_position_y_shot'] = -(position_data_['1_2'][shot_frame, :][1])
                    analysis_data['DF2_position_x_shot'] = position_data_['1_3'][shot_frame, :][0]
                    analysis_data['DF2_position_y_shot'] = -(position_data_['1_3'][shot_frame, :][1])
                    analysis_data['OF1_OF2_distance_shot'] = np.sqrt(np.sum(np.square(position_data_['0_0'][shot_frame]-position_data_['0_1'][shot_frame])))
                    analysis_data['OF1_DF1_distance_shot'] = np.sqrt(np.sum(np.square(position_data_['0_0'][shot_frame]-position_data_['1_3'][shot_frame])))
                    analysis_data['OF1_DF2_distance_shot'] = np.sqrt(np.sum(np.square(position_data_['0_0'][shot_frame]-position_data_['1_2'][shot_frame])))
                    analysis_data['OF2_DF1_distance_shot'] = np.sqrt(np.sum(np.square(position_data_['0_1'][shot_frame]-position_data_['1_3'][shot_frame])))
                    analysis_data['OF2_DF2_distance_shot'] = np.sqrt(np.sum(np.square(position_data_['0_1'][shot_frame]-position_data_['1_2'][shot_frame])))
                    analysis_data['DF1_DF2_distance_shot'] = np.sqrt(np.sum(np.square(position_data_['1_2'][shot_frame]-position_data_['1_3'][shot_frame])))
                break
            
        if shot == False:
            analysis_data['OF1_velocity_x_shot'] = None
            analysis_data['OF1_velocity_y_shot'] = None
            analysis_data['OF2_velocity_x_shot'] = None
            analysis_data['OF2_velocity_y_shot'] = None
            analysis_data['DF1_velocity_x_shot'] = None
            analysis_data['DF1_velocity_y_shot'] = None
            analysis_data['DF2_velocity_x_shot'] = None
            analysis_data['DF2_velocity_y_shot'] = None
            analysis_data['OF1_OF2_distance_shot'] = None
            analysis_data['OF1_DF1_distance_shot'] = None
            analysis_data['OF1_DF2_distance_shot'] = None
            analysis_data['OF2_DF1_distance_shot'] = None
            analysis_data['OF2_DF2_distance_shot'] = None
            analysis_data['DF1_DF2_distance_shot'] = None
            analysis_data['shot_person'] = None
            analysis_data['OF1_position_x_shot'] = None
            analysis_data['OF1_position_y_shot'] = None
            analysis_data['OF2_position_x_shot'] = None
            analysis_data['OF2_position_y_shot'] = None
            analysis_data['DF1_position_x_shot'] = None
            analysis_data['DF1_position_y_shot'] = None
            analysis_data['DF2_position_x_shot'] = None
            analysis_data['DF2_position_y_shot'] = None
            analysis_data['shot_t'] = None

        analysis_data_[externalID] = analysis_data

    print(f'frame_error:{frame_error}')

    """ with open(path_pkl4, 'wb') as p:
            pickle.dump(analysis_data_, p) """

    return analysis_data_


# plot shot position
def plot_shot_position(analysis_data):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    r1 = patches.Rectangle(xy=(6.0, 7.0), width=10.0, height=6.0, ec='#000000', fill=False)
    r2 = patches.Rectangle(xy=(-1.0, 8.5), width=1.0, height=3.0, ec='#000000', fill=False)
    a1 = patches.Arc(xy=(0.0, 8.5), width=12.0, height=12.0, angle=0.0, theta1=270.0, theta2=360.0)
    a2 = patches.Arc(xy=(0.0, 11.5), width=12.0, height=12.0, angle=0.0, theta1=0.0, theta2=90.0)
    ax.set_aspect('equal')
    ax.set_xlim(-1.0,20.0)
    ax.set_ylim(0.0,20.0)
    ax.add_patch(r1)
    ax.add_patch(r2)
    ax.add_patch(a1)
    ax.add_patch(a2)
    ax.plot([6.0, 6.0], [8.5, 11.5], color="black", lw=1)
    ax.plot([0.0, 0.0], [0.0, 20.0], color='black', lw=1)
    shot_successes_x = []
    shot_successes_y = []
    shot_failures_x = []
    shot_failures_y = []
    for externalID, analysis_data in analysis_data_.items(): # 動画ごとに処理
        #print('export: '+ externalID)
        # ax.cla()
        if np.all(analysis_data['shot_person'] != None):
            if analysis_data['result'] == 1:
                if analysis_data['shot_person'] == '0_0':
                    shot_successes_x.append(analysis_data['OF1_position_x_shot'])
                    shot_successes_y.append(analysis_data['OF1_position_y_shot'])
                else:
                    shot_successes_x.append(analysis_data['OF2_position_x_shot'])
                    shot_successes_y.append(analysis_data['OF2_position_y_shot'])
            else:
                if analysis_data['shot_person'] == '0_0':
                    shot_failures_x.append(analysis_data['OF1_position_x_shot'])
                    shot_failures_y.append(analysis_data['OF1_position_y_shot'])
                else:
                    shot_failures_x.append(analysis_data['OF2_position_x_shot'])
                    shot_failures_y.append(analysis_data['OF2_position_y_shot'])
    ax.scatter(shot_successes_x,shot_successes_y,c='red',s=5,label='shot success')
    ax.scatter(shot_failures_x,shot_failures_y,c='black',s=5,label='shot failure')
    ax.legend(fontsize=14)
    fig.tight_layout()
    save_dir = f"./figure/shot_position/shot_position.png"
    fig.savefig(save_dir)
    
    
# シュートを打った選手がOF1かOF2かによって色分け
def plot_shot_position_split_OF1_OF2(analysis_data):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    r1 = patches.Rectangle(xy=(6.0, 7.0), width=10.0, height=6.0, ec='#000000', fill=False)
    r2 = patches.Rectangle(xy=(-1.0, 8.5), width=1.0, height=3.0, ec='#000000', fill=False)
    a1 = patches.Arc(xy=(0.0, 8.5), width=12.0, height=12.0, angle=0.0, theta1=270.0, theta2=360.0)
    a2 = patches.Arc(xy=(0.0, 11.5), width=12.0, height=12.0, angle=0.0, theta1=0.0, theta2=90.0)
    ax.set_aspect('equal')
    ax.set_xlim(-1.0,20.0)
    ax.set_ylim(0.0,20.0)
    ax.add_patch(r1)
    ax.add_patch(r2)
    ax.add_patch(a1)
    ax.add_patch(a2)
    ax.plot([6.0, 6.0], [8.5, 11.5], color="black", lw=1)
    ax.plot([0.0, 0.0], [0.0, 20.0], color='black', lw=1)
    OF1_shot_x = []
    OF1_shot_y = []
    OF2_shot_x = []
    OF2_shot_y = []
    for externalID, analysis_data in analysis_data_.items(): # 動画ごとに処理
        #print('export: '+ externalID)
        # ax.cla()
        if np.all(analysis_data['shot_person'] != None):
            if analysis_data['shot_person'] == '0_0':
                OF1_shot_x.append(analysis_data['OF1_position_x_shot'])
                OF1_shot_y.append(analysis_data['OF1_position_y_shot'])
            else:
                OF2_shot_x.append(analysis_data['OF2_position_x_shot'])
                OF2_shot_y.append(analysis_data['OF2_position_y_shot'])
    ax.scatter(OF1_shot_x,OF1_shot_y,c='red',s=5,label='OF1 shot')
    ax.scatter(OF2_shot_x,OF2_shot_y,c='black',s=5,label='OF2 shot')
    ax.legend(fontsize=14)
    fig.tight_layout()
    save_dir = f"./figure/shot_position/shot_position_split_OF1_OF2.png"
    fig.savefig(save_dir)


def plot_shot_position_by_person(analysis_data):
    shot_person_list = ['0_0', '0_1']
    for shot_person in shot_person_list:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        r1 = patches.Rectangle(xy=(6.0, 7.0), width=10.0, height=6.0, ec='#000000', fill=False)
        r2 = patches.Rectangle(xy=(-1.0, 8.5), width=1.0, height=3.0, ec='#000000', fill=False)
        a1 = patches.Arc(xy=(0.0, 8.5), width=12.0, height=12.0, angle=0.0, theta1=270.0, theta2=360.0)
        a2 = patches.Arc(xy=(0.0, 11.5), width=12.0, height=12.0, angle=0.0, theta1=0.0, theta2=90.0)
        ax.set_aspect('equal')
        ax.set_xlim(-1.0,20.0)
        ax.set_ylim(0.0,20.0)
        # ax.set_xlabel('x')
        # ax.set_ylabel('y')
        ax.add_patch(r1)
        ax.add_patch(r2)
        ax.add_patch(a1)
        ax.add_patch(a2)
        ax.plot([6.0, 6.0], [8.5, 11.5], color="black", lw=1)
        ax.plot([0.0, 0.0], [0.0, 20.0], color='black', lw=1)
        shot_successes_x = []
        shot_successes_y = []
        shot_failures_x = []
        shot_failures_y = []
        for externalID, analysis_data in analysis_data_.items(): # 動画ごとに処理
            #print('export: '+ externalID)
            # ax.cla()
            if np.all(analysis_data['shot_person'] != None):
                if analysis_data['result'] == 1:
                    if shot_person == '0_0':
                        if analysis_data['shot_person'] == '0_0':
                            shot_successes_x.append(analysis_data['OF1_position_x_shot'])
                            shot_successes_y.append(analysis_data['OF1_position_y_shot'])
                    else:
                        if analysis_data['shot_person'] == '0_1':
                            shot_successes_x.append(analysis_data['OF2_position_x_shot'])
                            shot_successes_y.append(analysis_data['OF2_position_y_shot'])
                else:
                    if shot_person == '0_0':
                        if analysis_data['shot_person'] == '0_0':
                            shot_failures_x.append(analysis_data['OF1_position_x_shot'])
                            shot_failures_y.append(analysis_data['OF1_position_y_shot'])
                    else:
                        if analysis_data['shot_person'] == '0_1':
                            shot_failures_x.append(analysis_data['OF2_position_x_shot'])
                            shot_failures_y.append(analysis_data['OF2_position_y_shot'])
        ax.scatter(shot_successes_x,shot_successes_y,c='red',s=5,label='shot success')
        ax.scatter(shot_failures_x,shot_failures_y,c='black',s=5,label='shot failure')
        ax.legend(fontsize=14)
        fig.tight_layout()
        save_dir = f"./figure/shot_position/shot_position_{shot_person}.png"
        fig.savefig(save_dir)

# plot catch position
def plot_catch_position(analysis_data):
    fig_title = f"catch_position"
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for externalID, analysis_data in analysis_data_.items(): # 動画ごとに処理
        print('export: '+ externalID)
        # ax.cla()
        if np.all(analysis_data['shot_person'] != None):
            if analysis_data['result'] == 1:
                color = 'red'
            else:
                color = 'black'
            ax.plot(analysis_data['OF1_position_x_catch'], analysis_data['OF1_position_y_catch'], marker='o', color=color, markersize=2)
    fig.savefig(f"./figure/{fig_title}.png")


def frequency_table(data, stur=True):
    data_len = len(data)
    #print('データ数：', data_len)
    #スタージェンスの公式でbinの数を求める
    if stur is True:
        b = round(1 + np.log2(data_len))
        hist, bins = np.histogram(data, bins=b) 
    else:
        hist, bins = np.histogram(data)
    #データフレーム作成
    df = pd.DataFrame(
        {
            '以上': bins[:-1],
            '以下': bins[1:],
            '階級値': (bins[:-1]+bins[1:])/2,
            '度数': hist
        }
    )
 
    #相対度数の計算
    df['相対度数'] = df['度数'] / data_len

    #累積度数の計算
    df['累積度数'] = np.cumsum(df['度数'])

    #累積相対度数の計算
    df['累積相対度数'] = np.cumsum(df['相対度数'])
    # print(df)
    return df, b


def plot_hist(df, feature):
    ft, n_bins = frequency_table(df[feature])
    width = (max(df[feature]) - min(df[feature])) / n_bins
    fig = plt.figure(figsize=(18,16), tight_layout=True)
    ax = plt.subplot(111)
    ax.hist(df[feature], bins=n_bins, ec='black', color='#005AFF')
    save_dir = f'./figure/hist/{feature}_hist.png'
    fig.savefig(save_dir)
    results = np.zeros(n_bins)
    for index, row in df.iterrows():
        result = row['result']
        for i in range(n_bins):
            if (row[feature] > ft['以上'][i]) and (row[feature] < ft['以下'][i]):
                results[i] += result
    shot_success_rate = np.zeros(n_bins)
    for i in range(n_bins):
        if ft['度数'][i] != 0:
            shot_success_rate[i] = results[i] / ft['度数'][i]
        else:
            shot_success_rate[i] = 0
    fig = plt.figure(figsize=(18,16), tight_layout=True)
    ax = plt.subplot(111)
    ax.bar(ft['階級値'], shot_success_rate, align="center", width=width, ec='black', color='#005AFF')
    save_dir = f'./figure/hist/{feature}_shot_success_rate.png'
    fig.savefig(save_dir)
    
def calculate_shot_success_rate(df):
    df = pd.get_dummies(shot_df[['result', 'shot_type']])
    df_pivot_shot = df[df['shot_type_pivot shot'] == 1]
    df_pivot_shot = df_pivot_shot[['shot_type_pivot shot', 'result']]
    df_long_shot = df[df['shot_type_long shot'] == 1]
    df_long_shot = df_long_shot[['shot_type_long shot', 'result']]
    df_cut_in = df[df['shot_type_cut in'] == 1]
    df_cut_in = df_cut_in[['shot_type_cut in', 'result']]
    df_pivot_shot_bool = (df_pivot_shot == 1)
    df_long_shot_bool = (df_long_shot == 1)
    df_cut_in_bool = (df_cut_in == 1)
    """ print(df_pivot_shot_bool.sum())
    print(df_pivot_shot_bool.sum()['result'] / df_pivot_shot_bool.sum()['shot_type_pivot shot'])
    print(df_long_shot_bool.sum())
    print(df_long_shot_bool.sum()['result'] / df_long_shot_bool.sum()['shot_type_long shot'])
    print(df_cut_in_bool.sum())
    print(df_cut_in_bool.sum()['result'] / df_cut_in_bool.sum()['shot_type_cut in']) """
    
    
    
def predict_score_lr(df):
    print('Logistic regression:')
    acc= []
    f1 = []
    coef_ = []
    X = df.drop('result', axis=1)
    y = df['result']
    #print(f'features:{features}')
    # scale
    sc = StandardScaler()
    X_std = sc.fit_transform(X)
    test_size = 0.2
    for i in range(100):
        X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X_std, y, test_size=test_size, random_state=i)
        
        model = LogisticRegression(random_state=0, max_iter=10000)
        model.fit(X_train, Y_train)
        coef_.append(model.coef_[0])
        predict = model.predict(X_test)
        # print(f'predict = {predict}')
        # print(f'true = {Y_test.values.tolist()}')
        acc.append(accuracy_score(y_true=Y_test, y_pred=predict))
        f1.append(f1_score(y_true=Y_test, y_pred=predict))
    
    return coef_, acc, f1


def predict_score_svc(df):
    print('SVC:')
    acc= []
    f1 = []
    X = df.drop('result', axis=1)
    Y = df['result']
    #print(f'features:{features}')
    # scale
    sc = StandardScaler()
    X_std = sc.fit_transform(X)
    test_size = 0.2
    for i in range(100):
        X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X_std, Y, test_size=test_size, random_state=i)
        
        model = SVC(random_state=0, max_iter=10000)
        model.fit(X_train, Y_train)
        predict = model.predict(X_test)
        # print(f'predict = {predict}')
        # print(f'true = {Y_test.values.tolist()}')
        acc.append(accuracy_score(y_true=Y_test, y_pred=predict))
        f1.append(f1_score(y_true=Y_test, y_pred=predict))

    acc_mean = np.mean(acc)
    acc_sd = np.std(acc)
    f1_mean = np.mean(f1)
    f1_sd = np.std(f1)
    print(f"accuracy = {'{:.2f}'.format(round(acc_mean, 2))}±{'{:.2f}'.format(round(acc_sd, 2))}")
    print(f"f1 score = {'{:.2f}'.format(round(f1_mean, 2))}±{'{:.2f}'.format(round(f1_sd, 2))}")
    
    return acc, f1


def predict_score_random_forest(df):
    print('Random forest:')
    acc= []
    f1 = []
    fti = []
    X = df.drop('result', axis=1)
    Y = df['result']
    #print(f'features:{features}')
    # scale
    sc = StandardScaler()
    X_std = sc.fit_transform(X)
    test_size = 0.2
    for i in range(100):
        X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X_std, Y, test_size=test_size, random_state=i)
        
        model = RandomForestClassifier()
        model.fit(X_train, Y_train)
        fti.append(model.feature_importances_)
        predict = model.predict(X_test)
        # print(f'predict = {predict}')
        # print(f'true = {Y_test.values.tolist()}')
        acc.append(accuracy_score(y_true=Y_test, y_pred=predict))
        f1.append(f1_score(y_true=Y_test, y_pred=predict))

    """ acc_mean = np.mean(acc)
    acc_sd = np.std(acc)
    f1_mean = np.mean(f1)
    f1_sd = np.std(f1)
    for i in range(len(fti)):
        if i == 0:
            fti_mean = (fti[0] / len(fti))
        else:
            fti_mean += (fti_[i] / len(fti))
    for i in range(len(fti)):
        if i == 0:
            fti_abs_mean = (abs(fti[0]) / len(fti))
        else:
            fti_abs_mean += (abs(fti[i]) / len(fti))
    
    print(f"accuracy = {'{:.2f}'.format(round(acc_mean, 2))}±{'{:.2f}'.format(round(acc_sd, 2))}")
    print(f"f1 score = {'{:.2f}'.format(round(f1_mean, 2))}±{'{:.2f}'.format(round(f1_sd, 2))}")
 """    
    return fti, acc, f1



def plot_bar_(coef_mean, features, sorted_index, save_dir):
    color = [('#FF4B00' if 'distance' in x else ('#005AFF' if 'position' in x else ('#03AF7A' if 'velocity' in x else '#4DC4FF')))for x in features[sorted_index]]
    #color = [('#005AFF' if 'distance' in x else ('#FF4B00' if 'position' in x else '#03AF7A'))for x in features[sorted_index]]
    labels = []
    flag_distance = False
    flag_position = False
    flag_velocity = False
    flag_other = False
    for feature in features[sorted_index]:
        if 'distance' in feature:
            if flag_distance == False:
                labels.append('distance')
                flag_distance = True
            else:
                labels.append(None)
        elif 'position' in feature:
            if flag_position == False:
                labels.append('position')
                flag_position = True
            else:
                labels.append(None)
        elif 'velocity' in feature:
            if flag_velocity == False:
                labels.append('velocity')
                flag_velocity = True
            else:
                labels.append(None)
        else:
            if flag_other == False:
                labels.append('other')
                flag_other = True
            else:
                labels.append(None)
    #print(labels)
    fig = plt.figure(figsize=(26,24), tight_layout=True)
    plt.tick_params(labelsize=32)
    ax = plt.subplot(111)
    ax.set_xlabel("Importance",fontsize=32)
    ax.barh(features[sorted_index], np.sort(coef_mean)[::-1], align="center", color=color, label=labels)
    ax.legend(fontsize = 32)
    #ax.text(0.65, 0.85, f"accuracy = {'{:.2f}'.format(round(acc_mean, 2))}±{'{:.2f}'.format(round(acc_sd, 2))}", verticalalignment='top', transform=ax.transAxes)
    #ax.text(0.65, 0.80, f"f1 score = {'{:.2f}'.format(round(f1_mean, 2))}±{'{:.2f}'.format(round(f1_sd, 2))}", verticalalignment='top', transform=ax.transAxes)
    print(save_dir)
    fig.savefig(f'./figure/{save_dir}.png')
    plt.close()

def plot_bar(df, save_dir, coef_, acc, f1):
    features = np.array(df.drop('result', axis=1).columns)
    for i in range(len(coef_)):
        if i == 0:
            coef_mean = (coef_[0] / len(coef_))
        else:
            coef_mean += (coef_[i] / len(coef_))
    for i in range(len(coef_)):
        if i == 0:
            coef_abs_mean = (abs(coef_[0]) / len(coef_))
        else:
            coef_abs_mean += (abs(coef_[i]) / len(coef_))
    acc_mean = np.mean(acc)
    acc_sd = np.std(acc)
    f1_mean = np.mean(f1)
    f1_sd = np.std(f1)
    print(f"accuracy = {'{:.2f}'.format(round(acc_mean, 2))}±{'{:.2f}'.format(round(acc_sd, 2))}")
    print(f"f1 score = {'{:.2f}'.format(round(f1_mean, 2))}±{'{:.2f}'.format(round(f1_sd, 2))}")
    sorted_index = np.argsort(coef_mean)[::-1]
    sorted_index_abs = np.argsort(coef_abs_mean)[::-1]
    #print(sorted_index)
    #print(features[sorted_index])
    plot_bar_(coef_mean, features, sorted_index, save_dir)
    plot_bar_(coef_abs_mean, features, sorted_index_abs, save_dir+'_abs')
    
    

######### import positions data #####################
positions_data_path = 'positions_data.json'
if(os.path.exists(positions_data_path) == True):
# if False:
    with open(positions_data_path, 'rb') as p:
        positions_data = json.load(p)
    print("loading positions data completed")
else:
    print("Error: loading positions data is failed")
    sys.exit(1)

for externalID, position_data_ in positions_data.items():
    for objectID, position_data in position_data_.items():
        position_data_[objectID] = np.array(position_data)
    positions_data[externalID] = position_data_
# print(positions_data)
# get_animation(positions_data)

################ analyze handball 2vs2 ##################
analysis_data_path = 'analysis_data_.bin'
if(os.path.exists(analysis_data_path) == True):
# if False:
    with open(analysis_data_path, 'rb') as p:
        analysis_data_ = pickle.load(p)
else:
    analysis_data_ = get_analysis_data(positions_data)

plot_shot_position(analysis_data_)
#plot_shot_position_by_person(analysis_data_)
#plot_catch_position(analysis_data_)
plot_shot_position_split_OF1_OF2(analysis_data_)

df = pd.DataFrame.from_dict(analysis_data_, orient='index')
shot_df = df.dropna(subset='shot_person')
of1_shot_df = shot_df[shot_df['shot_person']=='0_0']
of1_long_shot_df = shot_df[shot_df['shot_type']=='long shot']
of1_cut_in_df = shot_df[shot_df['shot_type']=='cut in']
of2_shot_df = shot_df[shot_df['shot_person']=='0_1']

# 各種のシュート成功率を計算
calculate_shot_success_rate(df)

# ヒストグラム作成
plot_hist(shot_df, 'OF2_DF2_distance_catch')
plot_hist(shot_df, 'OF1_velocity_y_catch')
plot_hist(shot_df, 'OF1_velocity_x_catch')
plot_hist(shot_df, 'OF2_DF1_distance_catch')

if False:
    plot_hist(shot_df, 'OF1_OF2_distance_shot')
    plot_hist(shot_df, 'OF1_position_x_catch')
    plot_hist(shot_df, 'DF2_position_y_catch')
    plot_hist(shot_df, 'OF1_position_y_catch')
    plot_hist(of1_long_shot_df, 'shot_t')

columns_inc_distance = [column for column in shot_df.columns if 'distance' in column]
columns_inc_catch = [column for column in shot_df.columns if 'catch' in column]
columns_inc_shot = [column for column in shot_df.columns if 'shot' in column]
columns_inc_position = [column for column in shot_df.columns if 'position' in column]
columns_inc_velocity = [column for column in shot_df.columns if 'velocity' in column]

# only selected features(position, distance, OF1's velocity, and time of catch)
if True:
    """ # all shot
    df = shot_df[list(set(columns_inc_distance)&set(columns_inc_catch))+['result', 'OF1_velocity_x_catch', 'OF1_velocity_y_catch', 'OF1_position_x_catch', 'OF1_position_y_catch']]
    save_dir = 'all_shot/selected_importance'
    coef_, acc, f1 = predict_score_lr(df)
    plot_bar(df, save_dir, coef_, acc, f1)
    predict_score_svc(df)
    fti, acc, f1 = predict_score_random_forest(df)
    plot_bar(df, f'{save_dir}_random_forest', fti, acc, f1) """
    
    # OF1 shot
    df = of1_shot_df[list(set(columns_inc_distance)&set(columns_inc_catch))+['result', 'OF1_velocity_x_catch', 'OF1_velocity_y_catch', 'OF1_position_x_catch', 'OF1_position_y_catch']]
    save_dir = 'OF1_shot/selected_features'
    coef_, acc, f1 = predict_score_lr(df)
    plot_bar(df, save_dir, coef_, acc, f1)
    predict_score_svc(df)
    """ fti, acc, f1 = predict_score_random_forest(df)
    plot_bar(df, f'{save_dir}_random_forest', fti, acc, f1) """

    
    # OF2 shot
    """ df = of2_shot_df[list(set(columns_inc_distance)&set(columns_inc_catch))+['result', 'OF1_velocity_x_catch', 'OF1_velocity_y_catch', 'OF1_position_x_catch', 'OF1_position_y_catch']]
    save_dir = 'OF2_shot/selected_features'
    coef_, acc, f1 = predict_score_lr(df)
    plot_bar(df, save_dir, coef_, acc, f1)
    predict_score_svc(df)
    fti, acc, f1 = predict_score_random_forest(df)
    plot_bar(df, f'{save_dir}_random_forest', fti, acc, f1) """
    
    
    # 全特徴量
    # all shot
    """ df = pd.get_dummies(shot_df.drop(['shot_person', 'shot_type', 'other_results', 'block_success', 'OF1_dominant_hand'], axis=1))
    save_dir = 'all_shot/all_features'
    coef_, acc, f1 = predict_score_lr(df)
    plot_bar(df, save_dir, coef_, acc, f1)
    predict_score_svc(df)
    fti, acc, f1 = predict_score_random_forest(df)
    plot_bar(df, f'{save_dir}_random_forest', fti, acc, f1) """
    
    
    # OF1 shot
    df = pd.get_dummies(of1_shot_df.drop(['shot_person', 'shot_type', 'other_results', 'block_success', 'OF1_dominant_hand'], axis=1))
    save_dir = 'OF1_shot/OF1_all_features'
    coef_, acc, f1 = predict_score_lr(df)
    plot_bar(df, save_dir, coef_, acc, f1)
    predict_score_svc(df)
    fti, acc, f1 = predict_score_random_forest(df)
    plot_bar(df, f'{save_dir}_random_forest', fti, acc, f1)
    
    # OF2 shot
    """ df = pd.get_dummies(of2_shot_df.drop(['shot_person', 'shot_type', 'other_results', 'block_success', 'OF1_dominant_hand'], axis=1))
    save_dir = 'OF2_shot/OF2_all_features_importance'
    coef_, acc, f1 = predict_score_lr(df)
    plot_bar(df, save_dir, coef_, acc, f1)
    predict_score_svc(df)
    fti, acc, f1 = predict_score_random_forest(df)
    plot_bar(df, f'{save_dir}_random_forest', fti, acc, f1) """
    
    