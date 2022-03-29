# 加载依赖
from __future__ import division
import paddlehub as hub
import cv2 as cv
import time
import pandas as pd
import os
import numpy as np

division = 100


# 坐标变换
def transform(data, center):
    for idx, (name, point) in enumerate(data.items()):
        if point.any():
            point -= center


def delete_all_files(dir: str):
    for file in os.listdir(dir):
        os.remove(os.path.join(dir, file))


def save_csv(data, filename, base_dir='./work'):
    dat = pd.DataFrame(data)
    dat.index = ['x', 'y']
    dat.to_csv(os.path.join(base_dir, filename))


def process_csv(base_path='output', thresh=.3, output_path='user'):
    sz = len(os.listdir(base_path)) // 3  # 3个文件一套
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    else:
        delete_all_files(output_path)  # 删除先前所有文件

    proc = []

    pos_name = ('nose', 'neck', 'rshoulder', 'relbow', 'rhand', 'lshoulder',
                'lelbow', 'lhand', 'rhip', 'rknee', 'rankle', 'lhip', 'lknee',
                'lankle')

    for file in range(sz):
        candidate_dat = pd.read_csv(os.path.join(base_path, f'{file}_candidate.csv'))
        subset_dat = pd.read_csv(os.path.join(base_path, f'{file}_subset.csv'))
        # print(subset_dat.shape)
        if candidate_dat.empty:
            print('无人')
            continue
        if subset_dat.shape[0] > 1:
            print('多人！')
            continue

        pos = {}

        # 只需要看到14个就够了，不需要脸部数据
        for idx in range(1, 15):
            k = subset_dat.loc[0][idx]
            if k == -1:
                pos[pos_name[idx - 1]] = np.array([np.NAN, np.NAN])
                continue
            elif candidate_dat.loc[k][3] < thresh:
                pos[pos_name[idx - 1]] = np.array([np.NAN, np.NAN])
                continue
            else:
                pos[pos_name[idx - 1]] = np.array([int(candidate_dat.loc[k][1]), int(candidate_dat.loc[k][2])])
        transform(pos, pos['neck'].copy())
        save_csv(pos, filename=f'{file}.csv', base_dir=output_path)
        proc.append(pos)
    return proc


def process(model, base_path='./data', output_path='output'):
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    else:
        delete_all_files(output_path)
    for file in os.listdir(base_path):
        img = cv.imread(os.path.join(base_path, file))
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        ret = model.predict(img)
        out_img = ret['data']
        out_img = cv.cvtColor(out_img, cv.COLOR_RGB2BGR)
        cv.imwrite(os.path.join(output_path, f'{file[:-4]}_res.jpg'), out_img)
        candidate = ret['candidate']
        subset = ret['subset']
        candidate_dat = pd.DataFrame(candidate)
        subset_dat = pd.DataFrame(subset)
        candidate_dat.to_csv(os.path.join(output_path, f'{file[:-4]}_candidate.csv'))
        subset_dat.to_csv(os.path.join(output_path, f'{file[:-4]}_subset.csv'))


# 划分视频为10帧的图片
def split_video(video: str, base_path='./data', fps_limit=10) -> bool:
    if not os.path.exists(base_path):
        os.mkdir(base_path)
    delete_all_files(base_path)

    cap = cv.VideoCapture(video)
    elapsed = 1 / fps_limit
    if cap.isOpened():
        prev = time.time()
        i = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            elapsed_time = time.time() - prev
            if elapsed_time > elapsed:
                prev = time.time()
                cv.imwrite(os.path.join(base_path, f'{i}.jpg'), frame)
                i += 1
        return True
    return False


def dis(point1, point2):
    if np.isnan(point1).any() or np.isnan(point2).any():
        return 0
    ans = (point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2
    return ans


def series_dis(series1, series2):
    ans = 0
    for idx, (name, point1) in enumerate(series1.items()):
        point2 = series2[name]
        ans += dis(point1, point2)
    return ans


# DTW算法
def dynamic_time_warping(proc1, proc2):
    l1, l2 = len(proc1), len(proc2)
    dp = [[0x7f7f7f7f for j in range(l2 + 1)] for i in range(l1 + 1)]
    dp[0][0] = 0

    for i in range(1, l1 + 1):
        for j in range(1, l2 + 1):
            cost = series_dis(proc1[i - 1], proc2[j - 1])
            dp[i][j] = min(cost + dp[i - 1][j - 1], min(dp[i - 1][j], dp[i][j - 1]))
    return dp[l1][l2] / division


if __name__ == '__main__':
    out = process_csv(base_path='user_output', output_path='user')

    # 加载模型
    model = hub.Module(name='openpose_body_estimation')
    # splite_video('./test.mp4')
    # process(output_path='user_output')
    out = process_csv(base_path='user_output', output_path='user')
    criterion = process_csv()
    cost = dynamic_time_warping(criterion, out)
    print(cost)
