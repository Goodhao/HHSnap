from tkinter import *
from PIL import Image, ImageTk
import cv2
import numpy as np
import copy
import matplotlib.pyplot as plt
import random
from scipy.interpolate import splprep, splev
from scipy.spatial.distance import directed_hausdorff
from scipy.integrate import quad
import time
from vectorization import close_curve_for_snap
from collections import defaultdict
import sys
import warnings
import os
from loader import load, build_sketch
import argparse

parser = argparse.ArgumentParser(description='HHSnap')
parser.add_argument('--input', dest='filename', type=str, help='输入文件名，要求存放在img文件夹下')
parser.add_argument('--detect_corner', dest='detect_corner', help='是否检测图片的角点')
args = parser.parse_args()

filename = args.filename
detect_corner = bool(args.detect_corner) if args.detect_corner else False
path = os.path.join('img', filename)
original = cv2.imread(path)
img = copy.deepcopy(original)
sample = load(original)
path = os.path.join('img', f'out_{filename}')
cv2.imwrite(path, sample)
H, W = sample.shape[:2]
contour, _ = build_sketch(sample)
contour = [edge for edge in contour if len(edge) > 3]
if detect_corner:
    # 将图像转为灰度图
    gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    # 设定Shi-Tomasi角点检测的参数
    maxCorners = 100
    qualityLevel = 0.01
    minDistance = 10
    blockSize = 3
    k = 0.15
    # 进行Shi-Tomasi角点检测
    corners = cv2.goodFeaturesToTrack(gray, maxCorners, qualityLevel, minDistance, blockSize, useHarrisDetector=True, k=k)
    for corner in corners:
        x, y = corner.ravel()
        x = int(x)
        y = int(y)
        # cv2.circle(original, (int(x), int(y)), 3, (0, 0, 255), -1)
        d = 1e8
        idx1 = 0
        idx2 = 0
        for j, edge in enumerate(contour):
            for i, p in enumerate(edge):
                d2 = np.sqrt((p[0]-y)**2+(p[1]-x)**2) # 注意返回的x,y代表横纵坐标
                if d2 < d:
                    d = d2
                    idx1, idx2 = j, i
        original[contour[idx1][idx2][0], contour[idx1][idx2][1]] = [0, 0, 255]
    for j, edge in enumerate(contour):
        start = -1
        for i, p in enumerate(edge):
            if (original[p[0], p[1]] == [0, 0, 255]).all():
                start = i
                break
        if start == -1:
            continue
        start += 1
        edge2 = []
        contour2 = []
        for i in range(start, start + len(edge)):
            p = edge[i % len(edge)]
            edge2.append(p)
            if (original[p[0], p[1]] == [0, 0, 255]).all():
                contour2.append(edge2)
                edge2 = []
        contour.pop(j)
        for edge in contour2:
            contour.append(edge)

    for edge in contour:
        b, g, r = random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)
        for i, p in enumerate(edge):
            original[p[0], p[1]] = [b, g, r]
    # cv2.imwrite('Corner_Detection.png', original)

tangent = [[] for _ in range(len(contour))]
length = [[] for _ in range(len(contour))]
user_input = []

def distance(p1, p2):
    """
    计算两点之间的距离
    """
    return np.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)
def closest_end(c1, c2, exclude1=[], exclude2=[]):
    d = []
    d.append(distance(c1[0], c2[0]) if 0 not in exclude1 and 0 not in exclude2 else 1e8)
    d.append(distance(c1[0], c2[-1]) if 0 not in exclude1 and -1 not in exclude2 else 1e8)
    d.append(distance(c1[-1], c2[0]) if -1 not in exclude1 and 0 not in exclude2 else 1e8)
    d.append(distance(c1[-1], c2[-1]) if -1 not in exclude1 and -1 not in exclude2 else 1e8)
    i = np.argmin(d)
    if d[i] == 1e8:
        return -2, -2
    if i == 0:
        return 0, 0
    elif i == 1:
        return 0, -1
    elif i == 2:
        return -1, 0
    else:
        return -1, -1
def adjacency_list(coords, threshold):
    """
    为曲线计算邻接表
    """
    n = len(coords)
    adj_list = [[] for _ in range(n)]
    for i in range(n):
        for j in range(i+1, n):
            # 计算曲线i和曲线j的端点之间的距离
            d1 = distance(coords[i][0], coords[j][0])
            d2 = distance(coords[i][0], coords[j][-1])
            d3 = distance(coords[i][-1], coords[j][0])
            d4 = distance(coords[i][-1], coords[j][-1])
            # 如果距离不超过阈值，则将i和j加入彼此的邻接表中
            if d1 <= threshold or d2 <= threshold or d3 <= threshold or d4 <= threshold:
                adj_list[i].append(j)
                adj_list[j].append(i)
    return adj_list


win = Tk()
win.geometry(f'{W}x{H}')
canvas = Canvas(win, width=W, height=H)
canvas.pack()
bg = ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)))
canvas.create_image(W/2, H/2, image=bg, anchor='center')
canvas_img_id = canvas.find_all()[0]



# 弧长步长
ds = 5

def resample(i, edge):
    global contour
    global tangent

    if len(edge) == 0:
        return
    tck, u = splprep(np.array(edge).T, k=3, s=100)

    def integrand(u):
        # 求解曲线在u处的导数
        dxdu, dydu = splev(u, tck, der=1)
        # 返回导数的模长
        return np.sqrt(dxdu ** 2 + dydu ** 2)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        length[i], _ = quad(integrand, u.min(), u.max())

    num_points = int(length[i] / ds) + 1

    t = [u.min() + k * (u.max() - u.min()) / (num_points - 1) for k in range(num_points)]
    p = [splev(u, tck) for u in t]
    max_iter = 50
    last_offset = 0
    for j in range(1, num_points):
        for k in range(max_iter):
            dis = np.sqrt((p[j][0] - p[j-1][0]) ** 2 + (p[j][1] - p[j-1][1]) ** 2)
            offset = last_offset + dis - ds
            first_order = np.linalg.norm(splev(t[j], tck, der=1))
            second_order = np.linalg.norm(splev(t[j], tck, der=2))
            numerator = offset * first_order
            denominator = offset * second_order + first_order * first_order
            t[j] = t[j] - numerator / denominator
            t[j] = np.clip(t[j], t[j-1], u.max())
            p[j] = splev(t[j], tck)
        dis = np.sqrt((p[j][0] - p[j-1][0]) ** 2 + (p[j][1] - p[j-1][1]) ** 2)
        last_offset = last_offset + dis - ds

    x, y = splev(t, tck)
    contour[i] = [[x[_], y[_]] for _ in range(len(x))]
    tangent[i] = [splev(t[0], tck, der=1), splev(t[-1], tck, der=1)]


def get_sublist(lst, start, length):
    end = start + length
    if end <= len(lst):
        return lst[start:end]
    else:
        return lst[start:] + lst[:end % len(lst)]

def match(idx):
    if len(contour[idx]) > len(contour[-1]):
        tmp = contour[idx][:len(contour[-1])] 
    else:
        tmp = contour[idx]
    l = len(tmp)
    dd = []
    for i in range(len(contour[-1])):
        dd.append(directed_hausdorff(tmp, get_sublist(contour[-1], i, l))[0])
    assert len(dd) > 0
    return np.argmin(dd), np.min(dd)

for i, edge in enumerate(contour):
    resample(i, edge)

threshold = 15 # 两点距离若小于该阈值则认为是相邻的
threshold2 = 20 # 两条边的豪斯多夫距离若小于该阈值则认为是接近的
# global_adj_list = adjacency_list(contour, threshold)

# for i in range(len(adj_list)):
#     global_adj_list[i] = list(set(global_adj_list[i]))

contour.append(user_input) # 加入空的user_input占位
tangent.append([])
length.append([])

fig, ax = plt.subplots()
plt.xlim(0, W)
plt.ylim(0, H)
for i, edge in enumerate(contour):
    x, y = np.array([edge[_][0] for _ in range(len(edge))]), np.array([edge[_][1] for _ in range(len(edge))])
    ax.plot(y, H - x)
    ax.text(np.mean(y) + 1, np.mean(H - x) + 1, f"{i}", ha="center", va="center", fontsize=12, color="red")
plt.show()


start_move = 0
ovals = []
def on_key_press(event):
    global user_input
    global contour
    global ovals
    global start_move

    def take_design_part(v1, v2, v3, user_tck, user_u):
        i1 = np.argmin([distance(user_input[j], v1) for j in range(len(user_input))])
        i2 = np.argmin([distance(user_input[j], v3) for j in range(len(user_input))])
        part1 = []
        part2 = []
        for i in range(i1, i1 + len(user_input)):
            ii = i % len(user_input)
            part1.append(user_input[ii])
            if ii == i2:
                break
        for i in range(i2, i2 + len(user_input)):
            ii = i % len(user_input)
            part2.append(user_input[ii])
            if ii == i1:
                break
        vv = splev(user_u[i1], user_tck, der=1)
        if np.dot(vv, v2) > 0:
            return part1
        else:
            return part2

    for oval in ovals:
        canvas.delete(oval)
    ovals = []
    start_move = 0
    contour[-1] = user_input
    start_time = time.time()
    resample(len(contour) - 1, contour[-1])
    end_time = time.time()
    print(end_time - start_time, 's')

    user_tck, user_u = splprep(np.array(user_input).T, k=3, s=100)
    ss = []
    dd = []
    for i in range(len(contour) - 1):
        s, d_hd = match(i)
        ss.append(s)
        dd.append(d_hd)
    res = []
    for i in range(len(dd)):
        if dd[i] < min(threshold2, 0.7 * length[i]):
            res.append(i)
    seq = []
    for p in contour[-1]:
        mn = 1e8
        u = -1
        for i in res:
            m = np.min([(q[0] - p[0]) ** 2 + (q[1] - p[1]) ** 2 for q in contour[i]])
            if m < mn:
                mn = m
                u = i
        if u != -1 and u not in seq:
            seq.append(u)
    print('seq', seq)
    seq2 = []
    for i in seq:
        d1 = distance(contour[-1][ss[i]], contour[i][-1])
        d2 = distance(contour[-1][ss[i]], contour[i][0])
        e = np.argmin([d1, d2]) - 1
        if ss[i] + min(len(contour[i]), len(contour[-1])) - 1 < len(contour[-1]):
            seq2.append((i, e))
            seq2.append((i, -1 - e))
        else:
            seq2.append((i, -1 - e))
    if len(seq2) % 2:
        seq2.append((seq2[0][0], -1 - seq2[0][1]))
    print('seq2', seq2)

    ex_dict = defaultdict(list)
    for i in range(len(seq)):
        c1 = contour[seq[i-1]]
        c2 = contour[seq[i]]
        e1, e2 = closest_end(c1, c2)
        if (e1 != -2 and distance(c1[e1], c2[e2]) <= threshold):
            ex_dict[seq[i-1]].append(e1)
            ex_dict[seq[i]].append(e2)
    complete_pair = []
    for i in range(len(seq)):
        c1 = contour[seq[i-1]]
        c2 = contour[seq[i]]
        e1, e2 = closest_end(c1, c2, exclude1=ex_dict[seq[i-1]], exclude2=ex_dict[seq[i]])
        if (e1 != -2 and distance(c1[e1], c2[e2]) > threshold):
            print(f'{seq[i-1]}与{seq[i]}之间需要补全')
            complete_pair.append((seq[i-1], seq[i], e1, e2))
            ex_dict[seq[i-1]].append(e1)
            ex_dict[seq[i]].append(e2)

    def complete(pair):
        e1, e2 = pair[2], pair[3]
        v1 = np.array(contour[pair[0]][e1])
        v2 = np.array(tangent[pair[0]][e1])
        v3 = np.array(contour[pair[1]][e2])
        v4 = np.array(tangent[pair[1]][e2])
        if e1 == 0:
            v2 = -v2
        if e2 == 0:
            v4 = -v4
        design_part = take_design_part(v1, v2, v3, user_tck, user_u)
        v2 = v1 - v2 / np.linalg.norm(v2)
        v4 = v3 - v4 / np.linalg.norm(v4)
        res, _, _ = close_curve_for_snap([v1, v2, v4, v3], design_part)
        return res

    curve = []
    if len(seq) == 1:
        if distance(contour[seq[0]][0], contour[seq[0]][-1]) > threshold:
            # 单边补全成环
            complete_pair.append([seq[0], seq[0], -1, 0])
            curve += contour[seq[0]]
            curve += complete(complete_pair[0])
        else:
            # 单边自然成环
            curve += contour[seq[0]]
    else:
        for i in range(len(seq2)):
            pair = (seq2[i-1][0], seq2[i][0], seq2[i-1][1], seq2[i][1])
            if pair[0] == pair[1]:
                curve += contour[seq2[i-1][0]] if seq2[i-1][1] == 0 else list(reversed(contour[seq2[i-1][0]]))
            elif pair in complete_pair:
                curve += complete(pair)
    for p in curve:
        y, x = p[0], p[1]
        oval = canvas.create_oval(x, y, x, y, outline='red', fill='red', width=1)
        ovals.append(oval)

    user_input = []
    print('ok')


def on_mouse_move(event):
    global ovals
    global start_move
    x, y = event.x, event.y
    user_input.append([y, x])
    if start_move == 0:
        start_move = 1
        for oval in ovals:
            canvas.delete(oval)
        ovals = []
    oval = canvas.create_oval(x, y, x, y, outline='red', fill='red', width=5)
    ovals.append(oval)

canvas.focus_set()
canvas.bind('<B1-Motion>', on_mouse_move)
canvas.bind('<Key>', on_key_press)


win.mainloop()