import numpy as np
import copy
import random
from scipy.interpolate import splprep, splev
from scipy.spatial.distance import directed_hausdorff
from scipy.integrate import quad
import time
from vectorization import close_curve_for_snap
from collections import defaultdict
import sys
import warnings
import win32pipe, win32file
import os
from shapely.geometry import MultiLineString, LineString
import traceback


# 弧长步长
ds = 5

def distance(p1, p2):
    return np.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)
def closest_end(c1, c2, exclude1=[], exclude2=[]):
    d = []
    d.append(distance(c1[0], c2[0]) if 0 not in exclude1 and 0 not in exclude2 else 1e8)
    d.append(distance(c1[0], c2[-1]) if 0 not in exclude1 and -1 not in exclude2 else 1e8)
    d.append(distance(c1[-1], c2[0]) if -1 not in exclude1 and 0 not in exclude2 else 1e8)
    d.append(distance(c1[-1], c2[-1]) if -1 not in exclude1 and -1 not in exclude2 else 1e8)
    i = np.argmin(d)
    if i == 0:
        return 0, 0
    elif i == 1:
        return 0, -1
    elif i == 2:
        return -1, 0
    else:
        return -1, -1

def resample(contour, tangent, length, i, edge):
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

def filter_adj_same_point(edge):
    res = []
    for i, p in enumerate(edge):
        if i > 0 and p == edge[i - 1]:
            continue
        res.append(p)
    return res

def pre(contour):
    contour = [filter_adj_same_point(edge) for edge in contour]
    contour = [edge for edge in contour if len(edge) > 3]
    tangent = [[] for _ in range(len(contour))]
    length = [[] for _ in range(len(contour))]
    user_input = []

    for i, edge in enumerate(contour):
        resample(contour, tangent, length, i, edge)
    contour.append(user_input)
    tangent.append([])
    length.append([])
    return contour, tangent, length

def work(contour, tangent, length, user_input):
    threshold = 15 # 两点距离若小于该阈值则认为是相邻的
    threshold2 = 20 # 两条边的豪斯多夫距离若小于该阈值则认为是接近的

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

    start_move = 0
    contour[-1] = user_input
    resample(contour, tangent, length, len(contour) - 1, contour[-1])
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
    return curve



# 打开命名管道
handle = win32file.CreateFile("\\\\.\\pipe\\MyPipe", 
    win32file.GENERIC_READ | win32file.GENERIC_WRITE,
    0, None, 
    win32file.OPEN_EXISTING,
    0, None)

if handle == win32file.INVALID_HANDLE_VALUE:
    print("CreateFile failed with error", win32api.GetLastError())
    exit(1)

print('连接成功')

# 从管道读取图像的edge数据
_, readData = win32file.ReadFile(handle, 4, None)

# 输出收到的消息
n = int.from_bytes(readData, byteorder='little', signed=True)
print(f'{n}条边')
cur = 1
contour = []
for i in range(n):
    _, readData = win32file.ReadFile(handle, 4, None)
    m = int.from_bytes(readData, byteorder='little', signed=True)
    cur += 1
    edge = []
    for j in range(m):
        _, readData = win32file.ReadFile(handle, 8, None)
        x = int.from_bytes(readData[0:4], byteorder='little', signed=True)
        cur += 1
        y = int.from_bytes(readData[4:8], byteorder='little', signed=True)
        cur += 1
        edge.append([x, y])
    contour.append(edge)
print('ok1')

try:
    contour, tangent, length = pre(contour)
except Exception as e:
    print("pre函数出现了错误：", e)
    traceback.print_exc()
print('ok2')

while True:
    # 从管道读取用户输入轮廓数据
    _, readData = win32file.ReadFile(handle, 4, None)
    m = int.from_bytes(readData, byteorder='little', signed=True)
    cur = 1
    user_input = []
    for j in range(m):
        _, readData = win32file.ReadFile(handle, 8, None)
        x = int.from_bytes(readData[0:4], byteorder='little', signed=True)
        cur += 1
        y = int.from_bytes(readData[4:8], byteorder='little', signed=True)
        cur += 1
        user_input.append([x, y])
    try:
        curve = work(contour, tangent, length, user_input)
    except Exception as e:
        print("work函数出现了错误：", e)
        traceback.print_exc()

        
    res = b''
    res += len(curve).to_bytes(4, byteorder='little', signed=True)
    for i in range(len(curve)):
        x, y = round(curve[i][0]), round(curve[i][1])
        res += x.to_bytes(4, byteorder='little', signed=True)
        res += y.to_bytes(4, byteorder='little', signed=True)

    # import matplotlib.pyplot as plt

    # fig, ax = plt.subplots()
    # H, W = 1000, 1000
    # plt.xlim(0, W)
    # plt.ylim(0, H)
    # for i, edge in enumerate(contour):
    #     x, y = np.array([edge[_][0] for _ in range(len(edge))]), np.array([edge[_][1] for _ in range(len(edge))])
    #     ax.plot(y, H - x)
    #     ax.text(np.mean(y) + 1, np.mean(H - x) + 1, f"{i}", ha="center", va="center", fontsize=12, color="red")
    # ax.plot([p[1] for p in curve], [H - p[0] for p in curve], 'ro')
    # plt.show()

    win32file.WriteFile(handle, res)
    


# 关闭管道句柄
win32file.CloseHandle(handle)