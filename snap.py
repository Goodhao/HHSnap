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
from datetime import datetime
from vectorization import close_curve_for_snap
from collections import defaultdict
import sys
import warnings

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

contour = []
tangent = []
pixel_contour = []
user_input = []
# with open('Lpoint.txt') as f:
filename = sys.argv[1] if len(sys.argv) > 1 else 'corner.txt'
with open(filename, 'r') as f:
    hw = f.readline().strip().split()
    hw = [int(_) for _ in hw]
    H, W = hw[0], hw[1]
    img = np.zeros((H, W, 3), dtype=np.uint8)
    n = int(f.readline().strip())
    for i in range(n):
        data = f.readline().strip().split()
        data = [int(_) for _ in data]
        edge = [[data[i], data[i+1]] for i in range(0, len(data), 2)]
        if len(edge) > 0:
            contour.append(edge)
            tangent.append([])
        b, g, r = random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)
        for j in range(0, len(data), 2):
            x, y = data[j], data[j + 1]
            img[x, y, :] = [b, g, r]
pixel_contour = copy.deepcopy(contour)


win = Tk()
win.geometry(f'{W}x{H}')
canvas = Canvas(win, width=W, height=H)
canvas.pack()
bg = ImageTk.PhotoImage(Image.fromarray(img))
canvas.create_image(W/2, H/2, image=bg, anchor='center')
canvas_img_id = canvas.find_all()[0]



# 弧长步长
ds = 5

def bisection(start, tck):
    def integrand(u):
        # 求解曲线在u处的导数
        dxdu, dydu = splev(u, tck, der=1)
        # 返回导数的模长
        return np.sqrt(dxdu ** 2 + dydu ** 2)

    l = start
    r = 1
    while l <= r and abs(r - l) > 1e-7:
        x = (l + r) / 2
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            total_length, _ = quad(integrand, start, x)
        if abs(total_length - ds) < 1e-3:
            return x
        if total_length > ds:
            r = x
        else:
            l = x
    return -1

def resample(i, edge):
    global contour
    global curvature

    if len(edge) == 0:
        return
    tck, u = splprep(np.array(edge).T, k=3, s=100)
    now = 0
    t = [now]
    while True:
        now = bisection(now, tck)
        if now == -1:
            break
        t.append(now)
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
        return -1, -1e3
    l = len(contour[idx])
    dd = []
    for i in range(len(contour[-1])):
        dd.append(directed_hausdorff(contour[idx], get_sublist(contour[-1], i, l))[0])
    assert len(dd) > 0
    return np.argmin(dd), np.min(dd)

for i, edge in enumerate(contour):
    resample(i, edge)

threshold = 15
adj_list = adjacency_list(contour, threshold)

for i in range(len(adj_list)):
    adj_list[i] = list(set(adj_list[i]))
    print(i, adj_list[i])

contour.append(user_input) # 加入空的user_input占位
tangent.append([])
curvature = [[] for i in range(len(contour))]

fig, ax = plt.subplots()
plt.xlim(0, W)
plt.ylim(0, H)
for i, edge in enumerate(contour):
    x, y = np.array([edge[_][0] for _ in range(len(edge))]), np.array([edge[_][1] for _ in range(len(edge))])
    ax.plot(y, H - x)
    ax.text(np.mean(y) + 1, np.mean(H - x) + 1, f"{i}", ha="center", va="center", fontsize=12, color="red")
plt.show()

complete_pair = []

    

start_move = 0
ovals = []
def on_key_press(event):
    global user_input
    global contour
    global ovals
    global start_move
    global complete_pair

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
    resample(len(contour) - 1, contour[-1])
    user_tck, user_u = splprep(np.array(user_input).T, k=3, s=100)
    ss = []
    dd = []
    for i in range(len(contour) - 1):
        s, d_hd = match(i)
        ss.append(s)
        dd.append(d_hd)

    res = []
    for i in range(len(dd)):
        if dd[i] != -1e3 and dd[i] < 20:
            res.append(i)
            for p in pixel_contour[i]:
                y, x = p[0], p[1]
                oval = canvas.create_oval(x, y, x, y, outline='red', fill='red', width=1)
                ovals.append(oval)
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
    ex_dict = defaultdict(list)
    for i in range(len(seq)):
        c1 = contour[seq[i]]
        c2 = contour[seq[i-1]]
        e1, e2 = closest_end(c1, c2)
        if (distance(c1[e1], c2[e2]) <= threshold):
            ex_dict[seq[i]].append(e1)
            ex_dict[seq[i-1]].append(e2)
    for i in range(len(seq)):
        c1 = contour[seq[i]]
        c2 = contour[seq[i-1]]
        e1, e2 = closest_end(c1, c2)
        if (distance(c1[e1], c2[e2]) > threshold):
            print(f'{seq[i]}与{seq[i-1]}之间需要补全')
            e1, e2 = closest_end(c1, c2, exclude1=ex_dict[seq[i]], exclude2=ex_dict[seq[i-1]])
            complete_pair.append([seq[i], seq[i-1], e1, e2])
    if len(seq) == 1 and distance(contour[seq[0]][0], contour[seq[0]][-1]) > threshold:
        # 单边补全成环
        complete_pair.append([seq[0], seq[0], 0, -1])
    design_part = []
    for pair in complete_pair:
        flag = False
        if pair[0] != pair[1]:
            for pair2 in complete_pair:
                if pair2[1] == pair[0] and pair2[0] == pair[1]:
                    flag = True
        if flag:
            # 如果交换后的pair也存在，那么说明这两条边在两个端点处都相邻，要补全两条边
            if pair[0] < pair[1]:
                # 确保只计算一次
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
                for p in res:
                    y, x = p[0], p[1]
                    oval = canvas.create_oval(x, y, x, y, outline='red', fill='red', width=1)
                    ovals.append(oval)
                e1 = -1 - e1
                e2 = -1 - e2
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
                for p in res:
                    y, x = p[0], p[1]
                    oval = canvas.create_oval(x, y, x, y, outline='red', fill='red', width=1)
                    ovals.append(oval)

        else:
            e1, e2 = pair[2], pair[3]
            v1 = np.array(contour[pair[0]][e1])
            v2 = np.array(tangent[pair[0]][e1])
            v3 = np.array(contour[pair[1]][e2])
            v4 = np.array(tangent[pair[1]][e2])
            fig, ax = plt.subplots()
            if e1 == 0:
                v2 = -v2
            if e2 == 0:
                v4 = -v4
            design_part = take_design_part(v1, v2, v3, user_tck, user_u)
            plt.xlim(0, W)
            plt.ylim(0, H)
            for i, edge in enumerate(contour):
                x, y = np.array([edge[_][0] for _ in range(len(edge))]), np.array([edge[_][1] for _ in range(len(edge))])
                ax.plot(y, H - x)
                ax.plot([pp[1] for pp in design_part], [H - pp[0] for pp in design_part], 'ro')
                ax.plot(v2[1], H - v2[0], 'ro')
            plt.show()
            v2 = v1 - v2 / np.linalg.norm(v2)
            v4 = v3 - v4 / np.linalg.norm(v4)
            res, _, _ = close_curve_for_snap([v1, v2, v4, v3], design_part)
            for p in res:
                y, x = p[0], p[1]
                oval = canvas.create_oval(x, y, x, y, outline='red', fill='red', width=1)
                ovals.append(oval)


    user_input = []
    complete_pair = []
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