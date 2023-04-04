import cv2
import numpy as np
import random
# 读取图像
stroke = []
with open('case.txt', 'r') as f:
# with open('handle.txt', 'r') as f:
    hw = f.readline()
    lines = f.readlines()
    for line in lines:
        line = line.strip().split()
        line = list(map(int, line))
        stroke.append(line)


img = cv2.imread('out_case.png')
# img = cv2.imread('out_handle.png')
# for p in stroke:
    # img[p[0], p[1]] = [0, 0, 255]

# 将图像转为灰度图
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# 设定Shi-Tomasi角点检测的参数
maxCorners = 100
qualityLevel = 0.01
minDistance = 10
blockSize = 3
k = 0.15
# 进行Shi-Tomasi角点检测
corners = cv2.goodFeaturesToTrack(gray, maxCorners, qualityLevel, minDistance, blockSize, useHarrisDetector=True, k=k)
# 绘制角点
for corner in corners:
    x, y = corner.ravel()
    x = int(x)
    y = int(y)
    # cv2.circle(img, (int(x), int(y)), 3, (0, 0, 255), -1)
    d = 1e8
    idx = 0
    for i, p in enumerate(stroke):
        d2 = np.sqrt((p[0]-y)**2+(p[1]-x)**2) # 注意返回的x,y代表横纵坐标
        if d2 < d:
            d = d2
            idx = i
    img[stroke[idx][0], stroke[idx][1]] = [0, 0, 255]

strokes = []
stroke2 = []
start = 0
for i, p in enumerate(stroke):
    if (img[p[0], p[1]] == [0, 0, 255]).all():
        start = i
        break
start += 1
for i in range(start, start + len(stroke)):
    p = stroke[i % len(stroke)]
    stroke2.append(p)
    if (img[p[0], p[1]] == [0, 0, 255]).all():
        strokes.append(stroke2)
        stroke2 = []

for stroke in strokes:
    b, g, r = random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)
    for i, p in enumerate(stroke):
        img[p[0], p[1]] = [b, g, r]

with open('corner.txt', 'w') as f:
    f.write(hw)
    f.write(f'{len(strokes)}\n')
    for stroke in strokes:
        for p in stroke:
            f.write(f'{p[0]} {p[1]} ')
        f.write('\n')
# 显示结果
cv2.imwrite('Corner_Detection.png', img)
