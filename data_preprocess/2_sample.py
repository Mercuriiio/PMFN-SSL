import os
import shutil
import cv2
import numpy as np
import random
from entropy2dSpeedUp import calcEntropy2dSpeedUp

tcga_path = "segment/lusc/"
tcga_files = os.listdir(tcga_path)
dir_N = len(tcga_files)
dir_i = 0
output = 'segment/lusc_slide/entropy_v_s/'
if not os.path.exists(output):
    os.makedirs(output)

for tcga_file in tcga_files:
    dir_i += 1
    TCGA_svs = tcga_path + tcga_file
    files = os.listdir(TCGA_svs)

    var = []
    var_path = []
    image_var = []
    image_path = []
    i = 0
    N = len(files)

    for img in files:
        i += 1
        img_path = TCGA_svs + '/' + img
        print('(', dir_i, '/', dir_N, ')', '(', i, '/', N, ')', img_path)
        image = cv2.imread(img_path)
        image_ = cv2.resize(image, (128, 128), interpolation=cv2.INTER_CUBIC)

        # 直接采用图像信息熵准则
        # image_entropy = cv2.cvtColor(image_, cv2.COLOR_RGB2GRAY)
        # var.append(calcEntropy2dSpeedUp(image_entropy, 3, 3))

        # 采用信息熵+饱和度+明度准则
        image_gray = cv2.cvtColor(image_, cv2.COLOR_RGB2GRAY)
        image_entropy = calcEntropy2dSpeedUp(image_gray, 3, 3)
        image_hsv = cv2.cvtColor(image_, cv2.COLOR_RGB2HSV)
        H, S, V = cv2.split(image_hsv)
        # 明度（V）
        v = V.ravel()[np.flatnonzero(V)]  # 亮度非零的值
        average_v = sum(v) / len(v)
        # 饱和度（S）
        s = S.ravel()[np.flatnonzero(S)]
        if len(s) == 0:
            average_s = 0
        else:
            average_s = sum(s) / len(s)
        var.append(image_entropy**2*np.math.log(1+average_v*average_s, np.math.e)/1000)

        # 采用随机采样准则
        # var.append(random.random())

        var_path.append(img_path)

    if len(var) > 500 and len(var) <= 1000:
        img_select = 50
    elif len(var) > 1000:
        img_select = 100
    elif len(var) > 50 and len(var) <= 500:
        img_select = 30
    else:
        img_select = 1
    print(len(var), img_select)
    for i in range(len(var)):
        number = max(var)
        index = var.index(number)
        print(index, ':', var[index])
        image_path.append(var_path[index])
        image_var.append(var[index])
        var[index] = 0
        if i+1 == img_select:
            break

    # 保存采样图像
    img_out = output + tcga_file + '/'
    if not os.path.exists(img_out):
        os.makedirs(img_out)
    for path in image_path:
        index = image_path.index(path)
        shutil.copy2(path, img_out + str(image_var[index]) + '.png')
