import cv2
import numpy as np

def receptiveField(img, R=3, r=1, fac_r=-1, fac_R=6):
    img1 = np.float32(img)

    x, y = np.meshgrid(np.arange(1, R * 2 + 2), np.arange(1, R * 2 + 2))
    dis = np.sqrt((x - (R + 1)) ** 2 + (y - (R + 1)) ** 2)
    flag1 = (dis <= r)
    flag2 = np.logical_and(dis > r, dis <= R)
    kernal = flag1 * fac_r + flag2 * fac_R
    kernal /= kernal.sum()

    out = cv2.filter2D(img, -1, kernal)
    return out


def SimOTM(img):
    blur = cv2.blur(img, (3, 3))
    rec = receptiveField(img)
    result = cv2.merge([img, blur, rec])
    return result
