import cv2
import numpy as np
from PIL import Image
import time
initial=time.time()
store=[]
IMG='D:\College\Project\src\output2.png'
image = cv2.cvtColor(cv2.imread(IMG), cv2.COLOR_BGR2RGB)
img = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
def sobel(channel):
    sobelX = cv2.Sobel(channel, cv2.CV_16S, 1, 0)
    sobelY = cv2.Sobel(channel, cv2.CV_16S, 0, 1)
    sobel = np.hypot(sobelX, sobelY)
    sobel[sobel > 255] = 255
    return np.uint8(sobel)

def edge_detect(im):
    return np.max(np.array([sobel(im[:,:, 0]), sobel(im[:,:, 1]), sobel(im[:,:, 2]) ]), axis=0)

blurred = cv2.GaussianBlur(image, (5, 5), 18)
edges = edge_detect(blurred)
ret, edges = cv2.threshold(edges, 50, 255, cv2.THRESH_BINARY)
bw_image = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((20,20), np.uint8))


def union(a, b):
    x = min(a[0], b[0])
    y = min(a[1], b[1])
    w = max(a[0] + a[2], b[0] + b[2]) - x
    h = max(a[1] + a[3], b[1] + b[3]) - y
    return [x, y, w, h]


def intersect(a, b):
    x = max(a[0], b[0])
    y = max(a[1], b[1])
    w = min(a[0] + a[2], b[0] + b[2]) - x
    h = min(a[1] + a[3], b[1] + b[3]) - y
    if w < 0 or h < 0:
        return False
    return True


def group_rectangles(rec):
    tested = [False for i in range(len(rec))]
    final = []
    i = 0
    while i < len(rec):
        if not tested[i]:
            j = i + 1
            while j < len(rec):
                if not tested[j] and intersect(rec[i], rec[j]):
                    rec[i] = union(rec[i], rec[j])
                    tested[j] = True
                    j = i
                j += 1
            final += [rec[i]]
        i += 1

    return final


def text_detect(img, original):
    mask = np.zeros(img.shape, np.uint8)
    cnt, hierarchy = cv2.findContours(np.copy(img), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    index = 0
    boxes = []
    while (index >= 0):
        x, y, w, h = cv2.boundingRect(cnt[index])
        cv2.drawContours(mask, cnt, index, (255, 255, 255), cv2.FILLED)
        maskROI = mask[y:y + h, x:x + w]
        r = cv2.countNonZero(maskROI) / (w * h)
        # TODO Test h/w and w/h ratios
        if r > 0.1 and 2000 > w > 10 and 1600 > h > 10 and h / w < 3 and w / h < 10:
            boxes += [[x, y, w, h]]
        index = hierarchy[0][index][0]
    boxes = group_rectangles(boxes)
    bounding_boxes = np.array([0, 0, 0, 0])

    for (x, y, w, h) in boxes:
        s=(image[y:y + h, x:x + w].tolist())
        store.append(s)
        bounding_boxes = np.vstack((bounding_boxes, np.array([x, y, x + w, y + h])))

text_detect(bw_image,image)
store= np.asarray(store)
final=Image.fromarray(np.uint8(img))
final.save("output21.jpg")
final.show()

