import cv2
import pickle
import os

width = 130
height = 65

save_dir = 'cropped_img'

if not os.path.exists(save_dir):
    os.makedirs(save_dir)

def save_cropped_img(img, pos, index):
    cropped_img = img[pos[1]:pos[1]+height, pos[0]:pos[0]+width]
    save_path = os.path.join(save_dir, f'roi_{index}.png')
    cv2.imwrite(save_path, cropped_img)
    print(f'saved cropped image: {save_path}')


try:
    with open('carposition.pkl', 'rb') as f:
        positionList = pickle.load(f)
except :
    positionList = []

def mouseclick(events, x, y, flags, params):
    if events == cv2.EVENT_LBUTTONDOWN:
        positionList.append((x, y))
        save_cropped_img(cv2.resize(cv2.imread('car1.png'), (1280, 720)), (x, y), len(positionList))
    if events == cv2.EVENT_RBUTTONDOWN:
        for i, pos in enumerate(positionList):
            x1, y1 = pos
            if x1<x<x1+width and y1<y<y1+height:
                positionList.pop(i)
    with open('carposition.pkl', 'wb') as f:
        pickle.dump(positionList, f)

while True:
    image = cv2.imread('car1.png')
    image = cv2.resize(image, (1280, 720))
    for pos in positionList:
        cv2.rectangle(image, pos, (pos[0]+width, pos[1]+height), (255,0,255), 2)
    cv2.imshow("Image", image)
    cv2.setMouseCallback("Image", mouseclick)
    k = cv2.waitKey(1)
    if k==ord('q'):
        break