

import cv2

from model import Model





if __name__ == '__main__':


    capture = cv2.VideoCapture(0)
    ret, frame = capture.read()


    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    model6 = Model()

    label = model6.predict(image=frame)['label']
    print('predicted label: ', label)
    print(model6.predict(image=frame)['confidence'])
    cv2.imshow(label.title(), frame)
    cv2.waitKey(0)
