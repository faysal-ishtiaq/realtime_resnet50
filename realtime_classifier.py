import os
import cv2
import sys
import serial
import threading
import numpy as np
from resnet50 import ResNet50
from imagenet_utils import preprocess_input, decode_predictions

label = ''
frame = None


class MyThread(threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self)

    def run(self):
        global label
        #self.ard = serial.Serial(self.get_serial_port(), baudrate=9600, timeout=3)
        # Load the VGG16 network
        print("[INFO] loading network...")
        self.model = ResNet50(weights="imagenet")

        while (~(frame is None)):
            (inID, label) = self.predict(frame)

    def predict(self, frame):
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.float32)
        image = np.expand_dims(image, axis=0)
        image = preprocess_input(image)
        preds = self.model.predict(image)
        label = decode_predictions(preds)[0][0][0:2]
        #self.write_arduino(label)
        return label

    def write_arduino(self, label):
        str_label = str(label[1]).replace('_', ' ')
        a = bytearray()
        a.extend(str_label.encode())
        self.ard.write(a)

    def get_serial_port(self):
        return "/dev/" + os.popen("dmesg | egrep ttyACM | cut -f3 -d: | tail -n1").read().strip()


cap = cv2.VideoCapture(0)
if (cap.isOpened()):
    print("Camera OK")
else:
    cap.open()

keras_thread = MyThread()
keras_thread.start()

while (True):
    ret, original = cap.read()
    original = cv2.flip(original, 1)
    original = original[100: 540, 40: 440]
    frame = cv2.resize(original, (224, 224))
    cv2.putText(original, "Label: {}".format(label), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    cv2.imshow("Classification", original)

    if (cv2.waitKey(1) & 0xFF == ord('q')):
        break

cap.release()
frame = None
cv2.destroyAllWindows()
sys.exit()