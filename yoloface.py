# *******************************************************************
#
# Author : Thanh Nguyen, 2018
# Email  : sthanhng@gmail.com
# Github : https://github.com/sthanhng
#
# BAP, AI Team
# Face detection using the YOLOv3 algorithm
#
# Description : yoloface.py
# The main code of the Face detection using the YOLOv3 algorithm
#
# *******************************************************************

# Usage example:  python yoloface.py --image samples/outside_000001.jpg \
#                                    --output-dir outputs/
#                 python yoloface.py --video samples/subway.mp4 \
#                                    --output-dir outputs/
#                 python yoloface.py --src 1 --output-dir outputs/


import argparse
import sys
import os

from utils import *

import math
import time
import cv2
import numpy as np
from age_gender_ssrnet.SSRNET_model import SSR_net_general, SSR_net

#####################################################################
parser = argparse.ArgumentParser()
parser.add_argument('--model-cfg', type=str, default='./models/face-yolov3-tiny.cfg',
                    help='path to config file')
parser.add_argument('--model-weights', type=str,
                    default='./models/face-yolov3-tiny_41000.weights',
                    help='path to weights of model')
parser.add_argument('--image', type=str, default='',
                    help='path to image file')
parser.add_argument('--video', type=str, default='',
                    help='path to video file')
parser.add_argument('--src', type=int, default=0,
                    help='source of the camera')
parser.add_argument('--output-dir', type=str, default='outputs/',
                    help='path to the output directory')
args = parser.parse_args()

#####################################################################
# print the arguments
print('----- info -----')
print('[i] The config file: ', args.model_cfg)
print('[i] The weights of model file: ', args.model_weights)
print('[i] Path to image file: ', args.image)
print('[i] Path to video file: ', args.video)
print('###########################################################\n')

# Give the configuration and weight files for the model and load the network
# using them.
net = cv2.dnn.readNetFromDarknet(args.model_cfg, args.model_weights)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
######################## Agender model parameter ##################################
# Setup global parameters
face_size = 64
face_padding_ratio = 0.10
# Default parameters for SSR-Net
stage_num = [3, 3, 3]
lambda_local = 1
lambda_d = 1
# Initialize gender net
gender_net = SSR_net_general(face_size, stage_num, lambda_local, lambda_d)()
gender_net.load_weights('age_gender_ssrnet/ssrnet_gender_3_3_3_64_1.0_1.0.h5')
# Initialize age net
age_net = SSR_net(face_size, stage_num, lambda_local, lambda_d)()
age_net.load_weights('age_gender_ssrnet/ssrnet_age_3_3_3_64_1.0_1.0.h5')
################ from agender #######################
def predictAgeGender(faces):
    # Convert faces to N,64,64,3 blob
    blob = np.empty((len(faces), face_size, face_size, 3))
    for i, face_bgr in enumerate(faces):
        blob[i, :, :, :] = cv2.resize(face_bgr, (64, 64))
        blob[i, :, :, :] = cv2.normalize(blob[i, :, :, :], None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    # Predict gender and age
    genders = gender_net.predict(blob)
    ages = age_net.predict(blob)
    #  Construct labels
    labels = ['{},{}'.format('Male' if (gender >= 0.5) else 'Female', int(age)) for (gender, age) in zip(genders, ages)]
    return labels

def collectFaces(frame, face_boxes):
    faces = []
    # Process faces
    for i, box in enumerate(face_boxes):
        # Convert box coordinates from resized frame_bgr back to original frame
        box_orig = [
            int(round(box[0] * width_orig / width)),
            int(round(box[1] * height_orig / height)),
            int(round(box[2] * width_orig / width)),
            int(round(box[3] * height_orig / height)),
        ]
        # Extract face box from original frame
        face_bgr = frame[
            max(0, box_orig[1]):min(box_orig[3] + 1, height_orig - 1),
            max(0, box_orig[0]):min(box_orig[2] + 1, width_orig - 1),
            :
        ]
        faces.append(face_bgr)
    return faces
########################################################################

def _main():
    global width, height, height_orig, width_orig
    wind_name = 'face detection using YOLOv3'
    cv2.namedWindow(wind_name, cv2.WINDOW_NORMAL)

    cap = cv2.VideoCapture(args.src)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
    cap.set(cv2.CAP_PROP_XI_HEIGHT, 240)

    while True:
        has_frame, frame = cap.read()
        start_time = time.time()
        # source = frame.copy()
        ############## initial parameter of agender input type ##################
        height_orig, width_orig = frame.shape[:2]
        area = width * height
        width = int(math.sqrt(area * width_orig / height_orig))
        height = int(math.sqrt(area * height_orig / width_orig))
        #########################################################################

        # Stop the program if reached end of video
        if not has_frame:
            print('[i] ==> Done processing!!!')
            print('[i] ==> Output file is stored at', os.path.join(args.output_dir, output_file))
            cv2.waitKey(1000)
            break

        # Create a 4D blob from a frame.
        blob = cv2.dnn.blobFromImage(frame, 1 / 255, (IMG_WIDTH, IMG_HEIGHT),
                                     [0, 0, 0], 1, crop=False)

        # Sets the input to the network
        net.setInput(blob)

        # Runs the forward pass to get output of the output layers
        outs = net.forward(get_outputs_names(net))

        # Remove the bounding boxes with low confidence
        faces = post_process(frame, outs, CONF_THRESHOLD, NMS_THRESHOLD)
        if len(faces) > 0:
            #####################################
            # convert to agender input type
            face = collectFaces(frame, faces)
            # Get age and gender
            labels = predictAgeGender(face)
            for (x1, y1, x2, y2) in faces:
                cv2.rectangle(frame, (x1, y1), (x2, y2), color=(0, 255, 0), lineType=8)
            # Draw labels
            for (label, box) in zip(labels, faces):
                cv2.putText(frame, label, org=(box[0], box[1] - 10), fontFace=cv2.FONT_HERSHEY_PLAIN,
                           fontScale=1, color=COLOR_BLUE, thickness=1, lineType=cv2.LINE_AA)
            ######################################
            # source = source[faces[0][1]-20:faces[0][3]+20, faces[0][0]-20:faces[0][2]+20]
        print('[i] ==> # detected faces: {}'.format(len(faces)))
        print('#' * 60)
        end_time = time.time()
        # initialize the set of information we'll displaying on the frame
        info = [
            ('FPS', '{:.2f}'.format(1/(end_time-start_time)))
        ]

        for (i, (txt, val)) in enumerate(info):
            text = '{}: {}'.format(txt, val)
            cv2.putText(frame, text, (5, (i * 20) + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_RED, 2)

        # cv2.imshow("source", source)

        cv2.imshow(wind_name, frame)


        key = cv2.waitKey(1)
        if key == 27 or key == ord('q'):
            print('[i] ==> Interrupted by user!')
            break

    cap.release()
    cv2.destroyAllWindows()

    print('==> All done!')
    print('***********************************************************')


if __name__ == '__main__':
    width = 480
    height = 340
    _main()
