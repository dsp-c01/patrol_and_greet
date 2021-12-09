import argparse
import sys
import os

from utils import *
import pyrealsense2 as rs

import math
import time
import cv2
import numpy as np
from age_gender_ssrnet.SSRNET_model import SSR_net_general, SSR_net

import threading
# ======robot package ====== #
import imutils
import speech_recognition as sr
import pyaudio
import wave  # 讀音檔 使用mic可以不用
import requests  # robot library

# ============== face & agneder ================ #

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
parser.add_argument('--src', type=int, default=3,
                    help='source of the camera')
parser.add_argument('--output-dir', type=str, default='outputs/',
                    help='path to the output directory')
args = parser.parse_args()

# print the arguments
print('----- info -----')
print('[i] The config file: ', args.model_cfg)
print('[i] The weights of model file: ', args.model_weights)
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


#####################################################################################
def predictAgeGender(faces):
    # Convert faces to N,64,64,3 blob
    blob = np.empty((len(faces), face_size, face_size, 3))
    # print("faces: ", faces)
    for i, face_bgr in enumerate(faces):
        # print("face_bgr: ", face_bgr)
        blob[i, :, :, :] = cv2.resize(face_bgr, (64, 64))
        blob[i, :, :, :] = cv2.normalize(blob[i, :, :, :], None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    # Predict gender and age
    genders = gender_net.predict(blob)
    ages = age_net.predict(blob)
    #  Construct labels
    labels = ['{},{}'.format('Male' if (gender >= 0.7) else 'Female', int(age)) for (gender, age) in zip(genders, ages)]
    return labels


def collectFaces(frame, face_boxes):
    faces = []
    # Process faces
    # print("collect face_box:", face_boxes)
    for i, box in enumerate(face_boxes):
        # Convert box coordinates from resized frame_bgr back to original frame
        box_orig = [
            int(round(box[0] * width_orig / width)),
            int(round(box[1] * height_orig / height)),
            int(round(box[2] * width_orig / width)),
            int(round(box[3] * height_orig / height)),
        ]
        # print("collect box:", box_orig)
        # Extract face box from original frame
        face_bgr = frame[
                   max(0, box_orig[1]):min(box_orig[3] + 1, height_orig - 1),
                   max(0, box_orig[0]):min(box_orig[2] + 1, width_orig - 1),
                   :
                   ]
        # print("collect face_bgr:", face_bgr)
        faces.append(face_bgr)
    return faces


########################################################################
def agneder_yolo():
    print("start agender recognition")
    global width, height, labels, face, width_orig, height_orig, age, sal, recon_swi
    wind_name = 'face detection using YOLOv3'
    cv2.namedWindow(wind_name, cv2.WINDOW_NORMAL)
    cv2.namedWindow(wind_name, cv2.WINDOW_NORMAL)

    pipe = rs.pipeline()

    align_to = rs.stream.color
    align = rs.align(align_to)

    colorizer = rs.colorizer()

    cfg = rs.config()
    cfg.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    profile = pipe.start(cfg)

    # cap = cv2.VideoCapture(args.src)
    # cap.set(3, 640)
    # cap.set(4, 480)
    try:
        while True:
            frames = pipe.wait_for_frames()
            aligned_frames = align.process(frames)
            frame = np.asanyarray(frames.get_color_frame().get_data())
            aligned_depth_frame = aligned_frames.get_depth_frame()
            color_depth = np.asanyarray(colorizer.colorize(aligned_frames.get_depth_frame()).get_data())
            # has_frame, frame = cap.read()
            # frame_s = cv2.resize(frame, (256, 192),interpolation=cv2.INTER_AREA)
            start_time = time.time()
            ############## initial parameter of agender input type ##################
            height_orig, width_orig = frame.shape[:2]
            #########################################################################
            # start_time_yolo = time.time()
            # Create a 4D blob from a frame.
            blob = cv2.dnn.blobFromImage(frame, 1 / 255, (IMG_WIDTH, IMG_HEIGHT),
                                         [0, 0, 0], 1, crop=False)

            # Sets the input to the network
            net.setInput(blob)

            # Runs the forward pass to get output of the output layers
            outs = net.forward(get_outputs_names(net))

            # Remove the bounding boxes with low confidence
            face = post_process(frame, outs, CONF_THRESHOLD, NMS_THRESHOLD)
            # end_time_yolo = time.time()
            # print("yolo FPS: ", 1/(end_time_yolo-start_time_yolo))
            # print("initial label: ", labels)
            labels = []
            sal = ""
            b = 0
            if len(face) > 0:
                cen = [(face[0][0] + face[0][2])//2, (face[0][1] + face[0][3])//2]
                face[0][0] = 0 if face[0][0] <= 0 else face[0][0] - 20
                face[0][2] = width_orig if face[0][2] >= width_orig else face[0][2] + 20
                face[0][1] = 0 if face[0][1] <= 0 else face[0][1] - 20
                face[0][3] = height_orig if face[0][3] <= 0 else face[0][3] + 20

                for i in range(cen[0] - 10, cen[0] + 10, 1):
                    for j in range(cen[1] - 10, cen[1] + 10, 1):
                        # a += depth_frame.get_distance(i, j)
                        b += aligned_depth_frame.get_distance(i, j)
                #####################################
                # convert to agender input type
                faces = collectFaces(frame, face)
                # Get age and gender
                labels = predictAgeGender(faces)
                # print("label: ", labels)
                if len(labels) > 0:
                    labels_list = labels[0].split(",")
                    gender = labels_list[0]
                    age = int(labels_list[-1])
                    if 0 <= age < 16:
                        sal = "小朋友"
                    elif 16 <= age <= 60:
                        sal = "先生" if gender == "Male" else "小姐"
                    else:
                        sal = "阿伯" if gender == "Male" else "女士"
                for (x1, y1, x2, y2) in face:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color=(0, 255, 0), lineType=8)
                # Draw labels
                for (label, box) in zip(labels, face):
                    cv2.putText(frame, label, org=(box[0], box[1] - 10), fontFace=cv2.FONT_HERSHEY_PLAIN,
                                fontScale=1, color=(0, 64, 255), thickness=1, lineType=cv2.LINE_AA)

            end_time = time.time()
            # initialize the set of information we'll displaying on the frame
            # info = [
            #     ('FPS', '{:.2f}'.format(1/(end_time-start_time)))
            # ]
            #
            # for (i, (txt, val)) in enumerate(info):
            #     text = '{}: {}'.format(txt, val)
            #     cv2.putText(frame, text, (10, (i * 20) + 20),
            #                 cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_RED, 2)
            text1 = "FPS: {:.2f}".format(1 / (end_time - start_time))
            text2 = "distance: {:.2f} m".format(b/400)
            cv2.putText(frame, text1, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(color_depth, text2, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.imshow(wind_name, frame)
            cv2.imshow("depth", color_depth)

            key = cv2.waitKey(1)
            if key == 27 or key == ord('q') or recon_swi == 0:
                print('[i] ==> Interrupted by user!')
                break
    finally:
        # cap.release()
        pipe.stop()
        cv2.destroyAllWindows()

        print('==> All done!')
        print('***********************************************************')


########################################################################
# ======================= navigation =========================== #
def Voice_To_Text(i):
    voice_cmd = {"paraString": '{"time":"2018-08-08T12:00:00Z","requestId":"AAA","action":"start",'
                               '"deviceId":"SPEAKER","content":"請下達指令"}'}
    r = sr.Recognizer()  # 使用sr下的語音辨識class
    print("請開始說話:")
    response_voi = requests.post('http://169.254.246.191:8882/GeosatRobot/api/Device', data=voice_cmd)
    with sr.Microphone() as source:  # 將語音輸入丟到source
        time.sleep(3)
        r.adjust_for_ambient_noise(source)  # 調整麥克風的噪音
        audio = r.listen(source)
    try:
        Text = r.recognize_google(audio, language="zh-TW")  # 使用google語音辨識的api
    except sr.UnknownValueError:
        Text = "無法翻譯"
    except sr.RequestError as e:
        Text = ("無法翻譯{0}".format(e))

    # speech recognition dictionary
    voice_para1 = {"paraString": '{"time":"2018-08-08T12:00:00Z","requestId":"AAA","action":"start",'
                                 '"deviceId":"SPEAKER","content":"好的，阿寶將開始送餐"}'}
    voice_para2 = {"paraString": '{"time":"2018-08-08T12:00:00Z","requestId":"AAA","action":"start",'
                                 '"deviceId":"SPEAKER","content":"不客氣，這是阿寶應該做的"}'}
    voice_para3 = {"paraString": '{"time":"2018-08-08T12:00:00Z","requestId":"AAA","action":"start",'
                                 '"deviceId":"SPEAKER","content":"不好意思，請客人您再說一次"}'}
    voice_para4 = {"paraString": '{"time":"2018-08-08T12:00:00Z","requestId":"AAA","action":"start",'
                                 '"deviceId":"SPEAKER","content":"阿寶正在返回充電站"}'}
    Mic_paras = [voice_para1, voice_para2, voice_para3, voice_para4]

    lim = 0
    # speaker model
    if Text == '繼續送餐':
        print('阿寶準備前往下個送餐地點。')
        voice_mod = Mic_paras[0]
        i += 1
    elif Text == '謝謝':
        print('不客氣，這是阿寶應該做的。')
        voice_mod = Mic_paras[1]
        i += 1
    elif Text == "無法翻譯":
        print('不好意思，這位客人請再說一次。')
        voice_mod = Mic_paras[2]
    elif Text == '停止動作':
        print("阿寶返回充電站")
        voice_mod = Mic_paras[3]
        i = 4
        lim = 1  # trigger
    else:
        voice_mod = Mic_paras[2]
    if i == 4 & lim == 0:
        i = 0
    print("next waypoint: ", i)
    return Text, voice_mod, i


def navigation(url, i):
    print("start navigation")
    global sal, labels, voice_time, face
    print("this navigation id: ", i)
    labels = face = []
    sal = ""
    switch = 1
    nav_para1 = {
        "paraString": '{"time":"2018-08-08T12:00:00Z","requestId":"AAA","action":"start","deviceId":"NAVIGATION", "content":"A"}'}

    nav_para2 = {
        "paraString": '{"time":"2018-08-08T12:00:00Z","requestId":"AAA","action":"start","deviceId":"NAVIGATION", "content":"B"}'}

    nav_para3 = {
        "paraString": '{"time":"2018-08-08T12:00:00Z","requestId":"AAA","action":"start","deviceId":"NAVIGATION", "content":"C"}'}

    nav_para4 = {
        "paraString": '{"time":"2018-08-08T12:00:00Z","requestId":"AAA","action":"start","deviceId":"NAVIGATION", "content":"D"}'}

    nav_para5 = {
        "paraString": '{"time":"2018-08-08T12:00:00Z","requestId":"AAA","action":"start","deviceId":"BACKTODOCK",}'}

    nav_para6 = {
        "paraString": '{"time":"2018-08-08T12:00:00Z","requestId":"AAA","action":"start","deviceId":"NAVIGATIONSTATUS"}'}

    battery_ = {"paraString": '{"time":"2018-08-08T12:00:00Z","requestId":"AAA","action":"start","deviceId":"BATTERY"}'}
    Nav_paras = [nav_para1, nav_para2, nav_para3, nav_para4, nav_para5, nav_para6]

    response_wp = requests.post(url, data=Nav_paras[i])
    while True:
        response_rp = requests.post(url, data=Nav_paras[5])
        na_end_time = time.time()
        if len(labels) > 0 and len(sal) > 0 and na_end_time - voice_time > 5:
            print("導航打招呼 ==> label: {}, gender: {}".format(labels, sal))
            voice_cmd = {"paraString": '{"time":"2018-08-08T12:00:00Z","requestId":"AAA","action":"start",'
                                       '"deviceId":"SPEAKER","content":"' + sal + '你好"}'}
            response_voi = requests.post(url, data=voice_cmd)
            voice_time = time.time()
            labels = []
            sal = ""
        if 'COMPLETED' in response_rp.text and i != 4:
            print("=====已抵達" + str(i) + "點=====")
            break
        if i == 4:
            print("move to dock.")
            response_wp = requests.post(url, data=Nav_paras[4])
            while True:
                print("check battery loop.")
                response_battery = requests.post(url, data=battery_)
                if "Discharging" not in response_battery.text:
                    voice_cmd = {"paraString": '{"time":"2018-08-08T12:00:00Z","requestId":"AAA","action":"start",'
                                               '"deviceId":"SPEAKER","content":"阿寶正在充電"}'}
                    switch = 0
                    break
            response_voi = requests.post(url, data=voice_cmd)
            break
    print("report switch: ", switch)
    return switch


def rotation(url, rv):
    ro_right = {
        "paraString": '{"time":"2018-08-08T12:00:00Z","requestId":"AAA","action":"start","deviceId":"ROBOTBODYROTATE","content":"-2/0.4"}'}
    ro_left = {
        "paraString": '{"time":"2018-08-08T12:00:00Z","requestId":"AAA","action":"start","deviceId":"ROBOTBODYROTATE","content":"2/0.4"}'}
    rt = [ro_right, ro_left]
    ro = rt[0] if rv > 0 else rt[1]
    response_ro = requests.post(url, data=ro)


def rt_move(url):
    move = {
        "paraString": '{"time":"2018-08-08T12:00:00Z","requestId":"AAA","action":"start","deviceId":"ROBOTBODYMOVE","content":"0.1/0.1"}'}
    response_wv = requests.post(url, data=move)


def rt(url):
    global labels, face, width_orig, height_orig, sal, voice_time
    labels = face = []
    sal = ""
    while True:
        print("進入rotaion loop.")
        if face:
            if len(face[0]) == 4:
                face_rt = face
                area = (face_rt[0][2] - face_rt[0][0]) * (face_rt[0][3] - face_rt[0][1])
                print("Area: ", area)
                print("face: ", face_rt)
                rx = (face_rt[0][0] + face_rt[0][2]) // 2 - width_orig / 2
                ry = (face_rt[0][1] + face_rt[0][3]) // 2 - height_orig / 2
                re_center = (rx, ry)
                print("residual value: ", re_center)
                if rx > 120:
                    rotation(url, rx)
                elif rx < -120:
                    rotation(url, rx)
                if 120 >= rx >= -120:
                    if ry > 50:
                        rt_move(url)
                    else:
                        print("rotation ==> label: {}, gender: {}".format(labels, sal))
                        rt_end_time = time.time()
                        if len(labels) > 0 and len(sal) > 0 and rt_end_time - voice_time > 5:
                            print("旋轉打招呼")
                            voice_cmd = {"paraString": '{"time":"2018-08-08T12:00:00Z","requestId":"AAA","action":"start",'
                                                       '"deviceId":"SPEAKER","content":"' + sal + '你好"}'}
                            response_voi = requests.post(url, data=voice_cmd)
                            voice_time = time.time()
                            labels = []
                            sal = ""
                            break
        else:
            rotation(url, 1)
            sal = ""


def main():
    global recon_swi
    time.sleep(5)
    print("start main")
    url = 'http://169.254.246.191:8882/GeosatRobot/api/Device'
    i = 0
    switch = navigation(url, i)
    rt(url)
    # switch = 1    # loop initial value
    # response_voi = requests.post(url, data=voice_cmd)
    while True:
        text, voice_cmd, i = Voice_To_Text(i)
        print("Text: ", text)
        if text == '無法翻譯':
            print("無法辨識")
            # response_voi = requests.post(url, data=voice_cmd)
            # time.sleep(2)
        elif "你好" in text:
            print("1st.")
            print("voi: ", voice_cmd)
            response_voi = requests.post(url, data=voice_cmd)
            time.sleep(1)
        else:
            print("2nd.")
            print("voi: ", voice_cmd)
            response_voi = requests.post(url, data=voice_cmd)
            print("else_i: ", i)
            switch = navigation(url, i)
            if switch == 0:
                print("switch=0 break")
                break
            rt(url)
    recon_swi = 0


def multi_threading():
    t1 = threading.Thread(target=agneder_yolo)
    t2 = threading.Thread(target=main)

    t1.start()
    t2.start()

    t1.join()
    t2.join()

    print("threading end")


if __name__ == "__main__":
    # main()
    voice_time = time.time()
    face_boxes = []
    labels = []
    sal = ""
    recon_swi = 1
    width = 640
    height = 480
    width_orig = 640
    height_orig = 480
    center_x = 0
    center_y = 0
    # app.run(threaded=True)
    multi_threading()
    # agneder_yolo()
    print("whole program is finish!")
