# import cv2
# import socket
# import struct
# import pickle
#
# # client
# client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
# client_socket.connect(('192.168.0.193', 7777))
# encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]
#
# # server
# TCP_IP = "192.168.0.130"
# TCP_PORT = 8081
# BUFFER_SIZE = 2048  # 1024
# s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
# s.bind((TCP_IP, TCP_PORT))  # connect
# s.listen(10)
#
# # 選擇第二隻攝影機
# cap = cv2.VideoCapture(3)
#
# while True:
#     # 從攝影機擷取一張影像
#     ret, images = cap.read()
#     result, frame = cv2.imencode(".png", images, encode_param)
#     data = pickle.dumps(frame, 0)
#     size = len(data)
#     print("length: ", size)
#     client_socket.sendall(struct.pack(">L", size) + data)
#
#     conn, addr = s.accept()
#     data = conn.recv(BUFFER_SIZE)
#     data = pickle.loads(data)
#     print("data msg: ", data)
#
# # 顯示圖片
#     cv2.imshow('frame', images)
#
# # 若按下 q 鍵則離開迴圈
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
#
# # 釋放攝影機
# cap.release()
# # 關閉所有 OpenCV 視窗
# cv2.destroyAllWindows()
import requests   # robot library
import time
url = 'http://169.254.246.191:8882/GeosatRobot/api/Device'
nav_para = {"paraString": '{"time":"2018-08-08T12:00:00Z","requestId":"AAA","action":"start","deviceId":"BACKTODOCK",}'}
nav_para1 = {"paraString": '{"time":"2018-08-08T12:00:00Z","requestId":"AAA","action":"start","deviceId":"NAVIGATION", "content":"A"}'}
nav_para2 = {"paraString": '{"time":"2018-08-08T12:00:00Z","requestId":"AAA","action":"start","deviceId":"NAVIGATIONSTATUS"}'}
move = {"paraString": '{"time":"2018-08-08T12:00:00Z","requestId":"AAA","action":"start","deviceId":"ROBOTBODYMOVE","content":"0.1/0.1"}'}
ro_left = {"paraString": '{"time":"2018-08-08T12:00:00Z","requestId":"AAA","action":"start","deviceId":"ROBOTBODYROTATE","content":"1/0.2"}'}
voice_cmd = {"paraString": '{"time":"2018-08-08T12:00:00Z","requestId":"AAA","action":"start",'
                               '"deviceId":"SPEAKER","content":"請下達指令"}'}
# a=10
# b=20
# ro_init = {
#         "paraString": '{"time":"2018-08-08T12:00:00Z","requestId":"AAA","action":"start","deviceId":"ROBOTBODYROTATE","content":"' + str(
#             -(a-b)) + '/0.2"}'}
_ = requests.post(url, data=voice_cmd)
# a = time.time()
# print(int(a))
# print(int(a)%3)
# for i in range(3):
# print("未充電") if "Discharging" not in res1.text else print("充電中")
# while True:
# res1 = requests.post(url, data=move)
# print(res1.text)
# while True:
#     res2 = requests.post(url, data=nav_para2)
#     if 'COMPLETED' not in res2.text:
#         print(res2.text)
#     else:
#         print("completed.")
#         break
