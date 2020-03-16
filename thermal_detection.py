import cv2
import sys
import time
import numpy as np
import math
import threading
import pyqrcode
import qrcode
import base64
import RPi.GPIO as GPIO
import random
from screeninfo import get_monitors
#  mlx 90614 module
import board
import busio as io
import adafruit_mlx90614

class CameraSensor(object):
    def __init__(self):
        self.cascade_path = None
        self.faceCascade = None

    def set_face_cascade(self, path):
        self.cascade_path = path
        self.faceCascade = cv2.CascadeClassifier(path)

    def get_faces(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        if self.faceCascade is None:
            return []

        faces = self.faceCascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=2,
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE)
        return faces

    
    def set_sensor_rect(self, video_cap, rate=[0.5, 0.7], point=[0.5, 0.4]):
        frame_width = video_cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        frame_height = video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        fps = video_cap.get(cv2.CAP_PROP_FPS)

        rect_w = int(frame_width * rate[0])
        rect_h = int(frame_height * rate[1])

        rect_top = int(frame_height*point[1] - rect_h/2)
        rect_left = int(frame_width*point[0] - rect_w/2)
        return (rect_top, rect_left, rect_w, rect_h)

    def is_face_in_sensor(self, face_rect, sensor_rect):
        (fx, fy, fw, fh) = face_rect
        (sx, sy, sw, sh) = sensor_rect
        fcenter = [int(fx+fw/2), int(fy+fh/2)]
        scenter = [int(sx+sw/2), int(sy+sh/2)]

        ucl_distance = int(math.sqrt(math.pow(scenter[0]-fcenter[0], 2) + math.pow(scenter[1]-fcenter[1],2)))

        if ucl_distance < int(sh/2 * 0.5): # this is the distance between the center of yellow rect and the center of face rect
            return True
        else:
            return False

    def image_add(self, cam_img, thermal_img, face_rect, alpha=0.5):
        cw = cam_img.shape[1]
        ch = cam_img.shape[0]
        (fx, fy, fw, fh) = face_rect
        therm = cv2.resize(thermal_img, (fw, fh), interpolation=cv2.INTER_AREA)
        cam_img[fy:fy+fh, fx:fx+fw] = therm
        return cam_img

THERMAL_STATE_NO_DETECTION = 0
THERMAL_STATE_DETECTING    = 1
THERMAL_STATE_DETECTED     = 2

green = 27
red = 17
blue = 22
class ThermalSensor(object):
    def onled(self, color):
        #GPIO.cleanup()
        GPIO.setmode(GPIO.BCM)
        GPIO.setwarnings(False)
        #GPIO.setup(red,GPIO.OUT)
        #GPIO.setup(blue, GPIO.OUT)
        GPIO.setup(color, GPIO.OUT)
        #print "Red LED on"
        GPIO.output(color,GPIO.HIGH)
        #time.sleep(10)
        #print "LED off"
        #GPIO.output(color,GPIO.LOW)
        
        return 
    def __init__(self):
        self.thermal_stat = True
        self.back_temper_flag = False
        self.no_face_detection = True
        self.detection_state = THERMAL_STATE_NO_DETECTION
        self.body_temp = -1000
        self.back_temper = -1000
        i2c = io.I2C(board.SCL, board.SDA, frequency=100000)
        self.mlx = adafruit_mlx90614.MLX90614(i2c)
        self.sensor_thread = threading.Thread(target=self.detect_thread_func)
        #self.sensor_thread = threading.Thread(target=self.threadfunc)
        self.sensor_thread.start()
        
    def set_face_state(self, flag):
       self.no_face_detection = flag 
       
    def get_face_temper(self):
        #if self.back_temper_flag == False:
        #    print ("no background temperature!")
        #    return False, None
        if self.detection_state == THERMAL_STATE_NO_DETECTION:
            self.detection_state = THERMAL_STATE_DETECTING
        if self.detection_state == THERMAL_STATE_DETECTING:
            pass
        if self.detection_state == THERMAL_STATE_DETECTED:
            self.detection_state = THERMAL_STATE_NO_DETECTION
            return THERMAL_STATE_DETECTED, self.body_temp
        
        return self.detection_state, self.body_temp
        
        amb_temp = self.mlx.ambient_temperature
        obj_temp = self.mlx.object_temperature
        
    def terminate_sensor(self):
        self.thermal_stat = False
        time.sleep(5)
        self.sensor_thread.join()
        
    def detect_thread_func(self):
        cnt = 0
        while True:
            if self.detection_state == THERMAL_STATE_DETECTED:
                time.sleep(0.1)
                continue
            if self.no_face_detection == False:  # if face
                temp_sum = 0.0
                if self.detection_state == THERMAL_STATE_DETECTING:
                    while cnt < 6:# getting background temperature for 2 seconds 
                        vv = self.mlx.object_temperature
                        print (vv)
                        temp_sum += self.mlx.object_temperature
                        time.sleep(0.1)
                        cnt += 1
                        if self.no_face_detection == True:
                            break
                if cnt < 6:
                    self.detection_state = THERMAL_STATE_NO_DETECTION
                if cnt >= 5:
                    addnum=random.uniform(4.1,5.5)
                    self.body_temp = temp_sum/cnt + addnum
                    self.detection_state = THERMAL_STATE_DETECTED
            
            time.sleep(0.1)   
            cnt = 0
        return
    
    def threadfunc(self):
        cnt = 0
        
        while self.thermal_stat:
            if self.no_face_detection == True:
                temp_sum = 0.0
                while cnt < 30:# getting background temperature for 3 seconds 
                    vv = self.mlx.object_temperature
                    print (vv)
                    temp_sum += self.mlx.object_temperature
                    time.sleep(0.1)
                    cnt += 1
                    if self.no_face_detection == False:
                        break
                if cnt < 50:
                    back_temper_flag = False
                    break
                else:
                    #addnum=random.uniform(4.1,5.5)
                    self.back_temper = temp_sum/cnt 
                    print("return count****", cnt, temp_sum)
                    self.back_temper_flag = True
            else:  # can't measure background temper because of face
                print("Getting Background Temperature! Do not put any face in screen!")
                self.back_temper_flag = False

            if self.back_temper_flag == False:
                cnt = 0
                continue
            time.sleep(600)   # every 600 seconds
            cnt = 0
        return

QRCODE_SHOW_FRAME_COUNT = 100
ALERT_SHOW_FRAME_COUNT = 70   # this is a frame count that show  during  warnning msg
    
MSG_NO_FACE = "Please detect your body temperature!"
MSG_SMALL_FACE = "Please get close to the camera!"
MSG_OK_FACE = "Detecting..."
MSG_SCAN_QR = "Please scam the QR Code for fill in the health form"

WARRING_CONTACT_HOSPITAL = "Please contact our reception!"

def main():    
    save_flag = False
    alert_flag = False
    alert_frame_cnt = 0
    cam_sensor = CameraSensor()
    thermal_sensor = ThermalSensor()
    cascPath = "/home/pi/face_rect/haarcascade_frontalface_alt.xml"
    cam_sensor.set_face_cascade(cascPath)

    #video_capture = cv2.VideoCapture('1.mp4')
    video_capture = cv2.VideoCapture(0)
    #video_capture.set(10,65)
    #video_capture.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0)
    #cv2.VideoCapture.set(17,3)
    
    if True: #show full screen
        cv2.namedWindow('Video', cv2.WINDOW_NORMAL)
        cv2.setWindowProperty('Video', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                
    thermal_img = cv2.imread('thermal_back.png')

    ry, rx, rw, rh = cam_sensor.set_sensor_rect(video_capture)  
    sensor_rect = (rx, ry, rw, rh)

    
    qrimg_show_cnt = 0
    alert_show_cnt = 0
    qr_img = None
    temp_str = ''
    
    while True:
        # Capture frame-by-frame
        ret, frame = video_capture.read()
        bottom_msg = ''
          
        # Display the resulting frame
        if ret: 
            if alert_flag == True:
                bottom_msg = WARRING_CONTACT_HOSPITAL
                alert_show_cnt += 1
                if alert_show_cnt > ALERT_SHOW_FRAME_COUNT:
                    alert_show_cnt = 0
                    alert_flag = False
                    temp_str = ''
                    GPIO.cleanup() 
            else:
                if qr_img is None or qrimg_show_cnt == 0:
                    bottom_msg = MSG_NO_FACE
                    #time.sleep(0.5)
                    end_flag = False
                    #frame =cv2.flip(frame,-1)
                    faces = cam_sensor.get_faces(frame)
                    cv2.rectangle(frame, (rx,ry), (rx+rw, ry+rh), (0, 255, 255), 2) # out

                    if len(faces) == 0:                    
                        thermal_sensor.set_face_state(True)
                    else:
                        thermal_sensor.set_face_state(False)

                    print("face counet:", len(faces))
                    for face in faces:                
                        is_in_sensor = cam_sensor.is_face_in_sensor(face, sensor_rect)
                        (x, y, w, h) = face

                        if is_in_sensor == False or rw*0.4 > w:      # sensor_rect_width*0.5 > face rect width
                            if is_in_sensor == True:
                                bottom_msg = MSG_SMALL_FACE
                            continue
                        
                        ret, value = thermal_sensor.get_face_temper()
                    
                        print ("state of thermal:", ret)
                        bottom_msg = MSG_OK_FACE
                        if ret == THERMAL_STATE_DETECTING:
                            bottom_msg = MSG_OK_FACE   #show detecting
                        
                        if value < 36.6 and ret == THERMAL_STATE_DETECTED:     # this is temperature of face                    
                            cv2.putText(frame, str(value), (10,10), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255), 1)
                            qr = qrcode.QRCode(
                                version = 1,
                                error_correction = qrcode.constants.ERROR_CORRECT_H,
                                box_size = 15,
                                border = 6,
                            )
                            #temp_str = str(value) #str(34.444)
                            temp_str = "{:10.1f}".format(value)
                            encodedBytes = base64.b64encode(temp_str.encode("utf-8"))
                            encodedStr = str(encodedBytes, "utf-8")
                            qrdata = 'http://203.174.35.118/register_customer/'+encodedStr+'/china'
                            qr.add_data(qrdata)
                            qr.make(fit=True)
                            img = qr.make_image(fill_color="black", back_color="white")
                            
                            img.save('/home/pi/face_rect/qrimage/a.png')
                            qr_img = cv2.imread('/home/pi/face_rect/qrimage/a.png')   
                            qrimg_show_cnt += 1
                            frame = cam_sensor.image_add(frame, qr_img, sensor_rect, alpha=1.0)
                            
                            thermal_sensor.onled(green)
                        elif value > 36.6 and ret == THERMAL_STATE_DETECTED:
                            bottom_msg = WARRING_CONTACT_HOSPITAL
                            temp_str = "{:10.1f}".format(value)
                            alert_flag = True
                            alert_show_cnt += 1
                            thermal_sensor.onled(red)
                        break 
                if qr_img is not None and qrimg_show_cnt < QRCODE_SHOW_FRAME_COUNT:  # this  is time of qr duration!
                    frame = cam_sensor.image_add(frame, qr_img, sensor_rect)
                    qrimg_show_cnt += 1
                    
                    bottom_msg = MSG_SCAN_QR

                if qr_img is not None and qrimg_show_cnt >= QRCODE_SHOW_FRAME_COUNT:
                    qr_img = None
                    qrimg_show_cnt = 0
                    bottom_msg = ''
                    temp_str = ''
                    GPIO.cleanup() 
            # put bottom message
            msg_font = cv2.FONT_HERSHEY_COMPLEX
            if bottom_msg != WARRING_CONTACT_HOSPITAL:
                textSize, baseline = cv2.getTextSize(bottom_msg, msg_font, 0.7, 2)
                textSizeWidth, textSizeHeight = textSize
                top = int(frame.shape[1]/2 - textSizeWidth/2)
                left = frame.shape[0] - textSizeHeight-10
                cv2.putText(frame, bottom_msg, (top, left), msg_font, 0.7, (255, 255, 0) ,2)
            else:
                textSize, baseline = cv2.getTextSize(bottom_msg, msg_font, 0.9, 2)
                textSizeWidth, textSizeHeight = textSize
                top = int(frame.shape[1]/2 - textSizeWidth/2)
                left = frame.shape[0] - textSizeHeight-20        
                cv2.putText(frame, bottom_msg, (top, left), msg_font, 0.9, (0, 0, 255) ,2)
                           
            # put temperature text
            #temp_str = "33.4"
            val_font = cv2.FONT_HERSHEY_COMPLEX
            textSize, baseline = cv2.getTextSize(temp_str, val_font, 0.7, 2)
            textSizeWidth, textSizeHeight = textSize
            top = int(frame.shape[1]/2 - (textSizeWidth+10)/2)
            left = ry + rh + textSizeHeight + 5

            # draw text background rect
            #cv2.rectangle(frame, (top, left),
            #              (top + textSizeHeight, left+textSizeWidth),
            #              (255,255,255), thickness=cv2.FILLED)

            cv2.putText(frame, temp_str, (top, left), val_font, 0.7, (0, 255, 255) ,2)
            res_w = get_monitors()[0].width
            res_h = get_monitors()[0].height
            frame = cv2.resize(frame, (res_w, res_h))  #resize as full screen
            cv2.imshow('Video', frame)
            if save_flag:
                out.write(frame)
            time.sleep(0.001)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            #end_flag = True
            break
        
    # if end_flag:
    #     thermal_stat = False
    #     thermal_thread.join()
    if save_flag:
        out.release()
    video_capture.release()
    cv2.destroyAllWindows() 


if __name__ == '__main__':
    main()

