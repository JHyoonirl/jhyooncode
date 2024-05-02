import cv2
import os
import time
from datetime import datetime

capture = cv2.VideoCapture(0)
capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
save_images = []


########################
    # 이미지 캡쳐 #
########################
cur_time = time.time()
while capture.isOpened(): 
    ret, img = capture.read()           # 다음 프레임 읽기
    if ret:                         # 프레임 읽기 정상
        cv2.imshow('camera', img)
        if time.time() - cur_time > 0.3:
            save_images.append(img)
            cur_time = time.time()
    else:
        print('no frame')           # 다음 프레임을 읽을 수 없음.
        break
    
    if cv2.waitKey(1) == ord('q'): 
        break

    # time.sleep(0.5)

capture.release()                           # 캡쳐 자원 반납
cv2.destroyAllWindows()


########################
    # 이미지 저장 #
########################
current_path = os.getcwd()
print(current_path)
try:
    os.mkdir(current_path + '\images')
except Exception as ex:
    print('images folder already make it', ex)

now = datetime.now()

try:
    os.makedirs('images/{month}-{day}-{hour}'.format(month=now.month, day=now.day, hour=now.hour))
except Exception as ex:
    print('already make it', ex)

for image in save_images:
    now = datetime.now()
    cv2.imwrite('images/{month}-{day}-{hour}/{minute}-{second}-{micro}.PNG'.format(month=now.month, day=now.day, hour=now.hour, minute = now.minute, second=now.second, micro = int(now.microsecond/1000)), image)