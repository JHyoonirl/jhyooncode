import pyrealsense2 as rs
import numpy as np
import cv2

# --- 1. ArUco 검출기 설정 ---

# 사용할 ArUco 사전 정의 (이전과 동일)
dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
parameters = cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(dictionary, parameters)


# --- 2. RealSense 파이프라인 설정 ---

# 파이프라인: RealSense 장치와 센서 스트림을 관리하는 객체
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
pipeline.start(config)


# --- 3. 실시간 처리 루프 ---

try:
    while True:
        # 3-1. 프레임 받아오기
        # 파이프라인이 프레임을 받을 때까지 대기
        
        # 컬러 프레임과 깊이 프레임을 공간적으로 정렬
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()

        # 3-2. 프레임을 OpenCV에서 사용할 수 있는 numpy 배열로 변환
        # depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        # 3-3. ArUco 마커 검출
        corners, ids, rejected = detector.detectMarkers(color_image)

        # 3-4. 검출된 마커의 거리 측정 및 시각화
        if ids is not None:
            cv2.aruco.drawDetectedMarkers(color_image, corners, ids)

            for i, corner in enumerate(corners):
                # 마커의 중심 좌표 계산
                # 네 꼭짓점의 x, y 좌표의 평균을 구합니다.
                center_x = int(np.mean(corner[0][:, 0]))
                center_y = int(np.mean(corner[0][:, 1]))

                # 중심 좌표의 깊이 값(거리) 가져오기
                # depth_frame 객체의 get_distance() 메서드를 사용합니다.
                # distance = depth_frame.get_distance(center_x, center_y)

                # 0.0은 거리를 측정할 수 없는 경우 (예: 너무 멀거나 가까울 때)

        # 3-5. 결과 영상 출력
        cv2.imshow('RealSense ArUco Detection with Distance', color_image)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    # --- 4. 종료 처리 ---
    print("스트리밍을 중지합니다.")
    pipeline.stop()
    cv2.destroyAllWindows()