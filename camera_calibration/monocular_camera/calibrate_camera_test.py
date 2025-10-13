import cv2
import numpy as np
import sys
import argparse
import os

# --- 1. 유틸리티 함수: 카메라 매개변수 로드 ---

def load_camera_params(filename):
    """
    OpenCV FileStorage에서 카메라 매개변수 (내부 행렬 및 왜곡 계수)를 로드합니다.
    """
    camMatrix = np.array([])
    distCoeffs = np.array([])
    
    fs = cv2.FileStorage(filename, cv2.FILE_STORAGE_READ)
    
    if fs.isOpened():
        # 일반적으로 'cameraMatrix'와 'distCoeffs'라는 키를 사용합니다.
        camMatrix = fs.getNode("cameraMatrix").mat()
        distCoeffs = fs.getNode("distCoeffs").mat()
        fs.release()
        
        if camMatrix is None or distCoeffs is None:
            print(f"경고: {filename} 파일에서 'cameraMatrix' 또는 'distCoeffs'를 찾을 수 없습니다.")
            return np.array([]), np.array([])

        print(f"카메라 매개변수 로드 성공: {filename}")
        return camMatrix, distCoeffs
    else:
        print(f"오류: 카메라 매개변수 파일 {filename}을 열 수 없습니다.")
        return np.array([]), np.array([])

# --- 2. ChArUco 보드 및 감지기 매개변수 설정 ---

# 체스보드 칸 수 (X축, Y축)
SQUARES_X = 8
SQUARES_Y = 10
# 체스보드 한 칸의 길이 (단위: 미터 또는 임의의 길이 단위)
SQUARE_LENGTH = 0.02
# ArUco 마커의 길이 (단위: 위와 동일)
MARKER_LENGTH = 0.0145
# 사용할 ArUco 딕셔너리
DICTIONARY_ID = cv2.aruco.DICT_5X5_250
# 좌표 축의 길이 (시각화를 위해)
AXIS_LENGTH = 0.5 * min(SQUARES_X, SQUARES_Y) * SQUARE_LENGTH

# 딕셔너리 정의
dictionary = cv2.aruco.getPredefinedDictionary(DICTIONARY_ID)

# Charuco 보드 정의
board = cv2.aruco.CharucoBoard(
    (SQUARES_X, SQUARES_Y), SQUARE_LENGTH, MARKER_LENGTH, dictionary
)

# 감지기 매개변수 정의 및 ChArUcoDetector 생성
detectorParams = cv2.aruco.DetectorParameters()
charucoParams = cv2.aruco.CharucoParameters()
charucoDetector = cv2.aruco.CharucoDetector(board, charucoParams, detectorParams)


# --- 3. 메인 감지 로직 ---

def detect_charuco_on_image(image_path, calib_file):
    
    # 1. 이미지 로드
    frame = cv2.imread(image_path)
    if frame is None:
        print(f"오류: 이미지 파일 {image_path}를 로드할 수 없습니다.")
        return

    frame_copy = frame.copy()
    
    # 2. 카메라 매개변수 로드 (자세 추정을 위해)
    camMatrix, distCoeffs = load_camera_params(calib_file)
    # 카메라 매개변수가 로드되지 않았으면, 자세 추정은 건너뜁니다.
    pose_enabled = np.size(camMatrix) > 0 and np.size(distCoeffs) > 0
    if not pose_enabled:
        print("카메라 캘리브레이션 매개변수가 없거나 유효하지 않아 자세 추정을 건너뜀니다.")

    print("--- ChArUco 감지 시작 ---")
    
    # 3. 감지 및 모서리 보간 (C++의 detectBoard 함수)
    charucoCorners, charucoIds, markerCorners, markerIds = charucoDetector.detectBoard(frame)
    
    # 4. 자세 추정
    rvec = np.zeros((3, 1), dtype=np.float64)
    tvec = np.zeros((3, 1), dtype=np.float64)
    validPose = False
    
    # 감지된 Charuco 모서리가 4개 이상이고, 자세 추정이 활성화된 경우
    if pose_enabled and charucoIds is not None and len(charucoIds) >= 4:
        # 객체점(objPoints)과 이미지점(imgPoints) 계산
        objPoints, imgPoints = board.matchImagePoints(charucoCorners, charucoIds)
        
        # solvePnP를 사용하여 자세 추정
        success, rvec, tvec = cv2.solvePnP(objPoints, imgPoints, camMatrix, distCoeffs)
        if success:
            validPose = True

    # --- 5. 결과 시각화 ---
    
    # ArUco 마커 그리기
    if markerIds is not None and len(markerIds) > 0:
        cv2.aruco.drawDetectedMarkers(frame_copy, markerCorners, markerIds.flatten())

    # Charuco 코너 그리기 (파란색)
    if charucoIds is not None and len(charucoIds) > 0:
        cv2.aruco.drawDetectedCornersCharuco(frame_copy, charucoCorners, charucoIds.flatten(), (255, 0, 0))

    # 자세가 유효하면 좌표 축 그리기
    if validPose:
        cv2.drawFrameAxes(frame_copy, camMatrix, distCoeffs, rvec, tvec, AXIS_LENGTH)
        cv2.putText(frame_copy, "Pose Estimated", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    else:
        cv2.putText(frame_copy, "No Pose Estimated (or < 4 corners)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
    # 결과 이미지 출력
    cv2.imshow("ChArUco Detection Result", frame_copy)
    cv2.waitKey(0) # 아무 키를 누를 때까지 대기
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ChArUco 보드 이미지 감지 및 자세 추정.")
    parser.add_argument("-i", "--image", required=True, help="감지할 이미지 파일 경로.")
    parser.add_argument("-c", "--calib", default="camera_params.yml", help="카메라 캘리브레이션 파일 경로 (.yml 또는 .xml).")
    
    args = parser.parse_args()
    script_dir = os.path.dirname(os.path.abspath(__file__))
    image_path = os.path.join(script_dir, 'images', args.image)
    
    detect_charuco_on_image(image_path, args.calib)