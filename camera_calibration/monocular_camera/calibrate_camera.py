import cv2
import numpy as np
import argparse
import os

# TODO
'''
폴더 구조를 고려하여 자동화 필요
- monocular_camera
    - videos
        - undistorted_videos
        - raw_videos
    - param
    - images
'''

# --- 0.1. 유틸리티 함수: 카메라 매개변수 저장 ---

def save_camera_params(filename, camMatrix, distCoeffs, reproj_error, num_frames):
    """
    OpenCV FileStorage (YML) 형식으로 카메라 매개변수를 저장합니다.
    """
    fs = cv2.FileStorage(filename, cv2.FILE_STORAGE_WRITE)
    
    if fs.isOpened():
        fs.write("cameraMatrix", camMatrix)
        fs.write("distCoeffs", distCoeffs)
        fs.write("avg_reprojection_error", reproj_error)
        fs.write("num_calibration_frames", num_frames)
        fs.release()
        print(f"\n✅ 카메라 캘리브레이션 결과가 {filename}에 성공적으로 저장되었습니다.")
        return True
    else:
        print(f"\n❌ 오류: 카메라 매개변수 파일 {filename}을 저장할 수 없습니다.")
        return False

# --- 0.2. 카메라 내부 매트릭스 보정 함수 ---

def calibrate_camera_from_frames(all_charuco_corners, all_charuco_ids, 
                                board, frame_size, output_file, calib_flags=cv2.CALIB_FIX_K3):
    """
    ChArUco 데이터를 사용하여 카메라 내부 매개변수를 보정하고 결과를 저장합니다.
    """
    
    if len(all_charuco_corners) < 5: 
        print("경고: 캘리브레이션을 위한 유효 프레임 수가 5개 미만입니다. 보정을 건너뜁니다.")
        return np.array([]), np.array([])

    print(f"\n--- 카메라 캘리브레이션 시작: {len(all_charuco_corners)}개의 유효 프레임 사용 ---")

    h, w = frame_size[1], frame_size[0]
    camMatrix_init = np.array([
        [w, 0, w / 2],
        [0, h, h / 2],
        [0, 0, 1]
    ], dtype=np.float64)
    distCoeffs_init = np.zeros((5, 1), dtype=np.float64)
    print(camMatrix_init, distCoeffs_init)

    try:
        reproj_error, camMatrix, distCoeffs, rvecs, tvecs = cv2.aruco.calibrateCameraCharuco(
            charucoCorners=all_charuco_corners,
            charucoIds=all_charuco_ids,
            board=board,
            imageSize=frame_size,
            cameraMatrix=camMatrix_init, 
            distCoeffs=distCoeffs_init,
            flags=calib_flags
        )

        print("--- 카메라 캘리브레이션 완료 ---")
        print(f"최종 평균 재투영 오차 (RMSE): {reproj_error:.4f}")
        
        # 캘리브레이션 결과를 파일로 저장
        save_camera_params(output_file, camMatrix, distCoeffs, reproj_error, len(all_charuco_corners))
        
        return camMatrix, distCoeffs

    except Exception as e:
        print(f"❌ 오류: 캘리브레이션 중 예외 발생: {e}")
        return np.array([]), np.array([])

# --- 1. 유틸리티 함수: 카메라 매개변수 로드 ---
# (기존 코드와 동일)

def load_camera_params(filename):
    """
    OpenCV FileStorage에서 카메라 매개변수 (내부 행렬 및 왜곡 계수)를 로드합니다.
    """
    camMatrix = np.array([])
    distCoeffs = np.array([])
    
    if not os.path.exists(filename):
        print(f"오류: 카메라 매개변수 파일 {filename}을 찾을 수 없습니다.")
        return np.array([]), np.array([])

    fs = cv2.FileStorage(filename, cv2.FILE_STORAGE_READ)
    
    if fs.isOpened():
        camMatrix = fs.getNode("cameraMatrix").mat()
        distCoeffs = fs.getNode("distCoeffs").mat()
        fs.release()
        
        if camMatrix is None or distCoeffs is None:
            print(f"경고: {filename} 파일에서 'cameraMatrix' 또는 'distCoeffs'를 찾을 수 없습니다.")
            return np.array([]), np.array([])

        print(f"카메라 매개변수 로드 성공: {filename}")
        return camMatrix, distCoeffs
    else:
        return np.array([]), np.array([])

# -------------------------------------------------------------

# --- 2. ChArUco 보드 및 감지기 매개변수 설정 (동일) ---

SQUARES_X = 5
SQUARES_Y = 7
SQUARE_LENGTH = 0.032
MARKER_LENGTH = 0.019
DICTIONARY_ID = cv2.aruco.DICT_6X6_250

AXIS_LENGTH = 0.5 * min(SQUARES_X, SQUARES_Y) * SQUARE_LENGTH

dictionary = cv2.aruco.getPredefinedDictionary(DICTIONARY_ID)
board = cv2.aruco.CharucoBoard(
    (SQUARES_X, SQUARES_Y), SQUARE_LENGTH, MARKER_LENGTH, dictionary
)

detectorParams = cv2.aruco.DetectorParameters()
charucoParams = cv2.aruco.CharucoParameters()
charucoDetector = cv2.aruco.CharucoDetector(board, charucoParams, detectorParams)

# -------------------------------------------------------------

# --- 3. 메인 감지 및 캘리브레이션 로직 ---

def run_charuco_calibration_and_detection(video_path, calib_file):

    all_charuco_corners = []
    all_charuco_ids = []
    frame_size = None
    
    # 1. 비디오 캡처 객체 생성
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"오류: 비디오 파일 {video_path}를 열 수 없습니다.")
        return

    # 2. 캘리브레이션 결과가 있으면 로드, 없으면 자세 추정 비활성화
    camMatrix, distCoeffs = load_camera_params(calib_file)
    pose_enabled = np.size(camMatrix) > 0 and np.size(distCoeffs) > 0

    if not pose_enabled:
        print("\n**경고: 캘리브레이션 파일이 없으므로, 자세 추정은 비활성화됩니다. 데이터만 수집합니다.**")

    fps = cap.get(cv2.CAP_PROP_FPS)
    wait_time = int(1000 / fps) if fps > 0 else 10 

    print("--- ChArUco 동영상 감지 시작 (ESC 키로 종료) ---")
    
    totalIterations = 0

    while True:
        ret, frame = cap.read()
        if not ret: break

        frame_copy = frame.copy()
        totalIterations += 1
        
        # 3. 감지 및 모서리 보간
        charucoCorners, charucoIds, markerCorners, markerIds = charucoDetector.detectBoard(frame)
        
        # 4. 자세 추정 및 데이터 수집
        rvec, tvec, validPose = np.zeros((3, 1), dtype=np.float64), np.zeros((3, 1), dtype=np.float64), False
        
        if charucoIds is not None and len(charucoIds) >= 4:
            
            # 캘리브레이션용 데이터 수집
            all_charuco_corners.append(charucoCorners)
            all_charuco_ids.append(charucoIds)
            if frame_size is None:
                frame_size = frame.shape[1::-1] # (width, height)
            
            # 자세 추정 (Pose Estimation)
            if pose_enabled:
                objPoints, imgPoints = board.matchImagePoints(charucoCorners, charucoIds)
                success, rvec, tvec = cv2.solvePnP(objPoints, imgPoints, camMatrix, distCoeffs)
                if success: validPose = True

        # --- 5. 결과 시각화 ---
        
        if markerIds is not None and len(markerIds) > 0:
            cv2.aruco.drawDetectedMarkers(frame_copy, markerCorners, markerIds.flatten())
        if charucoIds is not None and len(charucoIds) > 0:
            cv2.aruco.drawDetectedCornersCharuco(frame_copy, charucoCorners, charucoIds.flatten(), (255, 0, 0))
            cv2.putText(frame_copy, f"Corners: {len(charucoIds)}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)


        if validPose:
            cv2.drawFrameAxes(frame_copy, camMatrix, distCoeffs, rvec, tvec, AXIS_LENGTH)
            cv2.putText(frame_copy, "Pose Estimated", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            cv2.putText(frame_copy, "Collecting Data" if charucoIds is not None and len(charucoIds) >= 4 else "No Board Found", 
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
        cv2.putText(frame_copy, f"Total Collected: {len(all_charuco_corners)}", 
                    (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
        cv2.imshow("ChArUco Detection & Data Collection", frame_copy)
        
        if cv2.waitKey(wait_time) == 27: # ESC
            break

        if len(all_charuco_corners) >= 50:
            print("✅ 50개 이상의 유효 프레임이 수집되어 자동으로 종료합니다.")
            break

    # 자원 해제
    cap.release()
    cv2.destroyAllWindows()

    # ------------------------------------------------
    # 6. 카메라 캘리브레이션 실행 (핵심)
    # ------------------------------------------------
    if len(all_charuco_corners) >= 5:
        print("\n--- 동영상 재생 종료: 캘리브레이션 시작 ---")
        calibrate_camera_from_frames(
            all_charuco_corners, 
            all_charuco_ids, 
            board, 
            frame_size,
            calib_file # 캘리브레이션 완료 후 이 파일에 결과를 저장
        )
    else:
        print(f"\n캘리브레이션을 위한 유효 데이터({len(all_charuco_corners)}개)가 부족합니다. (최소 5개 필요)")

# -------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ChArUco 보드 동영상 캘리브레이션 및 자세 추정.")
    parser.add_argument("-v", "--video", required=True, help="감지할 동영상 파일 이름 (예: charuco_test.mp4).") 
    parser.add_argument("-c", "--calib", default="camera_params.yml", help="카메라 캘리브레이션 파일을 저장/로드할 경로.")
    
    args = parser.parse_args()
    
    # 경로 처리: 스크립트 디렉토리 기준 'videos' 폴더에서 찾고, 없으면 스크립트 디렉토리에서 찾습니다.
    script_dir = os.path.dirname(os.path.abspath(__file__))
    video_filename = args.video
    
    path_attempt_1 = os.path.join(script_dir, 'videos', video_filename)
    path_attempt_2 = os.path.join(script_dir, video_filename)

    full_video_path = None
    
    if os.path.exists(path_attempt_1):
        full_video_path = path_attempt_1
    elif os.path.exists(path_attempt_2):
        full_video_path = path_attempt_2
    else:
        print(f"❌ 오류: 동영상 파일 '{video_filename}'을 찾을 수 없습니다.")
        print(f"   - 시도 1 (videos): {path_attempt_1}")
        print(f"   - 시도 2 (스크립트 폴더): {path_attempt_2}")
        import sys
        sys.exit(1)

    print(f"✅ 동영상 파일 로드 경로: {full_video_path}")
    
    run_charuco_calibration_and_detection(full_video_path, args.calib)