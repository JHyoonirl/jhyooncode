import cv2
import numpy as np
import argparse
import os
import sys

# --- 1. 유틸리티 함수: 카메라 매개변수 로드 ---

def load_camera_params(filename):
    """
    OpenCV FileStorage (YML/XML)에서 카메라 매개변수 (내부 행렬 및 왜곡 계수)를 로드합니다.
    """
    camMatrix = np.array([])
    distCoeffs = np.array([])
    
    if not os.path.exists(filename):
        print(f"❌ 오류: 카메라 매개변수 파일 '{filename}'을 찾을 수 없습니다.")
        return np.array([]), np.array([])

    fs = cv2.FileStorage(filename, cv2.FILE_STORAGE_READ)
    
    if fs.isOpened():
        # 파일에서 cameraMatrix와 distCoeffs 노드를 읽습니다.
        camMatrix = fs.getNode("cameraMatrix").mat()
        distCoeffs = fs.getNode("distCoeffs").mat()
        fs.release()
        
        if camMatrix is None or distCoeffs is None:
            print(f"경고: {filename} 파일에서 'cameraMatrix' 또는 'distCoeffs'를 찾을 수 없습니다.")
            return np.array([]), np.array([])

        print(f"✅ 카메라 매개변수 로드 성공: {filename}")
        return camMatrix, distCoeffs
    else:
        print(f"❌ 오류: 카메라 매개변수 파일 {filename}을 열 수 없습니다.")
        return np.array([]), np.array([])

# --- 2. 동영상 왜곡 보정 및 저장 함수 (메인 로직) ---

def undistort_and_save_video(input_video_path, calib_file):
    """
    캘리브레이션 결과를 사용하여 동영상을 읽고, 프레임별로 왜곡 보정한 후 새 동영상으로 저장합니다.
    """
    # 1. 캘리브레이션 파라미터 로드
    camMatrix, distCoeffs = load_camera_params(calib_file)
    if np.size(camMatrix) == 0:
        print("\n❌ 오류: 유효한 카메라 매개변수가 로드되지 않아 동영상을 보정할 수 없습니다.")
        return

    # 2. 입력 비디오 설정
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        print(f"\n❌ 오류: 입력 비디오 파일 {input_video_path}를 열 수 없습니다.")
        return

    # 비디오 속성 가져오기
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') # 비디오 코덱 (mp4v는 일반적인 mp4 코덱)
    
    # 3. 출력 비디오 설정
    # 출력 파일명 생성: 원본파일명_undistorted.mp4
    input_base, input_ext = os.path.splitext(os.path.basename(input_video_path))
    output_video_path = f"{input_base}_undistorted.mp4"
    
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    
    if not out.isOpened():
        print(f"\n❌ 오류: 출력 비디오 파일 {output_video_path}를 설정할 수 없습니다. (코덱 문제일 수 있음)")
        cap.release()
        return

    print(f"\n--- 동영상 왜곡 보정 및 저장 시작: {output_video_path} ---")
    
    # 새로운 카메라 매트릭스 계산 및 맵 생성
    # alpha=1.0: 모든 원본 픽셀 포함 (검은색 테두리 가능)
    # alpha=0.0: 유효 픽셀만 포함 (이미지 잘림 가능)
    new_camMatrix, roi = cv2.getOptimalNewCameraMatrix(
        camMatrix, distCoeffs, (width, height), 1, (width, height)
    )
    
    # 효율적인 보정을 위한 맵 생성 (cv2.remap에 사용)
    mapx, mapy = cv2.initUndistortRectifyMap(
        camMatrix, distCoeffs, None, new_camMatrix, (width, height), cv2.CV_32FC1 # 5 -> cv2.CV_32FC1
    )

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret: break

        # 4. 프레임 왜곡 보정 (cv2.remap 사용)
        undistorted_frame = cv2.remap(
            frame, mapx, mapy, cv2.INTER_LINEAR
        )

        # 5. 비디오 저장
        out.write(undistorted_frame)
        
        frame_count += 1
        if frame_count % 100 == 0:
            print(f"-> {frame_count} 프레임 처리 완료...", end='\r')

    # 자원 해제 및 완료
    cap.release()
    out.release()
    print(f"\n✅ 동영상 왜곡 보정 완료. 총 {frame_count} 프레임이 {output_video_path}에 저장되었습니다.")

# -------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="캘리브레이션 파일을 사용하여 동영상의 왜곡을 보정하고 새 파일로 저장합니다.")
    parser.add_argument("-v", "--video", required=True, help="보정할 원본 동영상 파일 이름 (예: input.mp4).") 
    parser.add_argument("-c", "--calib", default="camera_params.yml", help="카메라 캘리브레이션 파일 (.yml 또는 .xml) 경로.")
    
    args = parser.parse_args()
    
    # 경로 처리: 스크립트 디렉토리, 'videos' 폴더 등을 확인하여 파일 경로를 확정합니다.
    script_dir = os.path.dirname(os.path.abspath(__file__))
    video_filename = args.video
    calib_filename = args.calib
    
    video_path = os.path.join(script_dir, 'videos', video_filename)
    calib_path = os.path.join(script_dir, 'param', calib_filename)
    # path_attempt_2 = os.path.join(script_dir, video_filename)

    

    full_video_path = None
    
    if os.path.exists(video_path):
        full_video_path = video_path
    else:
        print(f"❌ 오류: 동영상 파일 '{video_filename}'을 찾을 수 없습니다.")
        sys.exit(1)

    if os.path.exists(calib_path):
        full_calib_path = calib_path
    else:
        print(f"❌ 오류: 캘리브레이션 파일 '{calib_filename}'을 찾을 수 없습니다.")
        sys.exit(1)

    print(f"✅ 동영상 파일 로드 경로: {full_video_path}")
    print(f"✅ 캘리브레이션 파일 로드 경로: {full_calib_path}")
    
    # 왜곡 보정 실행
    undistort_and_save_video(full_video_path, full_calib_path)
