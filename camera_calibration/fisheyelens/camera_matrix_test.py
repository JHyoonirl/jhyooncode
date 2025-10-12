import cv2
import numpy as np
import os

def create_overlay_video(input_path, output_path, K, D, alpha=0.5, blend_ratio=0.5):
    """
    원본 영상과 왜곡 보정된 영상을 투명하게 겹쳐서 비교하는 동영상을 생성하는 함수

    Args:
        input_path (str): 원본 동영상 파일 경로
        output_path (str): 비교 동영상을 저장할 파일 경로
        K (np.ndarray): 카메라 내부 파라미터 행렬
        D (np.ndarray): 왜곡 계수 행렬
        alpha (float): 0일수록 검은 영역 없이 잘라내고, 1일수록 모든 픽셀을 보존
        blend_ratio (float): 원본 영상의 투명도. 0.0 ~ 1.0 사이의 값.
    """
    # 1. 동영상 파일 열기
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"오류: '{input_path}' 동영상을 열 수 없습니다.")
        return

    # 2. 동영상 정보 가져오기
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    print(f"원본 동영상 정보: {width}x{height}, {fps:.2f} FPS")

    # 3. 출력 동영상 설정 (크기는 원본과 동일)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # 4. 최적의 새 카메라 행렬 및 왜곡 보정 맵 생성
    new_K, roi = cv2.getOptimalNewCameraMatrix(K, D, (width, height), alpha, (width, height))
    map1, map2 = cv2.initUndistortRectifyMap(K, D, None, new_K, (width, height), cv2.CV_32FC1)
    
    print("오버레이 비교 동영상 생성을 시작합니다...")

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        if frame_count % 100 == 0:
            print(f"{frame_count} 프레임 처리 중...")

        # 5. 왜곡 보정 적용
        undistorted_frame = cv2.remap(frame, map1, map2, interpolation=cv2.INTER_LINEAR)
        
        # --- ✨ 핵심 변경점: 두 프레임을 투명하게 합성 ---
        # cv2.addWeighted(src1, alpha, src2, beta, gamma)
        # 결과 = src1 * alpha + src2 * beta + gamma
        overlay_frame = cv2.addWeighted(frame, blend_ratio, undistorted_frame, 1 - blend_ratio, 0)
        
        # 합성된 프레임에 텍스트 추가
        text = f"Overlay (Original {blend_ratio*100}% vs Undistorted {(1-blend_ratio)*100}%)"
        cv2.putText(overlay_frame, text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
        
        # 합성된 프레임을 출력 동영상에 쓰기
        out.write(overlay_frame)
        # --------------------------------------------------

    # 6. 자원 해제
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    
    print(f"오버레이 비교 동영상이 완료되었습니다. '{output_path}' 파일로 저장되었습니다. 🎉")


if __name__ == '__main__':
    # --- 사용자 설정 영역 ---

    # 1. 동영상 경로 설정
    script_dir = os.getcwd()
    input_video_path = os.path.join(script_dir, 'camera_calibration','fisheyelens','raw_videos', 'GH019303.mp4')
    
    output_dir = os.path.join(script_dir, 'camera_calibration','fisheyelens', 'undistorted_videos')
    # 출력 파일 이름에 'overlay'를 추가하여 구분
    output_video_path = os.path.join(output_dir, 'GH019303_overlay_div_model.mp4')
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 2. DIVISION_UNDISTORTION 모델 파라미터
    fx = 453.8287113838226
    fy = 453.8287113838226 / 0.9816795507814547
    cx = 488.2482279122196
    cy = 273.7415339119115

    K = np.array([
        [fx, 0.0, cx],
        [0.0, fy, cy],
        [0.0, 0.0, 1.0]
    ])

    k1 = -1.4066165982928975e-06
    D = np.array([[k1, 0, 0, 0]]) # [k1, k2, p1, p2]
    
    # --- 설정 영역 끝 ---
    
    if not os.path.exists(input_video_path):
        print(f"오류: 입력 파일 '{input_video_path}'을(를) 찾을 수 없습니다.")
    else:
        # 겹쳐 보이게 하는 함수 호출
        create_overlay_video(input_path=input_video_path, 
                            output_path=output_video_path, 
                            K=K, 
                            D=D,
                            alpha=0.5,
                            blend_ratio=0.5) # 원본 50%, 보정본 50%로 설정

