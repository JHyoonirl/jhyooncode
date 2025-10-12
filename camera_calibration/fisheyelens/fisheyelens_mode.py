import cv2
import numpy as np
import glob
import argparse
import os

parser = argparse.ArgumentParser(description="Calibrate camera using standard lens model.")
parser.add_argument("video_name", type=str, help="Name of the video folder in calibration_images.")
args = parser.parse_args()

# 체커보드 설정 (내부 코너 개수)
CHECKERBOARD = (9, 6) 
square_size = 15.0  # 단위: mm

# 3D 월드 좌표 준비
objp = np.zeros((1, CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
objp[0,:,:2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
objp = objp * square_size

# 3D, 2D 포인트 저장을 위한 리스트
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.

# 스크립트 파일의 위치를 기준으로 경로를 설정합니다.
script_dir = os.path.dirname(os.path.abspath(__file__))
image_folder_path = os.path.join(script_dir, 'calibration_images', args.video_name)
images = glob.glob(os.path.join(image_folder_path, '*.jpg'))

if not images:
    print(f"Error: No images found in the directory: {image_folder_path}")
else:
    print(f"Found {len(images)} images. Processing...")
    
    gray = None

    # 첫 번째 이미지에서 이미지 크기를 가져오기 위함 (모든 이미지가 동일하다고 가정)
img_for_size = cv2.imread(images[0])
h, w = img_for_size.shape[:2]

for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 체스보드 코너 찾기
    ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, cv2.CALIB_CB_ADAPTIVE_THRESH +
                                            cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE)

    if ret == True:
        objpoints.append(objp)
        # 코너 정제 (서브픽셀 정확도 향상)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        imgpoints.append(corners2)

        # (옵션) 찾은 코너를 이미지에 그리고 보여주기
        # img = cv2.drawChessboardCorners(img, CHECKERBOARD, corners2, ret)
        # cv2.imshow('img', img)
        # cv2.waitKey(500)

    cv2.destroyAllWindows()

# ----------------- 캘리브레이션 수행 -----------------
# ✨ [수정] 캘리브레이션 전에 유효성 검사 추가
if not objpoints:
    print("Error: Checkerboard corners were not detected in any of the images.")
    print("Please check the image quality, lighting, and CHECKERBOARD pattern size.")
else:
    print(f"Checkerboard detected in {len(objpoints)} out of {len(images)} images.")
    
    # 이전에 제공된 캘리브레이션 코드는 이 'else' 블록 안으로 이동합니다.
    N_OK = len(objpoints)
    K = np.zeros((3, 3))
    D = np.zeros((4, 1))
    rvecs = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(N_OK)]
    tvecs = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(N_OK)]

    flags = (cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC |
            cv2.fisheye.CALIB_CHECK_COND |
            cv2.fisheye.CALIB_FIX_SKEW)
    
    # 캘리브레이션 실행
    rms, K, D, rvecs, tvecs = \
        cv2.fisheye.calibrate(
            objpoints,
            imgpoints,
            (w, h), # 이미지 크기
            K,
            D,
            rvecs,
            tvecs,
            flags,
            (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6)
        )

    print(f"\nRMS re-projection error: {rms}")
    print(f"K (Camera Matrix):\n {K}")
    print(f"D (Distortion Coefficients):\n {D}")

# ----------------- 왜곡 보정 적용 -----------------
# 첫 번째 이미지를 예시로 왜곡 보정
img_to_undistort = cv2.imread(images[0])
h_u, w_u = img_to_undistort.shape[:2]

# 왜곡 보정된 이미지의 새로운 카메라 매트릭스 계산
# balance=0.0: 원본 이미지의 모든 픽셀 유지 (블랙 테두리 생길 수 있음)
# balance=1.0: 최소한의 검은 테두리로 가장 큰 유효 영역
# balance=0.5: 그 중간
new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(K, D, (w_u, h_u), np.eye(3), balance=1.0)

# 왜곡 보정 맵 생성
map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), new_K, (w_u, h_u), cv2.CV_16SC2)

# 이미지 왜곡 보정
undistorted_img = cv2.remap(img_to_undistort, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

# 원본 이미지와 보정된 이미지 시각화
cv2.imshow('Original Image', img_to_undistort)
cv2.imshow('Undistorted Image', undistorted_img)
cv2.waitKey(0)
cv2.destroyAllWindows()