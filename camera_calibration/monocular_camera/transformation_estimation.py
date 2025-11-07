import cv2
import numpy as np
import argparse
import glob
import os
import sys
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.transform import Rotation as R


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
    

def load_images(pathname):
    """
    지정된 디렉토리(pathname)에서 모든 .jpg 이미지를 로드하여
    이미지 리스트와 파일 경로 리스트를 반환합니다.
    """
    if not os.path.isdir(pathname):
        print(f"❌ 오류: 디렉토리 '{pathname}'을(를) 찾을 수 없습니다.")
        return None, None

    # glob.glob의 결과를 정렬하여 이미지 순서를 일관성 있게 유지합니다.
    jpg_files = sorted(glob.glob(os.path.join(pathname, '*.jpg')))

    if not jpg_files:
        print(f"🟡 정보: '{pathname}' 디렉토리에서 JPG 이미지를 찾을 수 없습니다.")
        return [], []

    images = []
    image_paths = [] # 💡 파일 경로를 저장할 리스트 추가
    
    for img_path in jpg_files:
        img = cv2.imread(img_path)
        if img is not None:
            images.append(img)
            image_paths.append(img_path) # 💡 리스트에 경로 추가
        else:
            print(f"⚠️ 경고: '{img_path}' 파일을 읽는 데 실패했습니다.")

    print(f"✅ 이미지 로드 성공: 총 {len(images)}개의 이미지를 로드했습니다.")
    return images, image_paths # 💡 이미지와 경로를 함께 반환
    
def map_generator(camMatrix, distCoeffs, image_size):
    """
    주어진 카메라 매개변수로 왜곡 보정 맵을 생성합니다.
    """
    new_camMatrix, roi = cv2.getOptimalNewCameraMatrix(
        camMatrix, distCoeffs, image_size, 0, image_size
    )

    mapx, mapy = cv2.initUndistortRectifyMap(
        camMatrix, distCoeffs, None, new_camMatrix, image_size, cv2.CV_32FC1
    )
    
    return mapx, mapy



def initialize_aruco_detector(dictionary_type=cv2.aruco.DICT_6X6_250):
    """
    ArUco 검출을 위한 사전(dictionary)과 파라미터를 초기화합니다.
    
    Args:
        dictionary_type: 사용할 ArUco 딕셔너리 타입.
        
    Returns:
        ArUco 딕셔너리 객체, 검출기 파라미터 객체.
    """
    print(f"✅ ArUco 검출기 초기화 (사전: {dictionary_type})")
    aruco_dict = cv2.aruco.getPredefinedDictionary(dictionary_type)
    aruco_params = cv2.aruco.DetectorParameters()
    aruco_params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX 
    
    # 3. 사전과 파라미터를 사용하여 '탐지기' 객체를 생성합니다.
    detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)
    
    return detector

def detect_and_draw_aruco(image, cam_matrix, dist_coeffs, detector: cv2.aruco.ArucoDetector, marker_length=0.1):
    """
    한 이미지 내에서 ArUco 마커를 검출하고, Pose를 추정하여 축을 그립니다.
    
    Args:
        image: 입력 이미지.
        cam_matrix: 카메라 내부 행렬.
        dist_coeffs: 카메라 왜곡 계수.
        aruco_dict: ArUco 딕셔너리.
        aruco_params: ArUco 검출기 파라미터.
        marker_length: 실제 마커의 한 변 길이 (미터 단위).
        
    Returns:
        마커 정보가 그려진 결과 이미지, 회전 벡터 리스트, 이동 벡터 리스트.
    """
    # 1. 이미지 왜곡 보정
    undistorted_img = cv2.undistort(image, cam_matrix, dist_coeffs)
    
    # ✨ detector 객체의 detectMarkers 메서드를 호출합니다.
    corners, ids, rejected = detector.detectMarkers(undistorted_img)
    
    rvecs, tvecs = None, None
    if ids is not None:
        rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
            corners, marker_length, cam_matrix, dist_coeffs
        )
        cv2.aruco.drawDetectedMarkers(undistorted_img, corners, ids)
        for rvec, tvec in zip(rvecs, tvecs):
            if hasattr(cv2, 'drawFrameAxes'):
                cv2.drawFrameAxes(undistorted_img, cam_matrix, dist_coeffs, rvec, tvec, marker_length / 2)
            else:
                cv2.aruco.drawAxis(undistorted_img, cam_matrix, dist_coeffs, rvec, tvec, marker_length / 2)
                
    return undistorted_img, rvecs, tvecs

def process_camera_images(camera_name, images, paths, cam_matrix, dist_coeffs, aruco_detector, marker_length, save_dir_base):
    """
    지정된 카메라의 이미지 세트에 대해 ArUco 검출 및 저장을 수행합니다.
    """
    if not images:
        print(f"🟡 {camera_name} 카메라 이미지가 없어 처리를 건너뜁니다.")
        return

    # 카메라별 결과 저장 폴더 생성
    save_dir_specific = os.path.join(save_dir_base, camera_name)
    os.makedirs(save_dir_specific, exist_ok=True)
    print(f"\n▶️  '{camera_name}' 이미지 처리 시작... (결과 저장: '{save_dir_specific}')")

    for img, path in zip(images, paths):
        filename = os.path.basename(path)
        
        # ArUco 마커 검출 및 그리기
        result_img, rvecs, tvecs = detect_and_draw_aruco(
            img, cam_matrix, dist_coeffs, aruco_detector, marker_length
        )
        
        if rvecs is not None:
            print(f"  ✅ [{filename}] 마커 검출 완료.")
        else:
            print(f"  🟡 [{filename}] 마커를 찾지 못했습니다.")

        # 결과 저장 및 화면 출력
        save_path = os.path.join(save_dir_specific, filename)
        cv2.imwrite(save_path, result_img)
        cv2.imshow(f"{camera_name} - ArUco Result", result_img)

        if cv2.waitKey(250) & 0xFF == ord('q'): # 0.25초 대기
            print("사용자 요청으로 프로그램을 중단합니다.")
            return True # 중단 신호 반환
    return False # 정상 종료

def calculate_average_marker_pose(images, cam_matrix, dist_coeffs, detector: cv2.aruco.ArucoDetector, marker_length):
    """
    여러 이미지에서 ArUco 마커의 Pose를 검출하고 평균 회전 및 이동 벡터를 계산합니다.
    """
    all_rotations = []
    all_translations = []

    print(f"--- 총 {len(images)}개의 이미지에서 평균 Pose 계산 시작 ---")
    
    for i, image in enumerate(images):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = detector.detectMarkers(gray)

        if ids is not None and len(ids) == 1:
            rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(corners, marker_length, cam_matrix, dist_coeffs)
            rotation_matrix, _ = cv2.Rodrigues(rvecs[0])
            all_rotations.append(rotation_matrix)
            all_translations.append(tvecs[0])
            print(f"  [{i+1}/{len(images)}] ✅ 마커 검출 성공")
        else:
            print(f"  [{i+1}/{len(images)}] 🟡 마커 검출 실패 또는 여러 개 검출됨")

    if not all_rotations:
        return None, None

    avg_translation = np.mean(all_translations, axis=0)
    print(f"  ✅ 평균 Translation: {avg_translation}")
    avg_rotation_matrix = R.from_matrix(all_rotations).mean().as_matrix()
    print(f"  ✅ 평균 Rotation: {avg_rotation_matrix}")

    print(f"--- {len(all_rotations)}개의 유효 데이터로 평균 Pose 계산 완료 ---")
    return avg_rotation_matrix, avg_translation

def draw_axis(ax, R, t, label, length=0.1):
    """3D 플롯에 좌표계를 그리는 헬퍼 함수"""
    origin = t.flatten()
    
    # 각 축의 방향 벡터 (X:Red, Y:Green, Z:Blue)
    x_axis = R[:, 0]
    y_axis = R[:, 1]
    z_axis = R[:, 2]
    
    # Quiver를 사용하여 화살표(축) 그리기
    ax.quiver(origin[0], origin[1], origin[2], x_axis[0], x_axis[1], x_axis[2], length=length, color='r', normalize=True)
    ax.quiver(origin[0], origin[1], origin[2], y_axis[0], y_axis[1], y_axis[2], length=length, color='g', normalize=True)
    ax.quiver(origin[0], origin[1], origin[2], z_axis[0], z_axis[1], z_axis[2], length=length, color='b', normalize=True)
    
    # 좌표계 이름 표시
    ax.text(origin[0], origin[1], origin[2], f'  {label}', color='k')



if __name__ == "__main__": 
    
    realsense_camMatrix, realsense_distCoeffs = load_camera_params("param/realsense_camera_params.yml")
    unitree_camMatrix, unitree_distCoeffs = load_camera_params("param/unitree_camera_params.yml")

    if np.size(realsense_camMatrix) == 0 or np.size(unitree_camMatrix) == 0:
        print("\n❌ 오류: 유효한 카메라 매개변수가 로드되지 않아 동영상을 보정할 수 없습니다.")
        sys.exit(1)

    # realsense_mapx, realsense_mapy = map_generator(
    #     realsense_camMatrix, realsense_distCoeffs, (1280, 720)
    # )

    # unitree_mapx, unitree_mapy = map_generator(
    #     unitree_camMatrix, unitree_distCoeffs, (1280, 720)
    # )

    realsense_images, realsense_paths = load_images("images/realsense")
    unitree_images, unitree_paths = load_images("images/unitree")

    aruco_detector = initialize_aruco_detector()
    MARKER_LENGTH_M = 0.1  # 마커 한 변의 실제 길이 (5cm)
    SAVE_DIRECTORY_BASE = "aruco_results"

    
    print("\n[단계 1/2] Realsense 카메라의 평균 Pose를 계산합니다.")
    R_realsense_to_marker, t_realsense_to_marker = calculate_average_marker_pose(
        realsense_images, realsense_camMatrix, realsense_distCoeffs, aruco_detector, MARKER_LENGTH_M
    )

    print("\n[단계 2/2] Unitree 카메라의 평균 Pose를 계산합니다.")
    R_unitree_to_marker, t_unitree_to_marker = calculate_average_marker_pose(
        unitree_images, unitree_camMatrix, unitree_distCoeffs, aruco_detector, MARKER_LENGTH_M
    )
    
    if R_realsense_to_marker is None or R_unitree_to_marker is None:
        print("\n❌ 오류: 한 카메라라도 유효한 마커 Pose를 계산하지 못해 3D 시각화를 진행할 수 없습니다.")
        sys.exit(1)

    # 1. 각 변환을 4x4 동차 행렬로 만듭니다.
    # T(marker <- unitree): Unitree에서 Marker로의 변환
    T_marker_from_unitree = np.eye(4)
    T_marker_from_unitree[:3, :3] = R_unitree_to_marker
    T_marker_from_unitree[:3, 3] = t_unitree_to_marker.flatten()

    # T(marker <- realsense): Realsense에서 Marker로의 변환
    T_marker_from_realsense = np.eye(4)
    T_marker_from_realsense[:3, :3] = R_realsense_to_marker
    T_marker_from_realsense[:3, 3] = t_realsense_to_marker.flatten()

    # 2. 필요한 역행렬을 계산합니다.
    # T(unitree <- marker) = T(marker <- unitree)의 역행렬
    T_unitree_from_marker = np.linalg.inv(T_marker_from_unitree)

    # 3. 변환을 순서대로 곱합니다: T(unitree -> realsense) = T(unitree -> marker) @ T(marker -> realsense)
    # T(unitree -> realsense)는 T(realsense <- unitree)와 같습니다.
    T_realsense_from_unitree = np.linalg.inv(T_marker_from_realsense @ T_unitree_from_marker)

    # 4. 4x4 행렬에서 최종 회전(R)과 이동(t)을 다시 추출합니다.
    R_unitree_to_realsense = T_realsense_from_unitree[:3, :3]
    t_unitree_to_realsense = T_realsense_from_unitree[:3, 3]
    print(f"\n✅ Unitree -> Realsense 회전 행렬:\n{R_unitree_to_realsense}")
    print(f"\n✅ Unitree -> Realsense 이동 벡터:\n{t_unitree_to_realsense}")
    # === 3D 플롯 생성 ===

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # 1. Realsense 카메라 좌표계 그리기 (월드 원점)
    R_world = np.identity(3)
    t_world = np.array([[0], [0], [0]])
    draw_axis(ax, R_world, t_world, "world(unitree)")
    draw_axis(ax, R_unitree_to_realsense, t_unitree_to_realsense, "realsense cam")

    # 2. ArUco 마커 좌표계 그리기 (Realsense 기준)
    draw_axis(ax, R_unitree_to_marker, t_unitree_to_marker, "ArUco Marker_from_unitree")
    # draw_axis(ax, R_realsense_to_marker, t_realsense_to_marker, "ArUco Marker_from_realsense")

    # 3. Unitree 카메라 좌표계 그리기 (Realsense 기준)
    # draw_axis(ax, R_unitree_to_realsense, t_unitree_to_realsense, "Unitree Cam")

    # 플롯 설정
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.set_title("3D Visualization of Camera and Marker Poses")
    ax.grid(True)

    # 각 축의 스케일을 동일하게 설정하여 왜곡 방지
    max_range = 1
    ax.set_xlim(-max_range, max_range)
    ax.set_ylim(-max_range, max_range)
    ax.set_zlim(-max_range, max_range)
    ax.set_box_aspect([1,1,1]) # 축 비율을 1:1:1로 설정

    ax.view_init(elev=-90, azim=270)

    plt.show()

    # print("\n✅ 모든 작업이 완료되었습니다.")
    # cv2.destroyAllWindows()

    # print("\n✅ 모든 이미지의 왜곡 보정 및 저장이 완료되었습니다.")
    # cv2.destroyAllWindows() # 모든 창 닫기
