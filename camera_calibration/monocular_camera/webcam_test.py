import cv2
import cv2.aruco as aruco
import numpy as np
import pyrealsense2 as rs
import os # os 모듈 추가

# --- YAML 파일에서 DetectorParameters 로드 함수 ---
def load_detector_params_from_yaml(filepath):
    """
    YAML 파일에서 ChArUco/Aruco 감지 파라미터를 로드하여 DetectorParameters 객체를 반환합니다.
    """
    fs = cv2.FileStorage(filepath, cv2.FILE_STORAGE_READ)
    if not fs.isOpened():
        raise IOError(f"Error: Could not open file {filepath}")

    params = aruco.DetectorParameters()

    # YAML 파일에서 값을 읽어 DetectorParameters 객체에 설정
    # 파일에 정의된 각 파라미터에 대해 fs.getNode(key).real() 또는 .intValue()를 사용합니다.
    
    # 예시: cornerRefinementMethod는 상수로 처리해야 합니다. (1 -> aruco.CORNER_REFINE_SUBPIX)
    corner_ref_method_int = int(fs.getNode('cornerRefinementMethod').real())
    if corner_ref_method_int == 1:
        params.cornerRefinementMethod = aruco.CORNER_REFINE_SUBPIX
    elif corner_ref_method_int == 3:
        params.cornerRefinementMethod = aruco.CORNER_REFINE_APRILTAG
    else:
        # 다른 값에 대한 처리 (e.g., 0 -> aruco.CORNER_REFINE_NONE)
        params.cornerRefinementMethod = corner_ref_method_int


    # 다른 파라미터들 (일부는 DetectorParameters.h를 참조하여 정확한 이름 확인 필요)
    # nmarkers는 일반적으로 ArucoDetectorParameters에 직접 설정되지 않습니다.
    # 하지만 YAML 파일에 있는 값을 DetectorParameters의 속성에 맞춰 설정합니다.
    
    params.adaptiveThreshWinSizeMin = int(fs.getNode('adaptiveThreshWinSizeMin').real())
    params.adaptiveThreshWinSizeMax = int(fs.getNode('adaptiveThreshWinSizeMax').real())
    params.adaptiveThreshWinSizeStep = int(fs.getNode('adaptiveThreshWinSizeStep').real())
    # adaptiveThreshWinSize는 보통 Step으로 계산되므로 직접 설정할 필요가 없을 수 있으나, YAML에 있다면 설정
    # params.adaptiveThreshWinSize = int(fs.getNode('adaptiveThreshWinSize').real()) # ArucoDetectorParameters에는 이 필드가 없음
    params.adaptiveThreshConstant = fs.getNode('adaptiveThreshConstant').real()
    params.minMarkerPerimeterRate = fs.getNode('minMarkerPerimeterRate').real()
    params.maxMarkerPerimeterRate = fs.getNode('maxMarkerPerimeterRate').real()
    params.polygonalApproxAccuracyRate = fs.getNode('polygonalApproxAccuracyRate').real()
    params.minDistanceToBorder = int(fs.getNode('minDistanceToBorder').real())
    params.minMarkerDistanceRate = fs.getNode('minMarkerDistanceRate').real() # minMarkerDistance는 minMarkerDistanceRate로 설정

    params.cornerRefinementWinSize = int(fs.getNode('cornerRefinementWinSize').real())
    params.cornerRefinementMaxIterations = int(fs.getNode('cornerRefinementMaxIterations').real())
    params.cornerRefinementMinAccuracy = fs.getNode('cornerRefinementMinAccuracy').real()
    params.markerBorderBits = int(fs.getNode('markerBorderBits').real())

    params.perspectiveRemovePixelPerCell = int(fs.getNode('perspectiveRemovePixelPerCell').real())
    params.perspectiveRemoveIgnoredMarginPerCell = fs.getNode('perspectiveRemoveIgnoredMarginPerCell').real()
    params.maxErroneousBitsInBorderRate = fs.getNode('maxErroneousBitsInBorderRate').real()
    params.minOtsuStdDev = fs.getNode('minOtsuStdDev').real()
    params.errorCorrectionRate = fs.getNode('errorCorrectionRate').real()

    fs.release()
    return params

# --- 1. ChArUco 보드 설정 ---
SQUARES_X = 8
SQUARES_Y = 10
SQUARE_LENGTH = 0.02
MARKER_LENGTH = 0.0145
DICTIONARY = aruco.getPredefinedDictionary(aruco.DICT_5X5_250)

# ChArUco 보드 객체 생성
board = aruco.CharucoBoard(
    (SQUARES_X, SQUARES_Y), 
    SQUARE_LENGTH, 
    MARKER_LENGTH, 
    DICTIONARY
)

# --- 2. 감지 파라미터 설정 (os.path.join을 사용하여 경로 생성) ---
# 스크립트 파일이 위치한 디렉토리 경로를 얻습니다.
script_dir = os.path.dirname(os.path.abspath(__file__))
FILE_NAME = 'charuco_detector_params.yml'

# 스크립트 디렉토리 내에 있는 YAML 파일의 전체 경로를 만듭니다.
# 주의: 이 방법이 작동하려면 charuco_detector_params.yml 파일이 
# webcam_test.py 파일과 동일한 디렉토리(monocular_camera)에 있어야 합니다.
YAML_FILE = os.path.join(script_dir, FILE_NAME)

try:
    # YAML 파일에서 파라미터 로드
    detector_params = load_detector_params_from_yaml(YAML_FILE)
    print(f"'{YAML_FILE}' 파일에서 감지 파라미터를 성공적으로 로드했습니다.")
    # 로드된 파라미터 확인 (선택 사항)
    # print(f"Loaded adaptiveThreshWinSizeMax: {detector_params.adaptiveThreshWinSizeMax}")
    # print(f"Loaded cornerRefinementMethod: {detector_params.cornerRefinementMethod}")
except IOError as e:
    print(e)
    print("기본 파라미터로 계속 진행합니다.")
    detector_params = aruco.DetectorParameters()
    detector_params.cornerRefinementMethod = aruco.CORNER_REFINE_SUBPIX

# 파라미터를 적용하여 ChArUco 감지기 생성
charuco_detector = aruco.CharucoDetector(board, detectorParams=detector_params)

# --- 3. RealSense 카메라 설정 ---
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
pipeline.start(config)

print("RealSense 카메라 스트리밍을 시작합니다. 'q' 키를 누르면 종료됩니다.")

try:
    while True:
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            continue

        color_image = np.asanyarray(color_frame.get_data())

        # --- 4. ChArUco 보드 감지 ---
        charuco_corners, charuco_ids, marker_corners, marker_ids = charuco_detector.detectBoard(color_image)
        
        # --- 5. 결과 시각화 ---
        if marker_ids is not None and len(marker_ids) > 0:
            cv2.aruco.drawDetectedMarkers(color_image, marker_corners, marker_ids)

        if charuco_ids is not None and len(charuco_ids) > 0:
            cv2.aruco.drawDetectedCornersCharuco(color_image, charuco_corners, charuco_ids)
            
        cv2.imshow('RealSense ChArUco Detection', color_image)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()
    print("프로그램을 종료합니다.")