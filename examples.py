import pyrealsense2 as rs
import numpy as np
import cv2

# --- 설정 (기존과 동일) ---
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)

print("[INFO] RealSense 카메라 스트리밍을 시작합니다...")
profile = pipeline.start(config)

# # --- [추가] 필터 객체 생성 ---
# # 1. Decimation (해상도 줄이기)
# decimation_filter = rs.decimation_filter()
# decimation_filter.set_option(rs.option.filter_magnitude, 2) # 1280x720 -> 640x360

# # 2. Spatial (공간적 노이즈 제거)
# spatial_filter = rs.spatial_filter()
# spatial_filter.set_option(rs.option.filter_magnitude, 2)
# spatial_filter.set_option(rs.option.filter_smooth_alpha, 0.5)
# spatial_filter.set_option(rs.option.filter_smooth_delta, 20)
# spatial_filter.set_option(rs.option.holes_fill, 1) # Spatial 필터 내의 간단한 홀 채우기

# # 3. Temporal (시간적 노이즈 제거/안정화)
# temporal_filter = rs.temporal_filter()
# temporal_filter.set_option(rs.option.filter_smooth_alpha, 0.5) # 값이 낮을수록 안정화, 높을수록 반응성
# temporal_filter.set_option(rs.option.filter_smooth_delta, 20)

# # 4. Hole Filling (남아있는 홀 채우기)
# hole_filling_filter = rs.hole_filling_filter()
# hole_filling_filter.set_option(rs.option.holes_fill, 1) # 1: Fill from nearest, 2: Farest

# --- [추가] Depth-Color 정렬 객체 (매우 중요) ---
# Depth 이미지를 Color 이미지의 시점(viewpoint)으로 변환합니다.
# 이렇게 하면 (x, y) 픽셀이 color와 depth 모두에서 동일한 지점을 가리킵니다.
align_to_color = rs.align(rs.stream.color)


try:
    while True:
        frames = pipeline.wait_for_frames()
        
        # --- [수정] Depth-Color 정렬 ---
        # 정렬을 먼저 수행합니다.
        aligned_frames = align_to_color.process(frames)
        
        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()
        
        if not depth_frame or not color_frame:
            continue

        # --- [수정] 필터 체인 적용 ---
        # 필터 순서: Decimation -> Spatial -> Temporal -> Hole Filling
        # (순서는 사용 사례에 따라 다를 수 있습니다)
        # depth_frame = decimation_filter.process(depth_frame) # Decimation을 사용하면 해상도가 바뀜
        # depth_frame = spatial_filter.process(depth_frame)
        # depth_frame = temporal_filter.process(depth_frame)
        # depth_frame = hole_filling_filter.process(depth_frame) # 필요에 따라 사용
        
        # --- 프레임 처리 (기존과 유사) ---
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())
        
        # [참고] Decimation 필터를 사용했다면 color_image도 리사이즈해야 hstack 가능
        # if decimation_filter.get_option(rs.option.filter_magnitude) > 1:
        #     h, w = depth_image.shape
        #     color_image = cv2.resize(color_image, (w, h), interpolation=cv2.INTER_AREA)

        # 2. Depth 이미지 시각화 처리
        depth_image_8bit = cv2.convertScaleAbs(depth_image, alpha=0.03)
        depth_colormap = cv2.applyColorMap(depth_image_8bit, cv2.COLORMAP_JET)

        # --- 이미지 표시 ---
        images = np.hstack((color_image, depth_colormap))
        cv2.imshow('RealSense (RGB | Depth Colormap)', images)

        key = cv2.waitKey(1)
        if key & 0xFF == ord('q') or key == 27:
            cv2.destroyAllWindows()
            break

finally:
    print("[INFO] 스트리밍을 중지합니다.")
    pipeline.stop()