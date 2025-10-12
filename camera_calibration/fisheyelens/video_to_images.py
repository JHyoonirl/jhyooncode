import cv2
import os

def video_to_images(video_path, output_dir, interval=0.2):
    """
    Extract frames from a video at a specified interval and save them as images.

    Args:
        video_path (str): Path to the input video file.
        output_dir (str): Directory to save the extracted images.
        interval (float): Time interval between frames in seconds.
    """
    # Check if the video file exists
    if not os.path.exists(video_path):
        print(f"Video file {video_path} does not exist.")
        return

    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Open the video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Failed to open video file {video_path}.")
        return

    # Get video frame rate
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(fps * interval)

    frame_count = 0
    saved_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Save frame at the specified interval
        if frame_count % frame_interval == 0:
            image_name = f"frame_{saved_count:04d}.jpg"
            image_path = os.path.join(output_dir, image_name)
            cv2.imwrite(image_path, frame)
            saved_count += 1

        frame_count += 1

    cap.release()
    print(f"Saved {saved_count} images to {output_dir}.")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Extract frames from a video at a specified interval.")
    parser.add_argument("video_name", type=str, help="Name of the video file (without extension) in the raw_videos directory.")
    args = parser.parse_args()

    # --- 이 부분이 수정되었습니다 ---
    # 현재 스크립트 파일의 절대 경로를 가져옵니다.
    script_path = os.path.abspath(__file__)
    # 스크립트가 위치한 디렉토리 경로를 가져옵니다.
    script_dir = os.path.dirname(script_path)

    # 스크립트 디렉토리를 기준으로 폴더 경로를 설정합니다.
    raw_videos_dir = os.path.join(script_dir, "raw_videos")
    print(raw_videos_dir)
    calibration_images_dir = os.path.join(script_dir, "calibration_images")

    video_file = f"{args.video_name}.MP4"  # Assuming the video extension is .MP4
    video_path = os.path.join(raw_videos_dir, video_file)

    # Check if the video file exists
    if not os.path.exists(video_path):
        print(f"Video file {video_file} does not exist in {raw_videos_dir}.")
    else:
        # Create a folder in calibration_images with the same name as the video (without extension)
        output_dir = os.path.join(calibration_images_dir, args.video_name)

        # Extract frames from the video
        video_to_images(video_path, output_dir, interval=0.5)