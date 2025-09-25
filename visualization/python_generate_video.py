import os
import cv2

def generate_video_from_frames(video_folder, output_path, fps=30):
    """
    从帧生成视频
    :param video_folder: 包含帧的文件夹路径
    :param output_path: 输出视频文件的路径
    :param fps: 视频的帧率
    """
    # 获取帧文件列表并按文件名排序
    frame_files = [f for f in os.listdir(video_folder) if f.endswith(('.jpg', '.png'))]
    frame_files.sort()
    
    if not frame_files:
        print(f"No frames found in {video_folder}")
        return

    # 获取第一帧的尺寸
    first_frame_path = os.path.join(video_folder, frame_files[0])
    first_frame = cv2.imread(first_frame_path)
    height, width, _ = first_frame.shape

    # 定义视频编写器
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # 逐帧写入视频
    for frame_file in frame_files:
        frame_path = os.path.join(video_folder, frame_file)
        frame = cv2.imread(frame_path)
        if frame is None:
            print(f"跳过文件 {frame_path}，因为无法读取。")
            continue
        video_writer.write(frame)
    
    video_writer.release()
    print(f"视频已保存到 {output_path}")

def process_videos(root_folder, output_folder, fps=30):
    """
    处理根文件夹下的所有视频文件夹
    :param root_folder: 根文件夹路径
    :param output_folder: 输出视频文件夹路径
    :param fps: 视频的帧率
    """
    # 确保输出文件夹存在
    os.makedirs(output_folder, exist_ok=True)

    # 遍历根文件夹中的所有子文件夹
    for subfolder in os.listdir(root_folder):
        if subfolder not in ['dancingshoe', 'hand-9', 'hand3', 'kangaroo']:
            continue
        video_folder = os.path.join(root_folder, subfolder)
        
        if not os.path.isdir(video_folder):
            continue

        print(f"Processing video folder: {subfolder}")

        # 定义输出视频路径
        output_path = os.path.join(output_folder, f"{subfolder}.mp4")

        # 生成视频
        generate_video_from_frames(video_folder, output_path, fps)

# 设置文件夹路径
root_folder = '/home/deshui/pro/vos_pro/s3/out_0.72'
output_folder = '/home/deshui/pro/vos_pro/s3/visualization'
fps = 30  # 设置视频帧率

# 处理视频
process_videos(root_folder, output_folder, fps)
