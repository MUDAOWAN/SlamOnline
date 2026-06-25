"""
RealSense D455 数据采集脚本
分辨率: 720p (1280x720)
功能: 运动检测 + 质量控制，防止镜头移动时采集模糊数据

使用方法:
    python collect_dataset.py --output datasets/my_collection --max_frames 3000

数据保存格式 (TUM RGB-D 兼容):
    dataset_path/
        rgb/           - RGB 图像 (jpg)
        depth/         - 深度图 (png)
        rgb.txt        - RGB 时间戳索引
        depth.txt      - Depth 时间戳索引
        imu.txt        - IMU 数据 (用于运动检测)
"""

import os
import sys
import time
import argparse
from datetime import datetime
import csv

import cv2
import numpy as np
import torch

try:
    import pyrealsense2 as rs
except ImportError:
    print("Error: pyrealsense2 not found. Install with: pip install pyrealsense2")
    sys.exit(1)


class MotionDetector:
    """基于 IMU 和帧差的运动检测器"""

    def __init__(self, accel_threshold=0.5, gyro_threshold=0.3, frame_diff_threshold=30.0):
        """
        Args:
            accel_threshold: 加速度阈值 (m/s^2), 超过则认为运动剧烈
            gyro_threshold: 角速度阈值 (rad/s), 超过则认为运动剧烈
            frame_diff_threshold: 帧间像素差异阈值, 超过则认为模糊
        """
        self.accel_threshold = accel_threshold
        self.gyro_threshold = gyro_threshold
        self.frame_diff_threshold = frame_diff_threshold

        self.prev_gyro = None
        self.prev_image = None
        self.stable_count = 0
        self.motion_count = 0

    def is_stable(self, gyro_data, current_image):
        """
        判断当前帧是否稳定

        Args:
            gyro_data: [ax, ay, az, gx, gy, gz] 或 None
            current_image: 当前图像 (numpy array, BGR)

        Returns:
            bool: True 表示稳定可以采集, False 表示运动剧烈应跳过
        """
        is_motion_by_imu = False

        if gyro_data is not None:
            # gyro: [gx, gy, gz]
            gyro_magnitude = np.linalg.norm(gyro_data[3:6])

            if self.prev_gyro is not None:
                gyro_change = abs(gyro_magnitude - np.linalg.norm(self.prev_gyro[3:6]))
                if gyro_change > self.gyro_threshold:
                    is_motion_by_imu = True

            self.prev_gyro = gyro_data

        # 帧差检测 (检测模糊)
        is_motion_by_frame = False
        if self.prev_image is not None and current_image is not None:
            # 转灰度比较
            if len(current_image.shape) == 3:
                gray = cv2.cvtColor(current_image, cv2.COLOR_BGR2GRAY)
                prev_gray = cv2.cvtColor(self.prev_image, cv2.COLOR_BGR2GRAY)
            else:
                gray = current_image
                prev_gray = self.prev_image

            # 计算帧差
            diff = cv2.absdiff(gray, prev_gray)
            mean_diff = np.mean(diff)

            if mean_diff > self.frame_diff_threshold:
                is_motion_by_frame = True

        self.prev_image = current_image.copy() if current_image is not None else None

        # 稳定条件: IMU 和帧差都不检测到剧烈运动
        is_stable = not (is_motion_by_imu or is_motion_by_frame)

        if is_stable:
            self.stable_count += 1
            self.motion_count = 0
        else:
            self.motion_count += 1
            self.stable_count = 0

        return is_stable

    def get_stats(self):
        return {
            "stable_frames": self.stable_count,
            "motion_frames": self.motion_count
        }


class RealSenseCollector:
    """RealSense D455 数据采集器"""

    def __init__(self, output_dir, resolution=(1280, 720), fps=30,
                 motion_threshold=0.5, save_imu=True):
        """
        Args:
            output_dir: 保存路径
            resolution: 分辨率 (width, height)
            fps: 帧率
            motion_threshold: 运动阈值
            save_imu: 是否保存 IMU 数据
        """
        self.output_dir = output_dir
        self.w, self.h = resolution
        self.fps = fps
        self.save_imu = save_imu

        self.pipeline = rs.pipeline()
        self.config = rs.config()
        # 设置深度分辨率 (D455 支持 848x480)
        depth_w, depth_h = self._get_depth_resolution(resolution, fps)

        self.config.enable_stream(rs.stream.color, self.w, self.h, rs.format.bgr8, fps)
        self.config.enable_stream(rs.stream.depth, depth_w, depth_h, rs.format.z16, fps)
        self.config.enable_stream(rs.stream.accel, rs.format.motion_xyz32f, 63)
        self.config.enable_stream(rs.stream.gyro, rs.format.motion_xyz32f, 200)

        # 创建输出目录
        self.rgb_dir = os.path.join(output_dir, "rgb")
        self.depth_dir = os.path.join(output_dir, "depth")
        os.makedirs(self.rgb_dir, exist_ok=True)
        os.makedirs(self.depth_dir, exist_ok=True)

        # 初始化采集器
        self.profile = self.pipeline.start(self.config)

        # 获取深度 Scale
        depth_sensor = self.profile.get_device().first_depth_sensor()
        self.depth_scale = depth_sensor.get_depth_scale()

        # 设置 RGB 相机 (禁用自动曝光以获得一致画质)
        self.rgb_sensor = self.profile.get_device().query_sensors()[1]
        self.rgb_sensor.set_option(rs.option.enable_auto_exposure, False)
        self.rgb_sensor.set_option(rs.option.exposure, 200)  # 固定曝光
        self.rgb_sensor.set_option(rs.option.enable_auto_white_balance, False)
        self.rgb_sensor.set_option(rs.option.brightness, 0)  # 固定亮度

        # 对齐器 (深度对齐到 RGB)
        self.align = rs.align(rs.stream.color)

        # IMU 缓冲
        self.imu_buffer = []

        # 运动检测器
        self.motion_detector = MotionDetector(
            accel_threshold=motion_threshold,
            gyro_threshold=motion_threshold * 0.5,
            frame_diff_threshold=25.0
        )

        # 统计数据
        self.frame_count = 0
        self.saved_count = 0
        self.skipped_count = 0

    @staticmethod
    def resolution_to_realsense(resolution, fps):
        """转换分辨率为 RealSense 支持的格式"""
        w, h = resolution
        # D455 支持的深度分辨率: 480x270 @ 30fps, 480x270 @ 60fps, 848x480 @ 30fps, 848x480 @ 60fps
        # 使用 848x480 与 RGB 更接近
        return (848, 480)

    def _get_imu_data(self, frames):
        """从帧中提取 IMU 数据"""
        gyro_data = None
        accel_data = None
        ts = None

        # 遍历所有帧，找 gyro 和 accel
        for frame in frames:
            if frame.is_frameset():
                continue
            stream = frame.get_profile().stream_type()

            if stream == rs.stream.gyro:
                gyro_frame = frame.as_motion_frame()
                gyro_data = gyro_frame.get_motion_data()
                ts = gyro_frame.get_timestamp() / 1000.0
            elif stream == rs.stream.accel:
                accel_frame = frame.as_motion_frame()
                accel_data = accel_frame.get_motion_data()

        if gyro_data and accel_data:
            return [ts, accel_data.x, accel_data.y, accel_data.z,
                    gyro_data.x, gyro_data.y, gyro_data.z]
        return None

    def capture_frame(self, skip_motion_blur=True):
        """
        采集一帧

        Args:
            skip_motion_blur: True 则运动剧烈时跳过

        Returns:
            (rgb_image, depth_image, timestamp) 或 None (跳过时)
        """
        try:
            frames = self.pipeline.wait_for_frames(timeout_ms=5000)
        except Exception:
            return None

        # 对齐深度到 RGB
        aligned_frames = self.align.process(frames)

        # 获取 RGB
        color_frame = aligned_frames.get_color_frame()
        depth_frame = aligned_frames.get_depth_frame()

        if not color_frame or not depth_frame:
            return None

        timestamp = color_frame.get_timestamp() / 1000.0  # 秒

        # 获取图像
        rgb_image = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(depth_frame.get_data())

        # 获取 IMU
        imu_data = self._get_imu_data(frames)
        if imu_data and self.save_imu:
            self.imu_buffer.append(imu_data)

        # 运动检测
        if skip_motion_blur:
            is_stable = self.motion_detector.is_stable(imu_data, rgb_image)
            if not is_stable:
                self.skipped_count += 1
                return None

        self.frame_count += 1
        return (rgb_image, depth_image, timestamp)

    def save_frame(self, rgb_image, depth_image, timestamp):
        """保存一帧到磁盘"""
        frame_idx = self.saved_count

        # 保存 RGB (jpg)
        rgb_path = os.path.join(self.rgb_dir, f"frame_{frame_idx:06d}.jpg")
        cv2.imwrite(rgb_path, rgb_image, [cv2.IMWRITE_JPEG_QUALITY, 95])

        # 保存 Depth (png) - 转换为米为单位 (RealSense 默认是毫米)
        depth_m = (depth_image * self.depth_scale * 1000).astype(np.uint16)  # 转为 mm
        depth_path = os.path.join(self.depth_dir, f"frame_{frame_idx:06d}.png")
        cv2.imwrite(depth_path, depth_m)

        self.saved_count += 1

        return rgb_path, depth_path

    def save_index_files(self):
        """保存 TUM 格式的索引文件"""
        # 收集所有帧的时间戳 (通过文件名排序)
        rgb_files = sorted(os.listdir(self.rgb_dir))
        depth_files = sorted(os.listdir(self.depth_dir))

        # 生成 rgb.txt
        rgb_txt_path = os.path.join(self.output_dir, "rgb.txt")
        with open(rgb_txt_path, 'w') as f:
            f.write("# timestamp filename\n")
            # 时间戳使用相对时间 (从第一帧开始)
            for i, fname in enumerate(rgb_files):
                # 使用帧索引作为时间戳 (每帧间隔约 1/fps 秒)
                ts = i * (1.0 / self.fps)
                f.write(f"{ts:.6f} rgb/{fname}\n")

        # 生成 depth.txt
        depth_txt_path = os.path.join(self.output_dir, "depth.txt")
        with open(depth_txt_path, 'w') as f:
            f.write("# timestamp filename\n")
            for i, fname in enumerate(depth_files):
                ts = i * (1.0 / self.fps)
                f.write(f"{ts:.6f} depth/{fname}\n")

        # 保存 IMU 数据
        if self.save_imu and self.imu_buffer:
            imu_path = os.path.join(self.output_dir, "imu.txt")
            with open(imu_path, 'w') as f:
                f.write("# timestamp acc_x acc_y acc_z gyro_x gyro_y gyro_z\n")
                for row in self.imu_buffer:
                    f.write(" ".join(map(str, row)) + "\n")

        print(f"Index files saved to {self.output_dir}")

    def stop(self):
        """停止采集并保存索引"""
        self.pipeline.stop()

        # 保存索引文件
        self.save_index_files()

        print(f"\n=== Collection Summary ===")
        print(f"Total frames captured: {self.frame_count}")
        print(f"Frames saved: {self.saved_count}")
        print(f"Frames skipped (motion): {self.skipped_count}")
        print(f"IMU samples: {len(self.imu_buffer)}")
        print(f"Output: {self.output_dir}")

    def _get_depth_resolution(self, resolution, fps):
        """获取 RealSense D455 支持的深度分辨率"""
        w, h = resolution
        # D455 支持的深度分辨率: 480x270, 848x480
        return (848, 480)

    def get_intrinsics(self):
        """获取相机内参"""
        rgb_profile = rs.video_stream_profile(self.profile.get_stream(rs.stream.color))
        intrinsics = rgb_profile.get_intrinsics()

        return {
            "fx": intrinsics.fx,
            "fy": intrinsics.fy,
            "cx": intrinsics.ppx,
            "cy": intrinsics.ppy,
            "width": intrinsics.width,
            "height": intrinsics.height,
            "coeffs": intrinsics.coeffs
        }


def main():
    parser = argparse.ArgumentParser(description="RealSense D455 数据采集")
    parser.add_argument("--output", "-o", type=str, default="datasets/realsense_capture",
                        help="输出目录")
    parser.add_argument("--max_frames", type=int, default=5000,
                        help="最大采集帧数")
    parser.add_argument("--fps", type=int, default=30,
                        help="帧率")
    parser.add_argument("--motion_threshold", type=float, default=0.5,
                        help="运动阈值 (0.1-1.0)")
    parser.add_argument("--width", type=int, default=1280,
                        help="宽度")
    parser.add_argument("--height", type=int, default=720,
                        help="高度")
    parser.add_argument("--no_skip", action="store_true",
                        help="禁用运动跳过 (采集所有帧)")

    args = parser.parse_args()

    # 创建带时间戳的输出目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.output, timestamp)
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 50)
    print("RealSense D455 数据采集")
    print("=" * 50)
    print(f"Output: {output_dir}")
    print(f"Resolution: {args.width}x{args.height} @ {args.fps}fps")
    print(f"Motion threshold: {args.motion_threshold}")
    print(f"Skip motion blur: {not args.no_skip}")
    print("=" * 50)

    try:
        collector = RealSenseCollector(
            output_dir=output_dir,
            resolution=(args.width, args.height),
            fps=args.fps,
            motion_threshold=args.motion_threshold
        )

        # 打印相机内参
        intr = collector.get_intrinsics()
        print(f"\nCamera intrinsics:")
        print(f"  fx: {intr['fx']:.2f}, fy: {intr['fy']:.2f}")
        print(f"  cx: {intr['cx']:.2f}, cy: {intr['cy']:.2f}")

        # 保存内参到文件
        calib_path = os.path.join(output_dir, "calibration.txt")
        with open(calib_path, 'w') as f:
            f.write(f"# RealSense D455 Calibration @ {args.width}x{args.height}\n")
            f.write(f"fx: {intr['fx']}\n")
            f.write(f"fy: {intr['fy']}\n")
            f.write(f"cx: {intr['cx']}\n")
            f.write(f"cy: {intr['cy']}\n")
            f.write(f"width: {intr['width']}\n")
            f.write(f"height: {intr['height']}\n")
            f.write(f"distorted: True\n")
            f.write(f"depth_scale: {collector.depth_scale}\n")
        print(f"Calibration saved to {calib_path}")

        print("\n开始采集... 按 Ctrl+C 停止\n")

        # 主循环
        start_time = time.time()
        last_print_time = start_time

        while collector.saved_count < args.max_frames:
            result = collector.capture_frame(skip_motion_blur=not args.no_skip)

            if result is None:
                # 跳过或无数据
                continue

            rgb_image, depth_image, timestamp = result

            # 保存
            collector.save_frame(rgb_image, depth_image, timestamp)

            # 打印进度
            current_time = time.time()
            if current_time - last_print_time >= 2.0:
                elapsed = current_time - start_time
                fps = collector.saved_count / elapsed
                stats = collector.motion_detector.get_stats()
                print(f"Frames: {collector.saved_count}/{args.max_frames} | "
                      f"FPS: {fps:.1f} | "
                      f"Skipped: {stats['motion_frames']} | "
                      f"Stable: {stats['stable_frames']}")
                last_print_time = current_time

    except KeyboardInterrupt:
        print("\n采集停止 (Ctrl+C)")

    finally:
        collector.stop()

        # 生成 MonoGS 可用的 YAML 配置
        intr = collector.get_intrinsics()
        yaml_path = os.path.join(output_dir, "dataset_config.yaml")
        with open(yaml_path, 'w') as f:
            f.write(f"# Auto-generated config for MonoGS\n")
            f.write(f"# Dataset path: {output_dir}\n\n")
            f.write("Results:\n")
            f.write("  save_results: False\n")
            f.write("  save_dir: \"results\"\n")
            f.write("  use_gui: False\n")
            f.write("  eval_rendering: False\n\n")
            f.write("Dataset:\n")
            f.write(f"  dataset_path: \"{output_dir}\"\n")
            f.write(f"  type: 'tum'\n")
            f.write(f"  sensor_type: 'depth'\n")
            f.write("  pcd_downsample: 32\n")
            f.write("  pcd_downsample_init: 32\n\n")
            f.write("Calibration:\n")
            f.write(f"  fx: {intr['fx']}\n")
            f.write(f"  fy: {intr['fy']}\n")
            f.write(f"  cx: {intr['cx']}\n")
            f.write(f"  cy: {intr['cy']}\n")
            f.write("  k1: 0.0\n")
            f.write("  k2: 0.0\n")
            f.write("  p1: 0.0\n")
            f.write("  p2: 0.0\n")
            f.write("  k3: 0.0\n")
            f.write("  distorted: True\n")
            f.write(f"  width: {intr['width']}\n")
            f.write(f"  height: {intr['height']}\n")
            f.write(f"  depth_scale: {collector.depth_scale * 1000.0}\n\n")
            f.write("Training:\n")
            f.write("  eval_ate: False\n")

        print(f"\nMonoGS config saved to {yaml_path}")
        print(f"\n使用方法: python slam.py --config {yaml_path}")


if __name__ == "__main__":
    main()