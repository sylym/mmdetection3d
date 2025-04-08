import os
import cv2
import numpy as np
import shutil
from tqdm import tqdm


def process_kitti_dataset(input_root, output_root):
    """
    裁剪KITTI数据集并调整相关数据

    Args:
        input_root: 原始KITTI数据集路径
        output_root: 处理后数据集保存路径
    """
    # 创建输出目录
    for split in ['training', 'testing']:
        split_path = os.path.join(input_root, split)
        if not os.path.exists(split_path):
            print(f"警告: {split_path} 不存在，跳过。")
            continue

        for subdir in ['calib', 'image_2', 'velodyne']:
            subdir_path = os.path.join(input_root, split, subdir)
            if os.path.exists(subdir_path):
                os.makedirs(os.path.join(output_root, split, subdir), exist_ok=True)

        # 只为training创建label_2目录
        if split == 'training':
            label_path = os.path.join(input_root, split, 'label_2')
            if os.path.exists(label_path):
                os.makedirs(os.path.join(output_root, split, 'label_2'), exist_ok=True)

        # 处理每个数据集分割
        process_split(split, input_root, output_root)


def process_split(split, input_root, output_root):
    """处理特定的数据集分割（training或testing）"""
    print(f"处理 {split} 分割...")

    # 获取文件列表（以图像文件为参考）
    image_dir = os.path.join(input_root, split, 'image_2')
    if not os.path.exists(image_dir):
        print(f"警告: {image_dir} 不存在，跳过。")
        return

    image_files = sorted([f for f in os.listdir(image_dir) if f.endswith('.png')])

    for img_file in tqdm(image_files):
        file_id = os.path.splitext(img_file)[0]

        # 处理图像并获取裁剪信息
        img_path = os.path.join(input_root, split, 'image_2', img_file)
        img = cv2.imread(img_path)
        if img is None:
            print(f"警告: 无法读取图像 {img_path}，跳过。")
            continue

        orig_height, orig_width = img.shape[:2]

        # 计算裁剪边界，以图片上下中心线为基准，上下各取193像素
        center_y = orig_height // 2
        crop_half_height = 193  # 上下各取193像素

        # 计算裁剪的上下边界，确保不超出图像范围
        top = max(0, center_y - crop_half_height)
        bottom = min(orig_height, center_y + crop_half_height)

        # 如果碰到边界且有剩余像素，调整另一边界以确保总高度为386（如果可能）
        if top == 0 and bottom < orig_height:
            # 碰到上边界，调整下边界
            bottom = min(orig_height, top + 386)
        elif bottom == orig_height and top > 0:
            # 碰到下边界，调整上边界
            top = max(0, bottom - 386)

        # 裁剪并保存图像
        cropped_img = img[top:bottom, :]
        output_img_path = os.path.join(output_root, split, 'image_2', img_file)
        cv2.imwrite(output_img_path, cropped_img)

        # 处理标定文件
        calib_path = os.path.join(input_root, split, 'calib', f"{file_id}.txt")
        output_calib_path = os.path.join(output_root, split, 'calib', f"{file_id}.txt")
        if os.path.exists(calib_path):
            process_calibration(calib_path, output_calib_path, top)

        # 处理标签（仅限training分割）
        if split == 'training':
            label_path = os.path.join(input_root, split, 'label_2', f"{file_id}.txt")
            output_label_path = os.path.join(output_root, split, 'label_2', f"{file_id}.txt")
            if os.path.exists(label_path):
                process_label(label_path, output_label_path, top, bottom - top, orig_width)

        # 复制velodyne数据（点云数据不需要调整）
        velodyne_path = os.path.join(input_root, split, 'velodyne', f"{file_id}.bin")
        output_velodyne_path = os.path.join(output_root, split, 'velodyne', f"{file_id}.bin")
        if os.path.exists(velodyne_path):
            shutil.copy(velodyne_path, output_velodyne_path)


def process_calibration(input_path, output_path, top_crop):
    """
    调整相机标定参数以适应裁剪后的图像

    Args:
        input_path: 原始标定文件路径
        output_path: 调整后标定文件保存路径
        top_crop: 从顶部裁剪的像素数
    """
    with open(input_path, 'r') as f:
        calib_lines = f.readlines()

    output_lines = []
    for line in calib_lines:
        line = line.strip()
        if not line:
            output_lines.append('\n')
            continue

        parts = line.split(':', 1)
        if len(parts) != 2:
            output_lines.append(line + '\n')
            continue

        key, values = parts
        key = key.strip()
        values = values.strip()

        if key in ['P0', 'P1', 'P2', 'P3']:
            # 解析投影矩阵值
            matrix_vals = [float(x) for x in values.split()]

            # 调整cy（垂直主点）
            # KITTI投影矩阵格式：[f_x 0 c_x t_x 0 f_y c_y t_y 0 0 1 0]
            # c_y的索引应该是6
            matrix_vals[6] -= top_crop

            # 使用相同精度格式化回字符串
            formatted_vals = []
            for val in matrix_vals:
                # 使用科学计数法，保持与原格式一致
                formatted_vals.append(f"{val:.6e}")

            new_line = f"{key}: {' '.join(formatted_vals)}\n"
            output_lines.append(new_line)
        else:
            # 保持其他标定数据不变
            output_lines.append(f"{key}: {values}\n")

    with open(output_path, 'w') as f:
        f.writelines(output_lines)


def process_label(input_path, output_path, top_crop, crop_height, image_width):
    """
    调整标签信息以适应裁剪后的图像，并重新计算截断值

    Args:
        input_path: 原始标签文件路径
        output_path: 调整后标签文件保存路径
        top_crop: 从顶部裁剪的像素数
        crop_height: 裁剪后图像的高度
        image_width: 图像宽度
    """
    with open(input_path, 'r') as f:
        label_lines = f.readlines()

    output_lines = []

    for line in label_lines:
        parts = line.strip().split(' ')
        if len(parts) < 15:  # 标准KITTI标签有15+列
            output_lines.append(line)
            continue

        # 提取原始边界框坐标
        # KITTI格式: [left, top, right, bottom]
        bbox = [float(parts[4]), float(parts[5]), float(parts[6]), float(parts[7])]

        # 计算边界框的原始高度和宽度
        box_height = bbox[3] - bbox[1]
        box_width = bbox[2] - bbox[0]

        if box_height <= 0 or box_width <= 0:
            continue  # 跳过无效框

        # 调整y坐标以适应裁剪
        bbox[1] -= top_crop  # 调整bbox顶部
        bbox[3] -= top_crop  # 调整bbox底部

        # 计算截断值
        truncation = 0.0

        # 检查垂直方向截断
        if bbox[1] < 0:
            # 计算目标位于图像上方的部分比例
            truncation += min(1.0, -bbox[1] / box_height)
            bbox[1] = 0

        if bbox[3] > crop_height:
            # 计算目标位于图像下方的部分比例
            truncation += min(1.0, (bbox[3] - crop_height) / box_height)
            bbox[3] = crop_height

        # 检查水平方向截断（裁剪不会改变，但原始计算可能有误）
        if bbox[0] < 0:
            # 计算目标位于图像左侧的部分比例
            truncation += min(1.0, -bbox[0] / box_width)
            bbox[0] = 0

        if bbox[2] > image_width:
            # 计算目标位于图像右侧的部分比例
            truncation += min(1.0, (bbox[2] - image_width) / box_width)
            bbox[2] = image_width

        # 截断值上限为1.0
        truncation = min(1.0, truncation)

        # 如果目标完全不在图像内或截断过大，跳过该目标
        if bbox[0] >= bbox[2] or bbox[1] >= bbox[3] or truncation >= 0.99:
            continue

        # 更新边界框和截断值
        parts[1] = f"{truncation:.2f}"  # 更新截断值
        parts[4:8] = [f"{x:.2f}" for x in bbox]  # 更新边界框

        output_lines.append(' '.join(parts) + '\n')

    with open(output_path, 'w') as f:
        f.writelines(output_lines)


# 使用示例
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='裁剪KITTI数据集图像并调整相关数据')
    parser.add_argument('--input', default="../../data/CADC", help='原始KITTI数据集路径')
    parser.add_argument('--output', default="../../data/CADC_cropped", help='处理后数据集保存路径')

    args = parser.parse_args()

    process_kitti_dataset(args.input, args.output)