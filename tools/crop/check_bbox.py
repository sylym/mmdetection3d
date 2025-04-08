import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


def visualize_kitti_cropping(original_root, cropped_root, sample_ids=None, num_samples=5):
    """
    可视化原始和裁剪后的KITTI图像和检测框

    Args:
        original_root: 原始KITTI数据集路径
        cropped_root: 裁剪后数据集路径
        sample_ids: 指定要可视化的样本ID列表，如果为None则随机选择
        num_samples: 如果sample_ids为None，随机选择的样本数量
    """
    # 获取训练集图像列表
    original_img_dir = os.path.join(original_root, 'training', 'image_2')
    if not os.path.exists(original_img_dir):
        print(f"错误: 原始图像目录 {original_img_dir} 不存在")
        return

    image_files = sorted([f for f in os.listdir(original_img_dir) if f.endswith('.png')])

    # 如果没有指定样本ID，随机选择一些
    if sample_ids is None:
        if num_samples > len(image_files):
            num_samples = len(image_files)
        sample_indices = np.random.choice(len(image_files), num_samples, replace=False)
        sample_files = [image_files[i] for i in sample_indices]
    else:
        sample_files = [f"{id}.png" for id in sample_ids if os.path.exists(os.path.join(original_img_dir, f"{id}.png"))]

    for img_file in sample_files:
        file_id = os.path.splitext(img_file)[0]

        # 加载原始图像
        orig_img_path = os.path.join(original_root, 'training', 'image_2', img_file)
        orig_img = cv2.imread(orig_img_path)
        if orig_img is None:
            print(f"警告: 无法读取原始图像 {orig_img_path}")
            continue

        # 加载裁剪后的图像
        crop_img_path = os.path.join(cropped_root, 'training', 'image_2', img_file)
        crop_img = cv2.imread(crop_img_path)
        if crop_img is None:
            print(f"警告: 无法读取裁剪图像 {crop_img_path}")
            continue

        # 读取原始标签
        orig_label_path = os.path.join(original_root, 'training', 'label_2', f"{file_id}.txt")
        orig_boxes, orig_classes = [], []
        if os.path.exists(orig_label_path):
            with open(orig_label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split(' ')
                    if len(parts) < 15:
                        continue
                    obj_class = parts[0]
                    # KITTI格式: [left, top, right, bottom]
                    bbox = [float(parts[4]), float(parts[5]), float(parts[6]), float(parts[7])]
                    orig_boxes.append(bbox)
                    orig_classes.append(obj_class)

        # 读取裁剪后的标签
        crop_label_path = os.path.join(cropped_root, 'training', 'label_2', f"{file_id}.txt")
        crop_boxes, crop_classes = [], []
        if os.path.exists(crop_label_path):
            with open(crop_label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split(' ')
                    if len(parts) < 15:
                        continue
                    obj_class = parts[0]
                    # KITTI格式: [left, top, right, bottom]
                    bbox = [float(parts[4]), float(parts[5]), float(parts[6]), float(parts[7])]
                    crop_boxes.append(bbox)
                    crop_classes.append(obj_class)

        # 转换为RGB显示
        orig_img_rgb = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)
        crop_img_rgb = cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB)

        # 创建figure，使用GridSpec来控制子图大小
        fig = plt.figure(figsize=(18, 10))
        gs = GridSpec(1, 2, width_ratios=[orig_img.shape[1], crop_img.shape[1]])

        # 添加标题
        fig.suptitle(f'KITTI样本 {file_id} 裁剪前后对比', fontsize=16)

        # 绘制原始图像
        ax1 = plt.subplot(gs[0])
        ax1.imshow(orig_img_rgb)
        ax1.set_title(f'原始图像 {orig_img.shape[1]}x{orig_img.shape[0]}')

        # 在原始图像上绘制检测框
        for i, box in enumerate(orig_boxes):
            x1, y1, x2, y2 = box
            color = plt.cm.tab10(i % 10)
            rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor=color, linewidth=2)
            ax1.add_patch(rect)
            label_text = f"{orig_classes[i]}"
            ax1.text(x1, y1 - 5, label_text, color=color, fontsize=8,
                     backgroundcolor='white', alpha=0.7)

        # 绘制裁剪后的图像
        ax2 = plt.subplot(gs[1])
        ax2.imshow(crop_img_rgb)
        ax2.set_title(f'裁剪后图像 {crop_img.shape[1]}x{crop_img.shape[0]}')

        # 在裁剪后的图像上绘制检测框
        for i, box in enumerate(crop_boxes):
            x1, y1, x2, y2 = box
            color = plt.cm.tab10(i % 10)
            rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor=color, linewidth=2)
            ax2.add_patch(rect)
            label_text = f"{crop_classes[i]}"
            ax2.text(x1, y1 - 5, label_text, color=color, fontsize=8,
                     backgroundcolor='white', alpha=0.7)

        # 显示网格线便于参考
        ax1.grid(True, linestyle='--', alpha=0.3)
        ax2.grid(True, linestyle='--', alpha=0.3)

        # 调整布局
        plt.tight_layout(rect=[0, 0, 1, 0.96])  # 为标题留出空间
        plt.show()


def visualize_cropping_line(original_root, cropped_root, sample_ids=None, num_samples=5):
    """
    在原始图像上可视化裁剪线，展示裁剪区域

    Args:
        original_root: 原始KITTI数据集路径
        cropped_root: 裁剪后数据集路径
        sample_ids: 指定要可视化的样本ID列表，如果为None则随机选择
        num_samples: 如果sample_ids为None，随机选择的样本数量
    """
    # 获取训练集图像列表
    original_img_dir = os.path.join(original_root, 'training', 'image_2')
    if not os.path.exists(original_img_dir):
        print(f"错误: 原始图像目录 {original_img_dir} 不存在")
        return

    image_files = sorted([f for f in os.listdir(original_img_dir) if f.endswith('.png')])

    # 如果没有指定样本ID，随机选择一些
    if sample_ids is None:
        if num_samples > len(image_files):
            num_samples = len(image_files)
        sample_indices = np.random.choice(len(image_files), num_samples, replace=False)
        sample_files = [image_files[i] for i in sample_indices]
    else:
        sample_files = [f"{id}.png" for id in sample_ids if os.path.exists(os.path.join(original_img_dir, f"{id}.png"))]

    for img_file in sample_files:
        file_id = os.path.splitext(img_file)[0]

        # 加载原始图像
        orig_img_path = os.path.join(original_root, 'training', 'image_2', img_file)
        orig_img = cv2.imread(orig_img_path)
        if orig_img is None:
            print(f"警告: 无法读取原始图像 {orig_img_path}")
            continue

        # 加载裁剪后的图像（用于比较）
        crop_img_path = os.path.join(cropped_root, 'training', 'image_2', img_file)
        crop_img = cv2.imread(crop_img_path)
        if crop_img is None:
            print(f"警告: 无法读取裁剪图像 {crop_img_path}")
            continue

        # 计算裁剪边界
        orig_height = orig_img.shape[0]
        center_y = orig_height // 2
        crop_half_height = 193
        top = max(0, center_y - crop_half_height)
        bottom = min(orig_height, center_y + crop_half_height)

        # 调整上下边界，与处理中的逻辑保持一致
        if top == 0 and bottom < orig_height:
            bottom = min(orig_height, top + 386)
        elif bottom == orig_height and top > 0:
            top = max(0, bottom - 386)

        # 转换为RGB显示
        orig_img_rgb = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)
        crop_img_rgb = cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB)

        # 创建一个副本来显示裁剪线
        img_with_lines = orig_img_rgb.copy()
        # 绘制裁剪线（顶部为红色，底部为蓝色）
        img_with_lines[top, :] = [255, 0, 0]  # 红色表示顶部裁剪线
        img_with_lines[top - 1:top + 2, :] = [255, 0, 0]  # 加粗线条
        img_with_lines[bottom, :] = [0, 0, 255]  # 蓝色表示底部裁剪线
        img_with_lines[bottom - 1:bottom + 2, :] = [0, 0, 255]  # 加粗线条

        # 设置图像大小
        plt.figure(figsize=(18, 10))

        # 先显示带裁剪线的原始图像
        plt.subplot(2, 1, 1)
        plt.imshow(img_with_lines)
        plt.title(f'原始图像 {orig_img.shape[1]}x{orig_img.shape[0]}，裁剪区域 ({top} - {bottom})', fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.3)

        # 然后显示裁剪后的图像
        plt.subplot(2, 1, 2)
        plt.imshow(crop_img_rgb)
        plt.title(f'裁剪后图像 {crop_img.shape[1]}x{crop_img.shape[0]}', fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.3)

        plt.suptitle(f'KITTI样本 {file_id} 裁剪线可视化', fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.96])  # 为标题留出空间
        plt.show()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='可视化KITTI数据集裁剪前后对比')
    parser.add_argument('--original', default="../../data/CADC", help='原始KITTI数据集路径')
    parser.add_argument('--cropped', default="../../data/CADC_cropped", help='裁剪后数据集路径')
    parser.add_argument('--sample_ids', nargs='+', help='要可视化的样本ID（不指定则随机选择）')
    parser.add_argument('--num_samples', type=int, default=5, help='随机选择的样本数量')
    parser.add_argument('--mode', choices=['boxes', 'lines', 'both'], default='both',
                        help='可视化模式：boxes=显示检测框, lines=显示裁剪线, both=两者都显示')

    args = parser.parse_args()

    if args.mode in ['boxes', 'both']:
        visualize_kitti_cropping(args.original, args.cropped, args.sample_ids, args.num_samples)

    if args.mode in ['lines', 'both']:
        visualize_cropping_line(args.original, args.cropped, args.sample_ids, args.num_samples)