import matplotlib
matplotlib.rc("font", family='AR PL UKai CN')

def visualize_truncated_objects(dataset_path, num_samples=5):
    """
    可视化KITTI数据集，显示并标出截断值不为0的目标，检查截断值计算是否正确

    Args:
        dataset_path: KITTI数据集路径
        num_samples: 要可视化的样本数量
    """
    import random
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    import cv2
    import os
    import numpy as np

    # 仅可视化带有标签的训练数据
    image_dir = os.path.join(dataset_path, 'training', 'image_2')
    label_dir = os.path.join(dataset_path, 'training', 'label_2')

    if not os.path.exists(image_dir) or not os.path.exists(label_dir):
        print(f"错误：图像目录 {image_dir} 或标签目录 {label_dir} 未找到。")
        return

    image_files = [f for f in os.listdir(image_dir) if f.endswith('.png')]

    if not image_files:
        print(f"错误：在 {image_dir} 中未找到图像文件")
        return

    # 首先查找具有截断目标的文件
    truncated_files = []
    for img_file in image_files:
        file_id = os.path.splitext(img_file)[0]
        label_path = os.path.join(label_dir, f"{file_id}.txt")

        if not os.path.exists(label_path):
            continue

        # 检查标签文件中是否有截断值 > 0 的目标
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split(' ')
                if len(parts) >= 2 and float(parts[1]) > 0:
                    truncated_files.append(img_file)
                    break

    # 如果找到了有截断目标的文件，优先选择它们
    if truncated_files:
        print(f"找到 {len(truncated_files)} 个包含截断目标的文件。")
        # 如果多于请求数量，随机选择样本
        if len(truncated_files) > num_samples:
            selected_files = random.sample(truncated_files, num_samples)
        else:
            selected_files = truncated_files
    else:
        print("未找到包含截断目标的文件。选择随机文件。")
        # 如果多于请求数量，随机选择样本
        if len(image_files) > num_samples:
            selected_files = random.sample(image_files, num_samples)
        else:
            selected_files = image_files

    for img_file in selected_files:
        file_id = os.path.splitext(img_file)[0]
        img_path = os.path.join(image_dir, img_file)
        label_path = os.path.join(label_dir, f"{file_id}.txt")

        if not os.path.exists(label_path):
            print(f"警告：标签文件 {label_path} 未找到，跳过。")
            continue

        # 使用OpenCV读取图像并转换为RGB（用于matplotlib）
        img = cv2.imread(img_path)
        if img is None:
            print(f"警告：无法读取图像 {img_path}，跳过。")
            continue

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # 获取图像尺寸
        img_height, img_width = img.shape[:2]

        # 读取标签
        with open(label_path, 'r') as f:
            labels = f.readlines()

        # 创建带有足够空间用于注释的图形
        plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
        plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

        fig, ax = plt.subplots(1, figsize=(15, 10))
        ax.imshow(img)

        truncated_count = 0

        # 处理每个标签
        for label in labels:
            parts = label.strip().split(' ')
            if len(parts) < 15:  # 标准KITTI标签有15+列
                continue

            obj_type = parts[0]  # 目标类型（车辆、行人等）
            truncation = float(parts[1])  # 截断值
            # 边界框坐标 [左, 上, 右, 下]
            bbox = [float(parts[4]), float(parts[5]), float(parts[6]), float(parts[7])]

            # 计算边界框的宽度和高度
            width = bbox[2] - bbox[0]
            height = bbox[3] - bbox[1]

            # 为边界框创建矩形
            if truncation > 0:
                # 用红色高亮显示具有非零截断值的目标
                rect = patches.Rectangle((bbox[0], bbox[1]), width, height,
                                         linewidth=2, edgecolor='r', facecolor='none')
                truncated_count += 1

                # 添加截断方向的视觉指示器
                truncation_indicators = []

                if bbox[0] <= 0:  # 左侧截断
                    truncation_indicators.append("左侧")
                    # 绘制箭头指示截断方向
                    ax.arrow(5, bbox[1] + height / 2, -10, 0, head_width=5, head_length=5, fc='r', ec='r')

                if bbox[2] >= img_width:  # 右侧截断
                    truncation_indicators.append("右侧")
                    # 绘制箭头指示截断方向
                    ax.arrow(img_width - 5, bbox[1] + height / 2, 10, 0, head_width=5, head_length=5, fc='r', ec='r')

                if bbox[1] <= 0:  # 顶部截断
                    truncation_indicators.append("顶部")
                    # 绘制箭头指示截断方向
                    ax.arrow(bbox[0] + width / 2, 5, 0, -10, head_width=5, head_length=5, fc='r', ec='r')

                if bbox[3] >= img_height:  # 底部截断
                    truncation_indicators.append("底部")
                    # 绘制箭头指示截断方向
                    ax.arrow(bbox[0] + width / 2, img_height - 5, 0, 10, head_width=5, head_length=5, fc='r', ec='r')


                # 添加带有截断信息的标签文本
                label_text = f"{obj_type}（截断值: {truncation:.2f}）"

                # 添加截断方向信息
                if truncation_indicators:
                    label_text += f"\n截断方向: {', '.join(truncation_indicators)}"

                # 定位文本以使其视觉清晰
                text_y = max(0, bbox[1] - 20)

                ax.text(bbox[0], text_y, label_text, color='white',
                        backgroundcolor='red',
                        fontsize=9, weight='bold')
            else:
                # 用绿色标记正常目标
                rect = patches.Rectangle((bbox[0], bbox[1]), width, height,
                                         linewidth=2, edgecolor='g', facecolor='none')

                # 为非截断目标添加常规标签文本
                label_text = f"{obj_type}（截断值: {truncation:.2f}）"
                ax.text(bbox[0], bbox[1] - 5, label_text, color='white',
                        backgroundcolor='green', fontsize=8)

            ax.add_patch(rect)

        # 添加带有摘要信息的标题
        ax.set_title(f"图像: {img_file} - 发现 {truncated_count} 个截断目标")
        # 添加颜色编码说明
        ax.text(10, img_height - 10,
                "颜色编码: 红色 = 截断值, 绿色 = 无截断",
                color='white', backgroundcolor='black', fontsize=10)

        plt.tight_layout()
        plt.show()

        # 暂停以允许在进入下一个图像之前查看当前图像
        user_input = input("按Enter继续下一张图像（输入q退出）: ")
        if user_input.lower() == 'q':
            break


# 示例用法
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='检查KITTI数据集截断值计算')
    parser.add_argument('--dataset', default="../../data/CADC_cropped", help='KITTI数据集路径')
    parser.add_argument('--num_samples', type=int, default=5, help='可视化的样本数量')

    args = parser.parse_args()

    visualize_truncated_objects(args.dataset, args.num_samples)