import os
import glob
import re
from collections import Counter


def count_categories(directory_path):
    """统计目录中所有txt文件的物体类别"""
    category_counter = Counter()
    txt_files = glob.glob(os.path.join(directory_path, "*.txt"))

    if not txt_files:
        print(f"在'{directory_path}'中未找到txt文件")
        return category_counter

    print(f"正在处理{len(txt_files)}个txt文件...")

    for file_path in txt_files:
        with open(file_path, 'r', encoding='utf-8') as file:
            for line in file:
                if not line.strip():
                    continue

                parts = line.strip().split()
                if parts:
                    category = parts[0]
                    category_counter[category] += 1

    return category_counter


def rename_category(directory_path, old_category, new_category):
    """将指定类别批量更改为新名称"""
    if old_category == new_category:
        print("旧类别名和新类别名相同，无需更改")
        return 0

    txt_files = glob.glob(os.path.join(directory_path, "*.txt"))

    if not txt_files:
        print(f"在'{directory_path}'中未找到txt文件")
        return 0

    replacements_count = 0
    files_modified = 0

    for file_path in txt_files:
        file_modified = False
        updated_lines = []

        with open(file_path, 'r', encoding='utf-8') as file:
            lines = file.readlines()

        for line in lines:
            if not line.strip():
                updated_lines.append(line)
                continue

            parts = line.strip().split()
            if parts and parts[0] == old_category:
                # 使用正则表达式确保只替换第一列
                new_line = re.sub(f"^{re.escape(old_category)}(\\s+)", f"{new_category}\\1", line)
                updated_lines.append(new_line)
                replacements_count += 1
                file_modified = True
            else:
                updated_lines.append(line)

        if file_modified:
            with open(file_path, 'w', encoding='utf-8') as file:
                file.writelines(updated_lines)
            files_modified += 1

    print(f"已修改{files_modified}个文件中的{replacements_count}处实例")
    return replacements_count


def main():
    """主函数"""
    directory_path = input("请输入包含txt文件的目录路径: ")

    if not os.path.isdir(directory_path):
        print(f"错误: '{directory_path}'不是有效的目录")
        return

    # 统计类别
    category_counts = count_categories(directory_path)

    if not category_counts:
        print("文件中未找到类别")
        return

    print("\n类别统计:")
    for category, count in sorted(category_counts.items()):
        print(f"{category}: {count}")

    # 交互式重命名类别
    while True:
        print("\n重命名类别 (不输入任何内容直接按回车键退出):")
        old_category = input("请输入要替换的类别名: ")

        if not old_category:
            break

        if old_category not in category_counts:
            print(f"类别'{old_category}'在文件中未找到")
            continue

        new_category = input("请输入新的类别名: ")
        if not new_category:
            print("新类别名不能为空")
            continue

        replacements = rename_category(directory_path, old_category, new_category)

        # 更新内存中的类别计数
        if replacements > 0:
            if new_category in category_counts:
                category_counts[new_category] += replacements
            else:
                category_counts[new_category] = replacements

            category_counts[old_category] -= replacements
            if category_counts[old_category] == 0:
                del category_counts[old_category]

            print("\n更新后的类别统计:")
            for category, count in sorted(category_counts.items()):
                print(f"{category}: {count}")


if __name__ == "__main__":
    main()