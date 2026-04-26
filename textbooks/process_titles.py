"""
Markdown文件标题处理
"""

import os
import re
import sys


def process_markdown_file(filepath: str) -> int:
    """处理单个Markdown文件的标题（直接覆盖原文件）"""
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    lines = content.split('\n')
    processed_lines = []

    level_map = {
        1: '#',
        2: '##',
        3: '###',
        4: '####'
    }

    for line in lines:
        stripped_line = line.strip()

        if stripped_line.startswith('#'):
            title_content = stripped_line.lstrip('#').strip()

            if len(title_content) >= 3 and re.match(r'^第\d+章', title_content[:3]):
                level = 1
            elif title_content in ["继续阅读", "参考文献", "习题", "本章概要"]:
                level = 2
            elif len(title_content) >= 5 and re.match(r'^\d+\.\d+\.\d+', title_content[:5]):
                level = 3
            elif len(title_content) >= 3 and re.match(r'^\d+\.\d+', title_content[:3]):
                level = 2
            else:
                level = 4

            new_title = f"{level_map[level]} {title_content}"
            processed_lines.append(new_title)
        else:
            processed_lines.append(line)

    with open(filepath, 'w', encoding='utf-8') as f:
        f.write('\n'.join(processed_lines))

    return len([line for line in processed_lines if line.strip().startswith('#')])


def process_directory(input_dir: str = '.') -> dict:
    """处理目录中的所有Markdown文件"""
    if not os.path.exists(input_dir):
        print(f"❌ 目录不存在: {input_dir}")
        return {"success": 0, "failed": 0}

    md_files = [
        f for f in os.listdir(input_dir)
        if f.endswith('.md')
    ]

    if not md_files:
        print(f"❌ 目录中没有找到Markdown文件: {input_dir}")
        return {"success": 0, "failed": 0}

    print(f"\n📁 发现 {len(md_files)} 个Markdown文件")
    print("=" * 60)

    success_count = 0
    failed_count = 0

    for filename in md_files:
        filepath = os.path.join(input_dir, filename)

        try:
            count = process_markdown_file(filepath)
            print(f"✅ {filename} -> 完成 ({count} 个标题)")
            success_count += 1
        except Exception as e:
            print(f"❌ {filename} -> 失败: {e}")
            failed_count += 1

    print("=" * 60)
    print(f"📊 完成: 成功 {success_count} 个, 失败 {failed_count} 个")

    return {"success": success_count, "failed": failed_count}


def main():
    input_dir = sys.argv[1] if len(sys.argv) > 1 else '.'

    print("=" * 60)
    print("📚 Markdown标题处理工具")
    print("=" * 60)
    print(f"📂 处理目录: {os.path.abspath(input_dir)}")

    process_directory(input_dir)


if __name__ == "__main__":
    main()
