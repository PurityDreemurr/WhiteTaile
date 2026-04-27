import json
import re


def validate_json_file(file_path):
    try:
        # 1. 验证基础 JSON 语法
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        print(f"✅ 基础 JSON 语法验证通过！检测到 {len(data)} 条数据。")

        # 2. 验证业务逻辑格式（检查键值对和 state 标签）
        error_count = 0
        state_pattern = re.compile(r'<state>\{"mood": \d+, "emoji": ".*"\}</state>$')

        for index, item in enumerate(data):
            # 检查必要的键是否存在
            keys = ["instruction", "input", "output"]
            for key in keys:
                if key not in item:
                    print(f"❌ 第 {index} 条数据缺失键: '{key}'")
                    error_count += 1

            # 检查 output 末尾是否有正确的 state 标签
            output_content = item.get("output", "").strip()
            if not state_pattern.search(output_content):
                print(f"⚠️ 第 {index} 条数据的 output 末尾标签格式不规范。")
                print(f"   内容片段: {output_content[-50:]}")
                error_count += 1

        if error_count == 0:
            print("✨ 业务逻辑验证也完全正确！这是一份完美的“皓尾”数据集。")
        else:
            print(f"🚩 逻辑验证完成，共发现 {error_count} 处不规范。")

    except FileNotFoundError:
        print("❌ 错误：找不到指定的文件，请检查路径。")
    except json.JSONDecodeError as e:
        print(f"❌ JSON 语法错误：文件解析失败！")
        print(f"错误位置：行 {e.lineno}, 列 {e.colno}")
        print(f"错误详情：{e.msg}")
    except Exception as e:
        print(f"❌ 发生未知错误：{e}")


# 将 'haowei_data.json' 替换为你实际的文件名
validate_json_file('dataset.json')