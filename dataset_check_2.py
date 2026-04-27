import json
import os
import re


def check_json_array(file_path):
    print(f"🚀 开始解析数据集: {file_path}")

    if not os.path.exists(file_path):
        print("❌ 错误：找不到文件，请确认路径。")
        return

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        print(f"❌ JSON 语法错误：{e}")
        print("💡 建议：检查数组元素之间是否漏掉了逗号，或者结尾是否多了逗号。")
        return

    if not isinstance(data, list):
        print(f"❌ 格式错误：期望得到 [{{...}}, {{...}}] 数组，但解析到了 {type(data)}")
        return

    print(f"📊 检测到 {len(data)} 条数据，开始深度扫描...\n")

    error_count = 0
    # 状态标签的正则匹配
    state_pattern = re.compile(r'<state>\{.*\}</state>')

    for i, entry in enumerate(data):
        line_info = f"索引 [{i}]"

        # 1. 结构检查
        required_keys = ["instruction", "input", "output"]
        for key in required_keys:
            if key not in entry:
                print(f"🚩 {line_info}: 缺失核心字段 '{key}'")
                error_count += 1
                continue

            val = entry[key]
            # 2. 类型检查 (防止出现非字符串导致的 Tensor 报错)
            if not isinstance(val, str):
                print(f"🚩 {line_info}: 字段 '{key}' 不是字符串，当前为 {type(val)}")
                error_count += 1
                continue

            # 3. 空值检查
            if not val.strip():
                print(f"🚩 {line_info}: 字段 '{key}' 内容为空，这会导致训练时产生零维张量！")
                error_count += 1

            # 4. 非法不可见字符检查 (CUBLAS 报错的常见诱因)
            if "\x00" in val or "\ufffe" in entry[key]:
                print(f"🚩 {line_info}: 字段 '{key}' 包含非法 Unicode 字符 (NULL 或 BOM)")
                error_count += 1

        # 5. 状态标签专项检查
        output_text = entry.get("output", "")
        if "<state>" in output_text:
            if "</state>" not in output_text:
                print(f"🚩 {line_info}: 状态标签未闭合 (缺少 </state>)")
                error_count += 1
            else:
                # 尝试解析 JSON 状态内容
                try:
                    state_content = output_text.split("<state>")[1].split("</state>")[0]
                    json.loads(state_content)
                except Exception:
                    print(f"⚠️ {line_info}: <state> 内的 JSON 格式可能不标准")

    print("-" * 30)
    if error_count == 0:
        print("✅ 检查通过！数据集格式完美。")
        print("💡 如果依然报错，请删除 'unsloth_compiled_cache' 文件夹后重试。")
    else:
        print(f"❌ 发现 {error_count} 处潜在问题，请修复后再进行微调。")


if __name__ == "__main__":
    # 确保文件名和你本地一致
    check_json_array("dataset.json")