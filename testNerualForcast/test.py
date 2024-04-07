import json

# 读取 JSON 数据
with open('input.json', 'r') as file:
    json_data = json.load(file)

# 美化 JSON 数据
beautified_json = json.dumps(json_data, indent=4)

# 将美化后的 JSON 写入新文件
with open('swap.json', 'w') as file:
    file.write(beautified_json)
