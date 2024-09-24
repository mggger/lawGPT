import os
import pandas as pd

# 读取parquet文件
df = pd.read_parquet('dataset.parquet')

# 确保输出目录存在
output_dir = './ragtest/input'
os.makedirs(output_dir, exist_ok=True)

# 遍历每一行数据
for _, row in df.iterrows():
    # 构造文件名
    filename = f"{row['id']}.txt"
    filepath = os.path.join(output_dir, filename)

    # 将text内容写入文件
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(row['text'].replace('\\n', '\n'))

print("处理完成。文件已保存到 ./ragtest/input 目录中。")