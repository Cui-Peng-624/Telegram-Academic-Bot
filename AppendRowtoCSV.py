import pandas as pd
import os

def append_row_to_csv(csv_file, row_data):
    # 检查文件是否存在
    if os.path.exists(csv_file):
        # 读取现有的 CSV 文件
        df = pd.read_csv(csv_file)
    else:
        # 如果文件不存在，创建一个新的 DataFrame
        df = pd.DataFrame(columns=row_data.keys())
    
    # 将新的数据行转换为 DataFrame 并添加到现有的 DataFrame 中
    new_row_df = pd.DataFrame([row_data])
    df = pd.concat([df, new_row_df], ignore_index=True)
    
    # 将更新后的 DataFrame 写回 CSV 文件
    df.to_csv(csv_file, index=False)

# 示例使用
# csv_file = 'data/data_misclassified_user_requests/other.csv'
# new_row = {'Text': 'New Title', 'Label': 'http://example.com'}

# append_row_to_csv(csv_file, new_row)