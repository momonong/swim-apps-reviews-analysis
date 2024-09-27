import pandas as pd
import glob

# 找出所有符合模式的 CSV 檔案
files = glob.glob("data/grades_*.csv")

sum_count = 0
# 計算每個檔案的行數並打印
for file in files:
    df = pd.read_csv(file)
    row_count = len(df)
    sum_count += row_count
    print(f"{file} 的資料數量: {row_count}筆")

print(f"總資料數量: {sum_count}筆")
