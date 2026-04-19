import pandas as pd
import os

file_path = r'd:\tjjm_for_3\data_analysis\processed_data\final_merged_data.csv'

def analyze_missing_values():
    if not os.path.exists(file_path):
        print(f"文件不存在: {file_path}")
        return

    df = pd.read_csv(file_path)
    total_rows = len(df)
    
    print(f"数据总行数: {total_rows}")
    print("\n--- 各列缺失情况 (按缺失比例排序) ---")
    missing_info = df.isnull().sum()
    missing_pct = (missing_info / total_rows) * 100
    missing_df = pd.DataFrame({'缺失数': missing_info, '缺失比例(%)': missing_pct})
    print(missing_df.sort_values(by='缺失比例(%)', ascending=False))

    print("\n--- 按年份统计缺失比例 ---")
    # 排除 NLP 指标，主要看合并进来的社会经济指标
    eco_cols = [
        '人均地区生产总值（元）', '地区生产总值（万元）', '常住人口城镇化率（%）',
        '第三产业增加值占GDP比重（%）', '生活垃圾清运量（万吨）', '无害化处理量（万吨）',
        '市容环境卫生投资（万元）'
    ]
    # 只取存在的列
    available_eco_cols = [c for c in eco_cols if c in df.columns]
    
    if available_eco_cols:
        yearly_missing = df.groupby('年份')[available_eco_cols].apply(lambda x: x.isnull().mean() * 100)
        print(yearly_missing)

    print("\n--- 缺失值最严重的 10 个城市 ---")
    city_missing = df.groupby('地区')[available_eco_cols].apply(lambda x: x.isnull().mean() * 100).mean(axis=1)
    print(city_missing.sort_values(ascending=False).head(10))

if __name__ == "__main__":
    analyze_missing_values()
