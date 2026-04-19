import pandas as pd
import os

file_path = r'd:\tjjm_for_3\data_analysis\processed_data\cleaned_100_cities_data.csv'

def check_year_continuity():
    if not os.path.exists(file_path):
        print(f"文件不存在: {file_path}")
        return

    df = pd.read_csv(file_path)
    
    # 定义预期的年份范围
    expected_years = set(range(2013, 2024))
    
    city_year_info = df.groupby('地区')['年份'].apply(set).reset_index()
    
    missing_year_cities = []
    
    for _, row in city_year_info.iterrows():
        city = row['地区']
        actual_years = row['年份']
        missing = expected_years - actual_years
        
        if missing:
            missing_year_cities.append({
                '城市': city,
                '实际年数': len(actual_years),
                '缺失年份': sorted(list(missing))
            })
    
    if not missing_year_cities:
        print("所有 100 个城市在 2013-2023 年的数据都是完整的（每个城市都有 11 年数据）。")
    else:
        print(f"共有 {len(missing_year_cities)} 个城市存在年份缺失：")
        print("-" * 50)
        report_df = pd.DataFrame(missing_year_cities)
        print(report_df.to_string(index=False))

if __name__ == "__main__":
    check_year_continuity()
