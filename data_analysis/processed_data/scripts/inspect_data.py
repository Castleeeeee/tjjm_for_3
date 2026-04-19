import pandas as pd
import os

files = [
    r'd:\tjjm_for_3\data_analysis\original_data\地区生产总值构成.xlsx',
    r'd:\tjjm_for_3\data_analysis\original_data\常住人口城镇化率（仅2023）+地区生产总值+人均地区生产总值.xlsx',
    r'd:\tjjm_for_3\data_analysis\original_data\第三产业增加值占GDP比重.xlsx',
    r'd:\tjjm_for_3\data_analysis\original_data\生活垃圾清运量+无害化处理量+生活垃圾焚烧厂处理量+生活垃圾卫生填埋厂处理量.xlsx',
    r'd:\tjjm_for_3\data_analysis\original_data\市容环境卫生投资+市容环卫专用车辆设备总数.xlsx',
    r'd:\tjjm_for_3\data_analysis\processed_data\gov_report_env_stats.csv'
]

def inspect_files():
    for f in files:
        print(f"\n{'='*20}")
        print(f"File: {os.path.basename(f)}")
        try:
            if f.endswith('.xlsx'):
                df = pd.read_excel(f, nrows=5)
            else:
                df = pd.read_csv(f, nrows=5)
            print("Columns:", df.columns.tolist())
            print("First 2 rows:")
            print(df.head(2))
        except Exception as e:
            print(f"Error reading {f}: {e}")

if __name__ == "__main__":
    inspect_files()
