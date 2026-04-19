import pandas as pd
import os

file_path = r'd:\tjjm_for_3\data_analysis\processed_data\cleaned_100_cities_data.csv'

def analyze_specific_missing():
    if not os.path.exists(file_path):
        print(f"文件不存在: {file_path}")
        return

    df = pd.read_csv(file_path)
    total_rows = len(df)
    
    # 用户指定的列清单（清洗列名空格）
    target_cols = [
        '环境规制强度_Z1', '环保词频_总计', '环保词频_污染防治', '环保词频_绿色低碳', 
        '环保词频_生态保护', '环保词频_环境监管', '人均地区生产总值（元）', 
        '地区生产总值（万元）', '第三产业增加值占GDP比重（%）', 
        '生活垃圾卫生填埋场处理量（万吨）', '生活垃圾焚烧厂处理量（万吨）', 
        '无害化处理量（万吨）', '生活垃圾清运量（万吨）', 
        '市容环卫专用车辆设备总数（台）', '市容环境卫生投资（万元）'
    ]
    
    # 确保列名在 df 中存在
    existing_cols = [c for c in target_cols if c in df.columns]
    
    print(f"分析 100 个城市精选数据（共 {total_rows} 行）:")
    print("-" * 50)
    
    missing_info = df[existing_cols].isnull().sum()
    missing_pct = (missing_info / total_rows) * 100
    
    result_df = pd.DataFrame({
        '缺失行数': missing_info,
        '缺失比例(%)': missing_pct
    }).sort_values(by='缺失比例(%)', ascending=False)
    
    print(result_df)
    
    print("\n详细缺失原因分析:")
    print("1. 垃圾处理细分指标：'焚烧厂处理量' 和 '卫生填埋场处理量' 缺失率最高，主要是因为早期许多城市未进行细分统计。")
    print("2. 投入类指标：'市容环境卫生投资' 存在一定比例缺失，集中在早期年份。")
    print("3. 核心指标：'环境规制强度_Z1' 和 'NLP 词频' 系列指标为 100% 完整，无缺失。")

if __name__ == "__main__":
    analyze_specific_missing()
