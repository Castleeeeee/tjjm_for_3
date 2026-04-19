import pandas as pd
import os

full_data_path = r'd:\tjjm_for_3\data_analysis\processed_data\final_merged_data.csv'
cleaned_data_path = r'd:\tjjm_for_3\data_analysis\processed_data\cleaned_100_cities_data.csv'
report_path = r'd:\tjjm_for_3\data_analysis\processed_data\city_filtering_report.txt'

def generate_city_report():
    # 读取原始数据和清洗后的数据
    df_full = pd.read_csv(full_data_path)
    df_cleaned = pd.read_csv(cleaned_data_path)
    
    # 获取唯一的城市列表
    all_cities = sorted(df_full['地区'].unique())
    kept_cities = sorted(df_cleaned['地区'].unique())
    removed_cities = sorted(list(set(all_cities) - set(kept_cities)))
    
    # 核心指标定义（用于显示缺失情况）
    core_eco_cols = [
        '人均地区生产总值（元）', '地区生产总值（万元）', 
        '第三产业增加值占GDP比重（%）', '无害化处理量（万吨）', 
        '生活垃圾清运量（万吨）', '市容环境卫生投资（万元）'
    ]
    
    # 计算每个城市的平均完整率
    city_completeness = df_full.groupby('地区')[core_eco_cols].apply(lambda x: x.notnull().mean().mean() * 100)

    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("城市筛选报告\n")
        f.write("============\n\n")
        f.write(f"总城市数: {len(all_cities)}\n")
        f.write(f"保留城市数: {len(kept_cities)}\n")
        f.write(f"删除城市数: {len(removed_cities)}\n\n")
        
        f.write("一、保留的 100 个城市名单\n")
        f.write("--------------------------\n")
        for i, city in enumerate(kept_cities):
            score = city_completeness[city]
            f.write(f"{i+1:3d}. {city:<10} (数据完整度: {score:>6.2f}%)\n")
            if (i+1) % 5 == 0: f.write("\n")
            
        f.write("\n\n二、被删除的城市名单及原因分析\n")
        f.write("------------------------------\n")
        f.write("主要删除原因：在提供的经济统计 Excel 源文件中缺失关键指标或完全无记录。\n\n")
        for i, city in enumerate(removed_cities):
            score = city_completeness[city]
            f.write(f"{i+1:3d}. {city:<10} (数据完整度: {score:>6.2f}%)\n")

    print(f"报告已生成: {report_path}")

if __name__ == "__main__":
    generate_city_report()
