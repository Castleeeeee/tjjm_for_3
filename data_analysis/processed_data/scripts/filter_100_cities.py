import pandas as pd
import os

file_path = r'd:\tjjm_for_3\data_analysis\processed_data\final_merged_data.csv'
output_path = r'd:\tjjm_for_3\data_analysis\processed_data\cleaned_100_cities_data.csv'

def clean_and_filter_100_cities():
    if not os.path.exists(file_path):
        print(f"文件不存在: {file_path}")
        return

    df = pd.read_csv(file_path)
    
    # 1. 定义核心评价指标（这些指标应该在各年份都尽可能完整）
    core_eco_cols = [
        '人均地区生产总值（元）', '地区生产总值（万元）', 
        '第三产业增加值占GDP比重（%）', '无害化处理量（万吨）', 
        '生活垃圾清运量（万吨）', '市容环境卫生投资（万元）',
        '市容环卫专用车辆设备总数（台）'
    ]
    
    # 2. 计算每个城市的完整性得分
    # 得分 = 该城市所有年份中，核心指标非空的比例
    city_scores = df.groupby('地区')[core_eco_cols].apply(lambda x: x.notnull().mean().mean())
    
    # 3. 筛选得分最高的 100 个城市
    top_100_cities = city_scores.sort_values(ascending=False).head(100).index.tolist()
    
    print(f"筛选出的 100 个城市示例: {top_100_cities[:10]}")
    print(f"最低完整性得分: {city_scores.loc[top_100_cities[-1]]:.4f}")

    # 4. 过滤数据
    df_cleaned = df[df['地区'].isin(top_100_cities)].copy()
    
    # 5. 删除缺失率极高（>90%）的列（如仅 2023 或 2022 有数据的列）
    # 但如果用户之后需要这些数据，可以保留。根据“删除缺失很厉害的部分”指令，我们执行删除。
    missing_pct = df_cleaned.isnull().mean()
    cols_to_drop = missing_pct[missing_pct > 0.9].index.tolist()
    print(f"删除缺失率超过 90% 的列: {cols_to_drop}")
    df_cleaned.drop(columns=cols_to_drop, inplace=True)

    # 6. 整理列顺序
    # 保持基本信息在前
    base_cols = ['年份', '地区', '地区_归一化', '文本总词频', '环境规制强度_Z1']
    other_cols = [c for c in df_cleaned.columns if c not in base_cols]
    df_cleaned = df_cleaned[base_cols + other_cols]

    # 按照地区和年份排序
    df_cleaned.sort_values(by=['地区', '年份'], inplace=True)

    # 保存结果
    df_cleaned.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"\n清洗完成！共保留 {len(df_cleaned)} 行数据（100个城市 x 11年左右）。")
    print(f"结果已保存至: {output_path}")

if __name__ == "__main__":
    clean_and_filter_100_cities()
