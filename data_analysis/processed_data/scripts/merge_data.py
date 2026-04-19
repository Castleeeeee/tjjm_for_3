import pandas as pd
import numpy as np
import os

# 定义路径
original_dir = r'd:\tjjm_for_3\data_analysis\original_data'
processed_dir = r'd:\tjjm_for_3\data_analysis\processed_data'

# 1. 加载 gov_report_env_stats.csv (作为基础表)
base_file = os.path.join(processed_dir, 'gov_report_env_stats.csv')
df_base = pd.read_csv(base_file)

# 标准化城市名称的函数
def normalize_city(city):
    if not isinstance(city, str): return city
    # 去除空格
    city = city.strip()
    # 统一后缀处理：如果结尾没有"市"且不是省份，可以考虑补齐，但为了匹配，建议全部去掉"市"、"地区"、"自治州"等后缀
    for suffix in ['市', '地区', '自治州', '盟']:
        if city.endswith(suffix) and len(city) > 2:
            city = city[:-len(suffix)]
    return city

df_base['地区_归一化'] = df_base['地区'].apply(normalize_city)

# 2. 处理宽表 Excel 文件的通用函数
def process_wide_excel(file_path, indicator_col='Unnamed: 0', city_col='Unnamed: 1'):
    df = pd.read_excel(file_path)
    # 找到年份列 (2013-2023)
    year_cols = [col for col in df.columns if str(col).isdigit()]
    
    # 转换长表
    df_long = df.melt(id_vars=[indicator_col, city_col], value_vars=year_cols, 
                      var_name='年份', value_name='数值')
    
    # 清洗年份和城市
    df_long['年份'] = df_long['年份'].astype(int)
    df_long['地区_归一化'] = df_long[city_col].apply(normalize_city)
    
    # 透视：将指标名称转为列
    df_pivot = df_long.pivot_table(index=['年份', '地区_归一化'], 
                                   columns=indicator_col, 
                                   values='数值', 
                                   aggfunc='first').reset_index()
    return df_pivot

# 3. 处理各 Excel 文件
excel_files = {
    'urbanization': '常住人口城镇化率（仅2023）+地区生产总值+人均地区生产总值.xlsx',
    'tertiary_ratio': '第三产业增加值占GDP比重.xlsx',
    'waste': '生活垃圾清运量+无害化处理量+生活垃圾焚烧厂处理量+生活垃圾卫生填埋厂处理量.xlsx',
    'investment': '市容环境卫生投资+市容环卫专用车辆设备总数.xlsx'
}

merged_df = df_base.copy()

for key, filename in excel_files.items():
    print(f"正在处理: {filename}")
    path = os.path.join(original_dir, filename)
    df_processed = process_wide_excel(path)
    # 合并
    merged_df = pd.merge(merged_df, df_processed, on=['年份', '地区_归一化'], how='left')

# 4. 特殊处理: 地区生产总值构成.xlsx (2022年横截面数据)
print("正在处理: 地区生产总值构成.xlsx")
gdp_comp_path = os.path.join(original_dir, '地区生产总值构成.xlsx')
df_gdp_comp = pd.read_excel(gdp_comp_path, skiprows=4)
# 映射列名 (基于之前的 inspect 结果)
# 城市在第一列，第一产业比重(全市)在第三列，第二产业在第五列，第三产业在第七列
df_gdp_comp = df_gdp_comp.iloc[:, [0, 2, 4, 6]]
df_gdp_comp.columns = ['城市', '第一产业占比(2022)', '第二产业占比(2022)', '第三产业占比(2022)']
df_gdp_comp['地区_归一化'] = df_gdp_comp['城市'].apply(normalize_city)
df_gdp_comp['年份'] = 2022 # 假设这是2022年的数据

# 只取数值列进行合并
df_gdp_comp = df_gdp_comp[['地区_归一化', '年份', '第一产业占比(2022)', '第二产业占比(2022)', '第三产业占比(2022)']]
merged_df = pd.merge(merged_df, df_gdp_comp, on=['年份', '地区_归一化'], how='left')

# 5. 清理最终表
# 去除列名中的多余空格
merged_df.columns = [col.strip() for col in merged_df.columns]

# 删除中间过程的归一化列
# merged_df.drop(columns=['地区_归一化'], inplace=True)

# 按照年份和地区排序
merged_df = merged_df.sort_values(by=['地区', '年份'])

# 保存结果
output_path = os.path.join(processed_dir, 'final_merged_data.csv')
merged_df.to_csv(output_path, index=False, encoding='utf-8-sig')

print(f"数据整合完成！最终表保存至: {output_path}")
print(f"最终表列名: {merged_df.columns.tolist()}")
