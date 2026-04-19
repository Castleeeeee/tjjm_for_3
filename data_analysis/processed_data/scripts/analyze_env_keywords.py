import pandas as pd
import jieba
import os
import re

# 设置输入输出路径
input_file = r'd:\tjjm_for_3\data_analysis\original_data\gov_report.csv'
output_file = r'd:\tjjm_for_3\data_analysis\processed_data\gov_report_env_stats.csv'

# 定义环保相关关键词库（分维度）
keywords_dict = {
    '污染防治': [
        '污染', '减排', '排污', '治污', '烟粉尘', '二氧化硫', '氮氧化物', 'COD', '氨氮', 
        '污水', '废气', '废水', '废渣', '大气防治', '蓝天保卫战', '雾霾', '水质', 
        '黑臭水体', '土壤修复', '垃圾分类', '生活垃圾', '工业固废', '危险废物'
    ],
    '绿色低碳': [
        '节能', '降耗', '能耗', '低碳', '绿色', '碳达峰', '碳中和', '循环经济', 
        '清洁生产', '可再生能源', '新能源', '光伏', '风电', '节约用地', '集约利用', 
        '产业升级', '落后产能', '淘汰关闭', '绿色建筑'
    ],
    '生态保护': [
        '生态', '植被', '森林', '林地', '湿地', '绿化', '造林', '退耕还林', '封山育林', 
        '自然保护区', '水土保持', '荒漠化', '生物多样性', '生态补偿', '山水林田湖草', '修复', '宜居'
    ],
    '环境监管': [
        '环境', '环保', '环境保护', '环评', '达标', '监察', '督查', '执法', '整治', 
        '考核', '约束性指标', '法律法规', '强制性', '问责', '环保补贴', '环保税'
    ]
}

# 扁平化关键词列表用于快速匹配
all_env_keywords = [word for words in keywords_dict.values() for word in words]

def count_keywords(text):
    if not isinstance(text, str):
        return 0, {k: 0 for k in keywords_dict.keys()}
    
    # 使用 jieba 精确模式分词
    words = jieba.lcut(text)
    
    # 初始化各维度统计
    dim_counts = {k: 0 for k in keywords_dict.keys()}
    total_env_count = 0
    
    for word in words:
        # 统计各维度词频
        for dim, kws in keywords_dict.items():
            if word in kws:
                dim_counts[dim] += 1
                total_env_count += 1
                
    return total_env_count, dim_counts

def main():
    print(f"正在读取数据: {input_file}")
    # 读取数据，尝试 utf-8 编码，如果不行再尝试 gbk
    try:
        df = pd.read_csv(input_file, encoding='utf-8')
    except UnicodeDecodeError:
        df = pd.read_csv(input_file, encoding='gbk')

    print(f"数据读取完成，共 {len(df)} 行数据。正在进行 NLP 词频统计...")

    # 结果存储
    results = []

    for index, row in df.iterrows():
        text = row['报告全文']
        total_count, dim_counts = count_keywords(text)
        
        # 获取原有的基本信息
        res = {
            '年份': row['年份'],
            '地区': row['地区'],
            '文本总词频': row['文本总词频-精确模式(个)'],
            '环保词频_总计': total_count
        }
        # 添加各维度词频
        for dim, count in dim_counts.items():
            res[f'环保词频_{dim}'] = count
            
        # 计算强度 (Z1) = (环保词频总计 / 文本总词频) * 100
        if row['文本总词频-精确模式(个)'] > 0:
            res['环境规制强度_Z1'] = (total_count / row['文本总词频-精确模式(个)']) * 100
        else:
            res['环境规制强度_Z1'] = 0
            
        results.append(res)
        
        if (index + 1) % 50 == 0:
            print(f"已处理 {index + 1}/{len(df)} 篇报告...")

    # 转换为 DataFrame 并保存
    result_df = pd.DataFrame(results)
    result_df.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"统计完成！结果已保存至: {output_file}")

if __name__ == "__main__":
    main()
