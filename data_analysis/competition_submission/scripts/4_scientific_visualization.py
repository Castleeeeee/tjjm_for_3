import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# --- 配置路径 ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, 'results', 'ready_with_scores.csv')
GML_PATH = os.path.join(BASE_DIR, 'results', 'gml_results.csv')
PLOT_DIR = os.path.join(BASE_DIR, 'results')

# 设置全局科研绘图风格
# 尝试使用多种中文字体，解决不同环境下的乱码问题
font_list = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS', 'DejaVu Sans']
for font in font_list:
    try:
        plt.rcParams['font.sans-serif'] = [font]
        plt.rcParams['axes.unicode_minus'] = False
        # 测试是否生效
        fig = plt.figure()
        plt.close(fig)
        break
    except:
        continue

sns.set_theme(style="whitegrid", rc={"axes.facecolor": (0, 0, 0, 0)})

def plot_gml_trends(df_gml):
    """
    图表 3：GML 指数及其分解（TC/EC）年度趋势折线图
    展示技术进步与效率改善对全要素生产率的贡献对比
    """
    print("正在绘制 GML 指数趋势图...")
    # 计算全样本年度平均值
    annual_avg = df_gml.groupby('年份段')[['GML', 'EC', 'TC']].mean().reset_index()
    
    plt.figure(figsize=(10, 5))
    plt.plot(annual_avg['年份段'], annual_avg['GML'], marker='o', label='GML (TFP Change)', linewidth=2.5, color='#1f77b4')
    plt.plot(annual_avg['年份段'], annual_avg['EC'], marker='s', label='EC (Efficiency Change)', linestyle='--', color='#ff7f0e')
    plt.plot(annual_avg['年份段'], annual_avg['TC'], marker='^', label='TC (Technical Change)', linestyle='-.', color='#2ca02c')
    
    plt.axhline(y=1, color='gray', linestyle=':', alpha=0.7)
    plt.title("2013-2023年城市环境全要素生产率指数(GML)及其分解趋势", fontsize=14)
    plt.ylabel("Index Value")
    plt.xticks(rotation=45)
    plt.legend(frameon=True)
    sns.despine()
    plt.savefig(os.path.join(PLOT_DIR, 'gml_trends_analysis.png'), dpi=300, bbox_inches='tight')
    plt.close()

def plot_correlation_heatmap(df):
    """
    图表 4：核心变量相关性热力图
    展示 Z1、经济变量与效率得分之间的关联性
    """
    print("正在绘制变量相关性热力图...")
    cols = ['Super_SBM_Score', '环境规制强度_Z1', '人均地区生产总值（元）', '第三产业增加值占GDP比重（%）', 
            '无害化处理量（万吨）', '市容环境卫生投资（万元）']
    
    # 清洗列名
    df_clean = df.copy()
    df_clean.columns = ["".join(c.split()) for c in df_clean.columns]
    target_cols = ["".join(c.split()) for c in cols]
    
    corr = df_clean[target_cols].corr()
    
    plt.figure(figsize=(10, 8))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, cmap='RdBu_r', center=0, fmt='.2f', square=True, linewidths=.5)
    plt.title("环境治理效率与社会经济影响因素相关性热图", fontsize=14)
    plt.savefig(os.path.join(PLOT_DIR, 'variable_correlation_heatmap.png'), dpi=300, bbox_inches='tight')
    plt.close()

def plot_efficiency_evolution(df):
    """图表 1：效率演进山峦图"""
    print("正在绘制效率演进山峦图...")
    pal = sns.cubehelix_palette(10, rot=-.25, light=.7)
    g = sns.FacetGrid(df, row="年份", hue="年份", aspect=15, height=.75, palette=pal)
    g.map(sns.kdeplot, "Super_SBM_Score", bw_adjust=.5, clip_on=False, fill=True, alpha=1, linewidth=1.5)
    g.map(sns.kdeplot, "Super_SBM_Score", clip_on=False, color="w", lw=2, bw_adjust=.5)
    g.map(plt.axhline, y=0, lw=2, clip_on=False)

    def label(x, color, label):
        ax = plt.gca()
        ax.text(0, .2, label, fontweight="bold", color=color,
                ha="left", va="center", transform=ax.transAxes)
    g.map(label, "Super_SBM_Score")

    g.figure.subplots_adjust(hspace=-.25)
    g.set_titles("")
    g.set(yticks=[], ylabel="")
    g.despine(bottom=True, left=True)
    plt.xlabel("Super-SBM Efficiency Score", fontsize=12)
    plt.suptitle("2013-2023年城市环境治理效率动态演进趋势", fontsize=15, y=0.98)
    plt.savefig(os.path.join(PLOT_DIR, 'efficiency_evolution.png'), dpi=300, bbox_inches='tight')
    plt.close()

def plot_region_comparison(df):
    """图表 2：地区对比图"""
    print("正在绘制地区对比图...")
    jjj = ['北京市', '天津市', '石家庄市', '唐山市', '保定市']
    csj = ['上海市', '南京市', '苏州市', '无锡市', '杭州市', '宁波市']
    zsj = ['广州市', '深圳市', '佛山市', '东莞市', '珠海市']
    
    def get_region(city):
        if city in jjj: return '京津冀'
        if city in csj: return '长三角'
        if city in zsj: return '珠三角'
        return '其他区域'
    
    df_region = df.copy()
    df_region['经济圈'] = df_region['地区'].apply(get_region)
    
    plt.figure(figsize=(10, 6))
    sns.violinplot(data=df_region, x='经济圈', y='Super_SBM_Score', inner="quart", palette="Pastel1", split=True)
    sns.stripplot(data=df_region, x='经济圈', y='Super_SBM_Score', color="black", size=2, alpha=0.3, jitter=True)
    plt.title("典型经济圈环境治理效率空间差异对比", fontsize=14)
    plt.ylabel("Super-SBM Score")
    sns.despine()
    plt.savefig(os.path.join(PLOT_DIR, 'region_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()

def main():
    if not os.path.exists(DATA_PATH):
        print("找不到效率数据文件，请先运行 1_super_sbm_calculation.py")
        return

    df_final = pd.read_csv(DATA_PATH)
    
    # 运行原有的可视化
    plot_efficiency_evolution(df_final)
    plot_region_comparison(df_final)
    
    # 运行新增的优化可视化
    plot_correlation_heatmap(df_final)
    
    if os.path.exists(GML_PATH):
        df_gml = pd.read_csv(GML_PATH)
        plot_gml_trends(df_gml)
    else:
        print("未找到 GML 结果文件，跳过趋势图绘制。")
        
    print(f"所有科研级图表已保存至: {PLOT_DIR}")

if __name__ == "__main__":
    main()
