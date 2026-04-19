import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.base.model import GenericLikelihoodModel
import os

# --- 配置路径 ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, 'results', 'ready_with_scores.csv')
REPORT_PATH = os.path.join(BASE_DIR, 'results', 'enhanced_tobit_report.txt')

# --- Tobit 模型类定义 ---
class Tobit(GenericLikelihoodModel):
    def __init__(self, endog, exog, **kwargs):
        super(Tobit, self).__init__(endog, exog, **kwargs)

    def nloglikeobs(self, params):
        beta = params[:-1]
        sigma = params[-1]
        mu = np.dot(self.exog, beta)
        from scipy.stats import norm
        z = (self.endog - mu) / sigma
        prob_y_pos = norm.logpdf(z) - np.log(sigma)
        prob_y_zero = norm.logcdf(-mu / sigma)
        ll = np.where(self.endog > 0, prob_y_pos, prob_y_zero)
        return -ll

    def fit(self, start_params=None, maxiter=10000, **kwargs):
        if start_params is None:
            ols_res = sm.OLS(self.endog, self.exog).fit()
            start_params = np.append(ols_res.params, ols_res.resid.std())
        return super(Tobit, self).fit(start_params=start_params, maxiter=maxiter, **kwargs)

# --- 区域划分映射 ---
EAST = ['北京市', '天津市', '上海市', '石家庄市', '唐山市', '保定市', '廊坊市', '秦皇岛市', '张家口市', '承德市', '沧州市', '衡水市', '济南市', '青岛市', '烟台市', '潍坊市', '威海市', '南京市', '苏州市', '无锡市', '常州市', '镇江市', '扬州市', '泰州市', '南通市', '盐城市', '连云港市', '徐州市', '杭州市', '宁波市', '温州市', '嘉兴市', '湖州市', '绍兴市', '金华市', '舟山市', '台州市', '福州市', '厦门市', '泉州市', '广州市', '深圳市', '珠海市', '佛山市', '惠州市', '东莞市', '中山市', '江门市', '肇庆市', '海口市', '三亚市', '临沂市', '淄博市', '济宁市', '菏泽市', '汕头市', '湛江市', '茂名市', '淮安市', '宿迁市']
CENTRAL = ['太原市', '大同市', '郑州市', '洛阳市', '许昌市', '合肥市', '芜湖市', '马鞍山市', '铜陵市', '安庆市', '滁州市', '池州市', '宣城市', '武汉市', '襄阳市', '宜昌市', '长沙市', '株洲市', '衡阳市', '南昌市', '九江市', '阳泉市', '长治市', '新余市', '吉安市', '抚州市', '蚌埠市', '阜阳市', '荆州市']
WEST = ['呼和浩特市', '包头市', '鄂尔多斯市', '南宁市', '柳州市', '桂林市', '重庆市', '成都市', '绵阳市', '宜宾市', '贵阳市', '遵义市', '昆明市', '曲靖市', '拉萨市', '西宁市', '西安市', '宝鸡市', '兰州市', '银川市', '乌鲁木齐市', '南充市', '赤峰市', '德阳市', '泸州市', '榆林市', '梧州市', '玉林市']
NORTHEAST = ['沈阳市', '大连市', '盘锦市', '长春市', '哈尔滨市', '吉林市']

def get_region(city):
    if city in EAST: return 'East'
    if city in CENTRAL: return 'Central'
    if city in WEST: return 'West'
    if city in NORTHEAST: return 'Northeast'
    return 'Other'

def run_regression(df, y_col, x_cols, title):
    print(f"\n>>> 正在运行: {title}")
    Y = df[y_col]
    X = df[x_cols]
    X = sm.add_constant(X)
    model = Tobit(Y, X)
    results = model.fit(disp=0)
    return results

def main():
    if not os.path.exists(DATA_PATH):
        print("找不到数据文件！")
        return

    df = pd.read_csv(DATA_PATH)
    
    # 1. 数据预处理
    # 清理列名
    df.columns = ["".join(c.split()) for c in df.columns]
    
    # 定义核心变量名（清理后）
    z1_col = '环境规制强度_Z1'
    gdp_col = '人均地区生产总值（元）'
    third_col = '第三产业增加值占GDP比重（%）'
    y_col = 'Super_SBM_Score'
    
    # 分维度列名（模糊匹配）
    dim_cols = [c for c in df.columns if '环保词频_' in c and '总计' not in c]
    
    # 填充缺失值
    df[y_col] = df[y_col].fillna(1.0)
    df = df.dropna(subset=[z1_col, gdp_col, third_col, y_col])
    
    # 2. 深度挖掘 1：引入滞后项 (Lag Effect)
    df = df.sort_values(['地区', '年份'])
    df['L_Z1'] = df.groupby('地区')[z1_col].shift(1)
    
    # 3. 对数化处理
    for col in [z1_col, 'L_Z1', gdp_col, third_col] + dim_cols:
        df[f'ln_{col}'] = np.log(df[col] + 1)
    
    # 4. 区域标记
    df['Region'] = df['地区'].apply(get_region)
    
    # 5. 年份固定效应 (Year Dummies)
    year_dummies = pd.get_dummies(df['年份'], prefix='year', drop_first=True).astype(int)
    df = pd.concat([df, year_dummies], axis=1)
    year_cols = year_dummies.columns.tolist()

    all_results = []

    # --- 回归 1: 基准回归 (含年份固定效应) ---
    res_base = run_regression(df, y_col, [f'ln_{z1_col}', f'ln_{gdp_col}', f'ln_{third_col}'] + year_cols, "基准回归 (含年份FE)")
    all_results.append(("1. Baseline with Year FE", res_base))

    # --- 回归 2: 滞后项回归 (解决滞后性与部分内生性) ---
    df_lag = df.dropna(subset=['ln_L_Z1'])
    res_lag = run_regression(df_lag, y_col, ['ln_L_Z1', f'ln_{gdp_col}', f'ln_{third_col}'] + year_cols, "滞后一期回归 (L.Z1)")
    all_results.append(("2. Lagged Z1 (L.Z1)", res_lag))

    # --- 回归 3: 分维度挖掘 (Dimension Analysis) ---
    # 重点关注 '环境监管' 维度
    env_reg_col = [c for c in dim_cols if '环境监管' in c][0]
    res_dim = run_regression(df, y_col, [f'ln_{env_reg_col}', f'ln_{gdp_col}', f'ln_{third_col}'] + year_cols, "分维度分析 (环境监管维度)")
    all_results.append(("3. Dimension Analysis (Regulation)", res_dim))

    # --- 回归 4: 异质性分析 (东部 vs 非东部) ---
    res_east = run_regression(df[df['Region'] == 'East'], y_col, [f'ln_{z1_col}', f'ln_{gdp_col}', f'ln_{third_col}'], "异质性分析 (东部地区)")
    res_noneast = run_regression(df[df['Region'] != 'East'], y_col, [f'ln_{z1_col}', f'ln_{gdp_col}', f'ln_{third_col}'], "异质性分析 (非东部地区)")
    all_results.append(("4a. Heterogeneity (East)", res_east))
    all_results.append(("4b. Heterogeneity (Non-East)", res_noneast))

    # 6. 保存增强版报告
    with open(REPORT_PATH, 'w', encoding='utf-8') as f:
        f.write("增强版实证分析报告 - 2026年统计建模大赛专用\n")
        f.write("============================================\n\n")
        for title, res in all_results:
            f.write(f"\nMODEL: {title}\n")
            f.write("-" * 40 + "\n")
            f.write(res.summary().as_text())
            f.write("\n\n")
    
    print(f"\n增强版回归分析完成！详细学术报告已保存至: {REPORT_PATH}")

if __name__ == "__main__":
    main()
