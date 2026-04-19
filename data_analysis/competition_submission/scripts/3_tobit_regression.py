import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.base.model import GenericLikelihoodModel
import os

# --- 配置路径 ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, 'results', 'ready_with_scores.csv')

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

def main():
    if not os.path.exists(DATA_PATH):
        print(f"找不到数据文件: {DATA_PATH}，请先运行 1_super_sbm_calculation.py")
        return

    df_reg = pd.read_csv(DATA_PATH)
    
    # 彻底清理列名（去除所有空格、换行符、特殊不可见字符）
    def clean_column_name(name):
        return "".join(name.split())
    
    df_reg.columns = [clean_column_name(col) for col in df_reg.columns]
    
    # 模糊匹配列名
    def find_column(target, columns):
        target_clean = clean_column_name(target)
        for col in columns:
            if target_clean == col:
                return col
        for col in columns:
            if target_clean in col or col in target_clean:
                return col
        return None

    z_cols_raw = ['环境规制强度_Z1', '人均地区生产总值（元）', '第三产业增加值占GDP比重（%）']
    z_cols = [find_column(c, df_reg.columns) for c in z_cols_raw]
    z_cols = [c for c in z_cols if c is not None]
    y_col = find_column('Super_SBM_Score', df_reg.columns)

    # 填充因变量缺失值为 1 (有效前沿)
    if y_col and y_col in df_reg.columns:
        df_reg[y_col] = df_reg[y_col].fillna(1.0)

    # 预处理：删除包含 NaN 的行
    df_reg = df_reg.dropna(subset=z_cols + [y_col])
    
    if len(df_reg) == 0:
        print("错误：清洗后没有剩余样本！请检查原始数据中的缺失值或列名匹配情况。")
        return
    
    # 预处理：对数化
    for col in z_cols:
        df_reg[f'ln_{col}'] = np.log(df_reg[col] + 1)

    Y = df_reg[y_col]
    X = df_reg[[f'ln_{col}' for col in z_cols]]
    X = sm.add_constant(X)

    print(f"\n正在执行面板 Tobit 回归分析... (样本量: {len(df_reg)})")
    model = Tobit(Y, X)
    results = model.fit()

    print(results.summary(
        yname='Super-SBM Efficiency Score',
        xname=['Constant'] + [f'ln_{col}' for col in z_cols] + ['Sigma']
    ))

    # 保存摘要到文本文件
    summary_path = os.path.join(BASE_DIR, 'results', 'tobit_regression_summary.txt')
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write(results.summary().as_text())
    print(f"回归报告已保存至: {summary_path}")

if __name__ == "__main__":
    main()
