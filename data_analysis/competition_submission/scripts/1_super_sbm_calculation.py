import pandas as pd
import numpy as np
import pulp
import os

# --- 配置路径 ---
# 使用相对于脚本位置的路径，提高迁移性
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, 'data', 'ready.csv')
OUTPUT_PATH = os.path.join(BASE_DIR, 'results', 'ready_with_scores.csv')

def calculate_super_sbm(df, input_cols, output_good_cols, output_bad_cols):
    """
    计算包含非期望产出的超效率 SBM 模型 (Non-oriented)
    """
    dmus = df.index.tolist()
    n = len(dmus)
    m = len(input_cols)
    q1 = len(output_good_cols)
    q2 = len(output_bad_cols)
  # 提取数据矩阵并转换为可写副本
    X = df[input_cols].values.T.copy()
    Yg = df[output_good_cols].values.T.copy()
    Yb = df[output_bad_cols].values.T.copy()
    
    # 数据清洗：处理 0 或 NaN，避免除以 0 的错误
    # 在 DEA 中，通常用一个极小的正数 epsilon 替代 0
    epsilon = 1e-6
    X[X <= 0] = epsilon
    Yg[Yg <= 0] = epsilon
    Yb[Yb <= 0] = epsilon
    X = np.nan_to_num(X, nan=epsilon)
    Yg = np.nan_to_num(Yg, nan=epsilon)
    Yb = np.nan_to_num(Yb, nan=epsilon)
    
    scores = []
    
    for i in range(n):
        prob = pulp.LpProblem(f"Super_SBM_DMU_{i}", pulp.LpMinimize)
        t = pulp.LpVariable("t", lowBound=0)
        lambdas = [pulp.LpVariable(f"lambda_{j}", lowBound=0) for j in range(n)]
        s_minus = [pulp.LpVariable(f"s_minus_{k}", lowBound=0) for k in range(m)]
        s_plus_g = [pulp.LpVariable(f"s_plus_g_{r}", lowBound=0) for r in range(q1)]
        s_plus_b = [pulp.LpVariable(f"s_plus_b_{l}", lowBound=0) for l in range(q2)]
        
        # 目标函数：最小化转换后的投入松弛比例
        prob += t - (1/m) * pulp.lpSum([s_minus[k] / X[k, i] for k in range(m)])
        
        # 约束条件 1: 分母标准化约束
        prob += t + (1/(q1 + q2)) * (
            pulp.lpSum([s_plus_g[r] / Yg[r, i] for r in range(q1)]) + 
            pulp.lpSum([s_plus_b[l] / Yb[l, i] for l in range(q2)])
        ) == 1
        
        # --- 约束条件 2: 投入约束 (sum(lambda_j * x_jk) - s_minus <= t * x_ik) ---
        for k in range(m):
            prob += pulp.lpSum([lambdas[j] * X[k, j] for j in range(n) if j != i]) <= t * X[k, i] - s_minus[k]
            
        # --- 约束条件 3: 期望产出约束 (sum(lambda_j * y_jk^g) + s_plus_g >= t * y_ik^g) ---
        for r in range(q1):
            prob += pulp.lpSum([lambdas[j] * Yg[r, j] for j in range(n) if j != i]) >= t * Yg[r, i] + s_plus_g[r]
            
        # --- 约束条件 4: 非期望产出约束 (sum(lambda_j * y_jk^b) - s_plus_b <= t * y_ik^b) ---
        for l in range(q2):
            prob += pulp.lpSum([lambdas[j] * Yb[l, j] for j in range(n) if j != i]) <= t * Yb[l, i] - s_plus_b[l]

        prob.solve(pulp.PULP_CBC_CMD(msg=0))
        
        status = pulp.LpStatus[prob.status]
        if status == 'Optimal':
            scores.append(pulp.value(prob.objective))
        else:
            print(f"Warning: DMU {i} (Index) solver status: {status}")
            scores.append(np.nan)
            
    return scores

def main():
    if not os.path.exists(DATA_PATH):
        print(f"找不到数据文件: {DATA_PATH}")
        return

    df_ready = pd.read_csv(DATA_PATH)
    inputs = ['生活垃圾清运量（万吨）', '市容环卫专用车辆设备总数（台）', '市容环境卫生投资（万元）']
    outputs_good = ['无害化处理量（万吨）', '生活垃圾焚烧厂处理量（万吨）']
    outputs_bad = ['生活垃圾卫生填埋场处理量（万吨）']

    print("正在计算各年份 Super-SBM 效率...")
    results_list = []
    for year in sorted(df_ready['年份'].unique()):
        year_mask = df_ready['年份'] == year
        year_data = df_ready[year_mask].copy()
        scores = calculate_super_sbm(year_data, inputs, outputs_good, outputs_bad)
        year_data['Super_SBM_Score'] = scores
        results_list.append(year_data)
        print(f"年份 {year} 计算完成。")

    df_final = pd.concat(results_list)

    # 预处理：对自变量取对数处理 (ln)
    # 使用 find_column 匹配列名以增强鲁棒性
    def find_column(target, columns):
        target_clean = "".join(target.split())
        for col in columns:
            col_clean = "".join(col.split())
            if target_clean == col_clean:
                return col
        for col in columns:
            col_clean = "".join(col.split())
            if target_clean in col_clean or col_clean in target_clean:
                return col
        return None

    z_cols_raw = ['环境规制强度_Z1', '人均地区生产总值（元）', '第三产业增加值占GDP比重（%）']
    z_cols = [find_column(c, df_final.columns) for c in z_cols_raw]
    z_cols = [c for c in z_cols if c is not None]
    y_col = find_column('Super_SBM_Score', df_final.columns)

    # 填充 Super_SBM_Score 的缺失值为 1 (DEA 中的有效前沿)
    if y_col and y_col in df_final.columns:
        df_final[y_col] = df_final[y_col].fillna(1.0)
    
    # 预处理：删除包含 NaN 的行
    if z_cols and y_col:
        df_final = df_final.dropna(subset=z_cols + [y_col])

    df_final.to_csv(OUTPUT_PATH, index=False, encoding='utf-8-sig')
    print(f"结果已保存至: {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
