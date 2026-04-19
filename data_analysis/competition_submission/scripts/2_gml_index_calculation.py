import pandas as pd
import numpy as np
import pulp
import os

# --- 配置路径 ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, 'data', 'ready.csv')
OUTPUT_PATH = os.path.join(BASE_DIR, 'results', 'gml_results.csv')

def calculate_standard_sbm(target_df, frontier_df, input_cols, output_good_cols, output_bad_cols):
    """
    计算标准 SBM 效率（非超效率），用于 GML 指数分解。
    """
    X_target = target_df[input_cols].values.T.copy()
    Yg_target = target_df[output_good_cols].values.T.copy()
    Yb_target = target_df[output_bad_cols].values.T.copy()
    
    X_front = frontier_df[input_cols].values.T.copy()
    Yg_front = frontier_df[output_good_cols].values.T.copy()
    Yb_front = frontier_df[output_bad_cols].values.T.copy()
    
    # 数据清洗：处理 0 或 NaN，避免除以 0 的错误
    epsilon = 1e-6
    X_target[X_target <= 0] = epsilon
    Yg_target[Yg_target <= 0] = epsilon
    Yb_target[Yb_target <= 0] = epsilon
    X_target = np.nan_to_num(X_target, nan=epsilon)
    Yg_target = np.nan_to_num(Yg_target, nan=epsilon)
    Yb_target = np.nan_to_num(Yb_target, nan=epsilon)
    
    X_front[X_front <= 0] = epsilon
    Yg_front[Yg_front <= 0] = epsilon
    Yb_front[Yb_front <= 0] = epsilon
    X_front = np.nan_to_num(X_front, nan=epsilon)
    Yg_front = np.nan_to_num(Yg_front, nan=epsilon)
    Yb_front = np.nan_to_num(Yb_front, nan=epsilon)
    
    n_front = frontier_df.shape[0]
    n_target = target_df.shape[0]
    m, q1, q2 = len(input_cols), len(output_good_cols), len(output_bad_cols)
    
    eff_scores = []
    
    for i in range(n_target):
        prob = pulp.LpProblem(f"SBM_Target_{i}", pulp.LpMinimize)
        t = pulp.LpVariable("t", lowBound=0)
        lambdas = [pulp.LpVariable(f"lambda_{j}", lowBound=0) for j in range(n_front)]
        s_minus = [pulp.LpVariable(f"s_minus_{k}", lowBound=0) for k in range(m)]
        s_plus_g = [pulp.LpVariable(f"s_plus_g_{r}", lowBound=0) for r in range(q1)]
        s_plus_b = [pulp.LpVariable(f"s_plus_b_{l}", lowBound=0) for l in range(q2)]
        
        prob += t - (1/m) * pulp.lpSum([s_minus[k] / X_target[k, i] for k in range(m)])
        
        prob += t + (1/(q1 + q2)) * (
            pulp.lpSum([s_plus_g[r] / Yg_target[r, i] for r in range(q1)]) + 
            pulp.lpSum([s_plus_b[l] / Yb_target[l, i] for l in range(q2)])
        ) == 1
        
        for k in range(m):
            prob += pulp.lpSum([lambdas[j] * X_front[k, j] for j in range(n_front)]) <= t * X_target[k, i] - s_minus[k]
        for r in range(q1):
            prob += pulp.lpSum([lambdas[j] * Yg_front[r, j] for j in range(n_front)]) >= t * Yg_target[r, i] + s_plus_g[r]
        for l in range(q2):
            prob += pulp.lpSum([lambdas[j] * Yb_front[l, j] for j in range(n_front)]) <= t * Yb_target[l, i] - s_plus_b[l]

        prob.solve(pulp.PULP_CBC_CMD(msg=0))
        eff_scores.append(pulp.value(prob.objective) if pulp.LpStatus[prob.status] == 'Optimal' else np.nan)
        
    return eff_scores

def main():
    if not os.path.exists(DATA_PATH):
        print(f"找不到数据文件: {DATA_PATH}")
        return

    df_all = pd.read_csv(DATA_PATH)
    inputs = ['生活垃圾清运量（万吨）', '市容环卫专用车辆设备总数（台）', '市容环境卫生投资（万元）']
    outputs_good = ['无害化处理量（万吨）', '生活垃圾焚烧厂处理量（万吨）']
    outputs_bad = ['生活垃圾卫生填埋场处理量（万吨）']

    cities = df_all['地区'].unique()
    years = sorted(df_all['年份'].unique())
    results_gml = []

    print("开始计算 GML 指数及其分解...")

    for city in cities:
        city_data = df_all[df_all['地区'] == city].sort_values('年份')
        
        for t in range(len(years) - 1):
            y_curr = years[t]
            y_next = years[t+1]
            
            dmu_t = city_data[city_data['年份'] == y_curr]
            dmu_t1 = city_data[city_data['年份'] == y_next]
            front_t = df_all[df_all['年份'] == y_curr]
            front_t1 = df_all[df_all['年份'] == y_next]
            
            # 检查 dmu_t 和 dmu_t1 是否为空
            if dmu_t.empty or dmu_t1.empty:
                print(f"城市 {city} 在 {y_curr} 或 {y_next} 年份数据缺失，跳过此年份段。")
                continue
                
            # 计算四种效率
            scores_t_t = calculate_standard_sbm(dmu_t, front_t, inputs, outputs_good, outputs_bad)
            scores_t1_t1 = calculate_standard_sbm(dmu_t1, front_t1, inputs, outputs_good, outputs_bad)
            scores_G_t = calculate_standard_sbm(dmu_t, df_all, inputs, outputs_good, outputs_bad)
            scores_G_t1 = calculate_standard_sbm(dmu_t1, df_all, inputs, outputs_good, outputs_bad)

            if not scores_t_t or not scores_t1_t1 or not scores_G_t or not scores_G_t1:
                continue

            rho_t_t = scores_t_t[0]
            rho_t1_t1 = scores_t1_t1[0]
            rho_G_t = scores_G_t[0]
            rho_G_t1 = scores_G_t1[0]
            
            gml = rho_G_t1 / rho_G_t if rho_G_t != 0 else np.nan
            ec = rho_t1_t1 / rho_t_t if rho_t_t != 0 else np.nan
            tc = gml / ec if (ec != 0 and not np.isnan(ec)) else np.nan
            
            results_gml.append({
                '地区': city,
                '年份段': f"{y_curr}-{y_next}",
                'GML': gml,
                'EC': ec,
                'TC': tc
            })
        print(f"城市 {city} 计算完成。")

    df_gml_final = pd.DataFrame(results_gml)
    df_gml_final.to_csv(OUTPUT_PATH, index=False, encoding='utf-8-sig')
    print(f"GML 结果已保存至: {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
