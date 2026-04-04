import os
import re
import argparse
from statistics import mean

def extract_last_n_stats(file_path: str, n: int) -> list:
    """
    读取文件，提取最后 n 个 Stat[0] 的值（用于填充各个 Task 的详细分数）
    """
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    matches = re.findall(r"Stat\[0\]:\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)", content)
    
    if not matches:
        raise ValueError(f"No Stat[0] found in: {file_path}")
    if len(matches) < n:
        raise ValueError(f"Found only {len(matches)} Stat[0] entries, but expected at least {n} in: {file_path}")

    return [float(x) for x in matches[-n:]]

def extract_last_stat0(file_path: str) -> float:
    """
    【与参考代码一致】读取文件，只提取最后 1 个 Stat[0] 的值（用于获取该 Step 整体的 AP）
    """
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    matches = re.findall(r"Stat\[0\]:\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)", content)
    if not matches:
        raise ValueError(f"No Stat[0] found in: {file_path}")

    return float(matches[-1])

def build_paths():
    """
    构建路径字典，现在需要同时读取每任务(per_task)和整体(overall)的日志
    """
    per_task_paths = {}
    overall_paths = {}
    
    base_dir = "/data1/lsh/VITA_continue/output_2epiter1000/ytvis_2019_moe"
    
    # step0
    per_task_paths[0] = f"{base_dir}/step0/inference/evaluation_results.txt"
    overall_paths[0]  = f"{base_dir}/step0/inference/evaluation_results.txt"
    
    # step1 - step10
    for step in range(1, 11):
        per_task_paths[step] = f"{base_dir}/step{step}/inference/evaluation_results_per_task.txt"
        overall_paths[step]  = f"{base_dir}/step{step}/inference/evaluation_results.txt"
        
    return per_task_paths, overall_paths

def main():
    parser = argparse.ArgumentParser(description="Generate lower triangular matrix for continual learning tasks.")
    parser.add_argument("--root", default="", help="Optional root prefix path.")
    parser.add_argument("--strict", action="store_true", help="Fail if any step file is missing. Default: skip.")
    args = parser.parse_args()

    per_task_paths, overall_paths = build_paths()

    matrix = {}          # 存储 matrix[step][task] 的详细分数
    overall_aas = {}     # 存储每个 step 的整体 AP (来自 evaluation_results.txt)
    errors = {}

    for step in range(11):
        pt_file = os.path.join(args.root, per_task_paths[step]) if args.root else per_task_paths[step]
        ov_file = os.path.join(args.root, overall_paths[step]) if args.root else overall_paths[step]
        
        matrix[step] = {}
        try:
            if not os.path.exists(pt_file):
                raise FileNotFoundError(f"Missing {pt_file}")
            if not os.path.exists(ov_file):
                raise FileNotFoundError(f"Missing {ov_file}")

            # 1. 提取 per_task 详细分数
            num_tasks_to_extract = step + 1
            stats = extract_last_n_stats(pt_file, num_tasks_to_extract)
            for task_id, val in enumerate(stats):
                matrix[step][task_id] = val

            # 2. 提取 overall 整体 AP (与参考代码逻辑一致)
            overall_ap = extract_last_stat0(ov_file)
            overall_aas[step] = overall_ap

        except Exception as e:
            errors[step] = str(e)
            if args.strict:
                raise

    # ================= 打印漂亮的下三角矩阵表格 =================
    header = ["Step"] + [f"Task {t}" for t in range(11)] + ["AA (Avg)"]
    print("".join([f"{h:>10}" for h in header]))
    print("-" * (10 * len(header)))

    valid_step_aas = [] # 用于收集最后求总均值的整体AP集合

    # 打印每一行
    for step in range(11):
        row_str = f"{'step'+str(step):>10}"
        
        if step in errors:
            row_str += f"  [Error: {errors[step]}]"
            print(row_str)
            continue

        for task in range(11):
            if task <= step and task in matrix[step]:
                val = matrix[step][task]
                row_str += f"{val:10.4f}"
            else:
                row_str += f"{'-':>10}"

        # 最后一列：直接使用从 evaluation_results.txt 中提取出的 AP 值
        if step in overall_aas:
            aa = overall_aas[step]
            valid_step_aas.append(aa)
            row_str += f"{aa:10.4f}"
        else:
            row_str += f"{'-':>10}"

        print(row_str)
        
    # ================= 打印最后一行 (总 Mean) =================
    print("-" * (10 * len(header)))
    last_row_str = f"{'Mean':>10}"
    
    # 中间 11 列留空
    for _ in range(11):
        last_row_str += f"{'':>10}"
        
    # 在最后一列计算所有整体 AP 的平均分 (与参考代码的 aa = mean(ap_by_step.values()) 一致)
    if valid_step_aas:
        overall_aa = mean(valid_step_aas)
        last_row_str += f"{overall_aa:10.4f}"
    else:
        last_row_str += f"{'-':>10}"
        
    print(last_row_str)

if __name__ == "__main__":
    main()