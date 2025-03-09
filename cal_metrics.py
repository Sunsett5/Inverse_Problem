import csv
from typing import List, Dict
from cp_files import cal_metrics
# 假设cal_metrics函数已经定义
# def cal_metrics(path: str) -> Dict[str, float]:
#     # 这里应该是计算指标的逻辑
#     # 返回一个包含PSNR, SSIM, LPIPS, FID的字典
#     pass

def generate_path(algorithm: str, learned: bool, steps: int, noiseless: bool) -> str:
    # 根据算法名称、是否为learned、steps以及noise的情况生成路径
    # 这里只是一个示例，具体路径生成逻辑需要根据实际情况调整
    noise_type = 'noiseless' if noiseless else 'noisy'
    learned_suffix = '_learned' if learned else ''
    return f"exp/image_samples/{dataset}/{task}_{noise_type}/{algorithm}/{steps}steps{learned_suffix}"

def calculate_and_format_metrics(algorithm: str, learned: bool, steps: int) -> Dict[str, float]:
    # 计算noiseless和noisy情况下的指标
    metrics_noiseless = cal_metrics(generate_path(algorithm, learned, steps, True))
    metrics_noisy = cal_metrics(generate_path(algorithm, learned, steps, False))
    
    # 格式化指标
    formatted_metrics = {
        'PSNR_noiseless': round(metrics_noiseless['PSNR'], 2),
        'SSIM_noiseless': round(metrics_noiseless['SSIM'], 4),
        'LPIPS_noiseless': round(metrics_noiseless['LPIPS'], 4),
        'FID_noiseless': round(metrics_noiseless['FID'], 2),
        'PSNR_noisy': round(metrics_noisy['PSNR'], 2),
        'SSIM_noisy': round(metrics_noisy['SSIM'], 4),
        'LPIPS_noisy': round(metrics_noisy['LPIPS'], 4),
        'FID_noisy': round(metrics_noisy['FID'], 2),
    }
    return formatted_metrics

def compare_and_bold(metrics_a: Dict[str, float], metrics_b: Dict[str, float]) -> Dict[str, str]:
    # 比较两个算法的指标，将更好的值加粗
    better_metrics = {}
    for key in metrics_a:
        if key.startswith('PSNR') or key.startswith('SSIM'):
            # PSNR和SSIM越高越好
            if metrics_a[key] > metrics_b[key]:
                better_metrics[key] = f"<b>{metrics_a[key]}</b>"
            else:
                better_metrics[key] = f"{metrics_a[key]}"
        elif key.startswith('LPIPS') or key.startswith('FID'):
            # LPIPS和FID越低越好
            if metrics_a[key] < metrics_b[key]:
                better_metrics[key] = f"<b>{metrics_a[key]}</b>"
            else:
                better_metrics[key] = f"{metrics_a[key]}"
    return better_metrics

def write_metrics_to_html(algorithms: List[str], steps_list: List[int], output_file: str):
    with open(output_file, mode='w', encoding='utf-8') as file:
        # 写入HTML文件头
        file.write("<html><body>\n")
        file.write("<table border='1'>\n")
        # 写入表头
        file.write("<tr><th>steps</th><th>algorithm</th><th>PSNR_noiseless</th><th>SSIM_noiseless</th><th>LPIPS_noiseless</th><th>FID_noiseless</th>"
                   "<th>PSNR_noisy</th><th>SSIM_noisy</th><th>LPIPS_noisy</th><th>FID_noisy</th></tr>\n")
        
        for algorithm in algorithms:
            for steps in steps_list:
                print('{} {} steps'.format(algorithm, steps))
                # 计算原始算法和增强算法的指标
                metrics_original = calculate_and_format_metrics(algorithm, False, steps)
                metrics_learned = calculate_and_format_metrics(algorithm, True, steps)
                
                # 比较并加粗更好的值
                better_original = compare_and_bold(metrics_original, metrics_learned)
                better_learned = compare_and_bold(metrics_learned, metrics_original)
                
                # 写入原始算法的行
                file.write(f"<tr><td>{steps}</td><td>{algorithm}</td>"
                           f"<td>{better_original['PSNR_noiseless']}</td><td>{better_original['SSIM_noiseless']}</td>"
                           f"<td>{better_original['LPIPS_noiseless']}</td><td>{better_original['FID_noiseless']}</td>"
                           f"<td>{better_original['PSNR_noisy']}</td><td>{better_original['SSIM_noisy']}</td>"
                           f"<td>{better_original['LPIPS_noisy']}</td><td>{better_original['FID_noisy']}</td></tr>\n")
                
                # 写入增强算法的行
                file.write(f"<tr><td>{steps}</td><td>{algorithm}_learned</td>"
                           f"<td>{better_learned['PSNR_noiseless']}</td><td>{better_learned['SSIM_noiseless']}</td>"
                           f"<td>{better_learned['LPIPS_noiseless']}</td><td>{better_learned['FID_noiseless']}</td>"
                           f"<td>{better_learned['PSNR_noisy']}</td><td>{better_learned['SSIM_noisy']}</td>"
                           f"<td>{better_learned['LPIPS_noisy']}</td><td>{better_learned['FID_noisy']}</td></tr>\n")
        
        # 写入HTML文件尾
        file.write("</table>\n")
        file.write("</body></html>\n")

# 示例使用
task = 'cs2'
dataset = 'ffhq'
algorithms = ['ddnm', 'ddrm', 'pigdm', 'dps', 'reddiff', 'diffpir', 'dmps', 'resample', 'daps']  # 算法名称列表
# algorithms = ['dps', 'reddiff', 'diffpir', 'resample', 'daps']  # 算法名称列表
steps_list = [3, 4, 5, 7, 10, 15]  # 复杂度列表
# steps_list = [3]
output_file = 'metrics_{}_{}.html'.format(dataset, task)  # 输出文件名

write_metrics_to_html(algorithms, steps_list, output_file)