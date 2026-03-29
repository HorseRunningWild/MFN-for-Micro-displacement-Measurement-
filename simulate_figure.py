#!/usr/bin/env python3
import numpy as np
import pandas as pd
import os
import argparse
from scipy.ndimage import gaussian_filter
import seaborn as sns
import matplotlib.pyplot as plt

np.random.seed(42)

class InterferenceFringeSimulator:
    def __init__(self, stats_file_path, width=1280, height=1024, wavelength_nm=633.017, fit_scale=0.75):
        """
        stats_file_path: 包含 Mean 和 Std 的文本路径（与你给出的格式兼容）
        width, height: 要生成的原始图像尺寸（像素）
        wavelength_nm: 波长 (nm)
        fit_scale: 拟合时对图像做的下采样缩放因子
        """
        self.width = width
        self.height = height
        self.wavelength_nm = wavelength_nm
        self.fit_scale = float(fit_scale)
        self.stats = self._load_stats(stats_file_path)
        print(f"读取参数 {len(self.stats)} 个: {list(self.stats.keys())}")
        if abs(self.fit_scale - 1.0) > 1e-6:
            print(f"参数将从拟合尺度转换回原始图像尺度 (fit_scale={self.fit_scale})")

    def _load_stats(self, path):
        with open(path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        stats = {}
        for line in lines:
            if not line.strip():
                continue
            if line.strip().startswith('-') or 'Mean' in line or '平均值' in line or line.lower().startswith('name'):
                continue
            parts = line.strip().split()
            if len(parts) >= 3:
                name = parts[0]
                try:
                    mean = float(parts[1])
                    std = float(parts[2])
                    stats[name] = (mean, std)
                except ValueError:
                    continue
        return stats

    def _sample_params(self):
        """从 stats 中随机选取参数值（mean ± 3σ），并映射回原始图像尺度"""
        params = {}
        for name, (mean, std) in self.stats.items():
            low = mean - 3 * std
            high = mean + 3 * std
            val = np.random.uniform(low, high)
            params[name] = val

        # 空间尺度变换
        s = self.fit_scale
        if abs(s - 1.0) > 1e-9:
            for key in ('xc', 'yc', 'x0', 'y0'):
                if key in params:
                    params[key] = params[key] / s
            if 'a' in params:
                params['a'] = params['a'] * (s**2)
            if 'b' in params:
                params['b'] = params['b'] * (s**4)
            if 'c' in params:
                params['c'] = params['c'] * (s**2)
        return params

    def simulate_interference_pattern(self, params, m):
        I0 = params["I0"]
        a = params["a"]
        b = params["b"]
        c = params["c"]
        xc = params["xc"]
        yc = params["yc"]
        x0 = params["x0"]
        y0 = params["y0"]
        V = params["V"]
        k = params["k"]

        # 灰度空间为 0–255，不再归一化
        x = np.arange(self.width, dtype=np.float64)
        y = np.arange(self.height, dtype=np.float64)
        X, Y = np.meshgrid(x, y)

        r2 = (X - x0)**2 + (Y - y0)**2
        r2_env = (X - xc)**2 + (Y - yc)**2
        gaussian_env = np.exp(-c * r2_env)

        phase_unwrapped = m + a * r2 + b * (r2**2)
        phase_wrapped = (phase_unwrapped % 1) - 0.5
        phase = 2 * np.pi * phase_wrapped

        pattern = k + gaussian_env * I0 * (1 + V * np.cos(phase))

        # clip 到 0–255 灰度
        pattern = np.clip(pattern, 0.0, 255.0)
        return pattern

    def add_noise(
    self,
    image,
    power_drift_factor=1.0,
    photon_gain=2500.0,
    read_noise_std=2.5,
    blur_sigma_range=(1.2, 2.0),
    dark_current_e_per_sec=4.0,
    salt_pepper_ratio=0.0003,   
    stripe_noise_strength=1.8,  
    stripe_blur_sigma=14.0,     
    vignette_strength=0.005,     
    dead_pixel_ratio=0.0002,    
    noise_amplify_factor=1.1
    ):
        h, w = image.shape
        image = image.astype(np.float64)

        # 1. 功率漂移（缓慢全局波动）
        drift_jitter = np.random.normal(1.0, 0.005)
        image_drift = image * power_drift_factor * drift_jitter

        # 2. 光子散粒噪声（泊松型）
        photon_counts = np.clip(image_drift, 0, 255)
        noisy_photon = np.random.poisson(photon_counts).astype(np.float64)

        # 3. 读出噪声（高斯）
        read_noise_std_frame = read_noise_std * np.random.uniform(1.2, 1.8)
        noisy_readout = noisy_photon + np.random.normal(0, read_noise_std_frame, image.shape)

        # 4. 光学模糊（散焦 + 光学系统 PSF）
        sigma_blur = np.random.uniform(*blur_sigma_range)
        noisy_blurred = gaussian_filter(noisy_readout, sigma=sigma_blur)

        # 5. 暗电流噪声（低水平）
        exposure_time_ms = 1
        dark_noise = np.random.poisson(dark_current_e_per_sec * exposure_time_ms / 1000.0, size=image.shape)
        noisy_blurred += dark_noise

        # 6. 椒盐噪声
        if salt_pepper_ratio > 0:
            mask = np.random.rand(h, w) < salt_pepper_ratio
            noisy_blurred[mask] = np.random.choice([0.0, 255.0], size=np.count_nonzero(mask))

        # 7. 模糊型条纹噪声
        if stripe_noise_strength > 0:
            y_freq = np.random.uniform(8, 20)  
            Y, X = np.mgrid[0:h, 0:w]
            phase_jitter = np.random.normal(0, 0.2, size=(h, w))
            stripes = np.sin(2 * np.pi * (Y / y_freq) + phase_jitter)
            stripes = gaussian_filter(stripes, sigma=np.random.uniform(stripe_blur_sigma * 0.9, stripe_blur_sigma * 1.1))
            stripes *= np.random.uniform(stripe_noise_strength * 0.6, stripe_noise_strength * 0.9)
            noisy_blurred += stripes

        # 8. 背景渐变（光照不均）
        x = np.linspace(-1, 1, w)
        y = np.linspace(-1, 1, h)
        X, Y = np.meshgrid(x, y)
        gradient_background = 1.0 + 0.02 * X + 0.015 * Y
        noisy_blurred *= gradient_background

        # 9. 暗角（轻微）
        if vignette_strength > 0:
            r2 = X**2 + Y**2
            vignette = 1 - vignette_strength * r2
            noisy_blurred *= vignette

        # 10. 死像素
        if dead_pixel_ratio > 0:
            num_dead = int(dead_pixel_ratio * h * w)
            dead_x = np.random.randint(0, w, num_dead)
            dead_y = np.random.randint(0, h, num_dead)
            noisy_blurred[dead_y, dead_x] = np.random.choice([0.0, 255.0], size=num_dead)

        # 11. 噪声增强控制
        noise_component = noisy_blurred - image
        std_ratio = np.std(noise_component) / (np.std(image) + 1e-12)
        target_factor = noise_amplify_factor / (1.0 + 0.4 * std_ratio)
        amplified = image + target_factor * noise_component

        # 限幅
        amplified = np.clip(amplified, 0.0, 255.0)
        return amplified

def random_m_sequence(begin, end, min_images=5, max_images=15):
    """随机生成从 begin → end 的 m 序列"""
    if begin > end:
        begin, end = end, begin
    num_images = np.random.randint(min_images, max_images + 1)
    m_values = np.linspace(begin, end, num_images)
    perturb = np.random.uniform(-0.002, 0.002, size=num_images)
    m_values = np.clip(m_values + perturb, begin, end)
    m_values[0], m_values[-1] = begin, end
    return np.sort(m_values)

def simulate_dataset_group(simulator, output_dir, group_id, begin_value, end_value, noise_factor=1.0):
    """
    生成一组 (begin, end)，包含 up/down 两个文件夹，
    并引入组内缓慢变化的功率漂移。
    noise_factor: 控制整体噪声强度（>1 表示更强）
    """
    name_begin = f"{begin_value:.3f}".replace('.', '')
    name_end = f"{end_value:.3f}".replace('.', '')
    group_dir = os.path.join(output_dir, f"begin_{name_begin}_end_{name_end}")
    os.makedirs(group_dir, exist_ok=True)

    up_dir = os.path.join(group_dir, f"up_begin_{name_begin}_end_{name_end}")
    down_dir = os.path.join(group_dir, f"down_begin_{name_begin}_end_{name_end}")
    os.makedirs(up_dir, exist_ok=True)
    os.makedirs(down_dir, exist_ok=True)

    wavelength = simulator.wavelength_nm
    print(f"\n组 {group_id}: begin={begin_value:.3f}, end={end_value:.3f}")

    # ---- m 序列 ----
    m_up = random_m_sequence(begin_value, end_value)
    m_down = m_up[::-1]
    print(f" up m序列: {np.round(m_up, 4)}")
    print(f" down m序列: {np.round(m_down, 4)}")

    # ---- 随机采样组参数 ----
    group_params = simulator._sample_params()
    m_ref = m_up[0]

    # ---- 模拟功率漂移随时间缓变 ----
    drift_start = np.random.normal(1.0, 0.02)
    drift_end = np.random.normal(1.0, 0.02)
    power_drift_factors_up = np.linspace(drift_start, drift_end, len(m_up))
    power_drift_factors_down = power_drift_factors_up[::-1]

    # ---- UP ----
    displacement_up = []
    for i, (m, drift) in enumerate(zip(m_up, power_drift_factors_up)):
        pattern = simulator.simulate_interference_pattern(group_params, m)
        noisy = simulator.add_noise(pattern, power_drift_factor=drift, noise_amplify_factor=noise_factor)
        np.save(os.path.join(up_dir, f"image_{i+1:03d}.npy"), noisy)
        d_nm = (m - m_ref) * wavelength / 2
        displacement_up.append({
            "Image Index": f"{i+1:03d}",
            "m Value": m,
            "Power Drift Factor": drift,
            "Displacement (nm)": d_nm
        })
    pd.DataFrame(displacement_up).to_excel(
        os.path.join(up_dir, "displacement.xlsx"),
        index=False, float_format="%.10f"
    )

    # ---- DOWN ----
    displacement_down = []
    for i, (m, drift) in enumerate(zip(m_down, power_drift_factors_down)):
        pattern = simulator.simulate_interference_pattern(group_params, m)
        noisy = simulator.add_noise(pattern, power_drift_factor=drift, noise_amplify_factor=noise_factor)
        np.save(os.path.join(down_dir, f"image_{i+1:03d}.npy"), noisy)
        d_nm = (m - m_ref) * wavelength / 2
        displacement_down.append({
            "Image Index": f"{i+1:03d}",
            "m Value": m,
            "Power Drift Factor": drift,
            "Displacement (nm)": d_nm
        })
    pd.DataFrame(displacement_down).to_excel(
        os.path.join(down_dir, "displacement.xlsx"),
        index=False, float_format="%.10f"
    )

def main():
    parser = argparse.ArgumentParser(description="干涉条纹模拟（含功率漂移、散粒噪声、CCD噪声、随机模糊）")
    parser.add_argument("--stats", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--num_groups", type=int, default=5)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--fit_scale", type=float, default=0.75)
    parser.add_argument("--noise_factor", type=float, default=1.0,
                        help="整体噪声放大倍数")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    simulator = InterferenceFringeSimulator(args.stats, args.width, args.height, fit_scale=args.fit_scale)

    for g in range(args.num_groups):
        begin = np.random.uniform(0.0, 0.8)
        end = np.random.uniform(begin + 0.05, 1.0)
        simulate_dataset_group(simulator, args.output, g, begin, end, noise_factor=args.noise_factor)

if __name__ == "__main__":
    main()


## 注：”stats_file_path“里应txt文件为以下格式：
# 参数                      平均值(Mean)                 标准差(Std)
# -----------------------------------------------------------------
# I0                   1.142196e+02             1.824664e+00
# k                    2.111947e+01             6.883126e-02
# a                    2.844963e-05             3.680579e-08
# b                    5.459253e-15             3.045475e-15
# c                    2.357482e-06             8.577047e-08
# xc                   4.897949e+02             1.755068e+01
# yc                   4.814311e+02             1.640162e+01
# x0                   4.555675e+02             1.296875e+01
# y0                   3.858926e+02             1.655336e+01
# V                    7.622350e-01             2.486119e-02
