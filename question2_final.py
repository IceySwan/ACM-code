# -*- coding: utf-8 -*-
"""
2025华数杯全国大学生数学建模竞赛 C题
问题2：多通道LED最优权重设计
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from scipy import optimize, interpolate
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体，避免乱码
import platform

def setup_chinese_font():
    """设置中文字体"""
    system = platform.system()

    if system == "Windows":
        font_candidates = [
            'Microsoft YaHei',  # 微软雅黑
            'SimHei',           # 黑体
            'SimSun',           # 宋体
            'KaiTi',            # 楷体
            'FangSong'          # 仿宋
        ]
    elif system == "Darwin":  # macOS
        font_candidates = [
            'PingFang SC',      # 苹方
            'Arial Unicode MS', # Arial Unicode MS
            'Helvetica',        # Helvetica
            'STHeiti'           # 华文黑体
        ]
    else:  # Linux
        font_candidates = [
            'DejaVu Sans',      # DejaVu Sans
            'WenQuanYi Micro Hei', # 文泉驿微米黑
            'Noto Sans CJK SC', # Noto Sans CJK SC
            'SimHei'            # 黑体
        ]

    # 获取系统可用字体
    available_fonts = [f.name for f in fm.fontManager.ttflist]

    # 找到第一个可用的中文字体
    chinese_font = None
    for font in font_candidates:
        if font in available_fonts:
            chinese_font = font
            break

    if chinese_font:
        print(f"使用字体: {chinese_font}")
        plt.rcParams['font.sans-serif'] = [chinese_font, 'DejaVu Sans']
    else:
        print("未找到合适的中文字体，尝试使用系统默认字体")
        plt.rcParams['font.sans-serif'] = ['sans-serif']

    plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
    plt.rcParams['figure.dpi'] = 100
    plt.rcParams['savefig.dpi'] = 300

# 初始化字体设置
setup_chinese_font()

def clear_font_cache():
    """清理matplotlib字体缓存"""
    try:
        import matplotlib
        matplotlib.font_manager._rebuild()
        print("字体缓存已清理")
    except:
        pass

def ensure_chinese_display():
    """确保中文正常显示"""
    setup_chinese_font()
    plt.clf()
    plt.cla()

# 清理字体缓存（首次运行时）
clear_font_cache()

# 设置绘图风格
sns.set_style("whitegrid")
try:
    plt.style.use('seaborn-v0_8')
except:
    plt.style.use('seaborn')

print("="*60)
print("2025华数杯 C题 - 问题2：多通道LED最优权重设计")
print("="*60)
print()

class LEDLightAnalysis:
    """LED光源分析主类"""

    def __init__(self):
        """初始化"""
        self.wavelengths = np.arange(380, 781, 1)  # 380-780nm, 1nm间隔

        # CIE标准观察者函数和颜色匹配函数
        self.setup_cie_functions()

        # 普朗克轨迹数据
        self.setup_planck_locus()

        # 褪黑激素效应函数
        self.setup_melanopic_function()

    def setup_cie_functions(self):
        """设置CIE标准函数"""
        wl = self.wavelengths

        # X颜色匹配函数（改进的多峰拟合）
        x_main = 1.056 * np.exp(-0.5 * ((wl - 599.8) / 37.9)**2)
        x_secondary = 0.362 * np.exp(-0.5 * ((wl - 442.0) / 16.4)**2)
        x_negative = 0.065 * np.exp(-0.5 * ((wl - 501.1) / 20.9)**2)
        self.x_bar = x_main + x_secondary - x_negative

        # Y颜色匹配函数（视觉亮度函数，峰值在555nm）
        self.y_bar = np.exp(-0.5 * ((wl - 556.1) / 46.9)**2)

        # Z颜色匹配函数（峰值在449nm）
        self.z_bar = 1.669 * np.exp(-0.5 * ((wl - 449.8) / 19.4)**2)

        # 确保非负并归一化
        self.x_bar = np.maximum(self.x_bar, 0)
        self.y_bar = np.maximum(self.y_bar, 0)
        self.z_bar = np.maximum(self.z_bar, 0)

        # 标准化Y函数到683 lm/W（光度学常数）
        self.y_bar = self.y_bar / np.max(self.y_bar)

    def setup_planck_locus(self):
        """设置普朗克轨迹"""
        # 色温范围
        self.cct_range = np.arange(1000, 25001, 100)

        # 计算普朗克轨迹上的xy色坐标
        self.planck_x = []
        self.planck_y = []

        for cct in self.cct_range:
            spd_planck = self.planck_spd(cct)
            x, y, z = self.calculate_xyz(spd_planck)
            if x + y + z > 0:
                self.planck_x.append(x / (x + y + z))
                self.planck_y.append(y / (x + y + z))
            else:
                self.planck_x.append(0)
                self.planck_y.append(0)

        self.planck_x = np.array(self.planck_x)
        self.planck_y = np.array(self.planck_y)

    def setup_melanopic_function(self):
        """设置褪黑激素效应函数"""
        wl = self.wavelengths

        # 更准确的melanopic效应光谱函数
        mel_function = np.zeros_like(wl)

        # 短波段（380-500nm）：主要响应区域
        short_wave_mask = (wl >= 380) & (wl <= 500)
        wl_short = wl[short_wave_mask]
        alpha = 0.5 * ((np.log(wl_short) - np.log(480)) / 0.15)**2
        mel_function[short_wave_mask] = np.exp(-alpha)

        # 长波段（500-780nm）：较弱的响应
        long_wave_mask = (wl > 500) & (wl <= 780)
        wl_long = wl[long_wave_mask]
        mel_function[long_wave_mask] = 0.1 * np.exp(-(wl_long - 500) / 100)

        # 归一化到峰值为1
        self.mel_function = mel_function / np.max(mel_function)
        self.mel_function = np.maximum(self.mel_function, 0)

    def planck_spd(self, temperature):
        """计算普朗克黑体辐射SPD"""
        h = 6.626e-34  # 普朗克常数
        c = 3e8        # 光速
        k = 1.381e-23  # 玻尔兹曼常数

        wl_m = self.wavelengths * 1e-9  # 转换为米

        # 普朗克公式
        spd = (2 * h * c**2 / wl_m**5) / (np.exp(h * c / (wl_m * k * temperature)) - 1)

        # 归一化
        spd = spd / np.max(spd) * 100

        return spd

    def calculate_xyz(self, spd):
        """计算XYZ三刺激值"""
        if len(spd) != len(self.wavelengths):
            # 插值到标准波长
            f = interpolate.interp1d(np.linspace(380, 780, len(spd)), spd,
                                   bounds_error=False, fill_value=0)
            spd = f(self.wavelengths)

        # 计算XYZ
        X = np.trapz(spd * self.x_bar, self.wavelengths)
        Y = np.trapz(spd * self.y_bar, self.wavelengths)
        Z = np.trapz(spd * self.z_bar, self.wavelengths)

        return X, Y, Z

    def calculate_cct_duv(self, spd):
        """计算相关色温CCT和Duv"""
        X, Y, Z = self.calculate_xyz(spd)

        if X + Y + Z == 0:
            return 0, 0

        # 计算xy色坐标
        x = X / (X + Y + Z)
        y = Y / (X + Y + Z)

        # 使用Robertson方法计算CCT
        distances = np.sqrt((self.planck_x - x)**2 + (self.planck_y - y)**2)
        min_idx = np.argmin(distances)

        # 在最近点附近进行线性插值以提高精度
        if min_idx > 0 and min_idx < len(self.cct_range) - 1:
            idx_range = [min_idx-1, min_idx, min_idx+1]
            x_pts = self.planck_x[idx_range]
            y_pts = self.planck_y[idx_range]
            cct_pts = self.cct_range[idx_range]

            dists = [(x - x_pts[i])**2 + (y - y_pts[i])**2 for i in range(3)]
            weights = [1/max(d, 1e-10) for d in dists]
            total_weight = sum(weights)
            weights = [w/total_weight for w in weights]

            cct = sum(cct_pts[i] * weights[i] for i in range(3))
            closest_x = sum(x_pts[i] * weights[i] for i in range(3))
            closest_y = sum(y_pts[i] * weights[i] for i in range(3))
        else:
            cct = self.cct_range[min_idx]
            closest_x = self.planck_x[min_idx]
            closest_y = self.planck_y[min_idx]

        # Duv计算（到普朗克轨迹的距离）
        duv = np.sqrt((x - closest_x)**2 + (y - closest_y)**2)

        # 判断Duv符号（绿色偏移为正，洋红偏移为负）
        if y > closest_y:
            duv = duv  # 绿色偏移
        else:
            duv = -duv  # 洋红偏移

        return cct, duv

    def calculate_rf_rg(self, spd):
        """计算保真度指数Rf和色域指数Rg"""
        cct, _ = self.calculate_cct_duv(spd)

        # 选择参考光源
        if cct < 5000:
            ref_spd = self.planck_spd(cct)
        else:
            ref_spd = self.daylight_spd(cct)

        # 定义代表性测试颜色的反射光谱
        test_colors = self.get_test_color_reflectances()

        # 计算测试光源和参考光源下的颜色
        test_colors_under_test = []
        test_colors_under_ref = []

        for reflectance in test_colors:
            # 测试光源下的颜色
            test_spectrum = spd * reflectance
            X_t, Y_t, Z_t = self.calculate_xyz(test_spectrum)
            if X_t + Y_t + Z_t > 0:
                test_colors_under_test.append([X_t/(X_t+Y_t+Z_t), Y_t/(X_t+Y_t+Z_t)])
            else:
                test_colors_under_test.append([0.33, 0.33])

            # 参考光源下的颜色
            ref_spectrum = ref_spd * reflectance
            X_r, Y_r, Z_r = self.calculate_xyz(ref_spectrum)
            if X_r + Y_r + Z_r > 0:
                test_colors_under_ref.append([X_r/(X_r+Y_r+Z_r), Y_r/(X_r+Y_r+Z_r)])
            else:
                test_colors_under_ref.append([0.33, 0.33])

        # 计算色差
        color_differences = []
        for i in range(len(test_colors)):
            x_t, y_t = test_colors_under_test[i]
            x_r, y_r = test_colors_under_ref[i]
            delta_e = np.sqrt((x_t - x_r)**2 + (y_t - y_r)**2) * 100
            color_differences.append(delta_e)

        # Rf计算（保真度指数）
        avg_color_diff = np.mean(color_differences)
        rf = max(0, 100 - avg_color_diff * 4.6)

        # Rg计算（色域指数）
        test_area = self.calculate_color_gamut_area(test_colors_under_test)
        ref_area = self.calculate_color_gamut_area(test_colors_under_ref)

        if ref_area > 0:
            rg = (test_area / ref_area) * 100
        else:
            rg = 100

        rg = max(50, min(150, rg))

        return rf, rg

    def get_test_color_reflectances(self):
        """获取测试颜色的反射光谱"""
        colors = []
        wl = self.wavelengths

        # 红色（峰值在700nm）
        red = 0.1 + 0.8 * np.exp(-((wl - 700) / 50)**2)
        colors.append(red)

        # 橙色（峰值在600nm）
        orange = 0.1 + 0.7 * np.exp(-((wl - 600) / 40)**2)
        colors.append(orange)

        # 黄色（峰值在580nm）
        yellow = 0.2 + 0.6 * np.exp(-((wl - 580) / 35)**2)
        colors.append(yellow)

        # 绿色（峰值在530nm）
        green = 0.1 + 0.7 * np.exp(-((wl - 530) / 30)**2)
        colors.append(green)

        # 青色（峰值在480nm）
        cyan = 0.1 + 0.6 * np.exp(-((wl - 480) / 25)**2)
        colors.append(cyan)

        # 蓝色（峰值在450nm）
        blue = 0.1 + 0.8 * np.exp(-((wl - 450) / 30)**2)
        colors.append(blue)

        # 紫色（峰值在420nm）
        purple = 0.1 + 0.6 * np.exp(-((wl - 420) / 25)**2)
        colors.append(purple)

        # 皮肤色（宽峰）
        skin = 0.3 + 0.4 * np.exp(-((wl - 600) / 80)**2)
        colors.append(skin)

        return colors

    def calculate_color_gamut_area(self, color_points):
        """计算色域面积"""
        if len(color_points) < 3:
            return 0

        from scipy.spatial import ConvexHull
        try:
            hull = ConvexHull(color_points)
            return hull.volume  # 2D情况下volume就是面积
        except:
            return 0

    def daylight_spd(self, cct):
        """标准日光D系列SPD"""
        wl = self.wavelengths

        if cct <= 4000:
            spd = 100 * (1 + 0.3 * np.exp(-((wl - 650) / 100)**2))
        elif cct >= 8000:
            spd = 100 * (1 + 0.5 * np.exp(-((wl - 450) / 80)**2))
        else:
            spd = 100 * np.ones_like(wl)

        return spd

    def calculate_mel_der(self, spd):
        """计算褪黑激素日光效率比mel-DER"""
        mel_irradiance = np.trapz(spd * self.mel_function, self.wavelengths)
        photopic_irradiance = np.trapz(spd * self.y_bar, self.wavelengths)

        if photopic_irradiance == 0:
            return 0

        mel_der = mel_irradiance / photopic_irradiance

        return mel_der

# 创建分析实例
led_analyzer = LEDLightAnalysis()

def problem2_led_optimization():
    """问题2：多通道LED最优权重设计"""
    print("\n" + "="*50)
    print("问题2：多通道LED最优权重设计")
    print("="*50)

    try:
        # 读取五通道LED的SPD数据
        print("正在读取Problem 2 LED SPD数据...")

        excel_file = pd.ExcelFile('附录.xlsx')
        print(f"Excel文件包含的工作表: {excel_file.sheet_names}")

        # 查找Problem 2相关的工作表
        problem2_sheet = None
        for sheet in excel_file.sheet_names:
            if 'Problem' in sheet and '2' in sheet:
                problem2_sheet = sheet
                break

        if problem2_sheet is None:
            # 如果没找到，尝试第二个工作表
            if len(excel_file.sheet_names) > 1:
                problem2_sheet = excel_file.sheet_names[1]
            else:
                problem2_sheet = excel_file.sheet_names[0]

        print(f"使用工作表: {problem2_sheet}")

        # 读取数据
        led_data = pd.read_excel('attachment-Q2.xlsx')
        print(f"数据形状: {led_data.shape}")
        print("数据前5行:")
        print(led_data.head())
        print(f"列名: {list(led_data.columns)}")

        # 假设第一列是波长，后面5列是5个通道的SPD
        if len(led_data.columns) >= 6:
            wavelength_col = led_data.columns[0]
            led_channels = led_data.columns[1:6]  # 取前5个LED通道

            # 处理波长列
            wavelengths_raw = led_data[wavelength_col].values
            led_spds_raw = led_data[led_channels].values

            # 提取数值部分
            wavelengths = []
            led_spds_clean = []

            for i, wl in enumerate(wavelengths_raw):
                try:
                    # 处理波长
                    if isinstance(wl, str):
                        wl_num = float(wl.split('(')[0]) if '(' in wl else float(wl)
                    else:
                        wl_num = float(wl)

                    # 处理LED数据
                    led_row = []
                    valid_row = True
                    for j in range(min(5, len(led_channels))):
                        try:
                            val = float(led_spds_raw[i, j])
                            if np.isnan(val):
                                valid_row = False
                                break
                            led_row.append(val)
                        except:
                            valid_row = False
                            break

                    if valid_row and not np.isnan(wl_num) and len(led_row) == 5:
                        wavelengths.append(wl_num)
                        led_spds_clean.append(led_row)
                except:
                    continue

            wavelengths = np.array(wavelengths)
            led_spds = np.array(led_spds_clean)

            print(f"有效数据点数: {len(wavelengths)}")
            print(f"波长范围: {wavelengths.min():.1f} - {wavelengths.max():.1f} nm")
            print(f"LED通道名称: {list(led_channels)}")

            # 插值到标准波长网格
            led_spds_standard = np.zeros((len(led_analyzer.wavelengths), 5))
            for i in range(5):
                f = interpolate.interp1d(wavelengths, led_spds[:, i],
                                       bounds_error=False, fill_value=0)
                led_spds_standard[:, i] = f(led_analyzer.wavelengths)

            # 优化权重
            optimize_led_weights(led_spds_standard, led_channels)

        else:
            print("数据格式不正确，需要至少6列数据（波长+5个LED通道）")
            raise ValueError("数据格式错误")

    except Exception as e:
        print(f"读取数据时出错: {e}")
        print("使用模拟数据进行演示...")

        # 生成五通道LED的模拟SPD数据
        led_spds_standard = np.zeros((len(led_analyzer.wavelengths), 5))

        # 通道1: 深红光 (Deep Red) - 峰值在660nm
        deep_red = 100 * np.exp(-0.5 * ((led_analyzer.wavelengths - 660) / 20)**2)
        led_spds_standard[:, 0] = deep_red

        # 通道2: 绿光 (Green) - 峰值在530nm
        green = 100 * np.exp(-0.5 * ((led_analyzer.wavelengths - 530) / 25)**2)
        led_spds_standard[:, 1] = green

        # 通道3: 蓝光 (Blue) - 峰值在450nm
        blue = 100 * np.exp(-0.5 * ((led_analyzer.wavelengths - 450) / 20)**2)
        led_spds_standard[:, 2] = blue

        # 通道4: 暖白光 WW (~3000K)
        ww_spd = led_analyzer.planck_spd(3000)
        led_spds_standard[:, 3] = ww_spd

        # 通道5: 冷白光 CW (~6500K)
        cw_spd = led_analyzer.planck_spd(6500)
        led_spds_standard[:, 4] = cw_spd

        led_channels = ['深红光', '绿光', '蓝光', '暖白光WW', '冷白光CW']

        # 优化权重
        optimize_led_weights(led_spds_standard, led_channels)

def optimize_led_weights(led_spds, channel_names):
    """优化LED权重"""

    def objective_daylight(weights):
        """日间照明目标函数"""
        if np.sum(weights) == 0:
            return 1e10
        weights = weights / np.sum(weights)  # 归一化

        # 合成SPD
        combined_spd = np.dot(led_spds, weights)

        # 计算参数
        cct, duv = led_analyzer.calculate_cct_duv(combined_spd)
        rf, rg = led_analyzer.calculate_rf_rg(combined_spd)

        # 目标: CCT 6000±500K, Rf最大化(>88), Rg 95-105
        cct_penalty = 0
        if not (5500 <= cct <= 6500):
            cct_penalty = abs(cct - 6000) / 100

        rf_reward = rf if rf > 88 else rf - 100

        rg_penalty = 0
        if not (95 <= rg <= 105):
            if rg < 95:
                rg_penalty = (95 - rg) / 10
            else:
                rg_penalty = (rg - 105) / 10

        # 目标函数：最大���Rf，最小化CCT和Rg偏差
        return -(rf_reward - cct_penalty * 10 - rg_penalty * 5)

    def objective_night(weights):
        """夜间助眠目标函数"""
        if np.sum(weights) == 0:
            return 1e10
        weights = weights / np.sum(weights)

        combined_spd = np.dot(led_spds, weights)

        cct, duv = led_analyzer.calculate_cct_duv(combined_spd)
        rf, rg = led_analyzer.calculate_rf_rg(combined_spd)
        mel_der = led_analyzer.calculate_mel_der(combined_spd)

        # 目标: CCT 3000±500K, Rf≥80, mel-DER最小化
        cct_penalty = 0
        if not (2500 <= cct <= 3500):
            cct_penalty = abs(cct - 3000) / 100

        rf_penalty = 0
        if rf < 80:
            rf_penalty = (80 - rf) / 10

        # 目标函数：最小化mel-DER，满足CCT和Rf约束
        return mel_der + cct_penalty * 5 + rf_penalty * 3

    # 约束条件
    constraints = [
        {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},  # 权重和为1
    ]
    bounds = [(0, 1) for _ in range(5)]  # 权重非负且小于等于1

    # 初始猜测
    x0 = np.ones(5) / 5

    # 日间照明优化
    print("\n优化日间照明模式...")
    result_day = optimize.minimize(objective_daylight, x0, method='SLSQP',
                                 bounds=bounds, constraints=constraints)

    if result_day.success:
        weights_day = result_day.x
        weights_day = weights_day / np.sum(weights_day)  # 归一化
        combined_spd_day = np.dot(led_spds, weights_day)

        cct_day, duv_day = led_analyzer.calculate_cct_duv(combined_spd_day)
        rf_day, rg_day = led_analyzer.calculate_rf_rg(combined_spd_day)
        mel_der_day = led_analyzer.calculate_mel_der(combined_spd_day)

        print("日间照明最优权重:")
        for i, (name, weight) in enumerate(zip(channel_names, weights_day)):
            print(f"  {name}: {weight:.3f}")
        print(f"性能指标: CCT={cct_day:.0f}K, Rf={rf_day:.1f}, Rg={rg_day:.1f}, mel-DER={mel_der_day:.3f}")
    else:
        print("日间照明优化失败")
        weights_day = np.ones(5) / 5
        combined_spd_day = np.dot(led_spds, weights_day)
        cct_day, duv_day = led_analyzer.calculate_cct_duv(combined_spd_day)
        rf_day, rg_day = led_analyzer.calculate_rf_rg(combined_spd_day)
        mel_der_day = led_analyzer.calculate_mel_der(combined_spd_day)

    # 夜间助眠优化
    print("\n优化夜间助眠模式...")
    result_night = optimize.minimize(objective_night, x0, method='SLSQP',
                                   bounds=bounds, constraints=constraints)

    if result_night.success:
        weights_night = result_night.x
        weights_night = weights_night / np.sum(weights_night)
        combined_spd_night = np.dot(led_spds, weights_night)

        cct_night, duv_night = led_analyzer.calculate_cct_duv(combined_spd_night)
        rf_night, rg_night = led_analyzer.calculate_rf_rg(combined_spd_night)
        mel_der_night = led_analyzer.calculate_mel_der(combined_spd_night)

        print("夜间助眠最优权重:")
        for i, (name, weight) in enumerate(zip(channel_names, weights_night)):
            print(f"  {name}: {weight:.3f}")
        print(f"性能指标: CCT={cct_night:.0f}K, Rf={rf_night:.1f}, Rg={rg_night:.1f}, mel-DER={mel_der_night:.3f}")
    else:
        print("夜间助眠优化失败")
        weights_night = np.array([0.3, 0.1, 0.0, 0.6, 0.0])  # 偏向暖色
        combined_spd_night = np.dot(led_spds, weights_night)
        cct_night, duv_night = led_analyzer.calculate_cct_duv(combined_spd_night)
        rf_night, rg_night = led_analyzer.calculate_rf_rg(combined_spd_night)
        mel_der_night = led_analyzer.calculate_mel_der(combined_spd_night)

    # 创建图表 - 四个独立的图
    create_problem2_figures(led_spds, channel_names, weights_day, weights_night,
                          combined_spd_day, combined_spd_night,
                          cct_day, rf_day, rg_day, mel_der_day,
                          cct_night, rf_night, rg_night, mel_der_night)

    # 输出答案
    print("\n" + "="*30)
    print("问题2答案:")
    print("="*30)
    print("日间照明模式最优权重:")
    for i, (name, weight) in enumerate(zip(channel_names, weights_day)):
        print(f"  {name}: {weight:.3f}")
    print(f"  性能: CCT={cct_day:.0f}K, Rf={rf_day:.1f}, Rg={rg_day:.1f}, mel-DER={mel_der_day:.3f}")

    print("\n夜间助眠模式最优权重:")
    for i, (name, weight) in enumerate(zip(channel_names, weights_night)):
        print(f"  {name}: {weight:.3f}")
    print(f"  性能: CCT={cct_night:.0f}K, Rf={rf_night:.1f}, Rg={rg_night:.1f}, mel-DER={mel_der_night:.3f}")
    print("="*30)

def create_problem2_figures(led_spds, channel_names, weights_day, weights_night,
                          spd_day, spd_night, cct_day, rf_day, rg_day, mel_der_day,
                          cct_night, rf_night, rg_night, mel_der_night):
    """创建问题2的六个独立图表"""

    # 确保中文字体正常显示
    ensure_chinese_display()

    # 创建输出文件夹
    import os
    if not os.path.exists('Problem2_figures'):
        os.makedirs('Problem2_figures')

    # 计算Duv值
    _, duv_day = led_analyzer.calculate_cct_duv(spd_day)
    _, duv_night = led_analyzer.calculate_cct_duv(spd_night)

    # 图1: 日间、夜间参数柱状图
    plt.figure(figsize=(14, 8))

    # 参数名称和值
    params = ['CCT', 'Rf', 'Rg', 'Duv×100', 'mel-DER×1000']
    day_values = [cct_day, rf_day, rg_day, duv_day*100, mel_der_day*1000]
    night_values = [cct_night, rf_night, rg_night, duv_night*100, mel_der_night*1000]

    x = np.arange(len(params))
    width = 0.35

    bars1 = plt.bar(x - width/2, day_values, width, label='日间照明模式',
                   color='gold', alpha=0.8, edgecolor='black', linewidth=1)
    bars2 = plt.bar(x + width/2, night_values, width, label='夜间助眠模式',
                   color='navy', alpha=0.8, edgecolor='black', linewidth=1)

    plt.xlabel('性能参数', fontsize=14)
    plt.ylabel('参数值', fontsize=14)
    plt.title('日间、夜间照明模式性能参数对比', fontsize=16, fontweight='bold')
    plt.xticks(x, params, fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3, axis='y')

    # 添加数值标注
    for i, (bar1, bar2, d_val, n_val) in enumerate(zip(bars1, bars2, day_values, night_values)):
        # 日间模式标注
        plt.text(bar1.get_x() + bar1.get_width()/2., bar1.get_height() + max(max(day_values), max(night_values))*0.01,
                f'{d_val:.1f}' if i < 3 else f'{d_val:.3f}',
                ha='center', va='bottom', fontsize=10, fontweight='bold')

        # 夜间模式标注
        plt.text(bar2.get_x() + bar2.get_width()/2., bar2.get_height() + max(max(day_values), max(night_values))*0.01,
                f'{n_val:.1f}' if i < 3 else f'{n_val:.3f}',
                ha='center', va='bottom', fontsize=10, fontweight='bold')

    plt.tight_layout()
    plt.savefig('Problem2_figures/图1_日间夜间参数对比.png', dpi=300, bbox_inches='tight')
    plt.show()

    # 图2: 五通道LED光谱图
    plt.figure(figsize=(14, 8))
    colors = ['#FF4444', '#44AA44', '#4444FF', '#FF8800', '#00CCCC']
    linestyles = ['-', '-', '-', '--', '--']

    for i, (spd, name, color, ls) in enumerate(zip(led_spds.T, channel_names, colors, linestyles)):
        plt.plot(led_analyzer.wavelengths, spd, color=color, linewidth=3,
                label=f'通道{i+1}: {name}', alpha=0.9, linestyle=ls)

    plt.xlabel('波长 (nm)', fontsize=14)
    plt.ylabel('相对功率', fontsize=14)
    plt.title('五通道LED光谱功率分布', fontsize=16, fontweight='bold')
    plt.legend(fontsize=12, loc='upper right')
    plt.grid(True, alpha=0.3)
    plt.xlim(380, 780)

    # 添加波长区域标注
    plt.axvspan(380, 450, alpha=0.1, color='blue', label='蓝光区域')
    plt.axvspan(450, 550, alpha=0.1, color='green', label='绿光区域')
    plt.axvspan(550, 650, alpha=0.1, color='yellow', label='黄光区域')
    plt.axvspan(650, 780, alpha=0.1, color='red', label='红光区域')

    plt.tight_layout()
    plt.savefig('Problem2_figures/图2_五通道LED光谱分布.png', dpi=300, bbox_inches='tight')
    plt.show()

    # 图3: 最优权重参数图(柱状图)
    plt.figure(figsize=(14, 8))
    x = np.arange(len(channel_names))
    width = 0.35

    bars1 = plt.bar(x - width/2, weights_day, width, label='日间照明模式',
                   color='gold', alpha=0.8, edgecolor='black', linewidth=1)
    bars2 = plt.bar(x + width/2, weights_night, width, label='夜间助眠模式',
                   color='darkblue', alpha=0.8, edgecolor='black', linewidth=1)

    plt.xlabel('LED通道', fontsize=14)
    plt.ylabel('权重比例', fontsize=14)
    plt.title('最优权重配比参数', fontsize=16, fontweight='bold')
    plt.xticks(x, [f'{name}\n(通道{i+1})' for i, name in enumerate(channel_names)],
               fontsize=11, ha='center')
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3, axis='y')

    # 添加权重数值标注
    for bar, weight in zip(bars1, weights_day):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{weight:.3f}\n({weight*100:.1f}%)',
                ha='center', va='bottom', fontsize=10, fontweight='bold')

    for bar, weight in zip(bars2, weights_night):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{weight:.3f}\n({weight*100:.1f}%)',
                ha='center', va='bottom', fontsize=10, fontweight='bold')

    plt.ylim(0, max(max(weights_day), max(weights_night)) * 1.2)
    plt.tight_layout()
    plt.savefig('Problem2_figures/图3_最优权重配比��数.png', dpi=300, bbox_inches='tight')
    plt.show()

    # 图4: 五通道LED光谱图 (与图2相同但强调合成效果)
    plt.figure(figsize=(14, 8))

    # 绘制单独通道光谱（较浅颜色）
    for i, (spd, name, color) in enumerate(zip(led_spds.T, channel_names, colors)):
        plt.plot(led_analyzer.wavelengths, spd, color=color, linewidth=2,
                alpha=0.5, linestyle='--', label=f'{name}(单通道)')

    # 绘制加权合成光谱
    plt.plot(led_analyzer.wavelengths, spd_day, color='gold', linewidth=4,
             alpha=0.9, label='日间模式合成光谱')
    plt.plot(led_analyzer.wavelengths, spd_night, color='navy', linewidth=4,
             alpha=0.9, label='夜间模式合成光谱')

    plt.xlabel('波长 (nm)', fontsize=14)
    plt.ylabel('相对功率', fontsize=14)
    plt.title('五通道LED光谱及合���效果', fontsize=16, fontweight='bold')
    plt.legend(fontsize=10, loc='upper right', ncol=2)
    plt.grid(True, alpha=0.3)
    plt.xlim(380, 780)

    # 添加性能指标文本框
    textstr_day = f'日间模式:\nCCT: {cct_day:.0f}K\nRf: {rf_day:.1f}\nRg: {rg_day:.1f}'
    textstr_night = f'夜间模式:\nCCT: {cct_night:.0f}K\nRf: {rf_night:.1f}\nRg: {rg_night:.1f}'

    plt.text(0.02, 0.98, textstr_day, transform=plt.gca().transAxes, fontsize=11,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='gold', alpha=0.8))
    plt.text(0.02, 0.78, textstr_night, transform=plt.gca().transAxes, fontsize=11,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))

    plt.tight_layout()
    plt.savefig('Problem2_figures/图4_LED光谱合成效果.png', dpi=300, bbox_inches='tight')
    plt.show()

    # 图5: 综合性能雷达图
    from math import pi

    plt.figure(figsize=(12, 10))

    # 设置雷达图参数
    categories = ['CCT适宜性', '保真度Rf', '色域Rg', 'Duv性能', '节律调节']
    N = len(categories)

    # 标准化函数
    def normalize_cct(cct, is_day=True):
        target = 6000 if is_day else 3000
        return max(0, 1 - abs(cct - target) / 2000)

    def normalize_duv(duv):
        return max(0, 1 - abs(duv) / 0.01)  # Duv理想值接近0

    def normalize_mel_der(mel_der, is_day=True):
        if is_day:
            return min(1, mel_der / 0.5)  # 日间可以有一定mel-DER
        else:
            return max(0, 1 - mel_der / 0.3)  # 夜间mel-DER越小越好

    # 日间模式标准化数据
    day_values = [
        normalize_cct(cct_day, True),
        rf_day / 100,
        min(rg_day / 100, 1.2),
        normalize_duv(duv_day),
        normalize_mel_der(mel_der_day, True)
    ]

    # 夜间模式标准化数据
    night_values = [
        normalize_cct(cct_night, False),
        rf_night / 100,
        min(rg_night / 100, 1.2),
        normalize_duv(duv_night),
        normalize_mel_der(mel_der_night, False)
    ]

    # 计算角度
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]  # 闭合

    day_values += day_values[:1]
    night_values += night_values[:1]

    # 创建雷达图
    ax = plt.subplot(111, projection='polar')

    # 绘制数据
    ax.plot(angles, day_values, 'o-', linewidth=3, label='日间照明模式',
            color='gold', markersize=8)
    ax.fill(angles, day_values, alpha=0.25, color='gold')

    ax.plot(angles, night_values, 'o-', linewidth=3, label='夜间助眠模式',
            color='navy', markersize=8)
    ax.fill(angles, night_values, alpha=0.25, color='navy')

    # 设置标签
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=12)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=10)
    ax.grid(True)

    plt.title('综合性能雷达图', fontsize=16, fontweight='bold', pad=20)
    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0), fontsize=12)

    # 添加性能说明
    performance_text = """
    性能评��标准:
    • CCT适宜性: 日间6000K, 夜间3000K为理想值
    • 保真度Rf: 越接近100越好
    • 色域Rg: 接近100为理想
    • Duv性能: 越接近0越好
    • 节律调节: 日间适中, 夜间越低越好
    """

    plt.figtext(0.02, 0.02, performance_text, fontsize=10,
                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))

    plt.tight_layout()
    plt.savefig('Problem2_figures/图5_综合性能雷达图.png', dpi=300, bbox_inches='tight')
    plt.show()

    # 图6: 标准色盘上显示日间/夜间最优光照的位置
    plt.figure(figsize=(12, 10))

    # 绘制CIE 1931色度图轮廓
    # 创建光谱轨迹（单色光轨迹）
    wavelengths_mono = np.arange(380, 700, 5)
    spectrum_x = []
    spectrum_y = []

    for wl in wavelengths_mono:
        # 创建单色光谱（在��定波长处有峰值）
        mono_spd = np.exp(-0.5 * ((led_analyzer.wavelengths - wl) / 5)**2)
        X, Y, Z = led_analyzer.calculate_xyz(mono_spd)
        if X + Y + Z > 0:
            spectrum_x.append(X / (X + Y + Z))
            spectrum_y.append(Y / (X + Y + Z))

    # 闭合光谱轨迹（连接紫端）
    spectrum_x.append(spectrum_x[0])
    spectrum_y.append(spectrum_y[0])

    # 绘制光谱轨迹
    plt.plot(spectrum_x, spectrum_y, 'k-', linewidth=2, label='光谱轨迹', alpha=0.7)
    plt.fill(spectrum_x, spectrum_y, color='lightgray', alpha=0.3, label='可见光色域')

    # 绘制普朗克轨迹（黑体轨迹）
    plt.plot(led_analyzer.planck_x, led_analyzer.planck_y, 'k--',
             linewidth=2, label='普朗克轨迹', alpha=0.8)

    # 计算日间和夜间光照的xy色坐标
    X_day, Y_day, Z_day = led_analyzer.calculate_xyz(spd_day)
    X_night, Y_night, Z_night = led_analyzer.calculate_xyz(spd_night)

    if X_day + Y_day + Z_day > 0:
        x_day = X_day / (X_day + Y_day + Z_day)
        y_day = Y_day / (X_day + Y_day + Z_day)
    else:
        x_day, y_day = 0.33, 0.33

    if X_night + Y_night + Z_night > 0:
        x_night = X_night / (X_night + Y_night + Z_night)
        y_night = Y_night / (X_night + Y_night + Z_night)
    else:
        x_night, y_night = 0.33, 0.33

    # 绘制日间和夜间光照位置
    plt.scatter(x_day, y_day, s=300, c='gold', marker='*',
               edgecolor='black', linewidth=2, label=f'日间照明\n(CCT={cct_day:.0f}K)', zorder=10)
    plt.scatter(x_night, y_night, s=300, c='navy', marker='*',
               edgecolor='black', linewidth=2, label=f'夜间助眠\n(CCT={cct_night:.0f}K)', zorder=10)

    # 标注Duv线（到普朗克轨迹的距离）
    # 找到普朗克轨迹上最近的点
    distances_day = np.sqrt((led_analyzer.planck_x - x_day)**2 + (led_analyzer.planck_y - y_day)**2)
    min_idx_day = np.argmin(distances_day)

    distances_night = np.sqrt((led_analyzer.planck_x - x_night)**2 + (led_analyzer.planck_y - y_night)**2)
    min_idx_night = np.argmin(distances_night)

    # 绘制Duv线
    plt.plot([x_day, led_analyzer.planck_x[min_idx_day]],
             [y_day, led_analyzer.planck_y[min_idx_day]],
             'gold', linewidth=3, alpha=0.7, linestyle=':',
             label=f'日间Duv={duv_day:.4f}')

    plt.plot([x_night, led_analyzer.planck_x[min_idx_night]],
             [y_night, led_analyzer.planck_y[min_idx_night]],
             'navy', linewidth=3, alpha=0.7, linestyle=':',
             label=f'夜间Duv={duv_night:.4f}')

    # 标注一些关键色温点
    key_ccts = [2700, 3000, 4000, 5000, 6500, 10000]
    for cct in key_ccts:
        # 找到对应的xy坐标
        distances_cct = np.abs(led_analyzer.cct_range - cct)
        idx_cct = np.argmin(distances_cct)
        if idx_cct < len(led_analyzer.planck_x):
            plt.annotate(f'{cct}K',
                        (led_analyzer.planck_x[idx_cct], led_analyzer.planck_y[idx_cct]),
                        xytext=(5, 5), textcoords='offset points',
                        fontsize=9, alpha=0.7)

    # 添加等色温线（示意）
    for cct in [3000, 6000]:
        distances_cct = np.abs(led_analyzer.cct_range - cct)
        idx_cct = np.argmin(distances_cct)
        if idx_cct < len(led_analyzer.planck_x):
            # 绘制等色温线（垂直于普朗克轨迹）
            center_x = led_analyzer.planck_x[idx_cct]
            center_y = led_analyzer.planck_y[idx_cct]

            # 简化的等色温线
            line_length = 0.05
            if idx_cct > 0 and idx_cct < len(led_analyzer.planck_x) - 1:
                # 计算切线方向
                dx = led_analyzer.planck_x[idx_cct+1] - led_analyzer.planck_x[idx_cct-1]
                dy = led_analyzer.planck_y[idx_cct+1] - led_analyzer.planck_y[idx_cct-1]
                norm = np.sqrt(dx**2 + dy**2)
                if norm > 0:
                    # 垂直方向
                    perp_x = -dy / norm * line_length
                    perp_y = dx / norm * line_length

                    plt.plot([center_x - perp_x, center_x + perp_x],
                            [center_y - perp_y, center_y + perp_y],
                            'gray', linewidth=1, alpha=0.5)

    # 设置图表属性
    plt.xlabel('x色度坐标', fontsize=14)
    plt.ylabel('y色度坐标', fontsize=14)
    plt.title('CIE 1931色度图 - 日间/夜间最优光照位置', fontsize=16, fontweight='bold')
    plt.legend(fontsize=10, loc='upper right')
    plt.grid(True, alpha=0.3)

    # 设置坐标轴范围
    plt.xlim(0, 0.8)
    plt.ylim(0, 0.9)

    # 添加性能信息文本框
    info_text = f"""优化结果对比:
    
日间照明模式:
• 色坐标: ({x_day:.3f}, {y_day:.3f})
• CCT: {cct_day:.0f}K
• Duv: {duv_day:.4f}
• Rf: {rf_day:.1f}
• mel-DER: {mel_der_day:.3f}

夜间助眠模式:
• 色坐标: ({x_night:.3f}, {y_night:.3f})
• CCT: {cct_night:.0f}K
• Duv: {duv_night:.4f}
• Rf: {rf_night:.1f}
• mel-DER: {mel_der_night:.3f}"""

    plt.text(0.02, 0.98, info_text, transform=plt.gca().transAxes, fontsize=10,
             verticalalignment='top', horizontalalignment='left',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.9, edgecolor='gray'))

    # 添加色温说明
    temp_text = """色温说明:
• 2700K: 温馨暖光
• 3000K: 舒适暖光  
• 4000K: 自然白光
• 5000K: 日光白
• 6500K: 冷白光"""

    plt.text(0.98, 0.02, temp_text, transform=plt.gca().transAxes, fontsize=9,
             verticalalignment='bottom', horizontalalignment='right',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='lightblue', alpha=0.8))

    plt.tight_layout()
    plt.savefig('Problem2_figures/图6_CIE色度图最优光照位置.png', dpi=300, bbox_inches='tight')
    plt.show()

    print("问题2的六个图表已保存到 Problem2_figures/ 文件夹:")
    print("  - 图1: 日间、夜间参数对比柱状图")
    print("  - 图2: 五通道LED光谱分布")
    print("  - 图3: 最优权重配比参数")
    print("  - 图4: LED光谱合成效果")
    print("  - 图5: 综合性能雷达图")
    print("  - 图6: CIE色度图最优光照位置")

if __name__ == "__main__":
    print("LED光源分析系统初始化完成！")
    print("已设置CIE标准函数、普朗克轨迹和褪黑激素效应函数")
    print()

    # 运行问题2
    problem2_led_optimization()
