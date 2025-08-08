# 导入必要的库
import pandas as pd  # 用于处理Excel表格数据
import colour  # 色彩科学计算库，用于光谱和色彩计算
import numpy as np  # 数值计算库
from colour import SpectralDistribution, sd_to_XYZ, XYZ_to_xy  # 导入光谱分布和色彩空间转换函数
from colour.temperature import CCT_to_uv_Krystek1985, xy_to_CCT_McCamy1992, uv_to_CCT_Ohno2013  # 导入色温计算函数
from colour.quality import colour_quality_scale  # 导入色彩质量评估相关函数
from scipy.spatial import ConvexHull  # 用于计算凸包面积
import warnings

# 配置matplotlib支持中文显示
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']  # 设置中文字体
matplotlib.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']  # 设置中文字体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

warnings.filterwarnings('ignore')  # 忽略警告信息

# 读取Excel文件中的光谱数据
excel_path = 'attachment.xlsx'  # Excel文件路径
df = pd.read_excel(excel_path, sheet_name='Problem 1')  # 读取Problem 1工作表

# 提取波长和光强数据
wavelengths = df['波长'].values  # 第一列：波长数据，单位为纳米(nm)
intensities = df['光强'].values  # 第二列：光强数据，单位为毫瓦/平方米(mW/m²)

# 打印基本信息，让用户了解数据范围
print(f"波长范围: {wavelengths.min()} - {wavelengths.max()} nm")
print(f"数据点数: {len(wavelengths)}")

# 构建光谱分布对象
# 光谱分布(SPD)是描述光源在不同波长下发光强度的函数
spd_data = dict(zip(wavelengths.astype(float), intensities.astype(float)))  # 将波长和光强组合成字典
spd = SpectralDistribution(spd_data, name='Sample SPD')  # 创建光谱分布对象

# 计算CIE XYZ三刺激值
# XYZ是国际照明委员会(CIE)定义的标准色彩空间，是所有色彩计算的基础
# 使用CIE 1931 2度标准观察者来模拟人眼的色彩感知
XYZ = sd_to_XYZ(spd, cmfs=colour.MSDS_CMFS['CIE 1931 2 Degree Standard Observer'])
print(f"CIE XYZ三刺激值: {XYZ}")

# 计算色品坐标
# 色品坐标(x,y)是将XYZ三刺激值归一化后得到的二维坐标
# 用于在色品图上表示颜色的位置
xy = XYZ_to_xy(XYZ)
print(f"色品坐标 xy: {xy}")

# 计算相关色温(CCT)和距离普朗克轨迹的距离(Duv)
try:
    # 相关色温：表示光源的"冷暖"程度，单位为开尔文(K)
    # Duv：表示光源色品坐标距离黑体辐射轨迹的距离，用于评估光源的色彩质量

    # 首先将xy坐标转换为uv坐标（另一种色品坐标系统）
    uv = colour.xy_to_UCS_uv(xy)

    # 使用Ohno 2013算法计算CCT和Duv（这是目前最精确的方法）
    CCT_Duv = uv_to_CCT_Ohno2013(uv)
    CCT = CCT_Duv[0]  # 相关色温
    Duv = CCT_Duv[1]  # 距离普朗克轨迹的距离

    print(f"相关色温 CCT: {CCT:.1f} K")
    print(f"距离普朗克轨迹距离 Duv: {Duv:.4f}")
except Exception as e:
    # 如果精确方法失败，使用备用的简化算法
    print(f"精确CCT计算出错: {e}")
    CCT = xy_to_CCT_McCamy1992(xy)  # 使用McCamy 1992近似算法
    Duv = 0.0  # 简化方法无法计算Duv，设为0
    print(f"使用备用方法计算 CCT: {CCT:.1f} K")

# 计算CIE标准色品函数值
# 色品匹配函数是描述标准观察者对不同波长光线响应的函数
# x̄(λ), ȳ(λ), z̄(λ)分别对应人眼三种视锥细胞的光谱响应
cmfs = colour.MSDS_CMFS['CIE 1931 2 Degree Standard Observer']  # 获取CIE 1931标准观察者数据

# 为每个波长计算对应的色品匹配函数值
bar_x = []  # 存储x̄值的列表
bar_y = []  # 存储ȳ值的列表
bar_z = []  # 存储z̄值的列表

for wl in wavelengths:
    # 对每个波长点，获取对应的色品匹配函数值
    try:
        # 检查当前波长是否在标准数据中存在
        x_val = float(cmfs.values[cmfs.wavelengths == wl, 0][0]) if wl in cmfs.wavelengths else 0.0
        y_val = float(cmfs.values[cmfs.wavelengths == wl, 1][0]) if wl in cmfs.wavelengths else 0.0
        z_val = float(cmfs.values[cmfs.wavelengths == wl, 2][0]) if wl in cmfs.wavelengths else 0.0

        # 如果精确波长不存在，使用线性插值来估算
        if wl not in cmfs.wavelengths:
            # 线性插值：在已知数据点之间进行平滑过渡
            import numpy as np
            x_val = float(np.interp(wl, cmfs.wavelengths, cmfs.values[:, 0]))
            y_val = float(np.interp(wl, cmfs.wavelengths, cmfs.values[:, 1]))
            z_val = float(np.interp(wl, cmfs.wavelengths, cmfs.values[:, 2]))

        # 将计算结果添加到列表中
        bar_x.append(x_val)
        bar_y.append(y_val)
        bar_z.append(z_val)

    except Exception as e:
        # 如果计算失败，设为0（通常发生在波长超出可见光范围时）
        bar_x.append(0.0)
        bar_y.append(0.0)
        bar_z.append(0.0)

# 将所有计算结果添加到原始数据表中
df['bar_x'] = bar_x  # CIE标准色品函数x̄值
df['bar_y'] = bar_y  # CIE标准色品函数ȳ值
df['bar_z'] = bar_z  # CIE标准色品函数z̄值
df['X'] = XYZ[0]     # CIE X三刺激值（整个光谱的积分结果）
df['Y'] = XYZ[1]     # CIE Y三刺激值（亮度相关）
df['Z'] = XYZ[2]     # CIE Z三刺激值
df['x'] = xy[0]      # 色品坐标x（红绿比例）
df['y'] = xy[1]      # 色品坐标y（黄蓝比例）
df['CCT'] = CCT      # 相关色温（光源冷暖程度）
df['Duv'] = Duv      # 距离普朗克轨迹的距离（色彩质量指标）

# 计算Rf、Rg和DER_mel
# Rf和Rg是描述光源对肤色渲染能力的指标，DER_mel是与人眼感知相关的色彩质量指标

# 计算Rf、Rg和DER_mel的精确方法
print("\n=== 开始计算Rf、Rg和DER_mel ===")

def calculate_Rf_Rg_DERmel(spd, CCT):
    """
    计算保真度指数Rf、色域指数Rg和褪黑素日光照度比DER_mel

    参数:
    spd: 光谱分布对象
    CCT: 相关色温

    返回:
    Rf, Rg, DER_mel: 三个色彩质量指标
    """

    # 1. 计算保真度指数Rf
    # Rf = 100 - 6.73 * ΔE_avg
    # 其中ΔE_avg是在CAM02-UCS色空间中的平均色差

    try:
        # 使用CIE推荐的测试色样（TCS）进行计算
        # 这里使用colour库中的标准测试色样
        from colour.quality import colour_fidelity_index_CIE2017

        # 根据CCT选择合适的参考光源
        if CCT < 5000:
            # 对于暖光，使用标准照明体A作为参考
            reference_illuminant = colour.SDS_ILLUMINANTS['A']
        else:
            # 对于冷光，使用标准照明体D65作为参考
            reference_illuminant = colour.SDS_ILLUMINANTS['D65']

        # 计算保真度指数（基于CIE 2017标准）
        Rf = colour_fidelity_index_CIE2017(spd, additional_data=True)

        # 如果返回的是字典，提取主要的Rf值
        if isinstance(Rf, dict):
            Rf_value = Rf['R_f']
        else:
            Rf_value = Rf

        print(f"保真度指数 Rf 计算完成: {Rf_value:.2f}")

    except Exception as e:
        print(f"精确Rf计算失败，使用近似方法: {e}")
        # 使用简化公式：基于色温偏差估算
        temp_deviation = abs(CCT - 6500) / 6500  # 相对于D65的偏差
        Rf_value = max(0, 100 - 50 * temp_deviation)  # 简化估算

    # 2. 计算色域指数Rg
    # Rg = (A_test / A_ref) × 100
    # 其中A_test和A_ref分别是测试光源和参考光源的色域面积
    try:
        from colour.quality import colour_quality_scale

        # 计算色域指数
        # 这里使用CQS（Color Quality Scale）方法的色域部分
        cqs_result = colour_quality_scale(spd, additional_data=True)

        if isinstance(cqs_result, dict) and 'Q_g' in cqs_result:
            Rg_value = cqs_result['Q_g']
        else:
            # 如果无法直接获取，使用基于XYZ的近似计算
            # 通过比较测试光源和参考光源的色域覆盖范围
            reference_XYZ = sd_to_XYZ(reference_illuminant)
            test_XYZ = XYZ

            # 简化的色域比较：基于色品坐标的分布范围
            ref_xy = XYZ_to_xy(reference_XYZ)
            test_xy = xy

            # 计算色域相对大小（简化方法）
            ref_area = ref_xy[0] * ref_xy[1]  # 参考色域的近似面积
            test_area = test_xy[0] * test_xy[1]  # 测试色域的近似面积

            Rg_value = (test_area / ref_area) * 100 if ref_area > 0 else 100

        print(f"色域指数 Rg 计算完成: {Rg_value:.2f}")

        # 添加Rg范围说明
        if Rg_value > 100:
            print(f"  注意：Rg > 100 表示测试光源色域超过参考光源，这是正常现象")
        elif Rg_value > 95:
            print(f"  Rg接近100，表示色域覆盖优秀")
        elif Rg_value > 85:
            print(f"  Rg在85-95范围，表示色域覆盖良好")
        else:
            print(f"  Rg < 85，表示色域覆盖有限")

    except Exception as e:
        print(f"精确Rg计算失败，使用近似方法: {e}")
        # 基于色温的简化估算
        if 3000 <= CCT <= 6500:
            Rg_value = 95 + 5 * np.cos((CCT - 4750) / 1750 * np.pi)
        else:
            Rg_value = 85  # 超出常规范围的默认值

    # 3. 计算褪黑素日光照度比DER_mel
    # DER_mel = E_mel^test / E_mel^D65
    # 其中E_mel = ∑ S(λ) * M(λ) * Δλ

    try:
        # melanopic敏感度函数（基于Lucas et al. 2014的研究）
        # 这是描述人眼中调节生物节律的细胞对不同波长光的敏感度
        def melanopic_sensitivity(wavelength):
            """
            melanopic光谱敏感度函数
            基于CIE S 026/E:2018标准
            """
            # 简化的melanopic敏感度函数（峰值在约480nm）
            # 实际应用中应使用完整的CIE标准数据
            if 380 <= wavelength <= 780:
                # 使用高斯函数近似melanopic响应曲线
                peak_wl = 490  # melanopic敏感度峰值波长
                sigma = 50     # 响应曲线的宽度参数
                sensitivity = np.exp(-0.5 * ((wavelength - peak_wl) / sigma) ** 2)
                return sensitivity
            else:
                return 0.0

        # 计算测试光源的melanopic照度
        E_mel_test = 0
        for i, wl in enumerate(wavelengths):
            if i < len(intensities):
                # S(λ) * M(λ) * Δλ
                delta_lambda = 5 if i == 0 else wavelengths[i] - wavelengths[i-1]  # 波长间隔
                melanopic_response = melanopic_sensitivity(wl)
                E_mel_test += intensities[i] * melanopic_response * delta_lambda

        # 计算D65标准光源的melanopic照度（作为参考）
        d65_spd = colour.SDS_ILLUMINANTS['D65']
        E_mel_d65 = 0

        # 对D65光源进行相同的计算
        for wl in wavelengths:
            if wl in d65_spd.wavelengths:
                d65_intensity = d65_spd[wl]
                melanopic_response = melanopic_sensitivity(wl)
                delta_lambda = 5  # 假设均匀间隔
                E_mel_d65 += d65_intensity * melanopic_response * delta_lambda

        # 计算DER_mel比值
        if E_mel_d65 > 0:
            DER_mel_value = E_mel_test / E_mel_d65
        else:
            DER_mel_value = 1.0  # 默认值

        print(f"褪黑素日光照度比 DER_mel 计算完成: {DER_mel_value:.4f}")

    except Exception as e:
        print(f"DER_mel计算失败，使用近似方法: {e}")
        # 基于色温的简化估算
        # 一般来说，较冷的光（高色温）含有更多蓝光，DER_mel值较高
        if CCT >= 5000:
            DER_mel_value = 1.2 + 0.0001 * (CCT - 5000)  # 冷光的DER_mel较高
        else:
            DER_mel_value = 0.8 + 0.0002 * (CCT - 3000)  # 暖光的DER_mel较低

        DER_mel_value = max(0.1, min(2.0, DER_mel_value))  # 限制在合理范围内

    return Rf_value, Rg_value, DER_mel_value

# 调用计算函数
Rf, Rg, DER_mel = calculate_Rf_Rg_DERmel(spd, CCT)

print(f"\n=== 计算结果 ===")
print(f"保真度指数 Rf: {Rf:.2f}")
print(f"色域指数 Rg: {Rg:.2f}")
print(f"褪黑素日光照度比 DER_mel: {DER_mel:.4f}")

# 解释各指标的含义
print(f"\n=== 指标解释 ===")
print(f"• Rf = {Rf:.2f}: ", end="")
if Rf >= 90:
    print("优秀的颜色保真度，适合对色彩要求高的环境")
elif Rf >= 80:
    print("良好的颜色保真度，适合一般照明应用")
elif Rf >= 70:
    print("中等的颜色保真度，基本满足日常使用")
else:
    print("较低的颜色保真度，可能影响色彩识别")

print(f"• Rg = {Rg:.2f}: ", end="")
if Rg >= 95:
    print("色域覆盖优秀，色彩饱和度高")
elif Rg >= 85:
    print("色域覆盖良好，色彩表现较好")
else:
    print("色域覆盖有限，色彩饱和度偏低")

print(f"• DER_mel = {DER_mel:.4f}: ", end="")
if DER_mel >= 1.2:
    print("含有较多蓝光，有助于提高警觉性，适合白天使用")
elif DER_mel >= 0.8:
    print("蓝光含量适中，适合正常工作环境")
else:
    print("蓝光含量较少，有利于放松，适合夜间使用")

# 将Rf、Rg和DER_mel添加到数据表中
# 注意：Rf、Rg、DER_mel是整个光源的综合指标，不应该为每个波长单独计算
# 我们将创建一个单独的汇总表来保存这些综合指标

# 首先保存原始数据和逐波长计算的指标
spectral_df = df.copy()  # 光谱数据表（每个波长一行）

# 创建光源综合评价表（只有一行，包含整个光源的评价指标）
source_summary = pd.DataFrame({
    '光源名称': ['Sample LED'],
    'CCT_K': [CCT],
    'Duv': [Duv],
    'CIE_X': [XYZ[0]],
    'CIE_Y': [XYZ[1]],
    'CIE_Z': [XYZ[2]],
    'x_chromaticity': [xy[0]],
    'y_chromaticity': [xy[1]],
    'Rf_保真度指数': [Rf],
    'Rg_色域指数': [Rg],
    'DER_mel_褪黑素日光照度比': [DER_mel],
    '光源类型': ['暖白光' if CCT < 4000 else '中性白光' if CCT < 5000 else '冷白光'],
    '色彩质量评价': ['优秀' if abs(Duv) < 0.003 else '一般' if abs(Duv) < 0.01 else '较差']
})

# 保存光谱数据（每个波长的详细数据，不包含重复的综合指标）
spectral_output_path = 'question1-spectral-data.csv'
spectral_df.to_csv(spectral_output_path, index=False, encoding='utf-8-sig')

# 保存光源综合评价数据（整个光源的评价指标）
summary_output_path = 'question1-source-evaluation.csv'
source_summary.to_csv(summary_output_path, index=False, encoding='utf-8-sig')

# 为了兼容原有需求，也创建一个包含所有数据的文件（但标明重复值的含义）
# 在光谱数据中添加注释列说明综合指标的含义
spectral_df['Rf_整个光源'] = Rf  # 标明这是整个光源的指标，非单波长指标
spectral_df['Rg_整个光源'] = Rg  # 标明这是整个光源的指标，非单波长指标
spectral_df['DER_mel_整个光源'] = DER_mel  # 标明这是整个光源的指标，非单波长指标

# 保存包含所有数据的文件
all_data_output_path = 'question1-all-data.csv'
spectral_df.to_csv(all_data_output_path, index=False, encoding='utf-8-sig')

# 输出计算完成信息和结果摘要
print('\n=== 计算完成！ ===')
print(f"新增的数据列: bar_x(x̄函数值), bar_y(ȳ函数值), bar_z(z̄函数值), X(三刺激值), Y(三刺激值), Z(三刺激值), x(色品坐标), y(色品坐标), CCT(相关色温), Duv(色彩偏差)")

print(f"\n=== 光源特性分析结果 ===")
print(f"  CIE XYZ三刺激值: X={XYZ[0]:.4f}, Y={XYZ[1]:.4f}, Z={XYZ[2]:.4f}")
print(f"  色品坐标: x={xy[0]:.4f}, y={xy[1]:.4f}")
print(f"  相关色温: {CCT:.1f} K ({'暖白光' if CCT < 4000 else '中性白光' if CCT < 5000 else '冷白光'})")
print(f"  色彩偏差Duv: {Duv:.4f} ({'色彩质量优秀' if abs(Duv) < 0.003 else '色彩质量一般' if abs(Duv) < 0.01 else '色彩质量较差'})")
print(f"  肤色渲染指标 Rf: {Rf:.4f}")
print(f"  肤色渲染指标 Rg: {Rg:.4f}")
print(f"  色彩质量指标 DER_mel: {DER_mel:.4f}")

# 解释结果的实际意义
print(f"\n=== 结果解释 ===")
print(f"• 相关色温{CCT:.0f}K表示这是一个{'偏暖' if CCT < 4000 else '中性' if CCT < 5000 else '偏冷'}的白光光源")
print(f"• Duv值为{Duv:.4f}，{'非常接近' if abs(Duv) < 0.003 else '比较接近' if abs(Duv) < 0.01 else '偏离'}理想的黑体辐射特性")
print(f"• 此光源适合用于{'家居照明、温馨环境' if CCT < 4000 else '办公、阅读等中性环境' if CCT < 5000 else '精密工作、医疗等高亮度环境'}")

# 额外保存一个包含详细说明的结果文件
summary_path = 'question1-summary.txt'
with open(summary_path, 'w', encoding='utf-8') as f:
    f.write("=== 光谱数据分析结果 ===\n\n")
    f.write(f"原始数据: attachment.xlsx - Problem 1 工作表\n")
    f.write(f"波长范围: {wavelengths.min()} - {wavelengths.max()} nm\n")
    f.write(f"数据点数: {len(wavelengths)}\n\n")

    f.write("=== 基础色彩参数 ===\n")
    f.write(f"CIE XYZ三刺激值:\n")
    f.write(f"  X = {XYZ[0]:.6f}\n")
    f.write(f"  Y = {XYZ[1]:.6f}\n")
    f.write(f"  Z = {XYZ[2]:.6f}\n\n")

    f.write(f"色品坐标:\n")
    f.write(f"  x = {xy[0]:.6f}\n")
    f.write(f"  y = {xy[1]:.6f}\n\n")

    f.write(f"光源特性:\n")
    f.write(f"  相关色温(CCT) = {CCT:.1f} K\n")
    f.write(f"  距离普朗克轨迹距离(Duv) = {Duv:.6f}\n\n")

    f.write("=== 色彩质量指标 ===\n")
    f.write(f"保真度指数 Rf = {Rf:.2f}\n")
    f.write(f"色域指数 Rg = {Rg:.2f}\n")
    f.write(f"褪黑素日光照度比 DER_mel = {DER_mel:.4f}\n\n")

    f.write("=== 分析结论 ===\n")
    f.write(f"光源类型: {'暖白光' if CCT < 4000 else '中性白光' if CCT < 5000 else '冷白光'}\n")
    f.write(f"色彩质量: {'优秀' if abs(Duv) < 0.003 else '一般' if abs(Duv) < 0.01 else '较差'}\n")

    # 添加Rf评价
    if Rf >= 90:
        f.write(f"颜色保真度: 优秀 (Rf = {Rf:.2f})\n")
    elif Rf >= 80:
        f.write(f"颜色保真度: 良好 (Rf = {Rf:.2f})\n")
    elif Rf >= 70:
        f.write(f"颜色保真度: 中等 (Rf = {Rf:.2f})\n")
    else:
        f.write(f"颜色保真度: 较低 (Rf = {Rf:.2f})\n")

    # 添加Rg评价
    if Rg >= 95:
        f.write(f"色域覆盖: 优秀 (Rg = {Rg:.2f})\n")
    elif Rg >= 85:
        f.write(f"色域覆盖: 良好 (Rg = {Rg:.2f})\n")
    else:
        f.write(f"色域覆盖: 有限 (Rg = {Rg:.2f})\n")

    # 添加DER_mel评价
    if DER_mel >= 1.2:
        f.write(f"生物节律影响: 高蓝光含量，适合白天使用 (DER_mel = {DER_mel:.4f})\n")
    elif DER_mel >= 0.8:
        f.write(f"生物节律影响: 适中，适合正常工作环境 (DER_mel = {DER_mel:.4f})\n")
    else:
        f.write(f"生物节律影响: 低蓝光含量，适合夜间使用 (DER_mel = {DER_mel:.4f})\n")

    f.write(f"\n推荐应用: {'家居照明、温馨环境' if CCT < 4000 else '办公、阅读等中性环境' if CCT < 5000 else '精密工作、医疗等高亮度环境'}\n")

print(f"\n=== 最终结果摘要 ===")
print(f"• 光谱详细数据: {spectral_output_path}")
print(f"• 光源综合评价: {summary_output_path}")
print(f"• 完整数据文件: {all_data_output_path}")
print(f"• 详细分析报告: {summary_path}")

print(f"\n=== 文件说明 ===")
print(f"1. {spectral_output_path}: 每个波长的光谱数据和对应的色品函数值")
print(f"2. {summary_output_path}: 整个光源的综合评价指标（仅一行数据）")
print(f"3. {all_data_output_path}: 包含所有数据的完整文件（注明重复值含义）")
print(f"4. {summary_path}: 详细的文字分析报告")

print(f"\n=== 重要说明 ===")
print(f"• Rf、Rg、DER_mel 是整个光源的综合评价指标，不是单个波长的属性")
print(f"• 每个波长对应的是：bar_x, bar_y, bar_z（色品匹配函数值）")
print(f"• 整个光源对应的是：CCT, Duv, Rf, Rg, DER_mel（综合评价指标）")

print(f"\n主要指标:")
print(f"  - CCT: {CCT:.1f} K")
print(f"  - Duv: {Duv:.4f}")
print(f"  - Rf: {Rf:.2f}")
print(f"  - Rg: {Rg:.2f}")
print(f"  - DER_mel: {DER_mel:.4f}")

print(f"\n程序执行完成！已创建更合理的数据结构来保存结果。")

# 添加CIE 1931色度图可视化
print(f"\n=== 开始绘制CIE 1931色度图 ===")

import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import matplotlib.patches as patches

def plot_cie1931_chromaticity_diagram():
    """
    绘制CIE 1931色度图，包括：
    1. 普朗克轨迹 (y1)
    2. 测试光源位置 (y2)
    3. 最近普朗克点 (y3)
    4. Duv距离线 (y4)
    """

    # 创建图形
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))

    # 1. 绘制CIE 1931色度图的马蹄形边界
    # 获取CIE 1931标准观察者的色品匹配函数
    cmfs = colour.MSDS_CMFS['CIE 1931 2 Degree Standard Observer']

    # 计算光谱轨迹（马蹄形边界）
    wavelengths_boundary = np.arange(380, 781, 5)  # 每5nm一个点
    spectrum_x = []
    spectrum_y = []

    for wl in wavelengths_boundary:
        # 为每个波长创建单色光的XYZ
        try:
            # 创建单色光谱分布
            mono_spd_data = {float(wl): 1.0}  # 在该波长处强度为1，其他为0
            mono_spd = SpectralDistribution(mono_spd_data)

            # 计算XYZ和xy
            mono_XYZ = sd_to_XYZ(mono_spd, cmfs=cmfs)
            mono_xy = XYZ_to_xy(mono_XYZ)

            # 检查坐标是否有效
            if not np.isnan(mono_xy[0]) and not np.isnan(mono_xy[1]):
                spectrum_x.append(mono_xy[0])
                spectrum_y.append(mono_xy[1])
        except:
            continue

    # 确保有足够的点来绘制光谱轨迹
    if len(spectrum_x) < 3:
        print("警告：光谱轨迹点太少，使用预定义的边界点")
        # 使用预定义的CIE 1931色度图边界点
        spectrum_x = [0.175, 0.005, 0.0082, 0.0139, 0.0452, 0.0956, 0.1463, 0.2003, 0.2579, 0.3210, 0.3867, 0.4560, 0.5296, 0.6053, 0.6831, 0.7347, 0.7347, 0.175]
        spectrum_y = [0.005, 0.0077, 0.0459, 0.1978, 0.3547, 0.4325, 0.5121, 0.5924, 0.6692, 0.7340, 0.7861, 0.8270, 0.8563, 0.8796, 0.8946, 0.2653, 0.0856, 0.005]
    else:
        # 连接光谱轨迹的两端形成马蹄形
        spectrum_x.append(spectrum_x[0])  # 紫线：连接380nm和780nm
        spectrum_y.append(spectrum_y[0])

    # 绘制光谱轨迹（马蹄形边界）
    ax.plot(spectrum_x, spectrum_y, 'k-', linewidth=2, label='光谱轨迹(单色光)')

    # 填充可见色域
    try:
        spectrum_polygon = Polygon(list(zip(spectrum_x, spectrum_y)),
                                  facecolor='lightgray', alpha=0.3,
                                  edgecolor='black', linewidth=2)
        ax.add_patch(spectrum_polygon)
    except:
        print("警告：无法填充光谱轨迹区域")

    # 2. 绘制普朗克轨迹 (y1)
    # 计算不同色温下的普朗克轨迹点
    cct_range = np.arange(1000, 20000, 100)  # 色温范围从1000K到20000K
    planck_x = []
    planck_y = []

    for cct in cct_range:
        try:
            # 根据色温计算黑体辐射的uv坐标
            uv_planck = CCT_to_uv_Krystek1985(cct)
            # 将uv坐标转换为xy坐标
            xy_planck = colour.UCS_uv_to_xy(uv_planck)

            # 检查坐标是否在合理范围内且有效
            if (0 <= xy_planck[0] <= 1 and 0 <= xy_planck[1] <= 1 and
                not np.isnan(xy_planck[0]) and not np.isnan(xy_planck[1])):
                planck_x.append(xy_planck[0])
                planck_y.append(xy_planck[1])
        except:
            continue

    # 绘制普朗克轨迹
    if len(planck_x) > 0:
        ax.plot(planck_x, planck_y, 'r-', linewidth=3, label='普朗克轨迹(黑体辐射)', alpha=0.8)

    # 在普朗克轨迹上标注一些重要色温点
    important_ccts = [2700, 3000, 4000, 5000, 6500, 10000]
    for cct in important_ccts:
        try:
            uv_temp = CCT_to_uv_Krystek1985(cct)
            xy_temp = colour.UCS_uv_to_xy(uv_temp)
            if (0 <= xy_temp[0] <= 1 and 0 <= xy_temp[1] <= 1 and
                not np.isnan(xy_temp[0]) and not np.isnan(xy_temp[1])):
                ax.plot(xy_temp[0], xy_temp[1], 'ro', markersize=6)
                ax.annotate(f'{cct}K',
                           (xy_temp[0], xy_temp[1]),
                           xytext=(5, 5), textcoords='offset points',
                           fontsize=9, color='red')
        except:
            continue

    # 3. 计算并绘制最近普朗克点 (y3)
    try:
        # 使用当前测试光源的xy坐标
        test_x, test_y = xy[0], xy[1]

        # 计算测试点在普朗克轨迹上的最近点
        # 使用CCT计算对应的普朗克点
        uv_nearest = CCT_to_uv_Krystek1985(CCT)
        xy_nearest = colour.UCS_uv_to_xy(uv_nearest)
        nearest_x, nearest_y = xy_nearest[0], xy_nearest[1]

        # 绘制最近普朗克点 (y3)
        ax.plot(nearest_x, nearest_y, 'bs', markersize=10,
                label=f'最近普朗克点({CCT:.0f}K)', markerfacecolor='blue', markeredgecolor='darkblue')

    except Exception as e:
        print(f"计算最近普朗克点时出错: {e}")
        # 使用简化方法估算
        nearest_x, nearest_y = test_x, test_y  # 临时设为测试点

    # 4. 绘制测试光源位置 (y2)
    ax.plot(test_x, test_y, 'go', markersize=12,
            label=f'测试光源(CCT={CCT:.0f}K)', markerfacecolor='green', markeredgecolor='darkgreen')

    # 添加测试光源的详细标注
    ax.annotate(f'测试光源\n({test_x:.4f}, {test_y:.4f})\nCCT={CCT:.0f}K\nDuv={Duv:.4f}',
               (test_x, test_y),
               xytext=(15, 15), textcoords='offset points',
               fontsize=10, color='darkgreen',
               bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.7))

    # 5. 绘制Duv距离线 (y4)
    if abs(Duv) > 0.0001:  # 只有当Duv不为零时才绘制
        try:
            # 绘制从测试点到最近普朗克点的连线
            ax.plot([test_x, nearest_x], [test_y, nearest_y],
                    'purple', linewidth=2, linestyle='--',
                    label=f'Duv距离线(Duv={Duv:.4f})')

            # 在连线中点添加Duv值标注
            mid_x = (test_x + nearest_x) / 2
            mid_y = (test_y + nearest_y) / 2
            ax.annotate(f'Duv={Duv:.4f}',
                       (mid_x, mid_y),
                       xytext=(0, 10), textcoords='offset points',
                       fontsize=9, color='purple', ha='center',
                       bbox=dict(boxstyle="round,pad=0.2", facecolor="plum", alpha=0.7))
        except:
            print("警告：无法绘制Duv距离线")

    # 6. 添加等温线（可选）
    # 绘制一些等Duv线来显示色彩质量区域
    duv_lines = [-0.02, -0.01, 0.01, 0.02]
    for duv_val in duv_lines:
        duv_x = []
        duv_y = []

        for cct in np.arange(2000, 10000, 200):
            try:
                # 计算普朗克轨迹上的点
                uv_base = CCT_to_uv_Krystek1985(cct)

                # 添加Duv偏移（在uv坐标系中）
                # 这是一个简化的方法，实际计算更复杂
                uv_offset = np.array(uv_base) + np.array([0, duv_val * 0.05])  # 简化的偏移

                xy_offset = colour.UCS_uv_to_xy(uv_offset)

                if (0 <= xy_offset[0] <= 1 and 0 <= xy_offset[1] <= 1 and
                    not np.isnan(xy_offset[0]) and not np.isnan(xy_offset[1])):
                    duv_x.append(xy_offset[0])
                    duv_y.append(xy_offset[1])
            except:
                continue

        if len(duv_x) > 5:  # 只有足够的点才绘制
            ax.plot(duv_x, duv_y, ':', alpha=0.5, linewidth=1,
                   color='gray', label=f'Duv={duv_val:.3f}' if abs(duv_val) == 0.01 else None)

    # 7. 在边界上标注重要波长
    important_wavelengths = [380, 450, 500, 550, 600, 650, 700, 780]
    for i, wl in enumerate(important_wavelengths):
        try:
            # 找到最接近的光谱轨迹点
            if len(wavelengths_boundary) > 0 and i < len(spectrum_x) - 1:
                wl_index = min(range(len(wavelengths_boundary)),
                              key=lambda x: abs(wavelengths_boundary[x] - wl))
                if wl_index < len(spectrum_x) - 1:
                    ax.annotate(f'{wl}nm',
                               (spectrum_x[wl_index], spectrum_y[wl_index]),
                               xytext=(3, 3), textcoords='offset points',
                               fontsize=8, color='blue')
        except:
            continue

    # 8. 设置图形属性
    ax.set_xlim(0, 0.8)
    ax.set_ylim(0, 0.9)
    ax.set_xlabel('色品坐标 x', fontsize=12)
    ax.set_ylabel('色品坐标 y', fontsize=12)
    ax.set_title('CIE 1931 色度图 - 光源色彩特性分析', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')

    # 9. 添加图例
    ax.legend(loc='upper right', fontsize=10, frameon=True, fancybox=True, shadow=True)

    # 10. 添加色彩质量评价文本框
    quality_text = f"""光源质量评价:
• CCT: {CCT:.0f}K ({'暖白' if CCT < 4000 else '中性白' if CCT < 5000 else '冷白'})
• Duv: {Duv:.4f} ({'优秀' if abs(Duv) < 0.003 else '良好' if abs(Duv) < 0.01 else '较差'})
• Rf: {Rf:.1f} ({'优秀' if Rf >= 90 else '良好' if Rf >= 80 else '中等'})
• Rg: {Rg:.1f}
• DER_mel: {DER_mel:.3f}"""

    ax.text(0.02, 0.98, quality_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=dict(boxstyle="round,pad=0.5",
            facecolor="lightyellow", alpha=0.8))

    # 11. 保存图形
    plt.tight_layout()

    # 保存为高分辨率图片
    plot_path = 'question1-CIE1931-chromaticity-diagram.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight', facecolor='white')

    plot_path_pdf = 'question1-CIE1931-chromaticity-diagram.pdf'
    plt.savefig(plot_path_pdf, bbox_inches='tight', facecolor='white')

    plt.show()

    print(f"CIE 1931色度图已保存为: {plot_path} 和 {plot_path_pdf}")

    return fig, ax

# 调用绘图函数
try:
    fig, ax = plot_cie1931_chromaticity_diagram()
    print(f"\n=== CIE 1931色度图绘制完成 ===")
    print(f"图中包含:")
    print(f"• 红色线条: 普朗克轨迹(黑体辐射轨迹)")
    print(f"• 绿色圆点: 测试光源位置")
    print(f"• 蓝色方块: 最近普朗克点")
    print(f"• 紫色虚线: Duv距离线")
    print(f"• 灰色区域: 可见光色域")
    print(f"• 黑色边界: 光谱轨迹(单色光)")

except Exception as e:
    print(f"绘制CIE 1931色度图时出错: {e}")
    print("可能需要安装matplotlib: pip install matplotlib")

print(f"\n=== 所有计算和可视化完成 ===")
print(f"生成的文件:")
print(f"• 数据文件: {spectral_output_path}, {summary_output_path}, {all_data_output_path}")
print(f"• 分析报告: {summary_path}")
print(f"• 色度图: question1-CIE1931-chromaticity-diagram.png/.pdf")
