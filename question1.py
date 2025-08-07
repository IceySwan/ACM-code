# 导入必要的库
import pandas as pd  # 用于处理Excel表格数据
import colour  # 色彩科学计算库，用于光谱和色彩计算
import numpy as np  # 数值计算库
from colour import SpectralDistribution, sd_to_XYZ, XYZ_to_xy  # 导入光谱分布和色彩空间转换函数
from colour.temperature import CCT_to_uv_Krystek1985, xy_to_CCT_McCamy1992, uv_to_CCT_Ohno2013  # 导入色温计算函数

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

# 保存计算结果到CSV文件
output_csv_path = 'question1-output.csv'

# 将所有计算结果保存到CSV文件
df.to_csv(output_csv_path, index=False, encoding='utf-8-sig')  # 使用utf-8-sig编码支持中文

# 输出计算完成信息和结果摘要
print('\n=== 计算完成！ ===')
print(f'结果已保存到 {output_csv_path} 文件')
print(f"新增的数据列: bar_x(x̄函数值), bar_y(ȳ函数值), bar_z(z̄函数值), X(三刺激值), Y(三刺激值), Z(三刺激值), x(色品坐标), y(色品坐标), CCT(相关色温), Duv(色彩偏差)")

print(f"\n=== 光源特性分析结果 ===")
print(f"  CIE XYZ三刺激值: X={XYZ[0]:.4f}, Y={XYZ[1]:.4f}, Z={XYZ[2]:.4f}")
print(f"  色品坐标: x={xy[0]:.4f}, y={xy[1]:.4f}")
print(f"  相关色温: {CCT:.1f} K ({'暖白光' if CCT < 4000 else '中性白光' if CCT < 5000 else '冷白光'})")
print(f"  色彩偏差Duv: {Duv:.4f} ({'色彩质量优秀' if abs(Duv) < 0.003 else '色彩质量一般' if abs(Duv) < 0.01 else '色彩质量较差'})")

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

    f.write("=== 计算结果 ===\n")
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

    f.write("=== 分析结论 ===\n")
    f.write(f"光源类型: {'暖白光' if CCT < 4000 else '中性白光' if CCT < 5000 else '冷白光'}\n")
    f.write(f"色彩质量: {'优秀' if abs(Duv) < 0.003 else '一般' if abs(Duv) < 0.01 else '较差'}\n")
    f.write(f"推荐应用: {'家居照明、温馨环境' if CCT < 4000 else '办公、阅读等中性环境' if CCT < 5000 else '精密工作、医疗等高亮度环境'}\n")

print(f"\n详细分析报告已保存到: {summary_path}")
print(f"CSV数据文件已保存到: {output_csv_path}")
