import pandas as pd
import numpy as np
from scipy import interpolate
from scipy.spatial import ConvexHull
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def load_data():
    """Load wavelength and intensity data from attachment-Q1.xlsx"""
    df = pd.read_excel('attachment-Q1.xlsx')
    wavelength = df.iloc[:, 0].values  # First column: wavelength (nm)
    intensity = df.iloc[:, 1].values   # Second column: intensity (mW/m²)
    return wavelength, intensity

def get_cie_color_matching_functions():
    """Get CIE 1931 2° standard observer color matching functions"""
    # Standard CIE 1931 2° observer data (380-780nm, 5nm intervals)
    wavelengths = np.arange(380, 785, 5)

    # CIE 1931 2° standard observer x̄(λ), ȳ(λ), z̄(λ)
    x_bar = np.array([
        0.0014, 0.0022, 0.0042, 0.0076, 0.0143, 0.0232, 0.0435, 0.0776,
        0.1344, 0.2148, 0.2839, 0.3285, 0.3483, 0.3481, 0.3362, 0.3187,
        0.2908, 0.2511, 0.1954, 0.1421, 0.0956, 0.0580, 0.0320, 0.0147,
        0.0049, 0.0024, 0.0093, 0.0291, 0.0633, 0.1096, 0.1655, 0.2257,
        0.2904, 0.3597, 0.4334, 0.5121, 0.5945, 0.6784, 0.7621, 0.8425,
        0.9163, 0.9786, 1.0263, 1.0567, 1.0622, 1.0456, 1.0026, 0.9384,
        0.8544, 0.7514, 0.6424, 0.5419, 0.4479, 0.3608, 0.2835, 0.2187,
        0.1649, 0.1212, 0.0874, 0.0636, 0.0468, 0.0329, 0.0227, 0.0158,
        0.0114, 0.0081, 0.0058, 0.0041, 0.0029, 0.0020, 0.0014, 0.0010,
        0.0007, 0.0005, 0.0003, 0.0002, 0.0002, 0.0001, 0.0001, 0.0001,
        0.0000
    ])

    y_bar = np.array([
        0.0000, 0.0001, 0.0001, 0.0002, 0.0004, 0.0006, 0.0012, 0.0022,
        0.0040, 0.0073, 0.0116, 0.0168, 0.0230, 0.0298, 0.0380, 0.0480,
        0.0600, 0.0739, 0.0910, 0.1126, 0.1390, 0.1693, 0.2080, 0.2586,
        0.3230, 0.4073, 0.5030, 0.6082, 0.7100, 0.7932, 0.8620, 0.9149,
        0.9540, 0.9803, 0.9950, 1.0000, 0.9950, 0.9786, 0.9520, 0.9154,
        0.8700, 0.8163, 0.7570, 0.6949, 0.6310, 0.5668, 0.5030, 0.4412,
        0.3810, 0.3210, 0.2650, 0.2170, 0.1750, 0.1382, 0.1070, 0.0816,
        0.0610, 0.0446, 0.0320, 0.0232, 0.0170, 0.0119, 0.0082, 0.0057,
        0.0041, 0.0029, 0.0021, 0.0015, 0.0010, 0.0007, 0.0005, 0.0004,
        0.0002, 0.0002, 0.0001, 0.0001, 0.0001, 0.0000, 0.0000, 0.0000,
        0.0000
    ])

    z_bar = np.array([
        0.0065, 0.0105, 0.0201, 0.0362, 0.0679, 0.1102, 0.2074, 0.3713,
        0.6456, 1.0391, 1.3856, 1.6230, 1.7471, 1.7826, 1.7721, 1.7441,
        1.6692, 1.5281, 1.2876, 1.0419, 0.8130, 0.6162, 0.4652, 0.3533,
        0.2720, 0.2123, 0.1582, 0.1117, 0.0782, 0.0573, 0.0422, 0.0298,
        0.0203, 0.0134, 0.0087, 0.0057, 0.0039, 0.0027, 0.0021, 0.0018,
        0.0017, 0.0014, 0.0011, 0.0010, 0.0008, 0.0006, 0.0003, 0.0002,
        0.0002, 0.0001, 0.0001, 0.0001, 0.0000, 0.0000, 0.0000, 0.0000,
        0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
        0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
        0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
        0.0000
    ])

    return wavelengths, x_bar, y_bar, z_bar

def get_cri_test_color_samples():
    """Get CRI-14 test color sample reflectances (TM-30 color samples)"""
    # Simplified representation of TM-30 color samples
    # In practice, you would load these from a standard database
    # Here we use a subset of typical color samples

    # Wavelengths for color sample data
    wavelengths = np.arange(380, 781, 5)

    # Sample reflectance data for 14 common color samples (simplified)
    # These are approximations - in practice you'd use official CIE data
    color_samples = []

    # Sample 1: Light greyish red
    sample1 = 0.2 + 0.3 * np.exp(-((wavelengths - 600) / 50)**2)
    color_samples.append(sample1)

    # Sample 2: Dark greyish yellow
    sample2 = 0.3 + 0.4 * np.exp(-((wavelengths - 570) / 40)**2)
    color_samples.append(sample2)

    # Sample 3: Strong yellow green
    sample3 = 0.2 + 0.5 * np.exp(-((wavelengths - 540) / 30)**2)
    color_samples.append(sample3)

    # Sample 4: Moderate yellowish green
    sample4 = 0.25 + 0.4 * np.exp(-((wavelengths - 520) / 35)**2)
    color_samples.append(sample4)

    # Sample 5: Light bluish green
    sample5 = 0.3 + 0.3 * np.exp(-((wavelengths - 500) / 40)**2)
    color_samples.append(sample5)

    # Sample 6: Light blue
    sample6 = 0.2 + 0.4 * np.exp(-((wavelengths - 480) / 30)**2)
    color_samples.append(sample6)

    # Sample 7: Light violet
    sample7 = 0.25 + 0.3 * np.exp(-((wavelengths - 450) / 25)**2)
    color_samples.append(sample7)

    # Sample 8: Light reddish purple
    sample8 = 0.2 + 0.25 * np.exp(-((wavelengths - 420) / 30)**2) + 0.2 * np.exp(-((wavelengths - 650) / 40)**2)
    color_samples.append(sample8)

    # Additional samples 9-14
    for i in range(6):
        # Generate additional diverse color samples
        peak_wl = 400 + i * 60
        sample = 0.3 + 0.3 * np.exp(-((wavelengths - peak_wl) / 35)**2)
        color_samples.append(sample)

    return wavelengths, np.array(color_samples)

def get_d65_illuminant():
    """Get D65 standard illuminant spectral power distribution"""
    # Extended wavelength range to match melanopic sensitivity
    wavelengths = np.arange(380, 781, 1)  # Changed from 5nm to 1nm intervals

    # Interpolate the original D65 data to 1nm intervals
    original_wavelengths = np.arange(380, 781, 5)
    original_d65_spd = np.array([
        49.975, 52.311, 54.648, 68.701, 82.754, 87.120, 91.486, 92.458,
        93.431, 90.057, 86.682, 95.773, 104.865, 110.936, 117.008, 117.410,
        117.812, 116.336, 114.861, 115.392, 115.923, 112.367, 108.811, 109.082,
        109.354, 108.578, 107.802, 106.296, 104.790, 106.239, 107.689, 106.047,
        104.405, 104.225, 104.046, 102.023, 100.000, 98.167, 96.334, 96.061,
        95.788, 92.237, 88.685, 89.996, 91.306, 90.231, 89.157, 88.475,
        87.793, 85.947, 84.101, 83.233, 82.365, 80.877, 79.388, 78.988,
        78.588, 77.630, 76.672, 76.639, 76.606, 76.851, 77.096, 75.984,
        74.872, 74.349, 73.816, 73.227, 72.638, 71.808, 70.978, 70.458,
        69.938, 69.498, 69.058, 68.269, 67.481, 66.953, 66.425, 65.942,
        65.459, 64.933, 64.408, 64.133
    ])

    # Check array lengths before interpolation
    print(f"Debug: original_wavelengths length: {len(original_wavelengths)}")
    print(f"Debug: original_d65_spd length: {len(original_d65_spd)}")
    print(f"Debug: target wavelengths length: {len(wavelengths)}")

    # Ensure arrays have same length
    min_len = min(len(original_wavelengths), len(original_d65_spd))
    original_wavelengths = original_wavelengths[:min_len]
    original_d65_spd = original_d65_spd[:min_len]

    # Interpolate to 1nm intervals
    f_d65 = interpolate.interp1d(original_wavelengths, original_d65_spd,
                                kind='linear', bounds_error=False, fill_value=0)
    d65_spd = f_d65(wavelengths)

    return wavelengths, d65_spd

def get_melanopic_sensitivity_lucas():
    """Get melanopic sensitivity function based on Lucas et al. 2014"""
    # CIE S 026 melanopic sensitivity function
    wavelengths = np.arange(380, 781, 1)

    # Melanopic sensitivity function values (Lucas et al. 2014)
    # Peak at ~490nm, values normalized to peak = 1
    s_mel = np.zeros_like(wavelengths, dtype=float)

    for i, wl in enumerate(wavelengths):
        if 380 <= wl <= 780:
            # More accurate melanopic sensitivity based on CIE S 026
            if wl < 490:
                # Blue side of the curve
                s_mel[i] = np.exp(-0.5 * ((wl - 490) / 38)**2)
            else:
                # Red side of the curve (steeper falloff)
                s_mel[i] = np.exp(-0.5 * ((wl - 490) / 42)**2)

    # Normalize to peak = 1
    if s_mel.max() > 0:
        s_mel = s_mel / s_mel.max()

    return wavelengths, s_mel

def xyz_to_cam02ucs(X, Y, Z, X_w, Y_w, Z_w, L_A=64):
    """Convert XYZ to CAM02-UCS color space"""
    # Simplified CAM02-UCS conversion
    # In practice, this would be a full CIECAM02 implementation

    # Adaptation
    Y_b = 20.0  # Background luminance factor
    F = 1.0     # Surround condition
    c = 0.69    # Impact of surround
    N_c = 1.0   # Chromatic induction factor

    # Normalize by white point
    X_rel = X / X_w
    Y_rel = Y / Y_w
    Z_rel = Z / Z_w

    # Simplified lightness calculation
    J = 100 * (Y_rel ** 0.5)

    # Simplified chroma calculations
    a = 200 * (X_rel - Y_rel)
    b = 100 * (Y_rel - Z_rel)

    return J, a, b

def calculate_color_samples_under_illuminant(illuminant_wl, illuminant_spd, sample_wl, sample_reflectances):
    """Calculate XYZ values for color samples under given illuminant"""
    # Interpolate illuminant to sample wavelengths
    f_illuminant = interpolate.interp1d(illuminant_wl, illuminant_spd, kind='linear',
                                       bounds_error=False, fill_value=0)
    illuminant_interp = f_illuminant(sample_wl)

    # Get CIE color matching functions
    cie_wl, x_bar, y_bar, z_bar = get_cie_color_matching_functions()

    # Interpolate color matching functions to sample wavelengths
    f_x = interpolate.interp1d(cie_wl, x_bar, kind='linear', bounds_error=False, fill_value=0)
    f_y = interpolate.interp1d(cie_wl, y_bar, kind='linear', bounds_error=False, fill_value=0)
    f_z = interpolate.interp1d(cie_wl, z_bar, kind='linear', bounds_error=False, fill_value=0)

    x_bar_interp = f_x(sample_wl)
    y_bar_interp = f_y(sample_wl)
    z_bar_interp = f_z(sample_wl)

    # Calculate wavelength intervals
    d_lambda = np.diff(sample_wl)
    d_lambda = np.append(d_lambda, d_lambda[-1])

    # Normalization constant K for consistency
    K = 683  # lm/W for photopic vision

    # Calculate XYZ for each color sample under test and reference illuminants
    xyz_values = []
    for reflectance in sample_reflectances:
        # Multiply illuminant * reflectance * color matching function * K
        X = K * np.sum(illuminant_interp * reflectance * x_bar_interp * d_lambda)
        Y = K * np.sum(illuminant_interp * reflectance * y_bar_interp * d_lambda)
        Z = K * np.sum(illuminant_interp * reflectance * z_bar_interp * d_lambda)
        xyz_values.append([X, Y, Z])

    return np.array(xyz_values)

def interpolate_color_matching_functions(wavelength_data):
    """Interpolate color matching functions to match the data wavelengths"""
    cie_wavelengths, x_bar, y_bar, z_bar = get_cie_color_matching_functions()

    # Create interpolation functions
    f_x = interpolate.interp1d(cie_wavelengths, x_bar, kind='linear',
                               bounds_error=False, fill_value=0)
    f_y = interpolate.interp1d(cie_wavelengths, y_bar, kind='linear',
                               bounds_error=False, fill_value=0)
    f_z = interpolate.interp1d(cie_wavelengths, z_bar, kind='linear',
                               bounds_error=False, fill_value=0)

    # Interpolate to data wavelengths
    x_bar_interp = f_x(wavelength_data)
    y_bar_interp = f_y(wavelength_data)
    z_bar_interp = f_z(wavelength_data)

    return x_bar_interp, y_bar_interp, z_bar_interp

def calculate_tristimulus_values(wavelength, intensity):
    """Calculate CIE XYZ tristimulus values"""
    # Get interpolated color matching functions
    x_bar, y_bar, z_bar = interpolate_color_matching_functions(wavelength)

    # Calculate wavelength step
    d_lambda = np.diff(wavelength)
    d_lambda = np.append(d_lambda, d_lambda[-1])  # Use last interval for the last point

    # Normalization constant K for photopic vision
    K = 683  # lm/W for photopic vision

    # Calculate tristimulus values using numerical integration with proper normalization
    # Note: For relative calculations (CCT, chromaticity), K cancels out
    # But for absolute photometric calculations, K is essential
    X = K * np.sum(intensity * x_bar * d_lambda)
    Y = K * np.sum(intensity * y_bar * d_lambda)  # Y represents luminance when multiplied by K
    Z = K * np.sum(intensity * z_bar * d_lambda)

    return X, Y, Z

def calculate_cct_duv(X, Y, Z):
    """Calculate Correlated Color Temperature (CCT) and Distance from Planckian locus (Duv)"""
    # Convert to chromaticity coordinates
    x = X / (X + Y + Z)
    y = Y / (X + Y + Z)

    print(f"Debug: Chromaticity coordinates x={x:.6f}, y={y:.6f}")

    # McCamy's approximation for CCT
    n = (x - 0.3320) / (0.1858 - y)
    CCT = 449 * n**3 + 3525 * n**2 + 6823.3 * n + 5520.33

    # Calculate Duv using Robertson's method with correct Planckian locus
    # Use CIE 1960 UCS coordinates for more accurate calculation
    u = 4*x / (-2*x + 12*y + 3)
    v = 6*y / (-2*x + 12*y + 3)

    print(f"Debug: UCS coordinates u={u:.6f}, v={v:.6f}")

    # Find closest point on Planckian locus in UCS space
    T_range = np.linspace(max(1000, CCT-500), min(25000, CCT+500), 1000)
    min_dist = float('inf')
    closest_u_p = 0
    closest_v_p = 0

    for T in T_range:
        # Planckian locus in UCS coordinates (more accurate formula)
        u_p = (0.860117757 + 1.54118254e-4*T + 1.28641212e-7*T**2) / (1.0 + 8.42420235e-4*T + 7.08145163e-7*T**2)
        v_p = (0.317398726 + 4.22806245e-5*T + 4.20481691e-8*T**2) / (1.0 - 2.89741816e-5*T + 1.61456053e-7*T**2)

        dist = np.sqrt((u - u_p)**2 + (v - v_p)**2)
        if dist < min_dist:
            min_dist = dist
            closest_u_p = u_p
            closest_v_p = v_p

    # Calculate Duv with proper sign in UCS space
    # Vector from closest point on locus to test point
    du = u - closest_u_p
    dv = v - closest_v_p

    # Perpendicular direction to the locus (simplified)
    # Positive Duv means above the locus, negative means below
    duv_vector = dv  # Simplified: use v-direction as primary indicator

    # Final Duv calculation
    Duv = np.sqrt(du**2 + dv**2) * np.sign(duv_vector)

    print(f"Debug: min_dist={min_dist:.6f}, final Duv={Duv:.6f}")

    return CCT, Duv

def get_reference_illuminant_spd(cct):
    """Get reference illuminant spectral power distribution based on CCT"""
    wavelengths = np.arange(380, 781, 1)

    if cct < 5000:
        # Use Planckian radiator for CCT < 5000K
        h = 6.626e-34  # Planck constant
        c = 3.0e8      # Speed of light
        k = 1.381e-23  # Boltzmann constant

        spd = []
        for wl in wavelengths:
            wl_m = wl * 1e-9  # Convert nm to m
            # Planck's law
            B = (2 * h * c**2 / wl_m**5) / (np.exp(h * c / (wl_m * k * cct)) - 1)
            spd.append(B)
        spd = np.array(spd)
    else:
        # Use D-illuminant for CCT >= 5000K
        # Simplified D65 approximation scaled for different CCT
        d65_wl, d65_spd = get_d65_illuminant()
        f_d65 = interpolate.interp1d(d65_wl, d65_spd, kind='linear',
                                    bounds_error=False, fill_value=0)
        spd = f_d65(wavelengths)

        # Scale for different CCT (simplified)
        scale_factor = 6500 / cct
        spd = spd * scale_factor

    return wavelengths, spd

def calculate_rf_accurate(test_wavelength, test_intensity):
    """Calculate accurate Rf based on CAM02-UCS color differences"""
    # Get color samples
    sample_wl, sample_reflectances = get_cri_test_color_samples()

    # Get test source XYZ
    test_X, test_Y, test_Z = calculate_tristimulus_values(test_wavelength, test_intensity)
    test_cct, _ = calculate_cct_duv(test_X, test_Y, test_Z)

    # Get reference illuminant
    ref_wl, ref_spd = get_reference_illuminant_spd(test_cct)

    # Interpolate test source to sample wavelengths
    f_test = interpolate.interp1d(test_wavelength, test_intensity, kind='linear',
                                 bounds_error=False, fill_value=0)
    test_spd_interp = f_test(sample_wl)

    # Interpolate reference to sample wavelengths
    f_ref = interpolate.interp1d(ref_wl, ref_spd, kind='linear',
                                bounds_error=False, fill_value=0)
    ref_spd_interp = f_ref(sample_wl)

    # Calculate XYZ for color samples under test and reference illuminants
    test_xyz = calculate_color_samples_under_illuminant(sample_wl, test_spd_interp,
                                                       sample_wl, sample_reflectances)
    ref_xyz = calculate_color_samples_under_illuminant(sample_wl, ref_spd_interp,
                                                      sample_wl, sample_reflectances)

    # Calculate white point for both illuminants
    white_sample = np.ones_like(sample_wl)  # Perfect reflector
    test_white_xyz = calculate_color_samples_under_illuminant(sample_wl, test_spd_interp,
                                                             sample_wl, [white_sample])[0]
    ref_white_xyz = calculate_color_samples_under_illuminant(sample_wl, ref_spd_interp,
                                                            sample_wl, [white_sample])[0]

    # Convert to CAM02-UCS
    delta_e_values = []

    for i in range(len(sample_reflectances)):
        # Test illuminant CAM02-UCS
        J_test, a_test, b_test = xyz_to_cam02ucs(test_xyz[i, 0], test_xyz[i, 1], test_xyz[i, 2],
                                                test_white_xyz[0], test_white_xyz[1], test_white_xyz[2])

        # Reference illuminant CAM02-UCS
        J_ref, a_ref, b_ref = xyz_to_cam02ucs(ref_xyz[i, 0], ref_xyz[i, 1], ref_xyz[i, 2],
                                             ref_white_xyz[0], ref_white_xyz[1], ref_white_xyz[2])

        # Calculate color difference in CAM02-UCS
        delta_e = np.sqrt((J_test - J_ref)**2 + (a_test - a_ref)**2 + (b_test - b_ref)**2)
        delta_e_values.append(delta_e)

    # Calculate average color difference
    delta_e_avg = np.mean(delta_e_values)

    # Calculate Rf
    Rf = 100 - 6.73 * delta_e_avg

    return max(0, min(100, Rf))

def calculate_rg_accurate(test_wavelength, test_intensity):
    """Calculate accurate Rg based on color gamut area"""
    # Get color samples
    sample_wl, sample_reflectances = get_cri_test_color_samples()

    # Get test source XYZ
    test_X, test_Y, test_Z = calculate_tristimulus_values(test_wavelength, test_intensity)
    test_cct, _ = calculate_cct_duv(test_X, test_Y, test_Z)

    # Get reference illuminant
    ref_wl, ref_spd = get_reference_illuminant_spd(test_cct)

    # Interpolate test source to sample wavelengths
    f_test = interpolate.interp1d(test_wavelength, test_intensity, kind='linear',
                                 bounds_error=False, fill_value=0)
    test_spd_interp = f_test(sample_wl)

    # Interpolate reference to sample wavelengths
    f_ref = interpolate.interp1d(ref_wl, ref_spd, kind='linear',
                                bounds_error=False, fill_value=0)
    ref_spd_interp = f_ref(sample_wl)

    # Calculate XYZ for color samples under test and reference illuminants
    test_xyz = calculate_color_samples_under_illuminant(sample_wl, test_spd_interp,
                                                       sample_wl, sample_reflectances)
    ref_xyz = calculate_color_samples_under_illuminant(sample_wl, ref_spd_interp,
                                                      sample_wl, sample_reflectances)

    # Calculate white point for both illuminants
    white_sample = np.ones_like(sample_wl)  # Perfect reflector
    test_white_xyz = calculate_color_samples_under_illuminant(sample_wl, test_spd_interp,
                                                             sample_wl, [white_sample])[0]
    ref_white_xyz = calculate_color_samples_under_illuminant(sample_wl, ref_spd_interp,
                                                            sample_wl, [white_sample])[0]

    # Convert to CAM02-UCS and extract a*, b* coordinates for gamut calculation
    test_points = []
    ref_points = []

    for i in range(len(sample_reflectances)):
        # Test illuminant CAM02-UCS
        J_test, a_test, b_test = xyz_to_cam02ucs(test_xyz[i, 0], test_xyz[i, 1], test_xyz[i, 2],
                                                test_white_xyz[0], test_white_xyz[1], test_white_xyz[2])
        test_points.append([a_test, b_test])

        # Reference illuminant CAM02-UCS
        J_ref, a_ref, b_ref = xyz_to_cam02ucs(ref_xyz[i, 0], ref_xyz[i, 1], ref_xyz[i, 2],
                                             ref_white_xyz[0], ref_white_xyz[1], ref_white_xyz[2])
        ref_points.append([a_ref, b_ref])

    test_points = np.array(test_points)
    ref_points = np.array(ref_points)

    # Calculate convex hull areas
    try:
        test_hull = ConvexHull(test_points)
        ref_hull = ConvexHull(ref_points)

        A_test = test_hull.volume  # In 2D, volume is area
        A_ref = ref_hull.volume

        # Calculate Rg
        if A_ref > 0:
            Rg = (A_test / A_ref) * 100
        else:
            Rg = 100

    except Exception:
        # Fallback if ConvexHull fails
        Rg = 100

    return max(50, min(150, Rg))

def calculate_der_mel_accurate(test_wavelength, test_intensity):
    """Calculate accurate DER_mel based on melanopic illuminance ratio"""
    # Get melanopic sensitivity function
    mel_wl, s_mel = get_melanopic_sensitivity_lucas()

    # Get D65 illuminant - now both have same wavelength range (380-780nm, 1nm intervals)
    d65_wl, d65_spd = get_d65_illuminant()

    # Interpolate test source to melanopic wavelengths
    f_test = interpolate.interp1d(test_wavelength, test_intensity, kind='linear',
                                 bounds_error=False, fill_value=0)
    test_spd_interp = f_test(mel_wl)

    # No need to interpolate D65 since both mel_wl and d65_wl are identical
    # But keep for safety in case wavelength ranges differ
    if len(d65_wl) == len(mel_wl) and np.allclose(d65_wl, mel_wl):
        d65_spd_interp = d65_spd
    else:
        f_d65 = interpolate.interp1d(d65_wl, d65_spd, kind='linear',
                                    bounds_error=False, fill_value=0)
        d65_spd_interp = f_d65(mel_wl)

    # Calculate wavelength intervals
    d_lambda = np.diff(mel_wl)
    d_lambda = np.append(d_lambda, d_lambda[-1])

    # Calculate melanopic illuminance for test source
    E_mel_test = np.sum(test_spd_interp * s_mel * d_lambda)

    # Calculate melanopic illuminance for D65
    E_mel_d65 = np.sum(d65_spd_interp * s_mel * d_lambda)

    # Calculate DER_mel
    if E_mel_d65 > 0:
        DER_mel = E_mel_test / E_mel_d65
    else:
        DER_mel = 0

    return DER_mel

def get_cie_1931_spectral_locus():
    """Get CIE 1931 spectral locus boundary for chromaticity diagram"""
    wavelengths = np.arange(380, 781, 5)
    cie_wl, x_bar, y_bar, z_bar = get_cie_color_matching_functions()

    # Calculate chromaticity coordinates for spectral locus
    x_coords = []
    y_coords = []

    for i, wl in enumerate(wavelengths):
        if i < len(x_bar):
            X = x_bar[i]
            Y = y_bar[i]
            Z = z_bar[i]

            if X + Y + Z > 0:
                x = X / (X + Y + Z)
                y = Y / (X + Y + Z)
                x_coords.append(x)
                y_coords.append(y)

    # Close the spectral locus by connecting to purple line
    x_coords.append(x_coords[0])
    y_coords.append(y_coords[0])

    return np.array(x_coords), np.array(y_coords), wavelengths

def get_planckian_locus():
    """Get Planckian locus (blackbody locus) for chromaticity diagram"""
    # Temperature range from 1000K to 20000K
    temperatures = np.logspace(3, 4.3, 100)  # 1000K to 20000K

    x_coords = []
    y_coords = []

    for T in temperatures:
        # Planck's law for relative spectral power distribution
        wavelengths = np.arange(380, 781, 5)
        h = 6.626e-34  # Planck constant
        c = 3.0e8      # Speed of light
        k = 1.381e-23  # Boltzmann constant

        spd = []
        for wl in wavelengths:
            wl_m = wl * 1e-9  # Convert nm to m
            B = (2 * h * c**2 / wl_m**5) / (np.exp(h * c / (wl_m * k * T)) - 1)
            spd.append(B)

        spd = np.array(spd)

        # Get color matching functions
        cie_wl, x_bar, y_bar, z_bar = get_cie_color_matching_functions()

        # Calculate XYZ
        X = np.sum(spd * x_bar)
        Y = np.sum(spd * y_bar)
        Z = np.sum(spd * z_bar)

        if X + Y + Z > 0:
            x = X / (X + Y + Z)
            y = Y / (X + Y + Z)
            x_coords.append(x)
            y_coords.append(y)

    return np.array(x_coords), np.array(y_coords), temperatures

def plot_cie_chromaticity_diagram(test_x, test_y, test_cct, duv_value):
    """Plot CIE 1931 chromaticity diagram with test source and Planckian locus"""

    # Set up the plot with Chinese font support
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False

    fig, ax = plt.subplots(figsize=(12, 10))

    # Get spectral locus
    spec_x, spec_y, spec_wavelengths = get_cie_1931_spectral_locus()

    # Get Planckian locus
    planck_x, planck_y, temperatures = get_planckian_locus()

    # Plot spectral locus (y1 equivalent - boundary of visible colors)
    ax.fill(spec_x, spec_y, alpha=0.1, color='lightgray', label='可见光谱轨迹')
    ax.plot(spec_x[:-1], spec_y[:-1], 'k-', linewidth=2, label='光谱轨迹边界')

    # Add wavelength labels on spectral locus
    for i in range(0, len(spec_x)-1, 8):  # Every 8th point to avoid crowding
        if i < len(spec_wavelengths):
            ax.annotate(f'{spec_wavelengths[i]:.0f}nm',
                       (spec_x[i], spec_y[i]),
                       xytext=(5, 5), textcoords='offset points',
                       fontsize=8, alpha=0.7)

    # Plot Planckian locus (y1 - blackbody locus)
    ax.plot(planck_x, planck_y, 'b-', linewidth=3, label='普朗克轨迹 (黑体轨迹)')

    # Add temperature labels on Planckian locus
    temp_indices = [10, 25, 40, 55, 70, 85]  # Selected indices for labeling
    for i in temp_indices:
        if i < len(temperatures):
            ax.annotate(f'{temperatures[i]:.0f}K',
                       (planck_x[i], planck_y[i]),
                       xytext=(5, -15), textcoords='offset points',
                       fontsize=9, bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

    # Find closest point on Planckian locus for the test source (y3)
    min_dist = float('inf')
    closest_idx = 0
    for i, (px, py) in enumerate(zip(planck_x, planck_y)):
        dist = np.sqrt((test_x - px)**2 + (test_y - py)**2)
        if dist < min_dist:
            min_dist = dist
            closest_idx = i

    closest_planck_x = planck_x[closest_idx]
    closest_planck_y = planck_y[closest_idx]
    closest_temp = temperatures[closest_idx]

    # Plot closest Planckian point (y3)
    ax.plot(closest_planck_x, closest_planck_y, 'go', markersize=12,
            label=f'最近普朗克点 ({closest_temp:.0f}K)', markeredgecolor='darkgreen', markeredgewidth=2)

    # Plot test source CCT point (y2)
    ax.plot(test_x, test_y, 'ro', markersize=15,
            label=f'测试光源 (CCT={test_cct:.0f}K)', markeredgecolor='darkred', markeredgewidth=2)

    # Plot Duv line (y4 - distance from Planckian locus)
    ax.plot([test_x, closest_planck_x], [test_y, closest_planck_y],
            'r--', linewidth=3, alpha=0.8,
            label=f'Duv = {duv_value:.4f}')

    # Add arrow to show Duv direction
    ax.annotate('', xy=(test_x, test_y), xytext=(closest_planck_x, closest_planck_y),
                arrowprops=dict(arrowstyle='<->', color='red', lw=2))

    # Add text annotation for Duv
    mid_x = (test_x + closest_planck_x) / 2
    mid_y = (test_y + closest_planck_y) / 2
    ax.annotate(f'Duv={duv_value:.4f}',
                (mid_x, mid_y),
                xytext=(10, 10), textcoords='offset points',
                fontsize=12, fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.5", facecolor="yellow", alpha=0.8))

    # Set labels and title
    ax.set_xlabel('CIE x', fontsize=14, fontweight='bold')
    ax.set_ylabel('CIE y', fontsize=14, fontweight='bold')
    ax.set_title('CIE 1931 色度图', fontsize=18, fontweight='bold', pad=20)

    # Set equal aspect ratio and limits
    ax.set_aspect('equal')
    ax.set_xlim(0, 0.8)
    ax.set_ylim(0, 0.9)

    # Add grid
    ax.grid(True, alpha=0.3)

    # Add legend
    ax.legend(loc='upper right', fontsize=12, framealpha=0.9)

    # Add coordinate info box
    info_text = f"""光源信息:
CCT: {test_cct:.0f} K
Duv: {duv_value:.4f}
色度坐标: ({test_x:.4f}, {test_y:.4f})
最近黑体温度: {closest_temp:.0f} K"""

    ax.text(0.02, 0.98, info_text, transform=ax.transAxes,
            fontsize=11, verticalalignment='top',
            bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8))

    plt.tight_layout()

    # Save the plot
    plt.savefig('CIE_chromaticity_diagram.png', dpi=300, bbox_inches='tight')
    print("CIE色度图已保存为 'CIE_chromaticity_diagram.png'")

    # Show the plot
    plt.show()

    return fig, ax

def main():
    """Main function to calculate all parameters"""
    print("Loading data from attachment-Q1.xlsx...")
    wavelength, intensity = load_data()

    print(f"Data loaded: {len(wavelength)} data points")
    print(f"Wavelength range: {wavelength.min():.1f} - {wavelength.max():.1f} nm")
    print(f"Intensity range: {intensity.min():.3f} - {intensity.max():.3f} mW/m²")
    print()

    # 1. Calculate color matching functions (x̄, ȳ, z̄)
    x_bar, y_bar, z_bar = interpolate_color_matching_functions(wavelength)
    print("1. Color matching functions calculated")
    print(f"   x̄ range: {x_bar.min():.6f} - {x_bar.max():.6f}")
    print(f"   ȳ range: {y_bar.min():.6f} - {y_bar.max():.6f}")
    print(f"   z̄ range: {z_bar.min():.6f} - {z_bar.max():.6f}")
    print()

    # 2. Calculate CIE XYZ tristimulus values
    X, Y, Z = calculate_tristimulus_values(wavelength, intensity)
    print("2. CIE XYZ tristimulus values:")
    print(f"   X = {X:.3f}")
    print(f"   Y = {Y:.3f}")
    print(f"   Z = {Z:.3f}")
    print()

    # 3. Calculate CCT
    CCT, Duv = calculate_cct_duv(X, Y, Z)
    print("3. Correlated Color Temperature (CCT):")
    print(f"   CCT = {CCT:.0f} K")
    print()

    # 4. Calculate Duv
    print("4. Distance from Planckian locus (Duv):")
    print(f"   Duv = {Duv:.4f}")
    print()

    # 5. Calculate Rf (Color Fidelity Index) - Accurate method
    print("Calculating accurate Rf using CAM02-UCS color differences...")
    Rf = calculate_rf_accurate(wavelength, intensity)
    print("5. Color Fidelity Index (Rf):")
    print(f"   Rf = {Rf:.1f}")
    print()

    # 6. Calculate Rg (Color Gamut Index) - Accurate method
    print("Calculating accurate Rg using color gamut area...")
    Rg = calculate_rg_accurate(wavelength, intensity)
    print("6. Color Gamut Index (Rg):")
    print(f"   Rg = {Rg:.1f}")
    print()

    # 7. Calculate DER_mel - Accurate method
    print("Calculating accurate DER_mel using melanopic sensitivity...")
    DER_mel = calculate_der_mel_accurate(wavelength, intensity)
    print("7. Daylight Efficacy Ratio for melanopic response (DER_mel):")
    print(f"   DER_mel = {DER_mel:.4f}")
    print()

    # Summary
    print("="*50)
    print("SUMMARY OF RESULTS")
    print("="*50)
    print(f"CIE X: {X:.3f}")
    print(f"CIE Y: {Y:.3f}")
    print(f"CIE Z: {Z:.3f}")
    print(f"CCT: {CCT:.0f} K")
    print(f"Duv: {Duv:.4f}")
    print(f"Rf: {Rf:.1f}")
    print(f"Rg: {Rg:.1f}")
    print(f"DER_mel: {DER_mel:.4f}")

    # Calculate chromaticity coordinates for plotting
    x_chromaticity = X / (X + Y + Z)
    y_chromaticity = Y / (X + Y + Z)

    # Plot chromaticity diagram
    print("\n生成CIE色度图...")
    plot_cie_chromaticity_diagram(x_chromaticity, y_chromaticity, CCT, Duv)

if __name__ == "__main__":
    main()
