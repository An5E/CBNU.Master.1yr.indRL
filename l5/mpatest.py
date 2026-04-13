import numpy as np

def calculate_mpa(sun_altitude, sun_azimuth, panel_azimuth=0):
    """
    태양 위치 정보를 바탕으로 이론적 최적 경사각(MPA)을 계산
    sun_altitude: 태양 고도각 (degrees)
    sun_azimuth: 태양 방위각 (degrees, 남향 0도 기준)
    panel_azimuth: 패널이 고정된 방위각 (보통 0)
    """
    # Degree to Radian 변환
    alpha_s = np.radians(sun_altitude)
    gamma_s = np.radians(sun_azimuth)
    gamma_p = np.radians(panel_azimuth)
    
    # cos(theta)를 최대화하는 beta(경사각) 계산
    # tan(beta_opt) = cos(gamma_p - gamma_s) / tan(alpha_s)
    term = np.cos(gamma_p - gamma_s) / np.tan(alpha_s)
    beta_opt_rad = np.arctan(term)
    
    return np.degrees(beta_opt_rad)

# Figure 4의 샘플 데이터 (Albuquerque, 8월 2일 12시경 가정)
sample_sun_alt = 70  # 고도각 약 70도
sample_sun_azi = 10  # 방위각 약 10도

mpa = calculate_mpa(sample_sun_alt, sample_sun_azi)
print(f"계산된 이론적 최적 경사각(MPA): {mpa:.2f}도")
