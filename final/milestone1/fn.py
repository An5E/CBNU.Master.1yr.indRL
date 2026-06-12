
from pvlib import irradiance
from pvlib import solarposition
import pandas as pd
import numpy as np

from glb import startHr, endHr

IS_DEBUG = False

def debug_print(s:str):
    if IS_DEBUG:
        print(s)

# ! 태양 방위, 고도각 출력 함수 ( Fig4. )
def getHourlySolarPos():
    lat, lon = 35.08, -106.65 # ! 앨버커키 위도, 경도
    tz = 'MST' # Mountain Standard Time
    times = pd.date_range('2012-08-02 00:00:00', '2012-08-02 23:00:00', freq='1h', tz=tz)

    # 태양 위치 계산
    solpos = solarposition.get_solarposition(times, lat, lon)
    
    solpos['hr'] = solpos.index.strftime("%H")
    solpos.reset_index(drop=True, inplace=True)
    solpos.index = solpos['hr'].astype(int)
    
    surface_tilt = 30
    surface_azimuth = 180 # ! 남향
    aoi = irradiance.aoi(surface_tilt, surface_azimuth, solpos['apparent_zenith'], solpos['azimuth'])

    power_curve = np.cos(np.radians(aoi))
    power_curve[solpos['elevation'] < 0] = 0
    
    return solpos[['azimuth', 'zenith']]

def getMPAHourly(src: pd.DataFrame, startHour=6, endHour=18, max_power=220):
    ret = []
    
    src.reset_index(drop=True, inplace=True)
    
    data = {
        'hour': np.arange(startHour,  endHour+1), # 6시 ~ 18시 낮 시간대
        'azimuth': src.azimuth[startHour:endHour+1].values,
        'zenith': src.zenith[startHour:endHour+1].values
    }
    
    df = pd.DataFrame(data)
    tilt_range = np.linspace(0, 30, 31)  # X축: 패널 기울기 (0~30도, 1도 단위)
    panel_azimuth = 180  # 패널 설치 방향 (정남향)

    for i, row in df.iterrows():
        h = row['hour']
        s_azi = row['azimuth']
        s_zen = row['zenith']
        s_alt = 90 - s_zen  # 천정각을 고도각으로 변환
        
        powers = []
        for tilt in tilt_range:
            # 라디안 변환
            s_alt_r, s_azi_r = np.radians(s_alt), np.radians(s_azi)
            t_r, p_azi_r = np.radians(tilt), np.radians(panel_azimuth)
            
            # 입사각(AOI) 코사인 계산
            cos_theta = (np.sin(s_alt_r) * np.cos(t_r) + 
                        np.cos(s_alt_r) * np.sin(t_r) * np.cos(s_azi_r - p_azi_r))
            
            power = max_power * np.clip(cos_theta, 0, 1)
            powers.append(power)
        
        # 최대 발전 각도(MPA) 추출
        max_idx = np.argmax(powers)
        mpa_tilt = tilt_range[max_idx]
        max_p = powers[max_idx]
        
        ret.append((h, mpa_tilt, max_p, tilt_range, powers)) # ! 시간, MPA 경사각, 최대 발전량, 경사각 범위, 경사각별 발전량

    return ret

solpos = getHourlySolarPos()
l_mpa = getMPAHourly(solpos[['azimuth','zenith']], startHr, endHr, 220)

pd.DataFrame(l_mpa).to_csv(".\l_mpa2.csv")

# ! Hour input range: 6 ~ 18
def getRewardFromMPA(hour: int, tilt_angle: float, startHour=6, endHour=18):
    # print(f"  {hour}, {tilt_angle} => {int(tilt_angle)}")
    # print(l_mpa)
    # print(f"  grfmpa({hour},{tilt_angle}):: {l_mpa[hour-6][4][int(tilt_angle)]}")
    
    return l_mpa[hour-6][4][int(tilt_angle)] # if hour >= startHour and hour <= (endHour-1) else 0

def getSolarPower(hour, tilt_angle):        
        # ? {hour} 곡선에서 x={tilt_angle}인 y값 구하기. l_mpa에서 참조
        return max(0, getRewardFromMPA(hour, tilt_angle, startHour=startHr, endHour=endHr))