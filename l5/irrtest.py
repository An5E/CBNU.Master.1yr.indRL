import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def calculate_mpa(src: pd.DataFrame, max_power):    
    ret = []
    src.reset_index(drop=True, inplace=True)
    
    data = {
        'hour': np.arange(6, 19), # 6시 ~ 18시 낮 시간대
        'azimuth': src.azimuth[6:19].values,
        'zenith': src.zenith[6:19].values
    }
    
    df = pd.DataFrame(data)

    # 2. 설정 값
    tilt_range = np.linspace(0, 90, 100)  # X축: 패널 기울기 (0~90도)
    panel_azimuth = 180  # 패널 설치 방향 (정남향)

    # 3. 시각화 설정
    # plt.figure(figsize=(12, 8))
    # colors = cm.plasma(np.linspace(0, 1, len(df))) # 시간 흐름에 따른 색상

    # 4. 시간별(Row별) 루프 수행
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
            # cos(AOI) = sin(alt)*cos(tilt) + cos(alt)*sin(tilt)*cos(s_azi - p_azi)
            cos_theta = (np.sin(s_alt_r) * np.cos(t_r) + 
                        np.cos(s_alt_r) * np.sin(t_r) * np.cos(s_azi_r - p_azi_r))
            
            power = max_power * np.clip(cos_theta, 0, 1)
            powers.append(power)
        
        # 최대 발전 각도(MPA) 추출
        max_idx = np.argmax(powers)
        mpa_tilt = tilt_range[max_idx]
        max_p = powers[max_idx]
        
        ret.append((h, mpa_tilt, max_p, tilt_range, powers))
        
        # 그래프 그리기 (선형 차트)
        # plt.plot(tilt_range, powers, label=f'{int(h):02d}h (MPA:{mpa_tilt:.1f}°)', 
        #         color=colors[i], linewidth=1.8, alpha=0.8)
        
        # MPA 지점에 포인트 표시
        # plt.scatter(mpa_tilt, max_p, color=colors[i], s=40, edgecolor='black', zorder=5)

    return ret

if __name__ == "__main__":
    max_power = 4000
    # 5. 차트 레이아웃 구성
    plt.title("Solar Power Output by Panel Tilt (Based on Azimuth/Zenith Data)", fontsize=15, pad=20)
    plt.xlabel("Panel Tilt Angle (degrees)", fontsize=12)
    plt.ylabel("Output Power (W)", fontsize=12)
    plt.xlim(0, 90)
    plt.ylim(0, max_power + 300)
    plt.grid(True, linestyle='--', alpha=0.5)

    # 범례를 우측 상단에 배치
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1), title="Time & MPA", fontsize='small')

    plt.tight_layout()
    plt.show()