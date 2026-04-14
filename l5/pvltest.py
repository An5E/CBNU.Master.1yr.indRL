from pvlib import solarposition, irradiance
import pandas as pd
import matplotlib.pyplot as plt
import mpatest
import numpy as np
import irrtest

# 알버커키 위치 정보 및 시간 설정
lat, lon = 35.08, -106.65
tz = 'MST' # Mountain Standard Time
times = pd.date_range('2012-08-02 00:00:00', '2012-08-02 23:00:00', freq='1h', tz=tz)

# 태양 위치 계산
solpos = solarposition.get_solarposition(times, lat, lon)

solpos['hr'] = solpos.index.strftime("%H")
solpos.reset_index(drop=True, inplace=True)
solpos.index = solpos['hr'].astype(int)
# 결과 출력 (Apparent Elevation: 고도각, Azimuth: 방위각)
# print(solpos[['apparent_elevation', 'azimuth', 'hr']])
# solpos.rename(columns={'apparent_elevation': 'zenith'}, inplace=True)


# ! 
surface_tilt = 30
surface_azimuth = 180 # ! 남향
aoi = irradiance.aoi(surface_tilt, surface_azimuth, solpos['apparent_zenith'], solpos['azimuth'])

power_curve = np.cos(np.radians(aoi))
power_curve[solpos['elevation'] < 0] = 0



plt.figure(figsize=(16, 5))
plt.subplot(1, 3, 1)
plt.plot(solpos['azimuth'], marker='s', markersize=3, color='r', label="azimuth angle")
plt.plot(solpos['zenith'], marker='s', markersize=3, color='b', label="zenith angle")
plt.legend()
plt.ylabel('Angle (deg)')
plt.xlabel('Hour of Day (h)')
plt.title('Fig 4`. 2012-08-02 Solar position in Albuquerque')

# print(irrtest.calculate_mpa(solpos[['azimuth','zenith']], 4000))

plt.subplot(1, 3, 2)

l_mpa = irrtest.calculate_mpa(solpos[['azimuth','zenith']], 3500)

for h, mpa_tilt, max_p, tilt_range, powers in l_mpa:
    plt.plot(tilt_range, powers, label=f'{int(h):02d}:00 (MPA:{mpa_tilt:.1f}°)')
    plt.scatter(mpa_tilt, max_p, color='black', s=40, edgecolor='black', zorder=5)

# plt.plot(irrtest.calculate_mpa(solpos[['azimuth','zenith']]), marker='s', markersize=3, color='g', label="MPA curve")
# plt.plot(power_curve, marker='s', markersize=3, color='g', label="Power curve (cosine of AOI)")
plt.legend()
plt.ylabel('Power (W)')
plt.xlabel('Hour of Day (h)')
plt.title('Fig 5`. MPA Curves for Each Hour')


plt.subplot(1, 3, 3)
plt.title('Fig 6`. Q-learned Tracking vs Theoretical MPA')
plt.plot([h for h, _, _, _, _ in l_mpa], [mpa_tilt for _, mpa_tilt, _, _, _ in l_mpa], marker='s', markersize=3, linestyle='--', color='orange', label='Theoretical MPA')
plt.legend()
plt.xlabel('Hour of Day (h)')
plt.ylabel('Tilt Angle (deg)')
# plt.plot(mpatest.calculate_mpa(solpos['zenith'], solpos['azimuth']))
# # print(mpatest.calculate_mpa(solpos['zenith'], solpos['azimuth']))
# plt.title("Figure 5")
# plt.ylabel("Power (W)")
# plt.xlabel("Tilt Angle (deg)")

plt.tight_layout()
plt.show()