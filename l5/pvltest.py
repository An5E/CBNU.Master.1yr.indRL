from pvlib import solarposition
import pandas as pd
import matplotlib.pyplot as plt
import mpatest

# 알버커키 위치 정보 및 시간 설정
lat, lon = 35.08, -106.65
tz = 'MST' # Mountain Standard Time
times = pd.date_range('2012-08-02 00:00:00', '2012-08-02 23:00:00', freq='1H', tz=tz)

# 태양 위치 계산
solpos = solarposition.get_solarposition(times, lat, lon)

solpos['hr'] = solpos.index.strftime("%H")
solpos.reset_index(drop=True, inplace=True)
solpos.index = solpos['hr'].astype(int)
# 결과 출력 (Apparent Elevation: 고도각, Azimuth: 방위각)
# print(solpos[['apparent_elevation', 'azimuth', 'hr']])
# solpos.rename(columns={'apparent_elevation': 'zenith'}, inplace=True)

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(solpos['azimuth'], marker='s', markersize=3, color='r', label="azimuth angle")
plt.plot(solpos['zenith'], marker='s', markersize=3, color='b', label="zenith angle")
plt.legend()
plt.ylabel('Angle (deg)')
plt.xlabel('Hour of Day (h)')
plt.title('Fig 4. 2012-08-02 Solar position in Albuquerque')

plt.subplot(1, 2, 2)
plt.plot(mpatest.calculate_mpa(solpos['zenith'], solpos['azimuth']))

# print(mpatest.calculate_mpa(solpos['zenith'], solpos['azimuth']))

plt.title("Figure 5")
plt.ylabel("Power (W)")
plt.xlabel("Tilt Angle (deg)")

plt.show()