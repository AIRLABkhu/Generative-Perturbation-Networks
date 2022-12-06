import matplotlib.pyplot as plt
import numpy as np

method = ['vGPN-SA', 'vGPN-SS', 'cGPN-SS', 'mGPN-SS']
vGPN_SA = [74.21, 96.17, 72.94, 95.12]
vGPN_SS = [78.44, 95.57, 67.90, 90.44]
cGPN_SS = [79.82, 94.56, 70.15, 87.78]
mGPN_SS = [76.85, 94.93, 67.45, 92.12]

metric = ['FR(non-targeted)','Acc(targeted)','FR(non-targeted)','Acc(targeted)']
# 그림 사이즈, 바 굵기 조정
fig, ax = plt.subplots(figsize=(12,6))
bar_width = 0.2

# 연도가 4개이므로 0, 1, 2, 3 위치를 기준으로 삼음
index = np.arange(4)

# 각 연도별로 3개 샵의 bar를 순서대로 나타내는 과정, 각 그래프는 0.25의 간격을 두고 그려짐
b0 = plt.bar(index, vGPN_SA, bar_width, alpha=0.4, color='red', label='FR_within')

b1 = plt.bar(index + bar_width, vGPN_SS, bar_width, alpha=0.4, color='blue', label='Acc_within')

b2 = plt.bar(index + 2 * bar_width, cGPN_SS, bar_width, alpha=0.4, color='green', label='FR_crosss')

b3 = plt.bar(index + 3 * bar_width, mGPN_SS, bar_width, alpha=0.4, color='orange', label='Acc_cross')

# x축 위치를 정 가운데로 조정하고 x축의 텍스트를 year 정보와 매칭
plt.xticks(np.arange(0.33, 4 + bar_width, 1), metric, fontsize=19)

# x축, y축 이름 및 범례 설정
plt.xlabel("           Within-subject                Cross-subject     ", size = 22)
# plt.ylabel('revenue', size = 13)
plt.legend(method, loc='lower right', handlelength=2.2, handleheight=0.8, fontsize=20)
plt.show()