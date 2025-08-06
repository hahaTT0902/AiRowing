import csv
import matplotlib.pyplot as plt

# 读取 log.csv
time = []
leg = []
back = []
arm = []
phase = []
spm = []
switch_time = []
switch_type = []
leg_angle = []
back_angle = []
arm_angle = []

with open('log.csv', 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        t = float(row['Time'])
        time.append(t)
        leg.append(float(row['Leg Movement']))
        back.append(float(row['Back Movement']))
        arm.append(float(row['Arm Movement']))
        phase.append(row['Phase'])
        spm.append(float(row['SPM']) if row['SPM'] else 0)
        # 只在有切换时记录角度切换点
        if row['Switch']:
            switch_time.append(t)
            switch_type.append(row['Switch'])
            leg_angle.append(float(row['leg_drive_angle']) if row['leg_drive_angle'] else None)
            back_angle.append(float(row['back_angle']) if row['back_angle'] else None)
            arm_angle.append(float(row['arm_angle']) if row['arm_angle'] else None)

# 可选：过滤掉前5秒（调试用，后续可注释）
filter_idx = [i for i, t in enumerate(time) if t >= 5]
time = [time[i] for i in filter_idx]
leg = [leg[i] for i in filter_idx]
back = [back[i] for i in filter_idx]
arm = [arm[i] for i in filter_idx]
phase = [phase[i] for i in filter_idx]
spm = [spm[i] for i in filter_idx]

# 同步过滤切换点
switch_idx = [i for i, t in enumerate(switch_time) if t >= 5]
switch_time = [switch_time[i] for i in switch_idx]
switch_type = [switch_type[i] for i in switch_idx]
leg_angle = [leg_angle[i] for i in switch_idx]
back_angle = [back_angle[i] for i in switch_idx]
arm_angle = [arm_angle[i] for i in switch_idx]

fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 14), sharex=True)

# 第一子图：动作幅度曲线
ax1.plot(time, leg, label='Leg', color='green')
ax1.plot(time, back, label='Back', color='blue')
ax1.plot(time, arm, label='Arm', color='magenta')
if time:
    last_phase = phase[0]
    start_idx = 0
    for i, p in enumerate(phase):
        if p != last_phase or i == len(phase) - 1:
            ax1.axvspan(time[start_idx], time[i], color='#ffe6cc' if last_phase == 'Drive' else '#e6f2ff', alpha=0.2)
            last_phase = p
            start_idx = i
ax1.set_ylabel('Movement (px)')
ax1.set_title('Rowing Movement Curve')
ax1.legend()

# 第二子图：切换瞬间角度
ax2.plot(switch_time, leg_angle, 'o-', label='Leg Drive Angle', color='green')
ax2.plot(switch_time, back_angle, 'o-', label='Back Angle', color='blue')
ax2.plot(switch_time, arm_angle, 'o-', label='Arm Angle', color='magenta')
for t, s in zip(switch_time, switch_type):
    if s:
        ax2.axvline(t, color='gray', linestyle='--', alpha=0.3)
        ax2.text(t, ax2.get_ylim()[1], s, rotation=90, va='top', fontsize=8, alpha=0.6)
ax2.set_ylabel('Angle (°)')
ax2.set_title('Angles at Phase Switch')
ax2.legend()

# 第三子图：实时桨频曲线
ax3.plot(time, spm, label='SPM', color='orange')
ax3.set_ylabel('SPM')
ax3.set_xlabel('Time (s)')
ax3.set_title('Real-Time Stroke Per Minute')
ax3.legend()

plt.tight_layout()
plt.show()