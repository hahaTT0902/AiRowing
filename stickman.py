import csv
import numpy as np
import matplotlib.pyplot as plt

joint_indices = [11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]
bones = [
    (11, 13), (13, 15),  # 左臂
    (12, 14), (14, 16),  # 右臂
    (11, 12),            # 双肩
    (23, 24),            # 骨盆
    (11, 23), (12, 24),  # 躯干
    (23, 25), (25, 27),  # 左腿
    (24, 26), (26, 28),  # 右腿
]
bone_names = [f"{a}_{b}" for a, b in bones]

frames = []
with open('log.csv') as f:
    reader = csv.DictReader(f)
    for row in reader:
        try:
            joint_pos = {}
            # 骨盆中心为原点
            joint_pos[23] = np.array([0.0, 0.0])
            try:
                joint_pos[24] = joint_pos[23] + np.array([
                    float(row.get("vec_23_24_dx", 0)),
                    float(row.get("vec_23_24_dy", 0))
                ])
                # 躯干
                joint_pos[11] = joint_pos[23] + np.array([
                    float(row.get("vec_11_23_dx", 0)),
                    float(row.get("vec_11_23_dy", 0))
                ])
                joint_pos[12] = joint_pos[24] + np.array([
                    float(row.get("vec_12_24_dx", 0)),
                    float(row.get("vec_12_24_dy", 0))
                ])
                # 左臂
                joint_pos[13] = joint_pos[11] - np.array([
                    float(row.get("vec_11_13_dx", 0)),
                    float(row.get("vec_11_13_dy", 0))
                ])
                joint_pos[15] = joint_pos[13] - np.array([
                    float(row.get("vec_13_15_dx", 0)),
                    float(row.get("vec_13_15_dy", 0))
                ])
                # 右臂
                joint_pos[14] = joint_pos[12] - np.array([
                    float(row.get("vec_12_14_dx", 0)),
                    float(row.get("vec_12_14_dy", 0))
                ])
                joint_pos[16] = joint_pos[14] - np.array([
                    float(row.get("vec_14_16_dx", 0)),
                    float(row.get("vec_14_16_dy", 0))
                ])
                # 左腿
                joint_pos[25] = joint_pos[23] - np.array([
                    float(row.get("vec_23_25_dx", 0)),
                    float(row.get("vec_23_25_dy", 0))
                ])
                joint_pos[27] = joint_pos[25] - np.array([
                    float(row.get("vec_25_27_dx", 0)),
                    float(row.get("vec_25_27_dy", 0))
                ])
                # 右腿
                joint_pos[26] = joint_pos[24] - np.array([
                    float(row.get("vec_24_26_dx", 0)),
                    float(row.get("vec_24_26_dy", 0))
                ])
                joint_pos[28] = joint_pos[26] - np.array([
                    float(row.get("vec_26_28_dx", 0)),
                    float(row.get("vec_26_28_dy", 0))
                ])
                xs = [joint_pos[idx][0] for idx in joint_indices]
                ys = [joint_pos[idx][1] for idx in joint_indices]
                frames.append((xs, ys))
            except Exception as e:
                print(f"[Warn] 跳过一帧，数据异常: {e}")
                continue
        except Exception:
            continue

plt.ion()
fig, ax = plt.subplots(figsize=(5, 7))

bone_idx = [(joint_indices.index(a), joint_indices.index(b)) for a, b in bones]

def normalize(xs, ys):
    xs = np.array(xs)
    ys = np.array(ys)
    # 以骨盆中心为原点，保持纵向比例
    center_x = (xs[joint_indices.index(23)] + xs[joint_indices.index(24)]) / 2
    center_y = (ys[joint_indices.index(23)] + ys[joint_indices.index(24)]) / 2
    xs = xs - center_x
    ys = ys - center_y
    scale = max(np.max(np.abs(xs)), np.max(np.abs(ys)), 1e-6)
    xs = xs / scale
    ys = ys / scale
    return xs, ys

for xs, ys in frames:
    ax.cla()
    xs_n, ys_n = normalize(xs, ys)
    for a, b in bone_idx:
        ax.plot([xs_n[a], xs_n[b]], [ys_n[a], ys_n[b]], 'k-', lw=3)
    ax.scatter(xs_n, ys_n, c='r')
    # 假头
    left_shoulder = np.array([xs_n[joint_indices.index(11)], ys_n[joint_indices.index(11)]])
    right_shoulder = np.array([xs_n[joint_indices.index(12)], ys_n[joint_indices.index(12)]])
    neck = (left_shoulder + right_shoulder) / 2
    head_radius = 0.18
    head_center = neck + np.array([0, head_radius * 1.5])
    head = plt.Circle(head_center, head_radius, color='orange', fill=False, lw=3)
    ax.add_patch(head)
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1.5, 1.5)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title('2D Stickman (from relative vectors)')
    plt.pause(0.03)

plt.ioff()
plt.show()