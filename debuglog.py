import pandas as pd

df = pd.read_csv('log.csv')

for switch_type in ["Drive→Recovery", "Recovery→Drive"]:
    sub = df[df['Switch'] == switch_type]
    print(f"\n=== {switch_type} ===")
    for col in ['leg_drive_angle', 'back_angle', 'arm_angle']:
        vals = sub[col].dropna().astype(float)
        if len(vals) > 0:
            print(f"{col}: min={vals.min():.2f}, max={vals.max():.2f}")
        else:
            print(f"{col}: 无数据")