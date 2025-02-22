import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

center_line_df = pd.read_csv("../f1tenth_racetracks/Austin/Austin_centerline.csv")
center_line_df.rename(
    columns={
        old_name: old_name.replace("#", "").lstrip()
        for old_name in center_line_df.columns
    },
    inplace=True,
)

race_line_df = pd.read_csv(
    "../f1tenth_racetracks/Austin/Austin_raceline.csv",
    skiprows=2,
    sep=";",
)
race_line_df.rename(
    columns={
        old_name: old_name.replace("#", "").lstrip()
        for old_name in race_line_df.columns
    },
    inplace=True,
)

plt.plot(center_line_df["x_m"], center_line_df["y_m"], c="g", label="Center line")
# plt.plot(race_line_df["x_m"], race_line_df["y_m"], c="r", label="Race Line")


def get_hex_color(value):
    value = max(0, min(value, 1))
    red = int(255 * value)
    green = int(255 * (1 - value))
    return f"#{red:02X}{green:02X}00"


for i in range(0, len(race_line_df), 50):
    row = race_line_df.iloc[i]

    x = row["x_m"]
    y = row["y_m"]
    r = row["psi_rad"]
    v = row["vx_mps"]

    w = v * np.cos(r)
    h = v * np.sin(r)

    plt.arrow(x, y, w, h, head_width=2, head_length=2, color=get_hex_color(v / 8))

plt.legend()
plt.show()
