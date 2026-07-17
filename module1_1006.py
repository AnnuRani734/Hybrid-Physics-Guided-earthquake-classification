import requests
import numpy as np
import pandas as pd
from datetime import datetime

#
# Himalayan Region


START = "1970-01-01"
END   = "2025-12-31"

LAT_MIN, LAT_MAX = 25, 37
LON_MIN, LON_MAX = 72, 92



BETA1_BASE = 320.0      # m/s
BETA2_BASE = 3200.0     # m/s

RHO1 = 2200.0
RHO2 = 2700.0

MAG_REFERENCE = 4.0


# LOVE WAVE DISPERSION


def solve_dispersion(freq, H, beta1, beta2, mu1, mu2, B):

    omega = 2 * np.pi * freq

    #  wavenumbers
    def q1(c):
        val = (1.0 / beta1**2) - (1.0 / c**2)
        return omega * np.sqrt(max(val, 0.0))

    def q2(c):
        val = (1.0 / c**2) - (1.0 / beta2**2)
        return omega * np.sqrt(max(val, 0.0))

    #  Eq. (31)
    def dispersion(c):

        if c <= beta1 or c >= beta2:
            return np.nan

        try:
            q1v = q1(c)
            q2v = q2(c)

            if q1v == 0 or q2v == 0:
                return np.nan

            lhs = np.tan(q1v * H)

            rhs = (mu2 * q2v) / (mu1 * q1v * (1.0 - B * mu2 * q2v))

            return lhs - rhs

        except:
            return np.nan


    #


    c_grid = np.linspace(beta1 + 1.0, beta2 - 1.0, 400)

    f_vals = np.array([dispersion(c) for c in c_grid])

    for i in range(len(c_grid) - 1):

        if np.isnan(f_vals[i]) or np.isnan(f_vals[i+1]):
            continue

        if f_vals[i] * f_vals[i+1] < 0:

            return 0.5 * (c_grid[i] + c_grid[i+1])

    return np.nan



# GROUP VELOCITY


def group_velocity(cp_low, cp_high, f_low, f_high):

    try:
        w1 = 2 * np.pi * f_low
        w2 = 2 * np.pi * f_high

        k1 = w1 / cp_low
        k2 = w2 / cp_high

        if (k2 - k1) == 0:
            return np.nan

        return (w2 - w1) / (k2 - k1)

    except:
        return np.nan



# USGS DATA FETCH


def fetch_usgs():

    url = "https://earthquake.usgs.gov/fdsnws/event/1/query"

    params = {
        "format": "geojson",
        "starttime": START,
        "endtime": END,
        "minlatitude": LAT_MIN,
        "maxlatitude": LAT_MAX,
        "minlongitude": LON_MIN,
        "maxlongitude": LON_MAX,
        "minmagnitude": 2.5,
        "limit": 20000,
        "orderby": "time-asc"
    }

    r = requests.get(url, params=params)

    if r.status_code != 200:
        return pd.DataFrame()

    data = r.json()

    rows = []

    for eq in data["features"]:

        try:
            lon, lat, depth = eq["geometry"]["coordinates"]
            mag = eq["properties"]["mag"]

            t = datetime.utcfromtimestamp(
                eq["properties"]["time"] / 1000.0
            )

            rows.append([lat, lon, depth, mag, t])

        except:
            continue

    return pd.DataFrame(rows,
        columns=["latitude","longitude","depth_km","magnitude","time"])



# PHYSICS


def compute_physics_features(df):

    print("[Physics] Computing features...")

    out = []

    for _, row in df.iterrows():

        depth = row["depth_km"]
        mag = row["magnitude"]


        # Layer thickness

        H = max(500.0, min(depth * 1000.0, 30000.0))


        # Interface compliance (B)

        B = 1e-9 * (1.0 + 0.02 * np.log1p(depth + 1.0))


        # Shear velocities

        beta1 = BETA1_BASE * (1.0 + 0.002 * depth)
        beta2 = BETA2_BASE * (1.0 + 0.0005 * depth)

        mu1 = RHO1 * beta1**2
        mu2 = RHO2 * beta2**2


        # Phase velocities

        cp1 = solve_dispersion(1.0, H, beta1, beta2, mu1, mu2, B)
        cp2 = solve_dispersion(2.0, H, beta1, beta2, mu1, mu2, B)
        cp5 = solve_dispersion(5.0, H, beta1, beta2, mu1, mu2, B)

        if np.isnan(cp1):
            continue


        # Group velocities

        cg12 = group_velocity(cp1, cp2, 1.0, 2.0)
        cg25 = group_velocity(cp2, cp5, 2.0, 5.0)

        cg_mean = np.nanmean([cg12, cg25])


        # Interface perturbation

        B2 = 1.10 * B

        cp1_B = solve_dispersion(1.0, H, beta1, beta2, mu1, mu2, B2)
        cp2_B = solve_dispersion(2.0, H, beta1, beta2, mu1, mu2, B2)
        cp5_B = solve_dispersion(5.0, H, beta1, beta2, mu1, mu2, B2)

        delta_cp1 = abs(cp1 - cp1_B) if not np.isnan(cp1_B) else np.nan
        delta_cp2 = abs(cp2 - cp2_B) if not np.isnan(cp2_B) else np.nan
        delta_cp5 = abs(cp5 - cp5_B) if not np.isnan(cp5_B) else np.nan

        idi = np.nanmean([delta_cp1, delta_cp2, delta_cp5])


        # more PHYSICS FEATURES


        disp_slope_12 = cp2 - cp1

        disp_slope_25 = (cp5 - cp2) / 3.0

        disp_curvature = cp5 - 2.0 * cp2 + cp1

        dispersion_range = cp5 - cp1

        gp_ratio_12 = (
            cg12 / cp1
            if (not np.isnan(cg12) and cp1 > 0)
            else np.nan
        )

        gp_ratio_25 = (
            cg25 / cp5
            if (not np.isnan(cg25) and cp5 > 0)
            else np.nan
        )

        idi_norm = (
            idi / cg_mean
            if (
                    not np.isnan(cg_mean)
                    and cg_mean > 0
            )
            else np.nan
        )

        freq_sensitivity = (
                abs(cp5 - cp1)
                /
                (abs(cp1) + 1e-6)
        )
        out.append({

            "latitude": row["latitude"],
            "longitude": row["longitude"],
            "depth_km": depth,
            "magnitude": mag,
            "time": row["time"],

            "H_m": H,
            "B": B,

            "c_phase_1Hz": cp1,
            "c_phase_2Hz": cp2,
            "c_phase_5Hz": cp5,

            "c_group_12": cg12,
            "c_group_25": cg25,
            "c_group_mean": cg_mean,

            "delta_cpB_1Hz": delta_cp1,
            "delta_cpB_2Hz": delta_cp2,
            "delta_cpB_5Hz": delta_cp5,

            "interface_damage_index": idi,

            # NEW FEATURES

            "disp_slope_12": disp_slope_12,
            "disp_slope_25": disp_slope_25,

            "disp_curvature": disp_curvature,

            "dispersion_range": dispersion_range,

            "gp_ratio_12": gp_ratio_12,
            "gp_ratio_25": gp_ratio_25,

            "idi_norm": idi_norm,

            "freq_sensitivity": freq_sensitivity
        })
    return pd.DataFrame(out)



# MAIN


def main():

    df = fetch_usgs()

    if df.empty:
        print("No data retrieved.")
        return

    df_physics = compute_physics_features(df)

    print("[INFO] Generated:", len(df_physics))

    df_physics.to_csv("physics_dataset.csv", index=False)

    print("[SUCCESS] Saved physics_dataset.csv")


    # FIGURE : Phase Velocity vs Interface Compliance


    def plot_phase_velocity_vs_B():

        import numpy as np
        import matplotlib.pyplot as plt

        # Representative physical parameters
        depth = 15.0  # km

        H = depth * 1000.0

        beta1 = BETA1_BASE * (1.0 + 0.002 * depth)
        beta2 = BETA2_BASE * (1.0 + 0.0005 * depth)

        mu1 = RHO1 * beta1 ** 2
        mu2 = RHO2 * beta2 ** 2

        B_values = np.linspace(0.0, 2.0e-9, 40)

        cp1_list = []
        cp2_list = []
        cp5_list = []

        for B in B_values:
            cp1 = solve_dispersion(
                1.0, H,
                beta1, beta2,
                mu1, mu2, B
            )

            cp2 = solve_dispersion(
                2.0, H,
                beta1, beta2,
                mu1, mu2, B
            )

            cp5 = solve_dispersion(
                5.0, H,
                beta1, beta2,
                mu1, mu2, B
            )

            cp1_list.append(cp1)
            cp2_list.append(cp2)
            cp5_list.append(cp5)

        plt.figure(figsize=(8, 5))

        plt.plot(B_values, cp1_list,
                 marker='o', label='1 Hz')

        plt.plot(B_values, cp2_list,
                 marker='s', label='2 Hz')

        plt.plot(B_values, cp5_list,
                 marker='^', label='5 Hz')

        plt.xlabel('Interface Compliance B (m/Pa)')
        plt.ylabel('Phase Velocity (m/s)')

        plt.title(
            'Effect of Interface Compliance on Love-Wave Phase Velocity'
        )

        plt.grid(True)
        plt.legend()

        plt.tight_layout()

        plt.savefig(
            'Figure_PhaseVelocity_vs_B.png',
            dpi=600,
            bbox_inches='tight'
        )

        plt.show()


        # FIGURE


        def plot_group_velocity_vs_B():

            import numpy as np
            import matplotlib.pyplot as plt

            depth = 15.0

            H = depth * 1000.0

            beta1 = BETA1_BASE * (1.0 + 0.002 * depth)
            beta2 = BETA2_BASE * (1.0 + 0.0005 * depth)

            mu1 = RHO1 * beta1 ** 2
            mu2 = RHO2 * beta2 ** 2

            B_values = np.linspace(0.0, 2.0e-9, 40)

            cg12_list = []
            cg25_list = []

            for B in B_values:
                cp1 = solve_dispersion(
                    1.0, H,
                    beta1, beta2,
                    mu1, mu2, B
                )

                cp2 = solve_dispersion(
                    2.0, H,
                    beta1, beta2,
                    mu1, mu2, B
                )

                cp5 = solve_dispersion(
                    5.0, H,
                    beta1, beta2,
                    mu1, mu2, B
                )

                cg12 = group_velocity(
                    cp1, cp2,
                    1.0, 2.0
                )

                cg25 = group_velocity(
                    cp2, cp5,
                    2.0, 5.0
                )

                cg12_list.append(cg12)
                cg25_list.append(cg25)

            plt.figure(figsize=(8, 5))

            plt.plot(
                B_values,
                cg12_list,
                marker='o',
                label='1–2 Hz'
            )

            plt.plot(
                B_values,
                cg25_list,
                marker='s',
                label='2–5 Hz'
            )

            plt.xlabel('Interface Compliance B (m/Pa)')
            plt.ylabel('Group Velocity (m/s)')

            plt.title(
                'Effect of Interface Compliance on Love-Wave Group Velocity'
            )

            plt.grid(True)
            plt.legend()

            plt.tight_layout()

            plt.savefig(
                'Figure_GroupVelocity_vs_B.png',
                dpi=600,
                bbox_inches='tight'
            )
            plt.savefig("Figure_PhaseVelocity_vs_B.png", dpi=600)
            print("Saved Figure_PhaseVelocity_vs_B.png")
            plt.savefig("Figure_GroupVelocity_vs_B.png", dpi=600)
            print("Saved Figure_GroupVelocity_vs_B.png")
if __name__ == "__main__":
    main()
    