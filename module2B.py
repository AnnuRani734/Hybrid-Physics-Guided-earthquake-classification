import pandas as pd
import numpy as np
from tqdm import tqdm



INPUT_FILE = "physics_dataset.csv"
OUTPUT_FILE = "full_dataset_v2.csv"

SEQ_N = 6



def build_temporal_features(df):

    print("[Module 2B] Processing data...")

    df["time"] = pd.to_datetime(df["time"])



    df["event_id"] = (

        df["latitude"].round(4).astype(str)
        + "_"
        + df["longitude"].round(4).astype(str)
        + "_"
        + df["depth_km"].round(2).astype(str)
        + "_"
        + df["magnitude"].round(2).astype(str)
        + "_"
        + df["time"].astype(str)

    )


    df = df.sort_values("time").reset_index(drop=True)

    N = len(df)

    mags = df["magnitude"].values
    times = df["time"].values



    seq_mag = np.zeros((N, SEQ_N))
    seq_gap = np.zeros((N, SEQ_N))



    rolling_mean_5 = np.zeros(N)
    rolling_mean_10 = np.zeros(N)

    rolling_max_5 = np.zeros(N)
    rolling_max_10 = np.zeros(N)

    rolling_std_10 = np.zeros(N)

    event_count_30d = np.zeros(N)
    event_count_90d = np.zeros(N)
    event_count_365d = np.zeros(N)

    moment_sum_30d = np.zeros(N)
    moment_sum_90d = np.zeros(N)

    energy_sum_30d = np.zeros(N)
    energy_sum_90d = np.zeros(N)

    print(f"[Module 2B] Creating temporal features...")

    for i in tqdm(range(N)):



        for j in range(SEQ_N):

            idx = i - (j + 1)

            if idx >= 0:

                seq_mag[i, j] = mags[idx]

                seq_gap[i, j] = (
                    pd.Timestamp(times[i])
                    - pd.Timestamp(times[idx])
                ).days



        start5 = max(0, i - 5)
        start10 = max(0, i - 10)

        mags5 = mags[start5:i]
        mags10 = mags[start10:i]

        if len(mags5) > 0:

            rolling_mean_5[i] = np.mean(mags5)
            rolling_max_5[i] = np.max(mags5)

        if len(mags10) > 0:

            rolling_mean_10[i] = np.mean(mags10)
            rolling_max_10[i] = np.max(mags10)
            rolling_std_10[i] = np.std(mags10)


        current_time = pd.Timestamp(times[i])

        count30 = 0
        count90 = 0
        count365 = 0

        moment30 = 0.0
        moment90 = 0.0

        energy30 = 0.0
        energy90 = 0.0

        k = i - 1

        while k >= 0:

            dt_days = (
                current_time
                - pd.Timestamp(times[k])
            ).days

            if dt_days > 365:
                break

            count365 += 1

            M = mags[k]

            if dt_days <= 90:

                count90 += 1

                moment90 += 10 ** (1.5 * M + 9.1)
                energy90 += 10 ** (1.5 * M)

            if dt_days <= 30:

                count30 += 1

                moment30 += 10 ** (1.5 * M + 9.1)
                energy30 += 10 ** (1.5 * M)

            k -= 1

        event_count_30d[i] = count30
        event_count_90d[i] = count90
        event_count_365d[i] = count365

        moment_sum_30d[i] = np.log10(moment30 + 1.0)
        moment_sum_90d[i] = np.log10(moment90 + 1.0)

        energy_sum_30d[i] = np.log10(energy30 + 1.0)
        energy_sum_90d[i] = np.log10(energy90 + 1.0)


    mag_df = pd.DataFrame({

        f"mag_t-{j+1}":
            seq_mag[:, j]

        for j in range(SEQ_N)

    })

    gap_df = pd.DataFrame({

        f"dt_t-{j+1}":
            seq_gap[:, j]

        for j in range(SEQ_N)

    })


    advanced_df = pd.DataFrame({

        "rolling_mean_mag_5":
            rolling_mean_5,

        "rolling_mean_mag_10":
            rolling_mean_10,

        "rolling_max_mag_5":
            rolling_max_5,

        "rolling_max_mag_10":
            rolling_max_10,

        "rolling_std_mag_10":
            rolling_std_10,

        "event_count_30d":
            event_count_30d,

        "event_count_90d":
            event_count_90d,

        "event_count_365d":
            event_count_365d,

        "moment_sum_30d":
            moment_sum_30d,

        "moment_sum_90d":
            moment_sum_90d,

        "energy_sum_30d":
            energy_sum_30d,

        "energy_sum_90d":
            energy_sum_90d
    })



    df_features = df.drop(columns=["time"])

    df_final = pd.concat(

        [
            df_features,
            mag_df,
            gap_df,
            advanced_df
        ],

        axis=1

    )



    df_final = df_final.replace(
        [np.inf, -np.inf],
        np.nan
    )

    df_final = df_final.fillna(0)

    print("\nDataset Shape:", df_final.shape)

    df_final.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print(
        f"\n[SUCCESS] Saved: {OUTPUT_FILE}"
    )

    return df_final


if __name__ == "__main__":

    df = pd.read_csv(INPUT_FILE)

    build_temporal_features(df)