"""Load and clean the real Steel Industry Energy Consumption dataset (UCI #851)."""
import pandas as pd


def load_steel_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    df["date"] = pd.to_datetime(df["date"], format="%d/%m/%Y %H:%M")
    df = df.sort_values("date").reset_index(drop=True)
    return df
