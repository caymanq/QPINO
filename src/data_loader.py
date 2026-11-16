"""Download and load the JHU CSSE COVID-19 time series (archived 10 Mar 2023).

Source repository: https://github.com/CSSEGISandData/COVID-19
Files (csse_covid_19_data/csse_covid_19_time_series/):
  time_series_covid19_confirmed_US.csv     county-level cumulative confirmed cases, with
                                           FIPS, Admin2, Province_State, Lat, Long_
  time_series_covid19_deaths_US.csv        same rows + Population column (used for population)
  time_series_covid19_confirmed_global.csv country/province level (inspected only)
"""
from __future__ import annotations

import os
import shutil
import urllib.request

import numpy as np
import pandas as pd

BASE = ("https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/"
        "csse_covid_19_data/csse_covid_19_time_series/")
FILES = {
    "confirmed_US": "time_series_covid19_confirmed_US.csv",
    "deaths_US": "time_series_covid19_deaths_US.csv",
    "confirmed_global": "time_series_covid19_confirmed_global.csv",
}
META_COLS = ["UID", "iso2", "iso3", "code3", "FIPS", "Admin2", "Province_State",
             "Country_Region", "Lat", "Long_", "Combined_Key"]


def download(raw_dir="data/raw", names=("confirmed_US", "deaths_US", "confirmed_global"),
             fallback_dir=None):
    """Fetch the CSVs into raw_dir (skips files already present).  If the network fails and
    fallback_dir contains a copy (e.g. confirmed_US.csv), copy it instead."""
    os.makedirs(raw_dir, exist_ok=True)
    paths = {}
    for n in names:
        dst = os.path.join(raw_dir, FILES[n])
        if not os.path.exists(dst):
            try:
                urllib.request.urlretrieve(BASE + FILES[n], dst)
            except Exception as e:  # noqa: BLE001
                src = os.path.join(fallback_dir or "", f"{n}.csv")
                if fallback_dir and os.path.exists(src):
                    shutil.copy(src, dst)
                else:
                    raise RuntimeError(f"could not download {n}: {e}")
        paths[n] = dst
    return paths


def _date_cols(df):
    return [c for c in df.columns if c not in META_COLS + ["Population", "Province/State",
                                                           "Country/Region", "Lat", "Long"]]


def load_us(raw_dir="data/raw"):
    """Return (meta, cum) where meta has FIPS, county, state, lat, lon, population and cum is
    a DataFrame (rows = UID, columns = pd.Timestamp) of cumulative confirmed cases."""
    conf = pd.read_csv(os.path.join(raw_dir, FILES["confirmed_US"]))
    deaths = pd.read_csv(os.path.join(raw_dir, FILES["deaths_US"]), usecols=["UID", "Population"])
    conf = conf.merge(deaths, on="UID", how="left")
    dcols = _date_cols(conf)
    meta = conf[["UID", "FIPS", "Admin2", "Province_State", "Lat", "Long_", "Population",
                 "Combined_Key"]].set_index("UID")
    cum = conf.set_index("UID")[dcols]
    cum.columns = pd.to_datetime(cum.columns, format="%m/%d/%y")
    return meta, cum


def dataset_summary(raw_dir="data/raw"):
    """Facts used in the dataset documentation (all computed, nothing hard-coded)."""
    meta, cum = load_us(raw_dir)
    daily = cum.diff(axis=1)
    neg = (daily < 0)
    glob = pd.read_csv(os.path.join(raw_dir, FILES["confirmed_global"]))
    gd = _date_cols(glob)
    s = {
        "us_rows": int(len(meta)),
        "us_rows_with_coords": int(((meta.Lat != 0) & meta.Lat.notna()).sum()),
        "us_rows_with_population_gt0": int((meta.Population > 0).sum()),
        "us_unassigned_or_out_of_rows": int(meta.Combined_Key.str.contains("Unassigned|Out of", na=False).sum()),
        "us_first_date": str(cum.columns[0].date()), "us_last_date": str(cum.columns[-1].date()),
        "us_n_dates": int(cum.shape[1]),
        "us_missing_values": int(cum.isna().sum().sum()),
        "us_negative_daily_entries_total": int(neg.sum().sum()),
        "us_negative_daily_entries_2020_03_to_07": int(neg.loc[:, "2020-03-01":"2020-07-31"].sum().sum()),
        "us_negative_daily_mass_2020_03_to_07": float(-daily.loc[:, "2020-03-01":"2020-07-31"][neg].sum().sum()),
        "quantity": "cumulative confirmed cases (reported), not prevalence",
        "global_rows": int(len(glob)), "global_first_date": gd[0], "global_last_date": gd[-1],
        "global_resolution": "country / some provinces (single Lat/Long per row, no population)",
    }
    return s
