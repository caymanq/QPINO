"""Build a 1-D spatial COVID-19 field along the Interstate-10 corridor (Los Angeles ->
Jacksonville) from JHU county data.

Field definition
    u(x_j, t_n) = [7-day trailing mean of new reported confirmed cases per 100k residents in
                  transect bin j on day t_n] / S,
where S is the maximum of that quantity over all bins and the TRAINING days only.  It is
reported-case incidence (a testing-dependent proxy), not infection prevalence.

Leakage control: daily counts use cum[t] - cum[t-1]; negative revisions are clipped to 0 on
the day they appear (never redistributed backwards); smoothing uses only days t-6..t;
normalisation uses only training-period statistics.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# I-10 waypoints (lat, lon), west -> east
I10_WAYPOINTS = [
    ("Santa Monica CA", 34.02, -118.49), ("Los Angeles CA", 34.05, -118.24),
    ("San Bernardino CA", 34.11, -117.29), ("Palm Springs CA", 33.83, -116.55),
    ("Blythe CA", 33.61, -114.60), ("Phoenix AZ", 33.45, -112.07), ("Tucson AZ", 32.22, -110.97),
    ("Lordsburg NM", 32.35, -108.71), ("Las Cruces NM", 32.32, -106.78), ("El Paso TX", 31.76, -106.49),
    ("Fort Stockton TX", 30.89, -102.88), ("Junction TX", 30.49, -99.77), ("San Antonio TX", 29.42, -98.49),
    ("Houston TX", 29.76, -95.37), ("Beaumont TX", 30.08, -94.13), ("Lafayette LA", 30.22, -92.02),
    ("Baton Rouge LA", 30.45, -91.19), ("New Orleans LA", 29.95, -90.07), ("Mobile AL", 30.69, -88.04),
    ("Pensacola FL", 30.42, -87.22), ("Tallahassee FL", 30.44, -84.28), ("Lake City FL", 30.19, -82.64),
    ("Jacksonville FL", 30.33, -81.66),
]
R_EARTH = 6371.0


def haversine(lat1, lon1, lat2, lon2):
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dphi, dl = p2 - p1, np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * R_EARTH * np.arcsin(np.sqrt(a))


def densify(waypoints=I10_WAYPOINTS, step_km=2.0):
    lat = np.array([w[1] for w in waypoints]); lon = np.array([w[2] for w in waypoints])
    pts, s = [], [0.0]
    for i in range(len(lat) - 1):
        d = haversine(lat[i], lon[i], lat[i + 1], lon[i + 1])
        n = max(int(np.ceil(d / step_km)), 1)
        f = np.arange(n) / n
        pts.append(np.stack([lat[i] + f * (lat[i + 1] - lat[i]), lon[i] + f * (lon[i + 1] - lon[i])], 1))
    pts.append(np.array([[lat[-1], lon[-1]]]))
    P = np.concatenate(pts)
    seg = haversine(P[:-1, 0], P[:-1, 1], P[1:, 0], P[1:, 1])
    return P, np.concatenate([[0.0], np.cumsum(seg)])


def project_counties(meta, band_km=100.0, waypoints=I10_WAYPOINTS):
    """Arc-length coordinate s (km) of each county's nearest transect point and its distance."""
    P, s = densify(waypoints)
    ok = meta.Lat.notna() & (meta.Lat != 0) & (meta.Population > 0)
    m = meta[ok].copy()
    d = haversine(m.Lat.values[:, None], m.Long_.values[:, None], P[None, :, 0], P[None, :, 1])
    i = d.argmin(1)
    m["s_km"], m["dist_km"] = s[i], d[np.arange(len(m)), i]
    m = m[(m.dist_km <= band_km) & (m.s_km > 0) & (m.s_km < s[-1])]
    return m, float(s[-1])


def build_field(meta, cum, n_bins=64, band_km=100.0, start="2020-03-01", end="2020-07-31",
                train_end="2020-06-16", smooth=7):
    """Return dict with the processed field and all intermediate products."""
    counties, L_km = project_counties(meta, band_km)
    first = pd.Timestamp(start) - pd.Timedelta(days=smooth)  # trailing window needs history
    c = cum.loc[counties.index, first - pd.Timedelta(days=1):pd.Timestamp(end)]
    daily_raw = c.diff(axis=1).iloc[:, 1:]
    n_neg = int((daily_raw < 0).sum().sum())
    neg_mass = float(-daily_raw[daily_raw < 0].sum().sum())
    daily = daily_raw.clip(lower=0).fillna(0)
    edges = np.linspace(0, L_km, n_bins + 1)
    b = np.clip(np.digitize(counties.s_km.values, edges) - 1, 0, n_bins - 1)
    counties = counties.assign(bin=b)
    pop = np.bincount(b, weights=counties.Population.values, minlength=n_bins)
    cases_bin = np.zeros((daily.shape[1], n_bins))
    for j in range(n_bins):
        idx = counties.index[b == j]
        if len(idx):
            cases_bin[:, j] = daily.loc[idx].sum(0).values
    dates_all = daily.columns
    cases_sm = pd.DataFrame(cases_bin, index=dates_all).rolling(smooth, min_periods=smooth).mean().values
    empty = pop == 0
    inc = np.full_like(cases_sm, np.nan)
    inc[:, ~empty] = cases_sm[:, ~empty] / pop[~empty] * 1e5
    # fill empty bins by periodic linear interpolation in s (documented)
    centres = 0.5 * (edges[:-1] + edges[1:])
    if empty.any():
        for t in range(inc.shape[0]):
            inc[t, empty] = np.interp(centres[empty], centres[~empty], inc[t, ~empty], period=L_km)
    keep = dates_all >= pd.Timestamp(start)
    dates, inc = dates_all[keep], inc[keep]
    raw_daily_bin = np.zeros((len(dates), n_bins))
    raw_daily_bin[:, ~empty] = cases_bin[keep][:, ~empty] / pop[~empty] * 1e5
    train_mask = dates <= pd.Timestamp(train_end)
    S = float(np.nanmax(inc[train_mask]))
    u = inc / S
    x = 2 * np.pi * (np.arange(n_bins) + 0.5) / n_bins
    return dict(u=u, incidence_per100k=inc, raw_daily_per100k=raw_daily_bin, dates=dates, x=x,
                s_centres_km=centres, L_km=L_km, bin_pop=pop, empty_bins=np.where(empty)[0],
                counties=counties, scale_S=S, n_negative_clipped=n_neg, negative_mass_clipped=neg_mass,
                band_km=band_km, n_bins=n_bins)


def mirror_extend(u):
    """Even (mirror) extension: [u_0..u_{N-1}, u_{N-1}..u_0] -> periodic with no wrap jump.
    The resulting Fourier series is a cosine series (all Im a_k = 0)."""
    return np.concatenate([u, u[..., ::-1]], axis=-1)


def wrap_jump(u):
    """Relative size of the artificial discontinuity created by identifying x=0 with x=2pi."""
    return np.abs(u[..., -1] - u[..., 0]) / (np.abs(u).max(-1) + 1e-12)


def split_indices(dates, train_end="2020-06-16", val_end="2020-06-30", test_end="2020-07-31"):
    d = pd.DatetimeIndex(dates)
    tr = np.where(d <= pd.Timestamp(train_end))[0]
    va = np.where((d > pd.Timestamp(train_end)) & (d <= pd.Timestamp(val_end)))[0]
    te = np.where((d > pd.Timestamp(val_end)) & (d <= pd.Timestamp(test_end)))[0]
    return tr, va, te
