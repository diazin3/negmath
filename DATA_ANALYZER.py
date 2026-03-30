#!/usr/bin/env python3
"""
NegMath-Analyzer — Tuned Max-to-Sum Ratio Probe
"""

import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import label
import os
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

try:
    import cupy as cp
    xp = cp
    from cupy.fft import fftn
    print("✅ GPU (CuPy) enabled.")
except ImportError:
    xp = np
    from numpy.fft import fftn
    print("⚠️ CPU fallback.")

def get_energy_spectrum(u_phys, N):
    u_hat = fftn(u_phys, axes=(1, 2, 3))
    energy_3d = 0.5 * xp.sum(xp.abs(u_hat)**2, axis=0) / (N**6)
    k = xp.fft.fftfreq(N) * N
    kx, ky, kz = xp.meshgrid(k, k, k, indexing='ij')
    k_norm = xp.sqrt(kx**2 + ky**2 + kz**2).round().astype(int)
    k_max = N // 2
    e_1d = xp.zeros(k_max)
    for r in range(k_max):
        e_1d[r] = xp.sum(energy_3d[k_norm == r])
    return e_1d

def calculate_clusteredness(mask):
    mask_cpu = cp.asnumpy(mask) if xp == cp else mask
    structure = np.ones((3, 3, 3))
    _, count = label(mask_cpu, structure=structure)
    return count

def safe_log(series, clip_min=1e-8, clip_max=1e300):
    s = np.asarray(series, dtype=float)
    s = np.nan_to_num(s, nan=0.0, posinf=clip_max, neginf=clip_min)
    return np.clip(s, clip_min, clip_max)

def run_analysis(h5_path):
    print(f"🧐 Analyzing with tuned Max-to-Sum Ratio: {h5_path}")
    
    PHYS_SING_THRESHOLD = 1e20      # Good cutoff for now
    MAX_TO_SUM_TRIGGER   = 0.85     # Reduced false positives
    
    metrics = []
    t_phys_critical = None
    t_shadow = None
    prev_ratio = 0.0

    with h5py.File(h5_path, 'r') as f:
        steps = sorted(f.keys(), key=lambda x: float(x.split('_')[1]))
        N = f[steps[0]]['u_phys'].shape[1] if steps else 128

        for i, step_name in enumerate(steps):
            grp = f[step_name]
            t = grp.attrs.get('t', 0.0)
            tag = step_name.split('_')[-1]

            try:
                omega_phys = xp.asarray(grp['omega_phys'])
                stretch    = xp.asarray(grp['stretching'])
                shadow_both = xp.asarray(grp.get('shadow_both', xp.zeros_like(omega_phys)))

                E_k = get_energy_spectrum(xp.asarray(grp['u_phys']), N)
                bottleneck = float(E_k[-1] / (xp.max(E_k) + 1e-12))

                clean = shadow_both[xp.isfinite(shadow_both)]
                shadow_rms = float(xp.sqrt(xp.mean(clean**2))) if len(clean) > 0 else np.nan

                # Max-to-Sum Ratio
                if len(clean) > 0:
                    abs_clean = xp.abs(clean)
                    max_val = float(xp.max(abs_clean))
                    sum_val = float(xp.sum(abs_clean))
                    max_to_sum = max_val / (sum_val + 1e-30)
                else:
                    max_to_sum = 0.0

                # Physical ε bad fraction
                om_bad_eps = float(xp.mean(xp.abs(omega_phys) > PHYS_SING_THRESHOLD))
                stretch_bad_eps = float(xp.mean(xp.abs(stretch) > PHYS_SING_THRESHOLD))
                phys_bad_frac_eps = max(om_bad_eps, stretch_bad_eps)

                nan_c = int(xp.sum(xp.isnan(shadow_both)))
                inf_c = int(xp.sum(xp.isinf(shadow_both)))

                clusters = calculate_clusteredness(xp.isnan(shadow_both) | xp.isinf(shadow_both)) if (nan_c + inf_c) > 0 else 0

                # Combined risk (simple)
                risk_score = max_to_sum + 5.0 * phys_bad_frac_eps

                # Trigger logic with confirmation
                is_shadow_trigger = (max_to_sum > MAX_TO_SUM_TRIGGER and prev_ratio > 0.75)
                is_phys_critical = (phys_bad_frac_eps > 1e-5 or 
                                  grp.attrs.get('max_omega_finite', 0) > 1e25)

                if t_phys_critical is None and is_phys_critical:
                    t_phys_critical = t
                    print(f"💥 Physical ε-singularity at t={t:.6f} (bad_frac_eps={phys_bad_frac_eps:.4f})")

                if t_shadow is None and is_shadow_trigger:
                    t_shadow = t
                    print(f"⚠️ Max-to-Sum trigger at t={t:.6f} (ratio={max_to_sum:.4f}, risk={risk_score:.2f})")

                metrics.append({
                    'step': i, 't': t, 'tag': tag,
                    'bottleneck': bottleneck,
                    'shadow_rms': shadow_rms,
                    'max_to_sum_ratio': max_to_sum,
                    'phys_bad_frac_eps': phys_bad_frac_eps,
                    'risk_score': risk_score,
                    'nan_count': nan_c, 'inf_count': inf_c, 'clusters': clusters,
                    'palinstrophy': grp.attrs.get('palinstrophy', 0.0),
                    'max_omega_finite': grp.attrs.get('max_omega_finite', np.nan),
                    'max_stretching_finite': grp.attrs.get('max_stretching_finite', np.nan),
                    'bt_div': grp.attrs.get('bt_div', np.nan),
                    'phys_critical': int(is_phys_critical)
                })

                prev_ratio = max_to_sum

            except Exception as e:
                print(f"⚠️ Skip {step_name}: {e}")

    df = pd.DataFrame(metrics)
    df.to_csv("negmath_full_analysis.csv", index=False)
    print(f"\n✅ Analysis complete (Max-to-Sum trigger = {MAX_TO_SUM_TRIGGER})")

    if t_phys_critical and t_shadow:
        lead = t_phys_critical - t_shadow
        print(f"🎯 Lead time: {lead:.6f}s")

    # Plotting (same clean layout)
    fig, axs = plt.subplots(6, 1, figsize=(13, 18), sharex=True)

    axs[0].plot(df['t'], df['palinstrophy'], color='black', lw=1.2)
    axs[0].set_ylabel("Palinstrophy")
    axs[0].set_title("NegMath MRI — Max-to-Sum Ratio Probe + ε=1e20")

    axs[1].semilogy(df['t'], df['bottleneck'], color='blue', label="Spectral Bottleneck")
    axs[1].set_ylabel("Bottleneck")
    axs[1].legend()

    axs[2].plot(df['t'], df['max_to_sum_ratio'], color='darkred', lw=2, label="Max-to-Sum Ratio")
    axs[2].axhline(0.85, color='orange', ls='--', alpha=0.7, label="Trigger threshold")
    axs[2].set_ylabel("Max-to-Sum Ratio")
    if t_shadow:
        axs[2].axvline(t_shadow, color='orange', ls='-', lw=1.5, label=f"Trigger t={t_shadow:.4f}")
    axs[2].legend()

    axs[3].semilogy(df['t'], safe_log(df['max_omega_finite']), color='darkgreen', label="max |ω|")
    axs[3].semilogy(df['t'], safe_log(df['max_stretching_finite']), color='brown', label="max |stretching|")
    axs[3].set_ylabel("Max Finite")
    if t_phys_critical:
        axs[3].axvline(t_phys_critical, color='red', lw=2.5, label=f"ε-Singularity t={t_phys_critical:.4f}")
    axs[3].legend()

    axs[4].plot(df['t'], df['phys_bad_frac_eps'], color='purple', label="Phys Bad Frac (ε)")
    axs[4].set_ylabel("Bad Fraction (ε)")
    axs[4].legend()

    axs[5].semilogy(df['t'], safe_log(df['bt_div']), color='cyan', label="BT Divergence")
    axs[5].set_ylabel("BT Divergence")
    axs[5].set_xlabel("Time t")
    axs[5].legend()

    plt.tight_layout()
    plt.savefig("negmath_full_mri_plot.png", dpi=240, bbox_inches='tight')
    print("📈 Plot saved.")

if __name__ == "__main__":
    latest = max([d for d in os.listdir('.') if d.startswith("NS_NegMath_Raw_")], default=None)
    if latest:
        h5f = os.path.join(latest, "all_steps_snapshots.h5")
        if os.path.exists(h5f):
            run_analysis(h5f)
        else:
            print(f"❌ HDF5 not found in {latest}")
    else:
        print("❌ No run directory found.")