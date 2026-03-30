#!/usr/bin/env python3
"""
3D Incompressible Navier-Stokes: NegMath Singularity Snapshot Tool (Physical Tracking)
Updated: Robust ε regularization in shadows only + clear NaN vs Inf distinction
Goal: Detect when physical fields approach singularity (max_finite ~ threshold or bad_fraction rises)
"""

import os
import h5py
import numpy as np
import pandas as pd
from datetime import datetime

# ====================== GPU / CPU FALLBACK ======================
try:
    import cupy as cp
    from cupy.fft import fftn, ifftn, fftfreq
    xp = cp
    print("✅ GPU (CuPy) detected.")
    GPU_AVAILABLE = True
except ImportError:
    import numpy as xp
    from numpy.fft import fftn, ifftn, fftfreq
    print("⚠️ No GPU — falling back to CPU (NumPy).")
    GPU_AVAILABLE = False

np.seterr(divide='ignore', invalid='ignore', over='ignore')

# ====================== 3D SPECTRAL UTILITIES ======================
def make_wavenumbers_3d(N, L=2 * xp.pi):
    k = 2 * xp.pi * fftfreq(N, d=L/N)
    kx, ky, kz = xp.meshgrid(k, k, k, indexing='ij')
    k2 = kx**2 + ky**2 + kz**2
    return kx, ky, kz, k2

def dealias_mask_3d(N):
    kk = fftfreq(N) * N
    kx_i, ky_i, kz_i = xp.meshgrid(kk, kk, kk, indexing='ij')
    cutoff = N // 3
    mask = ((xp.abs(kx_i) <= cutoff) & 
            (xp.abs(ky_i) <= cutoff) & 
            (xp.abs(kz_i) <= cutoff)).astype(xp.float64)
    return mask

def poisson_solve_velocity(omega_hat, kx, ky, kz, k2, mask):
    with np.errstate(divide='ignore', invalid='ignore'):
        denom = k2
        ux = 1j * (ky * omega_hat[2] - kz * omega_hat[1]) / denom
        uy = 1j * (kz * omega_hat[0] - kx * omega_hat[2]) / denom
        uz = 1j * (kx * omega_hat[1] - ky * omega_hat[0]) / denom
    ux[0,0,0] = uy[0,0,0] = uz[0,0,0] = 0.0
    return xp.stack([ux * mask, uy * mask, uz * mask], axis=0)

# ====================== ROBUST STATUS COUNTER ======================
def count_field_status(field, eps=1e-12):
    """Return detailed statistics distinguishing finite, Inf, and NaN."""
    n_total = field.size
    n_finite = int(xp.sum(xp.isfinite(field)))
    n_inf = int(xp.sum(xp.isinf(field)))
    n_nan = int(xp.sum(xp.isnan(field)))
    bad_fraction = 1.0 - (n_finite / n_total)
    
    finite_abs = xp.abs(field[xp.isfinite(field)])
    max_finite = float(xp.max(finite_abs)) if finite_abs.size > 0 else np.inf
    
    return {
        'max_finite': max_finite,
        'nan_count': n_nan,
        'inf_count': n_inf,
        'bad_fraction': bad_fraction
    }

# ====================== UNIFIED NEGMATH COMPUTE CORE ======================
def compute_all_physical_and_shadows(omega_hat, kx, ky, kz, k2, nu, mask, eps=1e-12):
    """
    Unified helper with ε regularization ONLY in shadow fields.
    """
    N = omega_hat.shape[1]
    u_hat = poisson_solve_velocity(omega_hat, kx, ky, kz, k2, mask)
    u_phys = xp.real(ifftn(u_hat, axes=(1,2,3)))
    omega_phys = xp.real(ifftn(omega_hat, axes=(1,2,3)))
    ks = [kx, ky, kz]

    stretching = xp.zeros((3, N, N, N), dtype=xp.float64)
    for i in range(3):
        for j in range(3):
            grad_u_i_j = xp.real(ifftn(1j * ks[j] * u_hat[i]))
            stretching[i] += omega_phys[j] * grad_u_i_j

    visc_phys = xp.real(ifftn(nu * (-k2) * omega_hat, axes=(1,2,3)))

    # === SHADOW FIELDS with safe ε regularization ===
    omega2_safe = xp.maximum(omega_phys**2, eps**2)
    visc2_safe  = xp.maximum(visc_phys**2, eps**2)

    shadow_stretch = stretching / omega2_safe
    shadow_visc    = visc_phys / visc2_safe
    shadow_both    = shadow_stretch + shadow_visc 

    return {
        'u_phys': u_phys, 
        'omega_phys': omega_phys, 
        'stretching': stretching,
        'visc_phys': visc_phys, 
        'shadow_stretch': shadow_stretch,
        'shadow_visc': shadow_visc, 
        'shadow_both': shadow_both,
        'u_hat': u_hat
    }

# ====================== RHS FUNCTIONS (unchanged) ======================
def rhs_baseline(omega_hat, kx, ky, kz, k2, nu, mask):
    core = compute_all_physical_and_shadows(omega_hat, kx, ky, kz, k2, nu, mask)
    u_phys, stretching, visc_phys = core['u_phys'], core['stretching'], core['visc_phys']
    N, ks = omega_hat.shape[1], [kx, ky, kz]
    advection = xp.zeros((3, N, N, N), dtype=xp.float64)
    for i in range(3):
        for j in range(3):
            advection[i] += u_phys[j] * xp.real(ifftn(1j * ks[j] * omega_hat[i]))
    return fftn(stretching - advection + visc_phys, axes=(1,2,3)) * mask

def rhs_negmath_stretching(omega_hat, kx, ky, kz, k2, nu, mask):
    core = compute_all_physical_and_shadows(omega_hat, kx, ky, kz, k2, nu, mask)
    return -fftn(core['shadow_stretch'], axes=(1,2,3)) * mask - (nu * k2 * omega_hat)

def rhs_negmath_viscous(omega_hat, kx, ky, kz, k2, nu, mask):
    core = compute_all_physical_and_shadows(omega_hat, kx, ky, kz, k2, nu, mask)
    nonvisc_hat = rhs_baseline(omega_hat, kx, ky, kz, k2, nu, mask) + (nu * k2 * omega_hat)
    return nonvisc_hat + fftn(core['shadow_visc'], axes=(1,2,3)) * mask

def rhs_negmath_both(omega_hat, kx, ky, kz, k2, nu, mask):
    core = compute_all_physical_and_shadows(omega_hat, kx, ky, kz, k2, nu, mask)
    return -fftn(core['shadow_stretch'], axes=(1,2,3)) * mask + fftn(core['shadow_visc'], axes=(1,2,3)) * mask

# ====================== RK4 & SHORT HORIZON (unchanged) ======================
def rk4_step(omega_hat, dt, rhs_fn, kx, ky, kz, k2, nu, mask):
    k1 = rhs_fn(omega_hat, kx, ky, kz, k2, nu, mask)
    k2_ = rhs_fn(omega_hat + 0.5 * dt * k1, kx, ky, kz, k2, nu, mask)
    k3 = rhs_fn(omega_hat + 0.5 * dt * k2_, kx, ky, kz, k2, nu, mask)
    k4 = rhs_fn(omega_hat + dt * k3, kx, ky, kz, k2, nu, mask)
    return omega_hat + (dt / 6.0) * (k1 + 2 * k2_ + 2 * k3 + k4)

def short_horizon_divergence(omega_hat, dt, kx, ky, kz, k2, nu, mask, horizon, p_type):
    # ... (unchanged from original) ...
    w_base = omega_hat.copy()
    for _ in range(horizon):
        w_base = rk4_step(w_base, dt, rhs_baseline, kx, ky, kz, k2, nu, mask)
        w_base *= mask
    
    w_shadow = omega_hat.copy()
    last_known_stable = omega_hat.copy()
    rhs_p = {"nonlinear": rhs_negmath_stretching, "viscous": rhs_negmath_viscous}.get(p_type, rhs_negmath_both)
    
    try:
        for _ in range(horizon):
            w_next = rk4_step(w_shadow, dt, rhs_p, kx, ky, kz, k2, nu, mask)
            w_next *= mask
            if not xp.isfinite(w_next).all():
                return float('inf'), last_known_stable
            last_known_stable = w_next.copy()
            w_shadow = w_next
    except:
        return float('inf'), last_known_stable
        
    diff = xp.real(ifftn(w_shadow - w_base, axes=(1,2,3)))
    return float(xp.sqrt(xp.mean(diff**2))), last_known_stable

# ====================== HDF5 STORAGE ======================
def save_snapshot(h5_file, t, step, omega_hat, kx, ky, kz, k2, nu, mask, scalars, tag):
    grp_name = f"t_{t:07.4f}_{tag}"
    if grp_name in h5_file: 
        return scalars 
    grp = h5_file.create_group(grp_name)
    core = compute_all_physical_and_shadows(omega_hat, kx, ky, kz, k2, nu, mask)
    
    def to_f32_cpu(arr):
        cpu_arr = cp.asnumpy(arr) if GPU_AVAILABLE else arr
        return cpu_arr.astype(np.float32)

    dsets = ['u_phys', 'omega_phys', 'stretching', 'visc_phys', 'shadow_stretch', 'shadow_visc', 'shadow_both']
    for d in dsets:
        data = to_f32_cpu(core[d])
        c_shape = (data.shape[0], 32, 32, 32) if data.ndim == 4 else (32, 32, 32)
        grp.create_dataset(d, data=data, compression="gzip", compression_opts=6, chunks=c_shape)
    
    # Save NaN/Inf masks for shadow_both (useful for topology)
    for s_tag in ['stretch', 'visc', 'both']:
        field = core[f'shadow_{s_tag}']
        scalars[f'{s_tag}_nan_count'] = int(xp.isnan(field).sum())
        scalars[f'{s_tag}_inf_count'] = int(xp.isinf(field).sum())
        for m_type, m_func in [('nan', xp.isnan), ('inf', xp.isinf)]:
            m_data = to_f32_cpu(m_func(field)).astype(np.uint8)
            m_chunk = (m_data.shape[0], 32, 32, 32) if m_data.ndim == 4 else (32, 32, 32)
            grp.create_dataset(f"{m_type}_mask_{s_tag}", data=m_data, compression="lzf", chunks=m_chunk)

    # === ROBUST PHYSICAL METRICS ===
    omega_stats = count_field_status(core['omega_phys'])
    u_stats     = count_field_status(core['u_phys'])
    stretch_stats = count_field_status(core['stretching'])

    scalars.update({
        'max_omega_finite': omega_stats['max_finite'],
        'omega_nan': omega_stats['nan_count'],
        'omega_inf': omega_stats['inf_count'],
        'omega_bad_frac': omega_stats['bad_fraction'],
        
        'max_u_finite': u_stats['max_finite'],
        'u_nan': u_stats['nan_count'],
        'u_inf': u_stats['inf_count'],
        'u_bad_frac': u_stats['bad_fraction'],
        
        'max_stretching_finite': stretch_stats['max_finite'],
        'stretch_nan': stretch_stats['nan_count'],
        'stretch_inf': stretch_stats['inf_count'],
        'stretch_bad_frac': stretch_stats['bad_fraction'],
    })

    for k, v in scalars.items(): 
        grp.attrs[k] = v
    return scalars

# ====================== MAIN ======================
def main():
    outdir = f"NS_NegMath_Raw_{datetime.now().strftime('%Y%m%d_%H%M')}"
    os.makedirs(outdir, exist_ok=True)
    h5_path = os.path.join(outdir, "all_steps_snapshots.h5")

    N, nu, T, dt = 128, 1e-5, 0.1, 1e-4
    check_every, horizon = 20, 5 
    eps_reg = 1e-12                     # regularization for shadows only

    kx, ky, kz, k2 = make_wavenumbers_3d(N)
    mask = dealias_mask_3d(N)

    rng = np.random.default_rng(42)
    omega_hat = fftn(xp.asarray(0.5 * rng.normal(0, 1, (3, N, N, N))), axes=(1,2,3)) * mask

    history = []
    t, step = 0.0, 0
    print(f"🚀 RUNNING | N={N} | nu={nu} | ε={eps_reg} | H5: {h5_path} | RECORDING EVERY CHECK STEP\n" + "-"*180)

    with h5py.File(h5_path, "w") as h5_file:
        while t < T + 1e-10:
            if step % check_every == 0:
                palin = 0.5 * float(sum(xp.mean(xp.real(ifftn(1j*kc*omega_hat[i]))**2) 
                                      for i in range(3) for kc in [kx, ky, kz]))
                
                nl_div, nl_w = short_horizon_divergence(omega_hat, dt, kx, ky, kz, k2, nu, mask, horizon, "nonlinear")
                vi_div, vi_w = short_horizon_divergence(omega_hat, dt, kx, ky, kz, k2, nu, mask, horizon, "viscous")
                bt_div, bt_w = short_horizon_divergence(omega_hat, dt, kx, ky, kz, k2, nu, mask, horizon, "both")

                divs = [nl_div, vi_div, bt_div]
                stables = [nl_w, vi_w, bt_w]
                
                core = compute_all_physical_and_shadows(omega_hat, kx, ky, kz, k2, nu, mask, eps_reg)
                
                omega_stats = count_field_status(core['omega_phys'])
                u_stats     = count_field_status(core['u_phys'])

                scalars = {
                    't': t, 'step': step, 'palinstrophy': palin, 
                    'nl_div': nl_div, 'vi_div': vi_div, 'bt_div': bt_div,
                }

                # === TRIGGER LOGIC ===
                physics_threshold = 1e8      # tune this (related to 1/ν)
                bad_threshold     = 1e-5

                if (omega_stats['max_finite'] > physics_threshold or 
                    u_stats['max_finite'] > physics_threshold or
                    omega_stats['bad_fraction'] > bad_threshold or
                    u_stats['bad_fraction'] > bad_threshold):
                    trigger = "PHYSICS_NEAR_SINGULARITY"
                    save_w = omega_hat
                    failure_type = "INF" if omega_stats['inf_count'] > omega_stats['nan_count'] else "NAN"
                    print(f"🚨 PHYSICS NEAR SINGULARITY at t={t:.4f} | max|ω|={omega_stats['max_finite']:.2e} | "
                          f"bad_frac={omega_stats['bad_fraction']:.2e} | Type={failure_type}")
                elif any(np.isinf(divs)): 
                    trigger = "PRE_CRASH"
                    save_w = stables[np.argmax(np.isinf(divs))]
                elif (omega_stats['bad_fraction'] + u_stats['bad_fraction']) > 1e-6: 
                    trigger = "SINGULARITY"
                    save_w = omega_hat
                else: 
                    trigger = "STEP"
                    save_w = omega_hat

                phys_str = (f"maxω_f={omega_stats['max_finite']:.2e}  maxu_f={u_stats['max_finite']:.2e}  "
                           f"ω_bad={omega_stats['bad_fraction']:.2e}")

                full_row = save_snapshot(h5_file, t, step, save_w, kx, ky, kz, k2, nu, mask, scalars, trigger)
                history.append(full_row)
                
                color = "\033[91m" if trigger in ["PHYSICS_NEAR_SINGULARITY", "PRE_CRASH", "SINGULARITY"] else "\033[94m"
                print(f"{color}{t:6.4f} | P={palin:10.2f} | Tag: {trigger:22} | BT_Div={bt_div:8.2e} | "
                      f"ω_bad={omega_stats['bad_fraction']:.2e} | {phys_str}\033[0m")

            # Advance simulation
            omega_hat = rk4_step(omega_hat, dt, rhs_baseline, kx, ky, kz, k2, nu, mask) * mask
            t += dt
            step += 1

    pd.DataFrame(history).to_csv(os.path.join(outdir, "all_steps_metrics.csv"), index=False)
    print(f"\n✅ Finished. All steps recorded. Metrics saved to {outdir}/all_steps_metrics.csv")
    print(f"   HDF5 file: {h5_path}")

if __name__ == "__main__": 
    main()