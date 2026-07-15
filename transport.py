
import numpy as np

def transport_step(C, vx, vy, D, ku, dx, dt, boundary_value=1.0):
    # x-direction (columns) must be non-periodic: np.roll wraps column -1
    # (far tissue edge) directly onto column 0 (fixed vessel-wall source),
    # which "leaks" the C=1 boundary across the whole domain in one step
    # regardless of real transport. Left edge keeps its own value (it's
    # overwritten to the Dirichlet C=1 below anyway); right edge is
    # no-flux/Neumann (zero-gradient far-field tissue boundary).
    C_xm = np.empty_like(C); C_xm[:,1:] = C[:,:-1]; C_xm[:,0] = C[:,0]
    C_xp = np.empty_like(C); C_xp[:,:-1] = C[:,1:]; C_xp[:,-1] = C[:,-1]

    lap = (
        np.roll(C,1,0)+np.roll(C,-1,0)+
        C_xm + C_xp - 4*C
    )/dx**2

    # Upwind differencing for advection: central differencing oscillates and
    # overshoots C above 1.0 once the grid Peclet number (v*dx/D) exceeds ~2,
    # which happens routinely once velocity is non-negligible relative to D.
    dCdx_fwd = (C_xp-C)/dx
    dCdx_bwd = (C-C_xm)/dx
    dCdx = np.where(vx >= 0, dCdx_bwd, dCdx_fwd)

    dCdy_fwd = (np.roll(C,-1,0)-C)/dx
    dCdy_bwd = (C-np.roll(C,1,0))/dx
    dCdy = np.where(vy >= 0, dCdy_bwd, dCdy_fwd)

    C = C + dt*(D*lap - vx*dCdx - vy*dCdy - ku*C)
    # boundary_value defaults to 1.0 (instant fixed source, original behavior).
    # generate_dataset.py passes a time-ramping value here to model first-order
    # drug release kinetics: C(0,t) = 1 - exp(-k_release*t).
    C[:,0] = boundary_value
    return C

def penetration_depth(C, threshold, dx):
    idx = np.where(C.mean(axis=0) > threshold)[0]
    return idx.max()*dx if len(idx) else 0
