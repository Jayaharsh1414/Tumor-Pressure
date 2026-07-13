
import numpy as np
from parameters import K, mu

def solve_pressure(N, iterations=4000):
    P = np.zeros((N, N))
    P[:,0] = 1.0
    P[:,-1] = 0.0

    for _ in range(iterations):
        P_old = P.copy()
        P[1:-1,1:-1] = 0.25*(
            P_old[2:,1:-1] + P_old[:-2,1:-1] +
            P_old[1:-1,2:] + P_old[1:-1,:-2]
        )
    return P

def compute_velocity(P, dx, K_val=None):
    dPy, dPx = np.gradient(P, dx)
    if K_val is None:
        K_val = K
    vx = -(K_val/mu)*dPx
    vy = -(K_val/mu)*dPy
    return vx, vy
