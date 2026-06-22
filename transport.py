
import numpy as np

def transport_step(C, vx, vy, D, ku, dx, dt):
    lap = (
        np.roll(C,1,0)+np.roll(C,-1,0)+
        np.roll(C,1,1)+np.roll(C,-1,1)-4*C
    )/dx**2

    dCdx = (np.roll(C,-1,1)-np.roll(C,1,1))/(2*dx)
    dCdy = (np.roll(C,-1,0)-np.roll(C,1,0))/(2*dx)

    C = C + dt*(D*lap - vx*dCdx - vy*dCdy - ku*C)
    C[:,0] = 1.0
    return C

def penetration_depth(C, threshold, dx):
    idx = np.where(C.mean(axis=0) > threshold)[0]
    return idx.max()*dx if len(idx) else 0
