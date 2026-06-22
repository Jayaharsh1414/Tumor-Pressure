
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from parameters import *
from fluid_model import solve_pressure, compute_velocity
from transport import transport_step, penetration_depth

dx = L/(N-1)

# ---------------- Result 1 ----------------
P = solve_pressure(N)

plt.figure()
plt.contourf(P,30)
plt.colorbar(label='Pressure')
plt.title('Result 1: Pressure Contour')
plt.savefig('result1_pressure.png')

# ---------------- Result 2 ----------------
vx, vy = compute_velocity(P, dx)

plt.figure()
plt.quiver(vx[::5,::5], vy[::5,::5])
plt.title('Result 2: Velocity Field')
plt.savefig('result2_velocity.png')

# Grid Independence Study
grid_results = []
for grid in [50,100,200]:
    dxg = L/(grid-1)
    Pg = solve_pressure(grid)
    vxg, vyg = compute_velocity(Pg, dxg)
    vmax = np.sqrt(vxg**2 + vyg**2).max()
    grid_results.append([grid, vmax])

grid_df = pd.DataFrame(grid_results,
                       columns=['Grid_Size','Max_Velocity'])

grid_df.to_csv('grid_independence.csv', index=False)

# ---------------- Result 3 ----------------
sizes = [20,50,100,150,200]

all_data = []

plt.figure()

for size in sizes:

    D = kB*T/(3*np.pi*mu*(size*1e-9))

    C = np.zeros((N,N))
    C[:,0] = 1.0

    depths = []
    times = []

    for step in range(1200):

        C = transport_step(C,vx,vy,D,ku,dx,dt)

        t = step*dt

        if abs(t-20) < dt/2 and size==100:
            plt.figure()
            plt.imshow(C,origin='lower')
            plt.colorbar()
            plt.title('Concentration t=20')
            plt.savefig('result3_t20.png')

        if abs(t-40) < dt/2 and size==100:
            plt.figure()
            plt.imshow(C,origin='lower')
            plt.colorbar()
            plt.title('Concentration t=40')
            plt.savefig('result3_t40.png')

        if abs(t-60) < dt/2 and size==100:
            plt.figure()
            plt.imshow(C,origin='lower')
            plt.colorbar()
            plt.title('Concentration t=60')
            plt.savefig('result3_t60.png')

        depth = penetration_depth(C, threshold, dx)

        depths.append(depth)
        times.append(t)

        all_data.append([size,t,depth])

    plt.plot(times, depths, label=f'{size} nm')

# ---------------- Result 4 ----------------
plt.xlabel('Time')
plt.ylabel('Penetration Depth')
plt.title('Result 4: Penetration Depth vs Time')
plt.legend()
plt.savefig('result4_depth_time_all_sizes.png')

# ---------------- Result 5 ----------------
df = pd.DataFrame(all_data,
                  columns=['Particle_Size','Time','Penetration_Depth'])

final_depths = []

for size in sizes:
    temp = df[df['Particle_Size']==size]
    final_depths.append(temp['Penetration_Depth'].iloc[-1])

plt.figure()
plt.plot(sizes, final_depths, marker='o')
plt.xlabel('Particle Size (nm)')
plt.ylabel('Final Penetration Depth')
plt.title('Result 5: Penetration Depth vs Particle Size')
plt.savefig('result5_size_vs_depth.png')

df.to_csv('AI_dataset.csv', index=False)

print('Simulation Complete')
