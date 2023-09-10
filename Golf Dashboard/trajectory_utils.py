import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

def golf_ball_trajectory(initial_velocity, angular_spin, mass = 45.9, dT = 0.01, g = 9.80665, drag_coeff = 0.47, air_density = 1204, surface_area = 0.005854827, magnus_coeff = -0.1, plot = True):
    # dT = delta-T in sec
    # mass = 45.9 #mass of golf ball in grams
    # g = 9.80665 # acceleration due to gravity in m/s^2
    # air_density = 1204 # in gram/m^3
    # drag_coeff = 0.47
    # surface_area = 0.005854827 # of golf ball in m^2
    trajectory=[[0],[0],[0]]
    x,y,z=0,0,0
    duration = [0]
    Vx,Vy,Vz = initial_velocity[0],initial_velocity[1],initial_velocity[1]
    while True:
        Rx = 0.5*drag_coeff * air_density * surface_area * (Vx**2)
        Ry = 0.5*drag_coeff * air_density * surface_area * (Vy**2)
        Rz = 0.5*drag_coeff * air_density * surface_area * (Vz**2)
        air_resistance = [Rx,Ry,Rz]

        Mx = magnus_coeff*( angular_spin[1]*Vz - angular_spin[2]*Vy )
        My = magnus_coeff*( angular_spin[2]*Vx - angular_spin[0]*Vz )
        Mz = magnus_coeff*( angular_spin[0]*Vy - angular_spin[1]*Vx )
        magnus_force = [Mx,My,Mz]
        
        acc_X = ( -air_resistance[0] + magnus_force[0]) / mass
        x += Vx*dT + 0.5*acc_X*(dT**2)
        Vx += acc_X*dT 
        
        acc_Y = -g + ( -air_resistance[1] + magnus_force[1]) / mass
        y += Vy*dT + 0.5*acc_Y*(dT**2)
        Vy += acc_Y*dT 
        
        acc_Z = ( -air_resistance[2] + magnus_force[2]) / mass
        z += Vz*dT + 0.5*acc_Z*(dT**2)
        Vz += acc_Z*dT 

        # print(acc_X,acc_Y,acc_Z)
        
        if y == 0:
            trajectory[0].append(x)
            trajectory[1].append(y)
            trajectory[2].append(z)
            duration.append(duration[-1]+dT)
            break
        elif y < 0:
            trajectory[0].append(trajectory[0][-1])
            trajectory[1].append(0)
            trajectory[2].append(trajectory[2][-1])
            duration.append(duration[-1]+dT)
            break
        else:
            trajectory[0].append(x)
            trajectory[1].append(y)
            trajectory[2].append(z)
            duration.append(duration[-1]+dT)
    
    apex = max(trajectory[1])
    carry = ((trajectory[0][-1]**2)+(trajectory[2][-1]**2))**0.5

    if plot:
        fig = plt.figure()
        ax = plt.axes(projection='3d')
        # x,y,z = np.linspace(0, 1, 100),np.linspace(0, 1, 100),np.linspace(0, 1, 100)
        # x,y,z = [i for i in range(100)],[i for i in range(100)],[i for i in range(100)]
        x,y,z = trajectory[0],trajectory[2],trajectory[1]
        ax.plot3D(x,y,z,'green')
        ax.set_title('Golf Ball Trajectory')
        ax.set_xlabel('X')
        ax.set_ylabel('Z')
        ax.set_zlabel('Y')
        fig.set_figheight(15)
        fig.set_figwidth(15)
        return trajectory,duration[-1],apex,carry,fig
    return trajectory,duration[-1],apex,carry