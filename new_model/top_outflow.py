from constants import alpha
import numpy as np
from constants import alpha

rho_w = 1000    #water density
N_c = 1e9       #cloud droplet concentration

def Dc_func(X,Z_c):
    """
    Monodisperse cloud droplet diameter (from Murakami 1989)

    Parameters
        X (float): Cloud water content [kg/m2]
        Z_c (float): Cloud depth [m]    
    
    Returns
        D_c (float): Cloud droplet diameter [m]
    """
    D_c = ((6*X/Z_c)/(np.pi*N_c*rho_w))**(1/3)
    return D_c

def v_t(D):
    """
    Droplet fall velocity (from Georgakakos and Bras 1984a)

    Parameters
        D (float): Droplet diameter [m]
    """
    return alpha*D

def O_t(X,top_index,Z_b,Z,w_parcel_max,lfc_index,dt):
    """
    Cloud outflow rate

    Parameters
        X (float): Cloud water content [kg/m2]
        top_index (int): Index of the top of the cloud
        Z_b (float): Base of the cloud [m]
        Z (array): Height profile [m]
        w_parcel_max (float): Maximum parcel vertical velocity [m/s]
        lfc_index (int): Index of the LFC
        dt (float): Time step [s]

    Returns
        O_t (float): Cloud outflow rate [kg/m2/s]
    """
    
    #finding the characteristic heights
    Z_t = Z[top_index]
    D_cloud = Dc_func(X,Z_t-Z_b)

    if w_parcel_max>0:
        Z_mid = (Z_t + Z[lfc_index])/2
        Z_lim = ((Z_mid - Z_t)*(v_t(D_cloud)*dt + Z_t))/(w_parcel_max*dt + Z_mid - Z_t)
        Z_min = (v_t(D_cloud)*(Z_mid - Z[lfc_index])/w_parcel_max) + Z[lfc_index]
        Z_max = (v_t(D_cloud)*(Z_mid - Z_t)/w_parcel_max) + Z_t

    if Z_t <= Z_b or X == 0 or w_parcel_max<alpha*D_cloud:
            return 0
    else:
            flux_term = np.pi/6*rho_w*D_cloud**3*N_c
            if Z_min < Z_lim:
                Z_min = Z_lim

    
            if Z_min>Z_mid:
                    print('1')
                    Delta_Z = Z_max - Z_min

                    O_t = flux_term*( w_parcel_max/(Delta_Z)*((Z_max - Z_t)**2/2 - (Z_min - Z_t)**2/2)/(Z_mid - Z_t) \
                            - v_t(D_cloud)*(Z_max - Z_min))

            else:
                    print('2')
                    Delta_Z2 = Z_max - Z_mid
                    Delta_Z1 = Z_mid - Z_min

                    O_t = flux_term*((1/Delta_Z1)*( w_parcel_max*((Z_mid - Z[lfc_index])**2/2 - (Z_min - Z[lfc_index])**2/2)/(Z_mid - Z[lfc_index]) - v_t(D_cloud)*(Z_mid - Z_min))\
                            + (1/Delta_Z2)*( w_parcel_max*((Z_max - Z_t)**2/2 - (Z_mid - Z_t)**2/2)/(Z_mid - Z_t) - v_t(D_cloud)*(Z_max - Z_mid))) 

            return O_t

