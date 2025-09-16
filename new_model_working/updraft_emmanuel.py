import numpy as np
from constants import g

def calculate_parcel_updraft_velocity(Z, T_v_moist, T_v_env, lfc_index, top_index):
    """
    Calculate maximum parcel updraft velocity and CAPE (in height coordinates).

    Parameters:
        Z: height profile [m]
        T_v_moist: parcel virtual temperature profile [K]
        T_v_env: environment virtual temperature profile [K]
        lfc_index: index of the LFC
        top_index: index of the cloud top

    Returns:
        w_parcel_max: maximum updraft velocity [m/s]
        CAPE: convective available potential energy [J/kg]
    """

    if top_index > int(lfc_index):
        # Extract layers between LFC and cloud top
        Z_layer = Z[int(lfc_index):top_index+1]
        Tvp_layer = T_v_moist[int(lfc_index):top_index+1]
        Tve_layer = T_v_env[int(lfc_index):top_index+1]

        # Buoyancy [m/s^2]
        B = g * (Tvp_layer - Tve_layer) / Tve_layer

        # Layer thickness
        dz = np.diff(Z_layer)

        # Trapezoidal integration for CAPE
        B_mid = 0.5 * (B[:-1] + B[1:])
        CAPE = np.sum(B_mid * dz)

        # Prevent negative CAPE
        CAPE = max(CAPE, 0.0)

        # Max updraft velocity
        w_parcel_max = np.sqrt(2 * CAPE)
    else:
        CAPE = 0.0
        w_parcel_max = 0.0

    return w_parcel_max, CAPE
def w_profile(w_parcel_max,Z,lfc_index,top_index):

        """
        the scheme for calculating the updraft velocity profile

        Parameters:
            w_parcel_max: maximum updraft velocity [m/s]
            Z: height profile  [m]
            lfc_index: index of the LFC
            top_index: index of the top of the cloud

        Returns:
            w_profile: updraft velocity profile [m/s]
        """

        mid_index = int((lfc_index+top_index)/2)

        w_profile = np.zeros(top_index-lfc_index +1)
        
        w_profile[0:mid_index-lfc_index] = w_parcel_max*(Z[lfc_index:mid_index]-Z[lfc_index])/(Z[mid_index]-Z[lfc_index]) 
        w_profile[mid_index-lfc_index+1:top_index -lfc_index+1] = w_parcel_max*(Z[top_index]-Z[mid_index+1:top_index+1])/(Z[top_index]-Z[mid_index])
        w_profile[mid_index-lfc_index] = w_parcel_max
        return w_profile
