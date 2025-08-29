import numpy as np
from constants import R_dry,g
from scipy.interpolate import interp1d

"""
def calculate_parcel_updraft_velocity(p,T_v_moist,T_v_env,lfc_index,Z,top_index):

    
    the scheme for calculating the maximum updraft velocity (Emanuel 1991)

    Parameters:
        p: pressure profile  [Pa]"
        T_v_moist: parcel virtual temperature profile [K]"
        T_env: environment virtual temperature profile  [K]"
        lfc_index: index of the LFC
        top_index: index of the top of the cloud

    Returns:
        w_parcel_max: maximum updraft velocity [m/s]
        CAPE: convective available potential energy
    
    
    if top_index > 1 and top_index > int(lfc_index):
        Z_mid = 0.5*(Z[int(lfc_index)]+Z[top_index])
        before_mid_idx = np.where(Z < Z_mid)[0][-1]
        interp_parcel = interp1d(Z,T_v_moist,kind='linear')
        interp_env = interp1d(Z,T_v_env,kind='linear')
        T_v_parcel_mid = interp_parcel(Z_mid)   
        T_v_env_mid = interp_env(Z_mid)
        # Get the pressure and temperature slices first
        p_slice = p[int(lfc_index):top_index]
        #T_v_parcel_slice = T_v_moist[int(lfc_index):top_index]
        #T_v_env_slice = T_v_env[int(lfc_index):top_index]
        
        # Check if we have enough levels
        #if len(p_slice) < 3:
            #print("Warning: Only one level between LFC and top - CAPE may be inaccurate")
            #CAPE = 0
            #w_parcel_max = 0
        #else:
            # Calculate differences
        delta_ln_p = -np.diff(np.log(p_slice))
        dz = np.diff(Z[int(lfc_index):before_mid_idx+1])
            # Average between consecutive levels IN THE SLICE
            #T_v_parcel_avg = 0.5 * g * (T_v_parcel_slice[:-1] + T_v_parcel_slice[1:])
            #T_v_env_avg = 0.5 * (T_v_env_slice[:-1] + T_v_env_slice[1:])
            
            # Buoyancy calculation
            #buoyancy = 0.5* g *R_dry * (T_v_parcel_avg - T_v_env_avg)
        buoyancy = g *(T_v_moist[int(lfc_index):before_mid_idx+1] - T_v_env[int(lfc_index):before_mid_idx+1])/T_v_env[int(lfc_index):before_mid_idx+1]
        if len(buoyancy) < 2 or len(dz) < 2:
            print("Warning: Not enough buoyancy levels to compute average – skipping.")
            w_parcel_max = 0
            CAPE = 0

        
        #if before_mid_idx <= int(lfc_index):
            #print("Warning: before_mid_idx is not greater than lfc_index — skipping CAPE calculation.")
            #w_parcel_max = 0
            #CAPE = 0
        else:
            buoyancy_mid = g *(T_v_parcel_mid - T_v_env_mid)/T_v_env_mid
            buoyancy_avg = 0.5*(buoyancy[:-1] + buoyancy[1:])  #averaging the buoyancy
            
            CAPE = np.sum(buoyancy_avg * dz) + 0.5 * (buoyancy_mid + buoyancy[-1]) * (Z_mid - Z[before_mid_idx])
            w_parcel_max = np.sqrt(2 * CAPE)
    else:
        w_parcel_max = 0
        buoyancy = 0
        CAPE = 0
    return w_parcel_max,CAPE#,T_v_parcel_slice
"""
def calculate_parcel_updraft_velocity(p,T_v_moist,T_v_env,lfc_index,p_t,top_index):

    """
    the scheme for calculating the maximum updraft velocity (Emanuel 1991)

    Parameters:
        p: pressure profile  [Pa]"
        T_v_moist: parcel virtual temperature profile [K]"
        T_env: environment virtual temperature profile  [K]"
        lfc_index: index of the LFC
        top_index: index of the top of the cloud

    Returns:
        w_parcel_max: maximum updraft velocity [m/s]
        CAPE: convective available potential energy
    """
    
    if top_index>1 and top_index>int(lfc_index):
    #implementation of the CAPE equation and integration
        delta_ln_p = -np.diff(np.log(p[int(lfc_index):top_index+1]))
        print(delta_ln_p)
        buoyancy = R_dry * (T_v_moist - T_v_env)[int(lfc_index):top_index]
        CAPE = np.sum(buoyancy * delta_ln_p)
        w_parcel_max = np.sqrt(2 * CAPE)
    else:
        w_parcel_max = 0
        CAPE = 0
    return w_parcel_max,CAPE

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
