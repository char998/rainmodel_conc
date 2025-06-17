import numpy as np
from constants import R_dry

def w_parcel(p,T_v_moist,T_v_env,lfc_index,p_t,top_index):

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
