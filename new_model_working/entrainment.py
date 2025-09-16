import numpy as np
from physics_equations import w_sat

def entrainment_theta(entrainment_rate,theta_conserved,Z,theta_air,start_index):

    """
    Calculate the potential temperature profile change due to entrainment

    Parameters:
        entrainment_rate: entrainment rate       [1/m]
        theta_conserved: conserved parcel potential temperature [K]
        Z: height levels                          [m]
        theta_air: ambient air potential temperature profile  [K]

    Returns:
        theta_t: potential temperature profile with entrainment [K]
    """

    theta_t =  np.zeros(len(Z))
    dZ = np.diff(Z)
    theta_t[0:start_index] = theta_conserved
    for i in range(start_index, len(theta_t)):
        theta_t[i] = theta_t[i-1] - entrainment_rate * (theta_t[i-1] - theta_air[i-1]) * dZ[i-1]

    return theta_t

def entrainment_q_t(entrainment_rate, q_initial, Z, q_air,T_new,p_air, start_index,theta_type):
    q_t_new = np.zeros(len(Z) - start_index)
    q_t_updated = np.zeros(len(Z) - start_index)
    q_t_new[0] = q_initial
    q_t_updated[0] = q_initial
    for i in range(1, len(q_t_new)):
        dZ = Z[start_index + i] - Z[start_index + i - 1]
        q_t_new[i] = max(0, q_t_updated[i-1] - entrainment_rate * (q_t_updated[i-1] - q_air[start_index + i - 1]) * dZ)
        if theta_type == "theta_e" or theta_type == 'GB84':
            q_t_updated[i] = min(q_t_new[i], w_sat(T_new[start_index + i], p_air[start_index + i]))
        else:
            q_t_updated[i] = q_t_new[i]
    return q_t_new

def lcl_enrtainment_from_ground(T_parcel_new,p_air,T_d,Z):

    w0 = w_sat(T_d,p_air[0])
    w_sat_profile = w_sat(T_parcel_new, p_air)  # compute saturation mixing ratio profile

    # find indices where w_sat >= w0
    indices = np.where(w_sat_profile <= w0)[0]

    if len(indices) == 0:
        print("Parcel never reaches saturation in this profile.")
        return len(Z), T_parcel_new[-1], p_air[-1]
    else:
        LCL_index = indices[0]       # first height where saturation occurs
        # Linear interpolation weights
        f = ( Z[LCL_index] - Z[LCL_index - 1]) / (Z[LCL_index + 1] - Z[LCL_index - 1])

        # Interpolate temperature and pressure
        T_LCL = T_parcel_new[LCL_index - 1] + f * (T_parcel_new[LCL_index] - T_parcel_new[LCL_index - 1])
        p_LCL = p_air[LCL_index - 1] + f * (p_air[LCL_index] - p_air[LCL_index - 1])
        return LCL_index,T_LCL,p_LCL