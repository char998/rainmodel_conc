import numpy as np

def entraintment_theta(entraintment_rate,theta_conserved,Z,theta_air):

    """
    Calculate the potential temperature profile change due to entrainment

    Parameters:
        entraintment_rate: entrainment rate       [1/m]
        theta_conserved: conserved parcel potential temperature [K]
        Z: height levels                          [m]
        theta_air: ambient air potential temperature profile  [K]

    Returns:
        theta_t: potential temperature profile with entrainment [K]
    """

    theta_t =  np.zeros(len(Z))
    dZ = np.diff(Z)
    theta_t[0] = theta_conserved
    for i in range(1, len(theta_t)):
        theta_t[i] = theta_t[i-1] - entraintment_rate * (theta_t[i-1] - theta_air[i-1]) * dZ[i-1]

    return theta_t

def entraintment_q(entraintment_rate,q_t,Z,q_air):

    """
    Calculate the specific humidity profile change due to entrainment

    Parameters:
        entraintment_rate: entrainment rate       [1/m]
        q_t: conserved parcel specific humidity       [kg/kg]
        Z: height levels                          [m]
        q_air: ambient air specific humidity profile  [kg/kg]

    Returns:
        q_t_new: specific humidity profile with entrainment [kg/kg]
    
    """
    
    q_t_new = q_t*np.ones(len(Z))
    dZ = np.diff(Z)
    for i in range(1, len(q_t_new)):
        q_t_new[i] = max(0,q_t_new[i-1] - entraintment_rate * (q_t_new[i-1] - q_air[i-1])*dZ[i-1])

    return q_t_new