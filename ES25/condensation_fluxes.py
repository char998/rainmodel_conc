from constants import R_dry
from physics_equations import w_sat
import numpy as np


def rho_m(T_s,T_t,p_s,p_t):
    """
    average density inside the cloud

    Parameters:
        T_s: cloud base temperature [K]
        T_t: cloud top temperature [K]
        p_s: cloud base pressure [Pa]
        p_t: cloud top pressure [Pa]

    Returns:        
        rho_m: average density inside the cloud [kg/m3]
    """

    rho_m = (p_s/(R_dry*T_s) + p_t/(R_dry*T_t))/2
    return rho_m

def condensation_flux(rho,v,q_l_parcel,lcl_index,top_index,theta_type):
    """
    function to calculate the accumulated liquid water flux inside the cloud

    Parameters:
        rho: average density inside the cloud [kg/m3]
        v: average velocity inside the cloud [m/s]
        lcl_index: index of the LCL
        top_index: index of the cloud top
        theta_type: choose which approach to follow for 
                    the conserved potential temperature: 
                                                       - theta_e: pseudo-adiabatic equivalent potential temperature
                                                       - theta_e_reversible: reversible equivalent potential temperature
                                                       - theta_l: liquid water potential temperature    
                                                       - GB84: GB84 equivalent potential temperature
    Returns:        
        f: flux of liquid water inside the cloud [kg/m2/s]
    
    """
    if v == 0:
        return 0
    else:
        if theta_type == "GB84" or theta_type == "theta_e":         #in pseudoadiabatic approach water is constantly removed, 
                                                                    #so integrate over all cloud heights
            return rho*v*(np.sum(q_l_parcel[lcl_index:top_index+1]))
        else:                                                       #in reversible approach water is staying inside the parcel,
                                                                    #so total water is what reaches the cloud top
            return rho*v*(q_l_parcel[top_index])

