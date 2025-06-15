from constants import R_dry
from physics_equations import w_sat


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


# In[64]:


def f(T_d,p_0,p_t,T_t,rho,v):
    """
    function to calculate the flux of liquid water inside the cloud

    Parameters:
        T_d: dew point temperature in the surface  [K]
        p_0: surface pressure                  [Pa]
        p_t: cloud top pressure [Pa]
        T_t: cloud top temperature [K]
        rho: average density inside the cloud [kg/m3]
        v: average velocity inside the cloud [m/s]

    Returns:        
        f: flux of liquid water inside the cloud [kg/m2/s]
    
    """
    #specific humidity in the ground and cloud base
    w_0 = w_sat(T_d,p_0)
    w_s = w_sat(T_t,p_t)

    return (w_0 - w_s)*rho*v

