import numpy as np
from constants import *
from sympy import exp,Piecewise, Pow

Lv0 = 2.501e6
#equations for saturation vapor pressure, latent heat of condensation and specific humidity

def L(T):
    """
    latent heat of condensation (from Georgakakos and Bras 1984a)
    
    Parameters:
        T: temperature [K]
    
    Returns:
        L: latent heat of condensation [J/kg]
    """
    return A - B*(T - 273.15)


def e_s(T):
    """
    function to calculate saturation vapor pressure
    from Huang 2018 (A Simple Accurate Formula for Calculating Saturation Vapor Pressure of Water and Ice)

    Parameters: 
        T: temperature [K]

    Returns:
        e_s: saturation vapor pressure [Pa]
    """
    T_C = T - 273.15  # convert to Celsius

    e_s_val = np.empty_like(T)

    # Mask for water phase
    mask_water = T >= 273.15
    T_C_water = T_C[mask_water]
    e_s_val[mask_water] = (
        np.exp(34.494 - 4924.99 / (T_C_water + 237.1)) / (T_C_water + 105) ** 1.57
    )

    # Mask for ice phase
    mask_ice = ~mask_water
    T_C_ice = T_C[mask_ice]
    e_s_val[mask_ice] = (
        np.exp(43.494 - 6545.8 / (T_C_ice + 278)) / (T_C_ice + 868) ** 2
    )

    return e_s_val

def w_sat(T,P):
    """
    function to calculate saturation specific humidity

    Parameters:     
        T: temperature [K]  
        P: pressure [Pa]

    Returns:                
        w_sat: saturation specific humidity [kg/kg]
    """
    
    e_sat = e_s(T)
    return 0.622*e_s(T)/(P- e_sat)

def specific_humidity_from_rh(T, RH, p):
    """
    function to calculate specific humidity from relative humidity

    Parameters:
        T: temperature profile [K]
        RH: relative humidity profile [0-1]
        p: pressure profile [Pa]

    Returns:
        q: specific humidity [kg/kg]

    """

    e_sat = e_s(T)  # in Pa
    e = RH * e_sat
    q = 0.622 * e / (p - 0.378 * e)
    return q
#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#potential temperatures

def theta_GB84(T, p, p_0,qt):
    """
    function to calculate the pseudoadiabatic equivalent potential temperature (from Georgakakos and Bras 1984)

    Parameters:
        T: temperature profile [K]
        p: pressure profile [Pa]
        p_0: surface pressure [Pa]
        T_d: dew point temperature [K]

    Returns:
        pseudoadiabatic equivalent potential temperature [K]
    """

    #qt = w_sat(T_d,p_0)
    qv = min(qt, w_sat(T, p))
    ql = max(0.0, qt - qv)
    theta = T * (p_0 / p)**(R_dry / cpd)
    return theta * np.exp(L(T) * qv / (cpd * T))

def theta_e(T, p, p_0,qt):
    """
    function to calculate the pseudoadiabatic equivalent potential temperature (from Bryan 2008)

    Parameters:
        T: temperature profile [K]
        p: pressure profile [Pa]
        p_0: surface pressure [Pa]
        T_d: dew point temperature [K]

    Returns:
        pseudoadiabatic equivalent potential temperature [K]
    """

    #qt = w_sat(T_d,p_0)
    qv = min(qt, w_sat(T, p))
    ql = max(0.0, qt - qv)
    theta = T * (p_0 / p)**0.286
    return theta * np.exp(Lv0 * qv / (cpd * T))

# Theta_l (reversible) -- Liquid water potential temperature


def theta_l(T, p, p_0, qt):
    """
    function to calculate the pseudoadiabatic equivalent potential temperature (from AMS glossary)

    Parameters:
        T: temperature profile [K]
        p: pressure profile [Pa]
        p_0: surface pressure [Pa]
        T_d: dew point temperature [K]

    Returns:
        pseudoadiabatic equivalent potential temperature [K]
    """

    #qt = w_sat(T_d,p_0)
    qv = np.minimum(qt, w_sat(T, p))
    ql = np.maximum(0.0, qt - qv)
    poisson = 0.2854*(1 - 0.24*qv)
    theta = T * (p_0 / p)**(poisson)
    return theta * ((epsilon + qv) / (epsilon + qt))**poisson * (qv/qt)**(-qt*R_v / (cpd + qt * cpv)) * np.exp(-L(T) * ql / ((cpd + qt * cpv) * T))


def theta_e_reversible(T, p, p_0,qt):
    """
    function to calculate the pseudoadiabatic equivalent potential temperature (from AMS glossary)

    Parameters:
        T: temperature profile [K]
        p: pressure profile [Pa]
        p_0: surface pressure [Pa]
        T_d: dew point temperature [K]

    Returns:
        pseudoadiabatic equivalent potential temperature [K]
    """

    #qt = w_sat(T_d,p_0)
    qv = min(qt, w_sat(T, p))  # Saturation adjustment
    ql = max(0.0, qt - qv)
    # Reversible equivalent temperature formula
    theta = T * (p_0 / p)**(R_dry / (cpd + qt * cl))
    return theta * np.exp(L(T) * qv / ((cpd + qt * cl) * T))

#symbolic equations to be used in sympy numerical solvers

def theta_potential(T, p, p_0, qt,theta_type):
    if theta_type == 'GB84':
        return theta_GB84(T, p, p_0, qt)
    elif theta_type == 'theta_e':
        return theta_e(T, p, p_0, qt)
    elif theta_type == 'theta_l':
        return theta_l(T, p, p_0, qt)
    elif theta_type == 'theta_e_reversible':
        return theta_e_reversible(T, p, p_0, qt)

def Theta_air(T_air, p_air, q_air, theta_type):
    if theta_type == 'theta_e':
        Theta_air = T_air * (p_air[0] / p_air) ** 0.286 * np.exp(Lv0 * q_air / (cpd * T_air))

    elif theta_type == 'theta_l':
        Theta_air = T_air * (p_air[0] / p_air) ** 0.286     #no liquid water, θ_l = θ
   
    elif theta_type == 'theta_e_reversible':
        Theta_air = T_air * (p_air[0] / p_air) ** 0.286 * np.exp(L(T_air) * q_air / ((cpd + q_air * cl) * T_air))

    elif theta_type == 'GB84':
        Theta_air = T_air * (p_air[0] / p_air) ** 0.286 * np.exp(L(T_air) * q_air / (cpd * T_air))
    
    else:
        raise ValueError(f"Unknown theta_type: {theta_type}")
    return Theta_air

def theta_entrain(T, p, p_0, qv, qt_entr, theta_type):
    theta = T * (p_0 / p)**0.286
    if theta_type== 'theta_e':
        theta_entrain = theta * np.exp(Lv0 * qv / (cpd * T))
    elif theta_type== 'theta_l':
        ql = max(0.0, qt_entr - qv)
        theta_entrain = theta * ((epsilon + qv) / (epsilon + qt_entr))**0.286 * (qv/qt_entr)**(-qt_entr*R_v / (cpd + qt_entr * cpv)) * np.exp(-L(T) * ql / ((cpd + qt_entr * cpv) * T))
    elif theta_type== 'theta_e_reversible':
        theta_entrain = theta * np.exp(L(T) * qv / ((cpd + qt_entr * cl) * T))
    elif theta_type== 'GB84':
        theta_entrain = theta * np.exp(L(T) * qv / (cpd * T))
    return theta_entrain

def e_s_symbolic(T):
    T_C = T - 273.15
    return Piecewise(
        (exp(34.494 - 4924.99 / (T_C + 237.1)) / Pow(T_C + 105, 1.57), T >= 273.15),
        (exp(43.494 - 6545.8 / (T_C + 278)) / Pow(T_C + 868, 2), True)
    )

def w_sat_symbolic(T, P):
    e_sat = e_s_symbolic(T)
    return 0.622 * e_sat / (P - e_sat)
