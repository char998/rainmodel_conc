import numpy as np
from constants import *
from sympy import exp,Piecewise, Pow

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

    return np.where(
            T >= 273.15,
            np.exp(34.494 - 4924.99 / (T_C + 237.1)) / (T_C + 105) ** 1.57,
            np.exp(43.494 - 6545.8 / (T_C + 278)) / (T_C + 868) ** 2
        )

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
#symbolic equations to be used in sympy numerical solvers

def e_s_symbolic(T):
    T_C = T - 273.15
    return Piecewise(
        (exp(34.494 - 4924.99 / (T_C + 237.1)) / Pow(T_C + 105, 1.57), T >= 273.15),
        (exp(43.494 - 6545.8 / (T_C + 278)) / Pow(T_C + 868, 2), True)
    )

def w_sat_symbolic(T, P):
    e_sat = e_s_symbolic(T)
    return 0.622 * e_sat / (P - e_sat)
