#code for a scheme for alternative updraft velocity calculation based on
# Emmanuel 1991 (A Scheme for Representing Cumulus Convection in Large-Scale Models)
# The calculation is simplified to find the maximum updraft velocity at the middle of cloud
# as defined by (Cloud top - Cloud Base)/2, and so the relation is integrated only once with bounds 
# cloud top and cloud base. The average updraft velocity then is the 1/2 max_updr_vel

import numpy as np
from scipy.optimize import fsolve
from sympy import Symbol, Eq,exp, sqrt, lambdify

R_d = 287.05
epsilon = 0.622 
A_1 = 8e-4  # (kg/(m·s²·K^3.5))
A = 2.5e6  # (J/kg)
B = 2.38e3  # (J/(kg K))
cp = 1001

def w(T,P):
    "mixing ratio"
    return epsilon*A_1*(T - 223.15)**3.5/P

def L(T):
    "latent heat of condensation"
    return A - B*(T - 273.15)

def met_profiles(T_surf,p_surf,p_t):
    #ambient air pressure and  temperature (from the conserved potential temperature)
    p_air = np.linspace(p_surf, p_t, 100)
    T_air = T_surf*(p_air/p_surf)**0.286
    return p_air, T_air

def Temp_prof(T_surf,p_surf,T_d,p_air,T_air):

    p_s = (1/((T_surf - T_d)/223.15 + 1))**3.5 * p_surf
    T_s = (1/((T_surf - T_d)/223.15 + 1))* T_surf
    theta_moist_parcel = T_s*(p_s/p_surf)**0.286*np.exp(L(T_s)*w(T_s,p_s)/(cp*T_s))

    #find where the LCL is located
    lcl_index = (np.abs(p_air - p_s)).argmin()
    T_arr = T_air*np.ones(len(T_air))

    #parcel temperature profile given by the numerical solution of the equivalent potential
    #temperature equation
    for i in range(lcl_index,len(p_air)):
        #print(i)
        T_parcel = Symbol('T_parcel')
        T_func = Eq(theta_moist_parcel,T_parcel * (p_surf / p_air[i])**0.286 * exp((A - B * (T_parcel - 273.15)) * (epsilon*A_1*abs(T_parcel - 223.15)**3.5/(p_air[i])) / (cp * T_parcel)))
        T_solve = lambdify(T_parcel, T_func.lhs - T_func.rhs, 'numpy')
        T_parcel = fsolve(T_solve, T_air[i])
        T_arr[i] = T_parcel #parcel temperature at height with p_air[i]

    return T_arr,lcl_index


def T_v(T_surf,p_surf,p_air,T_arr,lcl_index):

    #parcel virtual temperature calculation 
    q_t = w(T_surf,p_surf)  #total humidity of parcel (conserved)
    q_v = np.zeros(len(T_arr))  #vapor specific humidity
    q_l = np.zeros(len(T_arr))  #liquid specific humidity

    #calculate vapor and liquid specific humidity above the LCL
    q_v[0:lcl_index] = q_t*np.ones(lcl_index)
    q_v[lcl_index:] = w(T_arr[lcl_index:],p_air[lcl_index:])
    q_l[0:lcl_index] = 0
    q_l[lcl_index:] = q_t - q_v[lcl_index:]

    T_v_moist = T_arr*(1 +0.61*q_v - q_l)

    return T_v_moist

def w_parcel(p_t,p_b,T_v_moist,T_air,lcl_index):

    #implementation of the CAPE equation and integration
    diff_T= T_v_moist[lcl_index:] - T_air[lcl_index:]
    diff_T = diff_T[~np.isnan(diff_T)]
    CAPE = max(0.0001,R_d*np.mean(diff_T)*np.log(p_b/p_t))
    print(np.mean(diff_T))
    #print(CAPE)
    w_parcel_max = np.sqrt(2*CAPE)

    return w_parcel_max,CAPE