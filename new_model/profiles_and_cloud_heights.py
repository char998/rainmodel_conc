import numpy as np
from constants import A,B,c_p,g,R_dry
from physics_equations import w_sat,specific_humidity_from_rh, w_sat_symbolic,L
from sympy import Symbol, Eq,exp, lambdify
from scipy.optimize import fsolve


def met_profiles(T_surf,p_surf,p_t):
    """
    function to calculate the temperature and pressure profiles from the dry adiabatic lapse rate

    Parameters:
        T_surf: surface temperature   [K]
        p_surf: surface pressure      [Pa]
        p_t: top pressure             [Pa]

    Returns:
        p_air: pressure profile       [Pa]
        T_air: temperature profile    [K]
    """
    p_air = np.linspace(p_surf, p_t, 1000)
    T_air = T_surf*(p_air/p_surf)**0.286
    return p_air, T_air
    
def Temp_prof(T_surf,p_surf,T_d,p_air):
    """
    function to calculate the temperature profile of the parcel given the pressure levels
    and using the irreversible process equivalent potential temperature

    Parameters:
        T_surf: surface temperature               [K]
        T_d: dew point temperature in the surface  [K]
        p_surf: surface pressure                  [Pa]
        p_air: pressure levels                    [Pa]
        T_arr: temperature profile                [K]

    Returns:
        T_arr: moist parcel temperature profile   [K]
        lcl_index: index of the LCL
        p_s: pressure at the LCL                  [Pa]
        T_s: temperature at the LCL               [K]
    """

    p_n = p_air[0]
    print(p_n)
    p_s = (1/((T_surf - T_d)/223.15 + 1))**3.5 * p_surf #pressure at the LCL
    #print(p_s)
    T_s = (1/((T_surf - T_d)/223.15 + 1))* T_surf       #Temperature at the LCL
    #print(T_s)

    theta_moist_parcel = T_s*(p_n/p_s)**0.286*np.exp(L(T_s)*w_sat(T_s,p_s)/(c_p*T_s))    #equivalent potential temperature inside the cloud (irreversible process)
    theta_dry = T_surf*(p_n/p_surf)**0.286               #equivalent potential temperature outside the cloud
    #print(theta_moist_parcel)

    #find where the LCL is located
    lcl_index = (np.abs(p_air - p_s)).argmin()


    T_arr = theta_dry*(p_air/p_n)**0.286*np.ones(len(p_air)) #dry adiabatic lapse rate temperature to use for initialization of the numerical solution
    #T_arr[lcl_index] = T_s

    for i in range(lcl_index,len(p_air)):
        #solve numerically the equivalent potential temperature equation to get the parcel temperature a each height
        T_parcel = Symbol('T_parcel')
        T_func = Eq(theta_moist_parcel,T_parcel * (p_n / p_air[i])**0.286 * exp((A - B * (T_parcel - 273.15)) * w_sat_symbolic(T_parcel,p_air[i]) / (c_p * T_parcel)))
        T_solve = lambdify(T_parcel, T_func.lhs - T_func.rhs, 'numpy')  #sympy equation
        T_parcel = fsolve(T_solve, T_arr[i-1])      #numerical solution (equation, initial guess)
        T_arr[i] = T_parcel     #parcel temperature at height with p_air[i]
        #print(T_parcel)

    return T_arr,lcl_index,p_s,T_s


def T_v(T_d,p_surf,p_air,T_arr,lcl_index):
    """
    function to calculate the virtual temperature of the parcel

    Parameters:
        T_d: dew point temperature in the surface  [K]
        p_surf: surface pressure                  [Pa]
        p_air: pressure levels                    [Pa]
        T_arr: temperature profile                [K]
    
    Returns:
        T_v_moist: parcel virtual temperature profile    [K]
        q_v: parcel vapor specific humidity profile       [kg/kg]
    """

    q_t = w_sat(T_d,p_surf)     #total humidity of parcel (conserved)
    q_v = np.zeros(len(T_arr))  #vapor specific humidity initialization
    q_l = np.zeros(len(T_arr))  #liquid specific humidity initialization
    q_v[0:lcl_index] = q_t*np.ones(lcl_index)   #vapor specific humidity below the LCL
    q_v[lcl_index:] = w_sat(T_arr[lcl_index:],p_air[lcl_index:]) #vapor specific humidity above the LCL
    q_l[0:lcl_index] = 0    #liquid specific humidity below the LCL
    q_l[lcl_index:] = q_t - q_v[lcl_index:] #liquid specific humidity above the LCL

    T_v_moist = T_arr*(1 +0.61*q_v)# - q_l)   #virtual temperature profile in !pseudo_adiabatic lapse rate!
                                              # for reversible processes, uncomment the -q_l term
    return T_v_moist,q_v

#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# FINDING THE LFC

#There are four possible cases:
    #1.The LFC is at the LCL
    #2.The LFC is above the LCL and the parcel has enough energy to reach it
    #3.The LFC is above the LCL and the parcel does not have enough energy to reach it
    #4.No LFC is found at the examined heights


def find_lfc(T_v_parcel,T_v_env,Z,lcl_index):
    """
    function to find he lfc height

    Parameters:
        T_v_parcel: parcel virtual temperature profile [K]
        T_v_env: environment virtual temperature profile  [K]
        Z: height profile  [m]

    Returns:
        lfc: the lfc height [m]
    """

    #case 1
    if T_v_parcel[lcl_index] > T_v_env[lcl_index]:
        return 'yes',lcl_index
    
    #case 2 and 3
    else:
        condition = T_v_parcel[lcl_index:]>T_v_env[lcl_index:]
        if np.any(condition):
            LFC_index = np.argmax(condition) + lcl_index
            #find the energy the parcel gains in the boundary layer
            BIN_lower = np.argmax(T_v_parcel>T_v_env)
            BIN_upper = np.argmax(T_v_parcel[BIN_lower:]<T_v_env[BIN_lower:]) + BIN_lower
            BIN = 0.0

            y_bin = g * (T_v_parcel[BIN_lower:BIN_upper] - T_v_env[BIN_lower:BIN_upper]) / T_v_parcel[BIN_lower:BIN_upper]
            BIN = np.trapz(y_bin, Z[BIN_lower:BIN_upper])
        
            #find how much energy needed to go above the CIN
            CIN = 0.0
            y_cin = g * (T_v_parcel[BIN_lower:BIN_upper] - T_v_env[BIN_lower:BIN_upper]) / T_v_parcel[BIN_lower:BIN_upper]
            CIN = abs(np.trapz(y_cin, Z[BIN_lower:BIN_upper]))

            #case 2
            if CIN <= BIN:
                 return 'yes',LFC_index
            #case 3
            else:
                return 'not reached',0
        
        #case 4
        else:
            return 'no LFC',0
        
#Finding the top of the cloud
def cloud_top(LFC_index,Z,Z_b,p,p_t,T_v_parcel,T_v_env,T_high,temp_parcel):
    """
    function to find the top of the cloud

    Parameters:
        LFC_index: index of the LFC
        Z: height profile  [m]
        Z_b: base of the cloud  [m]
        p: pressure profile  [Pa]
        p_t: top pressure - if there is no LNB [Pa]
        T_v_parcel: parcel virtual temperature profile [K]
        T_v_env: environment virtual temperature profile  [K]
        T_high: parcel temperature profile - if there is no LNB [K]
        temp_parcel: parcel temperature profile [K]

    Returns:
        Z_t - Z_b: cloud depth [m]
        Z_t: top of the cloud altitude [m]
        p_t: cloud top pressure [Pa]
    """

    condition = T_v_parcel[int(LFC_index[1]):]<T_v_env[int(LFC_index[1]):]
    if LFC_index[0] == 1.0:
        if np.any(condition):
            top_index = np.argmax(condition) + int(LFC_index[1])
            Z_t = Z[top_index-1] + (Z[top_index] - Z[top_index-1])/((T_v_parcel-T_v_env)[top_index-1] - (T_v_parcel-T_v_env)[top_index])*\
                                     ((T_v_parcel-T_v_env)[top_index-1])
            p_t = p[0]/np.exp(g*Z_t/R_dry/np.mean(temp_parcel[0:top_index+1]))
        else:
            Z_t = R_dry*np.mean(T_high)*np.log(p[0]/p_t)/g

         
    else:
            Z_t = 0 

    return max(1,Z_t - Z_b), Z_t, p_t
