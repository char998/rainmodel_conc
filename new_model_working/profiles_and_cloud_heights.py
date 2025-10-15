import numpy as np
from constants import cpd,g,R_dry,cl,Lv0
from physics_equations import w_sat,L,theta_e,theta_e_reversible,theta_l,theta_GB84
#from sympy import Symbol, Eq,exp, lambdify
from scipy.optimize import fsolve,brentq
from entrainment import entrainment_theta, entrainment_q_t


def calculate_parcel_Temp_profile(T_surf,p_surf,T_d,p_air,theta_type):
    """
    function to calculate the temperature profile of the parcel given the pressure levels
    and using the irreversible process equivalent potential temperature

    Parameters:
        T_surf: surface temperature               [K]
        T_d: dew point temperature in the surface  [K]
        p_surf: surface pressure                  [Pa]
        p_air: pressure levels                    [Pa]
        theta_type: choose which approach to follow for 
                    the conserved potential temperature: 
                    - theta_e: pseudo-adiabatic equivalent potential temperature with constant latent heat
                    - theta_e_reversible: reversible equivalent potential temperature
                    - theta_l: liquid water potential temperature
                    - GB84: pseudo-adiabatic equivalent potential temperature from Georgakakos and Bras 1984

    Returns:
        T_arr: moist parcel temperature profile   [K]
        lcl_index: index of the LCL
        p_s: pressure at the LCL                  [Pa]
        T_s: temperature at the LCL               [K]
    """
    
    #-----------------LCL-------------------------------------------------------------------------
    if theta_type == 'GB84':
        p_s = (1/((T_surf - T_d)/223.15 + 1))**3.5 * p_surf #pressure at the LCL
        T_s = (1/((T_surf - T_d)/223.15 + 1))* T_surf       #Temperature at the LCL


    else:
        if T_surf <= T_d:
            # Already saturated at the surface
            p_s = p_surf
            T_s = T_surf
            lcl_index = 0
        else:
            def solve_for_ws(p):
                """equation to find where w_sat = w0"""
                return w_sat(T_d,p_surf) - w_sat((p/p_surf)**0.286 * T_surf,p)
            p_min = np.min(p_air)   # 100 hPa (Pa) - minimum pressure
            p_max = p_surf
            p_s = brentq(solve_for_ws, p_min,p_max)  #pressure at the LCL
            T_s = (p_s/p_surf)**0.286 * T_surf       #Temperature at the LCL

    #find where the LCL is located
    lcl_index = (np.abs(p_air - p_s)).argmin()
    #-----------------LCL-------------------------------------------------------------------------
    
    #total parcel humidity (conserved for reversible, not conserved for irreversible)
    q_t = w_sat(T_d,p_surf)

    #temperature due to dry adiabatic lapse rate
    T_arr = T_surf*(p_air/p_surf)**0.286*np.ones(len(p_air))
    #T_arr[lcl_index] = T_s

    #Saturated: follow theta_e, theta_l, or theta_e_reversible
    for i in range(lcl_index+1, len(p_air)):
        
        if theta_type == 'theta_e':
            def f(T_new):
                return theta_e(T_new, p_air[i], p_air[0], q_t) - theta_e(T_s, p_s, p_air[0], q_t)
            
        elif theta_type == 'theta_l':
            def f(T_new):
                
                return theta_l(T_new, p_air[i], p_air[0], q_t) - theta_l(T_s, p_s, p_air[0], q_t)
            
        elif theta_type == 'theta_e_reversible':
            def f(T_new):
                return theta_e_reversible(T_new, p_air[i], p_air[0], q_t) - theta_e_reversible(T_s, p_s, p_air[0], q_t)
            
        elif theta_type == 'GB84':
            def f(T_new):
                return theta_GB84(T_new, p_air[i], p_air[0], q_t) - theta_GB84(T_s, p_s, p_air[0], q_t)
            
        else:
            raise ValueError(f"Unknown theta_type: {theta_type}")

        
        T_parcel = fsolve(f, T_arr[i-1])[0]

        #warnings
        if not np.isfinite(T_parcel) or T_parcel < 150 or T_parcel > 400:
            print(f"⚠️ Warning: Unphysical T_parcel = {T_parcel:.2f} K at p = {p_air[i]:.2f} Pa")
        if i > lcl_index and abs(T_parcel - T_arr[i-1]) > 20:
            print(f"⚠️ Large temperature jump detected: ΔT = {T_parcel - T_arr[i-1]:.2f} K at p = {p_air[i]:.2f} Pa")

        #temperature at pressure level p_air[i]
        T_arr[i] = T_parcel

    return T_arr,lcl_index,p_s,T_s

def calculate_parcel_T_v_profile(T_d,p_air,T_arr,lcl_index,theta_type):
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

    q_t = w_sat(T_d,p_air[0])     #total humidity of parcel (conserved for reversible, not conserved for irreversible above lcl)
    q_v = np.zeros(len(T_arr))  #vapor specific humidity initialization
    q_l = np.zeros(len(T_arr))  #liquid specific humidity initialization

    q_v[0:lcl_index+1] = q_t   #vapor specific humidity below the LCL
    q_l[0:lcl_index] = 0    #liquid specific humidity below the LCL

    if theta_type == 'theta_l' or theta_type == 'theta_e_reversible':
        q_v[lcl_index+1:] = np.minimum(q_t,w_sat(T_arr[lcl_index+1:],p_air[lcl_index+1:])) #vapor specific humidity above the LCL
    else:
        q_v[lcl_index+1:] = np.minimum(w_sat(T_arr[lcl_index:-1],p_air[lcl_index:-1]),w_sat(T_arr[lcl_index+1:],p_air[lcl_index+1:]))

    
    q_l[lcl_index:] = q_t - q_v[lcl_index:] #liquid specific humidity above the LCL

    #warnings
    if np.any(q_v < 0) or np.any(q_v - q_t > 1e-6):
        print("⚠️ Warning: Unphysical values in vapor specific humidity (q_v) detected.")
    # Check liquid water content
    if np.any(q_l < -1e-6):  # allow small negative tolerance due to numerical errors
        print("⚠️ Warning: Negative values in liquid water mixing ratio (q_l).")


    if theta_type == 'theta_e_reversible' or theta_type == 'theta_l':
        T_v_moist = T_arr*(1 +0.61*q_v - q_l)   #virtual temperature profile in !reversible processes!
                                              
    else:
        T_v_moist = T_arr*(1 +0.61*q_v) #virtual temperature profile in !pseudo_adiabatic lapse rate!

    #warnings
    if np.any(~np.isfinite(T_v_moist)) or np.any(T_v_moist < 150) or np.any(T_v_moist > 400):
        print("⚠️ Warning: Unphysical virtual temperatures detected.")

    return T_v_moist,q_v


def calculate_profiles_with_entrainment(T_s, T_d, T_air, T_parcel, p_s, p_air, q_air, Z, start_index, entrainment_rate, theta_type):
    """
    Calculate the moist parcel temperature profile and total humidity by considering entrainment

    Parameters:
        T_s: temperature at the LCL               [K]
        T_d: dew point temperature in the surface  [K]
        T_air: air temperature profile                [K]
        T_parcel: parcel temperature profile      [K]
        p_s: pressure at the LCL                  [Pa]
        p_air: pressure levels                    [Pa]
        q_air: environmental vapor specific humidity profile    [kg/kg]
        Z: height levels                          [m]
        lfc_index: index of the LFC
        entrainment_rate: entrainment rate       [1/m]
        theta_type: choose which approach to follow for 
                    the conserved potential temperature: 
                                                       - theta_e: pseudo-adiabatic equivalent potential temperature
                                                       - theta_e_reversible: reversible equivalent potential temperature
                                                       - theta_l: liquid water potential temperature    
                                                       - GB84: GB84 equivalent potential temperature

    Returns:    
        T_parcel_new: parcel temperature profile with entrainment [K]
        q_t_parcel_new: total specific humidity profile with entrainment [kg/kg]
    """

    q_t_parcel = w_sat(T_d,p_air[0])        #initial moisture content of parcel
    q_t_parcel_new = q_t_parcel*np.ones(len(Z))

    
    # Pass the full arrays but specify the starting value correctly
    q_above_start = entrainment_q_t(entrainment_rate, q_t_parcel, Z, q_air, T_parcel,p_air,start_index,theta_type)
    q_t_parcel_new[start_index:] = q_above_start

    T_parcel_new = T_parcel.copy()

    if theta_type == 'theta_e':
        Theta_air = T_air * (p_air / p_air[0]) ** 0.286 * np.exp(Lv0 * q_air / (cpd * T_air))
        theta_conserved = theta_e(T_s, p_s, p_air[0], q_t_parcel)

    elif theta_type == 'theta_l':
        Theta_air = T_air * (p_air / p_air[0]) ** 0.286     #no liquid water, θ_l = θ
        theta_conserved = theta_l(T_s, p_s, p_air[0], q_t_parcel)
   
    elif theta_type == 'theta_e_reversible':
        Theta_air = T_air * (p_air / p_air[0]) ** 0.286 * np.exp(L(T_air) * q_air / ((cpd + q_air * cl) * T_air))
        theta_conserved = theta_e_reversible(T_s, p_s, p_air[0], q_t_parcel)

    elif theta_type == 'GB84':
        Theta_air = T_air * (p_air / p_air[0]) ** 0.286 * np.exp(L(T_air) * q_air / (cpd * T_air))
        theta_conserved = theta_GB84(T_s, p_s, p_air[0], q_t_parcel)
    
    else:
        raise ValueError(f"Unknown theta_type: {theta_type}")
    
    #potential temperature of parcel after entrainment (not conserved anymore)
    #Theta_new = np.zeros(len(Z))
    #Theta_new[lcl_index+1:] = entrainment_theta(
        #entrainment_rate, theta_conserved, Z[lcl_index+1:], Theta_air[lcl_index+1:]
    #)
    #Theta_new[:lcl_index+1] = theta_conserved
    Theta_new= entrainment_theta(
        entrainment_rate, theta_conserved, Z, Theta_air,start_index)
    for i in range(1, len(p_air)):


        if theta_type == 'theta_e':
            def f(T_new):
                return theta_e(T_new, p_air[i], p_air[0], q_t_parcel_new[i]) - Theta_new[i]

        elif theta_type == 'theta_l':
            def f(T_new):
                return theta_l(T_new, p_air[i], p_air[0], q_t_parcel_new[i]) - Theta_new[i]

        elif theta_type == 'theta_e_reversible':

            def f(T_new):
                return theta_e_reversible(T_new, p_air[i], p_air[0], q_t_parcel_new[i]) - Theta_new[i]

        elif theta_type == 'GB84':
            def f(T_new):
                return theta_e_reversible(T_new, p_air[i], p_air[0], T_d) - Theta_new[i]

        T_parcel_new[i] = max(273.15-111,fsolve(f, T_parcel[i])[0])
             #capping the value for numerical stability
        #warnings
        if not np.isfinite(T_parcel_new[i]) or T_parcel_new[i] > 400:
            print(f"⚠️ Warning: Unphysical T_parcel = {T_parcel_new[i]:.2f} K at p = {p_air[i]:.2f} Pa")
        if i > start_index and abs(T_parcel_new[i] - T_parcel_new[i-1]) > 20:
            print(f"⚠️ Large temperature jump detected: ΔT = {T_parcel_new[i] - T_parcel_new[i-1]:.2f} K at p = {p_air[i]:.2f} Pa")

    return T_parcel_new,q_t_parcel_new





def calculate_humidity_profiles_with_entrainment(q_t_entr,p_air,T_arr,T_d,start_index):

    """
    Update the parcel humidity profiles after taking into account entrainment

    Parameters:
        T_d: dew point temperature in the surface  [K]
        p_surf: surface pressure                  [Pa]
        p_air: pressure levels                    [Pa]
        T_arr: parcel temperature profile                [K]
        q_t_entr: specific humidity profile with entrainment [kg/kg]

    Returns:
        q_l: liquid specific humidity profile       [kg/kg]
        q_v: vapor specific humidity profile        [kg/kg]
    """

    #q_t = np.zeros(len(T_arr))
    q_v = np.zeros(len(T_arr))
    q_l = np.zeros(len(T_arr))
    #q_t[0:lcl_index] = w_sat(T_d,p_surf)
    #q_t[lcl_index:] = q_t_entr[lcl_index:]
    q_v[0:start_index] =  w_sat(T_d,p_air[0])

    #if theta_type == 'theta_l' or theta_type == 'theta_e_reversible':
    q_v[start_index:] = np.minimum(q_t_entr[start_index:],w_sat(T_arr[start_index:],p_air[start_index:]))
    q_l = np.maximum(0, q_t_entr - q_v)
    #else:
        #q_v[lcl_index:] = entrainment_q_pseudo(entrainment_rate,T_arr[lcl_index:],p_air[lcl_index:],T_d,p_air[0],Z[lcl_index:],q_air[lcl_index:])[0]
        #q_l[lcl_index:] = entrainment_q_pseudo(entrainment_rate,T_arr[lcl_index:],p_air[lcl_index:],T_d,p_air[0],Z[lcl_index:],q_air[lcl_index:])[1]
    #warnings
    if np.any(q_v < 0):# or np.any(q_v > q_t):
        print("⚠️ Warning: Unphysical values in vapor specific humidity (q_v) detected.")

    return q_l,q_v

#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# FINDING THE LFC

#There are four possible cases:
    #1.The LFC is at the LCL
    #2.The LFC is above the LCL and the parcel has enough energy to reach it
    #3.The LFC is above the LCL and the parcel does not have enough energy to reach it
    #4.No LFC is found at the examined heights


def find_lfc_level(T_v_parcel,T_v_env,Z,lcl_index):
    """
    function to find he lfc height

    Parameters:
        T_v_parcel: parcel virtual temperature profile [K]
        T_v_env: environment virtual temperature profile  [K]
        Z: height profile  [m]

    Returns:
        lfc: the lfc height [m]
    """

    if T_v_parcel[lcl_index] > T_v_env[lcl_index]:
        CIN = 0
        return "yes", lcl_index, CIN  # Case 1: Already buoyant at LCL

    else:
        condition = T_v_parcel[lcl_index:] > T_v_env[lcl_index:]
        CIN = 0
        if np.any(condition):
            LFC_index = np.argmax(condition) + lcl_index

            # --- CIN: from LCL to LFC where parcel is negatively buoyant ---
            y_cin = g * (T_v_parcel[0:LFC_index] - T_v_env[0:LFC_index]) / T_v_env[0:LFC_index]
            y_cin = np.minimum(y_cin, 0)  # keep only negative buoyancy
            Z_cin = Z[0:LFC_index]
            CIN = np.abs(np.trapz(y_cin, Z_cin))
            
            # --- Decision ---
            if (CIN <= 200) and (lcl_index <= len(Z)-1):
                return "yes", LFC_index, CIN  # Case 2: CIN is overcome
            else:
                return 'not reached', 0, CIN   # Case 3: CIN too strong
        else:
            return 'no LFC', 0, CIN  # Case 4: no crossing of parcel above environment
        
#Finding the top of the cloud
def find_cloud_top_and_depth(LFC_index,Z,Z_b,p,T_v_parcel,T_v_env,temp_parcel):
    """
    function to find the top of the cloud

    Parameters:
        LFC_index: index of the LFC
        Z: height profile  [m]
        Z_b: base of the cloud  [m]
        p: pressure profile  [Pa]
        T_v_parcel: parcel virtual temperature profile [K]
        T_v_env: environment virtual temperature profile  [K]
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
                                     ((T_v_parcel-T_v_env)[top_index-1])            #interpolate to find the exact cloud top height
            p_t = p[0]/np.exp(g*Z_t/R_dry/np.mean(temp_parcel[0:top_index+1]))
        else:
            Z_t = Z[-1] 

            p_t = p[0]/np.exp(g*Z_t/R_dry/np.mean(temp_parcel))     #cloud top pressure from GB84
        
    else:
            Z_t = 0     #LFC was not reached or non-existent, no cloud top
            p_t = 0
    Z_c = Z_t - Z_b
    if Z_c < 100:
        Z_c = 1.0       # cloud depth is less than 100m - do not consider the system as cloud
    return Z_c, Z_t, p_t
