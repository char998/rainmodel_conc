import numpy as np
from constants import cpd,g,R_dry
from physics_equations import w_sat,L,theta_e,theta_e_reversible,theta_l,theta_GB84
#from sympy import Symbol, Eq,exp, lambdify
from scipy.optimize import fsolve,brentq
from entrainment import entrainment_theta, entrainment_q_t

"""
def met_profiles(T_surf,p_surf,p_t):
    
    function to calculate the temperature and pressure profiles from the dry adiabatic lapse rate

    Parameters:
        T_surf: surface temperature   [K]
        p_surf: surface pressure      [Pa]
        p_t: top pressure             [Pa]

    Returns:
        p_air: pressure profile       [Pa]
        T_air: temperature profile    [K]
    
    p_air = np.linspace(p_surf, p_t, 1000)
    T_air = T_surf*(p_air/p_surf)**0.286
    return p_air, T_air
"""


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

    p_n = p_air[0]
    
    if theta_type == 'GB84':
        p_s = (1/((T_surf - T_d)/223.15 + 1))**3.5 * p_surf #pressure at the LCL
        T_s = (1/((T_surf - T_d)/223.15 + 1))* T_surf       #Temperature at the LCL
    else:
        def solve_for_ws(p):
            return w_sat(T_d,p_surf) - w_sat((p/p_surf)**0.286 * T_surf,p)
        p_min = 10000.0   # 100 hPa (Pa)
        p_max = p_surf
        p_s = brentq(solve_for_ws, p_min,p_max)
        T_s = (p_s/p_surf)**0.286 * T_surf


    #theta_potential_parcel = T_s*(p_n/p_s)**0.286
    #theta_moist_parcel = T_s*(p_n/p_s)**0.286*np.exp(L(T_s)*w_sat(T_s,p_s)/(cpd*T_s))    #equivalent potential temperature inside the cloud (irreversible process)
    theta_dry = T_surf*(p_n/p_surf)**0.286               #equivalent potential temperature outside the cloud

    #find where the LCL is located
    lcl_index = (np.abs(p_air - p_s)).argmin()

    T_arr = theta_dry*(p_air/p_n)**0.286*np.ones(len(p_air)) #dry adiabatic lapse rate temperature to use for initialization of the numerical solution
    q_t = w_sat(T_d,p_surf)
    #Saturated: follow theta_e, theta_l, or theta_e_reversible
    for i in range(lcl_index, len(p_air)):
        
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

        T_arr[i] = T_parcel

    # for i in range(lcl_index,len(p_air)):
    #     #solve numerically the equivalent potential temperature equation to get the parcel temperature a each height
    #     T_parcel = Symbol('T_parcel')
    #     T_func = Eq(theta_moist_parcel,T_parcel * (p_n / p_air[i])**0.286 * exp((A - B * (T_parcel - 273.15)) * w_sat_symbolic(T_parcel,p_air[i]) / (cpd * T_parcel)))
    #     T_solve = lambdify(T_parcel, T_func.lhs - T_func.rhs, 'numpy')  #sympy equation
    #     T_parcel = fsolve(T_solve, T_arr[i-1])      #numerical solution (equation, initial guess)
    #     T_arr[i] = T_parcel     #parcel temperature at height with p_air[i]

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

    q_t = w_sat(T_d,p_air[0])     #total humidity of parcel (conserved)
    q_v = np.zeros(len(T_arr))  #vapor specific humidity initialization
    q_l = np.zeros(len(T_arr))  #liquid specific humidity initialization
    q_v[0:lcl_index] = q_t*np.ones(lcl_index)   #vapor specific humidity below the LCL
    q_v[lcl_index:] = np.minimum(q_t,w_sat(T_arr[lcl_index:],p_air[lcl_index:])) #vapor specific humidity above the LCL
    q_l[0:lcl_index] = 0    #liquid specific humidity below the LCL
    q_l[lcl_index:] = q_t - q_v[lcl_index:] #liquid specific humidity above the LCL

    #warnings
    if np.any(q_v < 0) or np.any(q_v > q_t):
        print("⚠️ Warning: Unphysical values in vapor specific humidity (q_v) detected.")
    # Check liquid water content
    if np.any(q_l < -1e-6):  # allow small negative tolerance due to numerical errors
        print("⚠️ Warning: Negative values in liquid water mixing ratio (q_l).")


    if theta_type == 'theta_e_reversible' or theta_type == 'theta_l':
        T_v_moist = T_arr*(1 +0.61*q_v - q_l)   #virtual temperature profile in !for reversible processes!
                                              
    else:
        T_v_moist = T_arr*(1 +0.61*q_v) #virtual temperature profile in !pseudo_adiabatic lapse rate!

    #warnings
    if np.any(~np.isfinite(T_v_moist)) or np.any(T_v_moist < 150) or np.any(T_v_moist > 400):
        print("⚠️ Warning: Unphysical virtual temperatures detected.")

    return T_v_moist,q_v


def calculate_profiles_with_entrainment(T_s, T_d, T_air, T_parcel, p_s, p_air, q_air, Z, lcl_index, entrainment_rate, theta_type):
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

    #if theta_type == 'theta_e_reversible' or theta_type == 'theta_l':
        # Store original values below LCL
        #original_q_below_lcl = q_t_parcel_new[:lcl_index+1].copy()
        
        #if lcl_index + 1 < len(Z):
            #q_t_parcel_new[lcl_index+1:] = entrainment_q(entrainment_rate,q_t_parcel,Z[lcl_index+1:],q_air[lcl_index+1:])
        
        # Restore original values below LCL
        #q_t_parcel_new[:lcl_index+1] = original_q_below_lcl
    #if theta_type == 'theta_e_reversible' or theta_type == 'theta_l':
    if lcl_index + 1 < len(Z):
            # Pass the full arrays but specify the starting value correctly
            start_index = 1 #index of height from which entrainment starts
            q_above_lcl = entrainment_q_t(entrainment_rate, q_t_parcel, Z, q_air, T_parcel,p_air,start_index,theta_type)
            q_t_parcel_new[start_index:] = q_above_lcl

    T_parcel_new = T_parcel.copy()

    if theta_type == 'theta_e':
        Theta_air = T_air * (p_air / p_air[0]) ** 0.286 * np.exp(L(T_air) * q_air / (cpd * T_air))
        theta_conserved = theta_e(T_s, p_s, p_air[0], q_t_parcel)

    elif theta_type == 'theta_l':
        Theta_air = T_air * (p_air / p_air[0]) ** 0.286     #no liquid water, θ_l = θ
        theta_conserved = theta_l(T_s, p_s, p_air[0], q_t_parcel)
   
    elif theta_type == 'theta_e_reversible':
        Theta_air = T_air * (p_air / p_air[0]) ** 0.286 * np.exp(L(T_air) * q_air / (cpd * T_air))
        theta_conserved = theta_e_reversible(T_s, p_s, p_air[0], q_t_parcel)

    elif theta_type == 'GB84':
        Theta_air = T_air * (p_air / p_air[0]) ** 0.286 * np.exp(L(T_air) * q_air / (cpd * T_air))
        theta_conserved = T_s * (p_air[0] / p_s) ** 0.286 * np.exp(L(T_s) * w_sat(T_s, p_s) / (cpd * T_s))
    
    else:
        raise ValueError(f"Unknown theta_type: {theta_type}")
    
    #potential temperature of parcel after entrainment (not conserved anymore)
    Theta_new = entrainment_theta(entrainment_rate,theta_conserved,Z, 
                                       Theta_air)
    
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
        if i > lcl_index and abs(T_parcel_new[i] - T_parcel_new[i-1]) > 20:
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
        return "yes", lcl_index  # Case 1: Already buoyant at LCL

    else:
        condition = T_v_parcel[lcl_index:] > T_v_env[lcl_index:]
        if np.any(condition):
            LFC_index = np.argmax(condition) + lcl_index

            # --- CIN: from LCL to LFC where parcel is negatively buoyant ---
            #mask_cin = T_v_parcel[lcl_index:LFC_index] < T_v_env[lcl_index:LFC_index]
            #if np.any(mask_cin):
                #y_cin = g * (T_v_parcel[lcl_index:LFC_index][mask_cin] - T_v_env[lcl_index:LFC_index][mask_cin]) / T_v_parcel[lcl_index:LFC_index][mask_cin]
                #Z_cin = Z[lcl_index:LFC_index][mask_cin]
                #CIN = -np.trapz(y_cin, Z_cin)
            #else:
                #CIN = 0.0
            y_cin = g * (T_v_parcel[0:LFC_index] - T_v_env[0:LFC_index]) / T_v_parcel[0:LFC_index]
            Z_cin = Z[0:LFC_index]
            CIN = np.abs(np.trapz(y_cin, Z_cin))
            #--- BIN: from LFC upward until buoyancy turns negative ---
            BIN_lower = LFC_index
            condition_bin = T_v_parcel[BIN_lower:] > T_v_env[BIN_lower:]
            if np.any(condition_bin):
                BIN_upper = np.argmax(~condition_bin) + BIN_lower
            else:
               BIN_upper = len(Z)  # buoyancy remains positive until top

            y_bin = g * (T_v_parcel[BIN_lower:BIN_upper] - T_v_env[BIN_lower:BIN_upper]) / T_v_parcel[BIN_lower:BIN_upper]
            Z_bin = Z[BIN_lower:BIN_upper]
            BIN = np.trapz(y_bin, Z_bin)

            # --- Decision ---
            if CIN <= 200:
                return "yes", LFC_index  # Case 2: CIN is overcome
            else:
                return 'not reached', 0   # Case 3: CIN too strong
        else:
            return 'no LFC', 0  # Case 4: no crossing of parcel above environment
        
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
                                     ((T_v_parcel-T_v_env)[top_index-1])
            p_t = p[0]/np.exp(g*Z_t/R_dry/np.mean(temp_parcel[0:top_index+1]))
        else:
            Z_t = Z[-1] 
            #Z_t = R_dry*np.mean(T_high)*np.log(p[0]/p_t)/g
            p_t = p[0]/np.exp(g*Z_t/R_dry/np.mean(temp_parcel))
        
    else:
            Z_t = 0 
            p_t = 0
    Z_c = Z_t - Z_b
    if Z_c < 100:
        Z_c = 1.0
    return Z_c, Z_t, p_t
