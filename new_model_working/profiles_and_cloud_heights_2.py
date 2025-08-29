import numpy as np
from constants import cpd,g,R_dry,cl,Lv0
from physics_equations import w_sat,theta_potential,theta_entrain,Theta_air
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
    T_arr[lcl_index] = T_s
    
    #Saturated: follow theta_e, theta_l, or theta_e_reversible
    for i in range(lcl_index+1, len(p_air)):
        
        
        def f(T_new):
            return theta_potential(T_new, p_air[i], p_air[0], q_t,theta_type) - theta_potential(T_s, p_s, p_air[0], q_t,theta_type)
                
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


def solve_entrained_parcel_state_robust(theta_target, q_t_total, p, p_ref, theta_type, T_initial_guess):
    """
    More robust solver using scipy.optimize as backup
    """
    
    # First try the custom iterative method
    try:
        T_sol, q_v_sol, q_l_sol = solve_entrained_parcel_state_debug(
            theta_target, q_t_total, p, p_ref, theta_type, T_initial_guess, 
            max_iter=30, tolerance=0.01, debug=False
        )
        
        # Verify the solution
        w_sat_val = w_sat(T_sol, p)
        q_v_check = min(q_t_total, w_sat_val)
        theta_check = theta_entrain(T_sol, p, p_ref, q_v_check, q_t_total, theta_type)
        
        if abs(theta_check - theta_target) < 0.1:  # Good enough
            return T_sol, q_v_sol, q_l_sol
    except:
        pass
    
    print(f"⚠️ Falling back to scipy solver...")
    
    # Fallback to scipy
    from scipy.optimize import brentq, minimize_scalar
    
    def theta_error(T):
        T = max(150.0, min(400.0, T))  # bounds
        try:
            w_sat_val = w_sat(T, p)
            q_v = min(q_t_total, w_sat_val)
            theta_calc = theta_entrain(T, p, p_ref, q_v, q_t_total, theta_type)
            return theta_calc - theta_target
        except:
            return 1e6  # large error if calculation fails
    
    try:
        # Try Brent's method if we can find a bracketing interval
        T_low, T_high = T_initial_guess - 50, T_initial_guess + 50
        T_low = max(150.0, T_low)
        T_high = min(400.0, T_high)
        
        error_low = theta_error(T_low)
        error_high = theta_error(T_high)
        
        if error_low * error_high < 0:  # Different signs - can use brentq
            T_solution = brentq(theta_error, T_low, T_high, xtol=0.01)
        else:
            # Use minimize_scalar
            result = minimize_scalar(lambda T: abs(theta_error(T)), 
                                   bounds=(150.0, 400.0), method='bounded')
            T_solution = result.x
            
        # Calculate final water partitioning
        w_sat_val = w_sat(T_solution, p)
        if q_t_total > w_sat_val:
            q_v = w_sat_val
            q_l = q_t_total - w_sat_val
        else:
            q_v = q_t_total
            q_l = 0.0
            
        return T_solution, q_v, q_l
        
    except Exception as e:
        print(f"⚠️ Scipy solver also failed: {e}")
        # Last resort - return initial guess with reasonable partitioning
        w_sat_val = w_sat(T_initial_guess, p)
        if q_t_total > w_sat_val:
            q_v = w_sat_val
            q_l = q_t_total - w_sat_val
        else:
            q_v = q_t_total
            q_l = 0.0
        return T_initial_guess, q_v, q_l
    
def solve_entrained_parcel_state_debug(theta_target, q_t_total, p, p_ref, theta_type, T_initial_guess, 
                                       max_iter=50, tolerance=0.01, debug=False):
    """
    Debug version with detailed logging and improved convergence
    """
    
    T_guess = T_initial_guess
    
    if debug:
        print(f"\n=== Solving for theta_target={theta_target:.2f} K, q_t={q_t_total*1000:.3f} g/kg, p={p/100:.1f} hPa ===")
    
    # Store previous values for secant method
    T_prev = None
    theta_prev = None
    
    for iteration in range(max_iter):
        # Calculate saturation mixing ratio at current temperature
        try:
            w_sat_val = w_sat(T_guess, p)
        except:
            print(f"⚠️ Error calculating w_sat at T={T_guess:.2f} K, p={p:.0f} Pa")
            return T_guess, 0.0, 0.0
        
        # Partition total water between vapor and liquid
        if q_t_total > w_sat_val:
            # Saturated - some condensation has occurred
            q_v = w_sat_val  # vapor is at saturation
            q_l = q_t_total - w_sat_val  # excess becomes liquid
            saturation_status = "SATURATED"
        else:
            # Unsaturated - all water is vapor
            q_v = q_t_total
            q_l = 0.0
            saturation_status = "UNSATURATED"
        
        # Calculate potential temperature with current T and water partitioning
        try:
            theta_calc = theta_entrain(T_guess, p, p_ref, q_v, q_t_total, theta_type)
        except Exception as e:
            print(f"⚠️ Error calculating theta_entrain: {e}")
            print(f"   T={T_guess:.2f}, p={p:.0f}, q_v={q_v*1000:.3f}, q_t={q_t_total*1000:.3f}")
            return T_guess, q_v, q_l
        
        # Check for numerical issues
        if not np.isfinite(theta_calc):
            print(f"⚠️ Non-finite theta_calc = {theta_calc} at T={T_guess:.2f} K")
            return T_guess, q_v, q_l
        
        # Calculate error
        error = theta_calc - theta_target
        
        if debug and (iteration < 5 or iteration % 10 == 0):
            print(f"  Iter {iteration:2d}: T={T_guess:6.2f} K, theta={theta_calc:6.2f} K, error={error:6.3f} K, {saturation_status}")
            print(f"           q_v={q_v*1000:5.2f} g/kg, q_l={q_l*1000:5.2f} g/kg, w_sat={w_sat_val*1000:5.2f} g/kg")
        
        # Check convergence
        if abs(error) < tolerance:
            if debug:
                print(f"  ✅ CONVERGED in {iteration+1} iterations")
            return T_guess, q_v, q_l
        
        # Calculate temperature adjustment
        if iteration == 0:
            # First iteration - simple proportional adjustment
            # For most theta types, dtheta/dT is roughly theta/T
            dT = -error * T_guess / max(theta_calc, 100)  # avoid division by small numbers
            dT = np.clip(dT, -10.0, 10.0)  # limit first step
            
        elif T_prev is not None and theta_prev is not None:
            # Use secant method with previous point
            if abs(theta_calc - theta_prev) > 1e-10:
                dT_secant = (T_guess - T_prev) * error / (theta_calc - theta_prev)
                dT = -dT_secant
            else:
                # Fall back to finite difference
                dT = -error * 0.1
        else:
            # Use finite difference approximation
            dT_test = min(1.0, 0.01 * T_guess)  # 1% perturbation or 1K, whichever is smaller
            T_test = T_guess + dT_test
            
            try:
                w_sat_test = w_sat(T_test, p)
                if q_t_total > w_sat_test:
                    q_v_test = w_sat_test
                else:
                    q_v_test = q_t_total
                
                theta_test = theta_entrain(T_test, p, p_ref, q_v_test, q_t_total, theta_type)
                
                if abs(theta_test - theta_calc) > 1e-10:
                    d_theta_dT = (theta_test - theta_calc) / dT_test
                    dT = -error / d_theta_dT
                else:
                    dT = -error * 0.1
            except:
                dT = -error * 0.1
        
        # Store values for next iteration
        T_prev = T_guess
        theta_prev = theta_calc
        
        # Apply temperature adjustment with adaptive damping
        # Reduce damping as we get closer to solution
        damping = max(0.3, min(0.8, 1.0 / (1 + iteration/10)))
        dT = np.clip(dT, -5.0, 5.0)  # limit step size
        T_guess = T_guess + damping * dT
        
        # Keep temperature in reasonable bounds
        T_guess = np.clip(T_guess, 150.0, 400.0)
        
        # Check if we're oscillating
        if iteration > 10 and abs(error) > abs(theta_prev - theta_target):
            # Error is increasing - reduce step size
            T_guess = 0.5 * (T_guess + T_prev)
            if debug:
                print(f"  ⚠️ Error increasing, averaging with previous T")
    
    # If we reach here, convergence failed
    print(f"⚠️ Warning: solve_entrained_parcel_state did not converge after {max_iter} iterations")
    print(f"   Final: T={T_guess:.2f} K, theta={theta_calc:.2f} K, target={theta_target:.2f} K")
    print(f"   Final error: {error:.6f} K, tolerance: {tolerance:.6f} K")
    print(f"   q_t={q_t_total*1000:.3f} g/kg, p={p/100:.1f} hPa, theta_type={theta_type}")
    
    # Return best estimate
    w_sat_val = w_sat(T_guess, p)
    if q_t_total > w_sat_val:
        q_v = w_sat_val
        q_l = q_t_total - w_sat_val
    else:
        q_v = q_t_total
        q_l = 0.0
        
    return T_guess, q_v, q_l

# Fix 2: Corrected main function loop
def calculate_profiles_with_entrainment_fixed(T_s, T_d, T_air, T_parcel, p_s, p_air, q_air, Z, start_index, entrainment_rate, theta_type):
    
    q_t_parcel = w_sat(T_d, p_air[0])        # initial moisture content of parcel
    q_t_parcel_new = q_t_parcel * np.ones(len(Z))
    q_v_parcel_new = np.zeros(len(Z))
    q_l_parcel_new = np.zeros(len(Z))
    n_levels = len(Z)
    
    # Apply pre-calculated entrainment to q_t for reversible processes only
    if theta_type == 'theta_l' or theta_type == 'theta_e_reversible':
        q_above_start = entrainment_q_t(entrainment_rate, q_t_parcel, Z, q_air, T_parcel, p_air, start_index, theta_type)
        q_t_parcel_new[start_index:] = q_above_start

    T_parcel_new = T_parcel.copy()
    
    theta_air = Theta_air(T_air, p_air, q_air, theta_type)
    theta_conserved = theta_potential(T_s, p_s, p_air[0], q_t_parcel, theta_type)
    Theta_new = entrainment_theta(entrainment_rate, theta_conserved, Z, theta_air, start_index)

    for i in range(n_levels):
        if i < start_index:
            # Below entrainment start (below LCL) - no condensation, just vapor
            T_parcel_new[i] = T_parcel[i]
            q_v_parcel_new[i] = q_t_parcel_new[i]  # All water is vapor below LCL
            q_l_parcel_new[i] = 0.0
                
        else:
            # Above entrainment start (above LCL) - apply entrainment and solve iteratively
            
            # For pseudo-adiabatic: apply entrainment to q_t at each level since ql is removed at each level
            if theta_type == 'theta_e' or theta_type == 'GB84':
                if i == start_index:
                    # First entrainment level
                    dz = Z[i] - Z[i-1] if i > 0 else 0
                    q_t_parcel_new[i] = q_t_parcel_new[i-1] + entrainment_rate * dz * (q_air[i] - q_t_parcel_new[i-1])
                else:
                    # Subsequent levels: entrain environmental air into current q_t
                    dz = Z[i] - Z[i-1]
                    q_t_parcel_new[i] = q_t_parcel_new[i-1] + entrainment_rate * dz * (q_air[i] - q_t_parcel_new[i-1])
            
            # Solve for T, q_v, q_l that satisfy the entrained theta and q_t
            T_solution, q_v_solution, q_l_solution = solve_entrained_parcel_state_debug(
                theta_target=Theta_new[i],
                q_t_total=q_t_parcel_new[i],
                p=p_air[i],
                p_ref=p_air[0],
                theta_type=theta_type,
                T_initial_guess=T_parcel_new[i-1] if i > 0 else T_parcel[i]
            )
            
            # Store results
            T_parcel_new[i] = T_solution
            q_v_parcel_new[i] = q_v_solution
            q_l_parcel_new[i] = q_l_solution
            
            # For pseudo-adiabatic: liquid water is removed instantly
            # Update q_t to reflect the loss of condensed water
            if theta_type == 'theta_e' or theta_type == 'GB84':
                q_t_parcel_new[i] = q_v_solution  # Only vapor remains
            
            # Safety checks
            if not np.isfinite(T_parcel_new[i]) or T_parcel_new[i] > 400 or T_parcel_new[i] < 150:
                print(f"⚠️ Warning: Unphysical T_parcel = {T_parcel_new[i]:.2f} K at level {i}, p = {p_air[i]:.2f} Pa")
                T_parcel_new[i] = max(150.0, min(400.0, T_parcel_new[i]))
            
            if i > start_index and abs(T_parcel_new[i] - T_parcel_new[i-1]) > 20:
                print(f"⚠️ Large temperature jump: ΔT = {T_parcel_new[i] - T_parcel_new[i-1]:.2f} K at level {i}")

    return T_parcel_new, q_t_parcel_new, q_l_parcel_new, q_v_parcel_new,Theta_new

def solve_temperature_from_theta_qt(theta_target, q_t_total, p, p_ref, theta_type, T_initial_guess, max_iter=20, tolerance=0.01):
    """
    Solve for temperature given theta and q_t using your existing functions
    """
    from scipy.optimize import fsolve
    
    def objective(T):
        # Use your existing theta_potential function 
        return theta_potential(T, p, p_ref, q_t_total, theta_type) - theta_target
    
    try:
        T_solution = fsolve(objective, T_initial_guess)[0]
        # Bounds checking
        T_solution = max(150.0, min(400.0, T_solution))
        return T_solution
    except:
        print(f"⚠️ fsolve failed, using iterative method")
        # Fallback to simple iteration
        T = T_initial_guess
        for iteration in range(max_iter):
            theta_calc = theta_potential(T, p, p_ref, q_t_total, theta_type)
            error = theta_calc - theta_target
            if abs(error) < tolerance:
                return T
            # Simple adjustment
            T = T - error * 0.1  # damped adjustment
            T = max(150.0, min(400.0, T))
        return T



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
            y_cin = g * (T_v_parcel[0:LFC_index] - T_v_env[0:LFC_index]) / T_v_parcel[0:LFC_index]
            Z_cin = Z[0:LFC_index]
            CIN = np.abs(np.trapz(y_cin, Z_cin))

            # --- Decision ---
            if CIN <= 200 & lcl_index <= len(Z)-1:
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
