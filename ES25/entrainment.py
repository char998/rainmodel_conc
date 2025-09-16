import numpy as np
from physics_equations import w_sat,Theta_air,theta_entrain,theta_potential

def entrainment_theta(entrainment_rate,theta_conserved,Z,theta_air,start_index):

    """
    Calculate the potential temperature profile change due to entrainment

    Parameters:
        entrainment_rate: fractional entrainment rate     [1/m]
        theta_conserved: conserved parcel potential temperature [K]
        Z: height levels                          [m]
        theta_air: ambient air potential temperature profile  [K]
        start_index: index of first height that is subject to enterainment

    Returns:
        theta_t: potential temperature profile with entrainment [K]
    """

    theta_t =  np.zeros(len(Z))
    dZ = np.diff(Z)
    theta_t[0:start_index] = theta_conserved
    for i in range(start_index, len(theta_t)):
        theta_t[i] = theta_t[i-1] - entrainment_rate * (theta_t[i-1] - theta_air[i-1]) * dZ[i-1]

    return theta_t

def entrainment_q_t(entrainment_rate, q_initial, Z, q_air,start_index):
    """
    Calculate the total humidity profile change due to entrainment (only for reversible approaches)

    Parameters:
        entrainment_rate: fractional entrainment rate       [1/m]
        q_initial: initial/conserved parcel total humidity [kg/kg]
        Z: height levels                          [m]
        q_air: ambient air humidity profile  [kg/kg]
        start_index: index of first height that is subject to enterainment

    Returns:
        q_t_new: parcel's total humidity profile after entrainment [kg/kg]
    """
    q_t_new = np.zeros(len(Z) - start_index)
    q_t_updated = np.zeros(len(Z) - start_index)
    q_t_new[0] = q_initial
    q_t_updated[0] = q_initial
    for i in range(1, len(q_t_new)):
        dZ = Z[start_index + i] - Z[start_index + i - 1]
        q_t_new[i] = max(0, q_t_updated[i-1] - entrainment_rate * (q_t_updated[i-1] - q_air[start_index + i - 1]) * dZ)
        q_t_updated[i] = q_t_new[i]
    return q_t_new

def solve_entrained_parcel_state_iterative(theta_target, q_t_total, p, p_ref, theta_type, T_initial_guess, 
                                       max_iter=50, tolerance=0.01, debug=False):
    """
    Solve for the temperature and partitioning of total water (vapor and liquid) in a rising parcel 
    under the influence of entrainment, using an iterative approach with optional debug logging.

    Parameters:
        theta_target : Target potential temperature of the parcel [K].
        q_t_total : Total water content of the parcel (vapor + liquid) [kg/kg].
        p : Ambient pressure at the parcel level [Pa].
        p_ref : Reference pressure for potential temperature calculation [Pa].
        theta_type : choose which approach to follow for calculating potential temperature: 
                    - theta_e: pseudo-adiabatic equivalent potential temperature with constant latent heat
                    - theta_e_reversible: reversible equivalent potential temperature
                    - theta_l: liquid water potential temperature
                    - GB84: pseudo-adiabatic equivalent potential temperature from Georgakakos and Bras 1984

        T_initial_guess : Initial guess for parcel temperature [K] to start the iteration.
        max_iter : Maximum number of iterations for convergence (default: 50) (optional)
        tolerance : Convergence criterion for the difference between calculated and target potential temperature (default: 0.01 K).
                    (optional)
        debug : If True, prints detailed iteration information for debugging (default: False). (optional)

    Returns:
        T_solution : Converged single level parcel temperature [K].
        q_v_solution : Water vapor mixing ratio of the parcel at the converged temperature [kg/kg].
        q_l_solution : Cloud liquid water mixing ratio of the parcel at the converged temperature [kg/kg].

    Notes:
        - The function partitions total water into vapor and liquid depending on saturation.
        - The iterative process adjusts the parcel temperature starting from T_initial_guess:
            1. Compute the saturation mixing ratio (w_sat) at the current temperature.
            2. Partition the total water into vapor (up to w_sat) and liquid (excess above w_sat).
            3. Compute the parcel potential temperature based on the current temperature and water partitioning.
            4. Compare the calculated potential temperature with theta_target.
            5. Adjust the temperature using a combination of proportional adjustment and secant method.
            6. Repeat steps 1-5 until the potential temperature converges within the specified tolerance or max_iter is reached.
        - Iterative approach uses a combination of proportional adjustment and secant method for estimation of initial value preturbation
          and to ensure robust convergence.
        - Optional debug mode provides detailed iteration information including temperature, theta, error, 
          vapor, liquid water, and saturation status.
        - If convergence is not reached within max_iter iterations, the function returns the best estimate.
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
def calculate_profiles_with_entrainment(T_s, T_d, T_air, T_parcel, p_s, p_air, q_air, Z, start_index, entrainment_rate, theta_type):
    """
    Calculate parcel thermodynamic profiles including entrainment effects.

    This function evolves a rising parcel through a vertical profile, accounting
    for entrainment of environmental air and solving for the partitioning of water
    between vapor and liquid. The treatment depends on the chosen potential
    temperature definition (reversible or pseudo-adiabatic).

    Parameters
    ----------
    T_s :  Surface (initial) parcel temperature [K].
    T_d : Surface dewpoint temperature [K].
    T_air : Environmental temperature profile [K].
    T_parcel : Initial parcel temperature profile [K].
    p_s : Surface pressure [Pa].
    p_air : Environmental pressure profile [Pa].
    q_air : Environmental specific humidity profile [kg/kg].
    start_index : Index where entrainment begins (LCL).
    Z : Height levels [m].
    start_index : index of first height that is subject to enterainment
    entrainment_rate : fractional entrainment rate [1/m].
    theta_type : choose which approach to follow for calculating potential temperature: 
                    - theta_e: pseudo-adiabatic equivalent potential temperature with constant latent heat
                    - theta_e_reversible: reversible equivalent potential temperature
                    - theta_l: liquid water potential temperature
                    - GB84: pseudo-adiabatic equivalent potential temperature from Georgakakos and Bras 1984

    Returns
    -------
    T_parcel_new : ndarray
        Updated parcel temperature profile [K].
    q_t_parcel_new : ndarray
        Parcel total water content profile [kg/kg].
    q_l_parcel_new : ndarray
        Parcel liquid water profile [kg/kg].
    q_v_parcel_new : ndarray
        Parcel vapor profile [kg/kg].
    Theta_new : ndarray
        Parcel potential temperature profile [K].

    Method
    ------
    - Initialize parcel total water content from surface dewpoint.
    - For reversible formulations ('theta_l', 'theta_e_reversible'):
        * Apply entrainment directly to the conserved total water content profile.
    - For pseudo-adiabatic formulations ('theta_e', 'GB84'):
        * Apply entrainment incrementally at each level, with condensed liquid
          removed instantly so only vapor is retained.
    - At each level above the start index:
        * Solve iteratively for parcel temperature, vapor, and liquid water
          that satisfy both (a) the entrained potential temperature and
          (b) the entrained total water content.
        * This uses `solve_entrained_parcel_state_iterative`
    - Apply safety checks for unphysical states (temperature bounds, large jumps).
    """
    q_t_parcel = w_sat(T_d, p_air[0])        # initial moisture content of parcel
    q_t_parcel_new = q_t_parcel * np.ones(len(Z))
    q_v_parcel_new = np.zeros(len(Z))
    q_l_parcel_new = np.zeros(len(Z))
    n_levels = len(Z)
    
    # Apply pre-calculated entrainment to q_t for reversible processes only
    if theta_type == 'theta_l' or theta_type == 'theta_e_reversible':
        q_above_start = entrainment_q_t(entrainment_rate, q_t_parcel, Z, q_air, start_index)
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
            T_solution, q_v_solution, q_l_solution = solve_entrained_parcel_state_iterative(
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


#------------------LCL-------------------------------------------------------------------------
#only for when entrainment starts from the ground

def lcl_enrtainment_from_ground(T_parcel_new,p_air,T_d,Z):

    """
    Estimate the lifting condensation level (LCL) index after entrainment starts from the ground.

    Parameters:
    ----------
    T_parcel_new : Parcel temperature profile [K].
    p_air : Environmental pressure profile [Pa].
    T_d : Surface dewpoint temperature [K].
    Z : Height levels [m].

    Returns
    -------
    LCL_index : Index of the first grid level where the parcel reaches saturation.
        If saturation is never reached, returns len(Z).
    T_LCL : Interpolated parcel temperature at the LCL [K].
    p_LCL : Interpolated parcel pressure at the LCL [Pa].
    """

    w0 = w_sat(T_d,p_air[0])
    w_sat_profile = w_sat(T_parcel_new, p_air)  # compute saturation mixing ratio profile

    # find indices where w_sat >= w0
    indices = np.where(w_sat_profile <= w0)[0]

    if len(indices) == 0:
        print("Parcel never reaches saturation in this profile.")

        return len(Z), T_parcel_new[-1], p_air[-1]
    else:
        LCL_index = indices[0]       # first height where saturation occurs
        # Linear interpolation weights
        f = ( Z[LCL_index] - Z[LCL_index - 1]) / (Z[LCL_index + 1] - Z[LCL_index - 1])

        # Interpolate temperature and pressure
        T_LCL = T_parcel_new[LCL_index - 1] + f * (T_parcel_new[LCL_index] - T_parcel_new[LCL_index - 1])
        p_LCL = p_air[LCL_index - 1] + f * (p_air[LCL_index] - p_air[LCL_index - 1])

        return LCL_index,T_LCL,p_LCL