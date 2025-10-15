import numpy as np
from constants import R_dry, g

def interpolate_ambient_values(values, pressures, target_pressure):
    """
    Linear interpolation for target_pressure in a descending pressure profile.
    """
    # Find indices surrounding target_pressure
    # Handle edge cases first
    if target_pressure >= pressures[0]:
        return values[0]  # Above highest pressure (surface)
    if target_pressure <= pressures[-1]:
        return values[-1]  # Below lowest pressure (top)
    
    # Find the first pressure <= target_pressure
    indices_below_or_equal = np.where(pressures <= target_pressure)[0]
    if len(indices_below_or_equal) == 0:
        # All pressures are greater than target - shouldn't happen with edge case handling
        return values[-1]
    
    idx_above = indices_below_or_equal[0]  # first pressure <= target
    
    # Make sure we have a valid idx_below
    if idx_above == 0:
        # Target pressure is very close to surface pressure
        return values[0]
    
    idx_below = idx_above - 1
    
    # Linear interpolation
    value = values[idx_below] + (values[idx_above] - values[idx_below]) * \
            (target_pressure - pressures[idx_below]) / (pressures[idx_above] - pressures[idx_below])
    
    return value

def insert_lcl_values(temp_parcel, p, Z, lcl_index, T_s_value, p_s_value, T_air, q_v):
    """
    Insert LCL values into atmospheric profiles with thermodynamically consistent height calculation
    
    Parameters:
        temp_parcel: parcel temperature profile [K]
        p: pressure levels [Pa] - should be in descending order
        Z: height profile [m]
        lcl_index: index of the closest level to LCL
        T_s_value: temperature at the LCL [K]
        p_s_value: pressure at the LCL [Pa]
        T_air: ambient temperature profile [K]
        q_v: water vapor mixing ratio profile [kg/kg]
        T_surf: surface temperature [K] - needed for thermodynamic height calculation
        p_surf: surface pressure [Pa] - needed for thermodynamic height calculation
    
    Returns:
        Updated profiles with LCL values inserted
    """
    top_index = len(p) - 1
   
    if abs(p[lcl_index] - p_s_value) < 0.1:
        # LCL pressure matches existing level closely
        insert_idx = lcl_index
        Z_lcl = Z[lcl_index]
        T_air_lcl = T_air[lcl_index]
        qv_lcl = q_v[lcl_index]
        
    elif lcl_index == top_index:
        # LCL is above the top of our profile - append
        insert_idx = top_index + 1  # append
        
        # Calculate height using hypsometric equation from surface
        Z_lcl = (R_dry * T_air[0] / g) * np.log(p[0] / p_s_value)
            
        T_air_lcl = interpolate_ambient_values(T_air, p, p_s_value)
        qv_lcl = interpolate_ambient_values(q_v, p, p_s_value)
        
    elif (p[lcl_index] - p_s_value) > 0:
        # LCL pressure is below our closest pressure level
        # LCL height is above our closest height level
        insert_idx = lcl_index + 1
        
        # Calculate height using hypsometric equation from surface
        # This ensures thermodynamic consistency
        Z_lcl = (R_dry * T_air[0] / g) * np.log(p[0] / p_s_value)
        
        T_air_lcl = interpolate_ambient_values(T_air, p, p_s_value)
        qv_lcl = interpolate_ambient_values(q_v, p, p_s_value)
        
    else:
        # LCL pressure is above our closest pressure level
        # LCL height is below our closest height level
        insert_idx = lcl_index
        
        # Calculate height using hypsometric equation from surface
        Z_lcl = (R_dry * T_air[0] / g) * np.log(p[0] / p_s_value)
            
        T_air_lcl = interpolate_ambient_values(T_air, p, p_s_value)
        qv_lcl = interpolate_ambient_values(q_v, p, p_s_value)
    
    # Insert values at insert_idx
    temp_parcel = np.insert(temp_parcel, insert_idx, T_s_value)
    p = np.insert(p, insert_idx, p_s_value)
    Z = np.insert(Z, insert_idx, Z_lcl)
    T_air = np.insert(T_air, insert_idx, T_air_lcl)
    q_v = np.insert(q_v, insert_idx, qv_lcl)
    
    new_lcl_index = insert_idx
    
    return temp_parcel, p, Z, new_lcl_index, T_air, q_v

