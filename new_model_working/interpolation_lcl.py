import numpy as np


def interpolate_ambient_values(values, pressures, target_pressure):
    """
    Linear interpolation for target_pressure in a descending pressure profile.
    """
    # Find indices surrounding target_pressure
    idx_above = np.where(pressures <= target_pressure)[0][0]  # first pressure <= target
    idx_below = idx_above - 1

    # Linear interpolation
    value = values[idx_below] + (values[idx_above] - values[idx_below]) * \
            (target_pressure - pressures[idx_below]) / (pressures[idx_above] - pressures[idx_below])
    return value

def insert_lcl_values(temp_parcel, p, Z, lcl_index, T_s_value, p_s_value, T_air, q_v):
    top_index = len(p) - 1
    
    if abs(p[lcl_index] - p_s_value) < 0.1:
        insert_idx = lcl_index
        Z_lcl = Z[lcl_index]
        T_air_lcl = T_air[lcl_index]
        qv_lcl = q_v[lcl_index] 
    elif lcl_index == top_index:
        insert_idx = top_index + 1  # append
        Z_lcl = Z[lcl_index] + (Z[lcl_index+1] - Z[lcl_index]) * \
        (p[lcl_index] - p_s_value) / (p[lcl_index] - p[lcl_index+1])
        T_air_lcl = interpolate_ambient_values(T_air, p, p_s_value)
        qv_lcl = interpolate_ambient_values(q_v, p, p_s_value)
    elif (p[lcl_index] - p_s_value) > 0:
        insert_idx = lcl_index + 1
        Z_lcl = Z[lcl_index] + (Z[lcl_index+1] - Z[lcl_index]) * \
        (p[lcl_index] - p_s_value) / (p[lcl_index] - p[lcl_index+1])
        T_air_lcl = interpolate_ambient_values(T_air, p, p_s_value)
        qv_lcl = interpolate_ambient_values(q_v, p, p_s_value)
    else:
        print('a',T_s_value)
        insert_idx = lcl_index
        Z_lcl = Z[lcl_index-1] + (Z[lcl_index] - Z[lcl_index-1]) * \
        (p_s_value - p[lcl_index-1]) / (p[lcl_index] - p[lcl_index-1])
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
