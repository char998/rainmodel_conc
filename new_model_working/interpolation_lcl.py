import numpy as np
from constants import R_dry, g

def interpolate_ambient_values(values, pressures, target_pressure, target_index):
    """
    Helper function to interpolate ambient air properties at LCL pressure
    """
    return values[target_index-1] + (
        (values[target_index + 1] - values[target_index-1]) /
        (pressures[target_index + 1] - pressures[target_index-1])
    ) * (target_pressure - pressures[target_index-1])

def insert_lcl_values(temp_parcel, p, Z, lcl_index, T_s_value, p_s_value, T_air, q_v):
    top_index = len(p) - 1

    if abs(p[lcl_index] - p_s_value) < 0.1:
        insert_idx = lcl_index
    elif lcl_index == top_index:
        insert_idx = top_index + 1  # append
    elif (p[lcl_index] - p_s_value) > 0:
        insert_idx = lcl_index + 1
    else:
        insert_idx = lcl_index

    # Insert values at insert_idx
    temp_parcel = np.insert(temp_parcel, insert_idx, T_s_value)
    p = np.insert(p, insert_idx, p_s_value)
    Z_lcl = R_dry * np.mean(temp_parcel[:insert_idx+1]) * np.log(p[0]/p[insert_idx]) / g
    Z = np.insert(Z, insert_idx, Z_lcl)

    T_air_lcl = interpolate_ambient_values(T_air, p, p_s_value, insert_idx)
    qv_lcl = interpolate_ambient_values(q_v, p, p_s_value, insert_idx)
    T_air = np.insert(T_air, insert_idx, T_air_lcl)
    q_v = np.insert(q_v, insert_idx, qv_lcl)

    new_lcl_index = insert_idx

    return temp_parcel, p, Z, new_lcl_index, T_air, q_v
