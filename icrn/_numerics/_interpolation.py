from ..utils.dict_utils import state_index, arr_mul, state_sub, state_sum

def _linear_interpolation(steps, dt_fractions, ys):
    steps0 = steps
    steps1 = steps0 + 1

    y0 = state_index(ys, steps0)
    y1 = state_index(ys, steps1)

    return state_sum(y0, arr_mul(state_sub(y1, y0), dt_fractions))

def _hermitian_interpolation(steps, dt_fractions, dt, ys, fs):
    steps0 = steps
    steps1 = steps0 + 1

    y0 = ys[steps0]
    y1 = ys[steps1]

    f0 = fs[steps0]
    f1 = fs[steps1]

    a1 = state_mul(y0, 1-dt_fractions)
    a2 = state_mul(y1, dt_fractions)
    
    b1 = state_mul(state_sub(y1, y0), 1-2*dt_fractions)
    b2 = state_mul(f0, dt * (dt_fractions - 1))
    b3 = state_mul(f1, dt * dt_fractions)
    B = state_sum(b1, b2, b3)

    return state_sum(a1, a2, state_mul(B, 1 - dt_fractions))