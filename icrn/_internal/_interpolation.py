from ..utils.dict_utils import dict_index, arr_mul, dict_sub, dict_sum
import jax


def _linear_interpolation(steps, dt_fractions, ys):
    steps0 = steps
    steps1 = steps0 + 1

    y0 = dict_index(ys, steps0)
    y1 = dict_index(ys, steps1)

    # print(steps0, steps1)
    # print(y0, y1)

    # print(dt_fractions, dt_fractions.shape)

    diff_y = dict_sub(y1, y0)

    def _mul_dt_fractions(x):
        dt_fractions_expanded = dt_fractions.reshape(
            dt_fractions.shape[0], *([1] * (x.ndim - 1))
        )
        return x * dt_fractions_expanded

    diff_y = jax.tree.map(_mul_dt_fractions, diff_y)

    return dict_sum(y0, diff_y)


def _hermitian_interpolation(steps, dt_fractions, dt, ys, fs):
    steps0 = steps
    steps1 = steps0 + 1

    y0 = ys[steps0]
    y1 = ys[steps1]

    f0 = fs[steps0]
    f1 = fs[steps1]

    a1 = arr_mul(y0, 1 - dt_fractions)
    a2 = arr_mul(y1, dt_fractions)

    b1 = arr_mul(dict_sub(y1, y0), 1 - 2 * dt_fractions)
    b2 = arr_mul(f0, dt * (dt_fractions - 1))
    b3 = arr_mul(f1, dt * dt_fractions)
    B = dict_sum(b1, b2, b3)

    return dict_sum(a1, a2, arr_mul(B, 1 - dt_fractions))
