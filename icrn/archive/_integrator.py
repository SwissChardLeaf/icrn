class Integrator:
    pass


def euler(concs_data, rate_data, dt, dynamics_func):
    dynamics = concs_data.init_with_dict(dynamics_func(concs_data | rate_data))
    return concs_data + dt * dynamics


def relu_euler(concs_data, rate_data, dt, dynamics_func):
    dynamics = concs_data.init_with_dict(dynamics_func(concs_data | rate_data))
    return jax_tree.tree_map(jax.nn.relu, concs_data + dt * dynamics)


def RK4(concs_data, rate_constant_data, dt, dynamics_func):
    k1 = concs_data.init_with_dict(
        dynamics_func(concs_data | rate_constant_data)
    )
    k2 = concs_data.init_with_dict(
        dynamics_func(concs_data + k1 * dt * 0.5 | rate_constant_data)
    )
    k3 = concs_data.init_with_dict(
        dynamics_func(concs_data + k2 * dt * 0.5 | rate_constant_data)
    )
    k4 = concs_data.init_with_dict(
        dynamics_func(concs_data + k3 * dt | rate_constant_data)
    )
    return concs_data + (k1 + (k2 * 2) + (k3 * 2) + k4) * (dt / 6)


def relu_RK4(concs_data, rate_constant_data, dt, dynamics_func):
    k1 = concs_data.init_with_dict(
        dynamics_func(
            jax_tree.tree_map(jax.nn.relu, concs_data) | rate_constant_data
        )
    )
    k2 = concs_data.init_with_dict(
        dynamics_func(
            jax_tree.tree_map(jax.nn.relu, concs_data) + k1 * dt * 0.5
            | rate_constant_data
        )
    )
    k3 = concs_data.init_with_dict(
        dynamics_func(
            jax_tree.tree_map(jax.nn.relu, concs_data) + k2 * dt * 0.5
            | rate_constant_data
        )
    )
    k4 = concs_data.init_with_dict(
        dynamics_func(
            jax_tree.tree_map(jax.nn.relu, concs_data) + k3 * dt
            | rate_constant_data
        )
    )
    return jax_tree.tree_map(
        jax.nn.relu, concs_data + (k1 + (k2 * 2) + (k3 * 2) + k4) * (dt / 6)
    )


def RK4_5(concs, dynamics, dt):
    pass
