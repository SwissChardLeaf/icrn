import unittest

from jax import numpy as jnp

from ..utils.dict_utils import dict_allclose
from ._reaction_numerics import (
    _euler_step,
    _patankar_euler_step,
    _RK4_step,
)


class TestEulerStep(unittest.TestCase):
    def test_euler_step_exponential_decay(self):
        def dyn_f(state, non_state):
            return -state

        state = jnp.array([1.0, 2.0, 3.0])
        non_state = jnp.array([4.0, 5.0, 6.0])
        dt = 0.5
        computed_state = _euler_step(state, non_state, dyn_f, dt)
        target_state = jnp.array([0.5, 1.0, 1.5])
        self.assertTrue(jnp.allclose(computed_state, target_state))

        # shouldn't change the target_state
        non_state = jnp.array([1.0, 5.0, 6.0])
        computed_state = _euler_step(state, non_state, dyn_f, dt)
        self.assertTrue(jnp.allclose(computed_state, target_state))

    def test_euler_step_exponential_decay_with_dicts(self):
        def dyn_f(state, non_state):
            return {
                "A": -non_state["k1"] * state["A"],
                "B": -non_state["k2"] * state["B"],
                "C": -non_state["k3"] * state["C"],
            }

        state = {
            "A": jnp.array(2.0),
            "B": jnp.array([4.0, 5.0, 6.0]),
            "C": jnp.array(
                [
                    [7.0, 8.0, 9.0],
                    [10.0, 11.0, 12.0],
                ]
            ),
        }

        non_state = {
            "k1": jnp.array(1.0),
            "k2": jnp.array([5.0, 6.0, 7.0]),
            "k3": jnp.array(
                [
                    [0.1, 0.2, 0.3],
                    [0.4, 0.5, 0.6],
                ]
            ),
        }

        dt = 0.75
        computed_state = _euler_step(state, non_state, dyn_f, dt)

        target_state = {
            "A": jnp.array(0.5),
            "B": state["B"] * (1 - non_state["k2"] * dt),
            "C": state["C"] * (1 - non_state["k3"] * dt),
        }
        self.assertTrue(dict_allclose(computed_state, target_state))


class TestPatankarEulerStep(unittest.TestCase):
    def test_decay_positivity(self):
        def pd_f(state, non_state):
            return (
                {"A": 0.0},
                {"A": non_state["k"] * state["A"]},
            )

        state = {"A": jnp.array(1.0)}
        non_state = {"k": jnp.array(2.0)}
        dt = 10.0  # large dt; explicit Euler would go negative

        computed_state = _patankar_euler_step(state, non_state, pd_f, dt)
        target = 1.0 / (1.0 + dt * 2.0)

        self.assertTrue(jnp.allclose(computed_state["A"], target))
        self.assertTrue(computed_state["A"] > 0)

    def test_production_only(self):
        def pd_f(state, non_state):
            return (
                {"A": non_state["p"]},
                {"A": 0.0},
            )

        state = {"A": jnp.array(2.0)}
        non_state = {"p": jnp.array(3.0)}
        dt = 0.5

        computed_state = _patankar_euler_step(state, non_state, pd_f, dt)
        target = 2.0 + dt * 3.0

        self.assertTrue(jnp.allclose(computed_state["A"], target))

    def test_multi_species(self):
        def pd_f(state, non_state):
            production = {"A": jnp.array(0.0), "B": state["A"]}
            destruction = {"A": state["A"], "B": jnp.array(0.0)}
            return production, destruction

        state = {"A": jnp.array(4.0), "B": jnp.array(1.0)}
        dt = 0.25

        computed_state = _patankar_euler_step(state, None, pd_f, dt)

        target_A = 4.0 * (4.0 + 0.0) / (4.0 + dt * 4.0)
        target_B = 1.0 * (1.0 + dt * 4.0) / (1.0 + 0.0)

        self.assertTrue(jnp.allclose(computed_state["A"], target_A))
        self.assertTrue(jnp.allclose(computed_state["B"], target_B))

    def test_array_state(self):
        def pd_f(state, non_state):
            return (
                {"A": jnp.zeros(3)},
                {"A": non_state["k"] * state["A"]},
            )

        state = {"A": jnp.array([1.0, 2.0, 3.0])}
        non_state = {"k": jnp.array([0.5, 1.0, 2.0])}
        dt = 0.5

        computed_state = _patankar_euler_step(state, non_state, pd_f, dt)
        target = state["A"] / (1.0 + dt * non_state["k"])

        self.assertTrue(jnp.allclose(computed_state["A"], target))
        self.assertTrue(jnp.all(computed_state["A"] > 0))


class TestRK4Step(unittest.TestCase):
    def test_RK4_step_exponential_decay(self):
        def dyn_f(state, non_state):
            return -state

        state = jnp.array(1.0)
        dt = 0.61
        computed_state = _RK4_step(state, None, dyn_f, dt)
        target_state = jnp.array(0.54398893375)
        self.assertTrue(jnp.allclose(computed_state, target_state))

        state = jnp.array(1.0)
        dt = jnp.log(2)
        computed_state = _RK4_step(state, None, dyn_f, dt)
        target_state = jnp.array(0.5011933468419623)
        self.assertTrue(jnp.allclose(computed_state, target_state))

        state = jnp.array([1.0, 2.0, 3.0])
        non_state = jnp.array([4.0, 5.0, 6.0])
        dt = 0.5
        computed_state = _RK4_step(state, non_state, dyn_f, dt)

        k1 = -state
        k2 = -(state + k1 * dt * 0.5)
        k3 = -(state + k2 * dt * 0.5)
        k4 = -(state + k3 * dt)

        next_step = (k1 + k2 * 2 + k3 * 2 + k4) * dt / 6

        target_state = state + next_step
        self.assertTrue(jnp.allclose(computed_state, target_state))

    def test_RK4_step_exponential_decay_with_dicts(self):
        def dyn_f(state, non_state):
            return {
                "A": -non_state["k1"] * state["A"],
                "B": -non_state["k2"] * state["B"],
                "C": -non_state["k3"] * state["C"],
            }

        state = {
            "A": jnp.array(2.0),
            "B": jnp.array([4.0, 5.0, 6.0]),
            "C": jnp.array(
                [
                    [7.0, 8.0, 9.0],
                    [10.0, 11.0, 12.0],
                ]
            ),
        }

        non_state = {
            "k1": jnp.array(1.0),
            "k2": jnp.array([5.0, 6.0, 7.0]),
            "k3": jnp.array(
                [
                    [0.1, 0.2, 0.3],
                    [0.4, 0.5, 0.6],
                ]
            ),
        }

        dt = 0.75
        computed_state = _RK4_step(state, non_state, dyn_f, dt)

        def manual_dyn_f(x, y):
            return -y * x

        manual_state_A = _RK4_step(
            state["A"], non_state["k1"], manual_dyn_f, dt
        )
        manual_state_B = _RK4_step(
            state["B"], non_state["k2"], manual_dyn_f, dt
        )
        manual_state_C = _RK4_step(
            state["C"], non_state["k3"], manual_dyn_f, dt
        )

        target_state = {
            "A": manual_state_A,
            "B": manual_state_B,
            "C": manual_state_C,
        }
        self.assertTrue(dict_allclose(computed_state, target_state))

    def test_RK4_step_with_return_dynamics(self):
        def dyn_f(state, non_state):
            return -non_state * state

        state = jnp.array(1.2)
        non_state = jnp.array(1.0)
        dt = 0.61
        computed_state, computed_dynamics = _RK4_step(
            state, non_state, dyn_f, dt, return_dynamics=True
        )
        target_state = jnp.array(0.6527867205000001)
        target_dynamics = jnp.array(-1.2)
        self.assertTrue(jnp.allclose(computed_state, target_state))
        self.assertTrue(jnp.allclose(computed_dynamics, target_dynamics))

        non_state = jnp.array(2.0)
        computed_state, computed_dynamics = _RK4_step(
            state, non_state, dyn_f, dt, return_dynamics=True
        )
        target_state = jnp.array(0.3766371279999998)
        target_dynamics = jnp.array(-2.4)
        self.assertTrue(jnp.allclose(computed_state, target_state))
        self.assertTrue(jnp.allclose(computed_dynamics, target_dynamics))

    def test_RK4_step_exponential_decay_with_dicts_return_dynamics(self):
        def dyn_f(state, non_state):
            return {
                "A": -non_state["k1"] * state["A"],
                "B": -non_state["k2"] * state["B"],
                "C": -non_state["k3"] * state["C"],
            }

        state = {
            "A": jnp.array(2.0),
            "B": jnp.array([4.0, 5.0, 6.0]),
            "C": jnp.array(
                [
                    [7.0, 8.0, 9.0],
                    [10.0, 11.0, 12.0],
                ]
            ),
        }

        non_state = {
            "k1": jnp.array(1.0),
            "k2": jnp.array([5.0, 6.0, 7.0]),
            "k3": jnp.array(
                [
                    [0.1, 0.2, 0.3],
                    [0.4, 0.5, 0.6],
                ]
            ),
        }

        dt = 0.75
        computed_state, computed_dynamics = _RK4_step(
            state, non_state, dyn_f, dt, return_dynamics=True
        )

        def manual_dyn_f(y, k):
            return -k * y

        manual_state_A, manual_dynamics_A = _RK4_step(
            state["A"], non_state["k1"], manual_dyn_f, dt, return_dynamics=True
        )
        manual_state_B, manual_dynamics_B = _RK4_step(
            state["B"], non_state["k2"], manual_dyn_f, dt, return_dynamics=True
        )
        manual_state_C, manual_dynamics_C = _RK4_step(
            state["C"], non_state["k3"], manual_dyn_f, dt, return_dynamics=True
        )

        target_state = {
            "A": manual_state_A,
            "B": manual_state_B,
            "C": manual_state_C,
        }

        target_dynamics = {
            "A": manual_dynamics_A,
            "B": manual_dynamics_B,
            "C": manual_dynamics_C,
        }
        self.assertTrue(dict_allclose(computed_state, target_state))
        self.assertTrue(dict_allclose(computed_dynamics, target_dynamics))
