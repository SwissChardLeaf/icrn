"""Default JAX to CPU for tests. Import before ``jax`` in test modules.

GPU subprocess workers set ``ICRN_TEST_JAX_DEVICE=gpu`` before importing JAX.
"""

import os

if os.environ.get("ICRN_TEST_JAX_DEVICE") != "gpu":
    os.environ["JAX_PLATFORMS"] = "cpu"
