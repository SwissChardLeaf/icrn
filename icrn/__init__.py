"""indexed chemical reaction networks in a differentiable, tensor-based
framework."""

from .reactions import (
    AbstractReaction,
    FastReaction,
    MassActionReaction,
)
from .solver import (
    solve_reaction_diffusion,
    solve_well_mixed,
    solve_with_ops,
)
from .symbols import (
    IndexSymbol,
    RateConstant,
    Species,
    TensorExpression,
    TensorFunction,
    TensorLiteral,
    many_index_symbols,
    many_rate_constants,
    many_species,
)

__version__ = "0.1.0"

__all__ = [
    "IndexSymbol",
    "Species",
    "RateConstant",
    "TensorExpression",
    "TensorFunction",
    "TensorLiteral",
    "many_index_symbols",
    "many_species",
    "many_rate_constants",
    "AbstractReaction",
    "MassActionReaction",
    "FastReaction",
    "solve_well_mixed",
    "solve_reaction_diffusion",
    "solve_with_ops",
    "__version__",
]
