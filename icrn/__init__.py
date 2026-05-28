"""indexed chemical reaction networks in a differentiable, tensor-based framework."""

from .symbols import (
    IndexSymbol,
    Species,
    RateConstant,
    TensorExpression,
    TensorFunction,
    TensorLiteral,
    many_index_symbols,
    many_species,
    many_rate_constants,
)

from .reactions import (
    AbstractReaction,
    MassActionReaction,
    FastReaction,
)

from .solver import (
    solve_well_mixed,
    solve_reaction_diffusion,
    solve_with_ops,
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
