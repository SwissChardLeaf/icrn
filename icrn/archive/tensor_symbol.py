from abc import ABC, abstractmethod
from dataclasses import dataclass, field, replace
import jax.numpy as jnp
from enum import Enum
from typing import Any


class TensorExpr(ABC):
    @abstractmethod
    def eval(self, data):
        pass

    def __call__(self, data):
        return self.eval(data)


class TensorSymbol(Expr):
    pass


@dataclass(frozen=True)
class Function(Expr):
    fn: jnp.ufunc
    args: list[Expr]
    aux: Any = None

    def eval(self, data):
        pass
