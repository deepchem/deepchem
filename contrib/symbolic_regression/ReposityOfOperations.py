
import torch

class OpSpec:
    def __init__(self, arity, fn, formatter=None):
        self.arity = arity
        self.fn = fn
        self.formatter = formatter


OP_REGISTRY = {}


def register_op(
    name,
    arity,
    fn,
    formatter=None,
):
    OP_REGISTRY[name] = OpSpec(arity=arity, fn=fn, formatter=formatter)


def _register_builtin_ops():
    register_op("add", 2, lambda a, b: a + b, lambda args: f"({args[0]} + {args[1]})")
    register_op("sub", 2, lambda a, b: a - b, lambda args: f"({args[0]} - {args[1]})")
    register_op("mul", 2, lambda a, b: a * b, lambda args: f"({args[0]} * {args[1]})")
    register_op("div", 2, lambda a, b: a / (b + 1e-12), lambda args: f"({args[0]} / {args[1]})")
    register_op("neg", 1, lambda a: -a, lambda args: f"-({args[0]})")
    register_op("sin", 1, lambda a: torch.sin(a), lambda args: f"sin({args[0]})")
    register_op("cos", 1, lambda a: torch.cos(a), lambda args: f"cos({args[0]})")
    register_op("log", 1, lambda a: torch.log(torch.abs(a) + 1e-8), lambda args: f"log({args[0]})")
    register_op("exp", 1, lambda a: torch.exp(torch.clamp(a, -50.0, 50.0)), lambda args: f"exp({args[0]})")


_register_builtin_ops()
