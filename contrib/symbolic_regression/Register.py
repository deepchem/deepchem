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
    def _safe_div(a, b):
        out = a / b
        return torch.where(
            torch.abs(b) > 1e-12, out, torch.full_like(out, float("nan"))
        )

    def _safe_log(a):
        return torch.where(a > 0.0, torch.log(a), torch.full_like(a, float("nan")))

    register_op("add", 2, lambda a, b: a + b, lambda args: f"({args[0]} + {args[1]})")
    register_op("sub", 2, lambda a, b: a - b, lambda args: f"({args[0]} - {args[1]})")
    register_op("mul", 2, lambda a, b: a * b, lambda args: f"({args[0]} * {args[1]})")
    register_op("div", 2, _safe_div, lambda args: f"({args[0]} / {args[1]})")
    register_op("neg", 1, lambda a: -a, lambda args: f"-({args[0]})")
    register_op("sin", 1, lambda a: torch.sin(a), lambda args: f"sin({args[0]})")
    register_op("cos", 1, lambda a: torch.cos(a), lambda args: f"cos({args[0]})")
    register_op("log", 1, _safe_log, lambda args: f"log({args[0]})")
    register_op("exp", 1, lambda a: torch.exp(a), lambda args: f"exp({args[0]})")


_register_builtin_ops()
