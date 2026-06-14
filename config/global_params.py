from dataclasses import dataclass
import torch

import sys
import os

def trace_print(*args, **kwargs):
    """Write formatted trace to trace_file if enabled, falling back to stdout."""
    f = settings.trace_file if settings.enable_trace and settings.trace_file else sys.stdout
    print(*args, file=f, **kwargs)

def trace_formula(template: str, result, **subs):
    """Print a formula with both its template form and substituted values.

    Example:
        trace_formula("max(0, {old} + {rate} * ({comp} - {gap}))", 0.14,
                      old=0.30, rate=1.0, comp=0.16, gap=0.50)
    Prints:
        [FORMULA] max(0, old + rate * (comp - gap))
        [VALUES]  max(0, 0.30 + 1.0 * (0.16 - 0.50)) = max(0, -0.04) = 0.14
    """
    f = settings.trace_file if settings.enable_trace and settings.trace_file else sys.stdout
    print(f"  [FORMULA] {template}", file=f)
    expr = template
    for k, v in subs.items():
        expr = expr.replace("{" + k + "}", str(v))
    print(f"  [VALUES]  {expr} = {result}", file=f)


@dataclass
class Settings:
    enable_print: bool = False
    enable_trace: bool = False
    exp_result_dir: str = "/root/autodl-tmp/res/"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_dir: str = ""
    plot_dir: str = ""
    weight_dir: str = ""
    resume_episode: int = 0
    seed: int = 0
    is_evaluate: bool = False
    trace_file = None
settings = Settings()
