import warnings
from collections.abc import Generator
from contextlib import contextmanager

_TORCH_JIT_SCRIPT_DEPRECATIONS = {
    "`torch.jit.script` is deprecated. Please switch to `torch.compile` or `torch.export`.",
    "`torch.jit.script` is not supported in Python 3.14+ and may break. "
    "Please switch to `torch.compile` or `torch.export`.",
}


@contextmanager
def suppress_cuequivariance_jit_deprecation() -> Generator[None, None, None]:
    """Suppress cuEquivariance's import-time torch.jit.script deprecation."""
    with warnings.catch_warnings():
        showwarning = warnings.showwarning

        def _showwarning(message, category, filename, lineno, file=None, line=None):
            if (
                issubclass(category, DeprecationWarning | FutureWarning)
                and str(message) in _TORCH_JIT_SCRIPT_DEPRECATIONS
            ):
                return
            showwarning(message, category, filename, lineno, file=file, line=line)

        warnings.showwarning = _showwarning
        yield
