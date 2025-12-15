import contextlib
import datetime as _datetime
import io
import json
import logging
import sys
import warnings
from pathlib import Path
from typing import Callable, Dict, Iterable, Optional, Tuple, Union


class _Tee(io.TextIOBase):
    """Mirror writes to multiple streams."""

    def __init__(self, streams: Iterable[io.TextIOBase]) -> None:
        self._streams = list(streams)

    def write(self, data: str) -> int:
        for stream in self._streams:
            stream.write(data)
        return len(data)

    def flush(self) -> None:
        for stream in self._streams:
            stream.flush()


def _default_base_dir() -> Path:
    return Path(__file__).resolve().parent


def ensure_run_dir(
    run_name: str, base_dir: Optional[Path] = None, run_dir: Optional[Union[Path, str]] = None
) -> Path:
    base = base_dir or _default_base_dir()
    run_path = Path(run_dir) if run_dir is not None else base / "runs" / run_name / _datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_path.mkdir(parents=True, exist_ok=True)
    return run_path


def find_latest_run_dir(run_name: str, base_dir: Optional[Path] = None) -> Optional[Path]:
    base = base_dir or _default_base_dir()
    runs_root = base / "runs" / run_name
    if not runs_root.exists():
        return None
    candidates = [p for p in runs_root.iterdir() if p.is_dir()]
    if not candidates:
        return None
    return max(candidates, key=lambda path: path.name)


def get_existing_run_dir(
    run_name: str, base_dir: Optional[Path] = None, run_dir: Optional[Union[Path, str]] = None
) -> Path:
    if run_dir is not None:
        run_path = Path(run_dir)
        if not run_path.exists():
            raise FileNotFoundError(f"Run directory does not exist: {run_path}")
        return run_path

    latest = find_latest_run_dir(run_name, base_dir=base_dir)
    if latest is None:
        raise FileNotFoundError(f"No previous runs found under '{run_name}'.")
    return latest


class CheckpointManager:
    """Save versioned checkpoints and retain only the most recent ones."""

    def __init__(self, directory: Path, prefix: str, keep: int = 5):
        self.directory = Path(directory)
        self.prefix = prefix
        self.keep = keep
        self.directory.mkdir(parents=True, exist_ok=True)

    def _checkpoint_name(self, iteration: int) -> str:
        return f"{self.prefix}_iter{iteration:06d}.zip"

    def _all_checkpoints(self):
        return sorted(self.directory.glob(f"{self.prefix}_iter*.zip"), key=lambda p: p.name)

    def save(self, iteration: int, saver: Callable[[Path], None]) -> Path:
        path = self.directory / self._checkpoint_name(iteration)
        saver(path)
        self._prune()
        return path

    def _prune(self):
        checkpoints = self._all_checkpoints()
        if self.keep and len(checkpoints) > self.keep:
            for old in checkpoints[:-self.keep]:
                old.unlink(missing_ok=True)

    def latest(self) -> Optional[Path]:
        checkpoints = self._all_checkpoints()
        if not checkpoints:
            return None
        return checkpoints[-1]


def resolve_latest_model(
    run_dir: Path,
    checkpoints_subdir: str,
    checkpoint_prefix: str,
    fallback_filename: str,
) -> Path:
    """
    Returns the latest checkpoint if it exists, otherwise falls back to the given filename.
    Raises FileNotFoundError when no model can be found.
    """
    manager = CheckpointManager(Path(run_dir) / checkpoints_subdir, checkpoint_prefix)
    latest = manager.latest()
    if latest:
        return latest

    fallback = Path(run_dir) / fallback_filename
    if fallback.exists():
        return fallback

    raise FileNotFoundError(f"No model found. Checked latest checkpoint under '{checkpoints_subdir}' and fallback '{fallback}'.")


@contextlib.contextmanager
def log_run(
    run_name: str,
    log_name: str,
    hyperparams: Dict[str, object],
    *,
    run_dir: Optional[Union[Path, str]] = None,
    base_dir: Optional[Path] = None,
    also_print: bool = True,
) -> Tuple[Path, Path]:
    run_path = ensure_run_dir(run_name, base_dir=base_dir, run_dir=run_dir)
    log_path = run_path / f"{log_name}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    stdout_orig, stderr_orig = sys.stdout, sys.stderr
    with open(log_path, "a", encoding="utf-8") as log_file:
        header = {
            "timestamp": _datetime.datetime.now().isoformat(),
            "run_dir": str(run_path),
            "hyperparameters": hyperparams,
        }
        log_file.write(json.dumps(header, indent=2, sort_keys=True))
        log_file.write("\n--- OUTPUT START ---\n")
        log_file.flush()

        stdout_streams = [log_file]
        stderr_streams = [log_file]
        # Use the real console streams when requested.
        # This prevents nested log_run contexts from leaking output into the parent's log file.
        if also_print:
            stdout_streams.append(sys.__stdout__)
            stderr_streams.append(sys.__stderr__)
        sys.stdout = _Tee(stdout_streams)
        sys.stderr = _Tee(stderr_streams)

        try:
            yield run_path, log_path
        except Exception:
            import traceback
            log_file.write("\n--- EXCEPTION ---\n")
            traceback.print_exc(file=log_file)
            log_file.flush()
            raise
        finally:
            sys.stdout = stdout_orig
            sys.stderr = stderr_orig
            log_file.write("\n--- OUTPUT END ---\n")
            log_file.flush()


def suppress_noisy_gymnasium_warnings() -> None:
    # CyberBattleSim uses numpy.int32 for some Discrete fields; this is safe for SB3.
    warnings.filterwarnings(
        "ignore",
        message=r".*obs returned by the `reset\\(\\)` method should be an int or np\\.int64.*",
        category=UserWarning,
    )
    warnings.filterwarnings(
        "ignore",
        message=r".*obs returned by the `step\\(\\)` method should be an int or np\\.int64.*",
        category=UserWarning,
    )
    # Deprecation warning from gymnasium wrapper attribute access (we avoid it, but keep quiet if upstream emits it)
    warnings.filterwarnings(
        "ignore",
        message=r".*env\\.bounds.*deprecated.*",
        category=UserWarning,
    )
    warnings.filterwarnings(
        "ignore",
        message=r".*env\\.environment.*deprecated.*",
        category=UserWarning,
    )
    warnings.filterwarnings(
        "ignore",
        message=r".*env\\.sample_valid_action.*deprecated.*",
        category=UserWarning,
    )
    warnings.filterwarnings(
        "ignore",
        message=r".*env\\.is_action_valid.*deprecated.*",
        category=UserWarning,
    )


def get_stdout_logger(name: str, *, level: int = logging.INFO) -> logging.Logger:
    """
    Return a logger that prints once to the current sys.stdout.
    Must be called from inside log_run() if you want output to land in the run log file.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False
    handler_stream = sys.stdout

    # Avoid accumulating handlers across repeated calls.
    for h in list(logger.handlers):
        if isinstance(h, logging.StreamHandler) and getattr(h, "stream", None) is handler_stream:
            break
    else:
        logger.handlers = []
        handler = logging.StreamHandler(handler_stream)
        handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(handler)

    return logger


@contextlib.contextmanager
def capture_cyberbattle_logs(*, level: int = logging.INFO):
    """
    Capture CyberBattleSim logs (e.g. BLOCKED TRAFFIC / discovered node / GOT REWARD)
    into the current sys.stdout (which is redirected to eval.log by log_run()).
    """
    cyber_logger = logging.getLogger("cyberbattle")
    old_level = cyber_logger.level
    old_handlers = list(cyber_logger.handlers)
    old_propagate = cyber_logger.propagate

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("%(message)s"))

    cyber_logger.handlers = [handler]
    cyber_logger.setLevel(level)
    cyber_logger.propagate = False
    try:
        yield
    finally:
        cyber_logger.handlers = old_handlers
        cyber_logger.setLevel(old_level)
        cyber_logger.propagate = old_propagate
