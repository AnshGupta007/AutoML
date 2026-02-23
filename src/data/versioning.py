"""
Data versioning with DVC (Data Version Control).
Provides a thin wrapper around DVC CLI for tracking and versioning datasets.
DVC integration is optional — system works without it.
"""
import subprocess
from pathlib import Path
from typing import Optional, Union

from src.utils.logger import get_logger

log = get_logger(__name__)


class DataVersioning:
    """Optional DVC-based data versioning wrapper.

    If DVC is not installed or not initialized, all methods
    no-op with a warning (the pipeline continues normally).

    Example:
        >>> dvc = DataVersioning()
        >>> dvc.add("data/raw/train.csv")
        >>> dvc.push()
    """

    def __init__(self, enabled: bool = True, remote: Optional[str] = None) -> None:
        self.enabled = enabled and self._is_dvc_available()
        self.remote = remote

        if enabled and not self._is_dvc_available():
            log.warning("DVC not installed. Data versioning is disabled.")

    def add(self, path: Union[str, Path]) -> bool:
        """Track a file or directory with DVC.

        Args:
            path: Path to file/directory to track.

        Returns:
            True if successful, False otherwise.
        """
        if not self.enabled:
            return False

        try:
            path = str(path)
            self._run_dvc(["add", path])
            log.info(f"DVC: tracking '{path}'")
            return True
        except Exception as exc:
            log.warning(f"DVC add failed for '{path}': {exc}")
            return False

    def push(self) -> bool:
        """Push tracked data to remote storage."""
        if not self.enabled:
            return False
        try:
            cmd = ["push"]
            if self.remote:
                cmd += ["--remote", self.remote]
            self._run_dvc(cmd)
            log.info("DVC: pushed data to remote")
            return True
        except Exception as exc:
            log.warning(f"DVC push failed: {exc}")
            return False

    def pull(self) -> bool:
        """Pull tracked data from remote storage."""
        if not self.enabled:
            return False
        try:
            cmd = ["pull"]
            if self.remote:
                cmd += ["--remote", self.remote]
            self._run_dvc(cmd)
            log.info("DVC: pulled data from remote")
            return True
        except Exception as exc:
            log.warning(f"DVC pull failed: {exc}")
            return False

    def get_file_hash(self, path: Union[str, Path]) -> Optional[str]:
        """Get the DVC hash (md5) of a tracked file.

        Args:
            path: Path to the file.

        Returns:
            MD5 hash string, or None if not tracked.
        """
        dvc_file = Path(str(path) + ".dvc")
        if not dvc_file.exists():
            return None
        try:
            import yaml
            with open(dvc_file) as f:
                meta = yaml.safe_load(f)
            return meta.get("outs", [{}])[0].get("md5")
        except Exception:
            return None

    @staticmethod
    def _is_dvc_available() -> bool:
        try:
            result = subprocess.run(
                ["dvc", "--version"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            return result.returncode == 0
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return False

    @staticmethod
    def _run_dvc(args: list[str]) -> str:
        result = subprocess.run(
            ["dvc"] + args,
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout
