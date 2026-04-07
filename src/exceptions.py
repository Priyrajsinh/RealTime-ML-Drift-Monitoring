"""Project-wide exception hierarchy for B5."""


class ProjectBaseError(Exception):
    """Base class for all B5 project exceptions."""


class DataLoadError(ProjectBaseError):
    """Raised when raw data cannot be loaded or parsed."""


class DataValidationError(ProjectBaseError):
    """Raised when pandera schema validation fails."""


class ModelNotFoundError(ProjectBaseError):
    """Raised when a saved model file is missing."""


class PredictionError(ProjectBaseError):
    """Raised when inference fails (shape mismatch, etc.)."""


class DriftDetectionError(ProjectBaseError):
    """Raised when drift detection computation fails."""


class ConfigError(ProjectBaseError):
    """Raised when config/config.yaml is missing or has invalid keys."""
