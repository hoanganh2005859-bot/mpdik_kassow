"""Project-specific exception types for config, asset, and dataset validation failures."""


class KR810DatasetError(Exception):
    """Base class for all project-specific errors."""


class AssetValidationError(KR810DatasetError):
    """Raised when an on-disk asset (model, URDF, mesh) fails a validation check."""


class ModelConfigurationError(KR810DatasetError):
    """Raised when a loaded MuJoCo model does not match the expected/configured robot definition."""


class KinematicsError(KR810DatasetError):
    """Base class for kinematics computation failures (FK, Jacobian, pose error)."""


class InvalidJointVectorError(KinematicsError):
    """Raised when a joint vector fails shape, finiteness, or range validation."""


class NumericalKinematicsError(KinematicsError):
    """Raised when a kinematics computation produces a non-finite or otherwise invalid numerical result."""


class DLSSolverError(KR810DatasetError):
    """Raised for unrecoverable Damped Least Squares solver failures (not the same as a converged=False result)."""
