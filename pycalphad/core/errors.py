class PycalphadError(Exception):
    """Base exception class for all PyCalphad errors."""
    pass


class CalculateError(PycalphadError):
    """Exception related to use of calculate() function."""
    pass


class DofError(PycalphadError):
    """Error due to missing degrees of freedom."""
    pass


class EquilibriumError(PycalphadError):
    """Exception related to calculation of equilibrium."""
    pass


class ConditionError(CalculateError, EquilibriumError):
    """Exception related to calculation conditions."""
    pass


class DatabaseError(PycalphadError):
    """Exception related to thermodynamic database parsing or manipulation."""
    pass


class DatabaseDialectError(DatabaseError):
    """Exception raised when encountering unknown or malformed database syntax."""
    pass


class ModelError(PycalphadError):
    """Exception related to thermodynamic model formulation or parameterization."""
    pass


class SolverError(EquilibriumError):
    """Exception related to numerical solver failure or convergence limits."""
    pass


class ConvergenceError(SolverError):
    """Exception raised when the local minimizer or hyperplane loop fails to converge."""
    pass


class InfeasibleEquilibriumError(SolverError):
    """Exception raised when no physically feasible equilibrium solution exists."""
    pass


class PropertyError(PycalphadError):
    """Exception related to physical/transport property calculations or metamodels."""
    pass


class KineticsError(PycalphadError):
    """Exception related to precipitation, transformation, or diffusion kinetics."""
    pass


__all__ = [
    "PycalphadError",
    "CalculateError",
    "DofError",
    "EquilibriumError",
    "ConditionError",
    "DatabaseError",
    "DatabaseDialectError",
    "ModelError",
    "SolverError",
    "ConvergenceError",
    "InfeasibleEquilibriumError",
    "PropertyError",
    "KineticsError",
]

