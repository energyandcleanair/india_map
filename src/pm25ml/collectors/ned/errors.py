"""NED errors."""

from pm25ml.collectors.export_pipeline import MissingDataError


class NedMissingDataError(MissingDataError):
    """Exception raised when expected data is missing from NASA Earthdata."""
