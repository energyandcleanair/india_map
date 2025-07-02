"""The logging configuration for the pm25ml package."""

from __future__ import annotations

import os
import sys
from datetime import datetime
from logging import DEBUG, Formatter, Logger, LogRecord, StreamHandler, getLogger

from pythonjsonlogger.json import JsonFormatter


def _in_cloud_run_job() -> bool:
    return bool(
        os.getenv("CLOUD_RUN_JOB") or os.getenv("CLOUD_RUN_TASK_INDEX"),
    )


class _CloudRunJsonFormatter(JsonFormatter):
    def formatTime(self, record: LogRecord, datefmt: str | None = None) -> str:  # noqa: ARG002, N802
        # Format the timestamp as RFC 3339 with microsecond precision
        isoformat = datetime.fromtimestamp(record.created).isoformat()  # noqa: DTZ006
        return f"{isoformat}Z"


logger: Logger = getLogger("pm25ml")
logger.setLevel(DEBUG)
logger.handlers.clear()

stream_handler = StreamHandler(sys.stdout)

if _in_cloud_run_job():
    # Production: structured JSON logs
    formatter = _CloudRunJsonFormatter(
        "%(asctime)s %(levelname)s %(threadName)s %(message)s "
        "%(otelTraceID)s %(otelSpanID)s %(otelTraceSampled)s",
        rename_fields={
            "levelname": "severity",
            "asctime": "timestamp",
            "otelTraceID": "logging.googleapis.com/trace",
            "otelSpanID": "logging.googleapis.com/spanId",
            "otelTraceSampled": "logging.googleapis.com/trace_sampled",
        },
    )
else:
    # Local: human-friendly with thread name
    formatter = Formatter(
        "%(asctime)s | %(name)s | %(levelname)s | [%(threadName)s] %(message)s",
        "%Y-%m-%d %H:%M:%S",
    )

stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)
