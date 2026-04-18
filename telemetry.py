import logging
import os

import openlit
from opentelemetry._logs import get_logger_provider
from opentelemetry.exporter.otlp.proto.http._log_exporter import OTLPLogExporter
from opentelemetry.sdk._logs import LoggingHandler
from opentelemetry.sdk._logs import LoggerProvider
from opentelemetry.sdk._logs.export import BatchLogRecordProcessor
from opentelemetry import trace

# Module-level tracer — imported by agent.py and tools so they can create spans
# without each having to call get_tracer() themselves.
tracer = trace.get_tracer("receipt-organizer")


def setup_telemetry() -> None:
    """Initialize OpenTelemetry: auto-instrument LangChain and route all Python
    logging through OTel to the local OpenLIT instance."""
    otlp_endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://127.0.0.1:4318")

    # Auto-instruments LangChain traces + sets up OTLP trace/metrics/logs exporters
    openlit.init(application_name="file_classifier", otlp_endpoint=otlp_endpoint)

    # Attach a log exporter to openlit's already-configured LoggerProvider
    logger_provider = get_logger_provider()
    if isinstance(logger_provider, LoggerProvider):
        logger_provider.add_log_record_processor(
            BatchLogRecordProcessor(OTLPLogExporter(endpoint=otlp_endpoint))
        )

    # Bridge Python's logging module → OTel log records
    handler = LoggingHandler(logger_provider=logger_provider)
    logging.getLogger().addHandler(handler)
    logging.getLogger().setLevel(logging.INFO)
