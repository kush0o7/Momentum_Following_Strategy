import logging


def configure_logging(level: str) -> None:
    """Configure app-wide logging once."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
