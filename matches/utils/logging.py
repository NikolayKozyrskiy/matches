import logging
import sys
from contextlib import contextmanager

import coloredlogs
from tqdm.auto import tqdm

from .object import AttrProxy


class StreamThroughTqdm(AttrProxy):
    def __init__(self, real_stream, stdout=sys.stdout, stderr=sys.stderr):
        self._stdout = stdout
        self._stderr = stderr
        self._real_stream = real_stream
        super().__init__("_real_stream")

    @contextmanager
    def _std_streams(self):
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        try:
            sys.stderr = self._stderr
            sys.stdout = self._stdout
            yield
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr

    def write(self, buf):
        # Set default streams back
        # They are required for correct tqdm.write work
        # See tqdm.tqdm.external_write_mode()
        with self._std_streams():
            for line in buf.rstrip().splitlines():
                tqdm.write(line.rstrip(), self._real_stream)

    def flush(self):
        pass


def configure_logging(file=sys.stdout):
    tqdm_stream = StreamThroughTqdm(file)
    remove_handlers()
    coloredlogs.install(isatty=True, stream=tqdm_stream)
    logging.captureWarnings(True)
    sys.stdout = tqdm_stream
    sys.stderr = tqdm_stream


def remove_handlers() -> None:
    for handler in logging.root.handlers:
        logging.root.removeHandler(handler)
