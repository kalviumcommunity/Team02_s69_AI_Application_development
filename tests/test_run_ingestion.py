"""Tests for the ingestion CLI's console-encoding guard."""
from __future__ import annotations

import io

from src.ingestion.run_ingestion import _ensure_utf8_stdout


class _FakeStream(io.StringIO):
    """A stream that records the encoding it was reconfigured with."""

    def __init__(self):
        super().__init__()
        self.reconfigured_with = None

    def reconfigure(self, encoding=None, errors=None):
        self.reconfigured_with = (encoding, errors)


def test_reconfigures_streams_that_support_it(monkeypatch):
    fake_stdout = _FakeStream()
    fake_stderr = _FakeStream()

    monkeypatch.setattr("sys.stdout", fake_stdout)
    monkeypatch.setattr("sys.stderr", fake_stderr)

    _ensure_utf8_stdout()

    assert fake_stdout.reconfigured_with == ("utf-8", "replace")
    assert fake_stderr.reconfigured_with == ("utf-8", "replace")


def test_does_not_crash_when_stream_lacks_reconfigure(monkeypatch):
    class _NoReconfigure:
        pass

    monkeypatch.setattr("sys.stdout", _NoReconfigure())
    monkeypatch.setattr("sys.stderr", _NoReconfigure())

    # Should simply do nothing, not raise.
    _ensure_utf8_stdout()
