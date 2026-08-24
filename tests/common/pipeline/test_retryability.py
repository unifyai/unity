"""Whether to retry is read from the failure, not guessed from its prose.

Retryability was decided by searching the rendered error text for tokens like
``"timeout"``, ``"500"`` and ``"429"``. That is a heuristic on prose standing in
for a typed decision, and it was wrong in both directions at once:

* a deterministic ``400`` was retried to the cap, because nothing in *"Invalid
  context name"* looks permanent -- and since the message is never acked, it was
  redelivered and retried again, indefinitely, by fifteen workers;
* any failure whose text merely contained ``"500"`` -- a row count, an id, a URL
  with a port -- was retried as though the server had faulted.

A refusal the caller caused refuses identically every time. Retrying it is not
just wasted work: it occupies a worker fleet.
"""

from __future__ import annotations

import pytest

from unify.common.pipeline.retry_policy import is_retryable_exception


class _WithStatus(Exception):
    def __init__(self, message: str, status_code: int) -> None:
        super().__init__(message)
        self.status_code = status_code


class _WithResponse(Exception):
    """The shape most HTTP clients raise: the status lives on a response."""

    class _Response:
        def __init__(self, status_code: int) -> None:
            self.status_code = status_code

    def __init__(self, message: str, status_code: int) -> None:
        super().__init__(message)
        self.response = self._Response(status_code)


class DuplicateLiveAttempt(RuntimeError):
    """Same name as the real one; classification is by type name."""


class TestClientErrorsAreNotRetried:
    @pytest.mark.parametrize("status", [400, 401, 403, 404, 409, 422])
    def test_a_client_error_is_permanent(self, status):
        assert is_retryable_exception(_WithStatus("nope", status)) is False

    def test_the_observed_context_name_rejection_is_permanent(self):
        # The exact failure that occupied the fleet. Nothing in its text looks
        # permanent, which is why prose matching retried it.
        exc = _WithStatus(
            "POST /project/Assistants/contexts failed with status code 400: "
            '{"detail":"Invalid context name. Names can only contain '
            "alphanumeric characters, underscores, dashes, and forward "
            'slashes. Consecutive slashes are not allowed."}',
            400,
        )
        assert is_retryable_exception(exc) is False

    @pytest.mark.parametrize("status", [408, 429])
    def test_the_two_retryable_client_codes_are_retried(self, status):
        # Timeout and rate limit are the only 4xx a repeat can resolve.
        assert is_retryable_exception(_WithStatus("slow down", status)) is True


class TestServerErrorsAreRetried:
    @pytest.mark.parametrize("status", [500, 502, 503, 504])
    def test_a_server_fault_is_retried(self, status):
        assert is_retryable_exception(_WithStatus("upstream", status)) is True

    def test_a_status_on_a_response_object_is_found(self):
        assert is_retryable_exception(_WithResponse("boom", 503)) is True
        assert is_retryable_exception(_WithResponse("nope", 400)) is False


class TestTheTextFallbackNoLongerMisfires:
    def test_a_row_count_that_looks_like_a_status_is_not_retried(self):
        # The mirror-image bug: no status to read, and the digits win.
        assert is_retryable_exception(RuntimeError("committed 500 rows")) is False

    def test_digits_that_look_like_a_status_are_not_retried(self):
        # Neither of these is a server fault, and both used to be retried purely
        # because the digits appeared somewhere in the text.
        assert is_retryable_exception(RuntimeError("wrote to shard 502")) is False
        assert (
            is_retryable_exception(RuntimeError("connect to host:502 failed")) is False
        )

    def test_the_transport_phrase_is_what_earns_a_retry(self):
        # Same digits, but now with wording that genuinely indicates transport
        # trouble -- which is the signal, rather than the number.
        assert is_retryable_exception(RuntimeError("connection reset on :502")) is True

    def test_transport_phrases_still_retry_without_a_status(self):
        for text in (
            "operation timed out",
            "connection reset by peer",
            "service unavailable",
            "rate limit exceeded",
        ):
            assert is_retryable_exception(RuntimeError(text)) is True, text

    def test_transport_exception_types_still_retry(self):
        assert is_retryable_exception(TimeoutError()) is True
        assert is_retryable_exception(ConnectionError()) is True
        assert is_retryable_exception(OSError("disk")) is True

    def test_an_empty_message_is_not_retried(self):
        assert is_retryable_exception(RuntimeError("")) is False


class TestALiveAttemptCollisionYields:
    def test_a_duplicate_attempt_is_not_retried(self):
        # Another attempt holds the lease and is committing. Retrying can only
        # contend with it; the caller should stand down and let it finish.
        exc = DuplicateLiveAttempt("Table 'table:1' is already being ingested")
        assert is_retryable_exception(exc) is False

    def test_it_is_decided_before_the_text_is_consulted(self):
        # Its message mentions neither a status nor a transport phrase, so this
        # pins the type check rather than an accident of wording.
        exc = DuplicateLiveAttempt("timed out waiting for the lease")
        assert is_retryable_exception(exc) is False
