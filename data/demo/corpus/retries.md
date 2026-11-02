# Retries and backoff

When a handler raises an error, Tidewell records the error message and schedules another attempt. By default a job is attempted at most 5 times in total, counting the first attempt. The limit can be changed per queue with `max_attempts` or per job when it is enqueued.

## Backoff schedule

The delay before the next attempt grows exponentially. The first retry waits 2 seconds, and each later retry waits twice as long as the one before, up to a ceiling of 10 minutes. A random jitter of plus or minus 20 percent is applied to every delay so that jobs which failed together do not all retry at the same instant and overload the dependency that caused the failure.

## Errors that should not be retried

Some failures will never succeed on a later attempt, such as a payload that fails validation. Raising `tidewell.PermanentError` in Python, or returning an error wrapped with `tidewell.Permanent(err)` in Go, moves the job straight to the `dead` state without using its remaining attempts.

## Timeouts

A handler that exceeds the queue's `timeout` (default 5 minutes) is treated as failed, and the attempt counts against the limit like any other error.
