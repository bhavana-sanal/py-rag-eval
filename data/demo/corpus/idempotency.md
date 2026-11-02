# Idempotency and duplicate jobs

Tidewell guarantees at-least-once delivery. A job can run more than once, for example when a worker crashes after finishing the work but before reporting success, and the lease expires. Handlers should therefore be idempotent: running them twice must have the same effect as running them once.

## Writing idempotent handlers

Common techniques are recording a processed marker keyed by the job id inside the same database transaction as the work, using upserts instead of inserts, and passing the job id as an idempotency key to payment or email providers that support one.

## Unique jobs

To avoid enqueuing the same work twice in the first place, a producer can set a `unique_key` when enqueuing. While a job with the same queue and key exists and is not in a terminal state, or finished less than `unique_for` ago, a second enqueue does not create a new job. The default `unique_for` window is one hour. Instead the API returns the existing job's id with HTTP 200 rather than 201, so the producer can tell the difference.
