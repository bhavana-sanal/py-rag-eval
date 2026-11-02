# Troubleshooting

## Jobs stay in the running state

A job remains `running` until its worker reports a result or its lease expires. If a worker was killed, the job is retried once the 30-second lease runs out. If jobs stay running much longer, check that the leader is healthy, since lease expiry is enforced by the leader.

## Queue depth keeps growing

Check, in order: whether the queue is paused (`tidewell queues pause` is easy to forget after an incident), whether a `rate_limit` or `max_running` cap is holding jobs back, and whether any workers are subscribed to the queue at all. The `tidewell_oldest_pending_age_seconds` metric shows whether jobs are moving slowly or not at all.

## Clock skew warnings

Each server compares its clock with the database clock on start-up and every minute afterwards. A difference of more than 2 seconds produces a `clock skew` warning, because scheduled jobs and lease expiry depend on consistent time. Fix it by enabling NTP on the host.

## Payload too large

HTTP 413 with the code `payload_too_large` means the payload exceeded the 1 MiB default limit. Store the data elsewhere and enqueue a reference to it.
