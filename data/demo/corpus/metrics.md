# Metrics

`tidewelld` exposes Prometheus metrics at `/metrics` on port 7421, separate from the API port so that the metrics endpoint can stay inside a private network.

## Key metrics

- `tidewell_jobs_enqueued_total`: counter of enqueued jobs, labelled by queue.
- `tidewell_jobs_completed_total`: counter of finished attempts, labelled by queue and `status` (`succeeded`, `failed` or `dead`).
- `tidewell_job_duration_seconds`: histogram of handler run time per queue.
- `tidewell_queue_depth`: gauge of pending jobs per queue.
- `tidewell_oldest_pending_age_seconds`: gauge of how long the oldest pending job in each queue has been waiting.

## What to alert on

Queue depth alone is a weak signal because a burst of enqueues can raise it without anything being wrong. The age of the oldest pending job is a better indicator of a stuck or under-provisioned queue, since it grows only when work is not being picked up. A common rule fires when that age stays above five minutes for ten minutes. Pair it with an alert on the rate of dead jobs.
