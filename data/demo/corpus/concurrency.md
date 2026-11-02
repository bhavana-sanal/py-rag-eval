# Concurrency limits

Tidewell offers two independent concurrency controls: one per worker process and one per queue across the whole cluster.

## Per-worker concurrency

The `concurrency` setting of a worker is the number of jobs it runs at the same time. It defaults to 4. Python workers run handlers in a thread pool of that size, and Go workers start that many goroutines. Raise it for handlers that mostly wait on network calls and keep it low for CPU-heavy handlers.

## Per-queue limits

The queue setting `max_running` caps how many jobs from that queue may be running at once across all workers. For example, `max_running = 2` on a `report-export` queue guarantees that no more than two exports hit the reporting database at the same time, however many workers are deployed. Jobs that cannot start because of the cap simply stay pending.

## Choosing values

Start with the defaults, watch the `tidewell_queue_depth` and `tidewell_job_duration_seconds` metrics, and adjust. A growing queue depth with idle CPU usually means concurrency is too low; rising job durations usually mean a downstream system is saturated.
