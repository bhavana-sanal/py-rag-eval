# Workers

A worker is a process that runs your handler code. Workers connect to `tidewelld` over HTTP, long-poll for jobs on the queues they subscribe to and report the result of each attempt.

## Leases and heartbeats

When a worker receives a job, it also receives a lease that is valid for 30 seconds. While the handler runs, the worker sends a heartbeat every 10 seconds and each heartbeat extends the lease. If the worker crashes or loses its network connection, the heartbeats stop and the lease expires. The server then records the attempt as failed with the error `lease expired` and schedules a retry like any other failure.

This mechanism is what makes delivery at-least-once: a job whose worker vanished halfway through will run again on another worker.

## Graceful shutdown

On SIGTERM a worker stops fetching new jobs and waits for running handlers to finish. If they have not finished after `shutdown_timeout`, which defaults to 30 seconds, the worker exits and the leases of the unfinished jobs expire normally. Set the shutdown timeout below your orchestrator's kill grace period so that workers are not killed mid-drain.
