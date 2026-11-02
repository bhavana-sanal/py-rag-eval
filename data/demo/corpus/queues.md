# Queues

Every job belongs to exactly one named queue. Queues group related work so that it can be paused, rate limited and scaled independently. A worker subscribes to one or more queues and only receives jobs from those queues.

## Creating queues

Queues are created automatically the first time a job is enqueued to them, so there is no separate setup step. A queue name may contain lowercase letters, digits and hyphens and can be at most 64 characters long. Names such as `emails`, `image-resize` and `billing-sync` are typical.

## Pausing and resuming

`tidewell queues pause emails` stops the server from handing out jobs from the `emails` queue. A paused queue still accepts new jobs, which simply wait in the pending state. `tidewell queues resume emails` restarts dispatching. Pausing is useful during an incident in a downstream system: work accumulates safely instead of failing and burning through retry attempts.

## Deleting queues

A queue can only be deleted with `tidewell queues delete` when it holds no pending, scheduled or running jobs. This prevents work from being dropped by accident.
