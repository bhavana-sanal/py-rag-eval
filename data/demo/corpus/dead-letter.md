# Dead jobs

A job becomes dead when it has used all of its attempts or when its handler raised a permanent error. Dead jobs are not deleted immediately. They are kept for 14 days so that an engineer can inspect the error history and decide what to do.

## Inspecting dead jobs

`tidewell dead list --queue billing-sync` shows dead jobs with the error message from each attempt. The same information is available from the HTTP API by filtering jobs on `state=dead`.

## Replaying

After fixing the underlying bug, `tidewell dead replay --queue billing-sync` moves the dead jobs of that queue back to `pending`. Replaying resets the attempt counter to zero, so each replayed job receives the full retry budget again. A single job can be replayed by passing `--id` with its identifier instead of a queue.

## Alerting

Because dead jobs represent work that did not happen, most teams alert on them. The metric `tidewell_jobs_completed_total` carries a `status` label, and an alert on the rate of `status="dead"` is a good starting point. Webhooks can also notify an external system whenever a job dies.
