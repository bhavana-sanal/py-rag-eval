# Delayed and recurring jobs

## Delayed jobs

A job can be enqueued with a `run_at` timestamp. It stays in the `scheduled` state until that moment and then becomes `pending`. Delays are useful for reminders, for example sending a follow-up email three days after a signup.

## Cron schedules

Recurring jobs are declared in the configuration file as `[[schedules]]` entries. Each entry names a queue, a payload and a standard five-field cron expression such as `0 3 * * *` for every day at 03:00. Schedules are evaluated in UTC unless the entry sets a `timezone` such as `Europe/Berlin`, in which case daylight-saving transitions are handled according to that zone.

Only the cluster leader evaluates schedules, so a recurring job is enqueued once per tick even when several servers are running.

## Missed runs

If the server was down when a schedule should have fired, the `misfire` policy decides what happens on restart. The default, `skip`, ignores missed ticks and waits for the next one. The alternative, `run_once`, enqueues a single catch-up job no matter how many ticks were missed. Tidewell never enqueues one job per missed tick, because a long outage would otherwise flood the queue.
