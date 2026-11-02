# Job priorities

Every job has a priority from 1 to 10. Higher numbers run first, and the default priority is 5. Within the same priority, jobs are handed out in the order they became pending.

## Choosing priorities

Use high priorities for work a user is actively waiting on, such as a password-reset email, and low priorities for bulk work such as reindexing. Priorities only order jobs within a single queue; they do not let one queue take capacity from another. To isolate heavy workloads, put them in their own queue with dedicated workers.

## Starvation protection

A queue that receives a steady stream of high-priority jobs could otherwise delay low-priority jobs forever. To prevent this, Tidewell ages waiting jobs: for every 60 seconds a job spends pending, its effective priority rises by one, up to the maximum of 10. The stored priority does not change; aging only affects the dispatch order. Aging can be disabled per queue with `priority_aging = false` when strict ordering matters more than fairness.
