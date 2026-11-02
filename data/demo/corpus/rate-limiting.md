# Rate limiting

Some jobs call external services that enforce their own quotas. A queue-level rate limit keeps Tidewell from dispatching jobs faster than such a service accepts them.

## Configuring a limit

Add a `rate_limit` to the queue's table in the configuration file, written as a count per unit of time, for example `rate_limit = "100/m"` for one hundred jobs per minute or `"5/s"` for five per second. The limit is enforced with a token bucket shared by all servers in the cluster. The optional `burst` setting controls how many jobs may start back to back after an idle period; it defaults to the per-unit count.

## Effect on jobs

A job that is held back by the rate limit stays in the `pending` state until a token is available. Waiting for a token is not an attempt, so throttling never pushes jobs toward the dead state. This is the main difference from handling HTTP 429 responses inside the handler, where every rejected call would consume a retry.

Rate limits apply to the start of a job, not to individual calls made by the handler. A handler that makes several API calls per job needs a correspondingly lower rate.
