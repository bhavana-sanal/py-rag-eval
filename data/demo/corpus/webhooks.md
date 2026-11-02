# Webhooks

Webhooks let external systems react to job events without polling the API. They are configured per queue with a URL, a secret and the list of events to deliver.

## Events

Two events are available: `job.succeeded` and `job.dead`. Most teams subscribe only to `job.dead`, for example to open a ticket automatically when a billing job gives up.

## Verifying signatures

Each delivery is an HTTP POST with a JSON body describing the job. The request carries a `Tidewell-Signature` header of the form `t=<unix timestamp>,v1=<hex digest>`. The digest is an HMAC-SHA256 of the timestamp, a period and the raw request body, computed with the webhook's secret. Receivers should recompute the digest, compare it in constant time and reject deliveries whose timestamp is more than five minutes old to block replay attacks.

## Delivery guarantees

A delivery succeeds when the receiver answers with any 2xx status. Failed deliveries are retried up to 3 times, one minute apart, after which the failure is logged and the event is dropped. Webhooks are therefore best-effort notifications and should not be the only record of a job's outcome.
