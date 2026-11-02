# Jobs and their lifecycle

A job is a record describing one unit of work. It has an identifier, a queue, a JSON payload, a priority, an optional run time and a counter of attempts made so far. Job identifiers are ULIDs, which sort by creation time and can be generated without coordination.

## Payload limits

The payload must be a JSON object and may be at most 1 MiB by default. The limit is controlled by `limits.max_payload_bytes`. Large inputs such as files should be stored elsewhere, for example in object storage, with only a reference placed in the payload. Requests over the limit are rejected with HTTP 413.

## States

A job moves through a small set of states:

- `pending`: ready to run and waiting for a worker.
- `scheduled`: waiting for its run time or for a retry backoff to elapse.
- `running`: leased by a worker.
- `succeeded`: finished without error.
- `failed`: the last attempt raised an error and another attempt will follow.
- `dead`: every attempt failed, or the handler raised a permanent error.
- `cancelled`: removed by a user before it started.

`succeeded`, `dead` and `cancelled` are terminal; a job in a terminal state never runs again unless it is explicitly replayed.
