# HTTP API

All endpoints live under the `/v1` prefix and exchange JSON. Requests must carry an API token in an `Authorization: Bearer` header.

## Enqueuing

`POST /v1/jobs` creates a job. The body contains the `queue`, the `payload` and optional `priority`, `run_at`, `max_attempts` and `unique_key` fields. A new job is answered with HTTP 201 and its id. `POST /v1/jobs:batch` accepts up to 500 jobs in one request and either creates all of them or none.

## Reading and cancelling

`GET /v1/jobs/{id}` returns a job including its state and the error of each failed attempt. `GET /v1/jobs` lists jobs and accepts `queue` and `state` filters. Lists are paginated with an opaque `cursor`: each response includes `next_cursor`, which is passed back to fetch the following page. The page size defaults to 50 and is capped at 200.

`DELETE /v1/jobs/{id}` cancels a job that is pending or scheduled. Cancelling a running job is not possible and returns HTTP 409, because the handler may already have produced side effects.

## Errors

Errors use a consistent body with a machine-readable `code` and a human-readable `message`, for example `payload_too_large` together with status 413.
