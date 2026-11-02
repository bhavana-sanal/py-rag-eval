# Logging

`tidewelld` writes structured logs as one JSON object per line to standard output, which works with any log collector. Each entry contains a timestamp, level, message and, where relevant, the job id and queue name, so that the full history of a job can be found with a single search.

## Log levels

The level is set with `log.level` in the configuration file or the `TIDEWELL_LOG__LEVEL` environment variable. Supported levels are `debug`, `info`, `warn` and `error`; the default is `info`. The `debug` level logs every lease and heartbeat and is too verbose for production.

## Redacting sensitive fields

Payloads are not logged by default. When `log.payloads = true` is set for debugging, fields listed in `log.redact_fields`, for example `["password", "card_number", "token"]`, are replaced with the string `[redacted]` before the line is written. Redaction matches field names at any nesting depth.

Workers written with the official client libraries use the same JSON format, so server and worker logs can be correlated by job id.
