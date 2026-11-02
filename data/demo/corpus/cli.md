# Command-line tool

The `tidewell` command talks to a running server over the HTTP API and also performs a few offline tasks. It reads the server address from `TIDEWELL_URL` and the API token from `TIDEWELL_TOKEN`.

## Common commands

- `tidewell init` writes a starter configuration file.
- `tidewell migrate` applies pending database migrations.
- `tidewell config check` validates a configuration without starting the server.
- `tidewell enqueue <queue> '<json>'` enqueues a job, which is handy for testing handlers.
- `tidewell jobs list --queue <queue> --state <state>` lists jobs with optional filters.
- `tidewell jobs cancel <id>` cancels a job that has not started yet.
- `tidewell queues pause <queue>` and `tidewell queues resume <queue>` control dispatching.
- `tidewell dead replay --queue <queue>` moves dead jobs back to pending.
- `tidewell tokens create --scope <scope>` creates an API token.
- `tidewell backup <path>` writes an online snapshot of a SQLite database.

Every command accepts `--output json` to print machine-readable output for scripts. Exit code 0 means success, 1 means the server rejected the request and 2 means the command line itself was invalid.
