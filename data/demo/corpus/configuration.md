# Configuration

Tidewell reads its settings from a TOML file, from environment variables and from command-line flags. When the same setting appears in several places, command-line flags win over environment variables, environment variables win over the file, and the file wins over built-in defaults.

## The configuration file

The server looks for `/etc/tidewell/tidewell.toml` unless a path is passed with `--config`. Settings are grouped into sections such as `[server]`, `[storage]`, `[log]`, `[retention]` and one `[queues.<name>]` table per queue that needs non-default behaviour.

## Environment variables

Every setting can be overridden with an environment variable that starts with `TIDEWELL_`. Nested keys are joined with a double underscore, so `storage.url` becomes `TIDEWELL_STORAGE__URL` and `log.level` becomes `TIDEWELL_LOG__LEVEL`. This is the recommended way to pass database passwords in container deployments.

## Validating a configuration

`tidewell config check` parses the file, applies environment overrides and reports unknown keys, invalid values and unreachable storage without starting the server. Running it in CI before a deploy catches most configuration mistakes early. Unknown keys are treated as errors rather than warnings because a misspelled key would otherwise silently fall back to its default.
