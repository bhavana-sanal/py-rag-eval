# Installing Tidewell

Prebuilt `tidewelld` binaries are published for Linux (x86-64 and ARM64) and macOS. An official container image, `tidewell/tidewelld`, is built from the same release artifacts.

## Requirements

A production deployment needs PostgreSQL 14 or newer. For local development the embedded SQLite backend works with no extra services. A small deployment handling a few hundred jobs per second runs comfortably on 2 CPU cores and 512 MB of memory.

## First start

1. Run `tidewell init` to write a starter configuration file to the current directory.
2. Edit the `storage.url` setting to point at your database.
3. Run `tidewell migrate` to create the schema.
4. Start the server with `tidewelld --config tidewell.toml`.

By default the HTTP API listens on port 7420 and the Prometheus metrics endpoint listens on port 7421. Both ports can be changed in the `[server]` section of the configuration.

## Verifying the install

`curl localhost:7420/healthz` returns `ok` once the server has connected to storage and applied all migrations. If migrations are missing the endpoint returns HTTP 503 and the log explains which migration is pending.
