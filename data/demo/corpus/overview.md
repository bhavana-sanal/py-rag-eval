# Tidewell overview

Tidewell is a self-hosted background job queue for services written in Python and Go. Applications enqueue small units of work, called jobs, and a pool of worker processes executes them outside the request path. Typical uses are sending emails, resizing images, syncing data with third-party APIs and running nightly reports.

The server is a single binary named `tidewelld`, written in Rust. It exposes an HTTP API for producers and workers, runs the cron scheduler and cleans up finished jobs. All state lives in a storage backend, either PostgreSQL or an embedded SQLite file, so the server itself is stateless and can be restarted at any time without losing work.

## Design goals

Tidewell favours predictable behaviour over raw throughput. Every job is delivered at least once, retries follow a documented backoff schedule, and every state change is recorded with a timestamp. The project was first released in 2022 and is distributed under the MIT license.

## Where to go next

Read the installation guide to start a local server, then the client libraries page to enqueue your first job.
