# Storage backends

Tidewell keeps every job, queue and schedule in a storage backend selected by the scheme of `storage.url`.

## PostgreSQL

PostgreSQL 14 or newer is the recommended backend for production. Workers claim jobs with `SELECT ... FOR UPDATE SKIP LOCKED`, which lets many servers dispatch from the same table without blocking each other. PostgreSQL is also required for running more than one server, because leader election uses its advisory locks.

## SQLite

The embedded SQLite backend stores everything in a single file and needs no separate database server. It runs in write-ahead logging mode so that reads do not block the writer. SQLite is intended for development, tests and small single-node deployments; it cannot be shared by several servers.

## Unsupported databases

MySQL and MariaDB are not supported. Their locking behaviour differs enough that the dispatch queries would need a separate implementation, and the maintainers have chosen to focus on the two backends above. Redis is also not supported as primary storage, because Tidewell relies on transactions spanning several tables.
