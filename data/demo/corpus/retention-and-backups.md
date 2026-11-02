# Retention and backups

## Retention

Finished jobs are deleted automatically so that the jobs table does not grow forever. By default succeeded and cancelled jobs are kept for 7 days and dead jobs for 14 days. The periods are set in the `[retention]` section, for example `succeeded = "3d"`. The cleanup task, called the janitor, runs on the cluster leader once per hour and deletes in small batches to avoid long-running transactions.

## Backing up PostgreSQL

With PostgreSQL, use the database's own tooling: `pg_dump` for logical backups or continuous archiving for point-in-time recovery. Tidewell needs no special treatment, because all of its state is in the database.

## Backing up SQLite

Copying a SQLite file while the server is writing to it can produce a corrupt copy. Use `tidewell backup <path>` instead, which takes a consistent online snapshot without stopping the server.

## Restoring

Restore the database, run `tidewell migrate` if the backup came from an older version, and start the server. Jobs that were running when the backup was taken will be in the `running` state with expired leases and are retried automatically.
