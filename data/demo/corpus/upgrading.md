# Upgrading

Tidewell follows semantic versioning. Patch releases contain only fixes, minor releases may add features and database migrations, and breaking changes are reserved for major releases and listed in the changelog.

## Upgrade procedure

1. Back up the database.
2. Run `tidewell migrate` using the new version of the command-line tool.
3. Replace the server binaries or container images.

Migrations are written so that the previous minor version keeps working against the new schema. This makes rolling upgrades possible: servers can be replaced one at a time while the cluster keeps serving traffic. Skipping minor versions in a rolling upgrade is not supported; upgrade through each minor version in turn, or stop all servers first.

## Downgrading

Migrations are forward-only. To return to an older version after migrating, restore the backup taken before the upgrade. For this reason the backup step should never be skipped.

## Client compatibility

The client libraries are compatible with any server of the same major version, so workers and producers can be upgraded independently of the servers.
