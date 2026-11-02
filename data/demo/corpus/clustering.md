# Running a cluster

Several `tidewelld` servers can run against the same PostgreSQL database for high availability. Any server can accept enqueues and dispatch jobs to workers, so a load balancer can spread traffic across all of them.

## Leader election

A few tasks must run on exactly one server: evaluating cron schedules, enforcing retention and expiring leases. The server that holds a PostgreSQL advisory lock is the leader and performs them. Because the lock belongs to a database session, it is released automatically when the leader's connection drops. Another server acquires it on its next attempt, so a new leader takes over within about 15 seconds of a crash.

## Requirements

All servers in a cluster must run the same minor version, except briefly during a rolling upgrade, and their clocks should be synchronised with NTP. Clustering is not available with the SQLite backend.

## Sizing

Servers are stateless, so adding servers increases API capacity, but dispatch throughput is ultimately limited by the database. Monitor database CPU before adding more servers.
