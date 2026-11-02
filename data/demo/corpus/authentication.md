# Authentication and access control

Every API request is authenticated with a bearer token. Tokens are created with `tidewell tokens create` and are shown exactly once; the server stores only an Argon2id hash, so a leaked database backup does not reveal usable tokens.

## Scopes

Each token has one or more scopes that limit what it can do:

- `enqueue` may create jobs, which is all a typical producer service needs.
- `read` may list and inspect jobs and queues, for dashboards and on-call tooling.
- `admin` may do everything, including pausing queues, replaying dead jobs and creating tokens.

Workers need a token with both `enqueue` and `read` scopes, because handlers often enqueue follow-up jobs. Following least privilege, producers should not receive `admin` tokens.

## Mutual TLS

In addition to tokens, the server can require workers to present a client certificate signed by a configured certificate authority. Enable it with `server.tls.client_ca`. Mutual TLS is recommended when workers run on networks shared with other tenants.

Tokens can be revoked at any time with `tidewell tokens revoke`, which takes effect immediately on all servers in a cluster.
