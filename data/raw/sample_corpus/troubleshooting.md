# Helix — Troubleshooting Guide

Common issues and their fixes.

## Ingestion

### `INGESTION_BACKPRESSURE` (503)

Indicates the Kafka topic for the dataset is at capacity. Helix auto-scales topic partitions but the scale-up takes up to 90 seconds. Clients should implement exponential backoff with a base of 60s and a max of 600s.

If backpressure persists for more than 10 minutes, page the data-plane on-call.

### Records silently dropped

Records that fail schema validation are returned in the `POST /v3/datasets/{id}/records` response with index and reason. They are NOT retried. If you see fewer rows than expected, inspect the response payload — silent drops are not a Helix bug.

## Queries

### `QUERY_TIMEOUT` (504)

Default query timeout is 60 seconds. To extend, pass `?timeout_seconds=300` (max 600). Queries scanning more than 1TB are billed at the standard rate even if they timeout.

### Cache misses on identical queries

Helix caches query results for 5 minutes keyed by (dataset_id, query_hash, schema_version). If schema changes, the cache is invalidated. To force a cache bypass, send `Cache-Control: no-cache` with the request.

## Auth

### "Token expired" right after login

Check the system clock on the client machine. Tokens validate `exp` against UTC; clock drift of more than 30 seconds will reject otherwise valid tokens.

### Persistent 403 INSUFFICIENT_SCOPE

Token scopes are issued at token creation time and cannot be modified afterward. Generate a new token with `helix auth login --scopes datasets:write,datasets:read`.

## Local Dev

### Kafka container exits immediately

The Confluent Kafka image requires `KAFKA_PROCESS_ROLES` to be set in single-node mode. Pull the latest `docker-compose.yml` from `helix-core@main` — older versions of the file are missing this variable.

### Port 5432 already in use

Another Postgres is already bound. Either stop the local instance (`brew services stop postgresql`) or override the port with `HELIX_DATABASE_URL=postgres://helix:helix@localhost:5433/helix` and update `docker-compose.yml` accordingly.
