# Helix Platform — Public API Reference (v3)

All endpoints live under `https://api.helix.example.com/v3` and require a bearer token in the `Authorization` header.

## Authentication

Tokens are obtained from `POST /v3/auth/token` with a client_id and client_secret. Tokens expire after 60 minutes. Refresh tokens are not supported in v3 — clients must re-authenticate.

Rate limits: 600 requests per minute per token. Exceeding the limit returns `429 Too Many Requests`.

## Datasets

### `POST /v3/datasets`

Create a new dataset.

Request body:

- `name` (string, required) — must match `^[a-z][a-z0-9_]{2,63}$`
- `schema` (object, required) — JSON schema describing record fields
- `retention_days` (integer, optional) — defaults to 90, max 730
- `partition_key` (string, optional) — must be a top-level field in the schema

Returns `201 Created` with the created dataset, including a generated `dataset_id`.

### `GET /v3/datasets/{dataset_id}`

Fetch dataset metadata.

### `DELETE /v3/datasets/{dataset_id}`

Soft-deletes the dataset. Records are purged after 30 days. The dataset name remains reserved during this window.

## Records

### `POST /v3/datasets/{dataset_id}/records`

Insert a batch of records. The request body must contain a `records` array with at most 1000 entries. Each record must conform to the dataset schema. Failed records are returned in the response with their index and error reason.

### `POST /v3/datasets/{dataset_id}/query`

Run a SQL query against the dataset. Helix supports a subset of ANSI SQL including JOINs across datasets in the same workspace. Window functions are supported; recursive CTEs are not.

The response includes `rows`, `row_count`, `bytes_scanned`, and `cache_hit`. Queries are billed by `bytes_scanned`.

## Common Errors

- `400 SCHEMA_MISMATCH` — record fields don't match the dataset schema
- `403 INSUFFICIENT_SCOPE` — token does not have `datasets:write`
- `409 NAME_RESERVED` — dataset name was used recently and is in the soft-delete window
- `429 Too Many Requests` — rate-limited
- `503 INGESTION_BACKPRESSURE` — Kafka topic is at capacity; retry after 60s
