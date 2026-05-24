# Helix Platform — Architecture Overview

## Components

Helix is a three-tier architecture:

- **Data plane** — handles record ingestion, storage, and query execution. Built on Apache Kafka, Apache Flink, and a custom columnar store named `Hyperion`.
- **Control plane** — exposes the public API, manages dataset metadata, enforces access control, and orchestrates resources. Built on FastAPI and Postgres.
- **Console** — the React-based admin UI for browsing datasets, running queries, and managing IAM.

## Data Flow

1. Client posts records to `POST /v3/datasets/{id}/records`.
2. The control plane validates the schema and writes records to a per-dataset Kafka topic.
3. A Flink job consumes the topic, partitions records by `partition_key`, and writes to Hyperion.
4. Hyperion writes immutable columnar segments to S3 and updates the manifest in Postgres.
5. Queries read manifest, push down predicates, and stream results back through the control plane.

## Hyperion Internals

Hyperion stores data in segments of approximately 1GB. Each segment contains:

- A column file per dataset field (Zstandard-compressed)
- A bloom filter for high-cardinality columns
- A min/max statistics block for skip-scan optimization
- A row index for point lookups

Compaction runs nightly: small segments under 256MB are merged into larger ones. The compaction worker maintains a target of 1 segment per ~50 million rows.

## Latency Budget

The published p99 latency targets are:

- Ingestion (record accepted): 250 ms
- Query (scan up to 100 GB): 8 seconds
- Query (scan over 100 GB): 30 seconds
- Console page load: 1.5 seconds

If any p99 exceeds the budget for 15 consecutive minutes, the SRE on-call is paged automatically.

## Multi-Region

Helix runs in three regions: `us-east-1` (primary), `eu-west-1`, and `ap-southeast-1`. Datasets are pinned to a single region; cross-region replication is offered as an Enterprise feature with eventual consistency (RPO 5 minutes).

The control plane is replicated globally via Postgres logical replication. Writes always go to `us-east-1`; reads can be served from any region.
