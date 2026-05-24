# Helix Platform — Billing & Glossary

## Pricing Tiers

Helix offers three tiers:

- **Starter** — $0/month. Up to 10GB stored, 1M records/month ingested, 100GB scanned/month. Community support only.
- **Growth** — $999/month base. 1TB stored included, then $0.025/GB-month. Ingest at $0.10 per million records. Query scans $5 per TB. 8x5 email support with 24-hour SLA.
- **Enterprise** — custom pricing. Includes BYOK, multi-region replication, dedicated infrastructure, 24x7 support with 1-hour SLA, and private bug bounty access.

## Billing Cycle

Invoices are issued on the 1st of each month for the prior month's usage. Net-30 payment terms by default; net-60 is available for Enterprise customers with annual prepayment.

Usage exceeding the included limits in Growth tier is billed at the overage rates above. There is no hard cap — customers are notified at 80%, 100%, and 150% of expected monthly spend.

## Glossary

- **Dataset** — a logical collection of records with a single schema. Analogous to a table in a relational database.
- **Workspace** — a tenant boundary. All datasets in a workspace share IAM and billing.
- **Schema** — JSON Schema document describing record fields. Schemas may evolve via additive changes only; removing fields requires creating a new dataset.
- **Partition key** — a top-level field used to physically partition Hyperion segments. Choosing a good partition key dramatically reduces `bytes_scanned` for filtered queries.
- **Manifest** — Postgres-stored index mapping dataset partitions to Hyperion segment files in S3.
- **Cold tier** — segments older than 90 days are migrated to S3 Glacier Instant Retrieval. Queries against cold-tier data carry a 2x bytes-scanned multiplier.
- **BYOK** — "Bring Your Own Key". Enterprise feature where the customer's KMS key encrypts their data; Helix never holds the unencrypted key.
- **Workspace owner** — the IAM principal with full administrative rights over a workspace, including billing and member management.

## Refund Policy

Helix does not issue refunds for unused capacity in monthly billing cycles. Annual prepayments may be refunded pro-rata if the customer terminates the contract for cause; refunds for convenience are at Helix's discretion.

Service credits for SLA violations are issued automatically if the monthly availability falls below the SLA target. The credit equals 10% of the monthly fee per 0.1% of availability shortfall, up to a maximum of 50% of the monthly fee.
