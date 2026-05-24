# Helix Platform — Engineering Setup Guide

This guide walks new engineers through setting up a local Helix development environment.

## Prerequisites

- macOS 13+ or Ubuntu 22.04+
- Python 3.11 or higher
- Docker Desktop 4.30+
- Node.js 20.x (for the admin console)
- A GitHub account added to the `helix-eng` organization

## Cloning the Repositories

The Helix codebase is split across three repositories:

- `helix-core` — the data plane and ingestion services
- `helix-control` — the control plane API
- `helix-console` — the React admin dashboard

Clone all three under `~/code/helix/` and run `make bootstrap` from `helix-core`.

## Environment Variables

Copy `.env.example` to `.env` in each repo. The required variables are:

- `HELIX_DATABASE_URL` — Postgres connection string. Default: `postgres://helix:helix@localhost:5432/helix`
- `HELIX_REDIS_URL` — Redis connection string. Default: `redis://localhost:6379/0`
- `HELIX_KAFKA_BROKERS` — comma-separated list of Kafka brokers. Default: `localhost:9092`
- `HELIX_S3_BUCKET` — name of the dev artifact bucket. Use `helix-dev-<your-username>`.
- `HELIX_AUTH_TOKEN` — issued via `helix auth login`; expires every 30 days.

## Bootstrapping the Stack

Run `docker compose up -d` from the root of `helix-core`. This launches Postgres, Redis, Kafka, and Zookeeper.

After containers are healthy, run database migrations with `helix db migrate`. Seed development data with `helix db seed --profile dev`.

## Running Tests

The full test suite is `pytest tests/ -m "not slow"`. The slow suite — including integration tests against real Kafka — runs in CI only. Unit tests must pass locally before opening a pull request.

## Common Issues

If `helix db migrate` fails with `relation "schema_migrations" does not exist`, run `helix db reset --force` first.

If Kafka refuses connections during integration tests, check that `KAFKA_ADVERTISED_LISTENERS` resolves to `localhost`, not the container name.
