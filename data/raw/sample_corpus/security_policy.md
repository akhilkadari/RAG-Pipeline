# Helix Platform — Security Policy

This document defines the security controls for the Helix production environment.

## Access Control

All production access requires SSO through Okta with hardware-backed MFA (YubiKey or platform authenticator). SMS-based MFA is forbidden for engineering accounts.

Just-in-time access to production is granted via the `helix-jit` tool and expires after 4 hours. JIT requests must be approved by a member of the on-call rotation in addition to the requesting engineer.

## Secret Management

Secrets are stored in HashiCorp Vault under `secret/helix/<env>/<service>`. Direct database credentials are forbidden in code; services obtain dynamic credentials from Vault at startup.

Static credentials in environment variables are permitted only for development. Any commit containing a string matching the Vault token format (`hvs.*`) is rejected by the pre-commit hook.

## Encryption

Data is encrypted at rest using AES-256-GCM. Per-dataset encryption keys are managed by AWS KMS with annual rotation. Customers on the Enterprise tier may bring their own KMS key (BYOK) — see `byok-onboarding.md`.

In transit, all Helix services require TLS 1.3. TLS 1.2 is permitted only for legacy customer integrations and must be approved by Security.

## Audit Logging

Every action against the control plane API generates an audit log entry written to the `helix-audit` Kafka topic. Audit logs are retained for 7 years per SOC 2 requirements.

Audit logs include: actor identity, action verb, resource ID, source IP, request ID, and result (success/failure). Sensitive payloads (records, query results) are NOT logged — only metadata is.

## Incident Response

Suspected security incidents must be reported within 30 minutes via the `#sec-incident` Slack channel and the on-call security engineer paged through PagerDuty.

The incident commander is the on-call security engineer until they explicitly hand off to another responder. Engineering responders must follow the directives of the IC without exception during an active incident.

## Vulnerability Disclosure

External researchers may submit vulnerabilities through `security@helix.example.com`. Helix commits to acknowledging reports within 48 hours and providing a status update within 7 days.

Helix runs a private bug bounty program through HackerOne. Payouts range from $250 (low) to $25,000 (critical) per the bounty rubric.
