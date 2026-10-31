---
title: Nimbus Workspace 3.0 Release Notes
version: "3.0"
date: 2025-04-20
kind: release_notes
---

# Nimbus Workspace 3.0 Release Notes

Released 2025-04-20. This release focuses on API scale, security and the administrator experience.

## Highlights

- A redesigned Admin Console with a single page for users, roles and invitations.
- Audit logs can now be exported as CSV for up to 12 months of history.
- SAML single sign-on supports multiple identity providers per workspace.

## API Changes

The API rate limit has been raised to 300 requests per minute per API key. Clients that already handle HTTP 429 responses do not need any changes.

Basic authentication is deprecated and will be removed in version 3.2. All integrations should switch to API keys sent with the Bearer scheme.

New endpoints were added for exporting audit logs: `GET /v3/audit-logs` returns events in pages of 50.

## Security Improvements

Admins can now require multi-factor authentication for every member of a workspace. API keys can be given an expiry date when they are created.

## Fixed Issues

- Fixed an issue where invitation emails were sometimes sent twice.
- Fixed incorrect storage totals for workspaces with files in the trash.
- Fixed a timeout when listing more than 10,000 users.
