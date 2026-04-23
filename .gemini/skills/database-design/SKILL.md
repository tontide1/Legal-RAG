---
name: database-design
description: Database design principles and decision-making. Use this skill when designing schemas, selecting ORMs, tuning performance with indexes, or choosing between database types (SQL vs NoSQL).
---

# Database Design

## Overview

This skill covers database design principles, schema modeling, and technology selection for modern applications. It emphasizes choosing the right tool for the job based on context rather than defaulting to a single technology.

## Decision Workflow

- Ask the user for database preferences when unclear.
- Choose between PostgreSQL, Neon, Turso, or SQLite based on deployment environment and scale.
- Select ORM (Drizzle, Prisma, Kysely) according to project needs.
- Design normalized schemas with clear primary keys and relationships.
- Plan indexing strategies (composite, GIN, BRIN) for performance tuning.

## Best Practices

- Use `EXPLAIN ANALYZE` for query optimization.
- Avoid `SELECT *` in production code.
- Store structured data correctly instead of defaulting to JSON.
- Implement safe migrations and plan for serverless database environments.
- Monitor and fix N+1 query issues.

## Anti-Patterns to Avoid

- Defaulting to PostgreSQL for simple applications where SQLite would suffice.
- Skipping proper indexing for frequently queried columns.
- Ignoring data relationship types during initial schema design.
