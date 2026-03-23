---
name: end-to-end-software-architecture
description: >-
  Guides end-to-end software architecture: system boundaries, layered design,
  interfaces, data and control flows, deployment topology, cross-cutting concerns
  (security, observability, reliability), and documentation (ADRs, diagrams).
  Use when designing or refactoring systems, drawing architecture diagrams,
  choosing between monolith vs services, defining APIs and contracts, planning
  CI/CD and environments, or when the user mentions system design, C4 model,
  scalability, resilience, or technical debt at the structural level.
---

# End-to-End Software Architecture

## When this skill applies

Use when shaping **how components fit together** from user or external touchpoints through data stores and back—not only class-level design inside one repo. Prefer **explicit boundaries**, **owned interfaces**, and **documented tradeoffs**.

## Core principles

1. **Separation of concerns**: Each module owns one reason to change; dependencies point **inward** toward domain/policy; infrastructure (DB, queues, HTTP) stays at the edges.
2. **Stable abstractions**: Define contracts (API schemas, event payloads, DB views) at boundaries; version or evolve them deliberately.
3. **Minimize coupling**: Prefer composition and interfaces over shared globals; avoid circular imports and hidden singletons across layers.
4. **Design for operability**: Logging, metrics, and health checks are part of the architecture, not an afterthought.

## Layering (typical reference)

Adapt names to the stack; keep direction of dependency clear:

| Layer | Responsibility | Depends on |
|-------|----------------|------------|
| Delivery (UI, CLI, HTTP handlers) | Adapt I/O, auth at edge, validate input | Application / use cases |
| Application / use cases | Orchestrate workflows, transactions | Domain |
| Domain | Business rules, entities, invariants | Nothing infrastructure-specific |
| Infrastructure | DB, messaging, external APIs | Domain interfaces (implementations injected) |

**Rule**: Domain does not import framework-specific types unless the project explicitly uses a different style (e.g. anemic CRUD app)—then document that choice.

## Boundaries and decomposition

**Choose split by**:

- **Team and release independence** → service boundaries (only when payoff exceeds ops cost).
- **Scaling or failure isolation** → separate processes or pools with clear async contracts.
- **Single deployable with clear modules** → monolith or modular monolith first; extract services when metrics justify it.

**Avoid**: Microservices by default without operational maturity; shared databases across services without a documented consistency story.

## Interfaces and contracts

- **Synchronous**: REST or RPC with versioned paths/schemas; OpenAPI or protobuf as source of truth where applicable.
- **Asynchronous**: Event names, schema versioning, idempotency keys, at-least-once handling, dead-letter policy.
- **Data**: Migration ownership per service; avoid cross-service joins; use sagas/outbox when distributed transactions are required.

## Cross-cutting checklist

- [ ] **Security**: AuthN/AuthZ at edge; secrets not in code; least privilege for runtime identities; validate all external input.
- [ ] **Observability**: Correlation IDs across calls; structured logs; RED/USE or equivalent metrics for services; traces for latency debugging.
- [ ] **Reliability**: Timeouts, retries with backoff (idempotent ops only), circuit breakers where appropriate; graceful degradation documented.
- [ ] **Performance**: Caching strategy with TTL and invalidation rules; hot paths identified; N+1 and fan-out risks called out.

## End-to-end flow (design workflow)

1. **Context**: Who are the actors? What are the critical user journeys and SLAs?
2. **Containers**: Major deployable units and their responsibilities (C4 Level 2–style).
3. **Components**: Major pieces inside each container and **owned interfaces**.
4. **Data**: Source of truth per entity; consistency model (strong vs eventual); retention and PII.
5. **Failure modes**: What breaks first under load or dependency outage? What is the safe default?
6. **Evolution**: Deprecation path for APIs; feature flags if releases are frequent.

## Documentation defaults

- **ADRs** (Architecture Decision Records): one short file per significant decision—context, decision, consequences, status.
- **Diagrams**: Keep one **context** and one **container** diagram current; deeper diagrams on demand.
- **README or docs/**: How to run locally, required env vars, and how pieces connect—no duplicate novel in every file.

## Anti-patterns to flag

- “Big ball of mud” with no module boundaries in a growing codebase.
- **Shared database** as integration layer between teams without governance.
- **Chatty synchronous chains** across many services without timeouts and budgets.
- **Leaky abstractions**: infrastructure errors and types bubbling into domain logic without translation.
- **Premature distribution**: multiple deployables before the problem or team structure requires it.

## Progressive disclosure

For stack-specific patterns (e.g. FastAPI layout, Kubernetes manifests), read the repo’s existing structure and extend it rather than imposing a greenfield template.
