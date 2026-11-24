# @foshol-lens/contracts

Single source of truth for the FosholLens API surface, shared by `apps/backend` and
`apps/frontend`.

- **Schemas** — zod schemas for every request/response payload (`advisory`, `weather`, `capture`).
- **Types** — inferred from the schemas (`z.infer`), never hand-written twice.
- **Routes** — canonical API route constants; no hardcoded URLs in apps.

## Usage

```ts
import { AdvisorySchema, API_ROUTES, apiUrl, type Advisory } from '@foshol-lens/contracts';
```

## Rules

1. Any change to the API happens here first. Apps depend on this package, never the other way
   around.
2. Types are derived from schemas. Do not declare a type the schema can produce.
3. Imports use explicit `.js` extensions (NodeNext ESM output consumed at runtime by the backend).
4. This package must stay dependency-light: `zod` only. No express, no react, no I/O.
