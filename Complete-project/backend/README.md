# AgentInTheShell API

Run from the repository root:

```powershell
.\.venv\Scripts\activate
uvicorn backend.main:app --reload
```

OpenAPI documentation is available at `http://127.0.0.1:8000/docs`.

Endpoints:

- `POST /analyze_text`
- `POST /analyze_thread`
- `POST /analyze_account`
- `POST /analyze_network`
- `GET /get_threat_dashboard`
- `GET /health`

SQLite is used by default at `data/aegis_api.db`. Set `DATABASE_URL` to a
SQLAlchemy PostgreSQL URL for production. Configure allowed frontend origins
with the comma-separated `CORS_ORIGINS` environment variable.

The handle endpoint is a deterministic demonstration and always returns
`data_source: simulated`; it does not fetch a live social-media account.

The API reports artifact provenance and trust rather than hiding fallbacks. The
checked-in text artifact is a lightweight smoke model, and the graph artifact
is below the configured cross-campaign transfer floor. Run notebooks 02 and 03
with `AEGIS_SMOKE_TEST=0` and validate held-out performance before production.

References:

- `../docs/API_REFERENCE.md` — complete request/response contracts and examples.
- `../docs/PROJECT_DOCUMENTATION.md` — architecture, model methodology,
  scoring, limitations, and production gaps.
- `../docs/GENERALIZATION_RUNBOOK.md` — full-data GPU training and acceptance.
