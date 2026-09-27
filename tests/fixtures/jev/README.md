# Jev fixtures — **captured from the live API**

These are **verbatim responses**, re-indented only. Nothing was added or removed, so a fixture
cannot describe an envelope the API does not actually produce.

Captured 2026-09-27 from `POST https://jevtypesafeai.com/api/v1/decide`, with
`model: jev-1.13.0`.

| File | What it pins |
|---|---|
| `response_choice.json` | a `choice` answer: `choice`, `confidence`, and `probabilities` **keyed by label** |
| `response_score.json` | a `score` answer: `score`, `confidence`, `legend`, and `probabilities` **keyed by level string** |
| `response_noul.json` | a `noul` answer: `noul` and **nothing else** — no `confidence` key at all |

That last file is the whole argument for the routing rule in one line: `{"type": "noul", "noul":
0.67}`. There is no confidence field to gate on, which is why `route()` refuses to route one.

Three things these captures settled that guessing had got wrong:

1. **`choice` keys `probabilities` by choice label; `score` keys them by level.** The two kinds
   genuinely differ, and the first draft of `docs/jev.md` claimed index strings for both.
2. **The API reports the cost.** `usage.cost_usd` and `usage.credits_remaining_usd` are present on
   every response. The doc had claimed the first-party API reported no cost field and that the
   multiplication was ours; it is now the fallback, not the source.
3. **The error envelope differs between the vendor's hosts.** This one answers
   `{"error": "Invalid or revoked API key."}`; `api.typesafe.ai` answers
   `{"detail": {"error_type": ..., "message": ...}}`. Both are parsed.

`error_401.json`, `error_403.json` and `models_200.json` are still hand-written, and are marked as
such: the first two were observed by probing, and `models_200.json` documents an endpoint that
**this host does not have** (`/api/v1/models` answers `unknown_endpoint`), so it is kept only for
the client test that exercises the older host's shape.
