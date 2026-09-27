# Jev fixtures — **constructed, not captured**

These files were **written by hand from the vendor's published schema**, not recorded from a live
successful response. Be precise about that, because the distinction matters:

- The **error** fixtures (`error_401.json`, `error_403.json`) are the real bodies, observed by
  probing `POST /v1/systemone` from this machine with no key (403) and with a key the server
  rejects (401).
- `models_200.json` and the three `response_*.json` files are **constructed**. The configured
  `TYPESAFE_API_KEY` was rejected with HTTP 401 for the whole of the session in which this client
  was written, so no successful response could be captured.

What they are derived from is stronger than a guess, though. The wire shapes come from the
official SDK's generated models (`typesafe_sdk/_schemas/models.py` in `typesafe-sdk==0.7.2`,
which mirrors the vendor's OpenAPI schema) and the HTTP surface was verified by live probe:

| Fact | Source |
|---|---|
| Response envelope is `{"model", "answers", "usage"}` | generated schema |
| `choice` answers carry `choice`, `confidence`, `probabilities` | generated schema |
| `score` answers carry `score`, `confidence`, `legend`, `probabilities` | generated schema |
| `noul` answers carry only `noul` — **no `confidence`** | generated schema |
| `usage` is `{"input_tokens", "output_tokens"}` | generated schema |
| A `choice` **question** takes `criteria` (a label→description mapping), not `options` | generated schema |
| `score` **question** takes `criteria` as an ordered sequence | generated schema |
| Errors arrive as `{"detail": {"error_type", "message"}}` | live probe |
| `GET /v1/models` answers 403 with no key, 401 with a rejected key | live probe |

**Two of those contradict what `docs/jev.md` originally claimed**, and the doc was corrected:

1. `choice` keys its `probabilities` by **choice label**, not by index string. It is `score` that
   keys them by level (`"0"`, `"1"`, ...). The two kinds genuinely differ.
2. `choice` questions are written with `criteria`, a mapping. `options` is not a field.

When a valid key exists, capture real responses here and delete this caveat — and at the same
time re-base the token budget in `tests/test_jev.py` on the `usage.input_tokens` the real API
reports, rather than on the character proxy it uses today.
