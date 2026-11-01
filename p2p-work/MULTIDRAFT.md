# PR #636 (multi-draft policy) port - branch p2p-145-md (~/project-maya-p2p145md)

## Findings
- #636 = 5 commits on v1.1.36; first four are in v1.1.45 via #587. Only a43df58 is new (policy + Model::draft_pass + --multi-check cache_hold).
- Claim "at N=4 nobody drafts" HOLDS: v1.1.45's N>1 round drafts only if `m.can_draft_multi(all drafters)`; that needs n*T <= kWin(16) and n*(T+1) <= 18. DSpark block T=5 -> max 3 per pass. N=4 -> gate fails -> all `tried=0` -> plain step of 4 rows. (N=2,3 draft.)
- DS4_DRAFT_COST (DraftCost) does NOT fix it: it runs after the gate, only resizing drafts of requests that already drafted (and plans windows by measured step cost). With N=4 the gate has already cleared everyone. No existing env fixes it; port needed.

## Design
- `MAYA_DS4_MULTI_DRAFT_POLICY=1` (default 0 here; PR default is 1). When 0 the legacy code path runs unchanged.
- Policy: engine times rounds by window rows (MultiCost.cost), drafter calls by conversations per call, per-request acc_rate / k_hat EMAs. Plans "j best-acceptance requests draft <= (16-n)/j each" priced with drafter calls (`draft_pass()` per call); best tokens/ms taken if it beats a no-draft round by MARGIN. Left-out requests probe every RETRY rounds, gap doubling to PROBE_CAP. Cold (fewer than 2 window sizes + 1 drafter call timed) = old gate.
- Drafts go through `draft_groups`: groups of <= draft_pass() via draft_multi, leftover alone via draft(). Uses `bind` for the single draft. Each call goes through Model::draft_multi so Engram decode read-ahead (eg_decode_ahead) and DRAFT_DEV drafter split are used as before (draft_multi/can_draft_multi are layout-agnostic; the pass size is bounded by kWin/d_block only). Cost is measured on the real setup, so a split drafter's higher call time is priced in automatically.
- When the policy is on, DraftCost's `sp` planning is disabled in multi rounds (policy replaces it); the cut uses room_all (the request's own room) then clamps to the plan's room. Logprob paths (tok_lp/got_lp, lp_take) untouched.
- Added: Model::draft_pass (ds4_model.cu/.hpp). Skipped: --multi-check changes (cache_hold), docs/SETTINGS.md.

## Knobs
MAYA_DS4_MULTI_DRAFT_POLICY (0), _MARGIN (0.03), _RETRY (8), _PROBE_CAP (128), _LOG (0). Also read as STRATA_*.

## Verify
Run server with `MAYA_DS4_MULTI_DRAFT_POLICY=1 MAYA_DS4_MULTI_DRAFT_LOG=1 --parallel 4`, 4 concurrent requests. Look for stderr lines
`maya ds4 multi draft: 4 answering, R rows (kept/offered rid:acc/n ...): drafter X ms, round Y ms, T tokens (.. tok/s; the plan expected E, without drafts W)`.
Drafts offered > 0 at N=4 (after warm-up of a few rounds) = working; "plan expected" should be near measured tok/s. Per-request averages are printed at request end. Compare aggregate tok/s vs POLICY=0 (~76 at N=4). Greedy output must be identical.
