# Engram lookups in decode (ds4_model.cu, p2p-145)

Findings
- Decode does NOT look up Engram synchronously inside the layer. forward() (~19330) and verify_multi() (~20606) call
  eg_issue() for every Engram layer as soon as the window's tokens are known (before forward_par/serial): cache lookup
  on the calling thread (counted in engram_ms), the misses are submitted to the EgReader threads asynchronously.
  The layer collects them with eg_finish(*eg_win[l.eg]) (~20056); engram_ms = issue time + the wait left there.
- engram_layer_ids = [1, 14] (EXL3 3.5bpw config). Layer 1 runs after only layer 0, so its rows get ~1 layer
  (~0.3 ms) of overlap; layer 14's rows ~4 ms (already hidden). The ~2.4 ms is mostly layer 1's SSD wait + the
  synchronous cache-lookup/issue work before the window starts.
- Verify windows: tokens y + drafts are all known only when draft() returns, and the window starts right after, so
  no earlier point exists for the drafts' rows. Only y (the token at pos) is known before draft(). Non-spec step():
  the token comes from sampling and the next step starts at once - no gap to hide reads in.
- Row cache (EgRowCache, ds4_engram_io.hpp): clock replacement, every read row enters it in eg_finish (also prompt
  read-ahead). Size DS4_ENGRAM_CACHE_MB (default min(256, ram/64) MB; cap rows = MB/264 B). Hit rate 37% is a
  capacity/locality matter: a larger cache (the main session tests 40000 MB) is the main lever.

Implemented (opt-in MAYA_DS4_EG_DECODE_AHEAD=1)
- eg_decode_ahead()/eg_decode_ahead_end(): at draft()/draft_multi() start, y's rows of all Engram layers go to the
  reader's low queue (one EgAhead per conversation, eg_dpre); they are finished into the row cache before the window's
  own eg_issue (forward, verify_multi), where they appear as hits. Bytes identical (cache holds file bytes); a miss =
  the normal synchronous path. Gain bounded: 1 of T rows of the window, hidden behind draft_ms.
- Counter: stats.eg_dahead, /metrics `eg_decode_ahead_rows`.
