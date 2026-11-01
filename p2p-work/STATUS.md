# P2P all-reduce (DS4) — stan 29.10.2026 ~17:30

Kontynuacja `~/project-maya/HANDOFF-p2p-allreduce.md`. Kod: worktree `~/project-maya-p2p`, gałąź `p2p-ar` (baza `minchunks-137`, niezacommitowane), build `build/maya-ds4`.

## Etap 1 — udział all-reduce (nsys, CLI, prompt 12K prawdziwego tekstu, chunk/window 4096)
- tp2u prefill: AR (`ar2_kernel`) = **31%** czasu (5,8 s z 18,6 s), sam transfer (~3-6 GB/s przez RAM hosta), na ścieżce krytycznej. Przy P2P teoretycznie do ~940 tok/s z 644.
- tp3 prefill: AR odsłonięty ~8,5% (`allreduce_dma` schowany na `ar_s`) + `ar_rs16_kernel` 9,7% = do ~18%, realnie 5-10%.
- dekod tp2u: transfer ~12% pracy GPU, realny zysk P2P 5-8%; tp3 dekod: AR 20-44% kerneli, ale ~80% to czekanie (nierównowaga GPU0/GPU3), transfer 5-7%.
- Baza CLI v1.1.37: tp2u prefill 645 tok/s, dekod 86 tok/s; tp3 538/93. Tokeny tp2u powtarzalne (128 pierwszych identyczne) -> kryterium poprawności.
- Pipeline: pary pracują szeregowo, xfer_get czeka 5-12 s (osobna dźwignia).

## Etap 2 — ustalenia
- `ds4_gguf_test --ar A B` z `DS4_AR_P2P=1/2` (dodane w tests/ds4_gguf_test.cu): bufory na GPU, poprawne bit w bit.
  GPU0/1: 4,4 -> 10,8 GB/s; GPU4/5: 3,6 -> 17,5-18,5 GB/s.
- Pary z GPU2 lub GPU3 (BAR1 256 MiB, bez ReBAR) WISZĄ w P2P nawet przy 64 KB buforach, w obu kierunkach, także 2 z 4. Kernelowy P2P wymaga ReBAR na obu kartach. -> bramka w `gmap`: grupa z kartą BAR1 < 1 GiB zostaje na hmap.
- Pełny silnik z jakimkolwiek `cudaDeviceEnablePeerAccess` (nawet bez buforów na GPU, tylko 1->0) => `illegal memory access`; coredump: `route_kernel` na GPU4 (Warp Out-of-range Address). Większy `--reserve-mib` nie pomaga.
- Samodzielny test `p2p-work/repro/p2prepro` (20 GB/GPU, 60 GB pinned, peer przed i po alokacjach) — przechodzi. Sterownik OK, problem specyficzny dla silnika.
- W toku: `DS4_AR_P2P_ENABLE_LATE=1` — peer włączany na końcu `load()` (po calibrate_paths/reset/blas_warm_start). Log: `p2p-work/late.log`.

## Zmienne testowe w gałęzi p2p-ar
DS4_AR_P2P=1, DS4_AR_P2P_MASK (1 ar.data, 2 ar.flags, 4 ar.pflags, 8 arb.data, 16 arb.flags, 32 arb.pflags, 64 sums/bflags), DS4_AR_P2P_LOG=1, DS4_AR_P2P_HOMES=0,4, DS4_AR_P2P_ENABLE_ONLY=1, DS4_AR_P2P_ENABLE_LATE=1.

## Uruchamianie
`MAYA_DIR=/home/michal/project-maya-p2p TMO=420 ~/project-maya/p2p-work/run.sh NAZWA 2 DS4_AR2=1 DS4_AR_P2P=1 ...` -> `p2p-work/NAZWA.log`, `.done`. Baza: bez MAYA_DIR (v1.1.37). Po restarcie cache modelu zimny (pierwszy przebieg wolniejszy).
Nie zapisywać dużych plików do /tmp (tmpfs w RAM; coredump CUDA miał 100 GB).

## WYNIK (17:40): DS4_AR_P2P_ENABLE_LATE=1 DZIAŁA
Peer włączony na końcu load() (po calibrate_paths/reset/blas_warm_start), bufory jeszcze hmap: prompt 12K przeszedł (490 tok/s, zimny cache).
=> włączenie peer W TRAKCIE load() psuje coś później w load (kandydaci: calibrate_paths, lane_setup, one_gpu_plan, alloc_chunks_fit).
Następny krok: w gmap tylko zapamiętać grupy/bufory; na końcu load() cudaMalloc buforów AR na GPU domowym, EnablePeerAccess, podmienić wskaźniki G.ar/G.arb (data/flags/pflags), zwolnić stare hmap. Potem: test tokenów vs baza (128 pierwszych), pomiar prefill tp2u.

## WYNIK KOŃCOWY (wieczór): p2p_place działa
gmap tylko notuje bufory (na razie hmap); `p2p_place()` na końcu load(): EnablePeerAccess + cudaMalloc na GPU domowym + podmiana wskaźników G.ar/G.arb. Tryby testowe ENABLE_ONLY/ENABLE_LATE i ćwiartka BAR usunięte.
A/B (ab-A*/ab-B*.log): prefill 659 -> 735 tok/s (+11,5%), dekod ~97 -> ~108. Tokeny OK. Zapis w WYNIKI-noc.md.

## PO FLASHU VBIOS (noc): wszystkie 6 kart BAR1 32 GB, P2P na 3 parach
prefill 647 -> 817 tok/s (+26%), dekod ~+5%. Bufory 1015 MB w VRAM. Szczegóły w WYNIKI-noc.md. Do zrobienia: serwer, tp3, commit/PR (na prośbę).

## SERWER (noc): prefill 12K 1041 -> 1290 tok/s (+24%), 24K 1116 -> 1399 (+25%), dekod N=1 +10%, N=4 +5%. Dalej: tp3 (qtp3.sh startuje po qsrv.done, wyniki t3-*.log).

## tp3 (noc): P2P bez zysku
t3-A*/B*: prefill 12K ~548 vs ~542 tok/s, dekod ~109 vs ~107 (rozrzut). W VRAM 850 MB, 480 MiB zostało w RAM (brak miejsca). Możliwy test: `EXTRA_ARGS="--reserve-mib 3000"` z B. Nie zrobiony.

## Test poprawności logitów (w toku, qdl.sh): dl-A1/A2 (bez P2P), dl-B1/B2 (P2P), tp2u, bez dspark, DS4_DUMP_LAST -> p2p-work/dl-*.logits.*; porównanie cmp.

## PLAN (po przerwie, kolejność)
1. wynik testu logitów; 2. porządki w kodzie (usunąć MASK/HOMES/kind, zwolnić stare hmap, rezerwa VRAM dla buforów P2P, wiersz w docs/SETTINGS.md, opt-in);
3. eksperymenty: DS4_AR2_DECODE=1 + P2P, przekazanie aktywacji między parami (dhand/xfer) na P2P, tp3 z --reserve-mib;
4. commit: ustawić autora (e-mail konta GitHub 486DX4-100), przenieść na `main` (osobna gałąź tylko z P2P), `check_hip_compat.py` OK; opis PR z liczbami (prefill tp2u +24-26% w serwerze i CLI, dekod +5-10%). Nic nie publikować bez prośby.
Zmiany niezacommitowane w `~/project-maya-p2p` (gałąź p2p-ar); kopie diffów: p2p-work/p2p-ar-wip2.patch (starsza wersja!) — zrobić świeży `git diff` przed pracą.

## WYNIK testu logitów (tp2u, 12K, bez dspark, ostatni wiersz promptu; dl-*.logits.0)
Bit w bit NIE wychodzi nawet bez P2P: A1 vs A2 maks |Δ| 2,8 (rms 0,38), a A vs B maks 2,2-3,8 (rms 0,36-0,52) — ten sam rząd wielkości. Argmax (token 85) taki sam we wszystkich 4 przebiegach; top10 wspólnych 9-10/10.
=> P2P nie wnosi mierzalnej różnicy ponad niedeterminizm bazy (ok. 2 rzędy większy niż zaokrąglenia FP). Źródło niedeterminizmu bazy NIEZBADANE (kandydat: CPU-lane/rozkład ekspertów GPU vs CPU, wg CONTRIBUTING „which expert runs on the CPU or GPU changes the last bits"; albo kolejność sumowania w ścieżce strumieniowania ekspertów). Prefill w tych przebiegach: A 644/644, B 802/801 tok/s (+25%).
Do PR: napisać uczciwie, że wyjście i bez P2P nie jest powtarzalne bitowo; sam transport sum nie zmienia (sumy w ar2_kernel identyczne, test --ar bit w bit).
