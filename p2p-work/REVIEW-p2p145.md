# Przegląd łatki DS4_AR_P2P na v1.1.45

Repozytorium: `~/project-maya-p2p145` (gałąź `p2p-145` od tagu `v1.1.45`, zmiany niezacommitowane;
`git diff` = 4 pliki, +151/−27). Porównane z `p2p-work/p2p-v141.patch` (ta sama łatka na v1.1.41)
i `p2p-v145.patch`.
Zakres: **tylko czytanie kodu** — nic nie zbudowano, nic nie uruchomiono na GPU (`build/` nietknięty),
żadnych commitów.
Pass 1: moje czytanie; pass 2: 3 subagenty (martwe wskaźniki hosta / widoczność zapisów + memcpy /
cykl życia + diff v141→v145). Wszystkie ich ustalenia sprawdziłem w kodzie; poniżej wersja scalona.

## Podsumowanie

Scalenie z v1.1.45 jest **strukturalnie wierne**: `gmap()` wstawiony dokładnie tam, gdzie .45
zakłada własne bufory hosta dla `G.arb` (`data`/`sums` tylko przy `G.hred == nullptr`,
`flags`/`bflags` zawsze przy `chunk > 0`), więc nie ma podwójnej alokacji, buforów zostawionych
w RAM-ie mimo P2P ani wycieku. Protokół danych jest OK (`__threadfence_system()` przed flagą,
`ld_fresh` = `__ldcv` przy odczycie danych peera). W porcie v141→v145 nic nie zginęło poza tym,
co wymusiła własna restrukturyzacja `G.arb` w .45.

**Trzy rzeczy blokują commit:** interakcja z `DS4_AR_HOST` (K1), odczyt flagi przez `volatile`
zamiast `ld_fresh` (W1), kolejność względem `blas_warm_start` (W2).

## KRYTYCZNE

### K1. `HostReducer` trzyma wskaźniki AR, które P2P przenosi do VRAM (i zwalnia)

`src/ds4/ds4_model.cu:4102-4117` + `src/ds4/ds4_kernels.cu:4696` (`ArGroup a`), `4726`/`4730`
(poll), `4745` (zapis), `4753` (`p_->a = a`).

`DS4_AR_HOST` (domyślnie 5) tworzy `HostReducer` dla grup o `nm >= host_from && nm <= 16`,
ale **po** `gmap(G.arb.flags)`/`gmap(G.arb.bflags)` (4102-4103). Wątek hosta czyta i **pisze**
`a.bflags` przez `*(volatile unsigned*)` — czyli wymaga pamięci czytelnej z CPU. Przy
`DS4_AR_HOST = 3..4` i grupie 3/4 kart bufory trafiają do VRAM home, a kopia `ArGroup`
w reducerze (`4753`, zrobiona przed `p2p_place`) wskazuje **zwolniony** przez `cudaFreeHost`
bufor → use-after-free / SIGSEGV / cisza w losowym miejscu pierwszego chunku promptu.
Osobno: `DS4_AR_HOST=2` przy `nm==2` daje `hred` z `bflags == nullptr` (guard `nm > 2`
w 4103) — błąd niezależny od P2P, obecny w samym .45.

Poprawka: policzyć `host_from`/`DS4_AR_DMA` **przed** linią 4102 (przenieść blok 4105-4117 wyżej)
i dla grupy sumowanej na hoście wołać `hmap`, a nie `gmap`; przy `host_from < 3` odrzucić
konfigurację w `load()` z jasnym błędem. Docelowo `HostReducer` powinien trzymać wskaźnik do
żywego `groups[].arb`, nie kopię.

## WYSOKIE

### W1. `wait_at_least()` czyta flagę przez `volatile`, nie przez `ld_fresh()`

`src/ds4/ds4_kernels.cu:3062-3068` (użycia: `3072`, `3081` w `wait_flags_kernel`, `3124`,
`3145`, `3399` w `ar2_kernel`).

To **jedyne miejsce na ścieżce peera**, które opiera się na niezdefiniowanym zachowaniu cache:
`volatile` omija L1, ale nie gwarantuje pobrania świeżej linii z L2 czytającej karty — a flaga
leży teraz w VRAM home i jest czytana po PCIe. Wszędzie indziej dane peera czytane są przez
`ld_fresh` (`__ldcv`), właśnie po to. Dla pamięci host-mapped (bez P2P) było niewrażliwe, bo
system memory i tak nie jest cache'owana. To także najprostsze wytłumaczenie historycznego
„ar2_kernel kręcił się w nieskończoność” (HANDOFF).

Poprawka (1 linia, koszt zero): `while ((int) (ld_fresh(flag) - seq) < 0)`. `__ldcv` na pamięci
host-mapped jest już w użyciu w tych samych kernelach (wiersze peera), więc oba tryby bezpieczne.

### W2. `p2p_place()` w wyścigu z wątkiem `blas_warm_start()`

`src/ds4/ds4_model.cu:4679-4680`, `2448-2492`, `2473`, `2484`.

`blas_warm_start()` startuje wątek robiący `cublas*` na tych samych kartach tuż przed
`p2p_place()`, który: (a) robi `cudaMalloc` ~1 GB na karcie home **konkurując z pulą cuBLAS** —
alokacja może niekiedy polec i bufor cicho zostanie w RAM („left host-mapped (no room)”), czyli
P2P „raz wchodzi, raz nie” zależnie od harmonogramu i A/B przestaje być miarodajne;
(b) `cudaDeviceSynchronize()` (2473) czeka na **całą** kartę home, czyli na kernela warm-upu →
`load()` wraca po rozgrzewce, gubiąc sens jej nakładania się z pierwszym żądaniem;
(c) `cudaFreeHost(old)` (2484) zwalnia pamięć zmapowaną w kontekstach **wszystkich** członków,
a zsynchronizowana jest tylko karta home — dziś bezpieczne tylko dlatego, że warm-up nie dotyka
buforów AR.

Poprawka: `reset(); p2p_place(); blas_warm_start();` (`p2p_place` niczego z warm-upu nie
potrzebuje), a przed `cudaFreeHost` jawnie `cudaDeviceSynchronize()` na każdej karcie grupy.

### W3. Podmiana wskaźników i cykl życia `p2p_todo` są kruche

`src/ds4/ds4_model.cu:2436`, `2442`, `2476`, `2477-2484`, `2491`.

Trzy problemy w jednym miejscu:
- `P2pBuf::where` to `(void**)&p` w element `std::vector<Group> groups` — działa tylko dlatego,
  że po `gmap` (4083) nic już nie robi `push_back`/`resize` na `groups` (brak `reserve()`, brak
  asercji);
- `p2p_todo` **nie jest czyszczone na żadnej ścieżce `return false` w `load()`**, więc po
  nieudanym `load()` zostają wpisy wskazujące na pola `groups[]` i `old` wskazujący już zwolnione
  bufory hosta (teardown 2300-2301). Dziś `m.load()` w `src/ds4/ds4_main.cpp:3619` jest wołane
  raz i kończy proces, więc to bomba z opóźnikiem; retry/reload w procesie = UB;
- podmiana to globalny sweep po równości `*f == old` w każdej grupie. Dziś poprawny (żadne dwa
  pola nie dzielą jednej alokacji: przy `chunk > 0` `arb.flags`/`bflags` dostają własne bufory,
  a przy `chunk == 0` dzielą jeden wskaźnik z `ar`, co jedna podmiana załatwia), ale gdyby
  kiedyś dwa pola współdzieliły bufor, drugi wpis odczytałby jako `old` wskaźnik device
  i zrobił `cudaFreeHost` na wskaźniku VRAM (błąd sticky, psuje kolejne wywołania CUDA).

Poprawka: zapisać `old` **w** `P2pBuf` przy `gmap` (zamiast czytać `*b.where`), przy podmianie
sprawdzić `*f == old`, `cudaFreeHost` wołać tylko gdy `old` faktycznie jest w `host_allocs`
(i czyścić `cudaGetLastError()`), `p2p_todo.clear()` na początku `load()` + `groups.reserve()`.

## ŚREDNIE

### Ś1. Bufory P2P nie są ujęte w budżecie VRAM

`src/ds4/ds4_model.cu:2464` (cudaMalloc), `2474-2475` (księgowanie w `d.allocs`).

Alokacja idzie **po** wszystkich decyzjach o miejscu (free0/`--reserve-mib`, eksperci, borrow/tail,
`Chunk::room`) i nie zwiększa `d.scratch`. `G.arb.data` to `nm*2*chunk*dim*4` B — przy chunk 4096
i dim 5120 ~336 MB dla 2 członków, ~672 MB dla 4 — lądujące na jednej karcie. Gdy wejdzie,
zjada rezerwę przewidzianą na późniejsze potrzeby (możliwy OOM w czasie działania, nie w load);
gdy nie wejdzie, cicho wraca do RAM — co najprawdopodobniej tłumaczy „tp3 bez zysku”
(STATUS: 480 MiB zostało host-mapped). `describe()`/Monitor też zaniża VRAM (brak w `d.scratch`).

Poprawka: zaplanować bufory P2P przed decyzjami o pokoju (albo odjąć ich rozmiar od `free0`
w load) i dopisać `d.scratch += bytes` po przeniesieniu.

### Ś2. Rozstrzygnięcia per bufor, nie per grupa; peer access nigdy nie wyłączany

`src/ds4/ds4_model.cu:2455-2461`, `2469-2472`.

`cudaDeviceEnablePeerAccess` jest wołane dla **każdego** bufora grupy (do 5-6 wpisów), a każdy
bufor decyduje o przeniesieniu osobno. Dziś niespójność „część grupy w VRAM, część w RAM” nie
występuje, bo błąd enable jest powtarzalny, ale to własność przypadku, nie konstrukcji. Peer
access zostaje włączony także wtedy, gdy żaden bufor nie został przeniesiony, i nie jest
wyłączany w teardownie.

Poprawka: raz na grupę — najpierw sprawdzić/włączyć peer dla wszystkich członków, potem
zaalokować wszystkie jej bufory; przy dowolnym niepowodzeniu zostawić **całą** grupę na hmap.

### Ś3. Próg `BAR1 >= 1 GiB` to proxy, nie spełnienie warunku

`src/ds4/ds4_model.cu:2418`.

Próg wyklucza okno 256 MiB, ale nie gwarantuje, że okno ReBAR pokrywa **sumę bajtów** grupy
(336-672 MB przy chunk 4096). ReBAR ograniczony w BIOSie do 1-2 GiB przejdzie bramkę, a loads
and stores przez okno nadal mogą zawiesić jądro — czyli dokładnie awaria opisana w komentarzu
(2413-2414).

Poprawka: porównywać `resource1` z realnym zapotrzebowaniem grupy (rozmiar największego bufora
albo suma), a nie ze stałym 1 GiB.

### Ś4. Log nie mówi prawdy w czterech przypadkach

`src/ds4/ds4_model.cu:2423-2428`, `2488-2490`.

- linia „sukcesu” drukuje się, gdy `p2p_todo` nie jest puste — **także gdy przeniesiono 0 B**:
  `0.0 MB ... peers reaching them directly, N MiB left host-mapped`;
- gdy żadna grupa się nie kwalifikuje, `p2p_place` wraca wcześnie i **nie ma żadnej linii** —
  a HOWTO/SETTINGS każą weryfikować log;
- przy `DS4_AR_P2P_LOG` wypis idzie **raz na bufor** (komentarz „once a group: its first buffer”
  jest nieprawdziwy): do ~18 linii przy 3 grupach tp2;
- HOWTO:15 mówi `DS4_AR_P2P_LOG=1`, a kod włącza log przy **dowolnej** obecności zmiennej
  (`getenv != nullptr`), więc `DS4_AR_P2P_LOG=0` też loguje.

Poprawka: linię sukcesu tylko przy `moved > 0`; osobne, zawsze przy `DS4_AR_P2P=1`, podsumowanie
„nothing moved” z powodem; deduplikacja LOG per grupa; `atoi`/`tri_of` zamiast `getenv != nullptr`.

### Ś5. `cudaMemcpyDefault` zmienione bezwarunkowo (11 miejsc) i bez sprawdzania błędów

`src/ds4/ds4_kernels.cu:4581,4582,4597,4609,4610,4617,4627,4629,4633,4634,4842`.

Dotyczy także trybu bez P2P (i HIP-a), więc formalnie łamie założenie 1 „bez `DS4_AR_P2P`
zachowanie identyczne z upstream”. Funkcjonalnie jest równoważne (dla `cudaHostAllocMapped`
`cudaMemcpyDefault` rozwiązuje się na DtoH/HtoD jak wcześniej; test bit-w-bit w STATUS to
potwierdza), ale: (a) na hot-path AR dochodzi rozwiązywanie typu wskaźnika per kopia, (b) **żadna
z tych kopii nie ma sprawdzenia wyniku** — przy błędzie (np. peer access wycofany) model liczy
dalej z zerami. Na ROCm `hipMemcpyDefault` na pinned-mapped bywa klasyfikowany inaczej — bez
pomiaru.

Poprawka: wybierać rodzaj kopii po trybie (`p2p ? cudaMemcpyDefault : cudaMemcpyDeviceToHost`),
albo owinąć kopie w `CK(...)`/sprawdzić `cudaGetLastError()` raz na wywołanie.

## NISKIE

### N1. `DS4_AR_P2P` czytane przez `std::getenv`, nie `ds4_env()`

`src/ds4/ds4_model.cu:2410`, `2423`.

Sąsiednie knoby (.45) idą przez `ds4_env()` (`13001`), które czyta `STRATA_*` i `DS4_*`, a
`tools/maya_names.py` (`alias_env`) zamienia `MAYA_X` na `STRATA_X`. Efekt: `MAYA_DS4_AR_P2P`
jest **po cichu ignorowane**, choć SETTINGS w sąsiednich wierszach używa nazw `MAYA_DS4_*`.
Poprawka: `ds4_env("DS4_AR_P2P")` i `ds4_env("DS4_AR_P2P_LOG")`.

### N2. Test `--ar`: `DS4_AR_P2P=2` znaczy co innego niż w silniku

`tests/ds4_gguf_test.cu:2873-2898`; silnik: `src/ds4/ds4_model.cu:2410`.

Silnik traktuje każdą wartość niezerową jak 1 (home zawsze pierwszy członek), test interpretuje
`2` jako „bufory na drugiej karcie”. Ten sam env daje różne zachowanie zależnie od binarki.
Dodatkowo test nie sprawdza BAR1 (na parze z BAR1 256 MiB zawiesi się — lepiej, by odmówił)
i zostawia „current device” na karcie home (2888), co przy `p2p == 2` zmienia kartę względem
ścieżki bez P2P. Poprawka: osobny knob na „home” testu + bramka BAR1 + jawne `cudaSetDevice`
przed pomiarami.

### N3. Dokumentacja

- `docs/SETTINGS.md:1165-1168`: wiersz `DS4_AR_P2P` wstawiony **w środek tabeli**, po nim akapit
  i pusta linia, więc tabela się urywa, a wiersz `MAYA_DS4_HIP_PEER_AR` (1168) zostaje
  osierocony poza tabelą — zepsuty render. Przenieść akapit za całą tabelę.
- `HOWTO-P2P.md:13` i założenie #4 mówią `~1015 MB`, kod drukuje `%.1f` → `1015.0 MB`.
  Albo `"~%.0f MB"`, albo poprawić HOWTO.
- `HOWTO-P2P.md:15` — patrz Ś4 (LOG działa przy każdej wartości zmiennej).

### N4. Drobne

- `src/ds4/ds4_model.cu:2457`, `2463`: brak sprawdzenia zwrotu `cudaSetDevice` w `p2p_place()` —
  przy złym `id` cała reszta wykona się na innej karcie, a błąd zostanie zamieciony przez
  `cudaGetLastError()`.
- `src/ds4/ds4_model.cu:2401-2406`, `2418`: `bar1_bytes()` zwraca 0 poza Linuksem → P2P nigdy
  nie działa (ciche, nieudokumentowane w SETTINGS).
- `ar2_kernel`/`ar_sum16_zc_kernel` zakładają `stride % 4 == 0` (wyrównanie 16 B dla
  `float4`/`uint2`); brak asercji przy ładowaniu. Nie regresja (to samo było z `cudaHostAlloc`),
  ale warto, by awaria była głośna.
- `src/ds4/ds4_model.cu:2474-2475` + `2294-2301`: przy `helper_keep = true` `host_allocs` nie są
  zwalniane, a bufory P2P (w `d.allocs`) są zawsze — niespójne intencją, ale helper ich nie czyta,
  więc tylko komentarz.

## Założenia projektu — weryfikacja wprost

1. **Wyłączone domyślnie, bez `DS4_AR_P2P` jak upstream** — TAK dla P2P: `gmap` → `hmap`,
   peer access nie jest włączany, `p2p_todo` puste. Zastrzeżenie: zmiana `cudaMemcpy*` →
   `cudaMemcpyDefault` jest **bezwarunkowa** (Ś5) — funkcjonalnie równoważna, ale to nie jest
   „identyczne”.
2. **Tylko grupy 2-4 kart, każda BAR1 ≥ 1 GiB, inaczej hmap** — TAK: `G.mem.size() > 4 → hmap`
   (2411), BAR1 z `resource1` ≥ 1 GiB **od każdego** członka + `cudaDeviceCanAccessPeer(m, home)`
   w dobrą stronę (2418-2420). Grupy 1-kartowe nie wołają `gmap` wcale (`nm > 1` w 4087) → tp1
   czysty; tp3 (3 karty) wchodzi świadomie; grupy 5+ nigdy. Zastrzeżenia: próg 1 GiB to proxy
   (Ś3), a fallback „no room” potrafi zostawić część buforów w RAM (Ś1, Ś2).
3. **Peer + przeniesienie dopiero na końcu `load()`** — TAK: `p2p_place()` w 4680, po
   `reset()`/`blas_warm_start()`. Ale ta kolejność jest źródłem wyścigu (W2).
4. **Log** — NIE literalnie: `1015.0 MB` zamiast `~1015 MB`, LOG przy dowolnej wartości zmiennej,
   wypis per bufor, a przy zerowym przeniesieniu „sukces” z `0.0 MB` (Ś4, N3).

## Co jest w porządku (nie wymaga zmian)

- **Brak podwójnej alokacji / wycieku `G.arb`**: `arb.data = arb.sums = nullptr` (4101) przed
  `gmap`; `arb.flags`/`bflags` dostają własne bufory dokładnie jak w .45 (4102-4104);
  `arb.pflags` osobno (4126), `ar.pflags` osobno (4135). Zero aliasów i wycieków.
- **Zwolnienie**: bufor P2P → `d.allocs` (2475) i `cudaFree` z właściwym `cudaSetDevice`
  w teardownie (2294-2296); stary host-bufor usunięty z `host_allocs` (2482-2483) i zwolniony
  raz (2484). Brak double-free.
- **Widoczność danych peera**: producent `__threadfence_system()` przed zapisem flagi
  (`set_flag_kernel:3095`, `ar_rs_kernel:3121/3142`, `ar2_kernel:3394/3396`), konsument czyta
  przez `ld_fresh`/`ld_fresh_v` (`__ldcv`) — dla VRAM-u peera to warunek co najmniej tak silny
  jak dla pamięci host-mapped. Jedyny wyjątek: sama flaga (W1).
- **Dane przed flagą w ścieżce DMA**: `cudaMemcpyAsync` i `set_flag_kernel` na tym samym
  strumieniu (4581-4583, 4597-4598) → kolejność wymuszona strumieniem.
- **Rozmiary/wyrównanie**: `cudaMalloc` daje ≥256 B (lepiej niż `cudaHostAlloc`), `+64`
  zachowane, typy 4-bajtowe → `n*sizeof(T)+64` spójne z `hmap`.
- **Arytmetyka**: `a.stride` jest `size_t`, więc `a.slots * a.stride / 2` nie przelewa `int`.
- **5+ kart / tp1**: patrz założenie 2.
- **`cudaMemcpyDefault` bez P2P**: rozwiązuje się na ten sam kierunek co wcześniej.

## Dalsze przyspieszenie (pomysły; wszystko do zmierzenia na GPU, nie w tej sesji)

Żadna z poprawek wyżej nie jest optymalizacją — dwie mogą jednak **pośrednio** odblokować zysk:
W2 (P2P czasem nie wchodzi → A/B niemiarodajne) i Ś1 (tp3: 480 MiB w RAM = część AR starą
ścieżką, najprostsze wytłumaczenie „tp3 bez zysku”).

1. **tp3: `--reserve-mib 3000`** i powtórzyć `t3-*` A/B. Darmowe; jeśli bufory nie wchodzą całe,
   wynik „bez zysku” był artefaktem.
2. **Najpierw nsys: ile AR zostało.** Jeśli udział spadł z ~31% do <10%, prefill jest już
   strumieniem ekspertów z RAM przez PCIe, nie all-reduce — wtedy punkty 3-5 ruszą tylko dekod.
3. **Model „pull” zamiast „push do karty domowej”** (największy potencjał). Dziś każdy członek
   *zapisuje* swoją połówkę do VRAM home (100% ruchu) i *czyta* połówkę drugiego (100%) ≈ 2×
   ruchu PCIe. Wariant pull: każdy pisze partial tylko do **swojej** karty, publikuje flagę,
   reszta czyta przez peer — zostaje sam odczyt, ~2× mniej ruchu. Wzorzec jest w repo: HIP-owe
   `PeerAr` (`src/ds4/ds4_kernels.cuh:501-504`, dwa sloty, timeout, flaga błędu). Bonus: bufory
   przestają obciążać jedną kartę (teraz home GPU niesie ~338 MB z 3 grup = tyle mniej cache
   ekspertów na tej karcie).
4. **Mniej fence'ów**: `ar2_kernel` robi `__threadfence_system()` na każdy piece
   (`ds4_kernels.cu:3394`) przy `kAr2Piece = 8192` float4 (128 KB) — ~10 fence'ów na blok na
   all-reduce. A/B na `kAr2Piece` 16K/32K — tanie, bo to jedna stała (`ds4_kernels.cuh:489`).
5. **tp3: `DS4_AR_ZC=0`** — ścieżka przez copy engine (`ar_sum16_kernel`) zamiast `ld_fresh`
   w kernelu (`allreduce_dma:4624-4635`). Stary pomiar 20,5 vs 11,5 ms dotyczył pamięci
   host-mapped; przy VRAM-ie peera kolejność może się odwrócić.
6. **`MAYA_DS4_AR2_DECODE=1` + P2P** — jest w SETTINGS, nigdy nietestowane z P2P; dla dekodu
   jeden launch mniej.
7. **Dekod: mniej flag.** Przy małym `n` dominuje round-trip 1-2 µs po PCIe, a `ar2_kernel` ma
   osobną flagę na blok (`kAr2Blocks = 64`); dla `n` w kilku blokach rozważyć jedną flagę.
8. **Nie warto** ruszać `cudaMemset` 1015 MB ani `cudaDeviceSynchronize` w `p2p_place` — to koszt
   `load()` (kilkadziesiąt ms), nie przepustowości.

Kolejność opłacalności: 1 → 2 → 3 (największy potencjał, najwięcej pracy) → 4/5/6 (tanie A/B).

## Jak sprawdzano

`git diff` w `~/project-maya-p2p145`; `diff -u p2p-v141.patch p2p-v145.patch`; czytanie
`src/ds4/ds4_model.cu` (gmap/p2p_place 2383-2503, pętla grup 4083-4138, koniec `load()` 4655-4682,
`reset()` 8880, `alloc_chunks_fit` 4891-4946, teardown 2294-2311), `src/ds4/ds4_kernels.cu`
(flagi 3062-3097, `ar_rs*` 3104-3150, `ar_sum16*` 3217-3283, `ar2_kernel` 3376-3411,
`allreduce_dma` 4542-4636, `allreduce_host` 4830-4875, `HostReducer` 4695-4800),
`src/ds4/ds4_kernels.cuh` (ArGroup 473-486), `tests/ds4_gguf_test.cu` (2869-2994),
`docs/SETTINGS.md` (1160-1175), `p2p-work/HOWTO-P2P.md`, `STATUS.md`, `HANDOFF-p2p-allreduce.md`;
`grep` po `host_allocs`, `helper_keep`, `ArGroup`, `cudaHostGetDevicePointer`, `ds4_env`.
Nic nie zbudowano, nic nie uruchomiono na GPU.
