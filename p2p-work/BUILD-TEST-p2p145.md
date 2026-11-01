# Jak zbudować i przetestować poprawki P2P (v1.1.45)

Drzewo poprawek: **`~/project-maya-p2p145-fix`** — git worktree, gałąź **`p2p-145-review`**
(utworzona z tagu `v1.1.45`), z nałożoną łatką `p2p-v145.patch` i poprawkami z przeglądu.
Oryginał `~/project-maya-p2p145` (gałąź `p2p-145`) **nie jest tknięty** — możesz go dalej używać.
Nic nie jest zacommitowane (zmiany leżą w worktree; `git status` pokazuje je jako zmodyfikowane).

Co zmieniono (szczegóły: `REVIEW-p2p145.md`, sekcja „pass 3”):

| # | plik:linia | zmiana |
|---|---|---|
| K1 | `ds4_model.cu` pętla grup | `DS4_AR_HOST` liczone **przed** alokacją flag; grupa sumowana na hoście dostaje `hmap` (hosta wątek czyta te flagi), `DS4_AR_HOST=1..2` odrzucane błędem, `hred` tworzony **po** flagach (żeby trzymał właściwe wskaźniki) |
| W1 | `ds4_kernels.cu:3062` | `wait_at_least` czyta flagę przez `ld_fresh` (`__ldcv`), nie `volatile` |
| W2 | `ds4_model.cu` koniec `load()` | `p2p_place()` **przed** `blas_warm_start()`; `p2p_place` synchronizuje wszystkie karty przed `cudaFreeHost` |
| W3 | `ds4_model.cu` `gmap`/`p2p_place`/`load` | `P2pBuf` trzyma `void* old` (nie `void**` w wektorze), wpis musi być w `host_allocs` (inaczej pomijany), `p2p_todo` czyszczone na wejściu `load()` |
| Ś2 | `p2p_place` | peer access włączany **raz na grupę**, `cudaSetDevice` sprawdzany |
| Ś3 | `gmap` | BAR1 musi pokryć **rozmiar bufora** (nie stałe 1 GiB) |
| Ś4 | `gmap`/`p2p_place` | log: `~1015 MB` (jak w HOWTO), osobna linia „nothing moved”, LOG raz na grupę i honoruje `=0` |
| N1 | `gmap` | `ds4_env()` zamiast `std::getenv` → działa też `MAYA_DS4_AR_P2P` |
| N3 | `docs/SETTINGS.md` | akapit przeniesiony za tabelę (wiersz `MAYA_DS4_HIP_PEER_AR` był osierocony) |

## 1. Zbudować (raz)

```bash
cd ~/project-maya-p2p145-fix
export PATH=/home/michal/project-maya/.venv/bin:/usr/local/cuda-12.9/bin:$PATH
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_CUDA_ARCHITECTURES=86 -DSTRATA_ENABLE_CUDA=ON \
      -DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++-12 \
      -DFETCHCONTENT_SOURCE_DIR_STRATA_LLAMACPP=/home/michal/project-maya-p2p145/build/_deps/strata_llamacpp-src
ninja -C build maya-ds4 ds4_gguf_test
```

`FETCHCONTENT_SOURCE_DIR_...` bierze ggml już pobrany do drzewa `p2p145` — bez tego nowy
katalog `build/` ściągałby go ponownie z sieci.

Build trwa ~10-20 min (dwa duże `.cu`). Same dwa pliki zmienione w poprawkach
(`ds4_model.cu`, `ds4_kernels.cu`) kompilowałem już `nvcc` z flagami z `build.ninja`
drzewa oryginalnego — czysto, tylko istniejące ostrzeżenie `NW` w `ds4_kernels.cu:3978`
(nie moje). To była kompilacja, nie uruchomienie na GPU.

## 2. Zamknąć produkcyjną Mayę

Skrypt sam tego nie zrobi i **przerywa**, jeśli coś już siedzi na kartach (wypisze PID-y).
Zamknij po PID: `kill -TERM <PID>` (nie `pkill -f`), poczekaj, aż `nvidia-smi` pokaże pusto.

## 3. Test (A/B + poprawność)

```bash
cd /home/michal/project-maya
p2p-work/ab145fix.sh 2                 # 2 rundy: A1 B1 B2 A2 (na przemian)
```

Co robi: dla każdej rundy startuje serwer z `p2p-work/cfg-fix-A.json` (bez `DS4_AR_P2P`) i
`cfg-fix-B.json` (`DS4_AR_P2P=1`) — **ta sama binarka**, jedyna różnica to zmienna; robi
rozgrzewkę 2K×2, prefill 12K×4 i 24K×3 (prawdziwy tekst, `pre_rt.py`), dekod N=1 i N=4,
notuje linię `DS4_AR_P2P` z logu silnika, licznik AER przed/po, i zatrzymuje **tylko swój** PID.
Przed A/B puszcza `ds4_gguf_test --ar 0 1 4096` bez P2P i z P2P (ma być `ALL OK` w obu).

Opcje:

```bash
SKIP_AR=1 p2p-work/ab145fix.sh 2        # bez testu --ar
HOST_SANITY=1 p2p-work/ab145fix.sh 2    # + sanity DS4_AR_HOST=4 (ma nie paść, 12K)
TREE=... PORT=8080 p2p-work/ab145fix.sh # inne drzewo/port
```

Wyniki: `p2p-work/ab145fix-<tag>.tsv` (surowe liczby + linia P2P), `ab145fix-<tag>.progress`
(log), `ab145fix-<tag>-{A,B}<runda>.{server,pre12,pre24,dec1,dec4,p2p}`, `...-ar-{off,on}.txt`.

## 4. Co mnie interesuje (do oceny wyników)

- tabela `B vs A` (prefill 12K, 24K, dekod N=1, N=4) — oczekiwane ~+24-26% / +5-10% jak na v1.1.41;
- linia P2P: **B** ma wypisać `~1015 MB ... peers reaching them directly`; **A** nie ma wypisać nic;
  jeśli B ma `left host-mapped (no room)`, albo `nothing moved`, to inna przyczyna niż wydajność
  (podnieś `--reserve-mib`, sprawdź BAR1 w nagłówku skryptu);
- `AER+` w każdej rundzie — > 50 nowych wpisów = runda niewiarygodna (PCIe), powtórz;
- `AR: --ar` — `ALL OK` przy obu wartościach;
- `HOST_SANITY=1` — ma przejść 12K bez crasha (to była dziura K1).

Do mnie wystarczy: plik `.tsv` + `ab145fix-<tag>-*-ar-on.txt` + (jeśli coś podejrzane)
`ab145fix-<tag>-B1.server`. Wtedy ocenię wyniki i zaproponuję dalsze optymalizacje
(pull zamiast push, `kAr2Piece`, `DS4_AR_ZC=0`, tp3 z `--reserve-mib`).
