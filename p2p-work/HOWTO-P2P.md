# Jak używać P2P all-reduce w Mayi (DS4, tp2) — 30.10.2026

**Produkcja (v1.1.41): `~/project-maya-p2p141` (gałąź `p2p-141`, niezacommitowane), binarka `~/project-maya-p2p141/build/maya-ds4`** — w configu ustaw `exe` i `cwd` na to drzewo. Sprawdzone 30.10: CLI tp2u 12K prefill 777/771/770 tok/s, dekod 100-123, bufory 1015 MB w VRAM.
Starsza wersja na v1.1.37: `~/project-maya-p2p` (gałąź `p2p-ar`, binarka `build/maya-ds4`) — nie dla produkcji.
Nie przebudowywać i nie kasować tej binarki, dopóki ktoś jej używa.

## Włączenie
1. Skopiuj działający config (`~/project-maya/maya-ds4.json`) np. do `maya-ds4-p2p.json` i zmień:
   - `"exe": "/home/michal/project-maya-p2p/build/maya-ds4"`, `"cwd": "/home/michal/project-maya-p2p"`
   - w `"env"` dodaj `"DS4_AR_P2P": "1"` (reszta bez zmian; AR2 zostaje `MAYA_DS4_AR2=1`).
   Gotowy wzór (z configu z 29.10): `~/project-maya/p2p-work/cfg-B.json` (bez P2P ten sam: `cfg-A.json`).
2. Start: `cd ~/project-maya && .venv/bin/python maya.py --config maya-ds4-p2p.json --port 8080 --plain --yes`
3. Sprawdź w logu silnika linię: `ds4: DS4_AR_P2P: ~1015 MB of all-reduce buffers in their group's first GPU, peers reaching them directly`.
   - dopisek `left host-mapped (no room)` = część buforów nie weszła do VRAM -> podnieś `--reserve-mib`
   - `DS4_AR_P2P_LOG=1` w env: wypisze grupę, która zostaje w RAM (BAR1 < 1 GiB albo brak peer access).
4. Wyłączenie: usuń `DS4_AR_P2P` z env (albo użyj configu A). Nic więcej.

## Warunki
- Grupy tensorowe po 2-4 karty; każda karta grupy musi mieć BAR1 >= 1 GiB (ReBAR; sprawdź `stat -c %s /sys/bus/pci/devices/0000:<bus>/resource1`). Inaczej grupa sama zostaje na RAM.
- tp3: bez zysku. tp2 (obecna produkcja): prefill +15..26%, dekod 0..+10% (zależnie od sprzętu — patrz WYNIKI-noc.md).
- Nie ustawiać peer access samemu w trakcie ładowania — silnik włącza go na końcu `load()` (wcześniej psuło `route_kernel`).

## Uwaga o sprzęcie (30.10)
Palit w slocie 02:00.0 (GPU1) + błędy PCIe na porcie 00:01.1 (GPU0): P2P dawał tam gorsze wyniki (serwer: prefill +15%, dekod N=1 -20%). Przed pomiarem: `journalctl -b 0 -k | grep -a -c aer_status` przed i po; >50 nowych = wynik niewiarygodny. Mierzyć A/B tą samą binarką, na przemian, z rozgrzanym cache.

## Pomiar
`~/project-maya/p2p-work/srv.sh NAZWA CONFIG` (start serwera, prefill 12K/24K, dekod N=1/4, zatrzymanie). UWAGA: skrypt zatrzymuje WSZYSTKIE serwery pasujące do `maya.py` / `project-maya*/build/maya-ds4`; przy cudzym teście ubijać po PID.
