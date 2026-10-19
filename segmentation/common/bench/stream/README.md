# STREAM memory bandwidth benchmark

Used in experiment A to measure the real RAM bandwidth of the Raspberry Pi 5 (roofline chart).

| Item | Value |
|---|---|
| Source | Official STREAM, John D. McCalpin, https://www.cs.virginia.edu/stream/FTP/Code/stream.c |
| Version | `stream.c,v 5.10 2013/01/17` |
| sha256 of `stream.c` | `a52bae5e175bea3f7832112af9c085adab47117f7d2ce219165379849231692b` |
| Downloaded | 2026-10-15 |
| Licence | Free to use and redistribute (header of `stream.c`). Published results must follow the STREAM run rules, or be labelled as based on a variant of STREAM if the code or run rules are changed |

## Not in git: compiled binary `stream`
The binary built from `stream.c` is excluded by `segmentation/.gitignore` (`common/bench/stream/stream`).
Rebuild it with (array size and flags recorded here once fixed in experiment A):

```bash
gcc -O3 -fopenmp -DSTREAM_ARRAY_SIZE=80000000 -DNTIMES=20 stream.c -o stream
OMP_NUM_THREADS=4 ./stream
```

Array size 80 M doubles = 610 MiB per array, 3 arrays ≈ 1.8 GiB, far larger than the 2 MB L3 cache,
so RAM is measured rather than cache.

## Results: Raspberry Pi 5 (16 GB, LPDDR4X), 2026-10-15
gcc 14.2.0, flags above, `performance` governor, 2.4 GHz, 43–53 °C, no throttling. Exploratory setup
(Claude Code open). Best rate in MB/s (10^6 bytes/s, as STREAM reports), 2 repeats × binding
`none` / `OMP_PROC_BIND=spread` = 4 runs per thread count; all runs within ±1.5%.

| Threads | Copy | Scale | Add | Triad |
|---|---|---|---|---|
| 1 | 11,840–11,955 | 11,544–11,590 | 11,966–12,034 | 12,229–12,254 |
| 2 | 10,194–10,402 | 10,182–10,472 | 9,311–9,365 | 9,982–10,061 |
| 4 | 8,320–8,466 | 8,274–8,370 | 7,882–7,974 | 7,948–7,983 |

More threads give **less** bandwidth: one core reaches about 12 GB/s (≈70% of the theoretical
~17 GB/s of 32-bit LPDDR4X-4267), four cores together about 8 GB/s. Same in every repeat and binding
mode. Likely cause (not verified): concurrent streams from several cores cause more DRAM row
conflicts in the memory controller. Used as the memory ceiling in the experiment A roofline
(4 threads: ≈ 8.2 GB/s, the setting ONNX Runtime uses).
