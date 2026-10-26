# Arena log check (`../../arena_log_check.py`)

One-time check of what ONNX Runtime's memory arena (BFCArena) does in runs 1 and 2, to explain the
run-2 RAM increase seen in the ORT memory settings experiment (`../ort_mem_settings/`).
Arena on, memory pattern on and off, one fresh process each, 1 image, 4 runs, ORT default logger at INFO
(arena events) plus RSS markers before and after every run.

| File | Contents | In git |
|---|---|---|
| `run_<timestamp>/log_<setting>.txt` | Raw stderr of the child process: ORT log and `@@MARK` lines with RSS / peak RSS | Yes |
| `run_<timestamp>/events.csv` | One row per arena event (region request, region taken, total, reserve) and the segment it fell in | Yes |
| `run_<timestamp>/segments.csv` | Per setting and segment (session, run1, run2, ...): regions taken, MiB from the OS, arena total, RSS change, peak RSS | Yes |
| `run_<timestamp>/summary.json` | The same with ORT version, threads and image | Yes |

Result of `run_20261026_122559` (ORT 1.30, 4 threads): in run 1 the arena takes three regions, 128 MiB
for the 120 MiB stem conv output, 128 MiB for the 30 MiB max-pool output and 256 MiB for the 142.5 MiB
Resize output, 513 MiB reserved in total, while RSS rises by 323 MiB (only written pages count). In run 2
no new region is taken in either setting; RSS rises by +7.5 MiB (pattern on) and +60 MiB (pattern off)
from pages written for the first time; runs 3 and 4 add nothing.
