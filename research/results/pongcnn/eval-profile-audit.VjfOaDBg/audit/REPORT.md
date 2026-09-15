# Pong evaluation/backend audit

Original: `/home/keith/Git/ml/cnn-5090/build/pongcnn/confirm-5090.full.JXYKB06r`.

The original 40 training runs and 312 successful / 8 timed-out evaluations remain unchanged.

## Evaluation panel

| Checkpoint | Original | Old exit / seconds | New exit / seconds | New last progress steps / matches |
|---|---|---|---|---|
| 00-a-flex_quality-s31002-2621440 | timeout | 124 / 30.096502 | 124 / 30.096114 | 18190336 / 13 |
| 01-a-flex_quality-s31002-3145728 | timeout | 124 / 30.103523 | 124 / 30.098822 | 17121280 / 0 |
| 02-a-flex_quality-s31002-3670016 | timeout | 124 / 30.088995 | 124 / 30.109231 | 18219008 / 3 |
| 03-a-flex_quality-s31002-4194304 | timeout | 124 / 30.101502 | 124 / 30.095702 | 17326080 / 1 |
| 04-a-nature_cnn-s31005-3145728 | timeout | 124 / 30.099728 | 124 / 30.103729 | 15132672 / 3 |
| 05-a-nature_cnn-s31005-3670016 | timeout | 124 / 30.088661 | 124 / 30.105026 | 16050176 / 1 |
| 06-a-nature_cnn-s31005-4194304 | timeout | 124 / 30.098098 | 124 / 30.10065 | 14639104 / 1 |
| 07-b-nature_cnn-s31004-4194304 | timeout | 124 / 30.102451 | 124 / 30.100683 | 16093184 / 2 |
| 08-a-flex_quality-s31001-524288 | ok | 0 / 4.826712 | 0 / 4.899081 | — / — |
| 09-a-nature_cnn-s31001-524288 | ok | 0 / 2.119793 | 0 / 2.153207 | — / — |
| 10-a-impala_cnn-s31001-524288 | ok | 0 / 17.374578 | 0 / 17.385927 | 1996800 / 448 |
| 11-a-impoola_cnn-s31001-524288 | ok | 0 / 6.301666 | 0 / 6.280568 | 661504 / 420 |
| 12-b-flex_quality-s31001-524288 | ok | 0 / 5.617216 | 0 / 5.587326 | 3276800 / 480 |
| 13-b-nature_cnn-s31001-524288 | ok | 0 / 2.58833 | 0 / 2.581528 | — / — |
| 14-b-impala_cnn-s31001-524288 | ok | 0 / 16.155126 | 0 / 16.208337 | 1980416 / 475 |
| 15-b-impoola_cnn-s31001-524288 | ok | 0 / 1.61783 | 0 / 1.701924 | — / — |

## Short performance diagnostics

131,072 decisions, seed 52001; shortened annealing schedule. Instrumented durations include profiler overhead and are not publication throughput.

| Cell | Mode | Valid | Process seconds | Native seconds | Updates |
|---|---|---|---:|---:|---:|
| a-flex_quality | unprofiled | True | 0.607579 | 0.31502723693847656 | 64 |
| a-flex_quality | instrumented | True | 2.723064 | 0.34293198585510254 | 64 |
| a-nature_cnn | unprofiled | True | 0.642851 | 0.33883237838745117 | 64 |
| a-nature_cnn | instrumented | True | 2.69134 | 0.3636205196380615 | 64 |
| a-impala_cnn | unprofiled | True | 4.4383 | 4.154934644699097 | 64 |
| a-impala_cnn | instrumented | True | 7.666875 | 4.316418647766113 | 64 |
| a-impoola_cnn | unprofiled | True | 4.414726 | 4.129648923873901 | 64 |
| a-impoola_cnn | instrumented | True | 7.708112 | 4.262478351593018 | 64 |
| b-flex_quality | unprofiled | True | 0.750086 | 0.464357852935791 | 192 |
| b-flex_quality | instrumented | True | 2.95063 | 0.49948740005493164 | 192 |
| b-nature_cnn | unprofiled | True | 0.785922 | 0.5022721290588379 | 192 |
| b-nature_cnn | instrumented | True | 3.208242 | 0.535348653793335 | 192 |
| b-impala_cnn | unprofiled | True | 10.801918 | 10.502366542816162 | 192 |
| b-impala_cnn | instrumented | True | 14.97858 | 10.813774108886719 | 192 |
| b-impoola_cnn | unprofiled | True | 10.696782 | 10.40699553489685 | 192 |
| b-impoola_cnn | instrumented | True | 14.92263 | 10.780262231826782 | 192 |

Raw configs, commands, low-frequency activity, progress and profiler output accompany each attempt. Full measurements are in `measurements.json`. Interpretation and profiler coverage are recorded separately in `FINDINGS.md`.
