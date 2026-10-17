# Sprite CNN overnight run on RTX 5090

This is a development architecture search, not held-out paper confirmation.
Run from this iteration's committed source in an idle5090 checkout. No coding
is required on the5090. Do not edit a checkout serving an ongoing campaign.

## Delivery status

Delivery uses **Kinvert/PufferLib, branch `cnn-research`**. The delivery merge
preserves both the existing remote image-research history and sprite tooling
through `d4527b4e`. Do not force-push, reset an active campaign, or use upstream.

In the idle5090 checkout, before preparation:

```bash
git switch cnn-research
git pull --ff-only origin cnn-research
git merge-base --is-ancestor d4527b4e HEAD
git merge-base --is-ancestor ba7a9b164e851925db283baa4c55c46f8d9d1e45 HEAD
```

If pull reports local divergence or modifications, stop and preserve that
checkout; use a separate clean checkout rather than discarding its work.
The launcher records the actual pulled revision. Publication does not mean
the full5090 campaign has run: that allocation remains PREPARED, NOT RUN.

Merged-source build, raster and host verification is recorded in
`research/results/sprite-delivery-reconciliation-20261016/README.md`.

## Existing environment

Reuse working CUDA/NCCL/raylib and uv Python3.12. No new Python packages,
Torch imports, CUDA installation, sudo or Docker. A fresh checkout may need
links to the existing venv/raylib; never replace existing folders.

If the established helper is outside this checkout:

```bash
export SPRITE_RUNTIME_ENV=/home/keith/Git/ml/cnn-5090/build/connect4cnn/runtime-5090.sh
```

Otherwise the launcher sources local `build/connect4cnn/runtime-5090.sh`
when present, then `ocean/connect4cnn/runtime_env.sh`. Explicit existing
`CUDA_HOME`/`NCCL_ROOT` paths also work.

## Prepare and launch

From the correct checkout/revision:

```bash
bash research/run_sprite_overnight.sh prepare build/sprite-night-5090-01
```

Preparation compiles six native binaries, native PROTEIN and independent
CUDA checks, runs scalar raster/host tests, and captures recipes/source/
binary/toolchain identities. It does not query/use the GPU. Fresh directories
are required; completed allocations are never automatically restarted.

When the5090 is idle, launch ONCE:

```bash
nohup bash research/run_sprite_overnight.sh run build/sprite-night-5090-01 \
  > build/sprite-night-5090-01/supervisor.txt 2>&1 < /dev/null &
```

The same command without nohup/backgrounding works in a persistent terminal.
Do not run both forms. No TUI watching is needed. Read `campaign/progress.json`
and `campaign/candidates/*/progress.json`; native output stays in log files.

## Allocation and loop

- Catch, Platform, Maze, Runner, Key and Push. Forage/Lane retain distinct
  contracts and are excluded from this first packed campaign.
- Same96-wide by192-high RGB player/gaze image plus four geometry values.
  Every agent learns its own movement and gaze actions from random weights.
- Fixed tile/Nature/IMPALA/Impoola controls, then up to24 native PROTEIN Flex
  proposals. Flex searches18 CNN coordinates on the packed image. Our current
  tile model is a fixed reference; its internals are not separately swept.
- EVERY candidate/control trains all six games at all seven fixed replicas:
 7/42/69/123/173/2027/9001. Never sweep, select or replace a seed.
- Common1,048,576-decision training, eight checkpoints at131,072-decision
  intervals, A32/H32/MB32. Per-game LR/entropy/gamma/core settings match across
  candidates. These conservative learners are not asserted to be equally
  optimal for every encoder family.
- Every checkpoint gets all three skins,16 exactly assigned development
  episodes/skin, suite420 and16 slots. Quotas do not favor early finishers.
- Feedback uses the final scheduled checkpoint: mean skins/replicas per game,
  then equal-game mean of declared0→1 utilities. Metrics still have different
  meanings; retain components and worst-game utility. Cost is total native
  training process seconds, charged once/job. Build/math/eval time is separate.
- Native PROTEIN observes each completed aggregate and proposes the next CNN.
  Random warmup is explicit; GP count must become nonzero after three observed
  proposals. Duplicates and inactive coordinates remain in the receipts.

Maximum allocation is168 reference trainings plus1008 proposal trainings;
it is not a promise all fit overnight. The12-hour cap and actual completion
status are authoritative. Do not report the allocation as completed results.

## Automatic checks and limits

Before full training: independent float64 CUDA positive/zero forward and
all-encoder-gradient tests for all five families at B1/3/32; repeated8192-
decision training; reload and graph/repeat/eager exact-tail evaluation on
every family/game. Each new Flex proposal gets its own B1/32 numerical tests.
These fixtures do not certify arbitrary ReLU branches, all inputs, or
independent MinGRU/optimizer derivatives.

`campaign/calibration.json` records actual canary SPS and conservative
training-time projections. Projections are estimates and do not change the
frozen budgets or select a seed/recipe.

The12-hour GPU execution cap includes validation, excludes preparation and
stopped-run reporting. Individual processes also have timeouts. Deadline/
storage stops preserve partial candidates without feeding incomplete means
to PROTEIN. A math/native failure stops the campaign. Do not silently retry,
lower tolerances, drop failed seeds or shrink budgets.

The shared lock and idle checks reject competing work. A5-second process
sampler records observed interference and stops only our process group.
Sampling cannot prove an uncooperative sub-five-second GPU job never ran.
Keep hardware collections separate; retain contaminated timing receipts.

Storage preserves20GiB free and caps artifacts at100GiB. All checkpoints stay
on the5090; none are automatically deleted. Request a new allocation if these
caps stop the run. Preserve `result.json`, logs and partial evidence.

## Inspect and return evidence

After stopping:

```bash
bash research/run_sprite_overnight.sh report build/sprite-night-5090-01
bash research/run_sprite_overnight.sh export build/sprite-night-5090-01
```

Start with `campaign/result.json`, `report.json`, `calibration.json`,
`curves.html`, `curves.csv`, and candidate `complete.json`/`progress.json`.
HTML keeps missing baselines visible and plots only complete assigned-replica
means, with every checkpoint/decline. No confirmed statistical frontier flags.

Export prints one compact `.tar.gz` plus checksum under
`build/hardware-artifacts`. It includes configs, receipts/logs, episode CSVs,
source snapshots, curves and hashes. Raw checkpoints/numerical arrays remain
on the5090, explicitly excluded from compact export. Keep them for auditing;
do not push weights/build directories into Git.

Suites2026101601/2026101602 remain reserved for later frozen-architecture
confirmation and are not adaptive feedback. No paper/SOTA/generalization
claim follows automatically. Fixed/random-gaze controls are a separate
ablation before claiming a benefit specifically from learned attention.
