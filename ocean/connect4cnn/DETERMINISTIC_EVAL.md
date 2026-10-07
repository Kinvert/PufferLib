# Dedicated deterministic Connect4 evaluation

This evaluates **exactly N assigned episodes once each**, using a frozen suite
shared by every checkpoint. It runs the normal native CUDA policy and native
Connect4 game without training. Python only prepares configs, launches the
binary and audits receipts. It supports the original state policy and all
existing pixel encoder selectors, retaining the supplied encoder shape,
projection, recurrent width and recurrent depth. GPU acceptance is recorded
separately below; accepting a configuration is not numerical verification of
every possible network.

## Create the suite once (no GPU)

From the repository root, using the existing Python 3.12 uv venv:

```bash
.venv/bin/python ocean/connect4cnn/deterministic_eval.py suite \
  --out build/connect4cnn/eval-suite-dev-01 \
  --seed 51005 --episodes 1000 --slots 64 --representation 0
```

Keep **both** `suite.json` and `episodes.csv`. The manifest lists each episode's
ID, environment seed, policy sampling seed, empty board and rendering. Its hash
is checked against the versioned mapping before every run. Output directories
must be new; partial and failed runs are never overwritten.

Use this identical suite for each model/checkpoint. `--offset` selects a block
of episode IDs; `--episodes` accepts 1 through 1,000,000, not only multiples of
the inference batch size. `--seed` is an unsigned 32-bit suite seed, independent
of training seeds and network initialization. Choose a separate fresh seed for
final held-out evaluation and mark it `--purpose heldout`; that label alone
does not prevent repeated test-set tuning. Freeze training/selection before
examining final held-out results.

Connect4 always starts with an empty 7×6 board. Diversity here comes from
opponent tie-breaking and policy sampling, not randomized board spawns.
The environment and policy RNG streams are separately keyed by suite seed and
episode ID. Different actions can produce different trajectories and legal
opponent choices; shared randomness does not force identical game trajectories.

## Build the matching policy executable

Use the normal build path and existing CUDA/NCCL installation. No installation
or global CUDA changes are necessary:

```bash
source ocean/connect4cnn/runtime_env.sh
NVCC_ARCH=sm_120 bash build.sh connect4cnn build/connect4cnn/eval-pixels --float
```

The default pixel binary supports numeric encoder IDs 0–5: tiny, experimental,
Nature, compact, Flex and Flex2. It uses the values in the **resolved training
INI**, not a hand-recreated subset. For models whose saved `policy.encoder=0`
selected a different compiled default, build accordingly:

```bash
NVCC_ARCH=sm_120 NVCC_EXTRA=-DC4_NATURE_CNN \
  bash build.sh connect4cnn build/connect4cnn/eval-nature --float
NVCC_ARCH=sm_120 NVCC_EXTRA=-DC4_IMPALA_CNN \
  bash build.sh connect4cnn build/connect4cnn/eval-impala --float
NVCC_ARCH=sm_120 NVCC_EXTRA=-DC4_IMPOOLA_CNN \
  bash build.sh connect4cnn build/connect4cnn/eval-impoola --float
NVCC_ARCH=sm_120 bash build.sh connect4 build/connect4cnn/eval-state --float
```

`eval_exact_info` prints the executable's environment, compiled default,
precision, game rules and receipt support without accessing a GPU. The runner
rejects a mismatch with your `--family` and resolved INI. A raw `.bin` does not
contain its architecture: retain its original training INI and build provenance.
Parameter counts cannot distinguish all configurations with equal-sized weights.

## Evaluate any selected checkpoint

Example for the local corrected-rules quality checkpoint (these artifact paths
exist on G240; the 5090 must use its own checkpoint and resolved INI):

```bash
source ocean/connect4cnn/runtime_env.sh
.venv/bin/python ocean/connect4cnn/deterministic_eval.py run \
  --suite build/connect4cnn/eval-suite-dev-01/suite.json \
  --binary build/connect4cnn/eval-pixels --family flex \
  --config build/connect4cnn/compare.a2lt6qft/flex_quality-s173/metrics/connect4cnn/trial.ini \
  --checkpoint build/connect4cnn/compare.a2lt6qft/flex_quality-s173/checkpoints/connect4cnn/trial/0000000013312000.bin \
  --out build/connect4cnn/eval-quality-dev-01
```

Select `--family state|tiny|experimental|nature|compact|flex|flex2|impala|impoola`
to match the supplied config/build. Change only those inputs and output folder
to evaluate another policy against the **same suite**. Full resolved native
INIs normally live at `metrics/<environment>/<run_id>.ini`.

The runner refuses an occupied GPU and acquires the local benchmark lock.
Default per-process timeout is 300 seconds, configurable with `--timeout`.
Failures retain logs and partial episode rows and produce `status=failed`, never
a completed score. A timeout must not be scored using only finished episodes.
No long training, sweep or encoder-5 GPU campaign is authorized by these commands.

## What is held constant and recorded

- Environment version `connect4-full-board-draw-v2`, initially empty boards,
  episode IDs, opponent RNG seeds, categorical sampling seeds and inference
  slot count. Checkpoint construction RNG is irrelevant after episode reseeding.
- Normal recurrent reset at every episode boundary, including every new wave.
  The evaluator uses PufferLib's actor CUDA stream so graph capture works.
- A wave contains at most `slots` assigned games. Finished slots stop stepping;
  dummy inference neither creates nor counts replacement episodes. The final
  partial wave counts only its assigned games. Every assigned game must finish
  within Connect4's 21 player decisions.
- Fixed pixel appearance for each suite (`--representation 0` through `9`).
  Slot-based mixed training appearances are deliberately rejected by the native
  exact path. Run matched suites for each desired rendering; do not pool an
  architecture's easiest rendering into a purported robustness result. A state
  policy sees the same underlying games; pixel appearance does not apply to it.
- Supplied and effective INIs; checkpoint/binary/config/suite hashes; all per-game
  results, action hashes, initial observation hashes, initial board and RNG
  receipts; hardware identity; native completion count and parameter count.
  The native path audits checkpoint float32 count and finite weights.

Changing a policy architecture or size never changes the suite mapping. Keep
batch size, CUDA/software and hardware fixed for bitwise repeatability claims;
floating-point behavior across GPU architectures is not promised identical.
The exact loop steps environments directly; a worker-count comparison would
not test parallel CPU stepping. It is single-GPU, float32, no self-play.

## Verification and evidence

[October 5 RTX 5060 acceptance](../../research/results/connect4cnn/deterministic-eval-20261005/README.md)
passed eight checkpoint configurations, exact 1,000-episode suites, repeated
processes, eager/graph equality and independent final-wave replay. One-episode
unsigned-boundary/rendering checks and malformed/nonfinite checkpoint rejection
also passed. Numeric encoder-5 GPU acceptance remains pending on the 5090.

GPU-free configuration/manifest/corruption tests:

```bash
.venv/bin/python ocean/connect4cnn/tests/test_deterministic_eval.py
.venv/bin/python ocean/connect4cnn/tests/test_claim_tools.py
```

Opt-in GPU acceptance uses **existing checkpoints**, without training:

```bash
source ocean/connect4cnn/runtime_env.sh
.venv/bin/python ocean/connect4cnn/tests/verify_deterministic_eval.py \
  --cases path/to/cases.json --suite path/to/suite.json \
  --out build/connect4cnn/eval-acceptance-NEW_ID
```

Cases JSON is a list of objects with `name`, `family`, `binary`, `config`, and
`checkpoint`. Each case runs graph inference twice, eager inference once, and
the last wave alone in a fresh process. It requires identical per-episode
outcomes/action hashes across repeats and eager/graph, and equality of the
independently replayed last wave. This directly tests recurrent reset and
episode identity across prior waves. Paths are relative to the invoking cwd.

This is a Connect4 evaluator. Pong and Flappy require their own episode/reset
and censoring contracts; their older pooled evaluators are not repaired by it.
Historical pooled-v1 scores remain historical. Old checkpoints may be used as
evaluator fixtures under corrected rules, but those reevaluations are not
reproductions of their original training experiment. Thousands of evaluation
games do not replace independently replicated training seeds or calibrated
uncertainty over the complete Pareto frontier.
