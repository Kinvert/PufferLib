# PongCNN

Native PufferLib Pong with a synthetic pixel observation. This is PufferLib's own game, **not ALE/Atari Pong**, and has no inherited Atari benchmark score. Copied from `ocean/pong/pong.h` (git blob `5fa3806cb1be8b43f93a831a4820c263abc015a9`). Original Pong is unchanged.

## Observation and game

Float32 CHW `[1,36,44]`, values in `[0,1]`, top row first. Rows 0 and 1 are left/right score bars: total row intensity divided by 44 equals score divided by the winning score. Fractional edge pixels preserve that ratio. The remaining 34 rows show the arena, with two columns on each side for the paddles and 40 columns for ball motion. Left paddle is 0.5, right/agent paddle 0.75, ball 1, background 0. Rectangle coverage uses floor/ceil bounds and clipping, so small visible objects retain pixel coverage. World y points up; image y points down. The 500×640 world is scaled into this image, not resized in the simulation.

Generation clears the observation buffer and fills rectangles directly in C, with no allocation, graphics context, screenshot, external renderer or RNG use. The existing Raylib viewer remains optional and reads game state. It is a human game view, not an exact preview of the policy image. Image generation happens once after the skipped physics frames, including the existing early-return reset paths. All physics, frame skip, three discrete actions, scripted opponent, rewards, score limits and same-step reset semantics are preserved. The environment also retains Pong's continuous-control branch; research configs use discrete control.

**Information differences matter:** state Pong explicitly exposes velocity, while a single pixel frame does not. The recurrent core must infer motion across decisions, and rasterization loses subpixel position precision. Also, the original state header computes both score observations with unsigned integer division, making them zero during normal play before the automatic reset. The pixel score bars expose the current score. This existing state behavior is preserved, not silently repaired. State/pixel comparisons therefore measure different observation interfaces, not strictly equal information. Encoder-to-encoder comparisons all receive exactly the same pixels and score bars.

Native `perf` is the episode-averaged fraction of points won, **not match win rate**. `score` is right score minus left score at match end. Preserve these definitions in reports; do not reuse the Connect4 report's win-rate labels.

## Encoders and configuration

The native custom-encoder hook also recognizes `PUFFER_PONGCNN`. It reuses the existing CUDA encoder implementations under `ocean/connect4cnn/`; no kernels are copied or changed. The initial shape deliberately satisfies their existing 1×36×44 contract. This does not make the kernels support arbitrary resolutions/RGB yet.

- `policy.encoder=4`: existing flexible family; default here is 16 channels, one 7×7/stride-4 layer, projection 64, flatten readout. This shape was selected on Connect4, not tuned on Pong.
- `policy.encoder=2`: adapted Nature.
- Build with `NVCC_PREPEND_FLAGS=-DC4_IMPALA_CNN` or `-DC4_IMPOOLA_CNN` and use `policy.encoder=0` for those references. The `C4_` names are existing shared-build selectors. IDs 1/3 and the compiled tiny default retain their existing meanings, but are outside this first canary panel.

`config/pongcnn.ini` starts from stock Pong's environment and learner/core settings with an added encoder. It is not tuned for pixels. For controlled plumbing, `compare.ini` supplies identical environment/learner/core settings to state, Flex, Nature, IMPALA and Impoola: hidden 128 × one recurrent layer, 64 agents, two threads, horizon/minibatch 32/2048, replay ratio 1, learning rate .001, synchronous float32 GPU learner. Several learner settings differ from tuned stock Pong; neither this state control nor the CNN is an untouched stock-tuning baseline.

## Validation and execution

CPU validation (existing shared Raylib, no window/GPU or installation):

```bash
bash ocean/pongcnn/tests/run_all.sh
```

Passed September 14, 2026: 49,152 transition traces match original Pong byte-for-byte, across eight seeds, frame skips 1/3/8 and discrete/continuous actions. Each trace entry hashes the complete environment state/logs/RNG with agent/client pointers cleared, and records rewards and terminals separately. Repeating the pixel run matches exactly. Explicit fixtures cover both sides' scores and match resets, corner placement, orientation, offscreen clipping, score bars, dirty-buffer overwrite, no velocity leakage into pixels, and observation guard values, under ASan/UBSan. These are same-host CPU checks, not GPU determinism or learning results.

Prepare the fixed five-model canary without compilation or GPU work:

```bash
bash ocean/pongcnn/canary.sh --prepare-only
```

After Connect4 confirmation has finished, run outside the sandbox:

```bash
NVCC_ARCH=sm_120 bash ocean/pongcnn/canary.sh
```

This creates a fresh directory, builds all five binaries, then serially trains each for 65,536 decisions with four checkpoints, reloads the final checkpoint and evaluates 64 requested matches on a separate seed. Each train/eval has a 120-second cap. Source/config snapshots and hashes, build/command receipts, checkpoint hashes, full native metrics, process wall time and raw evaluation outputs are stored under `build/pongcnn/canary.*`. Configs are isolated per job; the checkout defaults are never rewritten. The runner refuses to start a training job when another GPU compute process is present. All configuration/reporting orchestration here is Bash; training and evaluation are native C/CUDA. No W&B upload or PROTEIN search is launched by this canary.

**GPU builds, reload and learning remain unvalidated.** They were deliberately deferred while the Connect4 confirmation occupied the GPU. A canary is a plumbing test, not evidence that 65K decisions is an adequate learning budget. After it passes, establish learning-scale budgets and run the same frozen architectures across paired seeds; document the entire curves and the state-observation caveats before considering `cnn3` architecture search. Pong is now a development task, not an untouched final-test game.
