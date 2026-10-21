# Optional native autoencoder pretraining

October20,2026. Implementation work for Kinvert's optional5090pretraining request.
This work does not repair or waive the earlier compression memcheck failure.
No dataset collection or long pretraining is launched by preparing the tools.

October 20 evaluation clarification: a falling validation MSE is only a canary
for training-loop behavior, not encoder accuracy or proof that the ball survives
compression. Read [the literature review](ENCODER_EVALUATION_LITERATURE_20261020.md)
and [the proposed information-preservation evaluation](ENCODER_INFORMATION_EVALUATION.md)
before choosing an overnight pretraining objective. Per-object masks, latent
probes and paired nuisance tests are proposed there; they are not implemented by
the existing qualification command. Mathematical correctness and useful game
information are separate requirements.

The initial student is fullframe encoder8/compression_model0:full rendered RGB,
three narrow strided convolutions,4x4x16spatial pooling,128latent values. Reuse
its exact parameter order/import format. A training-only decoder maps128to
4x4x16,then nearest expansion plus3x3convolutions at ceil(H/16),ceil(H/4),H
(and corresponding widths),channels8/8/3. Final RGB prediction is linear.
The decoder is discarded on export. This is a deterministic reconstruction
autoencoder, with no VAE KL loss or quantized codebook.

Use FP32 native CUDA/cuBLAS; Python handles capture/configuration/manifests only.
No Torch/CUDA/dependency installation. Optional edge-weighted pixel MSE derives
weights from visible image contrasts, never simulator labels. Report ordinary
pixel MSE separately. Reconstruction quality is not downstream RL acceptance;
the128bottleneck may discard small objects or fine position.

Data are actual native headless full frames, stored in episode shards with
declared raw HWC/float32 NHWC and source/config/episode/skin/RNG identities.
Collection, dataset validation, pretraining and export are separate commands.
Train/validation trajectories have fixed disjoint environment/action seeds;
seeds are never swept. Track exact duplicate frames across splits instead of
claiming independent unseen imagery from episode separation alone. No labels
or hidden state enter the encoder. Skip post-terminal auto-reset observations
under the completed episode's identity. Fixed random actions are a transparent
first collection policy, not proof of broad game-state coverage.

Training must accept multiple raw shapes with shared weights, shape-specific
activation buffers and a deterministic balanced stream schedule. Preserve aspect
ratios. Per-shape native exports bind the same learned weights to each target's
strict fullframe-compression-encoder-v1contract. All pretraining/export inputs,
optimizer state, checkpoint steps and validation metrics remain hash-bound.
Resume restores optimizer/step/data order rather than silently restarting them.

Default collection/pretrain budgets are bounded pilots. Bigger5090runs are
explicit opt-ins with time/storage/step caps. Preparation is GPU-free. Before
any long run, require native math, repeatability, finite-loss/update, resume and
encoder import checks on that machine. WSL debugger/tool failure stays a failed
memory gate; do not change Windows or CUDA to conceal it.

Compare scratch, pretrained/fine-tuned and later pretrained/frozen/cached paths
using matched downstream learners. Frozen feature caching is a separate future
integration; pretrained initialization alone does not remove CNN computation.

Existing weights research:see PRETRAINED_BACKBONES_20261020.md. External weights
need their own architecture implementation/conversion and numerical equivalence;
they cannot be loaded into our small encoder merely because both produce latents.
