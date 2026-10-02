# Game-frame pretraining with pixel-space labels

September 26, 2026. Design/research only: **no dataset has been generated and no collector or pretraining trainer has been implemented.** Kinvert's requirement is to train a reusable CNN from game images, with targets located in the image itself. Architecture search and pretraining are separate experimental axes.

## The training contract

One sample is an actual game frame plus annotations in that final frame's pixel coordinates. The CNN input is the image alone. Native game state may generate supervision, but world coordinates, velocity, board arrays and hidden entities must not enter the CNN input or substitute for image-space targets.

Proposed fields:

| Field | Meaning |
|---|---|
| Image | Exact pixels offered to the encoder, width, height, channels, layout and value range |
| Box | Visible-object bounds `[x0, y0, x1, y1)` in final-image pixels, origin at top left |
| Class | A visual/semantic category such as ball, paddle or occupied piece; versioned class table |
| Instance | Optional identity persistent within an episode; never an additional CNN input |
| Mask | Optional visible-pixel instance mask, with the same height and width as the image |
| Center | Optional center defined explicitly as box center or visible-mask centroid |
| Provenance | Game/source revision, episode/frame ID, environment/appearance seeds, representation and rasterizer version |

Boxes are half-open: `0 <= x0 < x1 <= width`, `0 <= y0 < y1 <= height`. Normalized coordinates can be derived using the recorded width/height. A COCO export converts XYXY to XYWH explicitly; it must not silently reinterpret the same four numbers. A diagnostic overlay is a separate output: never draw target boxes into the training image.

Class names must describe what can be inferred from the supplied image/history. For example, do not classify two visually identical objects by a hidden allegiance. A role label such as player paddle is acceptable when the chosen image representation makes that role identifiable. Persistent IDs enable tracking targets, but arbitrary IDs are not meaningful classes to predict from a single frame.

## Derive annotations from the final rasterization

The strongest PufferLib path is an opt-in native collector around the existing policy pixel buffer. It should use the same geometry and draw order as pixel generation, and capture pixels and labels at the same simulation instant. This avoids a second renderer and per-environment Raylib installations.

For simple non-overlapping rectangles, reuse the rasterizer's rounded, clipped pixel extents. For shapes or overlapping objects, an instance-ID raster alongside the image can identify the pixels actually visible after draw order/occlusion. Derive visible boxes and masks from that raster. This is a proposed implementation, not existing code. Allocate annotation buffers once; keep collection disabled in ordinary training and measure overhead separately.

World geometry projected onto the screen is only an intermediate step. Labels must follow every crop, resize, letterbox, camera transform and appearance transform applied to the training image. Offscreen or fully hidden objects have no visible-object target. A partially hidden object's visible box encloses its surviving pixels; an amodal box, if ever included, must have a different explicit field and objective. Unsupported occlusion/alpha semantics should mark an annotation unsupported rather than pretend a projected rectangle is a visible mask.

Wrapping all Raylib draw calls is not a sufficient object annotator: one object may use many primitives, HUD elements may share primitives, and a window renderer may differ from policy observations. Associate primitives with semantic object IDs before rasterization if that renderer is the selected image source. For a future 3D game, projection/occlusion belongs in the collector; exported supervision still uses 2D final-frame pixels.

Capture terminal frames before same-step auto-reset if they are included. Never pair a pre-reset image with post-reset state. Record terminal/reset boundaries and frame order, and do not create temporal pairs across episodes.

## Concrete mapping in this checkout

The current policy images are normalized float grayscale, `1 x 36 x 44`; the human viewer is a separate image source. Any conversion to bytes would need a declared quantization rule and audit, particularly for Pong's fractional score-bar pixels. Prefer preserving the original float pixels in a first native prototype.

**Connect4CNN:** `ocean/connect4cnn/connect4cnn.h::compute_observation` maps the 7-column, 6-row board to pixels. Representation 0 uses filled 6-by-6 cells and one black column at each side. Board row 0 is rendered at the bottom. Other representations center the board and use per-owner glyph masks. A piece's visible box must enclose the rendered glyph pixels, rather than blindly use the full cell; disks, gaps and X/O change those extents. Empty cells are background for a piece detector. Board/grid targets could be a separate task if useful. Do not export bitboards as CNN inputs.

**PongCNN:** `ocean/pongcnn/pongcnn.h::pongcnn_rect` uses floor/ceil and clamps to the court below the two score rows. `compute_observations` scales world coordinates, reverses vertical direction, draws paddles then the ball, and applies representation transforms. Horizontal reflection maps a half-open box to `[W-x1, y0, W-x0, y1)`. Court-only vertical reflection uses `y0' = H + score_rows - y1`, `y1' = H + score_rows - y0`; score bars stay fixed. Grayscale remapping/inversion preserves geometry. Labels must come from the resulting frame and preserve semantic roles through a reflection; leftmost position is not automatically the player class. Do not label an occluded paddle pixel as visible. Single-frame supervision does not directly reveal ball velocity; temporal frame pairs are a separate option.

These are two integration examples, not a sufficient general-purpose corpus. A later corpus needs diverse games, scenes, object sizes, appearances and motion. State-derived perfect annotations only cover games whose render/state correspondence we actually implement and validate.

## Existing precedents and alternatives

[OCAtari](https://arxiv.org/abs/2306.08649v2) provides object extraction for Atari, and its [official code](https://github.com/k4ntz/OC_Atari) offers RAM- and vision-based extraction paths. It is a useful starting point for Atari screen-space annotations, not proof that every game's boxes are complete or exact. Pin a code revision and verify game-specific alignment against captured frames before using its labels. This is research into reuse; no Atari data have been collected here.

[SGI](https://arxiv.org/abs/2106.04799v1) demonstrates encoder pretraining from unlabeled Atari observations with latent dynamics and goal-conditioned objectives. It motivates keeping a temporal/self-supervised alternative alongside detection. It does not establish that bounding-box pretraining will improve our PPO-style learner or that ImageNet is optimal for small game frames.

The original [Nature DQN](https://www.nature.com/articles/nature14236) and [IMPALA](https://arxiv.org/abs/1802.01561v3) encoders were trained with their RL systems, not supplied as ImageNet-pretrained backbones. Their architecture names do not imply pretrained weights. The current native references in this checkout also start from random initialization.

## How the backbone would learn

Start, when collection is authorized, with object centers/classes and box sizes or visible masks. Attach a training-only detection/segmentation head to a spatial feature map **before global pooling or flattening**. At RL initialization import the matching encoder weights and discard that head; train the recurrent core and policy/value heads for the RL task. Detection is an auxiliary objective for useful visual features, not a claim that detecting objects teaches strategy.

Check whether the chosen small encoder retains enough spatial resolution for tiny balls/pieces. Excessive early stride can erase localization; a dataset cannot recover information lost by the architecture. Auxiliary multi-scale heads are an option to study, not a reason to permanently burden inference.

Do not assume one pretrained file initializes arbitrary sweep graphs. Start with one frozen encoder graph and a versioned shape/operation manifest. Reject incompatible imports. Compare scratch versus pretrained **on the same graph**, with a separate frozen-backbone control and fine-tuned-backbone run. Keep learnable activation coefficients in any weight/shape contract. Architecture inheritance/distillation can be investigated later.

For storage, a simple native shard of frame data plus fixed records is a candidate for fast training; COCO-style JSON is an interchange/inspection export. Retain a schema version, checksums and sample-to-label linkage. No need to write per-frame YAML during the simulation loop. Measure disk bandwidth and compression cost before choosing a format or claiming generation speed.

## Data and evidence requirements

Split by complete trajectories/level seeds, never random adjacent frames. For an across-game pretraining claim, keep held-out game identities out of collection, pretraining, architecture selection and objective tuning. Record policy used for collection and ensure random trajectories do not dominate with repeated empty/static frames. A future generator should deliberately cover reachable object configurations, appearance seeds and meaningful rare scenes, while documenting any synthetic states outside ordinary gameplay.

Before scale-up: inspect frame/label overlays, pixel-boundary fixtures, transforms and occlusions; verify same-seed sample/label hashes and terminal capture; audit annotation coverage and empty-label frames. CPU preparation/annotation auditing is allowed; model math and training validation belong on an available GPU.

For the paper, report frame count, unique trajectories/games, pretraining data sources, generation/pretraining GPU and CPU time, hardware and selection costs. Show both downstream training time and total time including pretraining, with any amortization assumptions. Give baseline encoders comparable pretraining opportunity when claiming a pretrained architecture advantage; otherwise label initialization and architecture effects separately. Maintain scratch controls and retain negative transfer results.

Next work is the collector/interface design and label fixtures, followed by explicit authorization for a bounded dataset pilot. No large dataset generation/download or pretraining run is part of the current task.
