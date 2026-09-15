# PongCNN CPU validation — September 14, 2026

`validation.txt` is the output of `bash ocean/pongcnn/tests/run_all.sh`, which compiled both environment variants with ASan/UBSan and compared original/pixel/repeated traces. `traces.sha256` records the full local trace files; these larger traces remain in ignored `build/pongcnn/`. `source.sha256` identifies the implementation, tests and configs at validation/archive time. Snapshot copies of the two environment configs and common comparison recipe are retained here.

Passed: 49,152 transitions across eight seeds, three frame skips and both action modes; all game-state hashes, rewards and terminal flags match original Pong, with exact repeat traces. Pixel fixtures and boundary guards passed. See [implementation documentation](../../../../ocean/pongcnn/README.md) for scope and observation differences.

Five-model GPU canary configs were prepared under `build/pongcnn/canary.NH9AYFJK` without compiling or executing a trainer. A separate configuration readback check confirmed that all five default+environment config combinations were identical, that environment settings matched `config/pong.ini`, and that Flex keys, 65,536-step budget and hidden-128 core were present. No GPU, learning, SPS or wall-clock performance claim follows from this CPU validation. Native canary execution remains pending after Connect4 confirmation.
