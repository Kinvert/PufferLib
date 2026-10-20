# Existing pretrained backbones: practical shortlist

Checked October20,2026 against primary model/code sources. These are available
weights/research candidates, not measured PufferLib speed or quality results.
No model becomes compatible with encoder8by matching its output length.

|Priority|Candidate|What is actually available|Fit and missing work|
|---|---|---|---|
|1|[TAESD](https://github.com/madebyollin/taesd),[model weights](https://huggingface.co/madebyollin/taesd)|Released tiny autoencoder; official encoder1,222,532params; RGB[0,1]→H/8×W/8×4latent; MIT model card|Closest ready autoencoder. Conv/ReLU/residual encoder needs its own native implementation or offline teacher path. Its encoder is about33times our36,776parameter student; “tiny” relative to Stable Diffusion is not proof of cheap RL. Test tiny-object/position preservation.|
|2|[MobileOne-S0](https://github.com/apple-aiml-research/ml-mobileone)|Official fused and unfused ImageNet checkpoints; fused inference removes training branches|Useful efficient recognition baseline; depthwise operations and folded normalization need a qualified native path. Published mobile-device latency is not5090latency. Check the linked checkpoint's licensing before redistribution.|
|3|[MobileNetV3-Small-050](https://huggingface.co/timm/mobilenetv3_small_050.lamb_in1k)|Released timm ImageNet weights,about1.6Mparameters including classifier,6.42MB safetensors,Apache2.0|The half-width model is a better first small recognition candidate than the2.54M full-width model. Remove classifier and retain spatial features. Depthwise/SE/hard-swish need implementation/conversion. Default224center crop drops game edges; declare full-frame preprocessing instead and test resulting domain shift.|
|4|[ATC](https://arxiv.org/abs/2009.08319),[official training/evaluation code](https://github.com/astooke/rlpyt/tree/master/rlpyt/ul)|Game-oriented unsupervised visual representation code|More relevant training domain, but no directly usable released weight URL verified in this pass. Code availability is not a found pretrained checkpoint.|
|5|[DC-AE](https://github.com/mit-han-lab/efficientvit/blob/master/applications/dc_ae/README.md)|Released ImageNet/mixed-data autoencoders with spatial factors32/64/128|Worth an offline teacher/reference. Large intermediate channels and additional operators make it a poor first assumption for the cheapest online encoder. Compression factor alone says nothing about encoder cost.|

TAESD acquisition is implemented without Torch or code execution:

```bash
.venv/bin/python research/backbone_assets.py list
.venv/bin/python research/backbone_assets.py fetch-taesd --out build/backbones/taesd-614f768
.venv/bin/python research/backbone_assets.py fetch-mobilenet --out build/backbones/mobilenetv3-small-050-5188129
```

Pinned Hugging Face revision:`614f76814bbe30edbe2e627ace1c2234c81a2c0e`.
Safetensors file9,793,292bytes; LFS SHA256:
`db169d69145ec4ff064e49d99c95fa05d3eb04ee453de35824a6d0f325513549`.
The tool downloads only weights/config/model card, validates the exact checkpoint
hash and every tensor span, and inventories parameters. It installs nothing,
unpickles nothing and executes no model. Inspection is not inference qualification.

The MobileNet checkpoint is also pinned,revision
`5188129022623c1fd7e36401459eda145a81d7e9`,6,420,754bytes,SHA256
`2e3f6937afd4b3704450518a2710168775d6c70ebfb7a0e9aaf06200c6fbe0c4`.
The same tool validates its checkpoint and records scalar normalization buffers
as stored elements,not trainable parameters. Both acquisition commands have
been executed locally; receipts live beside the weights under ignored build.

First comparison:our reconstruction student versus frozen TAESD features at an
explicit feature budget, with full-frame position/detail probes and complete
encoder latency. Then downstream scratch/fine-tuned/frozen controls. An offline
teacher can distill useful features into the much smaller native student, but
teacher data/compute must be counted. No external checkpoint was trained for
our games, and no transfer benefit is established by finding its weights.
