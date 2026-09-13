Please look in to this, get me things to read, terminology, where I'd look, if there is a direct answer, what I might try, etc. What project might I try and maximize it in some way, like some PufferLib pixel project where I get CNNs fast or some bullshit? Help me understand what he's directly asking, what he might be asking for with more detail, the info I need for this, the terminology, the paper he might be talking about, what existing potential answers might be, and how I might custom make something that is better for him.

Searched the web, read a page
Let me lay this out properly — what Joseph's actually asking, the vocabulary, the papers, and a concrete project.

What he's actually asking
Two nested questions:

"What's the FLOP-efficient conv encoder for RL from pixels?" — comparing the Nature DQN CNN (Mnih 2015: 3 conv layers, 32-8x8-s4 → 64-4x4-s2 → 64-3x3-s1, then FC) against the IMPALA-style encoder (Espeholt et al. 2018), which people loosely call "the IMPALA architecture" even though it's really just one component of that paper — a deeper conv stack with residual blocks, downsampling between stages. That component now typically gets called "Impala-CNN," and it's a 15-layer ResNet-inspired network that has become the default heavier encoder in image-based deep RL. 
GitHub
"If the CNN is just a feature extractor and the recurrent net does the actual reasoning, how should you split compute between them?" — i.e., for a fixed FLOP budget, what's the optimal allocation between encoder width/depth and recurrent core (LSTM/GRU/etc.) size? This is a compute-allocation question, structurally the same shape as "how do you split a fixed parameter budget between attention and FFN layers in a transformer," just nobody's published a clean answer for the RL encoder/core split the way people have for LLM component ratios.
Terminology to know
Nature CNN — the original DQN encoder, 3 conv layers, small, cheap, still shockingly hard to beat per-FLOP.
IMPALA-CNN / "Impala architecture" — 3 stacked conv+pool+residual-block groups from Espeholt et al. 2018, popularized further as the standard encoder in the Procgen benchmark (Cobbe et al. 2019/2020) and Phasic Policy Gradient.
RSSM (Recurrent State-Space Model) — the DreamerV3 term for the recurrent "core" that sits after the encoder and does the actual temporal reasoning/dynamics modeling. This is the cleanest existing example of literally what Joseph described: a CNN encoder that's kept comparatively small, feeding a recurrent model that gets scaled up.
Intrinsic performance — a term from Hilton et al. (OpenAI, 2023), defined as the minimum compute needed to hit a given return across a model-size family; their proxy for turning "does return improve smoothly" into something you can fit a power law to.
SR-SPR / BBF (Bigger, Better, Faster) — Nikishin/Schwarzer et al. 2023, data-efficient Atari, directly compares scaling width of Nature-CNN-style nets vs. Impala-CNN.
Impoola-CNN — a 2025 modification of Impala-CNN (global average pooling instead of flatten before the FC head) that improves both efficiency and generalization on Procgen.
SimBa / BRO — newer (2024) architectures for continuous control that use LayerNorm + residual MLPs to make scaling not collapse; less directly relevant to pixels but same underlying question (how do you scale DRL nets without the usual degradation).
Primacy bias — the phenomenon (Nikishin et al. 2022) that naive DRL network scaling often hurts unless you address plasticity loss; this is a big reason "just make the CNN bigger" doesn't trivially work and is probably part of why nobody's published a clean encoder/core compute-split law — bigger nets need many of the tricks the newer papers you'd read use, and adding size to the wrong component can just make training worse rather than better. This matters a lot for your project design.
The papers I'd send him, in order
Hilton, Tang, Schulman, "Scaling laws for single-agent reinforcement learning" (2023, arXiv:2301.13442) — the closest thing to a direct, general answer to "what's the right way to think about compute-optimal model size in RL." They introduce "intrinsic performance," a monotonic function of return defined as the minimum compute required to hit a given return across a family of differently-sized models, and find it scales as a power law in both model size and environment interactions, giving you a Chinchilla-style compute-optimal-size law for RL. It doesn't split encoder vs. recurrent core specifically, but it's the right mental framework and methodology to steal. 
arXiv
Espeholt et al., "IMPALA" (2018) — for the actual encoder Joseph's referring to, section on the "ResNet" variant.
Nikishin, Schwarzer, et al., "Bigger, Better, Faster" (BBF), 2023 (arXiv:2305.19452) — directly relevant finding: the smaller Impala-CNN ResNet, measured by both parameter count and FLOPs, outperforms larger/wider variants at all width scales they tried, and naive width-scaling of the encoder without other tricks (resets, etc.) actually collapses performance. This is close to empirical evidence that oversizing the encoder is actively bad, not just inefficient. 
arXiv
Hafner et al., "DreamerV3" (2023, arXiv:2301.04104) — this is the paper that most literally embodies what Joseph is describing: fixed, relatively modest CNN encoder, recurrent state-space model (the "reasoning" component) scaled up, and the paper explicitly demonstrates favorable scaling properties where increasing overall model size gives monotonic improvements in final performance and data-efficiency. Worth reading their appendix on exactly how they scale each component (encoder width, GRU/RSSM size, MLP heads) together as "model size" presets (S/M/L/XL). 
arXiv
Impoola, 2025 (arXiv:2503.05546) — good recent read specifically about Impala-CNN's actual bottleneck (flatten vs. global average pool) and shows you can beat larger models with a small architectural fix rather than more FLOPs — directly useful for "how do I make a better encoder" rather than just "how big should the encoder be."
Tuyls et al., "Scaling laws for imitation learning in single-agent games" (2023/2024, arXiv:2307.09423) — a sibling paper to Hilton et al., useful because they explicitly vary CNN width and separately the "final linear/core" width and can be read as an early sketch of a component-wise scaling law, done on Atari and NetHack.
Gwern's "RL scaling" reading list is a good one-stop directory if you want to go beyond this list — it's a live-updated bibliography. 
Gwern
Is there a direct answer?
No clean one. My honest read after searching this out: nobody has published the RL-equivalent of "attention vs FFN parameter ratio" for encoder-vs-recurrent-core. The closest things are (a) Hilton et al.'s general model-size scaling law, which doesn't split components, and (b) DreamerV3's empirically-tuned presets, which do split components but were tuned by hand/sweep, not derived from a law. The BBF result is suggestive evidence in the direction Joseph's intuition points (don't scale the encoder, it doesn't help and can hurt), but it's not a "here's the ratio" answer. So there's a real gap here — which is good news, because it means a focused experiment could produce something genuinely new and citable rather than just re-deriving a known result.

The PufferLib project
This is very doable and plays exactly to PufferLib's strengths. Concretely:

FLOP-matched encoder/core scaling sweep. Pick one or two PufferLib pixel environments (Atari via PufferLib's wrappers, or Procgen since it's designed for exactly this kind of generalization/capacity study — Procgen is cheap and was the original IMPALA-CNN benchmark). Build a small architecture zoo:

Nature CNN at 3-4 widths
Impala-CNN at 3-4 widths
Impoola-CNN (the pooling variant) at 3-4 widths
Cross each encoder variant with 3-4 recurrent core sizes (LSTM/GRU hidden dim), and train the full grid to convergence (or to a fixed environment-step budget) using PufferLib's fast vectorization — this is the part where PufferLib actually gives you an edge, since the paper you'd write needs a lot of runs and PufferLib is built to make that cheap. For each run log total FLOPs (forward+backward, separately for encoder and core so you can decompose), and plot return (or Hilton-style intrinsic performance) against total FLOPs, and against the encoder-fraction-of-FLOPs. That gives you the actual Pareto frontier and — much more usefully than a single number — the shape of the tradeoff curve, i.e., where it flattens.

A stronger version: don't just grid-search sizes, use PufferLib's own sweep tooling (they already sweep architecture hyperparameters for their topline Atari numbers, and note explicitly that problem complexity determines optimal network size) to search jointly over encoder-width and core-width under a fixed FLOP budget, which turns it into an actual compute-optimal-allocation search rather than a hand grid. That's the thing to send Joseph if you want to "maximize" it — not just "here's a chart" but "here's a sweep procedure that finds the optimal split automatically, and here's what it found." 
X

If you want it to be more than a fun weekend project and closer to something worth writing up: replicate Hilton et al.'s intrinsic-performance methodology but for the two-component (encoder, core) case instead of a single model-size axis. That's the actual missing paper.

What's a well thought out, intelligent, very brief question to ask him about this? Perhaps we have no questions at all right. But he'll be back on his livestream soon and I could ask as long as it's an intelligent question that gets us actual useful information. Again he isn't the expert in this, he is an RL expert he made PufferLib. But to clarify so I can help find him better answers or build this thing out or whatever

Pinpointing whether task memory demands drive the split.
Here's the one I'd ask — it's short, doesn't require him to have the answer already (he said himself he doesn't), but it forces a scoping decision that changes what you'd actually build:

"Do you expect that FLOP split to be roughly one universal ratio across pixel RL tasks, or does it shift a lot with how memory-heavy the task is — i.e. does a mostly-reactive game like Atari want a different encoder/core balance than something with real partial observability, like NMMO?"

Why this one: it's the actual fork in the road for your project. If he thinks it's roughly universal, you build one FLOP-matched sweep on one environment and report a ratio. If he thinks it depends on memory demand, you need at least two environments at opposite ends of the observability spectrum and you're measuring how the optimal split moves, which is a more interesting and more publishable result — and it's also the thing the existing literature hints at but doesn't test: Hilton et al.'s scaling-law paper found that varying task horizon length changes the coefficient of their power law but not the exponent, which is suggestive that the memory-demand of a task changes how much compute you should spend on the part of the network responsible for handling it, without necessarily changing the general shape of the tradeoff. Nobody's checked whether that holds for the specific encoder-vs-core split.

It's also a fair question to put to him specifically because it's not "what's the answer" (which he already told you he doesn't know) — it's "what's the right axis to test," which is a scoping call any RL person with his experience can answer fast, off the cuff, on a stream.

Meh that question is too big

Drafting a short, sharp, easy-to-answer question.
Drafting a short, sharp, easy-to-answer question.
Fair — that one required him to reason live. Here's a tighter one, and it's better anyway because it's grounded in something he's already done rather than something hypothetical:

"When your sweep found the smaller architecture for the 20M-step/s Breakout run — did it shrink the CNN, the LSTM, or both, and by roughly what ratio?"

He already has this answer sitting in a log somewhere — it's not a new thought, just a lookup. And it's directly useful to you: it's a real, empirical FLOP-matched-ish data point (his sweep was optimizing for wallclock/throughput at fixed performance, which is close enough to the tradeoff you care about) for exactly the question you're trying to build a project around. One real number from his own sweep is worth more to you as a starting point than a philosophical question about universality.



Write a message…


Claude is AI and can make mistakes. Please double-check responses.
