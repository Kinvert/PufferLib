# What the first search teaches us

October8. This is analysis of the completed12-trial campaign and its checked-in
native search implementation, not a result from the running36-additional-trial
continuation. No campaign input or optimizer history changes here.

## Measured patterns

1. The frozen quality architecture remains a strong starting point. It is one
   C16 7×7/stride4 convolution, flatten, projection64 and common H128/L1 core.
   Its final aggregate0.353913 at631.885s beats all12 searched aggregate scores
   and Nature's0.298506 at664.977s. This supports including that design as a
   search anchor; it does not establish that shallow models always win.
2. Parameter count does not predict training speed. On Connect4, quality-reference
   has160,736 parameters and mean final checkpoint75.076s. Quiet-owl-4 has81,440
   parameters but121.930s. Across all games, its aggregate score0.310761 costs
   1,720.332s. Spatial operations and measured runtime matter more than a
   parameter-only size ranking. Actual stage/kernel/pooling settings are still
   needed before attributing this result to a specific architectural feature.
3. Making a model cheaper can have poor value. Swift-fox-3's aggregate cost564.535s
   is only4.41% below happy-cat-1's590.555s, but its score is0.102099 versus
   0.311465. A frontier endpoint need not be a useful policy.
4. Strong per-game points can be hidden by the six-game average. Quiet-owl-11
   is a promising Maze candidate but has aggregate0.221413, far below the best
   aggregate. Nature still leads final Flappy/Snake means. A universal ranking
   does not replace reporting each game's entire frontier.
5. More decisions do not guarantee better observed evaluation. Breakout's
   quiet-owl-7 goes from score6.806061 at its first checkpoint to5.963636 at
   the final checkpoint. Other Maze/Flappy/Snake curves also decline at some
   stages. These observations could include seed/evaluation variation; they
   do not establish a particular optimization failure. Pong lower-bound changes
   additionally depend on censoring, so do not call them proven forgetting.

## Search-design limitations visible in code

`research/cross_game_feedback.py:aggregate` selects only the fixed final decision
budget in each game, then averages normalized lower scores equally across games
and returns once-per-job final training cost. This is a genuine final-score/cost
search, but the other three checkpoints do not affect PROTEIN's next suggestion.
The displayed full learning curves and the optimized objective are different.
If our desired advantage is early learning, that objective can undervalue it.

`research/feedback_campaign.py` prepares/trains quality and Nature as calibration
controls and paired references. They are not observations in the first search's
native history. Its default proposal is instead C8/kernel8/stride4/projection32.
The existing bridge insists on exact generated-proposal replay, so simply adding
the measured quality reference as an arbitrary history row is invalid. A future
anchored search needs an explicit, tested protocol/default and receipt reuse;
do not inject rows into the active continuation.

The declared shape menu has18 coordinates. From the checked-in ranges it has
2,304 nominal one-stage combinations,622,080 two-stage combinations and
167,961,600 three-stage combinations:168,585,984 total before effective-graph
equivalences. These are grammar combinations, not proven distinct functions.
Forty-eight trials cannot exhaust this menu or identify all feature effects.
PROTEIN can still exploit useful regions, but more trials alone do not fix a
poorly targeted objective or provide a controlled architectural ablation.

For depth1, the ten stage2/3 coordinates are inactive in the actual CNN, while
the native GP still receives the complete18-coordinate proposal vector.
Different proposed vectors can therefore describe the same effective graph.
Happy-cat-1/quiet-owl-6 are a reported example of an effective-graph duplicate;
their scores agree exactly but costs differ. All raw observations remain in the
ledger; native storage separately merges only sufficiently close coordinates.
Assess effective-graph diversity, not trial count alone. Changing representation
or masking inactive coordinates mid-continuation would break replay semantics.

## After the current continuation

Let the authorized continuation finish unchanged. It reuses the12 observations
and should give a more useful picture than the first sparse search, but cannot
promise a winner. Evaluate how many distinct graphs it tried, whether new points
improve the whole frontier and whether added compute yielded useful progress.

Obtain original candidate descriptors and seed/drawing checkpoint rows. Aggregate
exports cannot tell us causally which kernels, pooling or residuals work; network
parameter count alone cannot recover the graph. Keep all losing/duplicate points.

For a subsequent explicitly authorized campaign, first anchor the best known
quality design and the successful fast design, then use a more focused space
around them. Decide beforehand how learning-curve/budget information should
influence native PROTEIN while retaining fixed per-game learners and honest
compute charges. Do not feed several checkpoints at identical coordinates as
independent observations: native success storage replaces sufficiently close
coordinates. A budget-aware design needs a deliberate distinct configuration
coordinate/protocol, or one predeclared curve-derived objective per graph.

Finally freeze promising points and verify them on fresh training/evaluation
seeds. Continuing search on the existing two seeds adds discovery evidence;
it does not provide the independent confirmation needed for the paper.
