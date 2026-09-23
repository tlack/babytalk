# Ideas parking lot

Ideas that aren't in PLAN.md yet. Each entry: the idea as stated, then notes.

## Binary connectivity weights stored as Bloom filters (2026-09-23)

**Idea:** instead of float (or int8) weights, the network is binary: a node is
connected to an input or it isn't. Optimize for *sparse* connectivity, and store
each node's connection set as an integer that is a **Bloom filter** over the
indices of its connected inputs.

**Notes / why it might fit this hardware:**
- Storage per node is one fixed-width filter instead of a row of weights. That
  could shrink SD/flash bytes per expert by a large factor, which matters most for
  expert swap time (PLAN §2, SD → PSRAM wall).
- A node's output is (roughly) a popcount / sum over the inputs whose index tests
  positive in its filter. Membership tests are hash + bit tests: cheap scalar
  integer ops, with no MACs at all.
- **False positives are extra random connections.** They're deterministic given the
  hash, so training can see them (train through the filter, not the ideal set) and
  the net learns around them. Filter size / hash count trades bytes for noise.
- Evaluation direction matters: testing every input against every node's filter
  is O(inputs × nodes) hashes. Precomputing input-index hashes once per layer, then
  AND-ing bit masks, could make it vectorizable with PIE (128-bit AND + popcount).
- Training is the hard part: connectivity is discrete. Candidates: straight-through
  estimators (as in binary NNs), learned-mask / lottery-ticket-style pruning of a
  dense net then binarizing, or evolutionary search per block.
- Could combine with the MoE: experts as Bloom-filter connectivity *deltas* over a
  resident base (fits the LoRA-style-expert idea).

**Prior art to read first:** binary/XNOR nets (BinaryConnect, XNOR-Net), HashedNets
(hash-based weight sharing), lottery-ticket / supermask papers (fixed random
weights + learned binary masks), Bloom-filter embeddings (e.g. Bloom embeddings
for sparse inputs).

**Cheap first test:** on desktop, take one trained layer, keep its top-k
connections per node, encode as Bloom filters at a few sizes, and measure accuracy
vs bytes compared with int8 and int4.
