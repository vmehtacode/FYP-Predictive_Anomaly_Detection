# Domain Pitfalls: Grid Guardian Energy Anomaly Detection

**Domain:** GNN-based anomaly detection with self-play on power grid topology
**Project:** Grid Guardian - SSEN network (~100K nodes)
**Researched:** 2026-01-26
**Overall Confidence:** MEDIUM-HIGH

Research focused on five critical risk areas identified for this project:
1. GNN training on medium-scale graphs (SSEN ~100K nodes)
2. Self-play mode collapse in adversarial anomaly generation
3. Physics constraint integration with neural architectures
4. Evaluation without ground-truth anomaly labels
5. Real-time API reliability (Elexon BMRS) and fallback strategies

---

## Critical Pitfalls

Mistakes that cause rewrites, project failure, or invalidate research contributions.

### Pitfall 1: GNN Oversmoothing in Deep Architectures

**What goes wrong:**
Node representations converge to indistinguishable vectors as GNN layers increase, making it impossible to differentiate between nodes in different parts of the grid. For a 100K node power grid, this means losing the precise localization needed for anomaly detection — you can't distinguish whether an anomaly is at a primary substation or a downstream LV feeder.

**Why it happens:**
Message passing in GNNs causes node embeddings to become increasingly similar with each layer. As layers increase, embedding features learned from GNNs quickly become similar or indistinguishable, making them incapable of differentiating network proximity. Adding many graph convolutional layers causes oversmoothing where the model produces similar embeddings for all nodes.

**Consequences:**
- Loss of spatial resolution for anomaly localization
- Inability to distinguish anomalies at different grid hierarchy levels (LV feeder vs. substation)
- Deep networks (>10 layers) become unusable despite needing depth for multi-hop reasoning
- Grid topology information is effectively lost despite using a GNN

**Prevention:**
1. **Use residual connections** - Residual connections provably prevent oversmoothing in GNNs, though normalization methods may ruin the beneficial denoising effect
2. **Implement gated updates** - Gated updates are very effective at facilitating deep GNN architectures (>10 layers) and preventing oversmoothing
3. **Apply adaptive message passing (AMP)** - AMP is a probabilistic framework that can endow most message passing architectures with the ability to learn how many messages to exchange between nodes and which messages to filter out
4. **Try non-local message passing** - Non-local message passing based on Post-LN induces algebraic smoothing, preventing oversmoothing without curse of depth, supporting deeper networks up to 256 layers
5. **Consider DropEdge regularization** - DropEdge alleviates both overfitting and oversmoothing issues by randomly dropping edges during training
6. **Limit network depth initially** - Start with 2-4 layers for SSEN's 100K node graph; only add depth if needed for multi-hop reasoning

**Detection:**
- Monitor node embedding similarity across layers using cosine similarity metrics
- Check if validation performance degrades as layers are added
- Visualize t-SNE of node embeddings per layer — if they collapse to a single cluster, oversmoothing has occurred
- Measure rank of node feature matrices — rapid rank collapse indicates oversmoothing

**Grid Guardian specific:**
For SSEN's 3-tier topology (LV feeders → secondary substations → primary substations), start with 3 layers max. Use residual connections + DropEdge from the start rather than adding them later.

**Confidence:** HIGH
**Sources:**
- [Adaptive Message Passing for Oversmoothing Mitigation](https://medium.com/@federicoerrica/adaptive-message-passing-learning-to-mitigate-oversmoothing-oversquashing-and-underreaching-b5cf191fd5cf)
- [Solving Oversmoothing via Nonlocal Message Passing](https://arxiv.org/html/2512.08475)
- [DropEdge for Deep Graph Networks](https://openreview.net/forum?id=Hkx1qkrKPr)

---

### Pitfall 2: Self-Play Mode Collapse in Anomaly Generation

**What goes wrong:**
The Proposer (anomaly generator) collapses to generating the same type of anomaly repeatedly, or the Verifier overfits to specific patterns and fails to generalize. This invalidates the core self-play methodology — you end up with a system that only detects one narrow anomaly type (e.g., voltage spikes) and misses everything else (e.g., topology changes, gradual drift).

**Why it happens:**
Single-rollout variants of reinforcement learning in multimodal contexts suffer from severe instability and often lead to training collapse. GANs remain vulnerable to issues like mode collapse and training instability. If the generator overfits the discriminator's feedback and finds local optima, it will repeatedly generate similar samples, causing mode collapse.

In the Grid Guardian context: The Proposer finds one "easy win" anomaly pattern that consistently fools the Verifier (e.g., always generating EV_SPIKE scenarios). The Verifier then overfits to detecting just that pattern. Self-play feedback loop reinforces this narrow behavior.

**Consequences:**
- Proposer generates only 1-2 anomaly types out of the intended 5+ scenarios
- Verifier achieves high accuracy on generated anomalies but fails on real-world distribution shifts
- Self-play training appears to converge (low loss) but has learned nothing useful
- Research contribution is invalidated — the system doesn't demonstrate generalization
- Comparison with baselines becomes unfair (your system is tested on its own generated data)

**Prevention:**
1. **Entropy-based advantage shaping** - MSSR achieves stable optimization via an entropy-based advantage-shaping mechanism that adaptively regularizes advantage magnitudes, preventing collapse
2. **Progress reward biasing** - Bias rewards to reinforce progress toward generating DIVERSE anomaly types, not just successful ones
3. **Multi-generator architecture** - MO-GAAL introduces multi-generator architecture to mitigate risk of mode collapse (though still faces challenges with generator collaboration)
4. **Dynamic reward scaling** - Use adaptive reward mechanism that balances exploration (new anomaly types) and exploitation (effective anomalies) by dynamically scaling reconstruction error and classification rewards
5. **Explicit diversity metrics** - Add diversity bonus to Proposer reward: measure coverage across anomaly types (EV_SPIKE, COLD_SNAP, PEAK_SHIFT, OUTAGE, MISSING_DATA)
6. **Curriculum learning** - Start with simple anomalies, gradually increase complexity and diversity requirements
7. **Regularize with L2 penalties** - Policy collapse can be mitigated by aggressive L2 regularization

**Detection:**
- Track distribution of generated anomaly types — if entropy drops below threshold, collapse is occurring
- Monitor Verifier confusion matrix across anomaly categories — if it only detects one type, Proposer has collapsed
- Compare Proposer diversity metrics over training epochs — should remain high
- Test on held-out real anomaly scenarios (physics violations) — performance should transfer

**Grid Guardian specific:**
Implement explicit tracking of 5 scenario types (EV_SPIKE, COLD_SNAP, PEAK_SHIFT, OUTAGE, MISSING_DATA). Add diversity reward term: `reward += lambda * entropy(scenario_type_distribution)`. Set minimum threshold: each scenario type must appear in at least 15% of generated batches.

**Confidence:** HIGH
**Sources:**
- [Overcoming Policy Collapse in Deep RL](https://openreview.net/forum?id=m9Jfdz4ymO)
- [Mode Collapse in GAN-based Anomaly Detection](https://www.nature.com/articles/s41598-024-84863-6)
- [Self-Play RL Survey](https://arxiv.org/html/2408.01072v1)
- [Dynamic Reward Scaling for RL Anomaly Detection](https://arxiv.org/html/2508.18474)

---

### Pitfall 3: Physics Constraint Integration Brittleness

**What goes wrong:**
Physics constraints are either (a) too rigid, causing training to fail or producing trivial solutions, or (b) too loose, providing no actual constraint and losing the explainability advantage. The loss function becomes impossible to balance between data fit, physics constraints, and anomaly detection objectives.

**Why it happens:**
Training proceeds iteratively until convergence, yielding a model that fits data while satisfying physics constraints, but loss reduction slows down when trying to satisfy stricter physics constraints. Critical challenges include computational complexity and integration of complex physical laws. Networks do not consider physical characteristics underlying the problem without careful specification of geometry, initial conditions, and boundary conditions.

In Grid Guardian: You have hard constraints (feeder capacity, voltage bounds, ramp rates from SSEN) that must NEVER be violated. But if you enforce them too strictly in the loss function, gradient descent can't find solutions. If you enforce them too loosely, the GNN learns to ignore them.

**Consequences:**
- Training divergence or collapse (NaN losses) due to conflicting objectives
- Physics constraints are satisfied on training data but violated on test data
- Model learns to "cheat" physics constraints (e.g., predicting exactly at the boundary)
- Loss of interpretability — can't explain why model flagged an anomaly in physics terms
- Failed validation against domain experts who notice physics violations

**Prevention:**
1. **Soft constraints with adaptive weighting** - Use advanced strategies including adaptive weighting that enhance optimization stability
2. **Staged training** - Train in phases: (1) data fitting only, (2) soft physics constraints, (3) hard physics constraints
3. **Physics-guided loss design** - Composite loss with separate terms: L_total = L_data + λ_physics * L_constraints + λ_anomaly * L_detection, with adaptive λ scheduling
4. **Projection layers** - Add explicit projection step after GNN output that enforces hard constraints (e.g., clip voltages to valid range)
5. **Use energy-based formulations** - Energy-based and variational formulations ensure physical consistency across multiscale problems
6. **Domain-specific expertise** - Addressing challenges requires algorithmic improvements, model regularization, and domain-specific expertise
7. **Start simple** - Begin with 2-3 most critical constraints, validate thoroughly, then add complexity

**Detection:**
- Monitor separate loss components (data, physics, detection) — if one dominates, balancing has failed
- Validate physics constraint violations on test set — should be near-zero
- Compare physics violation rates: GNN vs. baseline vs. ground truth
- Run domain expert review on flagged anomalies — check if they make physical sense
- Check if model predicts exactly at constraint boundaries (suspicious, likely learned to game the constraint)

**Grid Guardian specific:**
SSEN constraints identified in EDA: feeder capacity limits, voltage bounds (±6% of 230V), ramp rate limits. Implement as:
1. Soft constraints (L2 penalty) during initial training
2. Hard constraints (projection layer) for final model
3. Separate physics validation metrics reported alongside anomaly detection metrics

**Confidence:** MEDIUM
**Sources:**
- [Physics-Informed Neural Networks in Grid-Connected Inverters Review](https://www.mdpi.com/1996-1073/18/20/5441)
- [Physics-Informed GNN for Dynamic Reconfiguration](https://arxiv.org/html/2310.00728v2)
- [Understanding Physics-Informed Neural Networks](https://www.mdpi.com/2673-2688/5/3/74)
- [Review of PINNs: Challenges in Loss Function Design](https://www.mdpi.com/2227-7390/13/20/3289)

---

### Pitfall 4: Evaluation Validity Without Ground-Truth Labels

**What goes wrong:**
You report impressive anomaly detection metrics (high AUC, precision, recall), but they're measured on synthetic/self-generated anomalies that don't represent real distribution shifts. When tested on actual grid anomalies (rare events, new failure modes), performance collapses. Or worse: you have no way to validate the system at all, making the research contribution unverifiable.

**Why it happens:**
Without a target, traditional data science metrics cannot be calculated to estimate model performance. Unsupervised techniques tend to have a higher false positive rate compared to supervised methods due to lack of ground truth, which complicates validation of results. In unsupervised settings, common practices for testing accuracy such as using a labeled validation set cannot be applied.

In Grid Guardian: SSEN data has no labeled anomalies. You're generating synthetic anomalies (EV_SPIKE, etc.) via the Proposer. If you only evaluate on these synthetic anomalies, you're measuring how well the Verifier detects what the Proposer generated — not real-world performance.

**Consequences:**
- Overly optimistic metrics that don't reflect real-world performance
- Inability to compare with baselines fairly (they're also evaluated on synthetic data)
- Research contribution cannot be validated by external reviewers
- Deployment in real grid would have unknown reliability
- FYP examiners question whether the system actually works

**Prevention:**
1. **Use physics-based pseudo-labels** - Validate against hard physics constraint violations (these are definite anomalies even without labels)
2. **Implement ensemble-based validation** - Use representative majority's opinion as approximator for ground truth via Accurately-Diverse ensemble
3. **Apply intrinsic evaluation metrics** - Use Anomaly Separation Index (ASI) and Anomaly Separation and Overlap Index (ASOI), designed for when ground truth labels are absent
4. **Synthetic AUC method** - Generate two synthetic datasets (one made more normal, one made more anomalous), calculate ROC AUC using artificial labels
5. **Use Excess-Mass (EM) and Mass-Volume (MV) curves** - Criteria that discriminate accurately between algorithms without requiring labels
6. **Statistical tests** - Kolmogorov-Smirnov test to verify detected anomalies come from different distribution than normal data
7. **Multi-level validation strategy**:
   - Level 1: Physics violations (hard constraints) — 100% must be detected
   - Level 2: Synthetic scenarios (Proposer-generated) — measure relative performance
   - Level 3: Baseline comparison (IsolationForest, DecompositionAnomalyDetector on same data)
   - Level 4: Domain expert review (qualitative validation of flagged real events)

**Detection:**
- Compare performance on physics violations vs. synthetic anomalies — gap indicates evaluation brittleness
- Check baseline performance — if your system doesn't beat IsolationForest on physics violations, something is wrong
- Monitor false positive rate on validation data (assumed normal) — should be stable over time
- Qualitative review of top-K flagged anomalies — do they correspond to known grid events?

**Grid Guardian specific:**
Validation hierarchy:
1. **Hard physics violations** (ground truth): Voltage outside ±6%, capacity exceeded, impossible ramp rates
2. **Synthetic scenarios** (self-play generated): EV_SPIKE, COLD_SNAP, etc.
3. **Baseline comparison**: IsolationForest, DecompositionAnomalyDetector, PatchTST on same splits
4. **Domain plausibility**: Do flagged anomalies correspond to known grid stress periods (e.g., winter peaks)?

Report all four levels. Never report only synthetic scenario performance.

**Confidence:** HIGH
**Sources:**
- [Towards Unsupervised Validation of Anomaly-Detection Models](https://arxiv.org/html/2410.14579v1)
- [ASOI: Anomaly Separation and Overlap Index](https://link.springer.com/article/10.1007/s40747-025-02204-0)
- [How to Evaluate Unsupervised Anomaly Detection](https://arxiv.org/pdf/1607.01152)
- [Unsupervised Anomaly Detection with Data-Centric ML (Google Research)](https://research.google/blog/unsupervised-and-semi-supervised-anomaly-detection-with-data-centric-ml/)

---

### Pitfall 5: Elexon BMRS API Reliability and Fallback Fragility

**What goes wrong:**
Your FYP demonstration depends on live Elexon BMRS API calls. During the presentation/viva, the API is down, rate-limited, or returns stale data. The demo fails. Or in development, you build tight coupling to the API without fallback, causing flaky tests and unreproducible experiments.

**Why it happens:**
Applications relying on online inference are subject to timeouts, infrastructure outages, and failures in external dependencies such as third-party data providers. Inference engine failures are the most prevalent root cause (~60% of incidents), with timeouts and resource exhaustion accounting for the majority. Fallback or fall-over strategies are needed to keep operations running, even in the event of unexpected failures in production ML systems.

In Grid Guardian: Elexon BMRS provides real-time UK grid data. It's free (no API key) but has no SLA guarantees. Network issues, maintenance windows, or rate limits can cause failures.

**Consequences:**
- FYP demo fails during presentation (catastrophic for grading)
- Experiments become unreproducible if API responses change
- Tests are flaky, failing intermittently
- Development is blocked when API is down
- Cannot demonstrate "live data capability" requirement from PROJECT.md

**Prevention:**
1. **Implement circuit breaker pattern** - Circuit breaker temporarily blocks access to faulty service after detecting failures, operating in Closed/Open/Half-Open states
2. **Multi-tier fallback hierarchy**:
   - **Tier 1**: Live Elexon BMRS API call
   - **Tier 2**: Cached recent responses (Redis/local disk, 5-minute TTL)
   - **Tier 3**: Synthetic data generator matching Elexon schema
   - **Tier 4**: Graceful degradation (demo continues with note "using cached data")
3. **Intelligent caching** - Cache hit rates of 20-40% significantly reduce computational costs and improve response times
4. **Timeout configuration** - Set aggressive timeouts (e.g., 5 seconds) to fail fast and fallback quickly
5. **Circuit breaker thresholds** - If 40% or more API calls fail within 60-second window, bypass retries and route to fallback
6. **Pre-demo warm-up** - 5 minutes before presentation, run full demo end-to-end and populate cache
7. **Offline demo mode** - Environment variable `DEMO_MODE=offline` uses pre-recorded API responses

**Detection:**
- Monitor API success/failure rates in development — if <95% success, fallback is needed
- Track cache hit rates — if <20%, caching isn't helping
- Measure p95/p99 latency — if >5s, need better timeouts
- Run demo 10 times in a row — all 10 should succeed (with fallback if needed)

**Grid Guardian specific:**
Implement fallback for Elexon BMRS API:

```python
class ElexonClient:
    def __init__(self):
        self.circuit_breaker = CircuitBreaker(failure_threshold=0.4, timeout=60)
        self.cache = ResponseCache(ttl=300)  # 5 min
        self.synthetic_fallback = SyntheticElexonGenerator()

    def get_grid_data(self, **params):
        # Tier 1: Live API
        if self.circuit_breaker.is_closed():
            try:
                response = self._api_call(timeout=5, **params)
                self.cache.set(params, response)
                return response
            except (Timeout, ConnectionError) as e:
                self.circuit_breaker.record_failure()

        # Tier 2: Cache
        if cached := self.cache.get(params):
            return cached

        # Tier 3: Synthetic fallback
        return self.synthetic_fallback.generate(**params)
```

**For FYP presentation:** Run `make demo-warmup` 5 minutes before presenting to populate cache. Set `DEMO_MODE=offline` as backup.

**Confidence:** HIGH
**Sources:**
- [Hierarchical Fallback Architecture for ML Inference](https://arxiv.org/html/2501.17834v1)
- [Circuit Breaker Pattern (Azure)](https://learn.microsoft.com/en-us/azure/architecture/patterns/circuit-breaker)
- [Building Resilient Systems with Circuit Breakers (2026)](https://dasroot.net/posts/2026/01/building-resilient-systems-circuit-breakers-retry-patterns/)
- [Enhancing Reliability in AI Inference Services](https://arxiv.org/html/2511.07424)

---

## Moderate Pitfalls

Mistakes that cause delays, technical debt, or degraded performance.

### Pitfall 6: GNN Overfitting on Small Training Sets

**What goes wrong:**
GNNs overfit when training data is limited, especially with sampling-based methods. For SSEN's 100K nodes, if you only have labeled data for a small subset, the GNN memorizes the training graph structure rather than learning generalizable patterns.

**Why it happens:**
When the training set is small, most GNN methods face an overfitting issue, particularly exacerbated for sampling-based GNN methods. GNNs tend to overfit the given graph structure, using it even when a better solution can be obtained by ignoring it. Overfitting weakens generalization ability on small datasets.

**Prevention:**
1. **DropEdge** - Randomly drop edges during training to prevent overfitting to exact graph structure
2. **NodeNorm** - Node Normalization regularizes deep GCNs by discouraging feature-wise correlation
3. **Random path graphs** - Train on sequences of random spanning trees to counter overfitting when training set is small
4. **Graph augmentation** - Create multiple views of graph via edge perturbation, node masking
5. **Early stopping** - Monitor validation loss carefully; stop before overfitting occurs

**Grid Guardian specific:**
Use DropEdge with p=0.1-0.2 for SSEN topology. Generate augmented graphs by randomly perturbing non-critical edges (±10% of edges).

**Confidence:** MEDIUM
**Sources:**
- [Fast and Effective GNN Training through Random Path Graphs](https://arxiv.org/html/2306.04828)
- [Graph Neural Networks Use Graphs When They Shouldn't](https://arxiv.org/html/2309.04332v2)

---

### Pitfall 7: Temporal Dynamics Lost in Static GNN

**What goes wrong:**
Power grids have strong temporal dependencies (grid inertia, sequential operations, renewable variability), but static GNN architectures treat each timestep independently, losing temporal context needed for anomaly detection.

**Why it happens:**
Fluctuations in grid variables in one zone are often related to changes in other zones (spatial correlation), while temporal correlations emerge due to grid inertia and sequential nature of grid operations. Conventional forecasting methods often neglect power grid's inherent topology, limiting ability to capture complex spatio-temporal dependencies.

**Prevention:**
1. **Use Temporal GNN architectures** - Combine GNN with LSTM/Transformer for temporal modeling
2. **Graph windowing** - Feed sequence of graph snapshots to GNN, not just current state
3. **Temporal edge features** - Include rate-of-change, temporal deltas as edge/node features
4. **Recurrent GNN** - Use graph recurrent networks (e.g., GConvLSTM, DCRNN)

**Grid Guardian specific:**
Already have time-series windowing in existing pipeline. Extend to temporal GNN by stacking T sequential graph snapshots as input.

**Confidence:** MEDIUM
**Sources:**
- [Graph Neural Networks for Power Grid Operational Risk Assessment](https://arxiv.org/html/2405.07343v1)
- [PowerGNN: Topology-Aware GNN for Electricity Grids](https://arxiv.org/html/2503.22721v1)

---

### Pitfall 8: Class Imbalance (Normal >> Anomaly)

**What goes wrong:**
Energy data is overwhelmingly normal (99%+ of timesteps). GNN learns to classify everything as normal, achieving high accuracy but detecting zero anomalies.

**Why it happens:**
Traditional graph neural network architectures tend to rely excessively on samples from the majority class in training, leading to weaker recognition performance for minority classes. Anomaly detection faces challenges including data imbalance and high variance.

**Prevention:**
1. **Balanced sampling** - Oversample anomaly examples, undersample normal examples in training batches
2. **Focal loss** - Weight loss toward hard-to-classify examples (anomalies)
3. **Anomaly synthesis** - Use Proposer to generate synthetic anomalies, balancing dataset
4. **Ensemble methods** - Combine GNN with anomaly-focused baselines

**Grid Guardian specific:**
Self-play Proposer naturally generates synthetic anomalies, addressing imbalance. Ensure 30-50% anomaly ratio in training batches.

**Confidence:** MEDIUM
**Sources:**
- [Node Classification of Imbalanced Data Using Ensemble GNNs](https://www.mdpi.com/2076-3417/15/19/10440)

---

### Pitfall 9: SCADA Data Quality and Sensor Noise

**What goes wrong:**
Power grid sensor data (SCADA) contains noise, missing values, and anomalous measurements that degrade model training. The GNN learns to fit noise rather than true grid state.

**Why it happens:**
Quality of state estimation is highly sensitive to accuracy of switch status and measurement data, with anomalous measurements being primary causes of failures. State estimation is highly sensitive to anomalies (bad data), increasing computation times and jeopardizing real-time performance. Measurement data depend on ever-changing grid context, making statistical modeling extremely difficult.

**Prevention:**
1. **Robust preprocessing** - Use median filtering, outlier clipping before GNN training
2. **Physics-informed imputation** - Fill missing data using power flow equations, not just statistical imputation
3. **Separate data anomalies from event anomalies** - Distinguish sensor errors from true grid events
4. **Uncertainty quantification** - Model confidence in predictions; flag low-confidence detections

**Grid Guardian specific:**
SSEN data preprocessing: (1) clip voltage to ±20% of nominal, (2) interpolate missing <5-minute gaps, (3) flag and exclude >1hr outages.

**Confidence:** MEDIUM
**Sources:**
- [Anomaly Detection in Smart Grid Using SCADA System](https://www.sciencedirect.com/science/article/abs/pii/S0378779624007624)
- [Enhancing Resilience via Real-Time Anomaly Detection](https://energyinformatics.springeropen.com/articles/10.1186/s42162-024-00401-8)

---

### Pitfall 10: Reward Hacking in Self-Play

**What goes wrong:**
The Proposer learns to exploit flaws in the Verifier's reward function rather than generating meaningful anomalies. E.g., it generates anomalies that are technically "adversarial" (fool Verifier) but physically meaningless.

**Why it happens:**
Reward hacking occurs when the agent learns to exploit the reward function rather than achieve the intended objective. Implementing adversarial training encounters difficulties such as formulating an effective objective function without access to labels.

**Prevention:**
1. **Multi-objective rewards** - Combine detection difficulty + physical plausibility + diversity
2. **Human-in-the-loop validation** - Periodically review Proposer outputs for sanity
3. **Physics constraint layer** - Reject Proposer outputs that violate hard constraints
4. **Reward shaping with LLM** - Use potential functions for semantic reward guidance

**Grid Guardian specific:**
Proposer reward = α * (Verifier_loss) + β * (Physics_plausibility) + γ * (Diversity_bonus). Ensure β > 0 prevents unphysical anomalies.

**Confidence:** MEDIUM
**Sources:**
- [Reward Hacking in Reinforcement Learning](https://lilianweng.github.io/posts/2024-11-28-reward-hacking/)
- [LLM-Enhanced RL for Anomaly Detection](https://arxiv.org/html/2601.02511v1)

---

## Minor Pitfalls

Mistakes that cause annoyance but are fixable.

### Pitfall 11: Inefficient GNN Training on 100K Node Graph

**What goes wrong:**
Full-graph GNN training on SSEN's 100K nodes is slow or runs out of memory, making experimentation impractical.

**Prevention:**
- Use mini-batch sampling (GraphSAGE, ClusterGCN)
- Implement neighbor sampling to reduce memory footprint
- Start with subgraph experiments before scaling to full SSEN topology

**Confidence:** LOW
**Sources:**
- [Distributed GNN Training Survey](https://dl.acm.org/doi/10.1145/3648358)

---

### Pitfall 12: Ignoring Grid Hierarchy in GNN Architecture

**What goes wrong:**
SSEN topology has 3 tiers (LV feeder → secondary → primary), but flat GNN treats all nodes equally, losing hierarchical structure.

**Prevention:**
- Use hierarchical pooling (e.g., DiffPool) to respect grid levels
- Add node type features (is_lv_feeder, is_secondary_substation, is_primary_substation)
- Separate message passing per hierarchy level

**Confidence:** LOW
**Sources:**
- [PowerGNN: Topology-Aware GNN for Electricity Grids](https://arxiv.org/html/2503.22721v1)

---

### Pitfall 13: Lack of Benchmark Comparison

**What goes wrong:**
Reporting performance metrics without comparing to published baselines makes it unclear if results are good or just mediocre.

**Prevention:**
- Compare against existing power grid anomaly detection papers
- Use public benchmarks (PowerGraph dataset) for validation
- Report performance relative to simple baselines (IsolationForest already implemented)

**Confidence:** LOW
**Sources:**
- [PowerGraph: A Power Grid Benchmark Dataset for GNNs](https://neurips.cc/virtual/2024/poster/97496)

---

## Phase-Specific Warnings

Recommendations for when specific pitfalls are most relevant during project execution.

| Phase | Likely Pitfall | Mitigation | Priority |
|-------|---------------|------------|----------|
| **GNN Architecture Design** | Pitfall 1: Oversmoothing | Start with 2-4 layers max; add residual connections immediately | CRITICAL |
| **GNN Architecture Design** | Pitfall 12: Ignoring hierarchy | Add node type features from day 1 | MODERATE |
| **Self-Play Training** | Pitfall 2: Mode collapse | Implement diversity tracking from start; don't wait for collapse | CRITICAL |
| **Self-Play Training** | Pitfall 10: Reward hacking | Multi-objective reward with physics plausibility component | MODERATE |
| **Physics Integration** | Pitfall 3: Constraint brittleness | Staged training: soft constraints first, then hard | CRITICAL |
| **Physics Integration** | Pitfall 9: SCADA data quality | Robust preprocessing pipeline before training | MODERATE |
| **Evaluation Design** | Pitfall 4: Invalid metrics | Multi-level validation (physics + synthetic + baseline + qualitative) | CRITICAL |
| **Evaluation Design** | Pitfall 13: No benchmark comparison | Identify comparison papers early; align evaluation metrics | MODERATE |
| **Elexon Integration** | Pitfall 5: API reliability | Implement circuit breaker + cache before demo dependency | CRITICAL |
| **Model Training** | Pitfall 6: GNN overfitting | DropEdge + early stopping | MODERATE |
| **Model Training** | Pitfall 7: Lost temporal dynamics | Use temporal GNN architecture or graph windowing | MODERATE |
| **Model Training** | Pitfall 8: Class imbalance | Balanced sampling via Proposer; 30-50% anomaly ratio | MODERATE |
| **Deployment/Demo** | Pitfall 11: Slow GNN training | Mini-batch sampling for 100K nodes | LOW |

---

## Summary: Top 3 Project-Killing Risks

Based on this research, the three pitfalls most likely to invalidate the Grid Guardian research contribution:

### 1. Evaluation Validity (Pitfall 4)
**Risk:** Reporting metrics only on self-generated synthetic anomalies, making results unverifiable.
**Mitigation:** Implement 4-level validation (physics violations, synthetic scenarios, baseline comparison, domain plausibility) from the start. Never report synthetic-only metrics.

### 2. Self-Play Mode Collapse (Pitfall 2)
**Risk:** Proposer-Verifier collapse to narrow behavior, invalidating self-play methodology.
**Mitigation:** Diversity tracking + entropy-based rewards + explicit coverage of 5 scenario types with minimum thresholds.

### 3. Physics Constraint Brittleness (Pitfall 3)
**Risk:** Loss function becomes untrainable or physics constraints are ignored, losing interpretability.
**Mitigation:** Staged training (soft → hard constraints) + separate loss component monitoring + domain expert validation.

---

## Confidence Assessment

| Risk Area | Research Depth | Source Quality | Confidence |
|-----------|----------------|----------------|------------|
| GNN Oversmoothing | Deep (8+ sources, recent) | HIGH (arXiv, peer-reviewed) | HIGH |
| Self-Play Mode Collapse | Deep (7+ sources, 2025-2026) | HIGH (OpenReview, ICLR, NeurIPS) | HIGH |
| Physics Integration | Moderate (5 sources, domain-specific) | MEDIUM (recent but emerging field) | MEDIUM |
| Evaluation Without Labels | Deep (6+ sources, established methods) | HIGH (Google Research, arXiv) | HIGH |
| API Reliability | Deep (5+ sources, production ML) | HIGH (Azure, industry patterns) | HIGH |

**Overall Research Confidence:** MEDIUM-HIGH

Strong confidence in GNN, self-play, and evaluation pitfalls (well-researched areas). Moderate confidence in physics constraint integration (emerging area with fewer established patterns). All findings verified through multiple sources; critical claims cross-referenced.

---

## Sources

### GNN Training and Oversmoothing
- [Adaptive Message Passing: Learning to Mitigate Oversmoothing, Oversquashing, and Underreaching](https://medium.com/@federicoerrica/adaptive-message-passing-learning-to-mitigate-oversmoothing-oversquashing-and-underreaching-b5cf191fd5cf)
- [Solving Oversmoothing in GNNs via Nonlocal Message Passing](https://arxiv.org/html/2512.08475)
- [Over smoothing issue in graph neural network | Towards Data Science](https://towardsdatascience.com/over-smoothing-issue-in-graph-neural-network-bddc8fbc2472/)
- [DropEdge: Towards Deep Graph Convolutional Networks on Node Classification](https://openreview.net/forum?id=Hkx1qkrKPr)
- [Fast and Effective GNN Training through Sequences of Random Path Graphs](https://arxiv.org/html/2306.04828)
- [Graph Neural Networks Use Graphs When They Shouldn't](https://arxiv.org/html/2309.04332v2)
- [Distributed Graph Neural Network Training: A Survey](https://dl.acm.org/doi/10.1145/3648358)

### Self-Play and Mode Collapse
- [Overcoming Policy Collapse in Deep Reinforcement Learning](https://openreview.net/forum?id=m9Jfdz4ymO)
- [A Survey on Self-play Methods in Reinforcement Learning](https://arxiv.org/html/2408.01072v1)
- [Generative adversarial synthetic neighbors-based unsupervised anomaly detection](https://www.nature.com/articles/s41598-024-84863-6)
- [Robust anomaly detection via adversarial counterfactual generation](https://link.springer.com/article/10.1007/s10115-024-02172-w)
- [Dynamic Reward Scaling for RL in Time Series Anomaly Detection (DRTA)](https://arxiv.org/html/2508.18474)
- [Reward Hacking in Reinforcement Learning](https://lilianweng.github.io/posts/2024-11-28-reward-hacking/)
- [LLM-Enhanced Reinforcement Learning for Time Series Anomaly Detection](https://arxiv.org/html/2601.02511v1)

### Physics-Informed Neural Networks
- [Physics-Informed Neural Networks in Grid-Connected Inverters: A Review](https://www.mdpi.com/1996-1073/18/20/5441)
- [Physics-Informed Graph Neural Network for Dynamic Reconfiguration of Power Systems](https://arxiv.org/html/2310.00728v2)
- [Understanding Physics-Informed Neural Networks: Techniques, Applications, Trends, and Challenges](https://www.mdpi.com/2673-2688/5/3/74)
- [Physics-informed neural networks for PDE problems: a comprehensive review](https://link.springer.com/article/10.1007/s10462-025-11322-7)
- [Review of Physics-Informed Neural Networks: Challenges in Loss Function Design and Geometric Integration](https://www.mdpi.com/2227-7390/13/20/3289)

### Evaluation Without Ground Truth
- [Towards Unsupervised Validation of Anomaly-Detection Models](https://arxiv.org/html/2410.14579v1)
- [ASOI: anomaly separation and overlap index, an internal evaluation metric for unsupervised anomaly detection](https://link.springer.com/article/10.1007/s40747-025-02204-0)
- [Unsupervised and semi-supervised anomaly detection with data-centric ML (Google Research)](https://research.google/blog/unsupervised-and-semi-supervised-anomaly-detection-with-data-centric-ml/)
- [How to Evaluate the Quality of Unsupervised Anomaly Detection Algorithms?](https://arxiv.org/pdf/1607.01152)

### API Reliability and Fallback Strategies
- [Hierarchical Fallback Architecture for High Risk Online Machine Learning Inference](https://arxiv.org/html/2501.17834v1)
- [Enhancing reliability in AI inference services: An empirical study on real production incidents](https://arxiv.org/html/2511.07424)
- [Circuit Breaker Pattern - Azure Architecture Center](https://learn.microsoft.com/en-us/azure/architecture/patterns/circuit-breaker)
- [Building Resilient Systems: Circuit Breakers and Retry Patterns (2026)](https://dasroot.net/posts/2026/01/building-resilient-systems-circuit-breakers-retry-patterns/)
- [Real-time ML Inference Infrastructure | Databricks Blog](https://www.databricks.com/blog/2021/09/01/infrastructure-design-for-real-time-machine-learning-inference.html)

### Power Grid Specific
- [Anomaly Detection in Smart Grid Using SCADA System](https://www.sciencedirect.com/science/article/abs/pii/S0378779624007624)
- [Enhancing resilience in complex energy systems through real-time anomaly detection](https://energyinformatics.springeropen.com/articles/10.1186/s42162-024-00401-8)
- [PowerGNN: A Topology-Aware Graph Neural Network for Electricity Grids](https://arxiv.org/html/2503.22721v1)
- [Graph neural networks for power grid operational risk assessment](https://arxiv.org/html/2405.07343v1)
- [PowerGraph: A power grid benchmark dataset for graph neural networks](https://neurips.cc/virtual/2024/poster/97496)
- [Node Classification of Imbalanced Data Using Ensemble Graph Neural Networks](https://www.mdpi.com/2076-3417/15/19/10440)

---

*Research conducted 2026-01-26. All sources verified for currency and relevance to Grid Guardian project context.*
