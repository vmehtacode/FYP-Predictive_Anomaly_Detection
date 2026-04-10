# Additional Diagrams

Mermaid diagrams for documentation reference. The system architecture and grid topology diagrams are in [README.md](../README.md).

---

## Hybrid Verifier Cascade

Three-layer verification ensemble with early-exit.

```mermaid
graph TD
    IN[Forecast\nper-node values] --> L1

    L1[Layer 1: PhysicsConstraintLayer\nVoltage / Capacity / Ramp Rate\nTolerance band scoring] --> EX{Severity > 0.9?}

    EX -- Yes --> EARLY[Early Exit\nPhysics score only\nSkip GNN]
    EX -- No --> L2

    L2[Layer 2: GATVerifier\nFrozen GNN inference\n3x GATv2Conv, 4 heads, 64 hidden] --> L3

    L3[Layer 3: CascadeLogicLayer\nNeighbour propagation check\nThreshold 0.3, decay 0.7/hop] --> ENS

    EARLY --> OUT
    ENS[Weighted Ensemble\n0.4 physics + 0.4 GNN + 0.2 cascade] --> OUT

    OUT[Combined Score\nper node in 0 to 1]
```

---

## Self-Play Loop

Propose-solve-verify training cycle with reward feedback.

```mermaid
graph TD
    P[ProposerAgent\nGenerate scenario\nEV_SPIKE / COLD_SNAP / OUTAGE / ...] --> S
    S[SolverAgent\nForecast under scenario\nPatchTST with quantile heads] --> V
    V[VerifierAgent\nPhysics constraint check\nReward in -1 to +1] --> R1

    R1[Verification reward] --> S
    R1 --> LR[Learnability reward\nr = 1 - success_rate\nif solvable, else 0]
    LR --> P

    P -.->|Curriculum 0.0 to 1.0\nHarder scenarios over time| P
```

---

## LED Board Mapping

6 physical LEDs mapped to a subtree of the SSEN grid for the viva demo.

```mermaid
graph TD
    subgraph Raspberry Pi
        LED0[LED 0 - PS1\nGPIO 17]
        LED1[LED 1 - SS1\nGPIO 27]
        LED2[LED 2 - SS2\nGPIO 22]
        LED3[LED 3 - LV1\nGPIO 5]
        LED4[LED 4 - LV2\nGPIO 6]
        LED5[LED 5 - LV3\nGPIO 13]

        LED0 --- LED1
        LED0 --- LED2
        LED1 --- LED3
        LED1 --- LED4
        LED2 --- LED5
    end

    subgraph Streamlit Dashboard
        HV[HybridVerifierAgent\ncombined_scores array] --> MAP[Extract 6 scores\nat mapped indices]
        MAP --> POST[POST /alert\nto Pi Flask server]
    end

    POST -->|JSON over USB| LED0
```

The Pi runs a Flask server on `raspberrypi.local:5050`. The dashboard extracts per-node scores from `details._breakdown.combined_scores` at the 6 mapped graph indices, thresholds them (off / warn / alert), and sends the state as JSON. The cascade demo animates LV -> SS -> PS propagation matching the 2-hop BFS with 0.7 decay from `CascadeLogicLayer`.
