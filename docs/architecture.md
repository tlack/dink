# Dink architecture (v1)

Colors: amber = computed once per device setup and cached; green = every query;
dashed grey = training only. The same diagram, with block internals and cost estimates,
is published as a page ("Dink network graph").

```mermaid
flowchart LR
  subgraph SETUP["Once per device setup (cached)"]
    direction TB
    SN["state names<br/>8 × up to 64 bytes<br/>e.g. kitchen thermostat"]
    TN["tool names<br/>8 × up to 64 bytes<br/>e.g. set kitchen temp"]
    TE1["text encoder T<br/>shared weights"]
    NV["name vectors<br/>16 × 64"]
    SN --> TE1
    TN --> TE1
    TE1 --> NV
  end

  subgraph QUERY["Every query"]
    direction TB
    P["prompt<br/>64 bytes"]
    TE2["text encoder T<br/>same weights"]
    PT["prompt tokens<br/>4 × 64"]
    SV["state values<br/>8 × (value, lo, hi)"]
    TV["tool values<br/>8 × (value, lo, hi)"]
    VE["value encoder V<br/>normalize to 0..1<br/>16 bins + magnitude"]
    FU["fuse F<br/>MLP(name ⊕ value)<br/>value-aware item tokens"]
    ST["state tokens<br/>8 × 64"]
    TT["tool tokens<br/>8 × 64"]
    REL["relation step<br/>tool finds its state field<br/>Δ = tool value − state value"]
    NONE["nothing-relevant token<br/>learned, 1 × 64"]
    MIX["mixer<br/>2 attention blocks over 21 tokens<br/>or 1 block applied 3 times"]
    SC["scorer<br/>64 → 1 per tool token<br/>and the nothing token"]
    OUT["masked softmax<br/>9 probabilities"]
    P --> TE2 --> PT
    SV --> VE
    TV --> VE
    VE --> FU
    FU --> ST
    FU --> TT
    ST --> REL
    TT --> REL
    PT --> MIX
    ST --> MIX
    REL --> MIX
    NONE --> MIX
    MIX --> SC --> OUT
  end

  subgraph TRAIN["Training only"]
    direction TB
    TEACH["teacher embedding model<br/>frozen, large"]
    VERB["verbalized items<br/>kitchen thermostat, set low"]
    VERB -.-> TEACH
  end

  NV --> FU
  TEACH -.->|"match vectors<br/>and similarities"| TE1
  TEACH -.->|"optional: match<br/>fused tokens"| FU

  classDef cached stroke:#b98a2e,stroke-width:2px
  classDef query stroke:#2e9e73,stroke-width:2px
  classDef train stroke:#7a8a86,stroke-width:1.5px,stroke-dasharray:5 4
  class SN,TN,TE1,NV cached
  class P,TE2,PT,SV,TV,VE,FU,ST,TT,REL,NONE,MIX,SC,OUT query
  class TEACH,VERB train
```

## Blocks

| block | what it does | when |
|---|---|---|
| text encoder T | bytes → embed 32 → 1×1 conv 32→128 → [depthwise k5 + 1×1 128→128] × 2 → attention pooling (4 prompt tokens, or 1 vector per name) | names once per setup (cached); prompt every query |
| value encoder V | (value, lo, hi) → normalized 0..1 → 16 piecewise-linear bins + signed log magnitude + has-value = 18 features | every query |
| fuse F | [name vector 64 ⊕ value features 18] → 128 → 64: value-aware item tokens ("thermostat, set low") | every query |
| relation step | each tool attends over state name vectors, takes the matching field's value, Δ = tool value − state value → bins → added to the tool token | every query |
| prompt numbers | numbers extracted from the prompt (2-3 slots, with a unit hint: raw or percent) are compared with each tool in that tool's own range: in range?, level, distance to the candidate value, exact match → added to the tool token | every query |
| mixer | 21 tokens × 64 (8 state + 4 prompt + 8 tool + nothing); 2 blocks of 4-head attention + FFN 64→256→64, or 1 block applied 3 times | every query |
| scorer | 64 → 1 per tool token and the nothing token; masked softmax over filled slots | every query |

Estimates: ~190 K parameters, ~6 M multiply-adds per query (~10 ms on the ESP32-S3).

## Drawing the real graph later

This diagram is the design. Once the PyTorch model exists, `torchview` can draw the
graph straight from the code (and Netron from the exported model), so the picture can
be checked against what is actually built.
