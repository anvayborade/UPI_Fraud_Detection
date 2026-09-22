# Architecture

## 1. Design objective

The architecture separates **novelty** from **malicious intent**. A high amount, new recipient or distant location is not sufficient to declare fraud. The system combines evidence from the payer, session, journey, recipient and money-flow network.

## 2. Three timing paths

### Precheck path

Triggered when a QR/recipient is selected. It returns cached recipient, complaint, QR and mule-risk information before the user reaches final approval.

### Fast scoring path

Runs synchronously before authorisation:

```text
Rules + rolling statistics + LightGBM transaction model
+ context/legitimate-novelty model + device/session features
+ cached sequence/graph scores
```

### Near-real-time intelligence path

Runs every few minutes:

```text
Time-aware sequence Transformer
+ typed temporal event-memory model
+ HGT graph snapshot encoders
+ multi-view graph attention
+ graph anomaly model
```

It writes updated entity scores to Redis so the next fast-path request only performs low-latency lookups.

## 3. Model experts

### Transaction expert

Answers: “Does this row look risky relative to the user and current conditions?”

### Legitimate-novelty expert

Answers: “Is the unusual transaction coherently explained by a trusted device, plausible journey, verified QR/merchant and low-risk recipient?”

### Sequence expert

Answers: “Does the ordered app/session event journey resemble account takeover or social manipulation?”

### Money-flow graph expert

Answers: “Does the recipient participate in rapid pass-through, splitting, circular flow or risky downstream paths?”

### Identity/device graph expert

Answers: “Are accounts, devices, VPAs and IPs linked in an account farm or coordinated cluster?”

### Context/trust graph expert

Answers: “Does the QR, merchant, location and complaint context support legitimacy or fraud?”

### Complaint intelligence expert

Answers: “Do complaint narratives expose a scam script or risky identifier?”

## 4. Preventing fan-in/fan-out false positives

Raw degree is never treated as proof of fraud. The repository adds:

- recipient archetype;
- peer-normalised fan-in/fan-out;
- median holding time;
- rapid outflow ratio;
- geographic and working-hour consistency;
- merchant/QR stability;
- complaint links;
- risky downstream connections.

A transport provider can therefore have high fan-in yet receive a low mule score when the pattern is stable, local, fare-like and lacks rapid laundering behaviour.

## 5. Fusion and policy

The mixture-of-experts model predicts overall and multi-label risks. The uncertainty module measures disagreement, missing context, graph sparsity and anomaly evidence. The policy engine then selects a proportionate action:

```text
ALLOW → CONFIRM → WARN → STEP_UP → REVIEW/HOLD → BLOCK
```

The classifier does not directly hard-code the operational decision.

## 6. LLM boundary

The LLM is not in the payment-critical loop. It receives redacted complaint text asynchronously and returns schema-constrained structured intelligence. Invalid or unavailable LLM output falls back to a local extractor.
