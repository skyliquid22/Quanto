# Product Requirements Document (PRD)
# Quanto Trading System

Version: 1.0  
Owner: Ahmed  
Purpose: Define WHAT the end-to-end trading system must deliver at a product level.  
Audience: PM Agent (MetaGPT/Custom PM), Codex Engineer Agents, Human developers.

---

# 1. Overview / Vision

Quanto is an institutional-grade, end-to-end trading research and execution platform. It automates the full lifecycle:

- Multi-vendor historical and real-time data ingestion
- Feature and label generation for ML & RL
- Reinforcement learning model training using FinRL
- Backtesting and performance evaluation
- Paper and (future) live execution with risk controls
- Continuous monitoring and reporting

The vision is a **modular, reproducible, extensible quant platform** where any component (data, features, models, execution) can evolve independently while maintaining safety and consistency.

---

# 2. Goals & Non-Goals

## 2.1 Goals
The system MUST:

- Provide a robust, vendor-agnostic ingestion pipeline supporting:
  - REST APIs
  - S3 flat files / bulk historical dumps
  - WebSocket real-time feeds (stored as bars)
- Compute institutional-quality features and labels
- Train RL agents reproducibly with FinRL
- Provide backtesting with realistic execution assumptions
- Offer a safe and controlled execution engine (paper → live)
- Provide strong observability (logging, metrics, reporting)
- Support internal Greek calculation (not sourced externally)
- Support Alpaca as the broker for Phase 2

## 2.2 Non-Goals
The system WILL NOT:

- Provide live trading in Phase 1
- Implement a GUI/dashboard initially
- Implement portfolio optimization
- Implement AutoML or automated feature discovery
- Support tick-level real-time ingestion (bars only)
- Serve as a high-frequency trading platform

---

# 3. Users / Personas

### **Quant Researcher**
Needs: clean data, consistent features, backtests, configurable pipelines.

### **ML/RL Engineer**
Needs: reproducible training loops, standardized datasets, configurable environment.

### **Execution Layer / Trader**
Needs: safe order routing, guardrails, logs, transparent risk management.

### **Founder (Ahmed)**
Needs: end-to-end clarity, extensibility, safety, maintainability.

---

# 4. Functional Requirements (FRs)

## **FR-1: Data Ingestion System**

The ingestion system MUST:

1. Support **multiple vendors** (Polygon + future ones).
2. Support **three ingestion modes**:
   - REST (async)
   - Flat files (S3 or equivalent)
   - WebSockets (real-time → bar aggregation only)
3. Provide **dynamic routing logic**:
   - REST for small requests
   - Flat files for large historical ranges
   - WebSocket for live incremental updates
4. Support data types:
   - OHLCV
   - Options chains, quotes, trades, open interest
   - Ticker metadata
   - Fundamentals
5. Normalize all vendor data into canonical schemas (defined in DATA_SPEC.md).
6. Store raw data in Parquet using a canonical directory structure.
7. Perform async REST requests with concurrency limits.
8. Perform multithreaded preprocessing for flat files.
9. Produce detailed ingestion logs + run manifests.
10. Be idempotent and resume-able.

## **FR-2: Data Validation**

System MUST:

- Enforce global validation rules:
  - timestamps normalized to UTC  
  - strict schema validation  
  - monotonicity checks  
  - duplicate removal  
  - missing data handling rules  
- Fail loudly on schema mismatch.
- Record validation results in a manifest file.

## **FR-3: Feature & Label Pipeline**

The system MUST:

- Read normalized raw data.
- Compute:
  - OHLCV-derived features (returns, volatility, microstructure metrics)
  - Options-derived features (internal Greek computation)
  - Cross-sectional features
  - ML/RL labels (forward returns, barrier labels, hit-strike labels)
- Store outputs in standardized Parquet form.
- Maintain versioned feature pipelines for reproducibility.

## **FR-4: RL Training Pipeline (FinRL)**

The system MUST:

- Load features/labels and construct FinRL-compatible trading environments.
- Support PPO as default algorithm (extensible to others).
- Save:
  - trained model files
  - evaluation metrics
  - episodic logs
  - plots
- Support training reproducibility through:
  - fixed seeds
  - config-based hyperparameters
  - manifest of training inputs

## **FR-5: Backtesting Engine**

The system MUST:

- Simulate model outputs across historical data.
- Support realistic execution assumptions:
  - slippage
  - fee schedules
  - bar-shifted execution (no lookahead)
  - market hours / holidays
- Produce:
  - PnL, Sharpe, Sortino
  - drawdown curves
  - trade logs
  - position histories
- Support multi-symbol backtesting.

## **FR-6: Execution Engine (Phase 2+)**

The execution system MUST:

- Use **Alpaca** as the Phase 2 broker.
- Convert signals → target positions → executable orders.
- Support paper trading first.
- Enforce risk constraints:
  - max leverage  
  - max daily loss  
  - max order notional  
  - max trades per minute  
- Write detailed order/fill logs.
- Include kill-switch mechanisms.

Details defined in EXECUTION_SPEC.md.

## **FR-7: Monitoring & Reporting**

System MUST:

- Provide structured logs (JSON)
- Produce ingestion metrics (rows/sec, failures, retries)
- Produce training and backtesting metrics
- Store plots under `monitoring/plots/`
- Summaries must be machine-readable (e.g., JSON manifests)

## **FR-8: Auditable Reproducibility**

System MUST:

- Store manifests for:
  - ingestion runs
  - feature pipeline runs
  - training runs
  - backtests  
- Hash input data files for integrity verification.
- Ensure rerunning with same configs yields identical results.

## **FR-9: Config-Driven Architecture**

System MUST:

- Use YAML configs for:
  - vendors
  - universes
  - data ranges
  - feature parameters
  - training hyperparameters
  - execution settings
- Validate config schemas via Pydantic or equivalent.
- Disallow hardcoded constants inside logic.

## **FR-10: Testing Requirements**

System MUST include:

- Unit tests for:
  - vendor adapters  
  - ingestion logic  
  - feature pipeline  
  - RL environment  
  - execution routing  
- Integration tests for:
  - data → features  
  - features → training  
  - training → backtesting  
- Mock vendors for deterministic tests.
- Target: ≥ 70% test coverage.

---

# 5. Non-Functional Requirements (NFRs)

### **Performance**
- REST ingestion: ≥ 50 req/sec (within vendor limits)
- Flat-file processing: ≥ 3GB/min on 8 cores
- Feature pipeline: must handle multiyear universes efficiently

### **Reliability**
- Automatic retry with exponential backoff  
- WebSocket failures must not crash execution  
- Execution failures must trigger kill-switch logic  

### **Security**
- All API keys must come from environment variables
- No secrets in repo or logs

### **Scalability**
- Support multiple vendors seamlessly
- Be able to extend schemas without breaking existing flows

### **Maintainability**
- Modular code boundaries
- One subsystem must not depend on internal details of another

---

# 6. Release Phases & Milestones

### **Phase 1 – Research Core**
- REST + flat-file ingestion  
- Feature pipeline  
- RL training  
- Backtesting  
- Manifest-based reproducibility  

### **Phase 2 – Execution**
- Alpaca paper trading  
- Risk module  
- Real-time bar ingestion  
- Execution logging + monitoring  

### **Phase 3 – Live Trading**
- Real Alpaca trading  
- WebSocket reliability enhancements  
- SLA-level observability  
- Production guardrails  

---

# 7. Dependencies & External Services

- Polygon REST API  
- Polygon S3 flat files  
- Polygon WebSocket streams  
- Alpaca trading API  
- AWS S3 or MinIO for flat-file storage  
- FinRL  
- PyTorch  
- Pydantic/YAML  

---

# 8. Open Questions (Resolved)

- Should Greeks be computed internally? → **Yes, internally**
- Which broker for Phase 2? → **Alpaca**
- How to store real-time ingestion? → **As bars only (not ticks)**

---

# 9. Acceptance Criteria

A release is **complete** when:

- Ingestion system dynamically routes REST vs flatfile correctly
- All data types pass validation rules
- Feature pipeline produces correct versioned outputs
- RL training is reproducible and documented
- Backtests match expected metrics
- Execution engine enforces all risk constraints
- Logs, manifests, and plots are generated
- Tests pass with coverage ≥ 70%

---

# End of PRD
