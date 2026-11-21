# Our Family Plan

**Team Fight Club** — InnoHack '26

> **Synthetic demonstration data — not financial advice.**

## Project Overview

**Our Family Plan** is a human-centered, four-agent AI financial resilience assistant for low-income Sri Lankan households with irregular daily-wage income.

### Milestone Progress

- **Milestone 1 (Complete):** Deterministic `Decimal`-based weekly cashflow engine (`src/models.py`, `src/finance.py`), fictional five-member Sri Lankan household dataset (`data/synthetic_family.json`), interactive Streamlit dashboard (`app.py`), rainy-day + clinic-expense shock simulation, and 12 automated financial tests (`tests/test_finance.py`).
- **Milestone 2 (Complete):** Real **Gemini + LangGraph** four-agent workflow (`src/config.py`, `src/agent_state.py`, `src/tools.py`, `src/agents.py`, `src/workflow.py`) with genuine Gemini tool calling across eight deterministic tools, bounded Critic revision loop (`max_revisions = 2`), curated Sri Lanka welfare & illustrative livelihood data (`data/sri_lanka_welfare_kb.json`, `data/community_livelihoods.json`), explicit approval controls, and 9 offline mocked-LLM tests (`tests/test_agents.py`).
- **Milestone 3 (Complete):** Judge-ready five-section narrative Streamlit experience (`app.py`, `src/demo_state.py`), visible round-by-round Critic revision loop, shock-to-replan agreement comparison preserving historical baseline agreements, live node-by-node progress via `st.status(...)`, deterministic warning deduplication, strict approval/fingerprint invalidation, and Milestone 3 demo transition tests (`tests/test_milestone3_demo.py`).

---

## Project Structure

```text
team-fight-club/
  app.py
  requirements.txt
  README.md
  .gitignore
  .env.example
  data/
    synthetic_family.json
    sri_lanka_welfare_kb.json
    community_livelihoods.json
  src/
    __init__.py
    models.py
    finance.py
    config.py
    agent_state.py
    tools.py
    agents.py
    workflow.py
    demo_state.py
  tests/
    test_finance.py
    test_agents.py
    test_milestone3_demo.py
  docs/
    AI_USAGE.md
```

---

## Complete Live Demonstration Sequence (InnoHack '26 Judge Walkthrough)

1. **Section 1 — Family Overview:** Review the fictional five-member Sri Lankan household (Sunil, Nimali, Kavindu, Tharushi, and grandmother Kamala), confirmed (`Rs. 3,300.00`) vs. uncertain (`Rs. 9,600.00`) income, existing weekly microfinance debt (`Rs. 2,800.00`), essential expenses, and the unverified `Rs. 20,000.00` business loan under consideration.
2. **Section 2 — Financial Health:** Inspect the deterministic daily cashflow calendar and chart showing how optimistic cash looks positive (`+Rs. 3,800.00`) while conservative cash (confirmed income only) hits a liquidity cliff on Thursday `2026-06-11` (`-Rs. 1,100.00`) and ends at `-Rs. 5,800.00`.
3. **Section 3 — Four-Agent AI Analysis:** Click **"Run 4-Agent AI Analysis"**. Observe live node-by-node progress as Gemini invokes the deterministic tools across all four agents, inspect each agent's main conclusion first, check the **Critic Audit & Revision Loop**, and open **Tool-Use & LangGraph Decision Trace**.
4. **Section 4 — Family Recovery Agreement:** Review the 2–3 proposed family actions (reasons, expected trade-offs, unresolved financial risks, and official confirmation notices). Click **"Approve Agreement"** (or **"Reject Agreement"**) to record an explicit family decision.
5. **Section 5 — Shock and Adaptation:** Click **"Simulate Shock: 2 Rainy Days + Clinic Expense"**.
   - Observe the exact removed uncertain wages (`-Rs. 5,600.00`) and added essential clinic expense (`+Rs. 3,200.00` on `2026-06-10`).
   - Note the honest accounting explanation: conservative cash drops strictly by `-Rs. 3,200.00` (clinic expense only, because conservative already excluded uncertain wages), while optimistic cash drops by `-Rs. 8,800.00` and flips negative (`-Rs. 5,000.00`).
   - Note that the previous baseline agreement is immediately preserved as **Historical Baseline** and its approval is **invalidated** for the shocked scenario.
6. **Reanalyze Shocked Scenario:** Click **"Reanalyze Shocked Scenario"** in Section 5. All four agents re-run on the shocked scenario using real Gemini tool calls.
7. **Before / After Agreement Comparison:** Inspect the side-by-side comparison in Section 5 showing exact deterministic financial shifts, which proposed actions changed and why, Critic status shifts, and newly surfaced unresolved risks.
8. **Fresh Family Approval:** Return to Section 4 and explicitly click **"Approve Agreement"** or **"Reject Agreement"** for the revised shocked agreement.
9. **Reset:** Click **"Reset Scenario"** at any time to restore the default synthetic baseline and clear shock/AI/approval state cleanly.

---

## Responsibilities of the Four Cooperating Agents

Why four agents instead of a single chatbot? Separating financial measurement, borrowing-risk evaluation, cautious community guidance, and independent safety auditing prevents a single prompt from hallucinating rupee totals, rubber-stamping risky debt, or promising unverified welfare payouts.

1. **Agent A — Financial Twin (`financial_twin`):** Calls `build_weekly_cashflow_calendar` and `detect_liquidity_cliffs` to explain date-specific liquidity cliffs, zero-income days, unfunded essential/debt payments, and confirmed vs. uncertain income.
2. **Agent B — Debt Shield (`debt_shield`):** Calls `run_loan_stress_test`, `generate_safer_alternatives`, and `compute_plan_diff` to test borrowing pressure under normal and lost-workday scenarios. When repayment terms are missing, the tool explicitly refuses to fabricate interest rates, schedules, or approvals.
3. **Agent C — Community Bridge (`community_bridge`):** Calls `query_sri_lanka_welfare_kb` and `match_community_livelihoods` to provide official `.gov.lk` reference pathways (always marked **"Requires official confirmation"**) and explicitly **illustrative** livelihood ideas.
4. **Agent D — Family Advocate & Critic (`family_advocate_critic`):** Independently audits the combined proposal via `audit_essential_protection` (and optionally `compute_plan_diff`) for essential-need cuts, unrealistic income guarantees, or unsupported claims, then routes conditionally.

---

## Critic Conditional Revision Logic & Display

```text
START
  -> financial_twin
  -> debt_shield
  -> community_bridge
  -> family_advocate_critic
       |-- verdict == "REVISE" and revision_count < 2 --> revise_plan --> family_advocate_critic
       |-- verdict == "ACCEPT" or  revision_count >= 2 --> finalize_agreement -> END
```

- **Never forced or hardcoded:** Gemini genuinely decides `ACCEPT` or `REVISE` after calling `audit_essential_protection` (with a deterministic safety floor if the audit tool itself raises essential-protection flags).
- **Transparent UI display:**
  - Accepted on round 0 -> displays **"Accepted without revision"** alongside Round 0 Initial Proposal and Critic Verdict #1.
  - Revised 1–2 times -> displays Round 0 Initial Proposed Actions -> Critic Verdict #1 + exact reasons for revision -> Revised Proposed Actions -> Critic Verdict #2 -> Final Agreement.
  - Still flagged after 2 revisions -> clearly displays **"Unresolved risks"** (`UNRESOLVED_RISKS_AFTER_MAX_REVISIONS`) without falsely declaring the plan safe.

---

## Baseline vs. Shocked Comparison & Human Approval Rules

- **Scenario fingerprinting:** Every analysis is tied to a SHA-256 `scenario_fingerprint` covering the active scenario events, balances, dates, baseline/shock label, and optional loan terms.
- **No stale analysis:** Any change to financial inputs or shock status invalidates the active view immediately; earlier baseline findings are preserved only under an explicit **Historical Baseline** label for comparison.
- **Explicit human control:** Agreements always initialize to `PENDING_FAMILY_APPROVAL`. Approval is stored per `scenario_fingerprint`, so approving the baseline agreement never automatically approves a shocked or edited scenario.

---

## Gemini Error Handling & Known Limitations

- **Graceful error handling (`src/config.py`):** Missing/placeholder API key, authentication failures (`401`/`403`), quota/rate limits (`429`), upstream high-demand spikes (`503`), and deadline/network timeouts (`504`/connection errors) produce clean, secret-free error messages while preserving previously completed historical analysis (labeled stale if inputs changed) and keeping the deterministic financial dashboard 100% functional.
- **Known limitations:**
  - All financial inputs represent synthetic demonstration data, not regulated financial advice.
  - Local Sri Lanka welfare entries are static curated references marked **"Requires official confirmation"** and do not fetch live government eligibility rules or amounts.
  - Community livelihood suggestions are illustrative only, not live job listings.
  - Live AI response latency and availability depend on Google Gemini API service capacity (default model `gemini-3.5-flash-lite` is configurable via `GEMINI_MODEL` in `.env`).

---

## Windows PowerShell Setup, Verification & Launch

### 1. Setup & Dependencies

```powershell
cd "C:\Users\Savina Methmica\Documents\GitHub\team-fight-club"
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

### 2. Local `.env` Configuration (Never Commit `.env`)

```powershell
Copy-Item .env.example .env
```

Set your key in `.env`:

```env
GEMINI_API_KEY=your_actual_gemini_api_key
GEMINI_MODEL=gemini-3.5-flash-lite
```

### 3. Run Offline Automated Tests & Compilation Checks (No API Key Required)

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m compileall app.py src tests
```

### 4. Launch the Streamlit Application

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Open `http://localhost:8501` in your browser.
