"""Our Family Plan — Milestone 3 Streamlit Application (Team Fight Club, InnoHack '26).

Five-section judge-ready dashboard combining:
- Milestone 1: Deterministic Decimal-based weekly cashflow engine & shock simulator.
- Milestone 2: Four cooperating Gemini + LangGraph agents with 8 deterministic tools
  and a bounded Critic revision loop (max 2 revisions).
- Milestone 3: Narrative UI, visible Critic revision history, shock-to-replan
  agreement comparison, honest live progress, and strict stale/approval guards.
"""

from __future__ import annotations

import importlib
import json
import sys
from datetime import date, datetime
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

import altair as alt
import pandas as pd
import streamlit as st


def _ensure_fresh_local_modules() -> None:
    """Reload stale cached ``src.*`` modules if Streamlit hot-reloads ``app.py``."""
    required_symbols = (
        ("src.agent_state", "consolidate_warnings"),
        ("src.agents", "sanitize_narrative_text"),
        ("src.demo_state", "classify_action_changes"),
        ("src.workflow", "run_family_plan_workflow"),
    )
    stale = any(
        mod_name in sys.modules and not hasattr(sys.modules[mod_name], symbol)
        for mod_name, symbol in required_symbols
    )
    if stale:
        for mod_name in (
            "src.agent_state",
            "src.agents",
            "src.demo_state",
            "src.workflow",
            "src",
        ):
            if mod_name in sys.modules:
                importlib.reload(sys.modules[mod_name])


_ensure_fresh_local_modules()

from src.agent_state import (
    ApprovalStatus,
    FamilyPlanState,
    compute_scenario_fingerprint,
    consolidate_warnings,
    dedupe_strings,
)
from src.agents import sanitize_narrative_text
from src.config import AIConfigStatus, check_ai_availability, classify_ai_error
from src.demo_state import (
    DemoAnalysisView,
    clear_demo_ai_store,
    compare_agreements,
    initialize_demo_ai_store,
    record_completed_analysis,
    resolve_demo_analysis_view,
    set_agreement_approval,
)
from src.finance import (
    SAME_DAY_ORDERING_CONVENTION,
    apply_rainy_days_and_clinic_shock,
    build_weekly_cashflow_calendar,
    calculate_scenario_summary,
    compare_scenarios,
    format_lkr,
    format_signed_lkr,
    load_scenario_from_json,
    select_default_rainy_day_income_ids,
)
from src.models import (
    DailyCashflowResult,
    ExpenseCategory,
    ExpenseClassification,
    ExpenseEvent,
    FinancialScenario,
    IncomeCategory,
    IncomeEvent,
    IncomeStatus,
    ScenarioSummary,
    ShockComparison,
    ZERO_LKR,
)
from src.workflow import run_family_plan_workflow

DATA_PATH = Path(__file__).resolve().parent / "data" / "synthetic_family.json"
DEFAULT_CLINIC_AMOUNT = Decimal("3200.00")


def _inject_restrained_styles() -> None:
    """Apply clean, modern, accessible styling with restrained accents and no animations."""
    st.markdown(
        """
        <style>
            .block-container {
                padding-top: 1.4rem;
                padding-bottom: 2.8rem;
                max-width: 1180px;
            }
            /* Prevent truncation of rupee figures and multi-word KPI labels */
            [data-testid="stMetricValue"] {
                font-size: 1.48rem !important;
                line-height: 1.25 !important;
                white-space: nowrap !important;
                overflow: visible !important;
                text-overflow: clip !important;
            }
            [data-testid="stMetricLabel"] {
                font-size: 0.86rem !important;
            }
            [data-testid="stMetricLabel"] p {
                white-space: normal !important;
                overflow: visible !important;
                text-overflow: clip !important;
            }
            [data-testid="stMetricDelta"] {
                font-size: 0.80rem !important;
            }
            [data-testid="stMetricDelta"] > div {
                white-space: normal !important;
                overflow: visible !important;
                text-overflow: clip !important;
            }
            .disclaimer-banner {
                background-color: #f8fafc;
                border: 1px solid #cbd5e1;
                border-left: 4px solid #0f766e;
                border-radius: 6px;
                padding: 0.75rem 1rem;
                margin: 0.5rem 0 1rem 0;
                color: #1e293b;
                font-size: 0.92rem;
            }
            .status-strip {
                display: flex;
                flex-wrap: wrap;
                gap: 0.5rem;
                margin: 0.35rem 0 1.2rem 0;
            }
            .pill {
                display: inline-block;
                padding: 0.28rem 0.7rem;
                border-radius: 999px;
                font-size: 0.82rem;
                font-weight: 600;
                border: 1px solid #cbd5e1;
                background-color: #f8fafc;
                color: #1e293b;
            }
            .pill-baseline {
                background-color: #f0fdfa;
                border-color: #99f6e4;
                color: #115e59;
            }
            .pill-shock {
                background-color: #fef2f2;
                border-color: #fecaca;
                color: #991b1b;
            }
            .pill-approved {
                background-color: #ecfdf5;
                border-color: #a7f3d0;
                color: #065f46;
            }
            .pill-rejected {
                background-color: #fef2f2;
                border-color: #fecaca;
                color: #991b1b;
            }
            .pill-pending {
                background-color: #fffbeb;
                border-color: #fde68a;
                color: #92400e;
            }
            .pill-historical {
                background-color: #f1f5f9;
                border-color: #94a3b8;
                color: #334155;
            }
            .section-kicker {
                font-size: 0.78rem;
                font-weight: 700;
                letter-spacing: 0.06em;
                text-transform: uppercase;
                color: #0f766e;
                margin-bottom: 0.1rem;
            }
            .section-note {
                color: #475569;
                font-size: 0.91rem;
                margin-bottom: 0.75rem;
            }
            .callout-card {
                background-color: #f8fafc;
                border: 1px solid #e2e8f0;
                border-radius: 6px;
                padding: 0.9rem 1.05rem;
                margin-bottom: 0.9rem;
                color: #0f172a;
            }
            .conclusion-box {
                background-color: #f8fafc;
                border-left: 3px solid #0f766e;
                padding: 0.55rem 0.75rem;
                border-radius: 4px;
                margin: 0.35rem 0 0.6rem 0;
                color: #0f172a;
                font-size: 0.94rem;
            }
            .warning-card {
                background-color: #fef2f2;
                border: 1px solid #fecaca;
                border-left: 4px solid #b91c1c;
                border-radius: 6px;
                padding: 0.85rem 1rem;
                margin-bottom: 0.7rem;
                color: #7f1d1d;
            }
            .caution-card {
                background-color: #fffbeb;
                border: 1px solid #fde68a;
                border-left: 4px solid #b45309;
                border-radius: 6px;
                padding: 0.85rem 1rem;
                margin-bottom: 0.7rem;
                color: #78350f;
            }
            .honest-note {
                background-color: #f0fdfa;
                border: 1px solid #99f6e4;
                border-left: 4px solid #0f766e;
                border-radius: 6px;
                padding: 0.85rem 1.05rem;
                margin: 0.7rem 0 0.95rem 0;
                color: #134e4a;
            }
            .historical-card {
                background-color: #f8fafc;
                border: 1px dashed #94a3b8;
                border-radius: 6px;
                padding: 0.9rem 1.05rem;
                margin: 0.75rem 0;
                color: #334155;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _render_section_header(number: int, title: str, subtitle: str) -> None:
    st.markdown(f'<div class="section-kicker">Section {number}</div>', unsafe_allow_html=True)
    st.subheader(title)
    if subtitle:
        st.markdown(f'<p class="section-note">{subtitle}</p>', unsafe_allow_html=True)


def _load_default_scenario() -> FinancialScenario:
    return load_scenario_from_json(DATA_PATH)


def _income_events_to_df(events: tuple[IncomeEvent, ...]) -> pd.DataFrame:
    rows = [
        {
            "event_id": e.event_id,
            "event_date": e.event_date,
            "description": e.description,
            "category": e.category.value,
            "status": e.status.value,
            "amount_lkr": float(e.amount),
            "earner": e.earner,
        }
        for e in events
    ]
    return pd.DataFrame(rows)


def _expense_events_to_df(events: tuple[ExpenseEvent, ...]) -> pd.DataFrame:
    rows = [
        {
            "event_id": e.event_id,
            "event_date": e.event_date,
            "description": e.description,
            "category": e.category.value,
            "classification": e.classification.value,
            "amount_lkr": float(e.amount),
        }
        for e in events
    ]
    return pd.DataFrame(rows)


def _coerce_ui_decimal(raw_value: Any, field_name: str) -> Decimal:
    """Convert a Streamlit numeric/string widget value safely into an exact Decimal."""
    if raw_value is None or (isinstance(raw_value, float) and pd.isna(raw_value)):
        raise ValueError(f"{field_name} is required.")
    if isinstance(raw_value, Decimal):
        return raw_value
    if isinstance(raw_value, int) and not isinstance(raw_value, bool):
        return Decimal(raw_value)
    if isinstance(raw_value, float):
        return Decimal(f"{raw_value:.2f}")
    if isinstance(raw_value, str):
        cleaned = raw_value.strip().replace(",", "")
        try:
            return Decimal(cleaned)
        except InvalidOperation as exc:
            raise ValueError(f"{field_name} must be a valid number, got {raw_value!r}.") from exc
    raise ValueError(f"Unsupported value for {field_name}: {raw_value!r}.")


def _coerce_ui_date(raw_value: Any, field_name: str) -> date:
    """Convert a Streamlit date/Timestamp cell into a Python date."""
    if raw_value is None or pd.isna(raw_value):
        raise ValueError(f"{field_name} is required.")
    if isinstance(raw_value, pd.Timestamp):
        return raw_value.date()
    if isinstance(raw_value, datetime):
        return raw_value.date()
    if isinstance(raw_value, date):
        return raw_value
    if isinstance(raw_value, str):
        return date.fromisoformat(raw_value.strip())
    raise ValueError(f"Invalid date for {field_name}: {raw_value!r}.")


def _build_scenario_from_ui(
    default_scenario: FinancialScenario,
    start_date: date,
    end_date: date,
    opening_balance_raw: Any,
    income_df: pd.DataFrame,
    expense_df: pd.DataFrame,
) -> FinancialScenario:
    """Validate and construct an immutable FinancialScenario from UI editor state."""
    opening_balance = _coerce_ui_decimal(opening_balance_raw, "Opening cash balance")

    incomes: list[IncomeEvent] = []
    for idx, row in income_df.iterrows():
        row_num = int(idx) + 1
        if (
            pd.isna(row.get("event_id"))
            and pd.isna(row.get("description"))
            and pd.isna(row.get("amount_lkr"))
        ):
            continue
        event_id = str(row.get("event_id") or "").strip() or f"inc-custom-{row_num}"
        incomes.append(
            IncomeEvent(
                event_id=event_id,
                event_date=_coerce_ui_date(row.get("event_date"), f"Income row {row_num} date"),
                description=str(row.get("description") or "").strip(),
                amount=_coerce_ui_decimal(row.get("amount_lkr"), f"Income row {row_num} amount"),
                category=str(row.get("category") or "").strip(),  # type: ignore[arg-type]
                status=str(row.get("status") or "").strip(),  # type: ignore[arg-type]
                earner=str(row.get("earner") or "").strip(),
            )
        )

    expenses: list[ExpenseEvent] = []
    for idx, row in expense_df.iterrows():
        row_num = int(idx) + 1
        if (
            pd.isna(row.get("event_id"))
            and pd.isna(row.get("description"))
            and pd.isna(row.get("amount_lkr"))
        ):
            continue
        event_id = str(row.get("event_id") or "").strip() or f"exp-custom-{row_num}"
        expenses.append(
            ExpenseEvent(
                event_id=event_id,
                event_date=_coerce_ui_date(row.get("event_date"), f"Expense row {row_num} date"),
                description=str(row.get("description") or "").strip(),
                amount=_coerce_ui_decimal(row.get("amount_lkr"), f"Expense row {row_num} amount"),
                category=str(row.get("category") or "").strip(),  # type: ignore[arg-type]
                classification=str(row.get("classification") or "").strip(),  # type: ignore[arg-type]
            )
        )

    return FinancialScenario(
        scenario_id=default_scenario.scenario_id,
        name=default_scenario.name,
        household=default_scenario.household,
        start_date=start_date,
        end_date=end_date,
        opening_balance=opening_balance,
        income_events=tuple(incomes),
        expense_events=tuple(expenses),
        notes=default_scenario.notes,
    )


def _calendar_to_display_df(
    calendar: list[DailyCashflowResult] | tuple[DailyCashflowResult, ...],
) -> pd.DataFrame:
    rows = []
    for day in calendar:
        if day.unfunded_essential_payments:
            unfunded_text = "; ".join(
                f"{u.description} (short {format_lkr(u.shortfall_amount)})"
                for u in day.unfunded_essential_payments
            )
        else:
            unfunded_text = "None"

        rows.append(
            {
                "Date": f"{day.day_date.isoformat()} ({day.day_date.strftime('%a')})",
                "Opening (Cons.)": format_lkr(day.conservative_opening_balance),
                "Confirmed In": format_lkr(day.confirmed_inflows),
                "Uncertain In": format_lkr(day.uncertain_inflows),
                "Essential Out": format_lkr(day.essential_outflows),
                "Debt Out": format_lkr(day.debt_outflows),
                "Nonessential Out": format_lkr(day.nonessential_outflows),
                "Total Out": format_lkr(day.total_outflows),
                "Conservative Closing": format_lkr(day.conservative_closing_balance),
                "Optimistic Closing": format_lkr(day.optimistic_closing_balance),
                "Unfunded Essential / Debt": unfunded_text,
            }
        )
    return pd.DataFrame(rows)


def resolve_ai_display_state(
    cached_result: FamilyPlanState | None,
    current_fingerprint: str,
) -> tuple[FamilyPlanState | None, str | None]:
    """Ensure stale AI findings are never presented as current after inputs change.

    Preserved for backward compatibility with Milestone 2 tests.
    """
    if cached_result is None:
        return None, None
    analyzed_fp = cached_result.get("scenario_fingerprint")
    if analyzed_fp != current_fingerprint:
        prev_label = cached_result.get("scenario_label", "an earlier scenario")
        return (
            None,
            f"Previous 4-agent AI analysis (for '{prev_label}') is now stale because financial "
            "inputs, scenario shock state, or loan terms changed. Click 'Run 4-Agent AI Analysis' "
            "to analyze the current scenario.",
        )
    return cached_result, None


def _approval_pill_html(status: ApprovalStatus | None) -> str:
    if status == "APPROVED_BY_FAMILY":
        return '<span class="pill pill-approved">Agreement: Approved by Family</span>'
    if status == "REJECTED_BY_FAMILY":
        return '<span class="pill pill-rejected">Agreement: Rejected by Family</span>'
    if status == "PENDING_FAMILY_APPROVAL":
        return '<span class="pill pill-pending">Agreement: Pending Family Approval</span>'
    return '<span class="pill">Agreement: Not Yet Generated</span>'


def _render_balance_chart(
    calendar: list[DailyCashflowResult] | tuple[DailyCashflowResult, ...],
    title: str,
) -> None:
    chart_rows = []
    for day in calendar:
        label = day.day_date.strftime("%b %d (%a)")
        chart_rows.append(
            {
                "Day": label,
                "ISO Date": day.day_date.isoformat(),
                "Track": "Conservative (Confirmed Income Only)",
                "Balance (LKR)": float(day.conservative_closing_balance),
            }
        )
        chart_rows.append(
            {
                "Day": label,
                "ISO Date": day.day_date.isoformat(),
                "Track": "Optimistic (Includes Uncertain Income)",
                "Balance (LKR)": float(day.optimistic_closing_balance),
            }
        )
    df = pd.DataFrame(chart_rows)
    day_order = [day.day_date.strftime("%b %d (%a)") for day in calendar]

    base = alt.Chart(df).encode(
        x=alt.X("Day:N", sort=day_order, title="Date"),
        y=alt.Y("Balance (LKR):Q", title="Closing Cash Balance (Rs.)"),
        color=alt.Color(
            "Track:N",
            scale=alt.Scale(
                domain=[
                    "Conservative (Confirmed Income Only)",
                    "Optimistic (Includes Uncertain Income)",
                ],
                range=["#b91c1c", "#0f766e"],
            ),
            legend=alt.Legend(orient="bottom", title=None, labelLimit=280),
        ),
        tooltip=["ISO Date", "Track", alt.Tooltip("Balance (LKR):Q", format=",.2f")],
    )

    lines = base.mark_line(point=True, strokeWidth=2.5)
    zero_rule = (
        alt.Chart(pd.DataFrame({"y": [0]}))
        .mark_rule(color="#475569", strokeDash=[5, 5], strokeWidth=1.5)
        .encode(y="y:Q")
    )

    chart = (
        (zero_rule + lines)
        .properties(title=title, height=290)
        .configure_axis(labelAngle=0)
    )
    st.altair_chart(chart, width="stretch")


def _render_comparison_chart(comparison: ShockComparison) -> None:
    chart_rows = []
    for base_day, shock_day in zip(comparison.baseline_calendar, comparison.shocked_calendar):
        label = base_day.day_date.strftime("%b %d (%a)")
        iso = base_day.day_date.isoformat()
        chart_rows.extend(
            [
                {
                    "Day": label,
                    "ISO Date": iso,
                    "Series": "Baseline Conservative",
                    "Balance (LKR)": float(base_day.conservative_closing_balance),
                },
                {
                    "Day": label,
                    "ISO Date": iso,
                    "Series": "Shocked Conservative",
                    "Balance (LKR)": float(shock_day.conservative_closing_balance),
                },
                {
                    "Day": label,
                    "ISO Date": iso,
                    "Series": "Baseline Optimistic",
                    "Balance (LKR)": float(base_day.optimistic_closing_balance),
                },
                {
                    "Day": label,
                    "ISO Date": iso,
                    "Series": "Shocked Optimistic",
                    "Balance (LKR)": float(shock_day.optimistic_closing_balance),
                },
            ]
        )

    df = pd.DataFrame(chart_rows)
    day_order = [day.day_date.strftime("%b %d (%a)") for day in comparison.baseline_calendar]

    lines = (
        alt.Chart(df)
        .mark_line(point=True, strokeWidth=2.2)
        .encode(
            x=alt.X("Day:N", sort=day_order, title="Date"),
            y=alt.Y("Balance (LKR):Q", title="Closing Cash Balance (Rs.)"),
            color=alt.Color(
                "Series:N",
                scale=alt.Scale(
                    domain=[
                        "Baseline Conservative",
                        "Shocked Conservative",
                        "Baseline Optimistic",
                        "Shocked Optimistic",
                    ],
                    range=["#dc2626", "#7f1d1d", "#0d9488", "#d97706"],
                ),
                legend=alt.Legend(orient="bottom", title=None, columns=2, labelLimit=240),
            ),
            strokeDash=alt.StrokeDash(
                "Series:N",
                scale=alt.Scale(
                    domain=[
                        "Baseline Conservative",
                        "Shocked Conservative",
                        "Baseline Optimistic",
                        "Shocked Optimistic",
                    ],
                    range=[[4, 3], [1, 0], [4, 3], [1, 0]],
                ),
                legend=None,
            ),
            tooltip=["ISO Date", "Series", alt.Tooltip("Balance (LKR):Q", format=",.2f")],
        )
    )
    zero_rule = (
        alt.Chart(pd.DataFrame({"y": [0]}))
        .mark_rule(color="#475569", strokeDash=[5, 5], strokeWidth=1.5)
        .encode(y="y:Q")
    )

    chart = (
        (zero_rule + lines)
        .properties(title="Baseline vs. Shocked Daily Closing Balances (Rs.)", height=310)
        .configure_axis(labelAngle=0)
    )
    st.altair_chart(chart, width="stretch")


def _render_shortage_warnings(summary: ScenarioSummary, heading: str) -> None:
    st.markdown(f"#### {heading}")
    if not summary.shortage_warnings:
        st.success(
            "No conservative cash shortages detected in this scenario. "
            "Confirmed income plus opening cash covers all dated expenses."
        )
        return

    st.caption(
        f"Conservative balance (confirmed income only) is negative on "
        f"{len(summary.conservative_negative_days)} day(s), starting "
        f"{summary.first_liquidity_cliff_date.isoformat() if summary.first_liquidity_cliff_date else 'N/A'}."
    )
    for warning in summary.shortage_warnings:
        card_class = "warning-card" if warning.severity == "critical" else "caution-card"
        if warning.unfunded_essential_payments:
            items = "".join(
                f"<li><strong>{u.description}</strong> ({u.classification.value.replace('_', ' ')}): "
                f"short {format_lkr(u.shortfall_amount)} of {format_lkr(u.amount)}</li>"
                for u in warning.unfunded_essential_payments
            )
            unfunded_lines = f"<ul style='margin:0.35rem 0 0 1.2rem;'>{items}</ul>"
        else:
            unfunded_lines = (
                "<div style='margin-top:0.3rem;'>No new essential or debt payments are unfunded "
                "on this specific day (deficit is from nonessential spending or carried forward).</div>"
            )

        st.markdown(
            f"""
            <div class="{card_class}">
                <strong>{warning.warning_date.isoformat()} ({warning.warning_date.strftime('%A')})</strong>
                — Conservative closing: <strong>{format_lkr(warning.conservative_closing_balance)}</strong>
                (shortage {format_lkr(warning.shortage_amount)}) |
                Optimistic closing: {format_lkr(warning.optimistic_closing_balance)}
                {unfunded_lines}
            </div>
            """,
            unsafe_allow_html=True,
        )


def _render_critic_revision_loop(result: FamilyPlanState) -> None:
    """Display the authentic Critic revision loop timeline without inventing revisions."""
    st.markdown("#### Critic Audit & Revision Loop")
    critic = result.get("critic_findings")
    if critic is None:
        st.caption("No Critic audit recorded.")
        return

    rev_count = result["revision_count"]
    agreement = result.get("final_agreement")
    critic_status = agreement["critic_status"] if agreement is not None else "UNKNOWN"

    critic_summary_clean = sanitize_narrative_text(
        critic["summary"], fallback="Critic audit completed."
    )
    if rev_count == 0 and critic["verdict"] == "ACCEPT":
        st.success(
            f"**Accepted without revision** — Agent D (Family Advocate & Critic) audited the "
            f"initial draft and accepted it on round 0. {critic_summary_clean}"
        )
    elif critic["verdict"] == "ACCEPT":
        st.success(
            f"**Accepted after {rev_count} revision(s)** — Critic requested revision(s) and "
            f"accepted the revised plan on round {rev_count}. {critic_summary_clean}"
        )
    else:
        st.error(
            f"**Unresolved risks** — Maximum revision limit ({result['max_revisions']}) reached "
            f"with Critic concerns still open ({critic_status}). {critic_summary_clean}"
        )

    snapshots = result.get("proposal_history") or []
    critic_history = result.get("critic_history") or []

    # Pair each proposal snapshot with the Critic verdict that audited it.
    max_rounds = max(len(snapshots), len(critic_history))
    for round_idx in range(max_rounds):
        snap = snapshots[round_idx] if round_idx < len(snapshots) else None
        crit = critic_history[round_idx] if round_idx < len(critic_history) else None

        round_title = (
            snap["stage_label"]
            if snap is not None
            else (f"Initial Proposal (Round 0)" if round_idx == 0 else f"Revision {round_idx}")
        )
        verdict_tag = f" -> Critic Verdict #{round_idx + 1}: {crit['verdict']}" if crit else ""
        expanded_default = (crit is not None and crit["verdict"] == "REVISE") or (round_idx > 0)

        with st.expander(f"{round_title}{verdict_tag}", expanded=expanded_default):
            left, right = st.columns(2)
            with left:
                st.markdown(f"**{round_title} Actions**")
                if snap is not None and snap["actions"]:
                    if snap.get("revision_summary"):
                        st.caption(sanitize_narrative_text(snap["revision_summary"]))
                    for idx, act in enumerate(snap["actions"], start=1):
                        st.markdown(
                            f"{idx}. **{sanitize_narrative_text(act['title'])}** *({act['owner']})*"
                        )
                else:
                    st.caption("No proposal snapshot recorded for this round.")
            with right:
                st.markdown(f"**Critic Review #{round_idx + 1}**")
                if crit is not None:
                    st.markdown(f"- **Verdict:** `{crit['verdict']}`")
                    st.markdown(
                        f"- **Essential-protection audit tool called:** `{crit['audit_tool_called']}`"
                    )
                    st.write(sanitize_narrative_text(crit["summary"], fallback="Critic audit recorded."))
                    if crit["verdict"] == "REVISE" and crit["revision_instructions"]:
                        st.markdown(
                            f"**Reasons for revision:** {sanitize_narrative_text(crit['revision_instructions'])}"
                        )
                    distinct_issues = consolidate_warnings(
                        [
                            *crit["issues"],
                            *crit["unrealistic_assumptions"],
                            *crit["unfunded_essentials"],
                            *crit["unsupported_claims"],
                        ]
                    )
                    if distinct_issues:
                        st.markdown("**Audit flags & concerns:**")
                        for issue in distinct_issues:
                            st.markdown(f"- {issue}")
                else:
                    st.caption("No Critic verdict recorded for this round.")


def _execute_ai_workflow(
    *,
    active_scenario: FinancialScenario,
    baseline_scenario: FinancialScenario,
    shocked_scenario: FinancialScenario | None,
    active_scenario_label: str,
    loan_terms_provided: dict[str, Any] | None,
    ai_status: AIConfigStatus,
) -> None:
    """Run the bounded 4-agent Gemini workflow with live node progress and rerun guard."""
    if not ai_status.is_available:
        st.session_state["ai_last_error"] = ai_status.reason
        return
    if st.session_state.get("ai_is_running"):
        return

    st.session_state["ai_is_running"] = True
    st.session_state["ai_last_error"] = None

    with st.status(
        f"Running 4 cooperating Gemini agents ({ai_status.model_name}) on {active_scenario_label}...",
        expanded=True,
    ) as status_box:
        def _on_progress(message: str) -> None:
            status_box.write(f"- {message}")

        try:
            result = run_family_plan_workflow(
                active_scenario=active_scenario,
                baseline_scenario=baseline_scenario,
                scenario_label=active_scenario_label,
                shocked_scenario=shocked_scenario,
                loan_terms_provided=loan_terms_provided,
                model_name=ai_status.model_name,
                on_progress=_on_progress,
            )
            record_completed_analysis(
                st.session_state,
                result,
                is_baseline_scenario=(shocked_scenario is None),
            )
            status_box.update(
                label=f"Completed 4-agent analysis for {active_scenario_label}.",
                state="complete",
                expanded=False,
            )
        except Exception as exc:
            st.session_state["ai_last_error"] = classify_ai_error(exc)
            status_box.update(
                label="Gemini analysis could not complete (previous historical analysis preserved).",
                state="error",
                expanded=True,
            )
        finally:
            st.session_state["ai_is_running"] = False

    st.rerun()


def _render_agreement_card_body(
    agreement: dict[str, Any],
    *,
    is_historical: bool = False,
) -> None:
    """Render proposed actions, trade-offs, unresolved risks, and uncertainties."""
    if is_historical:
        st.markdown(
            f'<div class="historical-card"><strong>Historical Reference Only ({agreement["scenario_label"]})</strong> '
            "— Preserved from an earlier scenario for comparison. This agreement does NOT apply to "
            "the current scenario.</div>",
            unsafe_allow_html=True,
        )
    elif agreement["critic_status"] != "ACCEPTED_BY_CRITIC":
        st.warning(
            "Unresolved risks: Maximum revision attempts were reached with Critic concerns still "
            "open. Review every risk below carefully before deciding."
        )

    if agreement["actions"]:
        for idx, act in enumerate(agreement["actions"], start=1):
            with st.container(border=True):
                clean_title = sanitize_narrative_text(act["title"], fallback="Proposed family action")
                clean_why = sanitize_narrative_text(
                    act["why_suggested"], fallback="Grounded in deterministic scenario analysis."
                )
                clean_tradeoff = sanitize_narrative_text(
                    act["expected_trade_off"], fallback="Requires household review before acting."
                )
                clean_evidence = sanitize_narrative_text(
                    act["grounded_evidence"], fallback="See deterministic tool trace."
                )
                st.markdown(f"**Action {idx}: {clean_title}** *(Owner: {act['owner']})*")
                st.markdown(f"- **Why suggested:** {clean_why}")
                st.markdown(f"- **Expected trade-off:** {clean_tradeoff}")
                st.caption(
                    f"Source agent: `{act['source_agent']}` | Grounded evidence: {clean_evidence}"
                )
    else:
        st.warning("No proposed actions were generated.")

    risks = consolidate_warnings(agreement.get("unresolved_risks", []))
    uncertainties = consolidate_warnings(agreement.get("uncertainties", []))

    r_col, u_col = st.columns(2)
    with r_col:
        st.markdown("**Unresolved Financial Risks**")
        if risks:
            for risk in risks:
                st.markdown(f"- {risk}")
        else:
            st.caption("No unresolved financial risks flagged.")
    with u_col:
        st.markdown("**Uncertainties & Official Verification Reminders**")
        if uncertainties:
            for unc in uncertainties:
                st.markdown(f"- {unc}")
        else:
            st.caption("No additional uncertainties listed.")


def _render_agreement_comparison(
    baseline_state: FamilyPlanState,
    shocked_state: FamilyPlanState,
    *,
    baseline_approval: ApprovalStatus | None,
    shocked_approval: ApprovalStatus,
) -> None:
    """Render before/after Baseline vs Shocked Agreement comparison in Section 5."""
    comp = compare_agreements(baseline_state, shocked_state)
    fin = comp["financial_comparison"]
    changes = comp["action_changes"]
    crit = comp["critic_comparison"]

    st.markdown("#### Before / After Agreement Comparison (Historical Baseline vs. Shocked Reanalysis)")
    k1, k2, k3, k4 = st.columns(4)
    k1.metric(
        "Shocked Conservative",
        format_lkr(Decimal(fin["shocked_conservative_ending_lkr"])),
        delta=format_signed_lkr(Decimal(fin["conservative_ending_delta_lkr"])),
    )
    k2.metric(
        "Shocked Optimistic",
        format_lkr(Decimal(fin["shocked_optimistic_ending_lkr"])),
        delta=format_signed_lkr(Decimal(fin["optimistic_ending_delta_lkr"])),
    )
    k3.metric(
        "Unfunded Essentials",
        format_lkr(Decimal(fin["shocked_unfunded_shortfall_lkr"])),
        delta=format_signed_lkr(Decimal(fin["unfunded_shortfall_delta_lkr"])),
        delta_color="inverse",
    )
    k4.metric(
        "Shocked First Cliff",
        str(fin["shocked_first_cliff"] or "None"),
        delta=f"Baseline: {fin['baseline_first_cliff'] or 'None'}",
        delta_color="off",
    )

    if comp["shortage_still_unresolved"]:
        st.info(
            "Honest status: Even after agent replanning, deterministic conservative cashflow "
            f"still ends at {format_lkr(Decimal(fin['shocked_conservative_ending_lkr']))}. "
            "No shortage is claimed resolved unless the calculations support it."
        )

    left, right = st.columns(2)
    with left:
        with st.container(border=True):
            st.markdown("**Historical Baseline Agreement**")
            st.markdown(
                f'<span class="pill pill-historical">Historical Baseline</span> '
                f'{_approval_pill_html(baseline_approval)}',
                unsafe_allow_html=True,
            )
            st.caption(
                f"Critic: `{crit['baseline_status']}` | Revisions: {crit['baseline_revisions']}"
            )
            base_ag = baseline_state.get("final_agreement")
            if base_ag and base_ag["actions"]:
                for idx, act in enumerate(base_ag["actions"], start=1):
                    st.markdown(f"{idx}. **{act['title']}**")
                    st.caption(f"Trade-off: {act['expected_trade_off']}")
    with right:
        with st.container(border=True):
            st.markdown("**Current Shocked Agreement**")
            st.markdown(
                f'<span class="pill pill-shock">Active Shocked Scenario</span> '
                f'{_approval_pill_html(shocked_approval)}',
                unsafe_allow_html=True,
            )
            st.caption(
                f"Critic: `{crit['shocked_status']}` | Revisions: {crit['shocked_revisions']}"
            )
            shock_ag = shocked_state.get("final_agreement")
            if shock_ag and shock_ag["actions"]:
                for idx, act in enumerate(shock_ag["actions"], start=1):
                    st.markdown(f"{idx}. **{act['title']}**")
                    st.caption(f"Trade-off: {act['expected_trade_off']}")

    st.markdown("**Action-by-Action Comparison (Conservative Intent Matching)**")
    kept_list = changes.get("kept_actions", [])
    updated_list = changes.get("updated_actions", [])
    possibly_related_list = changes.get("possibly_related_actions", [])
    new_list = changes.get("new_actions", [])
    replaced_list = changes.get("replaced_actions", [])

    if kept_list:
        for rec in kept_list:
            st.markdown(
                f"- **`Kept`** — **{rec['shocked_action']['title']}**  \n"
                f"  *{rec['change_explanation']}*"
            )

    if updated_list:
        for rec in updated_list:
            st.markdown(
                f"- **`Updated`** — **Baseline:** {rec['baseline_action']['title']}  \n"
                f"  **Shocked:** {rec['shocked_action']['title']}  \n"
                f"  *{rec['change_explanation']}*"
            )

    if possibly_related_list:
        for rec in possibly_related_list:
            st.markdown(
                f"- **`Possibly related / needs review`** — **Baseline:** {rec['baseline_action']['title']}  \n"
                f"  **Shocked:** {rec['shocked_action']['title']}  \n"
                f"  *{rec['change_explanation']}*"
            )

    if new_list:
        for act in new_list:
            st.markdown(
                f"- **`New`** — **{act['title']}**  \n"
                f"  *Why added under shock:* {act['why_suggested']}"
            )

    if replaced_list:
        for act in replaced_list:
            st.markdown(f"- **`Replaced`** — **Baseline only:** {act['title']}")

    if not any((kept_list, updated_list, possibly_related_list, new_list, replaced_list)):
        st.caption("No actions were available to compare.")

    new_risks = comp["unresolved_risks_comparison"]["new_risks_under_shock"]
    if new_risks:
        st.markdown("**New Unresolved Risks Surfaced Under Shock**")
        for risk in new_risks:
            st.markdown(f"- {risk}")


def _initialize_session_state(default_scenario: FinancialScenario) -> None:
    if "editor_version" not in st.session_state:
        st.session_state["editor_version"] = 0
    if "shock_active" not in st.session_state:
        st.session_state["shock_active"] = False
    if "base_start_date" not in st.session_state:
        st.session_state["base_start_date"] = default_scenario.start_date
    if "base_end_date" not in st.session_state:
        st.session_state["base_end_date"] = default_scenario.end_date
    if "base_opening_balance" not in st.session_state:
        st.session_state["base_opening_balance"] = float(default_scenario.opening_balance)
    if "base_income_df" not in st.session_state:
        st.session_state["base_income_df"] = _income_events_to_df(default_scenario.income_events)
    if "base_expense_df" not in st.session_state:
        st.session_state["base_expense_df"] = _expense_events_to_df(default_scenario.expense_events)
    initialize_demo_ai_store(st.session_state)


def _reset_to_default(default_scenario: FinancialScenario) -> None:
    st.session_state["shock_active"] = False
    st.session_state["base_start_date"] = default_scenario.start_date
    st.session_state["base_end_date"] = default_scenario.end_date
    st.session_state["base_opening_balance"] = float(default_scenario.opening_balance)
    st.session_state["base_income_df"] = _income_events_to_df(default_scenario.income_events)
    st.session_state["base_expense_df"] = _expense_events_to_df(default_scenario.expense_events)
    clear_demo_ai_store(st.session_state)
    st.session_state["editor_version"] = st.session_state.get("editor_version", 0) + 1
    st.rerun()


def main() -> None:
    st.set_page_config(
        page_title="Our Family Plan | Team Fight Club",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    _inject_restrained_styles()

    default_scenario = _load_default_scenario()
    _initialize_session_state(default_scenario)
    version = st.session_state["editor_version"]
    ai_status = check_ai_availability()

    # =========================================================================
    # HEADER
    # =========================================================================
    st.title("Our Family Plan")
    st.caption(
        "Team Fight Club · InnoHack '26 · Deterministic Financial Engine + "
        "4-Agent Gemini & LangGraph Assistant"
    )
    st.markdown(
        """
        <div class="disclaimer-banner">
            <strong>Synthetic demonstration data — not financial advice.</strong>
            All Sri Lankan Rupee (LKR) calculations come strictly from the deterministic
            <code>Decimal</code> financial engine. Four cooperating Gemini agents invoke these
            tools through LangGraph to explain cash-timing risks, evaluate borrowing pressure,
            surface officially-confirmable support pathways, and audit essential protection.
        </div>
        """,
        unsafe_allow_html=True,
    )

    # =========================================================================
    # SECTION 1: FAMILY OVERVIEW (+ Editable Financial Inputs)
    # =========================================================================
    _render_section_header(
        1,
        "Family Overview",
        "Fictional five-member Sri Lankan household facing irregular daily wages, fixed debt, and essential weekly expenses.",
    )

    household = default_scenario.household

    with st.expander(
        "Edit Family Financial Inputs (Opening Balance, Dates, Income & Expenses)",
        expanded=False,
    ):
        st.caption(
            "Adjust opening cash, scenario dates, income certainty, or dated expenses. "
            "Any edit recalculates deterministic balances immediately and invalidates stale AI findings."
        )
        in_col1, in_col2, in_col3 = st.columns(3)
        with in_col1:
            opening_balance_input = st.number_input(
                "Opening Cash Balance (Rs.)",
                min_value=0.0,
                step=100.0,
                value=float(st.session_state["base_opening_balance"]),
                format="%.2f",
                key=f"opening_balance_{version}",
            )
        with in_col2:
            start_date_input = st.date_input(
                "Scenario Start Date",
                value=st.session_state["base_start_date"],
                key=f"start_date_{version}",
            )
        with in_col3:
            end_date_input = st.date_input(
                "Scenario End Date",
                value=st.session_state["base_end_date"],
                key=f"end_date_{version}",
            )

        st.markdown("**Income Events** (`confirmed` vs. `uncertain`)")
        edited_income_df = st.data_editor(
            st.session_state["base_income_df"],
            num_rows="dynamic",
            hide_index=True,
            width="stretch",
            key=f"income_editor_{version}",
            column_config={
                "event_id": st.column_config.TextColumn("ID", required=True),
                "event_date": st.column_config.DateColumn("Date", format="YYYY-MM-DD", required=True),
                "description": st.column_config.TextColumn("Description", required=True, width="large"),
                "category": st.column_config.SelectboxColumn(
                    "Category",
                    options=[c.value for c in IncomeCategory],
                    required=True,
                ),
                "status": st.column_config.SelectboxColumn(
                    "Certainty",
                    options=[s.value for s in IncomeStatus],
                    required=True,
                ),
                "amount_lkr": st.column_config.NumberColumn(
                    "Amount (Rs.)",
                    min_value=0.0,
                    step=50.0,
                    format="%.2f",
                    required=True,
                ),
                "earner": st.column_config.TextColumn("Earner"),
            },
        )

        st.markdown("**Expenses & Existing Debt Obligations**")
        edited_expense_df = st.data_editor(
            st.session_state["base_expense_df"],
            num_rows="dynamic",
            hide_index=True,
            width="stretch",
            key=f"expense_editor_{version}",
            column_config={
                "event_id": st.column_config.TextColumn("ID", required=True),
                "event_date": st.column_config.DateColumn("Date", format="YYYY-MM-DD", required=True),
                "description": st.column_config.TextColumn("Description", required=True, width="large"),
                "category": st.column_config.SelectboxColumn(
                    "Category",
                    options=[c.value for c in ExpenseCategory],
                    required=True,
                ),
                "classification": st.column_config.SelectboxColumn(
                    "Type",
                    options=[c.value for c in ExpenseClassification],
                    required=True,
                ),
                "amount_lkr": st.column_config.NumberColumn(
                    "Amount (Rs.)",
                    min_value=0.0,
                    step=50.0,
                    format="%.2f",
                    required=True,
                ),
            },
        )

    # Validate baseline scenario from UI inputs
    try:
        baseline_scenario = _build_scenario_from_ui(
            default_scenario=default_scenario,
            start_date=start_date_input,
            end_date=end_date_input,
            opening_balance_raw=opening_balance_input,
            income_df=edited_income_df,
            expense_df=edited_expense_df,
        )
    except (ValueError, TypeError) as exc:
        st.error(f"Input validation error: {exc}")
        if st.button("Reset Scenario", key="reset_on_error"):
            _reset_to_default(default_scenario)
        st.stop()

    baseline_calendar = build_weekly_cashflow_calendar(baseline_scenario)
    baseline_summary = calculate_scenario_summary(baseline_scenario, baseline_calendar)

    # Resolve active shock scenario (if shock mode is enabled)
    uncertain_daily_wages = [
        inc
        for inc in sorted(baseline_scenario.income_events, key=lambda e: (e.event_date, e.event_id))
        if inc.status is IncomeStatus.UNCERTAIN and inc.category is IncomeCategory.DAILY_WAGE
    ]
    default_two_ids = select_default_rainy_day_income_ids(baseline_scenario, count=2)
    default_clinic_date = (
        uncertain_daily_wages[1].event_date
        if len(uncertain_daily_wages) >= 2
        else baseline_scenario.start_date
    )

    stored_shock_ids = st.session_state.get(f"shock_income_ids_{version}", default_two_ids)
    valid_wage_ids = {inc.event_id for inc in uncertain_daily_wages}
    effective_shock_ids = [eid for eid in stored_shock_ids if eid in valid_wage_ids]
    effective_clinic_date = st.session_state.get(f"clinic_date_{version}", default_clinic_date)
    if not (baseline_scenario.start_date <= effective_clinic_date <= baseline_scenario.end_date):
        effective_clinic_date = baseline_scenario.start_date
    effective_clinic_amount = _coerce_ui_decimal(
        st.session_state.get(f"clinic_amount_{version}", float(DEFAULT_CLINIC_AMOUNT)),
        "Clinic expense amount",
    )

    shock_comparison: ShockComparison | None = None
    active_shocked_scenario: FinancialScenario | None = None
    shock_change_record = None
    if st.session_state["shock_active"] and len(effective_shock_ids) == 2:
        active_shocked_scenario, shock_change_record = apply_rainy_days_and_clinic_shock(
            baseline_scenario,
            removed_income_ids=effective_shock_ids,
            clinic_expense_date=effective_clinic_date,
            clinic_expense_amount=effective_clinic_amount,
            include_clinic_expense=True,
        )
        shock_comparison = compare_scenarios(
            baseline_scenario, active_shocked_scenario, shock_change_record
        )

    is_shock_active = active_shocked_scenario is not None
    active_scenario = active_shocked_scenario if is_shock_active else baseline_scenario
    active_scenario_label = (
        "Shocked Scenario (2 Rainy Days + Clinic Expense)"
        if is_shock_active
        else "Baseline Scenario (Unshocked)"
    )
    active_calendar = (
        list(shock_comparison.shocked_calendar) if shock_comparison is not None else baseline_calendar
    )
    active_summary = (
        shock_comparison.shocked_summary if shock_comparison is not None else baseline_summary
    )

    # Resolve optional loan terms stored in session state for accurate status strip fingerprint
    provide_terms_state = bool(st.session_state.get(f"provide_loan_terms_{version}", False))
    loan_terms_provided: dict[str, Any] | None = None
    if provide_terms_state:
        loan_terms_provided = {
            "principal_lkr": str(
                _coerce_ui_decimal(
                    st.session_state.get(f"loan_principal_{version}", 20000.0), "Loan principal"
                )
            ),
            "repayment_amount_lkr": str(
                _coerce_ui_decimal(
                    st.session_state.get(f"loan_instalment_{version}", 1250.0), "Instalment amount"
                )
            ),
            "repayment_frequency": st.session_state.get(f"loan_freq_{version}", "weekly"),
            "first_repayment_date": st.session_state.get(
                f"loan_first_date_{version}", active_scenario.start_date
            ).isoformat(),
            "lost_workdays": 2,
        }

    current_fingerprint = compute_scenario_fingerprint(
        active_scenario,
        scenario_label=active_scenario_label,
        loan_terms_provided=loan_terms_provided,
    )
    demo_view: DemoAnalysisView = resolve_demo_analysis_view(
        st.session_state,
        current_fingerprint=current_fingerprint,
        is_shock_active=is_shock_active,
    )

    # Top status strip (visible immediately below header/Section 1 kickoff)
    scenario_pill = (
        '<span class="pill pill-shock">Active: Shocked Scenario (2 Rainy Days + Clinic)</span>'
        if is_shock_active
        else '<span class="pill pill-baseline">Active: Baseline Scenario</span>'
    )
    model_pill = (
        f'<span class="pill">Gemini Model: {ai_status.model_name}</span>'
        if ai_status.is_available
        else '<span class="pill pill-pending">Gemini: API Key Required</span>'
    )
    approval_pill = _approval_pill_html(
        demo_view.current_approval_status if demo_view.current_result is not None else None
    )
    st.markdown(
        f'<div class="status-strip">{scenario_pill}{model_pill}{approval_pill}</div>',
        unsafe_allow_html=True,
    )

    # Section 1 overview cards
    ov1, ov2, ov3, ov4 = st.columns(4)
    ov1.metric(
        "Confirmed Income",
        format_lkr(active_summary.total_confirmed_income),
        delta=f"Uncertain: {format_lkr(active_summary.total_uncertain_income)}",
        delta_color="off",
        help="Conservative cashflow uses confirmed income only. Uncertain daily wages are never treated as guaranteed.",
    )
    ov2.metric(
        "Essential Expenses",
        format_lkr(active_summary.total_essential_expenses),
        delta=f"Nonessential: {format_lkr(active_summary.total_nonessential_expenses)}",
        delta_color="off",
    )
    ov3.metric(
        "Existing Debt Due",
        format_lkr(active_summary.total_debt_obligations),
        delta="Fixed weekly microfinance",
        delta_color="off",
    )
    loan_under_eval = household.loan_under_consideration
    ov4.metric(
        "Possible Business Loan",
        format_lkr(loan_under_eval.principal_amount) if loan_under_eval else "None",
        delta="Not active (terms missing)",
        delta_color="off",
        help="Loan under consideration only — excluded from weekly cashflow until written repayment terms are verified.",
    )

    prof_left, prof_right = st.columns([3, 2])
    with prof_left:
        st.markdown(f"**{household.household_name}** — {household.location}")
        member_rows = [
            {
                "Member": m.name,
                "Role": m.role.replace("_", " ").title(),
                "Age": m.age,
                "Household Context": m.notes,
            }
            for m in household.members
        ]
        st.dataframe(pd.DataFrame(member_rows), hide_index=True, width="stretch")
    with prof_right:
        st.markdown("**Income Uncertainty & Loan Context**")
        for src_item in household.income_sources_summary:
            st.markdown(f"- {src_item}")
        if loan_under_eval is not None:
            st.markdown(
                f"""
                <div class="callout-card">
                    <strong>{loan_under_eval.description}</strong><br/>
                    Purpose: {loan_under_eval.purpose}<br/>
                    <small>{loan_under_eval.notes}</small>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.divider()

    # =========================================================================
    # SECTION 2: FINANCIAL HEALTH
    # =========================================================================
    _render_section_header(
        2,
        f"Financial Health — {active_scenario_label}",
        "Deterministic Decimal cashflow calendar separating confirmed cash from uncertain daily wages.",
    )

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Opening Cash", format_lkr(active_summary.opening_balance))
    m2.metric(
        "Conservative Ending",
        format_lkr(active_summary.conservative_ending_balance),
        help="Opening cash + CONFIRMED income only - dated expenses & debt.",
    )
    m3.metric(
        "Optimistic Ending",
        format_lkr(active_summary.optimistic_ending_balance),
        help="Includes both confirmed and uncertain expected income.",
    )
    active_cliff_label = (
        active_summary.first_liquidity_cliff_date.isoformat()
        if active_summary.first_liquidity_cliff_date is not None
        else "None"
    )
    m4.metric(
        "First Liquidity Cliff",
        active_cliff_label,
        delta=f"{len(active_summary.conservative_negative_days)} negative day(s)",
        delta_color="inverse" if active_summary.has_liquidity_cliff else "off",
    )

    _render_balance_chart(
        active_calendar,
        title=f"{active_scenario_label}: Daily Closing Balance (Rs.)",
    )

    st.markdown("#### Daily Cashflow Calendar")
    st.dataframe(
        _calendar_to_display_df(active_calendar),
        hide_index=True,
        width="stretch",
    )

    _render_shortage_warnings(
        active_summary,
        f"Liquidity Cliffs & Unfunded Essential Payments ({active_scenario_label})",
    )

    with st.expander("Accounting Methodology & Same-Day Transaction Ordering", expanded=False):
        st.write(SAME_DAY_ORDERING_CONVENTION)

    st.divider()

    # =========================================================================
    # SECTION 3: FOUR-AGENT AI ANALYSIS
    # =========================================================================
    _render_section_header(
        3,
        "Four-Agent AI Analysis",
        f"Real Gemini tool calling via LangGraph on '{active_scenario_label}' using model '{ai_status.model_name}'.",
    )

    if not ai_status.is_available:
        st.info(
            f"{ai_status.reason} Copy `.env.example` to `.env`, add `GEMINI_API_KEY`, and reload. "
            "No simulated or fake AI analysis is ever shown."
        )

    with st.expander(
        "Optional: Supply Explicit Repayment Terms to Stress-Test the Rs. 20,000 Loan",
        expanded=False,
    ):
        st.caption(
            "Leave unchecked to demonstrate Debt Shield honestly reporting missing loan terms "
            "without fabricating interest rates or repayment schedules."
        )
        provide_terms_cb = st.checkbox(
            "Provide explicit loan repayment terms for Debt Shield stress testing",
            value=provide_terms_state,
            key=f"provide_loan_terms_{version}",
        )
        if provide_terms_cb:
            lt1, lt2, lt3, lt4 = st.columns(4)
            with lt1:
                st.number_input(
                    "Loan Principal (Rs.)",
                    min_value=100.0,
                    step=500.0,
                    value=20000.0,
                    format="%.2f",
                    key=f"loan_principal_{version}",
                )
            with lt2:
                st.number_input(
                    "Instalment Amount (Rs.)",
                    min_value=50.0,
                    step=50.0,
                    value=1250.0,
                    format="%.2f",
                    key=f"loan_instalment_{version}",
                )
            with lt3:
                st.selectbox(
                    "Repayment Frequency",
                    options=["weekly", "biweekly", "monthly", "daily"],
                    index=0,
                    key=f"loan_freq_{version}",
                )
            with lt4:
                st.date_input(
                    "First Repayment Date",
                    value=active_scenario.start_date,
                    key=f"loan_first_date_{version}",
                )

    ai_btn_col, ai_info_col = st.columns([2, 3])
    with ai_btn_col:
        run_ai_clicked = st.button(
            "Run 4-Agent AI Analysis",
            type="primary",
            disabled=(not ai_status.is_available) or bool(st.session_state.get("ai_is_running")),
            width="stretch",
        )
    with ai_info_col:
        st.caption(
            f"Target scenario: **{active_scenario_label}** | Model: `{ai_status.model_name}` | "
            f"Scenario ID hash: `{current_fingerprint}`"
        )

    if run_ai_clicked:
        _execute_ai_workflow(
            active_scenario=active_scenario,
            baseline_scenario=baseline_scenario,
            shocked_scenario=active_shocked_scenario,
            active_scenario_label=active_scenario_label,
            loan_terms_provided=loan_terms_provided,
            ai_status=ai_status,
        )

    if st.session_state.get("ai_last_error"):
        st.error(st.session_state["ai_last_error"])

    if demo_view.stale_notice:
        st.warning(demo_view.stale_notice)
        if demo_view.stale_result is not None:
            with st.expander(
                f"View Preserved Stale / Historical Analysis ({demo_view.stale_result['scenario_label']})",
                expanded=False,
            ):
                st.caption(
                    "Labeled strictly as historical/stale — does NOT reflect the current scenario."
                )
                for key in ("financial_twin", "debt_shield", "community_bridge", "family_advocate_critic"):
                    fnd = demo_view.stale_result["agent_findings"].get(key)
                    if fnd:
                        st.markdown(
                            f"**{fnd['role_title']}:** "
                            f"{sanitize_narrative_text(fnd['summary'], fallback='Historical findings recorded.')}"
                        )

    if demo_view.current_result is not None:
        valid_result = demo_view.current_result
        st.markdown(
            f"""
            <div class="callout-card">
                <strong>Analyzed Scenario:</strong> {valid_result['scenario_label']} &nbsp;|&nbsp;
                <strong>Gemini Model:</strong> <code>{valid_result['model_name']}</code> &nbsp;|&nbsp;
                <strong>Revisions:</strong> {valid_result['revision_count']} / {valid_result['max_revisions']} &nbsp;|&nbsp;
                <strong>Tool Invocations:</strong> {len(valid_result['tool_invocations'])}
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Concise 2x2 findings grid — Main Conclusion First
        agent_order = [
            ("financial_twin", "Agent A: Financial Twin"),
            ("debt_shield", "Agent B: Debt Shield"),
            ("community_bridge", "Agent C: Community Bridge"),
            ("family_advocate_critic", "Agent D: Family Advocate & Critic"),
        ]
        r1 = st.columns(2)
        r2 = st.columns(2)
        for col, (agent_key, fallback_title) in zip([r1[0], r1[1], r2[0], r2[1]], agent_order):
            finding = valid_result["agent_findings"].get(agent_key)
            with col:
                with st.container(border=True):
                    if finding is None:
                        st.markdown(f"**{fallback_title}**")
                        st.caption("No output recorded.")
                        continue
                    st.markdown(f"**{finding['role_title']}**")
                    clean_summary = sanitize_narrative_text(
                        finding["summary"],
                        fallback="Structured findings recorded from deterministic tool outputs.",
                    )
                    st.markdown(
                        f'<div class="conclusion-box"><strong>Main conclusion:</strong> {clean_summary}</div>',
                        unsafe_allow_html=True,
                    )
                    key_pts = dedupe_strings(
                        [sanitize_narrative_text(p) for p in finding["key_points"]]
                    )
                    distinct_risks = consolidate_warnings(
                        [sanitize_narrative_text(r) for r in finding["risks"]]
                    )
                    distinct_unc = consolidate_warnings(
                        [sanitize_narrative_text(u) for u in finding["uncertainties"]]
                    )
                    if key_pts:
                        st.markdown("**Grounded Findings & Calculations**")
                        for pt in key_pts:
                            st.markdown(f"- {pt}")
                    if distinct_risks:
                        st.markdown("**Distinct Risks / Flags**")
                        for rk in distinct_risks:
                            st.markdown(f"- {rk}")
                    if distinct_unc:
                        st.markdown("**Unresolved Uncertainties**")
                        for un in distinct_unc:
                            st.markdown(f"- {un}")
                    tools_str = (
                        ", ".join(finding["tools_called"]) if finding["tools_called"] else "None"
                    )
                    st.caption(f"Tools called: `{tools_str}`")

        # Visible Critic Revision Loop
        _render_critic_revision_loop(valid_result)

        # Tool-Use & Decision Trace
        with st.expander("Tool-Use & LangGraph Decision Trace", expanded=False):
            st.markdown("**LangGraph Node Transitions & Critic Routing**")
            trace_df = pd.DataFrame(
                [
                    {
                        "Step": ev["step"],
                        "Node": ev["node"],
                        "Round": ev["revision_round"],
                        "Event": ev["event_type"],
                        "Summary": ev["summary"],
                    }
                    for ev in valid_result["trace_events"]
                ]
            )
            st.dataframe(trace_df, hide_index=True, width="stretch")

            st.markdown("**Deterministic Tool Invocations**")
            if valid_result["tool_invocations"]:
                tool_df = pd.DataFrame(
                    [
                        {
                            "Step": inv["step"],
                            "Agent": inv["agent"],
                            "Round": inv["revision_round"],
                            "Tool": inv["tool_name"],
                            "Arguments": json.dumps(inv["arguments"]),
                            "Status": inv["status"],
                            "Result Summary": inv["result_summary"],
                        }
                        for inv in valid_result["tool_invocations"]
                    ]
                )
                st.dataframe(tool_df, hide_index=True, width="stretch")
            else:
                st.caption("No tool invocations recorded.")

    st.divider()

    # =========================================================================
    # SECTION 4: FAMILY RECOVERY AGREEMENT
    # =========================================================================
    _render_section_header(
        4,
        "Family Recovery Agreement",
        "Actionable draft plan requiring explicit family review and approval — never pre-approved.",
    )

    if demo_view.current_result is not None and demo_view.current_result.get("final_agreement"):
        current_agreement = demo_view.current_result["final_agreement"]
        assert current_agreement is not None

        st.markdown(
            f'<div class="honest-note">{current_agreement["approval_requirement_notice"]}</div>',
            unsafe_allow_html=True,
        )
        _render_agreement_card_body(current_agreement, is_historical=False)

        # Explicit Approve / Reject buttons tied strictly to current_fingerprint
        app_col1, app_col2, app_col3 = st.columns([1.2, 1.2, 2.6])
        with app_col1:
            if st.button("Approve Agreement", type="primary", width="stretch"):
                set_agreement_approval(st.session_state, current_fingerprint, "APPROVED_BY_FAMILY")
                st.rerun()
        with app_col2:
            if st.button("Reject Agreement", width="stretch"):
                set_agreement_approval(st.session_state, current_fingerprint, "REJECTED_BY_FAMILY")
                st.rerun()
        with app_col3:
            if demo_view.current_approval_status == "APPROVED_BY_FAMILY":
                st.success(
                    f"Approved by Family for '{current_agreement['scenario_label']}'."
                )
            elif demo_view.current_approval_status == "REJECTED_BY_FAMILY":
                st.error(
                    f"Rejected by Family for '{current_agreement['scenario_label']}' — no actions accepted."
                )
            else:
                st.info(
                    "Pending Family Approval — click 'Approve Agreement' or 'Reject Agreement' to record your decision."
                )
    else:
        if is_shock_active:
            st.info(
                "No Family Recovery Agreement has been generated for the **Shocked Scenario** yet. "
                "Click **'Reanalyze Shocked Scenario'** in Section 5 (or **'Run 4-Agent AI Analysis'** "
                "in Section 3) to generate a revised agreement."
            )
            if (
                demo_view.historical_baseline_result is not None
                and demo_view.historical_baseline_result.get("final_agreement")
            ):
                with st.expander(
                    "Preserved Historical Baseline Agreement (Not Valid for Shocked Scenario)",
                    expanded=False,
                ):
                    _render_agreement_card_body(
                        demo_view.historical_baseline_result["final_agreement"],
                        is_historical=True,
                    )
        else:
            st.info(
                "Run **'Run 4-Agent AI Analysis'** in Section 3 to generate a proposed Weekly "
                "Family Recovery Agreement."
            )

    st.divider()

    # =========================================================================
    # SECTION 5: SHOCK AND ADAPTATION
    # =========================================================================
    _render_section_header(
        5,
        "Shock and Adaptation",
        "Simulate two rainy days plus an unexpected clinic expense, compare financial impact, and re-run all four agents to adapt the agreement.",
    )

    wage_options = {
        inc.event_id: f"{inc.event_id} | {inc.event_date.isoformat()} — {inc.description} ({format_lkr(inc.amount)})"
        for inc in uncertain_daily_wages
    }

    cfg1, cfg2, cfg3 = st.columns([2, 1, 1])
    with cfg1:
        selected_shock_ids = st.multiselect(
            "Uncertain daily-wage events lost to rain (select 2)",
            options=list(wage_options.keys()),
            default=default_two_ids,
            format_func=lambda eid: wage_options[eid],
            key=f"shock_income_ids_{version}",
        )
    with cfg2:
        clinic_date_input = st.date_input(
            "Clinic Expense Date",
            value=default_clinic_date,
            min_value=baseline_scenario.start_date,
            max_value=baseline_scenario.end_date,
            key=f"clinic_date_{version}",
        )
    with cfg3:
        clinic_amount_input = st.number_input(
            "Clinic Expense (Rs.)",
            min_value=0.0,
            step=100.0,
            value=float(DEFAULT_CLINIC_AMOUNT),
            format="%.2f",
            key=f"clinic_amount_{version}",
        )

    primary_btn_col1, primary_btn_col2 = st.columns(2)
    with primary_btn_col1:
        simulate_clicked = st.button(
            "Simulate Shock: 2 Rainy Days + Clinic Expense",
            type="primary" if not is_shock_active else "secondary",
            width="stretch",
        )
    with primary_btn_col2:
        reanalyze_clicked = st.button(
            "Reanalyze Shocked Scenario",
            type="primary" if is_shock_active else "secondary",
            disabled=(not is_shock_active)
            or (not ai_status.is_available)
            or bool(st.session_state.get("ai_is_running")),
            width="stretch",
        )

    sec_btn_col1, sec_btn_col2, _ = st.columns([1.2, 1.2, 2.6])
    with sec_btn_col1:
        reset_clicked = st.button("Reset Scenario", width="stretch")
    with sec_btn_col2:
        clear_shock_clicked = False
        if st.session_state["shock_active"]:
            clear_shock_clicked = st.button("Clear Shock", width="stretch")

    if reset_clicked:
        _reset_to_default(default_scenario)

    if clear_shock_clicked:
        st.session_state["shock_active"] = False
        st.rerun()

    if simulate_clicked:
        if len(selected_shock_ids) != 2:
            st.warning(
                "Please select exactly 2 uncertain daily-wage income events to simulate 2 rainy days."
            )
        else:
            st.session_state["shock_active"] = True
            st.rerun()

    if reanalyze_clicked and is_shock_active and active_shocked_scenario is not None:
        _execute_ai_workflow(
            active_scenario=active_shocked_scenario,
            baseline_scenario=baseline_scenario,
            shocked_scenario=active_shocked_scenario,
            active_scenario_label=active_scenario_label,
            loan_terms_provided=loan_terms_provided,
            ai_status=ai_status,
        )

    if st.session_state["shock_active"] and len(selected_shock_ids) != 2:
        st.warning(
            "Shock simulation paused because fewer or more than 2 uncertain daily-wage events are selected."
        )

    if shock_comparison is not None and shock_change_record is not None:
        st.markdown("#### Exact Events Changed by Shock")
        ev1, ev2 = st.columns(2)
        with ev1:
            st.markdown("**Removed Uncertain Daily-Wage Events (2 Rainy Days)**")
            for removed in shock_change_record.removed_income_events:
                st.markdown(
                    f"- `{removed.event_id}` · **{removed.event_date.isoformat()}** · "
                    f"{removed.description} — **-{format_lkr(removed.amount)}** *(uncertain)*"
                )
        with ev2:
            st.markdown("**Added Synthetic Shock Expense**")
            for added in shock_change_record.added_expense_events:
                st.markdown(
                    f"- `{added.event_id}` · **{added.event_date.isoformat()}** · "
                    f"{added.description} — **+{format_lkr(added.amount)}** *(essential medicine)*"
                )

        removed_unc_total = shock_change_record.removed_uncertain_income_total
        added_clinic_total = shock_change_record.added_expense_total
        st.markdown(
            f"""
            <div class="honest-note">
                <strong>Why Conservative and Optimistic Balances Change Differently:</strong><br/>
                Conservative cashflow <strong>already excludes uncertain daily wages</strong>, so
                removing the two rainy-day wage events ({format_lkr(removed_unc_total)}) contributes
                <strong>Rs. 0.00</strong> to the conservative delta. The conservative change of
                <strong>{format_signed_lkr(shock_comparison.conservative_ending_delta)}</strong>
                comes strictly from the added clinic expense ({format_lkr(added_clinic_total)}).<br/>
                Optimistic cashflow originally counted those wages, so it shifts by
                <strong>{format_signed_lkr(shock_comparison.optimistic_ending_delta)}</strong>
                ({format_signed_lkr(-removed_unc_total)} lost uncertain wages +
                {format_signed_lkr(-added_clinic_total)} clinic expense).
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("#### Baseline vs. Shocked Financial Comparison")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric(
            "Shocked Conservative",
            format_lkr(shock_comparison.shocked_summary.conservative_ending_balance),
            delta=format_signed_lkr(shock_comparison.conservative_ending_delta),
        )
        c2.metric(
            "Shocked Optimistic",
            format_lkr(shock_comparison.shocked_summary.optimistic_ending_balance),
            delta=format_signed_lkr(shock_comparison.optimistic_ending_delta),
        )
        c3.metric(
            "Shocked Lowest Cash",
            format_lkr(shock_comparison.shocked_summary.lowest_conservative_balance),
            delta=format_signed_lkr(shock_comparison.lowest_conservative_delta),
        )
        base_cliff = (
            baseline_summary.first_liquidity_cliff_date.isoformat()
            if baseline_summary.first_liquidity_cliff_date is not None
            else "None"
        )
        shock_cliff = (
            shock_comparison.shocked_summary.first_liquidity_cliff_date.isoformat()
            if shock_comparison.shocked_summary.first_liquidity_cliff_date is not None
            else "None"
        )
        c4.metric(
            "Shocked First Cliff",
            shock_cliff,
            delta=f"Baseline: {base_cliff}",
            delta_color="off",
        )

        comp_rows = []
        for b_day, s_day in zip(
            shock_comparison.baseline_calendar, shock_comparison.shocked_calendar
        ):
            cons_diff = s_day.conservative_closing_balance - b_day.conservative_closing_balance
            opt_diff = s_day.optimistic_closing_balance - b_day.optimistic_closing_balance
            comp_rows.append(
                {
                    "Date": f"{b_day.day_date.isoformat()} ({b_day.day_date.strftime('%a')})",
                    "Baseline Cons.": format_lkr(b_day.conservative_closing_balance),
                    "Shocked Cons.": format_lkr(s_day.conservative_closing_balance),
                    "Cons. Change": format_signed_lkr(cons_diff),
                    "Baseline Opt.": format_lkr(b_day.optimistic_closing_balance),
                    "Shocked Opt.": format_lkr(s_day.optimistic_closing_balance),
                    "Opt. Change": format_signed_lkr(opt_diff),
                }
            )
        st.dataframe(pd.DataFrame(comp_rows), hide_index=True, width="stretch")
        _render_comparison_chart(shock_comparison)

        # Before / After Agreement Comparison when both Baseline (historical) and Shocked (current) exist
        if (
            demo_view.historical_baseline_result is not None
            and demo_view.current_result is not None
        ):
            _render_agreement_comparison(
                demo_view.historical_baseline_result,
                demo_view.current_result,
                baseline_approval=demo_view.historical_baseline_approval_status,
                shocked_approval=demo_view.current_approval_status,
            )
        elif demo_view.historical_baseline_result is not None:
            st.info(
                "Historical Baseline Agreement is preserved. Click **'Reanalyze Shocked Scenario'** "
                "above to run all four agents on the shocked scenario and view the before/after "
                "agreement comparison."
            )

    st.divider()
    st.caption(
        "Our Family Plan · Team Fight Club (InnoHack '26) · Milestone 3 · "
        "Synthetic demonstration data — not financial advice."
    )


if __name__ == "__main__":
    main()
