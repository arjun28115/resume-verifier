"""Smart Resume Verification Tool - Streamlit front end.

QUICK START
-----------
    pip install -r requirements.txt
    export ANTHROPIC_API_KEY=sk-ant-...     # or paste it into the sidebar
    streamlit run app.py

The key can come from (in priority order):
  1. the sidebar text field (session only, never written to disk)
  2. the ANTHROPIC_API_KEY / OPENAI_API_KEY environment variable
  3. .streamlit/secrets.toml
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd
import streamlit as st

from verifier.engine import VerificationSettings, verify_batch
from verifier.fetch import FetchError, fetch_document, fetch_many, parse_url_list
from verifier.llm import ANTHROPIC_MODELS, EFFORT_LEVELS, OPENAI_MODELS, build_verifier
from verifier.pdf_utils import parse_upload
from verifier.schema import ResumeReport, Status
from verifier.scoring import band_for_report, describe_score

st.set_page_config(page_title="Smart Resume Verification Tool",
                   page_icon="🧾", layout="wide")

STATUS_STYLE = {
    Status.PASS: ("✅", "#1a7f37"),
    Status.WARNING: ("⚠️", "#9a6700"),
    Status.FAIL: ("❌", "#cf222e"),
}
SEVERITY_ICON = {"high": "🔴", "medium": "🟠", "low": "🔵"}


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

SECRET_PATHS = (
    Path(".streamlit/secrets.toml"),
    Path.home() / ".streamlit/secrets.toml",
)


def secret(name: str) -> str:
    """Read a key from the environment, falling back to secrets.toml.

    The existence check matters: touching ``st.secrets`` with no secrets file
    makes Streamlit print a "No secrets files found" warning into the sidebar,
    which looks like an error to the user.
    """
    if os.environ.get(name):
        return os.environ[name]
    if not any(p.exists() for p in SECRET_PATHS):
        return ""
    try:
        return st.secrets.get(name, "")
    except Exception:  # noqa: BLE001
        return ""


@st.cache_data(show_spinner=False)
def cached_parse(name: str, data: bytes):
    """Cache extraction by file content so re-runs don't re-parse PDFs."""
    return parse_upload(name, data)


@st.cache_data(show_spinner=False, ttl=3600)
def cached_fetch(url: str):
    """Cache downloads so a rerun (any widget change) doesn't re-hit the network."""
    return fetch_document(url)


@st.cache_data(show_spinner=False, ttl=3600)
def cached_fetch_many(urls: tuple[str, ...]):
    return fetch_many(list(urls))


def show_pdf(data: bytes, height: int = 620, key: str | None = None) -> None:
    """Render a PDF inline.

    st.pdf needs the `streamlit-pdf` extra; if it is missing we fall back to an
    <embed> of a data: URI, which every desktop browser renders natively. The
    viewer must never be the thing that breaks a verification run, so any
    failure degrades to a note rather than an exception.
    """
    try:
        st.pdf(data, height=height, key=key)
        return
    except Exception:  # noqa: BLE001 - missing extra, or an unreadable PDF
        pass
    try:
        import base64

        b64 = base64.b64encode(data).decode("ascii")
        st.components.v1.html(
            f'<embed src="data:application/pdf;base64,{b64}#view=FitH" '
            f'type="application/pdf" width="100%" height="{height}px" />',
            height=height,
        )
    except Exception:  # noqa: BLE001
        st.caption("Inline preview unavailable — use the download button below.")


def _short(name: str, keep: int = 34) -> str:
    """Shorten the long SPO filenames, which differ only in a trailing hash."""
    stem = name.rsplit(".", 1)[0]
    return name if len(stem) <= keep else f"{stem[:keep - 12]}…{stem[-8:]}"


def reports_to_dataframe(reports: list[ResumeReport]) -> pd.DataFrame:
    frame = pd.DataFrame([
        {
            "Status": (f"{STATUS_STYLE[r.status][0]} {r.status.value}"
                       + ("*" if r.score_provisional and r.status is Status.PASS else "")),
            "Match": f"{r.match_score}{'*' if r.score_provisional else ''}",
            "Identity": ("❌ different person" if r.identity_match is False
                         else "✅ same" if r.identity_match else "— unconfirmed"),
            "File": _short(r.filename),
            "Pages": r.page_count or "-",
            "Phone": "❌ found" if r.has_phone else "✅ clean",
            "JEE/AIR": "❌ found" if r.has_jee else "✅ clean",
            "Coverage": ("—" if r.identity_match is False
                         else f"{r.coverage:.0%}"),
            "Critical": r.blocking_count,
            "Warnings": r.warning_count,
            "Summary": r.headline(),
        }
        for r in reports
    ])
    return frame


def render_report(report: ResumeReport) -> None:
    """One expandable card per tailored resume."""
    icon, colour = STATUS_STYLE[report.status]
    label = f"{icon}  {report.filename}  —  {report.headline()}"

    with st.expander(label, expanded=report.status is not Status.PASS):
        cols = st.columns(5)
        cols[0].metric("Match /100", report.match_score,
                       help="How much of this resume is verified against the "
                            "master. Rephrasing never costs points.")
        short_verdict = {"WARNING": "WARN"}.get(report.status.value,
                                                 report.status.value)
        cols[1].metric(
            "Verdict",
            short_verdict + ("*" if report.score_provisional
                             and report.status is Status.PASS else ""))
        cols[2].metric("Pages", report.page_count or "?")
        cols[3].metric("Critical", report.blocking_count)
        cols[4].metric("Warnings", report.warning_count)

        band = band_for_report(report)
        st.progress(
            report.match_score / 100,
            text=f"{band} — {describe_score(report.match_score, report.score_lines)}",
        )
        if report.score_provisional:
            st.caption("\\* Provisional: no LLM cross-reference ran, so hallucinated "
                       "skills and altered wording are not reflected in this score.")

        with st.popover("How this score was calculated"):
            st.markdown("Starts at **100**; every deduction is itemised below.")
            for line in report.score_lines:
                if line.delta:
                    st.markdown(f"- **{line.delta:+d}** — {line.label}")
                else:
                    st.markdown(f"- {line.label}")
            st.markdown(f"**Total: {report.match_score}/100**")

        if report.identity_match is False:
            st.error(f"🚨 **Wrong candidate.** {report.identity_reason}")
        elif report.identity_match is None and report.identity_reason:
            st.info(f"ℹ️ Identity not confirmed — {report.identity_reason}")

        if report.error:
            st.error(f"Verification incomplete — {report.error}")
        for warning in report.parse_warnings:
            st.warning(warning)

        # --- Hard exclusions -------------------------------------------
        if report.has_phone or report.has_jee or not report.within_page_limit:
            st.markdown("##### 🚫 Strict exclusions")
            if report.has_phone:
                st.error("**Contains a phone number:** "
                         + ", ".join(f"`{h}`" for h in report.phone_hits))
            if report.has_jee:
                st.error("**Mentions a JEE rank:** "
                         + ", ".join(f"`{h}`" for h in report.jee_hits))
            if not report.within_page_limit:
                st.error(f"**Over the page limit:** {report.page_count} pages "
                         f"(limit {report.page_limit}).")

        if report.overall_assessment:
            st.markdown("##### 📋 Assessment")
            st.write(report.overall_assessment)

        # --- Findings ---------------------------------------------------
        content_findings = [f for f in report.findings if f.category != "OTHER"]
        if content_findings:
            st.markdown("##### 🔍 Flagged points")
            for f in content_findings:
                st.markdown(
                    f"{SEVERITY_ICON.get(f.severity, '⚪')} **{f.category}** "
                    f"· _{f.severity}_"
                )
                st.markdown(f"> {f.quote}")
                st.markdown(f"**Issue:** {f.issue}")
                st.markdown(f"**Master resume says:** {f.master_evidence}")
                st.markdown(f"**Fix:** {f.suggested_fix}")
                st.divider()
        elif not report.error and report.llm_used:
            st.success("No fabricated metrics, skills, or experiences detected.")

        # --- Supporting detail ------------------------------------------
        if report.identity_match is not False and report.unsupported_lines:
            st.markdown("##### 📄 Lines with no counterpart in the master")
            st.caption(
                f"Content coverage {report.coverage:.0%}. These lines had no close "
                "textual match in the master. This compares **wording**, so a "
                "heavily reworded but perfectly honest bullet lands here too, as "
                "does a word the PDF broke across a line. These are pointers to "
                "check, not verdicts: they do **not** affect the match score, but "
                "they do hold the verdict at ⚠️ WARNING until a human or the LLM "
                "has judged them."
            )
            for line in report.unsupported_lines[:15]:
                st.markdown(f"- {line}")
            if len(report.unsupported_lines) > 15:
                st.caption(f"…and {len(report.unsupported_lines) - 15} more.")

        if report.annotated_pdf:
            st.markdown("##### 📑 Review")
            if report.has_highlights:
                st.caption(
                    "🔴 no counterpart in the master · 🟠 metric with no match. "
                    "Hover a highlight for the reason."
                )
            else:
                st.caption("Nothing to highlight on this resume.")

            master_pdf = st.session_state.get("master_pdf")
            if master_pdf:
                left, right = st.columns(2)
                with left:
                    st.caption("**Master resume** (source of truth)")
                    show_pdf(master_pdf, key=f"master_{report.filename}")
                with right:
                    st.caption(f"**{_short(report.filename)}**"
                               + (" — highlighted" if report.has_highlights else ""))
                    show_pdf(report.annotated_pdf, key=f"tail_{report.filename}")
            else:
                show_pdf(report.annotated_pdf, key=f"tail_{report.filename}")

            if report.has_highlights:
                st.download_button(
                    "⬇️ Save the highlighted PDF",
                    data=report.annotated_pdf,
                    file_name=f"annotated_{report.filename}",
                    mime="application/pdf",
                    key=f"annot_{report.filename}",
                )

        if report.rank_mentions:
            st.markdown("##### 🏅 Rank claims (allowed — not JEE)")
            st.caption(
                "All India Rank / AIR claims outside a JEE context are legitimate "
                "achievements and do not fail the resume. Listed so you can "
                "eyeball them."
            )
            for mention in report.rank_mentions:
                st.markdown(f"- {mention}")

        left, right = st.columns(2)
        with left:
            if report.unverified_metric_candidates:
                st.markdown("##### 🔢 Metrics with no numeric match in the master")
                st.caption("Raw regex pre-scan — the LLM adjudicates these above.")
                st.code("\n".join(report.unverified_metric_candidates), language="text")
        with right:
            if report.verified_ok:
                st.markdown("##### ✅ Checked and confirmed")
                for note in report.verified_ok:
                    st.markdown(f"- {note}")


# --------------------------------------------------------------------------
# Sidebar - configuration
# --------------------------------------------------------------------------

st.sidebar.title("⚙️ Configuration")

provider = st.sidebar.selectbox("LLM provider", ["Anthropic (Claude)", "OpenAI"])
if provider == "Anthropic (Claude)":
    model = st.sidebar.selectbox("Model", ANTHROPIC_MODELS, index=0)
    env_key = secret("ANTHROPIC_API_KEY")
    key_label = "ANTHROPIC_API_KEY"
else:
    model = st.sidebar.selectbox("Model", OPENAI_MODELS, index=0)
    env_key = secret("OPENAI_API_KEY")
    key_label = "OPENAI_API_KEY"

api_key = st.sidebar.text_input(
    f"{key_label} (leave blank to use the environment)",
    value="", type="password",
    help="Paste a key here for this session only, or export it before launching.",
) or env_key

st.sidebar.caption(
    f"🔑 Key detected in environment: {'yes' if env_key else 'no'}"
)

effort = st.sidebar.select_slider(
    "Reasoning effort", options=EFFORT_LEVELS, value="high",
    help="Claude only. Lower is faster and cheaper for large batches; "
         "'high' or above is recommended for accuracy.",
) if provider == "Anthropic (Claude)" else "high"

st.sidebar.markdown("---")
st.sidebar.subheader("Verification rules")
max_pages = st.sidebar.number_input(
    "Maximum pages allowed", min_value=1, max_value=10, value=2, step=1,
    help="A \"single-pager\" is the short-resume format, not literally one "
         "sheet - two pages is common. Set your own house rule here.")
strict_page_limit = st.sidebar.checkbox(
    "Exceeding the page limit is an auto-fail", value=True,
    help="Off: going over the limit is a warning instead of a failure.")
check_years = st.sidebar.checkbox(
    "Also verify calendar years (2023, 2024…)", value=False,
    help="Off by default: dates are usually noise rather than achievement metrics.")
max_workers = st.sidebar.slider("Parallel resumes", 1, 8, 4)

# Default ON when no key is available: without one, an LLM run can only fail,
# so the app should start in the mode that actually works rather than asking
# the user to find a checkbox after hitting an error.
rule_only = st.sidebar.checkbox(
    "Rule-only mode (no LLM calls)", value=not bool(api_key),
    help="Runs the phone / JEE / page-count / metric and line-level subset checks without an API key. Hallucinated skills still need the LLM.")

st.sidebar.markdown("---")
st.sidebar.subheader("Master resume (source of truth)")

master_mode = st.sidebar.radio(
    "How do you want to provide it?",
    ["Upload a file", "Paste a link", "Paste the text"],
    key="master_mode", horizontal=False,
)

master_file, master_pasted, master_url = None, "", ""
if master_mode == "Upload a file":
    master_file = st.sidebar.file_uploader(
        "Upload the master resume", type=["pdf", "txt", "md"], key="master")
elif master_mode == "Paste a link":
    master_url = st.sidebar.text_input(
        "Public URL to the master resume",
        placeholder="https://example.com/master_resume.pdf",
        help="Must be a direct link to the file. Google Drive and Dropbox share "
             "links are converted to their download form automatically.",
    )
else:
    master_pasted = st.sidebar.text_area(
        "Paste the master resume text", height=200,
        placeholder="Paste the full master resume text if you don't have a PDF.")


# --------------------------------------------------------------------------
# Main area
# --------------------------------------------------------------------------

st.title("🧾 Smart Resume Verification Tool")
st.markdown(
    "Cross-check tailored single-page resumes against your **master resume**. "
    "Rephrasing is allowed — invented metrics, hallucinated experience, phone "
    "numbers and JEE ranks are not."
)

# Resolve the master resume from whichever input was used.
master_doc = None
st.session_state["master_pdf"] = None   # only set when the master is a real PDF
if master_file is not None:
    raw = master_file.getvalue()
    master_doc = cached_parse(master_file.name, raw)
    if master_file.name.lower().endswith(".pdf"):
        st.session_state["master_pdf"] = raw
elif master_url.strip():
    try:
        with st.spinner("Downloading the master resume…"):
            fetched = cached_fetch(master_url.strip())
        master_doc = cached_parse(fetched.name, fetched.data)
        if fetched.name.lower().endswith(".pdf"):
            st.session_state["master_pdf"] = fetched.data
    except FetchError as exc:
        st.error(f"Could not download the master resume — {exc}")
elif master_pasted.strip():
    master_doc = parse_upload("pasted_master.txt", master_pasted.encode("utf-8"))

if master_doc is None:
    st.info("👈 Start by uploading or pasting your **master resume** in the sidebar.")
elif master_doc.is_empty:
    st.error(
        f"Only {len(master_doc.text.strip())} characters were extracted from the "
        "master resume. If it is a scanned PDF, OCR it first — verifying against "
        "an empty master would flag everything."
    )
else:
    st.success(
        f"Master resume loaded: **{master_doc.name}** — "
        f"{len(master_doc.text.split()):,} words, {master_doc.page_count} page(s), "
        f"extracted via `{master_doc.extractor}`."
    )
    with st.expander("Preview the master resume"):
        if st.session_state.get("master_pdf"):
            show_pdf(st.session_state["master_pdf"], height=680, key="master_preview")
        st.caption("Extracted text (what the checks actually run on):")
        st.text(master_doc.text[:4000] + ("\n…" if len(master_doc.text) > 4000 else ""))

st.subheader("Tailored resumes")
tab_upload, tab_link = st.tabs(["📂 Upload files", "🔗 Paste links"])

with tab_upload:
    uploaded = st.file_uploader(
        "Drag and drop your tailored single-pager resumes (multiple allowed)",
        type=["pdf", "txt", "md"], accept_multiple_files=True, key="tailored",
    )

with tab_link:
    url_blob = st.text_area(
        "Public URLs — one per line",
        height=140, key="tailored_urls",
        placeholder="https://example.com/resume_google.pdf\n"
                    "https://example.com/resume_quant.pdf",
        help="Direct links to the files. Google Drive / Dropbox / GitHub share "
             "links are rewritten to their download form automatically.",
    )
    st.caption("Both tabs can be used together — uploads and links are merged "
               "into a single batch.")

# Both sources feed one batch, so you can mix uploads and links freely.
files: list[tuple[str, bytes]] = [(f.name, f.getvalue()) for f in (uploaded or [])]

urls = parse_url_list(url_blob)
if urls:
    with st.spinner(f"Downloading {len(urls)} file(s)…"):
        fetched_files, fetch_errors = cached_fetch_many(tuple(urls))
    for bad_url, message in fetch_errors:
        st.error(f"**{bad_url}** — {message}")
    if fetched_files:
        st.success("Downloaded: " + ", ".join(f"`{f.name}`" for f in fetched_files))
    files.extend((f.name, f.data) for f in fetched_files)

ready = master_doc is not None and not master_doc.is_empty and bool(files)
if files and not ready:
    st.warning("Load a usable master resume before verifying.")

if not rule_only and not api_key:
    st.warning(
        f"No {key_label} found. Add one in the sidebar, or tick **Rule-only mode** "
        "to run the deterministic checks without an LLM."
    )
elif rule_only:
    st.info(
        "**Rule-only mode** — identity, phone, JEE, page count, metrics and "
        "line-level matching all run. Hallucinated skills and altered wording "
        "need an API key."
    )

if st.button("🔎 Verify resumes", type="primary", disabled=not ready):
    verifier = None
    if not rule_only:
        try:
            verifier = build_verifier(provider, api_key, model, master_doc.text, effort)
        except ImportError as exc:
            st.error(f"SDK not installed: {exc}. Run `pip install -r requirements.txt`.")
            st.stop()
        except Exception as exc:  # noqa: BLE001
            st.error(f"Could not initialise the {provider} client: {exc}")
            st.stop()

    settings = VerificationSettings(
        max_pages=int(max_pages),
        strict_page_limit=strict_page_limit,
        check_calendar_years=check_years,
        max_workers=max_workers,
    )

    progress = st.progress(0.0, text="Starting…")

    def on_progress(done: int, total: int, result: ResumeReport) -> None:
        progress.progress(done / total, text=f"Verified {done}/{total} — {result.filename}")

    with st.spinner("Cross-referencing against the master resume…"):
        st.session_state["reports"] = verify_batch(
            files, master_doc, verifier, settings, on_progress
        )
    progress.empty()

# --------------------------------------------------------------------------
# Results dashboard
# --------------------------------------------------------------------------

reports: list[ResumeReport] = st.session_state.get("reports", [])
if reports:
    st.markdown("---")
    st.header("📊 Results")

    passed = sum(1 for r in reports if r.status is Status.PASS)
    warned = sum(1 for r in reports if r.status is Status.WARNING)
    failed = sum(1 for r in reports if r.status is Status.FAIL)

    avg = round(sum(r.match_score for r in reports) / len(reports))
    best = max(reports, key=lambda r: r.match_score)

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Resumes", len(reports))
    c2.metric("Avg match /100", avg)
    c3.metric("✅ Pass", passed)
    c4.metric("⚠️ Warning", warned)
    c5.metric("❌ Fail", failed)
    st.caption(f"Strongest match: **{_short(best.filename)}** "
               f"({best.match_score}/100)")

    st.dataframe(reports_to_dataframe(reports), use_container_width=True,
                 hide_index=True)

    st.subheader("Detailed findings")
    for report in sorted(reports, key=lambda r: (r.match_score, r.filename)):
        render_report(report)

    st.markdown("---")
    dl1, dl2 = st.columns(2)
    dl1.download_button(
        "⬇️ Download full report (JSON)",
        data=json.dumps([r.model_dump(mode="json") for r in reports], indent=2),
        file_name="resume_verification_report.json", mime="application/json",
    )
    dl2.download_button(
        "⬇️ Download summary (CSV)",
        data=reports_to_dataframe(reports).to_csv(index=False),
        file_name="resume_verification_summary.csv", mime="text/csv",
    )
