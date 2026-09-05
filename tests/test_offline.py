"""Offline tests - no API key required.

Run:  python tests/test_offline.py      (or: python -m pytest tests/)
"""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from verifier import rules
from verifier.engine import VerificationSettings, verify_one
from verifier.llm import _extract_json
from verifier.schema import Finding, LLMReport, Status

# Enough real prose to clear the "looks like a scanned PDF" emptiness guard.
_FILLER = ("SKILLS Python, C++, PyTorch, numpy, Kubernetes, Docker, SQL, pandas.\n"
           "PROJECTS Training-free segmentation pipeline evaluated on image pairs.\n"
           "EDUCATION Indian Institute of Technology Kanpur, Electrical Engineering.\n")

MASTER = """Arjun Singla | a@example.com | Mobile: +91 98765 43210
IIT Kanpur, B.Tech EE, 2022-2026. CPI 8.7/10. JEE Advanced All India Rank 1204.
- Built an LOB simulator in C++ processing 2.5M messages per day.
- Reduced backtest runtime by 40% using vectorised numpy.
- Worked with a team of five to migrate research jobs to Kubernetes.
- Cut p99 latency from 800ms to 200ms."""


def test_phone_detection() -> None:
    positives = [
        "Ph: 9876543210", "Mobile: +91 98765 43210", "(415) 555-0132",
        "+1 555 123 4567", "Contact no - 98765 43210", "415.555.0132",
    ]
    for text in positives:
        assert rules.find_phone_numbers(text), f"missed phone in {text!r}"

    # These must NOT trip the detector.
    negatives = [
        "IIT Kanpur, 2022-2026",                 # date range
        "May 2025 - Jul 2025",                   # employment dates
        "Processed 2,500,000 messages per day",  # a metric
        "arjun1234567890@example.com",           # email
        "github.com/arjun/repo-1234567890",      # URL
        "CPI 8.7/10 and 40% improvement",        # metrics
    ]
    for text in negatives:
        hits = rules.find_phone_numbers(text)
        assert not hits, f"false positive in {text!r}: {[h.match for h in hits]}"


def test_jee_detection() -> None:
    """Explicit JEE mentions, and ranks that sit in a JEE context."""
    for text in ["JEE Advanced AIR 1204", "IIT-JEE 2022",
                 "Joint Entrance Examination", "AIR 993 in JEE Mains 2024",
                 "* AIR 5 in Mains 2024"]:
        assert rules.find_jee_references(text), f"missed JEE in {text!r}"

    for text in ["Air quality sensor project", "Airbus internship",
                 "Ranked top 5 in the cohort"]:
        hits = rules.find_jee_references(text)
        assert not hits, f"false positive in {text!r}: {[h.match for h in hits]}"


def test_non_jee_ranks_are_not_flagged() -> None:
    """Regression: real IIT-K resumes hold legitimate olympiad All India Ranks.

    Failing a resume for "All India Rank 14 in American Mathematics
    Competitions" was a false positive - the exclusion is a JEE rank, not the
    word "rank".
    """
    legitimate = [
        "* All India Rank 1 in Canadian Senior Mathematics Contest (CSMC), 2022",
        "* All India Rank 14 in American Mathematics Competitions (AMC) 12A, 2023",
        "* All India Rank 3 in the National qualifiers of the Hanoi Open Mathematics",
        "* AIR 10, Srinivasa Ramanujan Mathematics Competition (SRMC) 2025",
    ]
    for text in legitimate:
        hits = rules.find_jee_references(text)
        assert not hits, f"false positive in {text!r}: {[h.match for h in hits]}"
        assert rules.find_rank_mentions(text), f"should be reported as info: {text!r}"


def test_jee_context_does_not_leak_across_bullets() -> None:
    """The bullet immediately after a JEE bullet must not inherit its context."""
    text = ("* AIR 993 in JEE Mains 2024 out of 1.5 million candidates, and in "
            "the top 1% in JEE Advanced 2024 nationally.\n"
            "* AIR 10, Srinivasa Ramanujan Mathematics Competition (SRMC) 2025.")
    flagged = {h.match for h in rules.find_jee_references(text)}
    assert "AIR 993" in flagged, flagged
    assert "AIR 10," not in flagged, f"leaked across bullets: {flagged}"
    assert any("AIR 10" in h.match for h in rules.find_rank_mentions(text))


def test_rank_mentions_never_change_the_verdict() -> None:
    """A resume full of olympiad ranks and nothing else must not fail."""
    body = ("Aradhya Goel | a@example.com\n"
            "* All India Rank 1 in Canadian Senior Mathematics Contest (CSMC), 2022\n"
            "* AIR 10, Srinivasa Ramanujan Mathematics Competition (SRMC) 2025\n" + _FILLER)
    report = verify_one("olympiad.txt", body.encode(), body,
                        rules.extract_metrics(body), None, VerificationSettings())
    assert not report.has_banned_exam, report.jee_hits
    assert len(report.rank_mentions) == 2, report.rank_mentions
    assert report.status is not Status.FAIL, report.headline()


def test_equivalent_notation_is_not_flagged() -> None:
    """Honest rewording of the same number must never reach the LLM as a candidate."""
    master = rules.extract_metrics(rules.strip_contact_noise(
        "Improved throughput by 40 percent over 12 months, saving $1,500,000."))
    tailored = rules.extract_metrics(rules.strip_contact_noise(
        "Improved throughput 40% in 1 year, saving $1.5M."))
    unmatched = rules.unmatched_metrics(tailored, master)
    assert not unmatched, f"equivalent notation flagged: {[m.raw for m in unmatched]}"


def test_fabricated_metrics_are_caught() -> None:
    master = rules.extract_metrics(rules.strip_contact_noise(MASTER))
    tailored = rules.extract_metrics(rules.strip_contact_noise(
        "Reduced backtest runtime by 65% and increased PnL by $2.3M."))
    raw = {m.raw for m in rules.unmatched_metrics(tailored, master)}
    assert "65%" in raw and "$2.3M" in raw, raw


def test_calendar_years_excluded_by_default() -> None:
    master = rules.extract_metrics("Worked 2022-2024.")
    tailored = rules.extract_metrics("Worked 2019-2021.")
    assert not rules.unmatched_metrics(tailored, master)
    assert rules.unmatched_metrics(tailored, master, include_years=True)


def test_json_extraction_survives_prose_and_fences() -> None:
    payload = {"overall_assessment": "ok", "findings": [], "verified_ok": []}
    for variant in (
        json.dumps(payload),
        "```json\n" + json.dumps(payload) + "\n```",
        "Here is the audit:\n" + json.dumps(payload) + "\nLet me know if...",
    ):
        assert _extract_json(variant) == payload


# --------------------------------------------------------------------------
# LLM path, with a stubbed Anthropic client
# --------------------------------------------------------------------------

VALID_RESPONSE = json.dumps({
    "overall_assessment": "Two fabricated metrics found.",
    "findings": [{
        "category": "UNVERIFIED_METRIC", "severity": "high",
        "quote": "reduced runtime by 65%",
        "issue": "Master states 40%, not 65%.",
        "master_evidence": "Reduced backtest runtime by 40%",
        "suggested_fix": "Restore the 40% figure.",
    }],
    "verified_ok": ["'5-person team' matches 'a team of five'."],
})


class _Block:
    def __init__(self, text): self.type, self.text = "text", text


class _Response:
    stop_reason = "end_turn"
    def __init__(self, text): self.content = [_Block(text)]


def _install_stub(fail_variants: int = 0):
    """Replace anthropic.Anthropic with a recorder; optionally reject the first
    N calls with BadRequestError to exercise the degradation ladder."""
    import anthropic
    calls: list[dict] = []

    class _Messages:
        def create(self, **kwargs):
            calls.append(kwargs)
            if len(calls) <= fail_variants:
                raise anthropic.BadRequestError(
                    "unsupported parameter",
                    response=types.SimpleNamespace(status_code=400, headers={},
                                               request=types.SimpleNamespace()),
                    body=None,
                )
            return _Response(VALID_RESPONSE)

    class _Client:
        def __init__(self, **_): self.messages = _Messages()

    original = anthropic.Anthropic
    anthropic.Anthropic = _Client
    return calls, (lambda: setattr(anthropic, "Anthropic", original))


def test_anthropic_request_shape() -> None:
    from verifier.llm import AnthropicVerifier

    calls, restore = _install_stub()
    try:
        v = AnthropicVerifier("test-key", "claude-opus-5", MASTER, "high")
        report = v.verify("Reduced runtime by 65%.", [], "t.pdf")
    finally:
        restore()

    assert isinstance(report, LLMReport) and len(report.findings) == 1
    assert report.findings[0].category == "UNVERIFIED_METRIC"

    kwargs = calls[0]
    assert kwargs["model"] == "claude-opus-5"
    assert kwargs["thinking"] == {"type": "adaptive"}
    assert kwargs["output_config"]["effort"] == "high"
    assert kwargs["output_config"]["format"]["type"] == "json_schema"
    # The master resume must be the cached system block, and the tailored
    # resume must live in messages (after the cache breakpoint).
    system = kwargs["system"]
    assert system[1]["cache_control"] == {"type": "ephemeral"}
    assert "<master_resume>" in system[1]["text"]
    assert "65%" in kwargs["messages"][0]["content"]


def test_degradation_ladder() -> None:
    """A model/SDK rejecting thinking or output_config must still produce a report."""
    from verifier.llm import AnthropicVerifier

    calls, restore = _install_stub(fail_variants=2)
    try:
        v = AnthropicVerifier("test-key", "claude-opus-5", MASTER, "high")
        report = v.verify("Reduced runtime by 65%.", [], "t.pdf")
    finally:
        restore()

    assert len(calls) == 3, f"expected 3 attempts, got {len(calls)}"
    assert "thinking" not in calls[2] and "output_config" not in calls[2]
    # The final, bare attempt must inline the schema in the prompt instead.
    assert "json_schema" not in calls[2]["messages"][0]["content"]
    assert '"overall_assessment"' in calls[2]["messages"][0]["content"]
    assert len(report.findings) == 1


def test_end_to_end_with_stubbed_llm() -> None:
    from verifier.llm import AnthropicVerifier

    calls, restore = _install_stub()
    try:
        verifier = AnthropicVerifier("test-key", "claude-opus-5", MASTER, "high")
        dirty = ("Arjun Singla | a@example.com | Ph: 9876543210\n"
                 "JEE Advanced AIR 1204.\n- Reduced runtime by 65%.\n" + _FILLER)
        report = verify_one("dirty.txt", dirty.encode(), MASTER,
                            rules.extract_metrics(rules.strip_contact_noise(MASTER)),
                            verifier, VerificationSettings())
    finally:
        restore()

    assert report.status is Status.FAIL
    assert report.has_phone and report.has_banned_exam
    assert report.llm_used
    # Deterministic findings + the LLM finding are merged into one list.
    assert report.blocking_count >= 3


def test_llm_failure_is_surfaced_not_swallowed() -> None:
    class Boom:
        def verify(self, *a, **k): raise RuntimeError("network down")

    clean = ("Arjun Singla | a@example.com\n"
             "- Built a C++ LOB simulator processing 2.5M messages per day.\n" + _FILLER)
    report = verify_one("x.txt", clean.encode(), MASTER,
                        rules.extract_metrics(rules.strip_contact_noise(MASTER)),
                        Boom(), VerificationSettings())
    assert report.error and "network down" in report.error
    assert report.status is Status.WARNING, "a failed audit must never read as PASS"


# --------------------------------------------------------------------------
# URL fetching - exercised against a real local HTTP server
# --------------------------------------------------------------------------

def _serve(routes: dict[str, tuple[int, str, bytes]]):
    """Start a throwaway HTTP server. routes: path -> (status, content_type, body)."""
    import http.server, threading

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802
            if self.path not in routes:
                self.send_error(404, "Not Found")
                return
            status, ctype, body = routes[self.path]
            self.send_response(status)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *a):  # silence
            pass

    server = http.server.HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, f"http://127.0.0.1:{server.server_port}"


def test_share_link_normalisation() -> None:
    from verifier.fetch import normalise_share_url

    assert normalise_share_url("https://drive.google.com/file/d/1AbC_x-9/view?usp=sharing") \
        == "https://drive.google.com/uc?export=download&id=1AbC_x-9"
    assert normalise_share_url("https://www.dropbox.com/s/abc/cv.pdf?dl=0").endswith("dl=1")
    assert normalise_share_url("https://github.com/me/repo/blob/main/cv.pdf") \
        == "https://raw.githubusercontent.com/me/repo/main/cv.pdf"
    plain = "https://example.com/cv.pdf"
    assert normalise_share_url(plain) == plain


def test_fetch_real_pdf_and_reject_bad_responses() -> None:
    from verifier.fetch import FetchError, fetch_document, fetch_many

    pdf_bytes = b"%PDF-1.4\n" + b"x" * 500 + b"\n%%EOF"
    server, base = _serve({
        "/cv.pdf": (200, "application/pdf", pdf_bytes),
        "/noext": (200, "application/pdf", pdf_bytes),
        "/share": (200, "text/html", b"<!DOCTYPE html><html><body>Sign in</body></html>"),
        "/notes.txt": (200, "text/plain", b"plain text resume"),
    })
    try:
        got = fetch_document(f"{base}/cv.pdf")
        assert got.name == "cv.pdf" and got.data == pdf_bytes

        # Magic bytes win over a missing extension.
        assert fetch_document(f"{base}/noext").name.endswith(".pdf")

        # A Drive-style HTML share page must fail with an actionable message.
        try:
            fetch_document(f"{base}/share")
            raise AssertionError("HTML share page should have been rejected")
        except FetchError as exc:
            assert "web page" in str(exc).lower()

        # 404 surfaces as a FetchError, not a traceback.
        try:
            fetch_document(f"{base}/missing.pdf")
            raise AssertionError("404 should have been rejected")
        except FetchError as exc:
            assert "404" in str(exc)

        # One bad link must not abort the rest of the batch.
        ok, errs = fetch_many([f"{base}/cv.pdf", f"{base}/share", f"{base}/notes.txt"])
        assert len(ok) == 2 and len(errs) == 1
    finally:
        server.shutdown()


def test_non_http_scheme_rejected() -> None:
    from verifier.fetch import FetchError, fetch_document

    for bad in ("file:///etc/passwd", "ftp://example.com/cv.pdf"):
        try:
            fetch_document(bad)
            raise AssertionError(f"{bad} should have been rejected")
        except FetchError as exc:
            assert "http/https" in str(exc)


def test_url_list_parsing() -> None:
    from verifier.fetch import parse_url_list

    blob = "https://a.com/1.pdf\nhttps://b.com/2.pdf,  https://c.com/3.pdf\n\n"
    assert parse_url_list(blob) == ["https://a.com/1.pdf", "https://b.com/2.pdf",
                                    "https://c.com/3.pdf"]
    assert parse_url_list("") == []


# --------------------------------------------------------------------------
# Match score
# --------------------------------------------------------------------------

def _score(text: str, verifier=None, master: str = MASTER):
    # _FILLER is scaffolding to clear the "looks like a scanned PDF" guard, so
    # it belongs in BOTH documents - otherwise it reads as invented content and
    # every scenario here would carry an unrelated coverage penalty.
    full_master = master + "\n" + _FILLER
    return verify_one("r.txt", (text + "\n" + _FILLER).encode(), full_master,
                      rules.extract_metrics(rules.strip_contact_noise(full_master))
                      + rules.word_number_metrics(full_master),
                      verifier, VerificationSettings())


def test_clean_resume_scores_full_marks() -> None:
    report = _score("- Built an LOB simulator in C++ processing 2.5M messages per day.\n"
                    "- Reduced backtest runtime by 40% using vectorised numpy.")
    assert report.match_score == 100, (report.match_score, report.score_lines)
    assert report.score_band == "Verified"


def test_heavy_rewrite_does_not_cost_points() -> None:
    """Regression: line-level coverage must never feed the score.

    Fuzzy matching scores "Engineered a C++ LOB simulator" against
    "Built a limit order book simulator in C++" as unsupported. Charging for
    that cost 27 points for changing nothing but words.
    """
    master = ("Aradhya Goel a@iitk.ac.in\n"
              "* Built a limit order book simulator in C++ processing 2.5M messages per day.\n"
              "* Reduced backtest runtime by 40% using vectorised numpy operations.\n"
              "* Worked with a team of five to migrate research jobs to Kubernetes.\n"
              + _FILLER)

    def run(body):
        return verify_one("r.txt", (body + "\n" + _FILLER).encode(), master,
                          rules.extract_metrics(rules.strip_contact_noise(master))
                          + rules.word_number_metrics(master),
                          None, VerificationSettings())

    verbatim = run("Aradhya Goel a@iitk.ac.in\n"
                   "* Built a limit order book simulator in C++ processing 2.5M messages per day.\n"
                   "* Reduced backtest runtime by 40% using vectorised numpy operations.\n"
                   "* Worked with a team of five to migrate research jobs to Kubernetes.")
    rewritten = run("Aradhya Goel a@iitk.ac.in\n"
                    "* Engineered a C++ LOB simulator handling 2.5M messages daily.\n"
                    "* Cut backtest runtime 40% via vectorised numpy.\n"
                    "* Collaborated on a 5-person Kubernetes migration of research jobs.")

    assert verbatim.match_score == rewritten.match_score == 100, (
        verbatim.match_score, rewritten.match_score)
    # The rewrite genuinely trips the fuzzy matcher - it is still reported as
    # evidence, it just must not be scored.
    assert rewritten.coverage < 1.0, "expected fuzzy matching to miss the rewrite"
    assert not any("coverage" in line.label.lower() for line in rewritten.score_lines)


def test_rephrasing_does_not_cost_points() -> None:
    """The whole premise: an honest rewrite must score the same as a copy-paste."""
    verbatim = _score("- Built an LOB simulator in C++ processing 2.5M messages per day.\n"
                      "- Reduced backtest runtime by 40% using vectorised numpy.")
    rewritten = _score("- Engineered a C++ limit order book simulator handling "
                       "2.5M messages daily.\n"
                       "- Cut backtest runtime 40% via vectorised numpy operations.")
    assert rewritten.match_score == verbatim.match_score == 100, (
        verbatim.match_score, rewritten.match_score)


def test_exclusions_and_findings_deduct() -> None:
    from verifier.scoring import PENALTY_BANNED_EXAM, PENALTY_PHONE

    phone = _score("Ph: 9876543210\n- Built an LOB simulator in C++.")
    assert phone.match_score == 100 - PENALTY_PHONE, phone.score_lines

    jee = _score("- AIR 993 in JEE Mains 2024.\n- Built an LOB simulator in C++.")
    assert jee.match_score == 100 - PENALTY_BANNED_EXAM, jee.score_lines

    # An olympiad rank costs nothing *as an exclusion* when the master
    # supports it. (It is still metric-checked like any other claim - being a
    # non-JEE rank exempts it from the auto-fail, not from verification.)
    olympiad_master = MASTER + "\n- All India Rank 14 in American Mathematics Competitions."
    olympiad = _score("- All India Rank 14 in American Mathematics Competitions.",
                      master=olympiad_master)
    assert olympiad.match_score == 100, olympiad.score_lines
    assert not any("JEE" in line.label for line in olympiad.score_lines)

    # And with no master support it is charged as an unverified metric, never
    # as a JEE violation.
    unsupported = _score("- All India Rank 14 in American Mathematics Competitions.")
    assert not unsupported.has_banned_exam
    assert any("metrics unmatched" in line.label for line in unsupported.score_lines)


def test_score_is_itemised_and_adds_up() -> None:
    report = _score("Ph: 9876543210\n- AIR 993 in JEE Mains 2024.")
    total = 100 + sum(line.delta for line in report.score_lines)
    assert report.match_score == max(0, total), (report.match_score, report.score_lines)
    assert len(report.score_lines) >= 2


def test_llm_findings_feed_the_score() -> None:
    from verifier.scoring import PENALTY_HIGH, PENALTY_MEDIUM

    class Stub:
        def verify(self, *a, **k):
            return LLMReport(
                overall_assessment="",
                findings=[
                    Finding(category="UNVERIFIED_METRIC", severity="high", quote="q",
                            issue="i", master_evidence="NOT FOUND IN MASTER", suggested_fix="f"),
                    Finding(category="HALLUCINATED_SKILL", severity="medium", quote="q",
                            issue="i", master_evidence="NOT FOUND IN MASTER", suggested_fix="f"),
                ],
                verified_ok=[])

    report = _score("- Built an LOB simulator in C++.", verifier=Stub())
    assert report.match_score == 100 - PENALTY_HIGH - PENALTY_MEDIUM, report.score_lines
    assert not report.score_provisional, "LLM ran, so the score is not provisional"


def test_rule_only_score_is_marked_provisional() -> None:
    report = _score("- Built an LOB simulator in C++.")
    assert report.score_provisional, "no LLM ran - the score must be flagged partial"


def test_score_never_leaves_0_100() -> None:
    awful = _score("Ph: 9876543210 | Mobile: +91 98765 43210\n"
                   "- AIR 993 in JEE Mains 2024, 99.9% percentile, $50M raised, "
                   "300% growth, 45x returns, 12 years experience.")
    assert 0 <= awful.match_score <= 100, awful.match_score


def test_band_labels_are_descriptive_not_editorial() -> None:
    """The tool reports what it observed; it does not rule on intent or sendability."""
    from verifier.scoring import BANDS

    labels = " ".join(label for _, label in BANDS).lower()
    for judgement in ("sendable", "fabrication", "lie", "fraud", "reject"):
        assert judgement not in labels, f"editorial word in band labels: {judgement}"


def test_score_description_matches_the_actual_deductions() -> None:
    """A canned per-band sentence can assert a cause that did not occur."""
    from verifier.schema import ScoreLine
    from verifier.scoring import describe_score

    # Below 50 purely from findings - no exclusion violation at all.
    only_findings = [ScoreLine(label="5 high-severity findings", delta=-60)]
    text = describe_score(40, only_findings)
    assert "5 high-severity findings" in text
    assert "exclusion" not in text.lower(), text

    # Exclusion present -> it is named.
    with_exclusion = [ScoreLine(label="Mentions a JEE rank", delta=-35)]
    assert "JEE" in describe_score(65, with_exclusion)

    # Nothing deducted -> a clean generic sentence, not an empty string.
    clean = describe_score(100, [ScoreLine(label="No deductions", delta=0)])
    assert clean and "master resume" in clean


# --------------------------------------------------------------------------
# Identity gate
# --------------------------------------------------------------------------

_ARADHYA = ("Aradhya Goel\nThird Year Undergraduate, IIT Kanpur\n"
            "Mechanical Engineering, IIT Kanpur aradhyag24@iitk.ac.in\n" + _FILLER)
_ROHIT = ("Rohit Verma\nFourth Year Undergraduate, IIT Kanpur\n"
          "Civil Engineering, IIT Kanpur rohitv22@iitk.ac.in\n" + _FILLER)


def test_identity_extraction() -> None:
    from verifier.identity import extract_identity

    ident = extract_identity(_ARADHYA)
    assert ident.name == "Aradhya Goel", ident
    assert "aradhyag24@iitk.ac.in" in ident.emails
    # Section headers and the institute line must not be mistaken for a name.
    assert extract_identity("Education\nIndian Institute of Technology Kanpur").name == ""


def test_different_person_is_caught() -> None:
    """The regression: a master for one candidate and a single-pager for
    another used to score 88/100 because nothing compared the two names."""
    report = verify_one("rohit.txt", _ROHIT.encode(), _ARADHYA,
                        rules.extract_metrics(_ARADHYA), None, VerificationSettings())
    assert report.identity_match is False, report.identity_reason
    assert report.status is Status.FAIL
    assert report.match_score == 0, report.score_lines
    assert "Rohit Verma" in report.identity_reason


def test_same_person_passes_the_gate() -> None:
    tailored = ("Aradhya Goel\nMechanical Engineering, IIT Kanpur "
                "aradhyag24@iitk.ac.in\n" + _FILLER)
    report = verify_one("a.txt", tailored.encode(), _ARADHYA,
                        rules.extract_metrics(_ARADHYA), None, VerificationSettings())
    assert report.identity_match is True, report.identity_reason
    assert report.match_score == 100


def test_same_person_with_a_different_email_is_not_a_mismatch() -> None:
    """A personal address on one resume and an institute address on the other
    is normal. The name still agrees, so it must not fail."""
    from verifier.identity import compare_identities, extract_identity

    personal = ("Aradhya Goel\nMechanical Engineering aradhya.goel@gmail.com\n" + _FILLER)
    verdict = compare_identities(extract_identity(_ARADHYA), extract_identity(personal))
    assert verdict.same_person is True, verdict.reason


def test_missing_identity_is_unconfirmed_not_a_failure() -> None:
    """Absence of a name/email means 'cannot verify', never 'different person'."""
    from verifier.identity import compare_identities, extract_identity

    anonymous = "EXPERIENCE\n- Built an LOB simulator in C++.\n" + _FILLER
    verdict = compare_identities(extract_identity(_ARADHYA), extract_identity(anonymous))
    assert verdict.same_person is None, verdict.reason

    report = verify_one("anon.txt", anonymous.encode(), _ARADHYA,
                        rules.extract_metrics(_ARADHYA), None, VerificationSettings())
    assert report.status is not Status.FAIL, report.headline()


def test_spelling_variant_is_not_a_mismatch() -> None:
    from verifier.identity import compare_identities, extract_identity

    variant = "Aradhya Goyal\nMechanical Engineering, IIT Kanpur\n" + _FILLER
    verdict = compare_identities(extract_identity(_ARADHYA), extract_identity(variant))
    assert verdict.same_person is not False, verdict.reason


def test_identity_mismatch_skips_the_llm() -> None:
    """No point paying a model to describe a stranger's resume."""
    calls = []

    class Spy:
        def verify(self, *a, **k):
            calls.append(1)
            raise AssertionError("LLM must not be called on an identity mismatch")

    report = verify_one("rohit.txt", _ROHIT.encode(), _ARADHYA,
                        rules.extract_metrics(_ARADHYA), Spy(), VerificationSettings())
    assert not calls and report.match_score == 0


# --------------------------------------------------------------------------
# Line-level subset matching (ported from resume_processor)
# --------------------------------------------------------------------------

_LM_MASTER = """Aradhya Goel aradhyag24@iitk.ac.in
* Built a limit order book simulator in C++ processing 2.5M messages per day.
* Reduced backtest runtime by 40% using vectorised numpy operations.
* Worked with a team of five to migrate research jobs to Kubernetes.
"""


def test_verbatim_and_reworded_lines_both_count_as_covered() -> None:
    """Rephrasing is allowed, so a reworded line must not read as unsupported."""
    from verifier import linematch

    verbatim = linematch.match_lines(
        "* Reduced backtest runtime by 40% using vectorised numpy operations.",
        _LM_MASTER)
    assert verbatim and verbatim[0].level == "matched", verbatim

    reworded = linematch.match_lines(
        "* Cut backtest runtime 40% with vectorised numpy.", _LM_MASTER)
    assert reworded and reworded[0].level != "unsupported", reworded
    assert linematch.coverage(reworded) == 1.0


def test_fabricated_line_is_unsupported() -> None:
    from verifier import linematch

    results = linematch.match_lines(
        "* Shipped a Kafka streaming layer at Bridgewater Associates.", _LM_MASTER)
    assert results and results[0].level == "unsupported", results
    assert linematch.coverage(results) == 0.0


def test_section_headers_are_ignored() -> None:
    """Short furniture lines carry no signal and must not be scored."""
    from verifier import linematch

    assert linematch.match_lines("SKILLS\nEDUCATION\n* * *", _LM_MASTER) == []


def test_line_matching_survives_missing_optional_deps() -> None:
    """rapidfuzz/pymupdf are optional - the tool must degrade, not crash."""
    import builtins
    from verifier import linematch

    real_import = builtins.__import__

    def blocked(name, *a, **k):
        if name in ("rapidfuzz", "fitz"):
            raise ImportError(f"{name} not installed")
        return real_import(name, *a, **k)

    builtins.__import__ = blocked
    try:
        assert linematch.match_lines("* anything at all here", _LM_MASTER) == []
        assert linematch.extract_pdf_lines(b"%PDF-1.4 fake") == []
    finally:
        builtins.__import__ = real_import


def test_annotation_returns_none_without_drawable_lines() -> None:
    from verifier.annotate import annotate_pdf

    assert annotate_pdf(b"%PDF-1.4 not really a pdf", []) is None


def test_annotated_pdf_is_excluded_from_json_export() -> None:
    """The JSON report must stay serialisable and small."""
    from verifier.schema import ResumeReport

    report = ResumeReport(filename="x.pdf", annotated_pdf=b"%PDF-1.4 bytes")
    dumped = report.model_dump(mode="json")
    assert "annotated_pdf" not in dumped, dumped.keys()
    import json
    json.dumps(dumped)   # must not raise


def test_unsupported_lines_reach_the_prompt() -> None:
    from verifier.prompts import build_user_prompt

    prompt = build_user_prompt("resume text", [], "f.pdf",
                               ["* Shipped Kafka at Bridgewater Associates."])
    assert "Bridgewater" in prompt
    assert "lines_without_a_textual_match" in prompt
    # The model must be told the scan compares wording, not meaning.
    assert "rewording" in prompt.lower() or "reworded" in prompt.lower()


# --------------------------------------------------------------------------
# Page limit
# --------------------------------------------------------------------------

class SkipTest(Exception):
    """Raised when an optional dev-only dependency is unavailable."""


def _pdf(pages: int) -> bytes:
    """A minimal multi-page PDF with enough text to clear the emptiness guard.

    Needs reportlab (see requirements-dev.txt). Absent it, the page-limit tests
    skip rather than fail - the core suite must pass on a plain install.
    """
    from io import BytesIO
    try:
        from reportlab.lib.pagesizes import LETTER
        from reportlab.pdfgen import canvas
    except ImportError:
        raise SkipTest("reportlab not installed (pip install -r requirements-dev.txt)")

    buf = BytesIO()
    c = canvas.Canvas(buf, pagesize=LETTER)
    for _ in range(pages):
        y = 740
        for line in ("Aradhya Goel a@iitk.ac.in",
                     "* Built a limit order book simulator in C++.",
                     _FILLER.replace("\n", " ")[:90],
                     _FILLER.replace("\n", " ")[90:180]):
            c.setFont("Helvetica", 10); c.drawString(54, y, line); y -= 15
        c.showPage()
    c.save()
    return buf.getvalue()


def _run_pages(pages: int, **kw):
    master = "Aradhya Goel a@iitk.ac.in\n* Built a limit order book simulator in C++.\n" + _FILLER
    return verify_one(f"{pages}p.pdf", _pdf(pages), master,
                      rules.extract_metrics(rules.strip_contact_noise(master)),
                      None, VerificationSettings(**kw))


def test_two_page_resume_is_allowed_by_default() -> None:
    """A "single-pager" is the short-resume format, not literally one sheet."""
    report = _run_pages(2)
    assert report.page_count == 2, report.page_count
    assert report.within_page_limit, report.headline()
    assert report.status is not Status.FAIL
    assert not any("pages" in line.label for line in report.score_lines), report.score_lines


def test_over_the_limit_fails_and_is_charged_per_extra_page() -> None:
    from verifier.scoring import PENALTY_PER_EXTRA_PAGE

    report = _run_pages(4)          # limit 2 -> two pages over
    assert not report.within_page_limit
    assert report.status is Status.FAIL
    assert report.match_score == 100 - 2 * PENALTY_PER_EXTRA_PAGE, report.score_lines
    assert "limit 2" in report.headline()


def test_page_limit_is_configurable() -> None:
    assert _run_pages(3, max_pages=3).within_page_limit
    assert not _run_pages(3, max_pages=1).within_page_limit
    # Strictness is separable from the limit itself.
    lenient = _run_pages(4, max_pages=2, strict_page_limit=False)
    assert not lenient.within_page_limit
    assert lenient.status is Status.WARNING, lenient.status


def test_clean_resume_passes_in_rule_only_mode() -> None:
    """A clean resume must be able to reach PASS without an API key.

    Forcing WARNING whenever the LLM had not run made every resume a warning,
    so the status column carried no information. The partial-audit caveat lives
    on `score_provisional` instead.
    """
    report = _score("- Built an LOB simulator in C++ processing 2.5M messages per day.")
    assert report.status is Status.PASS, (report.status, report.headline())
    assert report.score_provisional, "a rule-only pass must still be marked partial"


def test_llm_error_still_downgrades_to_warning() -> None:
    """An audit that tried to run the LLM and failed is NOT a clean pass."""
    class Boom:
        def verify(self, *a, **k): raise RuntimeError("network down")

    report = _score("- Built an LOB simulator in C++.", verifier=Boom())
    assert report.status is Status.WARNING, report.status
    assert report.error


def test_unreviewed_lines_block_a_clean_pass() -> None:
    """Reporting flagged lines while the verdict says PASS is incoherent.

    Real case: five resumes showed "lines with no counterpart in the master"
    and simultaneously scored 100/100 PASS.
    """
    master = ("Aniket Misra a@iitk.ac.in\n"
              "* Built a limit order book simulator in C++.\n" + _FILLER)
    body = ("Aniket Misra a@iitk.ac.in\n"
            "* Built a limit order book simulator in C++.\n"
            "* Led the acquisition of Bridgewater Associates for four billion dollars.\n"
            + _FILLER)
    report = verify_one("r.txt", body.encode(), master,
                        rules.extract_metrics(rules.strip_contact_noise(master))
                        + rules.word_number_metrics(master),
                        None, VerificationSettings())
    assert report.unsupported_lines, "expected the invented bullet to be flagged"
    assert report.status is Status.WARNING, report.status
    # Not scored - fuzzy matching cannot tell invention from an honest rewrite.
    assert report.match_score == 100, report.score_lines
    from verifier.scoring import describe_score
    text = describe_score(report.match_score, report.score_lines)
    assert "needs review" in text, text
    assert "traces back" not in text, text


def test_hyphenated_master_words_still_match() -> None:
    """Column layouts break words across lines: "Materials char-\nacterization"."""
    from verifier import linematch

    master = ("Relevant Courses\n"
              "Materials char-\nacterization (SEM, TEM, XRD), Semiconductor characterization\n")
    results = linematch.match_lines(
        "* Materials characterization (SEM, TEM, XRD) coursework completed.", master)
    assert results, results
    assert results[0].level != "unsupported", (results[0].level, results[0].score)


# --------------------------------------------------------------------------
# GATE and the configurable exam list
# --------------------------------------------------------------------------

def test_gate_is_banned_on_any_mention() -> None:
    """GATE is treated like JEE: the exam may not appear at all.

    This deliberately fails a legitimate M.Tech credential line
    ("Qualified Graduate Aptitude Test in Engineering (GATE)") - the rule is
    about the exam being named, not about a rank being quoted.
    """
    for text in ["GATE AIR 245",
                 "Qualified Graduate Aptitude Test in Engineering (GATE) - Physics",
                 "GATE score 812"]:
        assert rules.find_exam_references(text), f"missed GATE in {text!r}"


def test_rank_only_exams_need_a_score_word() -> None:
    """CAT/NEET are barred only when a rank or score is attached."""
    with_score = rules.find_exam_references("CAT 99.8 percentile", ("CAT",))
    assert with_score, "a CAT percentile should be flagged"
    bare = rules.find_exam_references("Common Admission Test qualified", ("CAT",))
    assert not bare, f"a bare CAT mention should be allowed: {[h.match for h in bare]}"


def test_exam_list_is_configurable() -> None:
    text = "Qualified GATE - Physics Paper 2024"
    assert rules.find_exam_references(text, ("JEE", "GATE"))
    assert not rules.find_exam_references(text, ("JEE",)), "GATE off = not flagged"


def test_olympiad_ranks_survive_the_wider_exam_list() -> None:
    """Adding GATE must not resurrect the olympiad false positive."""
    for text in ["* All India Rank 14 in American Mathematics Competitions (AMC) 12A",
                 "* AIR 10, Srinivasa Ramanujan Mathematics Competition (SRMC) 2025"]:
        assert not rules.find_exam_references(text), text
        assert rules.find_rank_mentions(text), text


def test_gate_failure_is_scored_and_named() -> None:
    from verifier.scoring import PENALTY_BANNED_EXAM

    report = _score("- Qualified GATE - Physics (PH) Paper in 2024.")
    assert report.has_banned_exam
    assert report.status is Status.FAIL
    assert report.match_score == 100 - PENALTY_BANNED_EXAM, report.score_lines
    assert any("GATE" in line.label for line in report.score_lines), report.score_lines


if __name__ == "__main__":
    failures = skipped = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"  PASS  {name}")
            except SkipTest as exc:
                skipped += 1
                print(f"  SKIP  {name}: {exc}")
            except AssertionError as exc:
                failures += 1
                print(f"  FAIL  {name}: {exc}")
            except Exception as exc:  # noqa: BLE001
                failures += 1
                print(f"  ERROR {name}: {exc.__class__.__name__}: {exc}")
    tail = f" ({skipped} skipped)" if skipped else ""
    print("\n" + ("All tests passed." + tail if not failures
                  else f"{failures} test(s) failed.{tail}"))
    sys.exit(1 if failures else 0)
