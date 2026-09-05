# Smart Resume Verification Tool

Cross-checks tailored single-pager resumes against a **master resume** (the
source of truth) and flags anything the master doesn't support: invented
metrics, hallucinated experience, phone numbers, JEE ranks, over-length
documents — and resumes that belong to a different candidate entirely.

Runs entirely on your own machine. **No API key is required** for the
deterministic checks.

---

## Quick start

### 1. Install Python 3.10 or newer

- **Windows** — <https://www.python.org/downloads/>, and **tick "Add Python to
  PATH"** in the installer.
- **Mac** — same page, or `brew install python`.
- **Linux** — almost certainly already there: `python3 --version`.

### 2. Run it

Double-click **`run.bat`** (Windows) or run **`./run.sh`** (Mac/Linux). The
first run creates a virtual environment and installs everything; later runs
start immediately.

Doing it by hand instead:

```bash
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

### 3. Open <http://localhost:8501>

Load the **master resume** in the sidebar (upload, paste a link, or paste the
text), then add the tailored resumes in the main panel — upload them, paste
their URLs, or both. Click **Verify resumes**.

Press **Ctrl+C** in the terminal to stop.

---

## What you get without an API key

Everything deterministic: the identity gate, phone numbers, JEE ranks, the page
limit, metric cross-checking, line-level subset matching, the annotated PDF
side-by-side view, and the match score. This is **rule-only mode**, on by
default when no key is present.

Adding an `ANTHROPIC_API_KEY` (sidebar, or an environment variable) additionally
catches hallucinated skills and meaning-changing rewrites — the things only a
language model can judge. That is the only part that costs money.

---

## Design in one paragraph

The checks are split into two layers on purpose. **Hard exclusions** — phone
number, JEE/AIR rank, page count — are pure regex/metadata checks; they are
objective, auditable, and never delegated to a model. **Semantic checks** —
fabricated metrics, hallucinated skills/employers, title inflation — go to an
LLM, but not blind: a deterministic pre-scan first extracts every number from
both documents, normalises them (so `12 months` == `1 year` and `$1.5M` ==
`$1,500,000`), and hands the model the short list of tailored-resume metrics
with *no* numeric counterpart in the master. The model adjudicates that list
rather than hunting for numbers itself.

## Project structure

```
resume-verifier/
├── run.sh / run.bat        # one-click launcher (sets up on first run)
├── app.py                  # Streamlit UI: uploads, dashboard, cards, exports
├── requirements.txt
├── requirements-dev.txt    # adds reportlab, needed only for the tests
├── .env.example
├── verifier/
│   ├── pdf_utils.py        # pdfplumber -> pypdf extraction + page count
│   ├── fetch.py            # public-URL downloads + share-link normalisation
│   ├── rules.py            # phone / JEE regexes, metric harvesting + matching
│   ├── prompts.py          # THE PROMPT (system rulebook + per-resume message)
│   ├── schema.py           # JSON contract + Pydantic validation
│   ├── identity.py         # candidate identity gate (same person?)
│   ├── linematch.py        # line-level subset matching vs the master
│   ├── annotate.py         # highlights the findings onto the PDF
│   ├── scoring.py          # the 0-100 Match Score + itemised deductions
│   ├── llm.py              # Anthropic / OpenAI adapters
│   └── engine.py           # orchestration, merge, PASS/WARNING/FAIL verdict
└── tests/test_offline.py   # 10 tests, no API key needed
```

## Providing the resumes

Both the master and the tailored batch accept **either** input, and the two can
be mixed — uploads and links are merged into one batch.

| | Upload | Link |
|---|---|---|
| **Master resume** | sidebar file picker | sidebar URL field (or paste the raw text) |
| **Tailored resumes** | drag-and-drop, multiple files | "Paste links" tab, one URL per line |

Links must resolve to the actual file. Google Drive `/file/d/<id>/view`, Dropbox
`?dl=0` and GitHub `/blob/` URLs are rewritten to their download form
automatically; a share page that returns HTML is rejected with a message saying
so rather than being parsed as an empty resume. Downloads are capped at 25 MB,
restricted to http/https, and cached for an hour so widget changes don't re-hit
the network.

## API key (optional)

Read from, in order: the sidebar field (session only, never written to disk) →
`ANTHROPIC_API_KEY` / `OPENAI_API_KEY` → `.streamlit/secrets.toml`. Nothing in
this repo writes a key to disk, and `.gitignore` excludes `secrets.toml` and
`.env` so one can't be committed by accident.

## Verdicts

| Verdict | Meaning |
|---|---|
| ❌ **FAIL** | Phone number, JEE rank, over the page limit, or any `high`-severity content finding |
| ⚠️ **WARNING** | Only `medium`/`low` findings; lines the deterministic pass could not match and no LLM has judged; or the LLM leg was attempted and errored |
| ✅ **PASS** | Fully supported by the master resume. Shown as **PASS\*** in rule-only mode — it passed every check that ran, but hallucinated skills and altered wording were never examined. |

## Line-level matching & annotated PDFs

Every line of the tailored resume is fuzzy-matched against the master, so the
tool can say "this whole bullet has no counterpart in the master" — not just
"this number is unsupported" — **with no API key**.

Flagged lines are drawn onto the PDF and shown **inline, side by side with the
master**, so you review it in the page rather than downloading a file (red = no
textual match or a strict exclusion, orange = unsupported metric; hover a
highlight for the reason). A "Save the highlighted PDF" button is offered as a
secondary action. Inline rendering uses `st.pdf` and falls back to an `<embed>`
data URI if the `streamlit-pdf` extra is missing.

Strict-exclusion lines (phone, JEE rank) *are* highlighted even though they are
kept out of the content findings and the coverage figure — the point of the
marked-up PDF is to show a reviewer where every problem sits.

Unmatched lines hold the verdict at **⚠️ WARNING** rather than PASS. They do
not move the score — but reporting flagged lines under a green PASS was
incoherent, so they ask for a human look instead of being silently ignored.

Words the PDF broke across a line (`Materials char-` / `acterization`, common in
column layouts) are re-joined before matching; without that repair, content that
genuinely is in the master gets reported as invented.

**These findings never move the Match Score, deliberately.** Fuzzy matching
compares *wording*, and this tool's premise is that wording may change freely.
Measured: an honest rewrite of three bullets ("Built a limit order book
simulator in C++" → "Engineered a C++ LOB simulator") drops coverage to 75% —
scoring that cost 27 points for changing nothing but words. Coverage is reported
as evidence and passed to the LLM, which can judge meaning; the number stays put.

The same reasoning fixed a second leak: the master's metric pool now counts
spelled-out numbers, so rewriting "a team of five" as "5-person team" is no
longer an unverified metric.

Lines already charged as a strict exclusion are dropped from this pass — a phone
number has no counterpart in the master by definition, and reporting it twice
would double-charge one violation.

## Identity gate

Before anything else, the tool checks that the two documents describe the **same
candidate**, comparing email, roll number and name. Without this, a master for
one student and a single-pager for another scored 88/100 — the metric matcher
just saw "some numbers are unsupported", which reads like a tidy resume with a
few unverified claims rather than the wrong file entirely.

A mismatch is an immediate **FAIL at 0/100**, and the LLM call is skipped (no
point paying a model to describe a stranger's resume as unsupported).

The gate only fires on a *positive contradiction*. A missing name or email means
"could not confirm", never "different person" — single-pagers legitimately drop
the contact block. Agreement on any strong signal wins outright, so a personal
Gmail on one resume and an institute address on the other is not a mismatch when
the name still agrees.

## Match Score

Each resume gets a **0–100 Match Score**: how much of it is verified against the
master. It starts at 100 and every deduction is itemised in the UI, so a number
can always be traced to the lines that produced it.

| Deduction | Cost |
|---|---|
| Phone number present | −35 |
| JEE rank present | −35 |
| Each page over the limit | −25 |
| Each high-severity finding | −12 |
| Each medium / low finding | −5 / −2 |
| Unmatched metrics (rule-only mode) | up to −25, scaled by the ratio |

Bands: **90+ Verified** · **75+ Minor issues** · **50+ Needs review** · **<50 Major issues**

Band labels are descriptive rather than editorial, and the line under the score
is built from the deductions that actually applied — not a fixed sentence per
band. A canned blurb states a cause that may not hold: a resume can fall below
50 purely by accumulating findings, with no exclusion violation and nothing that
warrants the word "fabrication". The tool reports what it observed against the
master; whether a resume should be sent is the reviewer's call.

Two properties are deliberate and covered by tests:

* **Rephrasing never costs points.** The obvious implementation — lexical
  overlap against the master — would punish exactly what the tool permits, so a
  truthful rewrite would score worse than a copy-paste. The score is computed
  from checkable facts (metrics, findings, exclusions), never from wording.
* **No double counting.** When the LLM runs, an unsupported metric is already a
  finding, so the raw metric ratio isn't also charged. In rule-only mode there
  are no findings, so the ratio stands in — and the score is marked
  `provisional` (shown with a `*`), because the deterministic layer alone cannot
  see hallucinated skills.

The score is not asked of the model: an LLM-generated number would be neither
reproducible nor auditable.

## Rules the LLM enforces

* **Rephrasing is allowed.** Rewording, compressing, merging and reordering
  bullets is expected and never flagged. Restating a number in an equivalent
  unit is also fine.
* **Metrics are strict.** Every number in the tailored resume must already exist
  in the master, on the same accomplishment. New, altered, exaggerated and
  *derived* metrics are all flagged — correct arithmetic (`800ms→200ms` stated
  as `75%`) is explicitly not a defence.
* **Hallucination check.** New skills, employers, projects, degrees, or inflated
  titles/scope.

### On the page limit

A "single-pager" is the short-resume **format**, not literally one sheet — two
pages is normal. The limit is therefore a setting (**Maximum pages allowed**,
default **2**), not a hardcoded rule, and whether exceeding it fails or merely
warns is a separate toggle. Only pages *over* the limit are charged.

### On the JEE rule specifically

The exclusion is a **JEE rank**, not the word "rank". An "All India Rank" or
"AIR" is only treated as a JEE rank when JEE context appears in the *same
bullet* — so "AIR 993 in JEE Mains 2024" fails, while "All India Rank 14 in
American Mathematics Competitions" does not. Non-JEE rank claims are listed in
the report as informational context and never affect the verdict. Bullet-level
scoping is deliberate: a character window would leak "JEE" from one bullet into
the next.

The full rulebook lives in `verifier/prompts.py::SYSTEM_RULES`.

## Notes

* **Prompt caching**: the system rulebook + master resume are one cached prefix
  and the tailored resume goes in `messages`, so a batch of N resumes pays for
  the master once and reads it from cache N−1 times.
* **Scanned PDFs**: if under 200 characters are extracted, the file is failed
  with an OCR warning rather than silently "passing" on empty text.
* **Model**: defaults to `claude-opus-5` with adaptive thinking. The request
  degrades gracefully (thinking → structured output → plain JSON) if a
  model or SDK rejects a parameter.

## Note when developing

Streamlit reruns `app.py` on save but keeps imported modules in `sys.modules`,
so **edits under `verifier/` need a full server restart** (Ctrl+C and re-run),
not just a browser refresh. A stale module shows up as a `TypeError` about an
unexpected keyword argument.

## Tests

```bash
pip install -r requirements-dev.txt
python tests/test_offline.py
```

They run without an API key. The page-limit tests skip if `reportlab` is absent
rather than failing, so the suite also passes on a plain install.

Covers phone/JEE true and false positives, unit-equivalence, fabricated-metric
detection, JSON extraction, request shape (including the cache breakpoint), the
degradation ladder, that an LLM failure never renders as a PASS, and URL
fetching against a real local HTTP server (PDF, HTML share page, 404, bad
scheme, partial batch failure).
