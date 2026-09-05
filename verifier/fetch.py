"""Fetch resume documents from public URLs.

Uses only the standard library so the app gains no new dependency. Raw bytes
are downloaded (never a text-converted proxy) because the single-page check
needs the real PDF page count.

Anything fetched here is treated strictly as data to be audited - the contents
of a downloaded resume are never interpreted as instructions.
"""

from __future__ import annotations

import re
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass

MAX_BYTES = 25 * 1024 * 1024      # generous for a resume; stops runaway downloads
TIMEOUT_SECONDS = 30
_UA = "Mozilla/5.0 (compatible; ResumeVerifier/1.0)"


class FetchError(RuntimeError):
    """Raised with a message intended to be shown directly to the user."""


@dataclass
class FetchedFile:
    name: str
    data: bytes
    source_url: str


def normalise_share_url(url: str) -> str:
    """Rewrite common 'share page' links into direct-download links.

    A Google Drive /file/d/<id>/view URL serves an HTML viewer, not the PDF;
    the uc?export=download form serves the bytes. Same idea for Dropbox's
    ?dl=0 preview links.
    """
    url = url.strip()

    drive = re.search(r"drive\.google\.com/file/d/([A-Za-z0-9_-]+)", url)
    if drive:
        return f"https://drive.google.com/uc?export=download&id={drive.group(1)}"

    drive_open = re.search(r"drive\.google\.com/open\?id=([A-Za-z0-9_-]+)", url)
    if drive_open:
        return f"https://drive.google.com/uc?export=download&id={drive_open.group(1)}"

    if "dropbox.com" in url:
        return re.sub(r"[?&]dl=0", "?dl=1", url) if "dl=0" in url else url

    if "github.com" in url and "/blob/" in url:
        return url.replace("github.com", "raw.githubusercontent.com").replace("/blob/", "/")

    return url


def _filename_from(url: str, headers) -> str:
    disposition = headers.get("Content-Disposition", "") if headers else ""
    match = re.search(r'filename\*?=(?:UTF-8\'\')?"?([^";]+)"?', disposition)
    if match:
        return urllib.parse.unquote(match.group(1).strip())

    path = urllib.parse.urlparse(url).path
    name = urllib.parse.unquote(path.rsplit("/", 1)[-1]) if path else ""
    if name and "." in name:
        return name
    return "downloaded_resume.pdf"


def fetch_document(url: str, timeout: int = TIMEOUT_SECONDS) -> FetchedFile:
    """Download one document, or raise :class:`FetchError` with a clear reason."""
    raw_url = url.strip()
    if not raw_url:
        raise FetchError("Empty URL.")

    target = normalise_share_url(raw_url)
    parsed = urllib.parse.urlparse(target)
    if parsed.scheme not in ("http", "https"):
        raise FetchError(f"Only http/https URLs are supported (got '{parsed.scheme or 'no scheme'}').")

    request = urllib.request.Request(target, headers={"User-Agent": _UA})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            declared = response.headers.get("Content-Length")
            if declared and int(declared) > MAX_BYTES:
                raise FetchError(f"File is larger than {MAX_BYTES // 1024 // 1024} MB.")
            data = response.read(MAX_BYTES + 1)
            name = _filename_from(response.geturl(), response.headers)
            content_type = (response.headers.get("Content-Type") or "").lower()
    except urllib.error.HTTPError as exc:
        raise FetchError(
            f"Server returned HTTP {exc.code} ({exc.reason}). "
            "If this is a private share link, make it public or upload the file instead."
        ) from exc
    except urllib.error.URLError as exc:
        raise FetchError(f"Could not reach the URL: {exc.reason}") from exc
    except (TimeoutError, OSError) as exc:
        raise FetchError(f"Download failed: {exc}") from exc

    if len(data) > MAX_BYTES:
        raise FetchError(f"File is larger than {MAX_BYTES // 1024 // 1024} MB.")
    if not data:
        raise FetchError("The URL returned an empty file.")

    # A share/preview page returns HTML, not a PDF - catch it here with an
    # actionable message rather than letting the parser report "0 pages".
    looks_html = data[:1024].lstrip()[:15].lower().startswith((b"<!doctype html", b"<html"))
    if looks_html or "text/html" in content_type:
        raise FetchError(
            "That URL returned a web page, not a file. Use a direct link to the "
            "PDF (one that downloads the file), or upload it instead."
        )

    is_pdf = data[:5] == b"%PDF-"
    if not is_pdf and not name.lower().endswith((".txt", ".md", ".markdown")):
        # Trust the magic bytes over the extension, but keep text files working.
        if "text/plain" not in content_type:
            raise FetchError(
                f"'{name}' does not look like a PDF or a text file "
                f"(content-type: {content_type or 'unknown'})."
            )
    if is_pdf and not name.lower().endswith(".pdf"):
        name += ".pdf"

    return FetchedFile(name=name, data=data, source_url=raw_url)


def fetch_many(urls: list[str]) -> tuple[list[FetchedFile], list[tuple[str, str]]]:
    """Fetch several URLs. Returns (successes, [(url, error_message), ...]).

    One bad link must never abort the batch - the caller shows the failures
    alongside the resumes that did download.
    """
    fetched: list[FetchedFile] = []
    errors: list[tuple[str, str]] = []
    for url in urls:
        url = url.strip()
        if not url:
            continue
        try:
            fetched.append(fetch_document(url))
        except FetchError as exc:
            errors.append((url, str(exc)))
    return fetched, errors


def parse_url_list(text: str) -> list[str]:
    """Split a textarea blob into URLs (newline, comma or whitespace separated)."""
    return [u for u in re.split(r"[\s,]+", text or "") if u.strip()]
