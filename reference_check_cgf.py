"""Reference checker for CGF-format (Eurographics / Computer Graphics Forum) PDF submissions.

CGF/Eurographics reference format:
  [Key] SURNAME, FIRSTNAME, SURNAME2, FIRSTNAME2, and SURNAME3, FIRSTNAME3.
        "Title". Journal/Conf Volume.Issue (Year), Pages.
  [Key] SURNAME, FIRSTNAME, et al.
        Title. Year. arXiv:XXXX.XXXXX
  [Key] SURNAME, FIRSTNAME. Title. URL:https://... Year.

Usage is identical to reference_check_eccv.py (same CLI flags), but the
PDF parser is tuned for CGF/Eurographics [Key]-style references.
"""

import argparse
import sys
import os
import re
import logging

from reference_check import (
    normalize_styled_unicode, repair_ligatures, normalize_title,
    normalize_name, normalize_for_comparison, _normalize_alnum,
    _get_name_parts, download_pdf, extract_text,
    is_arxiv_ref, extract_arxiv_id, search_arxiv, search_dblp,
    search_crossref, is_match, calculate_similarity, is_venue_title,
    check_reference, parse_bib_file, parse_bbl_file,
    run_check_on_bib, process_batch_txt, process_pdf_folder,
    CROSSREF_MAILTO, _format_hit,
)
import reference_check as _rc

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# CGF-specific helpers
# ---------------------------------------------------------------------------

def _strip_cgf_page_headers(text):
    """Remove CGF/Eurographics running page headers and footers.

    CGF submission pages carry footers like:
        submitted to EUROGRAPHICS 2026.
        12 of 12FPsecond-1051 / Shape2Script
        FPsecond-1051 / Shape2Script11 of 12
    """
    # "submitted to CONFERENCE YYYY."
    text = re.sub(
        r'submitted[ \t]+to[ \t]+\S.*?(?:\d{4}\.?)',
        '',
        text,
        flags=re.IGNORECASE,
    )
    # "NN of NNSubmissionID / PaperTitle" or "SubmissionID / PaperTitle NN of NN"
    text = re.sub(
        r'\d+[ \t]+of[ \t]+\d+[A-Za-z].*?(?=\n|\[)',
        '',
        text,
    )
    text = re.sub(
        r'[A-Za-z]\S+[ \t]/[ \t]\S+\d+[ \t]+of[ \t]+\d+',
        '',
        text,
    )
    return text


def _normalize_broken_url(text):
    """Collapse space-fragmented URLs introduced by PDF line breaks.

    pypdf sometimes yields ``https : / / example . com / path`` for URLs
    broken across lines.  We collapse internal whitespace so URL-removal
    regex can match them.
    """
    def _collapse(m):
        return re.sub(r'\s+', '', m.group(0))

    return re.sub(
        r'https?\s*:\s*/\s*/\s*\S[^\s(]*(?:\s+[^\s(]+)*',
        _collapse,
        text,
        flags=re.IGNORECASE,
    )


def _normalize_broken_doi(text):
    """Collapse space-fragmented DOIs like ``10 . 5281 / zenodo . 3908559``."""
    def _collapse(m):
        return re.sub(r'\s+', '', m.group(0))

    return re.sub(
        r'10\s*\.\s*\d{4,}\s*/\s*\S[^\s,)]*(?:\s+[^\s,)]+)*',
        _collapse,
        text,
    )


def find_references_section_cgf(text):
    """Locate the References section in a CGF/Eurographics paper."""
    text = text.replace('\r\n', '\n')

    header_re = re.compile(
        r'\n\s*(?:References|Bibliography)\s*\n', re.IGNORECASE
    )

    best_pos = -1
    for match in header_re.finditer(text):
        if match.start() > best_pos:
            best_pos = match.end()

    if best_pos != -1:
        return text[best_pos:]

    # Fallback: look for "[Xxx00]" citation key pattern near end of document
    logger.warning(
        "Explicit 'References' header not found. "
        "Attempting to detect reference list structure."
    )
    match1 = re.search(r'\n\[[A-Z][A-Za-z*]+\d{2}\]', text)
    if match1 and match1.start() > len(text) * 0.5:
        return text[match1.start():]

    return ""


def _strip_back_references(text):
    """Remove trailing in-paper page back-references like ' 2, 3.' or '10.'

    CGF references end with a list of page numbers pointing back into the
    paper body, e.g. '...ACM Trans. on Graphics 29.4 (2010), 104:1–10 3.'
    or '...Vision & Pattern Recognition. 2022, 9871–9880 3.' These must be
    removed so they don't contaminate title/venue parsing.
    """
    # Trailing: optional whitespace, then digits/commas/spaces (back-refs), then end
    return re.sub(r'\s+\d[\d,\s]*\.?\s*$', '', text.rstrip())


# Unicode curly/typographic quotes used in CGF PDFs, plus ASCII fallback
_OPEN_QUOTE = r'["“„]'   # " or „ or "
_CLOSE_QUOTE = r'["”“]'  # " or "


def _split_cgf_author_title(block):
    """Split a CGF reference block into (authors_raw, rest_of_entry).

    CGF author patterns (after [Key]):
      SURNAME, FIRSTNAME, SURNAME2, FIRSTNAME2, and SURNAME3, FIRSTNAME3. "Title"
      SURNAME, FIRSTNAME, et al. Title
      SURNAME, FIRSTNAME. Title
    """
    # Case 1: "et al." marks end of author list
    m = re.match(r'^(.*?et al\.)\s*(.*)', block, re.DOTALL)
    if m:
        return m.group(1).strip(), m.group(2).strip()

    # Case 2: quoted title (handles ASCII " and Unicode curly quotes " ")
    # Matches '. "' with optional space between period and opening quote
    m = re.search(r'\.\s*' + _OPEN_QUOTE, block)
    if m:
        return block[:m.start()].strip(), block[m.end() - 1:].strip()

    # Case 3: unquoted title — find first '.' after which the next token is
    # not an all-caps word (to avoid splitting within author list), including
    # when the period is directly adjacent to the first title word (no space).
    for m in re.finditer(r'\.', block):
        after = block[m.end():]
        # Skip leading whitespace to find the first real character
        after_stripped = after.lstrip()
        first_word_m = re.match(r'([A-Za-z]+)', after_stripped)
        if not first_word_m:
            continue
        first_word = first_word_m.group(1)
        # If the first word after '.' is not ALL-CAPS, it's the title start
        if not first_word.isupper():
            return block[:m.start()].strip(), after_stripped.strip()
        # Also treat as title start if it looks like a URL, arXiv, or DOI
        if re.match(r'(?:https?://|arXiv:|DOI:)', after_stripped, re.IGNORECASE):
            return block[:m.start()].strip(), after_stripped.strip()

    # Fallback: no clear split found
    return '', block


def _extract_quoted_title(rest):
    """Extract a double-quoted title from the start of *rest* (handles curly quotes)."""
    m = re.match(_OPEN_QUOTE + r'(.*?)' + _CLOSE_QUOTE, rest, re.DOTALL)
    if m:
        return m.group(1).strip()
    return None


def _extract_unquoted_title(rest):
    """Extract the title from an unquoted CGF entry.

    These are arXiv preprints or books where the title is not wrapped
    in double quotes.  The title runs up to the year or venue marker.
    """
    # Strip year marker and everything after
    rest_no_year = re.sub(r'\(\d{4}\).*', '', rest).strip()
    # Strip trailing URL or DOI
    rest_no_url = re.sub(r'(?:https?://|DOI:|arXiv:)\S+', '', rest_no_year).strip()
    # Strip publication venue clues: "YYYY." standalone year without parens
    rest_no_url = re.sub(r'\b\d{4}\b\.?\s*$', '', rest_no_url).strip()
    # The title is everything up to the first period that ends a sentence
    pos = rest_no_url.find('. ')
    if pos > 0:
        return rest_no_url[:pos].strip()
    return rest_no_url.rstrip('.').strip()


def extract_references_list_cgf(ref_section_text):
    """Extract structured references from a CGF/Eurographics reference block.

    Returns a list of dicts with keys: id, authors, year, title, text, raw.
    Compatible with ``check_reference`` from reference_check.py.
    """
    text = ref_section_text

    # 1. Remove CGF page headers/footers
    text = _strip_cgf_page_headers(text)

    # 2. Reconnect words hyphenated across line breaks
    text = re.sub(r'-\s*\n\s*', '', text)

    # 3. Collapse multiple newlines
    text = re.sub(r'\n+', '\n', text)

    # 4. Split into individual reference blocks on "[Key]" at start of line
    entries = re.split(r'\n(\[[A-Z][A-Za-z*0-9]+\d{2}\])\s*', '\n' + text.strip())
    # layout: [pre-text, key1, block1, key2, block2, ...]

    refs = []
    for i in range(1, len(entries), 2):
        if i + 1 >= len(entries):
            break
        key = entries[i]          # e.g. "[BWS10]"
        block = entries[i + 1].strip()
        block = ' '.join(block.split())   # collapse all internal whitespace

        if not block:
            continue

        # Collapse space-fragmented URLs and DOIs
        block = _normalize_broken_url(block)
        block = _normalize_broken_doi(block)

        # Remove trailing back-references before any further processing
        block = _strip_back_references(block)

        # Extract year: last "(YYYY)" in the block
        year_matches = list(re.finditer(r'\((\d{4})\)', block))
        if year_matches:
            year = year_matches[-1].group(1)
        else:
            # Try bare year at end for arXiv-style "2023."
            ym = re.search(r'\b(20\d{2}|19\d{2})\b', block)
            year = ym.group(1) if ym else ''

        # Split author block from rest of entry
        authors_raw, rest = _split_cgf_author_title(block)

        # Normalize author string: remove "et al.", "and", trailing commas
        authors_clean = re.sub(r',?\s*et al\.?', '', authors_raw).strip()
        authors_clean = re.sub(r'\band\b', ',', authors_clean)
        authors_clean = re.sub(r',\s*,', ',', authors_clean).strip().strip(',').strip()

        # Extract title
        if rest and rest[0] in '"“„':
            title = _extract_quoted_title(rest) or ''
        else:
            title = _extract_unquoted_title(rest)

        title = repair_ligatures(title)
        title = title.rstrip('.')
        title = ' '.join(title.split())

        full_ref_text = f"{key} {authors_clean}: {title} ({year})"

        refs.append({
            'id': key,
            'authors': authors_clean,
            'year': year,
            'title': title,
            'text': full_ref_text,
            'raw': block,
        })

    return refs


# ---------------------------------------------------------------------------
# Main run function (mirrors reference_check_eccv.run_check_on_file)
# ---------------------------------------------------------------------------

def run_check_on_file(url, submission_id=None, title=None, use_local=False):
    """Run reference verification on a single CGF-format PDF."""
    output_lines = []

    def log_print(s=""):
        output_lines.append(str(s))
        print(s)

    log_print(f"Checking Submission {submission_id}: {title}")
    log_print(f"URL/Path: {url}")
    log_print("=" * 60)

    if use_local:
        pdf_path = url
    else:
        pdf_path = download_pdf(url)

    try:
        full_text = extract_text(pdf_path)
        ref_section = find_references_section_cgf(full_text)

        if not ref_section:
            log_print(
                "ERROR: Could not locate References section. "
                "Formatting might be non-standard."
            )
            return "\n".join(output_lines)

        references = extract_references_list_cgf(ref_section)
        log_print(f"Extracted {len(references)} references.")

        if not references:
            log_print("WARNING: Reference section found but no references extracted.")
            return "\n".join(output_lines)

        fake_refs = []
        swapped_name_refs = []

        log_print("Verifying references against Crossref/DBLP/ArXiv...")
        log_print(f"{'ID':<12} {'Status':<10} {'Details'}")
        log_print("-" * 60)

        for ref in references:
            valid, hit, query_details, names_swapped = check_reference(ref)

            status = "OK" if valid else "not found"
            if not valid:
                ref['failed_queries'] = query_details
                ref['closest_match'] = hit
                fake_refs.append(ref)
                log_print(
                    f"{ref['id']:<12} {status:<10} "
                    f"Queries tried: {query_details}"
                )
            else:
                log_print(
                    f"{ref['id']:<12} {status:<10} "
                    f"Detected: {hit['info'].get('title', '')[:40]}... "
                    f"(Query: {query_details})"
                )
                if names_swapped:
                    swapped_name_refs.append((ref, hit))

        log_print("-" * 60)
        if fake_refs:
            log_print(
                f"\nFound {len(fake_refs)} references that could not be matched:"
            )
            for ref in fake_refs:
                log_print(f"\n{ref['id']} {ref['text']}")
                log_print(f"Queries tried: {ref.get('failed_queries', 'N/A')}")
                closest = ref.get('closest_match')
                if closest:
                    log_print(f"Closest match:\n{_format_hit(closest.get('info', {}))}")
        else:
            log_print("\nAll references verified successfully.")

        if swapped_name_refs:
            log_print(
                f"\nFound {len(swapped_name_refs)} references with authors' "
                f"names backwards:"
            )
            for ref, hit in swapped_name_refs:
                log_print(f"\n{ref['id']} {ref['text']}")
                log_print(f"Closest match:\n{_format_hit(hit.get('info', {}))}")

    finally:
        if not use_local and os.path.exists(pdf_path):
            os.remove(pdf_path)

    return "\n".join(output_lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Detect potential fake references in CGF/Eurographics PDF submissions."
    )
    parser.add_argument(
        "source", nargs='?',
        help=(
            "URL to the PDF file OR path to a .txt file containing URLs "
            "OR path to a folder of PDF files"
        ),
    )
    parser.add_argument(
        "--bib", metavar="BIB_FILE",
        help="Path to a .bib or .bbl file to check directly (skips PDF parsing)",
    )
    parser.add_argument(
        "--mailto",
        help="Email for Crossref polite pool (optional, faster queries)",
    )
    args = parser.parse_args()

    _rc.CROSSREF_MAILTO = args.mailto

    if args.bib:
        bib_path = args.bib
        if not os.path.isfile(bib_path):
            logger.error(f"Bib file not found: {bib_path}")
            sys.exit(1)
        log_content = run_check_on_bib(bib_path)
        submission_id = os.path.splitext(os.path.basename(bib_path))[0]
        log_dir = os.path.join(os.getcwd(), "reference_checks")
        os.makedirs(log_dir, exist_ok=True)
        log_path = os.path.join(log_dir, f"{submission_id}.log")
        with open(log_path, 'w', encoding='utf-8') as f_out:
            f_out.write(log_content)
        return

    if not args.source:
        parser.error("source is required unless --bib is specified")

    if os.path.isdir(args.source):
        log_dir = os.path.join(os.getcwd(), "reference_checks")
        os.makedirs(log_dir, exist_ok=True)
        process_pdf_folder(args.source, log_dir)

    elif args.source.endswith(".txt") and os.path.isfile(args.source):
        log_dir = os.path.join(os.getcwd(), "reference_checks")
        os.makedirs(log_dir, exist_ok=True)
        process_batch_txt(args.source, log_dir)

    else:
        use_local = not args.source.startswith("http")
        log_content = run_check_on_file(
            args.source, "SingleCheck", "Manual Run", use_local=use_local
        )
        if use_local:
            filename = os.path.basename(args.source)
            submission_id = os.path.splitext(filename)[0]
            log_dir = os.path.join(os.getcwd(), "reference_checks")
            os.makedirs(log_dir, exist_ok=True)
            log_path = os.path.join(log_dir, f"{submission_id}.log")
            with open(log_path, 'w', encoding='utf-8') as f_out:
                f_out.write(log_content)


if __name__ == "__main__":
    main()
