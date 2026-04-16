"""Reference checker for ECCV-format (Springer LNCS) PDF submissions.

ECCV/Springer LNCS reference format:
  N. Surname, F., Surname, F.: Title text. In: Venue (Year)
  N. Surname, F., et al.: Title. arXiv preprint arXiv:XXXX (Year)
  N. Organization: Title. https://url (Year)

Usage is identical to reference_check.py (same CLI flags), but the
PDF parser is tuned for ECCV/LNCS numbered references rather than
ACM TOG author-year references.
"""

import argparse
import sys
import os
import re
import glob
import logging

# Import all shared utilities from the original checker
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
# ECCV-specific helpers
# ---------------------------------------------------------------------------

def _strip_eccv_line_numbers(text):
    """Remove ECCV submission line-number annotations embedded in the text.

    ECCV blind submissions annotate every line with a pair of identical
    3-4 digit numbers, e.g. ``474 474``, so that reviewers can cite
    specific lines.  pypdf extracts these inline and they must be removed
    before parsing.

    The numbers may be directly attached to the preceding word
    (e.g. ``References471 471``) so we do not rely on a leading word
    boundary.
    """
    # Matches "NNN NNN" or "NNNN NNNN" where both numbers are the same.
    # Trailing \b ensures we only strip when the second number ends at a
    # word boundary (not mid-number).
    text = re.sub(r'(\d{3,4})\s+\1\b', '', text)
    return text


def _strip_lncs_page_headers(text):
    """Remove LNCS/ECCV running page headers and footers.

    ECCV submission pages carry a header like::

        ECCV 2026 Submission #8157 15
        14 ECCV 2026 Submission #8157

    pypdf extracts this inline between reference entries and must be
    removed so it does not contaminate reference blocks.

    We use ``[ \\t]`` (horizontal space only) to avoid consuming newlines
    and accidentally eating the reference number that follows the header.
    """
    # Handles both orderings.  [ \t] instead of \s prevents crossing newlines.
    text = re.sub(
        r'[ \t]*\d{0,3}[ \t]*ECCV[ \t]+\d{4}[ \t]+Submission[ \t]+#\d+'
        r'(?:[ \t]+\d{1,3})?',
        '',
        text,
    )
    return text


def _normalize_broken_url(text):
    """Collapse space-fragmented URLs introduced by PDF line breaks.

    pypdf sometimes yields ``https: //example.com / path`` when a URL is
    broken across lines.  We collapse internal whitespace inside URL-like
    tokens so that the URL-removal regex can match them.
    """
    # Match "https?:  // ..." with arbitrary internal spaces and collapse them
    def _collapse(m):
        return re.sub(r'\s+', '', m.group(0))

    return re.sub(
        r'https?\s*:\s*/\s*/\s*\S[^\s(]*(?:\s+[^\s(]+)*',
        _collapse,
        text,
        flags=re.IGNORECASE,
    )


def find_references_section_eccv(text):
    """Locate the References section in an ECCV/Springer LNCS paper.

    ECCV submissions can have the header as plain ``References`` or with
    embedded line-number annotations like ``References471 471``.
    """
    text = text.replace('\r\n', '\n')

    # Strip line-number annotations before searching for the header
    cleaned = _strip_eccv_line_numbers(text)

    header_re = re.compile(
        r'\n\s*\d*\s*(?:References|Bibliography)\s*\n', re.IGNORECASE
    )

    best_pos = -1
    for match in header_re.finditer(cleaned):
        if match.start() > best_pos:
            best_pos = match.end()

    if best_pos != -1:
        return cleaned[best_pos:]

    # Fallback: look for "1. " near the end of the document
    logger.warning(
        "Explicit 'References' header not found. "
        "Attempting to detect reference list structure."
    )
    match1 = re.search(r'\n1\.\s+', cleaned)
    if match1 and match1.start() > len(cleaned) * 0.5:
        return cleaned[match1.start():]

    return ""


def _split_title_from_venue(rest_no_year):
    """Split ``rest_no_year`` into (title, venue) at the title-venue boundary.

    In ECCV refs the structure after the author colon is:
      ``Title text. In: Conference``  or
      ``Title text. Journal/Acronym``  or
      ``Title text. arXiv preprint arXiv:XXXX``  or
      ``Title text. https://url``
    """
    # Explicit venue markers (ordered by specificity)
    patterns = [
        r'\.\s+In:\s+',           # ". In: Conference"
        r'\.\s+arXiv\b',          # ". arXiv preprint ..."
        r'\.\s*https?://',        # ". https://..." (no space when broken across line)
    ]
    for pat in patterns:
        m = re.search(pat, rest_no_year, re.IGNORECASE)
        if m:
            return rest_no_year[:m.start()].strip()

    # General: last ". " separates title sentence from venue sentence
    pos = rest_no_year.rfind('. ')
    if pos > 0:
        potential_venue = rest_no_year[pos + 2:]
        # Only split here if what follows looks like a short venue label
        # (avoids splitting inside a long title that ends without a venue)
        if len(potential_venue) < 80:
            return rest_no_year[:pos].strip()

    return rest_no_year.strip()


def extract_references_list_eccv(ref_section_text):
    """Extract structured references from an ECCV/Springer LNCS reference block.

    Returns a list of dicts with keys: id, authors, year, title, text, raw.
    Compatible with ``check_reference`` from reference_check.py.
    """
    # 1. Remove line-number annotations
    text = _strip_eccv_line_numbers(ref_section_text)

    # 2. Remove LNCS page headers/footers that bleed between reference entries
    text = _strip_lncs_page_headers(text)

    # 3. Reconnect words hyphenated across line breaks
    text = re.sub(r'-\s*\n\s*', '', text)

    # 4. Normalize whitespace (collapse multiple newlines)
    text = re.sub(r'\n+', '\n', text)

    # 4. Split into individual reference blocks on "N. " at start of line
    #    We prepend a newline so the first reference is also captured.
    entries = re.split(r'\n(\d+)\.\s+', '\n' + text.strip())
    # entries layout: [pre-text, num1, block1, num2, block2, ...]

    refs = []
    for i in range(1, len(entries), 2):
        if i + 1 >= len(entries):
            break
        num = entries[i]
        block = entries[i + 1].strip()
        block = ' '.join(block.split())   # collapse all whitespace

        if not block:
            continue

        # ----------------------------------------------------------------
        # Collapse space-fragmented URLs before any further processing
        # ----------------------------------------------------------------
        block = _normalize_broken_url(block)

        # ----------------------------------------------------------------
        # Extract year: take the LAST "(YYYY)" in the block.
        # We use findall rather than anchoring to $ so that stray page
        # header text after the year does not hide it.
        # ----------------------------------------------------------------
        year_matches = list(re.finditer(r'\((\d{4})\)', block))
        if year_matches:
            year_match = year_matches[-1]
            year = year_match.group(1)
        else:
            year_match = None
            year = ''

        # ----------------------------------------------------------------
        # Split authors from rest at the colon that follows the author list.
        # ECCV authors end with "Initial." or "al." before ":"
        # e.g. "Feng, Y.:" or "et al.:"
        # ----------------------------------------------------------------
        author_match = re.match(
            r'^((?:[^:]+?(?:et al\.|[A-Z][a-z]*\.))\s*):\s+(.+)$',
            block, re.DOTALL
        )
        if author_match:
            authors_raw = author_match.group(1).strip()
            rest = author_match.group(2).strip()
        else:
            # Fallback for organization names without initials (e.g. "Anthropic:")
            colon_pos = block.find(': ')
            if colon_pos > 0:
                authors_raw = block[:colon_pos].strip()
                rest = block[colon_pos + 2:].strip()
            else:
                authors_raw = ''
                rest = block

        # Remove "et al." from author string
        authors_raw = re.sub(r',?\s*et al\.?', '', authors_raw).strip().rstrip(',').strip()

        # ----------------------------------------------------------------
        # Extract title from rest (everything between ":" and venue/year)
        # ----------------------------------------------------------------
        # Remove from the year marker to end of block (drops trailing page
        # header garbage as well as the year itself)
        if year_match:
            # year_match position is relative to `block`; adjust for author prefix
            rest_year_pos = rest.rfind('(' + year + ')')
            if rest_year_pos >= 0:
                rest_no_year = rest[:rest_year_pos].strip().rstrip('.').strip()
            else:
                rest_no_year = rest.strip()
        else:
            rest_no_year = rest.strip()

        # Remove any bare URLs (they may remain after year removal)
        rest_no_url = re.sub(r'https?://\S*', '', rest_no_year).strip()

        title = _split_title_from_venue(rest_no_url if rest_no_url else rest_no_year)

        # Repair ligature artifacts and clean up
        title = repair_ligatures(title)
        title = title.rstrip('.')
        title = ' '.join(title.split())

        full_ref_text = f"{authors_raw}: {title} ({year})"

        refs.append({
            'id': f"ref_{num}",
            'authors': authors_raw,
            'year': year,
            'title': title,
            'text': full_ref_text,
            'raw': block,
        })

    return refs


# ---------------------------------------------------------------------------
# Main run function (mirrors reference_check.run_check_on_file)
# ---------------------------------------------------------------------------

def run_check_on_file(url, submission_id=None, title=None, use_local=False):
    """Run reference verification on a single ECCV-format PDF."""
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
        ref_section = find_references_section_eccv(full_text)

        if not ref_section:
            log_print(
                "ERROR: Could not locate References section. "
                "Formatting might be non-standard."
            )
            return "\n".join(output_lines)

        references = extract_references_list_eccv(ref_section)
        log_print(f"Extracted {len(references)} references.")

        if not references:
            log_print("WARNING: Reference section found but no references extracted.")
            return "\n".join(output_lines)

        fake_refs = []
        swapped_name_refs = []

        log_print("Verifying references against Crossref/DBLP/ArXiv...")
        log_print(f"{'ID':<8} {'Status':<10} {'Details'}")
        log_print("-" * 60)

        for ref in references:
            valid, hit, query_details, names_swapped = check_reference(ref)

            status = "OK" if valid else "not found"
            if not valid:
                ref['failed_queries'] = query_details
                ref['closest_match'] = hit
                fake_refs.append(ref)
                log_print(
                    f"[{ref['id']}] {status:<10} "
                    f"Queries tried: {query_details}"
                )
            else:
                log_print(
                    f"[{ref['id']}] {status:<10} "
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
                log_print(f"\n[{ref['id']}] {ref['text']}")
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
                log_print(f"\n[{ref['id']}] {ref['text']}")
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
        description="Detect potential fake references in ECCV/Springer LNCS PDF submissions."
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

    # Propagate mailto to the shared module
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
