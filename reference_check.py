import argparse
import sys
import os
import re
import requests
import tempfile
import logging
from difflib import SequenceMatcher
import time
import xml.etree.ElementTree as ET
import glob
from pypdf import PdfReader
import unicodedata

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

DBLP_API_URL = "https://dblp.org/search/publ/api"
ARXIV_API_URL = "http://export.arxiv.org/api/query"

CROSSREF_API_URL = "https://api.crossref.org/works"
CROSSREF_MAILTO = None  # if set, gets you into the "polite" pool (faster)

def normalize_styled_unicode(text):
    text = text.replace('\u2018', "'").replace('\u2019', "'")  # curly single quotes
    text = text.replace('\u201c', '"').replace('\u201d', '"')  # curly double quotes
    # Strip LaTeX accents: \'e, \"o, \`a, \^i, \~n, \=u, \.z, \v{c}, \c{c} etc.
    text = re.sub(r"\\{1,2}['\"`^~=.vc]\{?(\w)\}?", r'\1', text)
    text = text.replace("’", "'")
    text = text.replace("&apos;", "'")
    # NFKD decompose, then strip combining marks (accents)
    text = unicodedata.normalize('NFKD', text)
    text = ''.join(c for c in text if unicodedata.category(c) != 'Mn')
    return text

LIGATURES = ['ff', 'fi', 'fl', 'tt', 'ffi', 'ffl']

# common words that might have those ligatures in them
KNOWN_WORDS = {"diffusion", "different", "field", "finite", "flexible", "flow",
               "influence", "efficient", "offline", "profile", "filter", "rectified",
               "refine", "reflect", "simplified", "simplification", "shuffle", "traffic", "affine", "tutte"}

def repair_ligatures(text):
    def fix_word(m):
        prefix, bad_char, suffix = m.group(1), m.group(2), m.group(3)
        for lig in LIGATURES:
            candidate = prefix + lig + suffix
            for w in KNOWN_WORDS:
                if candidate.lower().startswith(w) or candidate.lower().endswith(w):
                    return candidate
        return m.group(0)
    
    # iterate until no more changes (handles multiple ligatures per word)
    prev = None
    while prev != text:
        prev = text
        # ligatures are sometimes read as !, #, or a unicode character like \uffff
        text = re.sub(r'(\w*)([!#\ufffd-\uffff])(\w*)', fix_word, text)
    return text

def normalize_title(text):
    # Strip HTML tags (e.g., <i>, </i>, <sub>, <sup>)
    text = re.sub(r'<[^>]+>', '', text)
    text = normalize_styled_unicode(text)
    text = repair_ligatures(text)
    text = text.replace('first ed', 'first edition')
    text = text.replace('1 ed', 'first edition')
    text = text.replace('1st ed', 'first edition')
    text = text.replace('second ed', 'second edition')
    text = text.replace('2 ed', 'second edition')
    text = text.replace('2nd ed', 'second edition')
    text = text.replace('third ed', 'third edition')
    text = text.replace('3 ed', 'third edition')
    text = text.replace('3rd ed', 'third edition')
    text = text.replace('edition.', 'edition')
    text = text.replace('-', ' ')
    text = text.replace('LeastSquare', 'Least Square')
    return text

def normalize_name(text):
    # Strip LaTeX accents: \'e, \"o, \`a, \^i, \~n, \=u, \.z, \v{c}, \c{c} etc.
    text = re.sub(r"\\['\"`^~=.vc]\{?(\w)\}?", r'\1', text)
    # ß → ss (before NFKD, which doesn't expand ß)
    text = text.replace('ß', 'ss')
    # NFKD decompose, then strip combining marks (accents)
    text = unicodedata.normalize('NFKD', text)
    text = ''.join(c for c in text if unicodedata.category(c) != 'Mn')
    return text

def normalize_for_comparison(text):
    text = normalize_name(text)
    text = normalize_title(text)
    return text.lower()

def download_pdf(url):
    try:
        logger.info(f"Downloading PDF from {url}...")
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers, stream=True)
        response.raise_for_status()
        
        fd, path = tempfile.mkstemp(suffix=".pdf")
        with os.fdopen(fd, 'wb') as tmp:
            for chunk in response.iter_content(chunk_size=8192):
                tmp.write(chunk)
        return path
    except Exception as e:
        logger.error(f"Failed to download PDF: {e}")
        sys.exit(1)

def extract_text(pdf_path):
    try:
        logger.info("Parsing PDF content...")
        reader = PdfReader(pdf_path)
        text = []
        for page in reader.pages:
            t = page.extract_text()
            if t:
                text.append(t)
        return "\n".join(text)
    except Exception as e:
        logger.error(f"Failed to parse PDF: {e}")
        return ""

def find_references_section(text):
    # Normalize newlines
    text = text.replace('\r\n', '\n')
    
    # Common headers for References section
    # We look for the last occurrence of these headers, 
    # as sometimes "References" appears in TOC or text.
    headers = [
        r'\n\s*REFERENCES\s*\n',
        r'\n\s*References\s*\n',
        r'\n\s*Bibliography\s*\n'
    ]
    
    best_pos = -1
    patterns_re = [re.compile(h, re.IGNORECASE) for h in headers]
    for p in patterns_re:
        for match in p.finditer(text):
            # strict check: uppercase or isolated
            if match.start() > best_pos:
                best_pos = match.end()
    
    if best_pos != -1:
        return text[best_pos:]
    
    # Fallback: Search for sequence of [1], [2], [3] near end of file
    # If we find [1] and later [2], assume start is at [1]
    logger.warning("Explicit 'References' header not found. Attempting to detect reference list structure.")
    
    match1 = re.search(r'\[1\]', text)
    if match1:
        # Check if we are in the last 20% of the document? Or just assume valid.
        if match1.start() > len(text) * 0.5:
             return text[match1.start():]
             
    return ""

def clean_reference_text(ref_text):
    """Clean up extracted reference text (remove newlines, extra spaces)."""
    return " ".join(ref_text.split())

def extract_references_list(ref_section_text):
    """
    Extracts structured references (Authors, Year, Title) from the text block.
    """
    # Use raw text to preserve newlines for boundary detection
    text = ref_section_text
    
    # Regex for Year Anchor: . 2005. 
    # We allow flexible whitespace/newlines
    year_pattern = re.compile(r'\.\s+((?:19|20)\d{2})\.\s+', re.DOTALL)
    
    matches = list(year_pattern.finditer(text))
    
    if not matches:
        return []

    refs = []
    for i, match in enumerate(matches):
        year = match.group(1)
        year_end = match.end()
        year_start = match.start()
        
        # TITLE extraction
        # Take text from year_end until the next period
        rest_of_text = text[year_end:]
        next_period = rest_of_text.find('.')
        
        if next_period != -1:
            title = rest_of_text[:next_period].strip()
        else:
            title = rest_of_text.strip() # End of string case
        
        # Remove hyphenation artifacts: "word- " or "word-\n" -> "word"
        title = title.replace('-\n', '').replace('- ', '')
        
        # Normalize title (remove newlines etc) for clean output
        title = " ".join(title.split())

        # AUTHOR extraction
        # Look backwards from year_start.
        
        # We assume authors start after the previous reference ended.
        # Heuristic: Previous reference ends with a newline, often followed by a URL or empty line.
        # But specifically, we look for the line break that separates two refs.
        
        cursor = year_start
        start_of_authors = 0
        
        # We iterate backwards looking for newlines
        while cursor > 0:
            last_newline = text.rfind('\n', 0, cursor)
            
            if last_newline == -1:
                start_of_authors = 0
                break
                
            # Check the line BEFORE the newline
            # If it ends with a URL or Pages, or looks like the end of a ref block text.
            line_before = text[0:last_newline].strip() 
            line_after = text[last_newline:year_start].strip() # Current candidate author line
            
            # If line_before contains "doi.org" or "http", it is almost certainly the previous ref
            if "doi.org" in line_before.lower() or "http" in line_before.lower():
                start_of_authors = last_newline + 1
                break
                
            # If line_before ends with a year pattern? (unlikely if we matched unique years, but possible)
            
            # If current 'line_after' starts with a minimal indentation or looks like a start?
            # Hard to tell in plain text.
            
            # Check 'comma connection'. If line_before ends with ',' or 'and', it belongs to current ref.
            if line_before.endswith(',') or line_before.lower().endswith(' and'):
                cursor = last_newline
                continue
                
            # If line_before ends with '.' or digit, it might be the end of previous ref.
            # But "Jr." ends with dot.
            # "Vol. 4." ends with dot.
            if line_before.endswith('.'):
                # Check if it looks like an initial "A." (Capital then Dot)
                if len(line_before) >= 2 and line_before[-2].isupper() and (len(line_before) < 3 or line_before[-3] in ' ,'):
                    # Likely initial, continue back
                    cursor = last_newline
                    continue
                else:
                    # Likely end of sentence/previous ref
                    start_of_authors = last_newline + 1
                    break

            # If line_before ends with digits (pages), likely previous ref
            if line_before and line_before[-1].isdigit():
                 start_of_authors = last_newline + 1
                 break
                 
            # If we reached here, it's ambiguous. 
            # If the current accumulated authors string is getting too long (>300 chars), stop.
            if year_start - last_newline > 300:
                start_of_authors = last_newline + 1
                break
                
            # Default: Keep going back (assume wrap)
            cursor = last_newline
            
        authors_raw = text[start_of_authors:year_start]
        authors = " ".join(authors_raw.split()) # Clean newlines
        
        # Removing leading "." if it was captured from previous line end
        authors = authors.lstrip('.').strip()
        authors = authors.replace(', and', ',')
        authors = authors.replace('-\n', '').replace('- ', '')
        
        # Remove 'et al.' from authors list
        authors = re.sub(r'\bet al\.?', '', authors, flags=re.IGNORECASE).strip()
        # Clean up trailing punctuation that might remain
        authors = authors.strip(' .,')
        
        full_ref_text = f"{authors}. {year}. {title}."
        
        refs.append({
            'id': f"ref_{i+1}",
            'authors': authors,
            'year': year,
            'title': title,
            'text': full_ref_text,
            'raw': full_ref_text
        })
        
    return refs

def is_arxiv_ref(ref_text):
    return "arxiv" in ref_text.lower()

def extract_arxiv_id(text):
    # Matches patterns like arXiv:2103.12345 or arXiv:2103.12345v1
    match = re.search(r'arXiv[:\s]?(\d{4}\.\d{4,5}(?:v\d+)?)', text, re.IGNORECASE)
    if match:
        return match.group(1)
    
    # Old format
    match_old = re.search(r'arXiv[:\s]?([a-z\-]+/\d{7})', text, re.IGNORECASE)
    if match_old:
        return match_old.group(1)
        
    return None

def search_arxiv(query, query_type='ti'):
    # query_type: 'ti' (title) or 'id_list' (ids)
    params = {
        'start': 0,
        'max_results': 1
    }
    
    if query_type == 'id_list':
        params['id_list'] = query
    else:
        # ArXiv query syntax: ti:"title"
        # We need to sanitize quotes
        clean_query = query.replace('"', '').replace(':', ' ')
        params['search_query'] = f"ti:\"{clean_query}\""

    try:
        # Rate limit friendly (ArXiv requests 3s delay, but for single sequential script 1s is likely ok if not spamming)
        time.sleep(1.0) 
        resp = requests.get(ARXIV_API_URL, params=params, timeout=15)
        
        if resp.status_code != 200:
            return None
            
        # Parse XML
        root = ET.fromstring(resp.content)
        # Namespace map usually needed. Atom namespace is default.
        ns = {'atom': 'http://www.w3.org/2005/Atom'}
        entries = root.findall('atom:entry', ns)
        
        if not entries:
            return None
            
        entry = entries[0]
        title = entry.find('atom:title', ns).text
        authors = [a.find('atom:name', ns).text for a in entry.findall('atom:author', ns)]
        
        return {
            'info': {
                'title': title.replace('\n', ' ').strip(),
                'authors': {'author': [{'text': a} for a in authors]},
                'type': 'ArXiv Preprint' 
            }
        }

    except Exception as e:
        logger.error(f"ArXiv search failed: {e}")
        return None

def search_dblp(query):
    # DBLP allows searching by title string
    params = {
        'q': query,
        'format': 'json',
        'h': 5,  # Fetch top 5 hits
    }
    try:
        # Rate limit friendly
        time.sleep(0.3)
        resp = requests.get(DBLP_API_URL, params=params, timeout=10)
        if resp.status_code == 429:  # Too many requests
            wait_time = int(resp.headers.get("Retry-After", 45))
            logger.warning(f"DBLP rate limit hit, waiting {wait_time} seconds...")
            time.sleep(wait_time)
            resp = requests.get(DBLP_API_URL, params=params, timeout=10)
        
        # Check for 420 (Policy Violation / Limit Exceeded)
        if resp.status_code == 420:
            logger.warning("DBLP query limit exceeded/blocked (Status 420).")
            return []
            
        if resp.status_code != 200:
            return []
            
        data = resp.json()
        hits = data.get('result', {}).get('hits', {}).get('hit', [])
        return hits
    except Exception:
        return []

def search_crossref(query):
    """Search Crossref and return results normalized to DBLP-like hit format."""
    params = {
        'query.bibliographic': query,
        'rows': 5,
    }
    # Add mailto if provided — still works without it, just slower
    if CROSSREF_MAILTO:
        params['mailto'] = CROSSREF_MAILTO
    try:
        time.sleep(0.1)  # lighter rate limit than DBLP; polite pool is generous
        resp = requests.get(CROSSREF_API_URL, params=params, timeout=15)

        if resp.status_code == 429:
            wait_time = int(resp.headers.get("Retry-After", 30))
            logger.warning(f"Crossref rate limit hit, waiting {wait_time}s...")
            time.sleep(wait_time)
            resp = requests.get(CROSSREF_API_URL, params=params, timeout=15)

        if resp.status_code != 200:
            return []

        items = resp.json().get('message', {}).get('items', [])

        # Normalize each item into the same shape as a DBLP hit so that
        # is_match / calculate_similarity / update_best all work unchanged.
        hits = []
        for item in items:
            titles = item.get('title', [])
            title = titles[0] if titles else ''

            # Build author string: "First Last, First Last, ..."
            authors_list = item.get('author', [])
            author_names = []
            for a in authors_list:
                given = a.get('given', '')
                family = a.get('family', '')
                author_names.append(f"{given} {family}".strip())

            # Year: try published-print, then published-online, then issued
            year = ''
            for date_field in ('published-print', 'published-online', 'issued'):
                parts = item.get(date_field, {}).get('date-parts', [[]])
                if parts and parts[0] and parts[0][0]:
                    year = str(parts[0][0])
                    break

            venues = item.get('container-title', [])
            venue = venues[0] if venues else ''

            hit = {
                'source': 'crossref',
                'info': {
                    'title': title,
                    'authors': {
                        'author': [{'text': name} for name in author_names]
                    },
                    'year': year,
                    'venue': venue,
                    'type': item.get('type', ''),
                    'doi': item.get('DOI', ''),
                    'url': item.get('URL', ''),
                },
            }
            hits.append(hit)

        return hits
    except Exception:
        return []

def is_match(ref, hit):
    """
    Determines if a DBLP hit matches the reference object.
    Uses title and author overlap.
    
    Returns:
        (bool, bool): (is_match, names_swapped)
    """
    info = hit.get('info', {})
    hit_title = info.get('title', '')
    
    # --- Title Match ---
    
    ref_title = ref.get('title', '')
    ref_text = ref.get('text', '')
    
    def normalize(s):
        return "".join(c.lower() for c in normalize_for_comparison(s) if c.isalnum())
    
    norm_hit_title = normalize(hit_title)
    
    title_matches = False
    
    # Try strict title match pattern
    if len(ref_title) > 5:
        norm_ref_title = normalize(ref_title)
        # Check basic containment or fuzzy match
        if norm_hit_title == norm_ref_title or (len(norm_hit_title)>10 and norm_hit_title in norm_ref_title):
             title_matches = True
        else:
             # Fuzzy match
             sm = SequenceMatcher(None, norm_hit_title, norm_ref_title)
             match = sm.find_longest_match(0, len(norm_hit_title), 0, len(norm_ref_title))
             if match.size > len(norm_hit_title) * 0.8:
                 title_matches = True

    # If explicit title didn't match (or wasn't found), try full text
    if not title_matches:
        norm_ref_text = normalize(ref_text)
        if norm_hit_title in norm_ref_text:
             title_matches = True
        else:
             sm = SequenceMatcher(None, norm_hit_title, norm_ref_text)
             match = sm.find_longest_match(0, len(norm_hit_title), 0, len(norm_ref_text))
             if match.size > len(norm_hit_title) * 0.8:
                 title_matches = True

    if not title_matches:
        return False, False
        
    # --- Author Match ---

    # If title matches, we MUST check authors to avoid "Hallucination by common title"
    # or "Matching the wrong paper with same title".
    
    hit_authors_data = info.get('authors', {}).get('author', [])
    if isinstance(hit_authors_data, dict):
        hit_authors_data = [hit_authors_data]
        
    hit_authors = [a.get('text', '') for a in hit_authors_data if isinstance(a, dict) and 'text' in a]
    # sometimes it's just a string in rare cases or different structure, but usually list of dicts or dict
    if not hit_authors and isinstance(hit_authors_data, list):
        # Try to handle if it's just a list of strings (unlikely in DBLP API but possible)
        hit_authors = [str(a) for a in hit_authors_data]

    ref_authors_str = ref.get('authors', '')
    if not ref_authors_str:
        # If we couldn't extract authors from PDF, we rely solely on title match
        # This is a weak check but better than failing valid refs
        return True, False
    
    # Check for author implementation
    def get_name_parts(text):
        """
        Extract (first_names, surnames) from an author string.
        Returns two sets of normalized name parts.
        """
        parts = text.replace(' and ', ',').split(',')
        first_names = set()
        surnames = set()
        for p in parts:
            p = p.strip()
            if not p:
                continue
            words = p.split()
            if len(words) >= 2:
                # Assume "First [Middle...] Last" format
                first_names.add(normalize(words[0]))
                surnames.add(normalize(words[-1]))
            elif len(words) == 1:
                # Single name — could be either; add to both
                surnames.add(normalize(words[0]))
        return first_names, surnames
    
    # Collect first names and surnames from hit authors
    hit_first_names = set()
    hit_surnames = set()
    for auth in hit_authors:
        firsts, lasts = get_name_parts(auth)
        hit_first_names.update(firsts)
        hit_surnames.update(lasts)
    
    # Extract what ref claims as surnames (last word of each part)
    _, ref_surnames = get_name_parts(ref_authors_str)
    
    # Check normal match: ref surnames ∩ hit surnames
    normal_overlap = len(ref_surnames & hit_surnames)
    
    # Check swapped match: ref surnames ∩ hit first names
    swapped_overlap = len(ref_surnames & hit_first_names)
    
    if normal_overlap > 0:
        return True, False
    
    if swapped_overlap > 0:
        # Names appear swapped: what ref lists as surnames are actually first names
        return True, True
    
    # No author overlap — suspicious
    if len(norm_hit_title) > 50 and title_matches:
        # High confidence title match override? 
        # Risky for "Survey on..." titles.
        return False, False
        
    return False, False

def calculate_similarity(ref, hit):
    info = hit.get('info', {})
    hit_title = info.get('title', '')
    ref_title = ref.get('title', '')
    
    if not hit_title:
        return 0.0
        
    def normalize(s):
        return "".join(c.lower() for c in s if c.isalnum())
    
    norm_hit = normalize(hit_title)
    if len(ref_title) > 5:
        norm_ref = normalize(ref_title)
    else:
        # Fallback if title extraction failed
        norm_ref = normalize(ref.get('text', ''))
        
    return SequenceMatcher(None, norm_hit, norm_ref).ratio()

def is_venue_title(text):
    """
    Heuristic to check if a text string looks like a venue/proceedings title 
    rather than a paper title.
    """
    venue_keywords = [
        "proceedings", "conference", "symposium", "workshop", "journal", 
        "transactions", "digest", "advances in", "notes in", "handbook of"
    ]
    text_lower = text.lower()
    for kw in venue_keywords:
        if kw in text_lower:
            return True
    return False

def check_reference(ref):
    """
    Checks a single reference. 
    1. Extract potential title (heuristic)
    2. Check ArXiv ID (if present)
    3. Search DBLP
    4. Search ArXiv by title (fallback)
    """
    text = ref.get('text', '')
    title_extracted = normalize_title(ref.get('title', ''))
    
    queries_tried = []
    best_candidate = None
    best_score = 0.0

    def update_best(hit):
        nonlocal best_candidate, best_score
        score = calculate_similarity(ref, hit)
        if score > best_score:
            best_score = score
            best_candidate = hit
    
    # --- 1. ArXiv Direct Check (by ID) ---
    arxiv_id = extract_arxiv_id(text)
    if arxiv_id:
        queries_tried.append(f"ArXiv ID: {arxiv_id}")
        # If we have an ID, we trust it and check it first.
        hit = search_arxiv(arxiv_id, query_type='id_list')
        if hit:
            # For ArXiv ID lookup, we strongly trust the return if ID matches.
            # But we should still verification basic title/author match to ensure 
            # the ID wasn't just hallucinated (e.g. real ID for DIFFERENT paper).
            match, names_swapped = is_match(ref, hit)
            if match:
                return True, hit, f"ArXiv ID: {arxiv_id}", names_swapped
            else:
                 # ID found but content mismatch -> Suspicious!
                 # But maybe just bad extraction? Let's fall through to DBLP search.
                 update_best(hit)
                 pass
    
    # Decide on search query
    search_queries = []
    
    # Try to include all author surnames to improve relevance
    author_surnames_str = ""
    authors_str = ref.get('authors', '')
    if authors_str:
        # Simple extraction of surnames
        parts = authors_str.replace(' and ', ', ').split(',')
        surnames = []
        for p in parts:
            p = p.strip()
            if p and p[0].isupper():
                surnames.append(p.split()[-1]) # take last word as surname
        
        author_surnames_str = " ".join(surnames)
    
    if len(title_extracted) > 5:
        # Query 1: Title + Author Surnames (Most precise)
        if author_surnames_str:
            search_queries.append(f"{title_extracted} {normalize_name(author_surnames_str)}")
        else:
            # Query 2: Just Title
            search_queries.append(title_extracted)
            
    
    # Fallback: legacy splitting if we had no good title
    else:
        parts = [p.strip() for p in text.split('.') if len(p.strip()) > 3]
        candidates = sorted(parts, key=len, reverse=True)
        search_queries.extend(candidates[:2])

    # --- 2. Crossref Check ---
    for q in search_queries:
        if len(q) < 5: continue

        queries_tried.append(f"Crossref: {q}")
        hits = search_crossref(q)
        if not hits:
            continue

        for hit in hits:
            update_best(hit)
            match, names_swapped = is_match(ref, hit)
            if match:
                return True, hit, f"Crossref: {q}", names_swapped
    
    # Also add a query for the whole text if it's not too long
    # DBLP sometimes works well with full ref
    # search_queries.append(text[:200])

    # --- 3. DBLP Check ---
    for q in search_queries:
        if len(q) < 5:
            continue
        
        queries_tried.append(f"DBLP: {q}")
        hits = search_dblp(q)
        if not hits:
            continue
            
        for hit in hits:
            hit_title = hit.get('info', {}).get('title', '')
            
            update_best(hit)

            # Use improved matching logic
            match, names_swapped = is_match(ref, hit)
            if match:
                type_ = hit.get('info', {}).get('type', '')
                if type_ in ['Editorship', 'Data'] or is_venue_title(hit_title):
                    continue
                
                return True, hit, f"DBLP: {q}", names_swapped

    # --- 3. ArXiv Title Search Fallback ---
    # If DBLP failed, but it claims to be ArXiv (or we just want to be thorough)
    if is_arxiv_ref(text):
         # Search ArXiv by title
         if len(title_extracted) > 10:
             q = title_extracted
         else:
             # Fallback to longest part
             q = search_queries[0] if search_queries else text[:50]

         queries_tried.append(f"ArXiv Title: {q}")
         hit = search_arxiv(q, query_type='ti')
         if hit:
             update_best(hit)
             match, names_swapped = is_match(ref, hit)
             if match:
                 return True, hit, f"ArXiv Title: {q}", names_swapped
    
    return False, best_candidate, "; ".join(queries_tried), False

def run_check_on_file(url, submission_id=None, title=None, use_local=False):
    """
    Runs the reference check on a single PDF URL.
    Returns the output log as a string.
    """
    output_lines = []
    def log_print(s=""):
        output_lines.append(str(s))
        print(s)  # Also print to stdout? Yes

    log_print(f"Checking Submission {submission_id}: {title}")
    log_print(f"URL: {url}")
    log_print("=" * 60)

    if use_local:
        pdf_path = url
    else:
        pdf_path = download_pdf(url)
    try:
        full_text = extract_text(pdf_path)
        ref_section = find_references_section(full_text)
        
        if not ref_section:
            log_print("ERROR: Could not locate References section. Formatting might be non-standard.")
            return "\n".join(output_lines)
            
        references = extract_references_list(ref_section)
        log_print(f"Extracted {len(references)} references.")
        
        if not references:
            log_print("WARNING: Reference section found but no references extracted.")
            return "\n".join(output_lines)

        fake_refs = []
        swapped_name_refs = []
        
        log_print("Verifying references against DBLP...")
        log_print(f"{'ID':<5} {'Status':<10} {'Details'}")
        log_print("-" * 60)
        
        for ref in references:
             valid, hit, query_details, names_swapped = check_reference(ref)
             
             status = "OK" if valid else "not found"
             if not valid:
                 ref['failed_queries'] = query_details
                 ref['closest_match'] = hit
                 fake_refs.append(ref)
                 log_print(f"[{ref['id']}] {status:<10} Queries tried: {query_details}")
             else:
                 log_print(f"[{ref['id']}] {status:<10} Detected: {hit['info'].get('title', '')[:30]}... (Query: {query_details})")
                 if names_swapped:
                    swapped_name_refs.append((ref, hit))

        log_print("-" * 60)
        if fake_refs:
            log_print(f"\nFound {len(fake_refs)} references that could not be matched to DBLP:")
            for ref in fake_refs:
                log_print(f"\n[{ref['id']}] {ref['text']}")
                log_print(f"Queries tried: {ref.get('failed_queries', 'N/A')}")
                
                closest = ref.get('closest_match')
                if closest:
                    info = closest.get('info', {})
                    c_title = info.get('title', 'Unknown')
                    c_url = info.get('url', '')
                    c_authors = info.get('authors', 'Unknown')
                    log_print(f"Closest match: {c_title} ({c_url})\n\t by {c_authors}")
        else:
            log_print("\nAll references verified successfully.")
        if swapped_name_refs:
            log_print(f"\nFound {len(swapped_name_refs)} references that had authors' names backwards:")
            for ref, hit in swapped_name_refs:
                log_print(f"\n[{ref['id']}] {ref['text']}")
                info = hit.get('info', {})
                c_title = info.get('title', 'Unknown')
                c_url = info.get('url', '')
                c_authors = info.get('authors', 'Unknown')
                log_print(f"Closest match: {c_title} ({c_url})\n\t by {c_authors}")

    finally:
        if os.path.exists(pdf_path) and not use_local:
            os.remove(pdf_path)
            
    return "\n".join(output_lines)

def process_batch_txt(txt_path, log_dir):
    try:
        with open(txt_path, 'r', encoding='utf-8') as f:
            urls = [line.strip() for line in f if line.strip()]
            
        logger.info(f"Found {len(urls)} URLs in {txt_path}")
        
        for i, url in enumerate(urls):
            submission_id = f"batch_{i+1:03d}"
            
            # Try to infer title from filename if possible, otherwise generic
            filename = url.split('/')[-1]
            title = filename if filename else f"Paper {i+1}"
            
            log_filename = f"{submission_id}.log"
            log_path = os.path.join(log_dir, log_filename)
            
            if os.path.exists(log_path):
                logger.info(f"Skipping {submission_id} (log exists)")
                continue

            local_pdf = url[:4] != "http"
            
            try:
                log_content = run_check_on_file(url, submission_id, title, use_local=local_pdf)
                
                with open(log_path, 'w', encoding='utf-8') as f_out:
                    f_out.write(log_content)
                    
            except Exception as e:
                logger.error(f"Failed to process {url}: {e}")
    except Exception as e:
        logger.error(f"Failed to process batch file {txt_path}: {e}")

def process_pdf_folder(folder_path, log_dir):
    try:
        pdf_files = glob.glob(os.path.join(folder_path, "*.pdf"))
        logger.info(f"Found {len(pdf_files)} PDF files in {folder_path}")
        
        for i, pdf_file in enumerate(sorted(pdf_files)):
            filename = os.path.basename(pdf_file)
            submission_id = os.path.splitext(filename)[0]
            title = filename
            
            log_filename = f"{submission_id}.log"
            log_path = os.path.join(log_dir, log_filename)
            
            if os.path.exists(log_path):
                logger.info(f"Skipping {submission_id} (log exists)")
                continue

            try:
                # use_local=True since we are processing a local PDF file
                log_content = run_check_on_file(pdf_file, submission_id, title, use_local=True)
                
                with open(log_path, 'w', encoding='utf-8') as f_out:
                    f_out.write(log_content)
                    
            except Exception as e:
                logger.error(f"Failed to process {pdf_file}: {e}")
            print("-------------------------------------------\n\n")
    except Exception as e:
        logger.error(f"Failed to process PDF folder {folder_path}: {e}")

def main():
    parser = argparse.ArgumentParser(description="Detect potential fake references in PDF submissions.")
    parser.add_argument("source", help="URL to the PDF file OR path to a .txt file containing URLs OR path to a folder of PDF files")
    parser.add_argument('--mailto', help='Email for Crossref polite pool (optional, but recommended for faster queries)')
    args = parser.parse_args()
    CROSSREF_MAILTO = args.mailto

    if os.path.isdir(args.source):
        # Folder mode (PDFs)
        log_dir = os.path.join(os.getcwd(), "reference_checks")
        os.makedirs(log_dir, exist_ok=True)
        
        process_pdf_folder(args.source, log_dir)

    elif args.source.endswith(".txt") and os.path.isfile(args.source):
        # Batch mode
        log_dir = os.path.join(os.getcwd(), "reference_checks")
        os.makedirs(log_dir, exist_ok=True)
        
        process_batch_txt(args.source, log_dir)
            
    else:
        local_pdf = args.source[:4] != "http"
        log_content = run_check_on_file(args.source, "SingleCheck", "Manual Run", use_local=local_pdf)
        if local_pdf:
            filename = os.path.basename(args.source)
            submission_id = os.path.splitext(filename)[0]

            log_dir = os.path.join(os.getcwd(), "reference_checks")
            os.makedirs(log_dir, exist_ok=True)
            
            log_filename = f"{submission_id}.log"
            log_path = os.path.join(log_dir, log_filename)
            with open(log_path, 'w', encoding='utf-8') as f_out:
                f_out.write(log_content)

if __name__ == "__main__":
    main()
