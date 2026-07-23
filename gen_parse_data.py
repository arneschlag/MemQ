"""
SPARQL parser for CWQ/WebQSP datasets.
Converts my_{dataset}_{split}.json into {dataset}_{split}_parse.json
for consumption by build_graph_train.py / build_graph_test.py.
"""

import json
import re
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable


# ── regex patterns ───────────────────────────────────────────────────────────

# Extract answer variable from SELECT DISTINCT ?x
RE_SELECT = re.compile(r"SELECT\s+DISTINCT\s+(\?\w+)", re.IGNORECASE)

# Matches a FILTER(NOT EXISTS { … } || EXISTS { … }) block
RE_EXISTS_BLOCK = re.compile(
    r"FILTER\s*\(\s*NOT\s+EXISTS\s*\{\s*"   # FILTER(NOT EXISTS {
    r"([^}]+)"                                #   triple(s) — must not contain }
    r"\s*\}\s*\|\|\s*EXISTS\s*\{\s*"          # } || EXISTS {
    r"([^}]+)"                                #   triple(s) inside EXISTS
    r"\s*\}\s*\)",                            # })
    re.IGNORECASE | re.DOTALL
)

# Matches a standalone FILTER(…) expression (with possible nested parens)
_RE_FILTER_INNER = r"\((?:[^()]|\([^()]*\))*\)"
RE_FILTER = re.compile(
    r"FILTER\s*(" + _RE_FILTER_INNER + r")",
    re.IGNORECASE | re.DOTALL
)

# Matches { body } UNION { body } (possibly repeated)
RE_UNION_BLOCK = re.compile(
    r"\{\s*([^}]*?)\s*\}\s*UNION\s*\{\s*([^}]*?)\s*\}"
    r"(?:\s*UNION\s*\{\s*([^}]*?)\s*\})*",
    re.DOTALL
)

# Object can be: ?var, ns:m.xxx, "literal", "literal"@en,
# "literal"^^xsd:type, 123, etc.
RE_OBJECT = (
    r"("
    r"[?\w][\w.:\"\-\^/]*(?::[?\w][\w.:\"\-\^/]*)*"  # ?var or ns:m.xxx
    r"|"
    r'"[^"]*"(?:@\w+|(?:\^\^)[\w.:\/\-#]+)?'         # "literal" with optional
    r")"                                                #   @lang or ^^type
)

# Triple pattern: subject predicate object
RE_TRIPLE = re.compile(
    r"([?\w][\w.\-]*(?::[?\w][\w.\-]*)*)"   # subject: ?var or ns:m.xxx
    r"\s+"
    r"(ns:[\w.]+)"                            # predicate: ns:rel.path
    r"\s+"
    + RE_OBJECT,
)

# Semicolon continuation: ; pred obj
RE_SEMICOLON = re.compile(
    r"\s*;\s*"
    r"(ns:[\w.]+)"                            # predicate
    r"\s+"
    + RE_OBJECT,
)

# Sub-query block: { SELECT … WHERE { … } }
RE_SUBQUERY = re.compile(
    r"\{\s*SELECT\s[^}]*WHERE\s*\{([^}]*)\}\s*\}",
    re.IGNORECASE | re.DOTALL
)


# ── helpers ──────────────────────────────────────────────────────────────────

def find_matching_brace(text, start):
    """Find position of matching closing brace starting from `start`
    (which points to the opening brace)."""
    depth = 1
    i = start + 1
    while i < len(text) and depth > 0:
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
        i += 1
    return i - 1 if depth == 0 else -1


def strip_comments(sparql):
    """Remove #-style comments from SPARQL text."""
    lines = []
    for line in sparql.split("\n"):
        # Keep everything before the first unquoted #
        # Be conservative: just strip lines that start with # or have # after whitespace
        stripped = line.strip()
        if stripped.startswith("#"):
            # Entire line is comment
            line = re.sub(r"#.*$", "", line, count=1)
        elif "#" in line:
            # Inline comment: remove from # onward (rough, but works for CWQ)
            line = re.sub(r"\s*#.*$", "", line)
        lines.append(line)
    return "\n".join(lines)


def normalize_whitespace(text):
    """Normalize whitespace but preserve structure."""
    return re.sub(r"[ \t]+", " ", text)


def parse_triples(text, where_list):
    """Parse triple patterns from text (including semicolon expansion and
    multi-triple lines).

    Handles:
      - Standard triples:  ?s ns:pred ?o .
      - Semicolons:        ?s ns:pred1 ?o1 ; ns:pred2 ?o2 .
      - Multi-triple line: ?s ns:p1 ?o1 . ?o1 ns:p2 ?o2 .
    """
    lines = text.split("\n")
    current_subj = None

    for line in lines:
        line = line.strip()
        if not line:
            continue

        # Skip FILTER remnants and other non-triple lines
        if line.upper().startswith("FILTER"):
            continue
        if line.upper().startswith("ORDER"):
            continue
        if line.upper().startswith("LIMIT"):
            continue
        if line.upper().startswith("OFFSET"):
            continue

        # Process all triples on this line
        pos = 0
        while pos < len(line):
            # Skip whitespace and dots (but NOT semicolons — those are
            # handled by RE_SEMICOLON)
            while pos < len(line) and line[pos] in ' \t.':
                pos += 1
            if pos >= len(line):
                break

            # 1) Semicolon continuation with leading ";"
            semi_match = RE_SEMICOLON.match(line, pos)
            if semi_match and current_subj:
                where_list.append([current_subj, semi_match.group(1),
                                   semi_match.group(2)])
                pos = semi_match.end()
                # Consume trailing dot
                while pos < len(line) and line[pos] in ' \t.':
                    pos += 1
                continue

            # 2) Full triple match (has subject and predicate)
            triple_match = RE_TRIPLE.match(line, pos)
            if triple_match:
                subj = triple_match.group(1)
                pred = triple_match.group(2)
                obj = triple_match.group(3)
                current_subj = subj
                where_list.append([subj, pred, obj])
                pos = triple_match.end()
                # Consume trailing . or ;
                while pos < len(line) and line[pos] in ' \t.;':
                    pos += 1
                continue

            # 3) Implicit semicolon continuation (line starts with
            #    "ns:pred obj" — no leading ";")
            if current_subj:
                rest = line[pos:].strip()
                # Remove trailing . or ; before splitting
                if rest.endswith(' .') or rest.endswith(' ;'):
                    rest = rest[:-2].strip()
                elif rest.endswith('.') or rest.endswith(';'):
                    rest = rest[:-1].strip()
                parts = rest.split(None, 1)
                if len(parts) == 2 and parts[0].startswith("ns:"):
                    where_list.append([current_subj, parts[0], parts[1]])
                    pos = len(line)  # consume rest of line
                    continue

            # Can't parse — go to next line
            break

            # Try implicit semicolon continuation:
            # "ns:pred obj" without leading semicolon
            if current_subj:
                rest = line[pos:].strip()
                parts = rest.split(None, 1)
                if len(parts) == 2 and parts[0].startswith("ns:"):
                    # Check that this isn't the start of a new triple
                    # (it would have a subject before the predicate)
                    where_list.append([current_subj, parts[0], parts[1]])
                    pos += len(line[pos:]) - len(rest) + len(parts[0]) + 1 + len(parts[1])
                    continue

            # Can't parse — skip remaining of line
            break


def extract_exists_blocks(text, exists_list):
    """Extract FILTER(NOT EXISTS{...}||EXISTS{...}) blocks.

    The output format is [placeholder_var, exists_triple_str, filter_expr]
    where:
      - placeholder_var: the variable from NOT EXISTS (e.g., ?sk0)
      - exists_triple_str: the triple from the EXISTS part (e.g.,
        "?y ns:rel ?sk1") — its variable should match the filter refs
      - filter_expr: the FILTER expression from the EXISTS part
    """

    def replace_exists(match):
        not_exists_part = match.group(1).strip()
        exists_part = match.group(2).strip()

        # Extract the triple from the EXISTS part (the part before the
        # first occurrence of " ." followed by FILTER or end-of-block).
        # Use RE_TRIPLE to find the triple — it stops at the first
        # non-matching char after the object.
        triple_match = RE_TRIPLE.search(exists_part)
        if triple_match:
            exists_triple_str = (
                f"{triple_match.group(1)} "
                f"{triple_match.group(2)} "
                f"{triple_match.group(3)}"
            )
        else:
            # Fallback: use first 3 whitespace-separated tokens that
            # look like subj pred obj
            parts = exists_part.split()
            # Find the predicate (starts with "ns:")
            for i, p in enumerate(parts):
                if p.startswith("ns:") and i > 0 and i < len(parts) - 1:
                    exists_triple_str = (
                        f"{parts[i - 1]} {parts[i]} {parts[i + 1]}"
                    )
                    break
            else:
                return ""

        # Find the placeholder variable from NOT EXISTS (the one that
        # does NOT appear in the EXISTS part)
        not_vars = set(re.findall(r"\?\w+", not_exists_part))
        exists_vars = set(re.findall(r"\?\w+", exists_part))
        placeholder_vars = not_vars - exists_vars

        if placeholder_vars:
            placeholder_var = list(placeholder_vars)[0]
        else:
            placeholder_var = "?sk0"

        # Extract FILTER expression from EXISTS part
        filter_expr = ""
        filter_match = re.search(
            r"FILTER\s*(\(.+\))", exists_part, re.IGNORECASE | re.DOTALL
        )
        if filter_match:
            filter_expr = filter_match.group(1).strip()

        exists_list.append([placeholder_var, exists_triple_str, filter_expr])
        return ""  # Remove from text

    return re.sub(RE_EXISTS_BLOCK, replace_exists, text)


def find_matching_paren(text, start):
    """Find position of matching closing paren starting from `start`
    (which points to the opening paren)."""
    depth = 1
    i = start + 1
    while i < len(text) and depth > 0:
        if text[i] == "(":
            depth += 1
        elif text[i] == ")":
            depth -= 1
        i += 1
    return i - 1 if depth == 0 else -1


def extract_filters(text, filter_list):
    """Extract standalone FILTER(…) expressions using paren-matching."""
    result = []
    i = 0
    while i < len(text):
        # Look for FILTER keyword
        pos = text.find("FILTER", i)
        if pos == -1:
            result.append(text[i:])
            break

        # Check if preceded by NOT or EXISTS context — skip if part of
        # FILTER(NOT EXISTS ...) which is handled by extract_exists_blocks
        prefix = text[max(0, pos - 20):pos].upper()
        if "NOT" in prefix or "EXISTS" in prefix:
            # This FILTER is part of EXISTS handling, skip it
            result.append(text[i:pos + 6])
            i = pos + 6
            continue

        # Add text before FILTER
        result.append(text[i:pos])

        # Find opening paren of FILTER
        paren_open = text.find("(", pos)
        if paren_open == -1:
            result.append(text[pos:])
            i = len(text)
            break

        # Find matching closing paren
        paren_close = find_matching_paren(text, paren_open)
        if paren_close == -1:
            result.append(text[pos:])
            i = len(text)
            break

        # Extract inner content (strip outer parens)
        inner = text[paren_open + 1:paren_close].strip()
        filter_list.append(inner)

        i = paren_close + 1

    return "".join(result)


def flatten_union_blocks(text, where_list):
    """Handle UNION blocks: extract triples from each branch."""
    # Find all { ... } UNION { ... } patterns
    # First find positions of UNION keywords
    while True:
        # Find a group that contains UNION
        union_start = -1
        union_pos = -1

        for m in re.finditer(r"UNION", text):
            pos = m.start()
            # Look backward for opening brace
            brace_start = text.rfind("{", 0, pos)
            if brace_start == -1:
                continue
            # Check if there's a closing brace before this UNION
            brace_end = find_matching_brace(text, brace_start)
            if brace_end == -1:
                continue
            # Now find the next opening brace after UNION
            next_brace = text.find("{", pos)
            if next_brace == -1:
                continue
            next_brace_end = find_matching_brace(text, next_brace)
            if next_brace_end == -1:
                continue

            # Extract bodies
            body1 = text[brace_start + 1:brace_end].strip()
            body2 = text[next_brace + 1:next_brace_end].strip()

            # Parse triples from both bodies
            parse_triples(body1, where_list)
            parse_triples(body2, where_list)

            # Replace the entire UNION expression with empty string
            # Find the full extent: from brace_start to next_brace_end
            union_start = brace_start
            union_pos = next_brace_end
            break

        if union_start == -1:
            break

        text = text[:union_start] + text[union_pos + 1:]

    return text


def extract_subqueries(text, where_list):
    """Extract triples from { SELECT ... WHERE { ... } } subqueries."""

    def replace_subquery(match):
        inner_where = match.group(1).strip()
        parse_triples(inner_where, where_list)
        return ""

    return re.sub(RE_SUBQUERY, replace_subquery, text)


def parse_sparql(sparql, beg_e, qid, question):
    """Parse a single SPARQL query into the _parse.json format."""
    result = {
        "id": qid,
        "question": question,
        "ori_sparql": sparql,
        "BegE": beg_e,
        "AnsE": None,
        "where": [],
        "filter": [],
        "exists": [],
    }

    # ── 1. Preprocess ──
    sparql_clean = strip_comments(sparql)
    sparql_clean = normalize_whitespace(sparql_clean)
    # Remove leading/trailing whitespace per line while keeping structure
    sparql_clean = "\n".join(line.strip() for line in sparql_clean.split("\n") if line.strip())

    # ── 2. Extract AnsE ──
    select_match = RE_SELECT.search(sparql_clean)
    if select_match:
        result["AnsE"] = select_match.group(1)

    # ── 3. Extract WHERE clause content ──
    where_start = sparql_clean.find("WHERE")
    if where_start == -1:
        return result

    # Find opening brace of WHERE
    brace_open = sparql_clean.find("{", where_start)
    if brace_open == -1:
        return result

    brace_close = find_matching_brace(sparql_clean, brace_open)
    if brace_close == -1:
        return result

    where_content = sparql_clean[brace_open + 1:brace_close].strip()

    # ── 4. Parse WHERE content ──
    # Order matters: EXISTS blocks first, then subqueries, then unions, then filters, then triples

    # 4a. Extract EXISTS blocks
    where_content = extract_exists_blocks(where_content, result["exists"])

    # 4b. Extract subqueries
    where_content = extract_subqueries(where_content, result["where"])

    # 4c. Handle UNION blocks
    where_content = flatten_union_blocks(where_content, result["where"])

    # 4d. Extract standalone FILTER expressions
    where_content = extract_filters(where_content, result["filter"])

    # 4e. Parse remaining triples
    parse_triples(where_content, result["where"])

    # 4f. Sort clause -> order spec (drives the Rank step downstream).
    # parse_triples deliberately skips ORDER BY / LIMIT lines, so without this a
    # superlative question loses its extremum: the query returns every candidate
    # instead of the top one. The sort variable's binding triple is kept above,
    # so only the direction/limit need recovering here.
    order = _extract_order(sparql, result["where"])
    if order is not None:
        result["order"] = order

    return result


_ORDER_RE = re.compile(
    r"ORDER\s+BY\s+(.*?)\s+LIMIT\s+(\d+)(?:\s+OFFSET\s+(\d+))?", re.I | re.S)


def _extract_order(sparql, where):
    """Recover {order, var, start, len} from a gold ORDER BY ... LIMIT clause."""
    m = _ORDER_RE.search(sparql or "")
    if not m:
        return None
    expr = m.group(1).strip()
    direction = "DESC" if re.match(r"DESC", expr, re.I) else "ASC"
    expr = re.sub(r"^(DESC|ASC)\s*", "", expr, flags=re.I).strip()
    variables = re.findall(r"\?[A-Za-z0-9_]+", expr)
    if not variables or not any(variables[0] in str(t) for t in (where or [])):
        return None
    # Keep the xsd cast so graph_explain can type the Rank step; for an uncast
    # key hand over the bare variable (graph_explain wants "?num", not "(?num)").
    has_cast = any(c in expr.lower() for c in ("datetime", "float", "integer"))
    return {"order": direction, "var": expr if has_cast else variables[0],
            "start": int(m.group(3)) if m.group(3) else 0, "len": int(m.group(2))}


# ── main processing ──────────────────────────────────────────────────────────

def process_dataset(input_path, output_path, desc=""):
    """Process a single dataset file."""
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    parsed = []
    for d in tqdm(data, desc=desc):
        try:
            entry = parse_sparql(
                sparql=d["ori_sparql"],
                beg_e=d["BegE"],
                qid=d["id"],
                question=d["question"],
            )
            parsed.append(entry)
        except Exception as e:
            print(f"Error parsing {d['id']}: {e}")
            # Still add basic entry so pipeline doesn't break
            parsed.append({
                "id": d["id"],
                "question": d["question"],
                "ori_sparql": d["ori_sparql"],
                "BegE": d["BegE"],
                "AnsE": None,
                "where": [],
                "filter": [],
                "exists": [],
            })

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(parsed, f, indent=2)

    # Print stats
    empty_where = sum(1 for p in parsed if len(p["where"]) == 0)
    has_exists = sum(1 for p in parsed if len(p["exists"]) > 0)
    has_filter = sum(1 for p in parsed if len(p["filter"]) > 0)
    print(f"  {desc}: {len(parsed)} entries — "
          f"empty_where={empty_where}, "
          f"has_exists={has_exists}, "
          f"has_filter={has_filter}")


if __name__ == "__main__":
    import os

    # Ensure output directory exists
    os.makedirs("output", exist_ok=True)

    # Train datasets
    process_dataset(
        "output/my_webqsp_train.json",
        "output/webqsp_train_parse.json",
        desc="webqsp_train",
    )

    process_dataset(
        "output/my_cwq_train.json",
        "output/cwq_train_parse.json",
        desc="cwq_train",
    )

    # Test datasets (if they exist)
    for ds_name in ["webqsp_test", "cwq_test"]:
        in_path = f"output/my_{ds_name}.json"
        out_path = f"output/{ds_name}_parse.json"
        if os.path.exists(in_path):
            process_dataset(in_path, out_path, desc=ds_name)
        else:
            print(f"  {ds_name}: skipping — {in_path} not found")

    print("\nDone. Generated _parse.json files in output/")
