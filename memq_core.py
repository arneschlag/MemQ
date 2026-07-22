"""MemQ reconstruction core — shared, **database-free**.

Factored out of the original all-in-one `reconstruct.py` so the lookup pass
(memory retrieval + SPARQL-string assembly + structure accuracy) can run in the
background with NO Freebase service. The only thing that touches the DB is the
*answer* execution, which lives in `score_answers.py`.

Retrieval is configurable via env vars so Exp 2/3 don't need code edits:
  MEMQ_EMBED_MODEL  sentence-transformers model/path
                     (default sentence-transformers/all-MiniLM-L6-v2)
  MEMQ_RETRIEVAL    legacy | adaptive            (default legacy = today's reranker)
  MEMQ_GAMMA1       adaptive: top-1 cutoff       (default 0.90)
  MEMQ_GAMMA2       adaptive: multi-recall cutoff(default 0.80)
  MEMQ_KEY_EXPLAIN  memory file                  (default output/key_explain.json)

Importing this module builds the embedding index once (needs torch). Keep it out
of `score_answers.py`, which must stay torch-free.
"""
import os
import re
import json
import time
import numpy as np
from collections import Counter
from scipy.spatial.distance import cdist
import networkx as nx
from sentence_transformers import SentenceTransformer

# ---------------------------------------------------------------- regex patterns
variablepattern = r"\?[A-Za-z0-9_]+"
midnamepattern = r"\*(.+)\*"
integer_pattern = r'[-+]?\d+'
filterfloatpattern = r"a float (\"[-+]?\d+\.\d+\")"
filterstrpattern = r"a string (\"[^\"]+\")"
filterdatetimepattern = r"(\d{4}-\d{2}-\d{2})"
sortdatetimepattern = r"datetime (\?[A-Za-z0-9_]+)"
sortintegerpattern = r"integer (\?[A-Za-z0-9_]+)"
sortfloatpattern = r"float (\?[A-Za-z0-9_]+)"
findpattern = r"Find (.+), assign it to (\?[A-Za-z0-9_]+)\."
makesurepattern = r"Make sure (.+)\."
sortpattern = r"Sort the result based on (.+) in (descending|ascending) order and keep the (.+) result\."
finallypattern = r"Finally the answer is (\?[A-Za-z0-9_]+)\."
countpattern = r"Count the number of (\?[A-Za-z0-9_]+)\."  # GrailQA aggregation
existspattern = r"Find (.+), assign it to (\?[A-Za-z0-9_]+)\. If (\?[A-Za-z0-9_]+) exists, ([^.]+)\."

SPARQL_TEMPLATE = """PREFIX ns: <http://rdf.freebase.com/ns/>\nSELECT DISTINCT {ansE}\nWHERE{{\n{where}\n}}\n{sort_sparql}"""

# ---------------------------------------------------------------- config
EMBED_MODEL = os.environ.get(
    "MEMQ_EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
)
RETRIEVAL = os.environ.get("MEMQ_RETRIEVAL", "adaptive")  # adaptive (paper, default) | legacy
GAMMA1 = float(os.environ.get("MEMQ_GAMMA1", "0.90"))   # adaptive: exact-match cutoff
GAMMA2 = float(os.environ.get("MEMQ_GAMMA2", "0.80"))   # adaptive: multi-recall cutoff
KEY_EXPLAIN = os.environ.get("MEMQ_KEY_EXPLAIN", "output/key_explain.json")
# legacy reranker constants (paper-divergent, kept for the baseline)
gamma = 0.6
alpha = 0.9

# ---------------------------------------------------------------- memory index
with open(KEY_EXPLAIN, "r") as f:
    _all_key = json.load(f)

explain_key = {}
for k in _all_key:
    explain = _all_key[k]
    if explain in explain_key:
        if len(k.split(".\n")) == 3:
            assert explain_key[explain]["is_tri"] is True
        else:
            assert explain_key[explain]["is_tri"] is False
        explain_key[explain]["infounit"].append(k)
    else:
        if len(k.split(" .\n")) == 3:
            explain_key[explain] = {"infounit": [k], "is_tri": True}
        else:
            explain_key[explain] = {"infounit": [k], "is_tri": False}

explain_list = list(explain_key.keys())
model = SentenceTransformer(EMBED_MODEL)
existing_embeddings = model.encode(explain_list, convert_to_tensor=False)

MID_NAMES = os.environ.get("MEMQ_MID_NAMES", "output/All_cached_mid_names.json")
with open(MID_NAMES, "r") as f:
    mid_names = json.load(f)


def get_mid_by_name(name):
    mids = [key for key, value in mid_names.items() if value == name]
    if len(mids) == 0:
        return [key for key, value in mid_names.items() if value.lower() == name.lower()]
    return mids


# ---------------------------------------------------------------- retrieval
def common_words_similarity(explain, ref_explain):
    words1 = explain.lower().split()
    words2 = ref_explain.lower().split()
    common_words = set(words1) & set(words2)
    return len(common_words) / len(set(words1))


def get_infounit(explain, is_tri=False):
    query_embedding = model.encode([explain], convert_to_tensor=False)
    distances = cdist(query_embedding, existing_embeddings, metric='cosine')[0]
    similarities = 1 - distances
    # restrict to the right pool (tri vs non-tri)
    if is_tri:
        for i, tmp_explain in enumerate(explain_list):
            if not explain_key[tmp_explain]['is_tri']:
                similarities[i] = 0
    else:
        for i, tmp_explain in enumerate(explain_list):
            if explain_key[tmp_explain]['is_tri']:
                similarities[i] = 0

    top_indices = np.argsort(similarities)[-8:][::-1]

    # tri relations: paper + legacy both take the single best
    if is_tri:
        return explain_key[explain_list[top_indices[0]]]["infounit"], similarities[top_indices[0]]

    if RETRIEVAL == "adaptive":
        # Paper Eq.4 adaptive recall — pure cosine, no word-overlap reranker.
        best = similarities[top_indices[0]]
        if best >= GAMMA1:
            return explain_key[explain_list[top_indices[0]]]["infounit"]
        out = []
        for i in top_indices:
            if similarities[i] >= GAMMA2:
                out.extend(explain_key[explain_list[i]]["infounit"])
        if not out:  # nothing cleared gamma2 -> fall back to top-1
            out = list(explain_key[explain_list[top_indices[0]]]["infounit"])
        return out

    # legacy: exact-match shortcut, else common-words rerank over top-8
    if similarities[top_indices[0]] > 0.99:
        return explain_key[explain_list[top_indices[0]]]["infounit"]
    for i in top_indices:
        similarities[i] += gamma * common_words_similarity(explain, explain_list[i])
    top_similarity_idx = np.argsort(similarities)[-8:][::-1][0]
    top_similarity = similarities[top_similarity_idx]
    out = []
    for i in top_indices:
        if similarities[i] > alpha * top_similarity:
            out.extend(explain_key[explain_list[i]]["infounit"])
    return out


# ---------------------------------------------------------------- FILTER parsing
def split_by_operators(s):
    pattern = r' ((?:>=|<=|!=|>|<|=)) '
    parts = re.split(pattern, s)
    return [part for part in parts if part]


def process_filter(f, idx=None):
    # v9 model sometimes wraps the condition in FILTER(...) syntax
    f = f.strip()
    if f.startswith("FILTER(") and f.endswith(")"):
        f = f[7:-1].strip()  # strip FILTER( ... )
    f = f.strip()
    f = f.replace("should not be smaller than", ">=").replace("should not be earlier than", ">=")
    f = f.replace("should not be larger than", "<=").replace("should not be later than", "<=")
    f = f.replace("should be smaller than", "<").replace("should be earlier than", "<")
    f = f.replace("should be larger than", ">").replace("should be later than", ">")
    f = f.replace("should be", "=").replace("should not be", "!=")
    f = split_by_operators(f)
    assert len(f) == 3, f"unable to parse filter {f}"
    e1, op, e2 = f[0], f[1], f[2]
    if re.fullmatch(variablepattern, e1) and re.fullmatch(variablepattern, e2):
        return f"FILTER({e1} {op} {e2})"
    elif re.fullmatch(variablepattern, e1) and e2 == "*NOW*":
        return f"FILTER(xsd:datetime({e1}) {op} \"2015-08-10\"^^xsd:dateTime)"
    elif re.fullmatch(variablepattern, e1) and re.fullmatch(midnamepattern, e2) and e2 != "*NOW*":
        e2 = re.fullmatch(midnamepattern, e2).group(1)
        e2mids = get_mid_by_name(e2)
        expr = [f"{e1} {op} {mid}" for mid in e2mids]
        if op == "=":
            expr = " OR ".join(expr)
        elif op == "!=":
            expr = " AND ".join(expr)
        else:
            raise Exception(f"DEBUG f: {f}")
        if len(e2mids) == 0:
            return ""
        return f"FILTER({expr})"
    elif re.fullmatch(variablepattern, e1) and re.fullmatch(filterfloatpattern, e2):
        e2 = re.fullmatch(filterfloatpattern, e2).group(1)
        return f"FILTER(xsd:float({e1}) {op} {e2}^^xsd:float)"
    elif re.fullmatch(variablepattern, e1) and re.fullmatch(filterstrpattern, e2):
        e2 = re.fullmatch(filterstrpattern, e2).group(1)
        return f"FILTER(str({e1}) {op} {e2})"
    elif re.fullmatch(variablepattern, e1) and re.fullmatch(filterdatetimepattern, e2):
        e2 = re.fullmatch(filterdatetimepattern, e2).group(1)
        return f"FILTER(xsd:datetime({e1}) {op} \"{e2}\"^^xsd:dateTime)"
    #  v9 model emits xsd:dateTime literals in Make sure steps: "YYYY"^^xsd:dateTime etc.
    elif re.fullmatch(variablepattern, e1) and re.fullmatch(r'"(\d{4}(?:-\d{2}(?:-\d{2})?)?)"\^\^xsd:dateTime', e2):
        e2val = re.fullmatch(r'"(\d{4}(?:-\d{2}(?:-\d{2})?)?)"\^\^xsd:dateTime', e2).group(1)
        # normalize partial dates to the start of the period
        if "-" not in e2val:        # YYYY -> YYYY-01-01
            e2val = f"{e2val}-01-01"
        elif e2val.count("-") == 1:  # YYYY-MM -> YYYY-MM-01
            e2val = f"{e2val}-01"
        # else YYYY-MM-DD -> as-is
        if op == "=":
            parts = e2val.split("-")
            y, m, d = int(parts[0]), int(parts[1]), int(parts[2])
            if m == 12:
                end = f"{y+1}-01-01"
            else:
                end = f"{y}-{m+1:02d}-01"
            return f"FILTER(xsd:datetime({e1}) >= \"{e2val}\"^^xsd:dateTime && xsd:datetime({e1}) < \"{end}\"^^xsd:dateTime)"
        else:
            return f"FILTER(xsd:datetime({e1}) {op} \"{e2val}\"^^xsd:dateTime)"
    elif re.fullmatch(variablepattern, e1) and re.fullmatch(integer_pattern, e2):
        return f"FILTER(xsd:integer({e1}) {op} \"{e2}\"^^xsd:integer)"
    # v9 model sometimes emits literal SPARQL expressions that split_by_operators
    # awkwardly destructures. When the parts look like SPARQL already (xsd: cast,
    # variable-var comparison, arithmetic), just reconstruct and wrap.
    elif ((re.fullmatch(r'xsd:\w+\(\s*\?\w+\s*\)', e1) and (re.fullmatch(variablepattern, e2) or re.fullmatch(r'[-+]?\d+', e2)))
          or (re.fullmatch(variablepattern, e1) and re.fullmatch(r'xsd:\w+\(\s*\?\w+\s*\)', e2))
          or any(op in str(f) for op in (" + ", " - "))):
        return f"FILTER({e1} {op} {e2})"
    else:
        raise Exception(f"{idx} ## Filter DEBUG f: {f}")


# ---------------------------------------------------------------- FIND step -> triples
def process_find(e_new, explain, G, cvt_node_cnt, seen_type2, main_entity, idx=None):
    assert re.fullmatch(variablepattern, e_new), e_new
    all_entities = re.findall(r"(\*.+\*)", explain)
    all_variables = re.findall(variablepattern, explain)
    related_nodes = list(all_entities) + list(all_variables)

    if len(related_nodes) == 1:
        tmp = explain.replace(related_nodes[0], "?entity1")
        tmp = "?entity2 is " + tmp
        if re.fullmatch(midnamepattern, related_nodes[0]):
            all_e1_mid = get_mid_by_name(related_nodes[0][1:-1])
            if len(all_e1_mid) != 1:
                if main_entity in all_e1_mid:
                    e1 = main_entity
                else:
                    raise Exception(f"{idx}: more than 1 matched e1 for {related_nodes[0][1:-1]}")
            else:
                e1 = all_e1_mid[0]
        elif re.fullmatch(variablepattern, related_nodes[0]):
            e1 = related_nodes[0]
        else:
            raise Exception(f"{idx}: not expected e1 {related_nodes[0]}")

        infounit = get_infounit(tmp, is_tri=False)
        return_infounit = []
        for iu in infounit:
            if len(iu.split(" .\n")) == 2:
                seen_type2[(e_new, e1)] = f"?cvt_{cvt_node_cnt}"
                tmp_iu_sparql = iu.replace("?entity2", e_new).replace("?entity1", e1).replace("?cvt", f"?cvt_{cvt_node_cnt}")
                return_infounit.append(tmp_iu_sparql)
                G.add_edge(e1, f"?cvt_{cvt_node_cnt}", relation="")
                G.add_edge(f"?cvt_{cvt_node_cnt}", e_new, relation=tmp_iu_sparql)
                cvt_node_cnt += 1
            else:
                tmp_iu_sparql = iu.replace("?entity2", e_new).replace("?entity1", e1)
                return_infounit.append(tmp_iu_sparql)
                G.add_edge(e1, e_new, relation=tmp_iu_sparql)
        return G, cvt_node_cnt, seen_type2, return_infounit

    elif len(related_nodes) == 2:
        tmp1 = explain.replace(related_nodes[0], "?entity1").replace(related_nodes[1], "?entity2")
        tmp1 = "?entity3 is " + tmp1
        infounit1, sim1 = get_infounit(tmp1, is_tri=True)
        tmp2 = explain.replace(related_nodes[0], "?entity2").replace(related_nodes[1], "?entity1")
        tmp2 = "?entity3 is " + tmp2
        infounit2, sim2 = get_infounit(tmp2, is_tri=True)
        infounit = infounit1 if sim1 > sim2 else infounit2
        if re.fullmatch(variablepattern, related_nodes[0]):
            e1 = related_nodes[0]
        elif re.fullmatch(midnamepattern, related_nodes[0]):
            all_e1_mid = get_mid_by_name(related_nodes[0][1:-1])
            if len(all_e1_mid) != 1:
                if main_entity in all_e1_mid:
                    e1 = main_entity
                else:
                    raise Exception(f"{idx}: more than 1 matched e1 for {related_nodes[0][1:-1]}")
            else:
                e1 = all_e1_mid[0]
        else:
            raise Exception(f"{idx} not implement type3 infounit")

        if re.fullmatch(variablepattern, related_nodes[1]):
            e2 = related_nodes[1]
        elif re.fullmatch(midnamepattern, related_nodes[1]):
            all_e1_mid = get_mid_by_name(related_nodes[1][1:-1])
            if len(all_e1_mid) != 1:
                raise Exception(f"{idx}: more than 1 matched e1 for {related_nodes[1][1:-1]}")
            else:
                e1 = all_e1_mid[0]
        else:
            raise Exception(f"{idx} not implement type3 infounit")

        if (e1, e2) in seen_type2:
            cvtnode = seen_type2[(e1, e2)]
        elif (e2, e1) in seen_type2:
            cvtnode = seen_type2[(e2, e1)]
        else:
            return G, cvt_node_cnt, seen_type2, []
        tmp_iu_sparql = infounit[0].split(" .\n")[-1].replace("?cvt", cvtnode).replace("?entity3", e_new)
        return_infounit = [tmp_iu_sparql]
        G.add_edge(cvtnode, e_new, relation=tmp_iu_sparql)
        return G, cvt_node_cnt, seen_type2, return_infounit
    else:
        raise Exception(f"{idx}: more than 2 nodes error")


# ---------------------------------------------------------------- direction fallback (historical v9 compatibility)
_GUARD_RELS = ("ns:type.object.name", "ns:type.object.type")


def _flip_one_triple(triple):
    """Return a UNION that tries both directions of one plain RDF triple."""
    parts = triple.strip().split()
    if len(parts) != 3:
        return None
    subject, relation, obj = parts
    if not relation.startswith("ns:") or relation in _GUARD_RELS:
        return None
    return f"{{ {subject} {relation} {obj} }}UNION{{ {obj} {relation} {subject} }}"


def _both_directions(step):
    """Recreate v9's recall fallback for direction-ambiguous relations.

    The original v9 memory used direction-agnostic Type-1 descriptions. This
    fallback is constructed for every query, but is executed only when the
    scorer receives ``MEMQ_DIRFB=1`` and all prior variants are empty.
    """
    if "FILTER" in step or "UNION" in step:
        return step
    return " .\n".join(_flip_one_triple(triple) or triple for triple in step.split(" .\n"))


# ---------------------------------------------------------------- plan -> SPARQL (DB-free)
def build_reconstruction(d):
    """Parse d['test_plan'] into reconstructed SPARQL strings. Pure string/graph
    work — no Freebase calls. Returns a dict; raises on a malformed plan."""
    main_entity = d["main_path"][0] if "main_path" in d else d["BegE"]
    cvt_node_cnt = 0
    plan = d['test_plan']
    plan = re.compile(r'(\?[A-Za-z0-9_]+)').sub(r' \1', plan)  # llama3: space before vars
    # Doppelte Blanks einebnen (NICHT \s, sonst verschwinden die Zeilenumbrueche,
    # an denen die Schritte getrennt werden). Zwei Quellen: 5.6% der DeepSeek-
    # Memory-Beschreibungen beginnen mit Whitespace ("Find " + " the gender of"),
    # und die Zeile darueber setzt ein Blank vor jede Variable, auch wenn dort
    # schon eines steht ("assign it to  ?x"). findpattern erwartet aber genau
    # eines, sonst schlaegt jede Rekonstruktion fehl.
    plan = re.sub(r'[ \t]{2,}', ' ', plan)
    steps = plan.split("\n")

    sort_sparql = ""
    ansE = ""
    is_count = False
    all_step_sparql = []
    seen_type2 = {}
    G = nx.DiGraph()
    for s in steps:
        step = re.sub(r'Step\d+:\s*', '', s)
        findmatch = re.fullmatch(findpattern, step)
        makesurematch = re.fullmatch(makesurepattern, step)
        sortmatch = re.fullmatch(sortpattern, step)
        finallymatch = re.fullmatch(finallypattern, step)
        existsmatch = re.fullmatch(existspattern, step)
        countmatch = re.fullmatch(countpattern, step)
        if findmatch:
            e_new = findmatch.group(2)
            explain = findmatch.group(1)
            G, cvt_node_cnt, seen_type2, return_infounit = process_find(e_new, explain, G, cvt_node_cnt, seen_type2, main_entity)
            if len(return_infounit) == 1:
                all_step_sparql.append(return_infounit[0])
            else:
                tmp = ["{ " + x + " }" for x in return_infounit]
                all_step_sparql.append("UNION".join(tmp))
        elif makesurematch:
            f = makesurematch.group(1)
            filt = process_filter(f)
            if filt != "":
                all_step_sparql.append(filt)
        elif sortmatch:
            order = "DESC" if sortmatch.group(2) == "descending" else "ASC"
            var = sortmatch.group(1)
            if re.fullmatch(variablepattern, var):
                sort_sparql = f"ORDER BY {order}({var})\n"
            elif re.fullmatch(sortdatetimepattern, var):
                var = re.fullmatch(sortdatetimepattern, var).group(1)
                sort_sparql = f"ORDER BY {order}(xsd:datetime({var}))\n"
            elif re.fullmatch(sortintegerpattern, var):
                var = re.fullmatch(sortintegerpattern, var).group(1)
                sort_sparql = f"ORDER BY {order}(xsd:float({var}))\n"
            elif re.fullmatch(sortfloatpattern, var):
                var = re.fullmatch(sortfloatpattern, var).group(1)
                sort_sparql = f"ORDER BY {order}(xsd:float({var}))\n"
            else:
                raise Exception(f"Sort DEBUG var: {var}")
            sortlen = sortmatch.group(3)
            if sortlen == "first":
                sort_sparql += "LIMIT 1"
            elif sortlen == "second":
                sort_sparql += "LIMIT 1\nOFFSET 1"
            else:
                raise Exception(f"Sort DEBUG sortlen: {sortlen}")
        elif countmatch:
            is_count = True
        elif finallymatch:
            ansE = finallymatch.group(1)
            all_step_sparql.append(f"FILTER (!isLiteral({ansE}) OR lang({ansE}) = '' OR langMatches(lang({ansE}), 'en'))")
        elif existsmatch:
            assert existsmatch.group(2) == existsmatch.group(3)
            e_new = existsmatch.group(2)
            explain = existsmatch.group(1)
            G, cvt_node_cnt, seen_type2, return_infounit = process_find(e_new, explain, G, cvt_node_cnt, seen_type2, main_entity)
            if len(return_infounit) > 0:
                exist_filter = process_filter(existsmatch.group(4))
                exist_search = return_infounit[0]
                all_step_sparql.append(f"FILTER(NOT EXISTS {{ {exist_search} }} || EXISTS {{ {exist_search} .{exist_filter} }})")
        else:
            raise Exception(f"not match {s}")

    where = " .\n".join(all_step_sparql)
    # GrailQA COUNT: project the aggregate as ?value; the WHERE still binds ansE.
    proj = f"(COUNT(DISTINCT {ansE}) AS ?value)" if is_count else ansE
    out_ansE = "?value" if is_count else ansE
    primary = SPARQL_TEMPLATE.format(ansE=proj, where=where, sort_sparql=sort_sparql)

    # fallback 1: drop all FILTERs (degradation in original reconstruct.py)
    steps1 = [x for x in all_step_sparql if "FILTER" not in x]
    steps1.append(f"FILTER({ansE} != {main_entity})")
    fb1 = SPARQL_TEMPLATE.format(ansE=proj, where=" .\n".join(steps1), sort_sparql="")

    # fallback 2: keep only the longest main path through the graph
    fb2 = None
    try:
        UG = nx.Graph(G)
        paths = list(nx.all_simple_paths(UG, source=main_entity, target=ansE))
        if paths:
            longest = max(paths, key=len)
            steps2 = []
            for i in range(len(longest) - 1):
                u, v = longest[i], longest[i + 1]
                ea = G.get_edge_data(u, v) or G.get_edge_data(v, u)
                if ea and ea['relation'] != "":
                    steps2.append(ea['relation'])
            fb2 = SPARQL_TEMPLATE.format(ansE=proj, where=" .\n".join(steps2), sort_sparql="")
    except Exception:
        fb2 = None

    # Historical v9 fallback 3: retry each relation in either direction.  Keep
    # it separate from the primary query so modern runs remain unchanged unless
    # score_answers.py is called with MEMQ_DIRFB=1.
    steps3 = [_both_directions(step) for step in all_step_sparql]
    fb3 = SPARQL_TEMPLATE.format(ansE=proj, where=" .\n".join(steps3), sort_sparql=sort_sparql)

    return {
        "reconstruct_sparql": primary,
        "reconstruct_sparql1": fb1,
        "reconstruct_sparql2": fb2,
        "reconstruct_sparql3": fb3,
        "ansE": out_ansE,
        "main_entity": main_entity,
    }


# ---------------------------------------------------------------- structure accuracy (DB-free)
# GrailQA's gold queries bind the Freebase namespace to ":" instead of "ns:"
# (PREFIX : <http://rdf.freebase.com/ns/>), so a pattern anchored on "ns:" finds
# zero gold relations there and structure accuracy is 0 by construction. Accept
# both prefixes and normalise to "ns:" so the multisets stay comparable.
_REL_RE = re.compile(r'(?<![A-Za-z0-9_])(?:ns)?:([A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)+)')


# A type constraint ("?x :type.object.type :opera.opera_designer_gig") puts a
# CLASS in object position that matches the relation pattern. GrailQA uses these
# heavily, WebQSP not at all, so counting them would compare classes against
# relations and depress GrailQA structure accuracy for the wrong reason.
_TYPE_TRIPLE_RE = re.compile(
    r'(?:ns)?:type\.object\.type\s+(?:ns)?:[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)+')


def _relation_multiset(sparql):
    """Bag of relation IRIs in a query, ignoring type/name guards and the class
    IRIs that appear as their object."""
    sparql = _TYPE_TRIPLE_RE.sub(" ", sparql or "")
    rels = ("ns:" + r for r in _REL_RE.findall(sparql))
    return sorted(r for r in rels if r not in ("ns:type.object.name", "ns:type.object.type"))


def structure_accuracy(reconstruct_sparql, ori_sparql):
    """1 if the reconstructed query uses exactly the same multiset of relation
    predicates as the gold query (hop pattern), else 0. DB-free proxy for the
    paper's structural-correctness signal — directly catches a single-hop
    relation being wrongly reconstructed as a two-hop CVT path."""
    return 1 if _relation_multiset(reconstruct_sparql) == _relation_multiset(ori_sparql) else 0


# ---------------------------------------------------------------- EHR / GoldGED (paper 5.3)
# Golden graph = d['where'] (list of [subj, pred, obj] triples, already present in
# every test-plan entry). Reconstructed graph = triples parsed back out of
# reconstruct_sparql. Grounded terms (mids) must match exactly; all variables are
# treated as an interchangeable "VAR" placeholder, since the pipeline invents
# fresh names for intermediate CVT nodes (?cvt_0, ...) that have no counterpart
# in the gold variable names (?y, ?c, ...). The paper doesn't specify the exact
# node-correspondence rule, so this is a documented approximation, not a verbatim
# reproduction of the (unpublished) original implementation.
def node_label(n):
    return n if not n.startswith("?") else "VAR"


def extract_where_block(sparql_text):
    start = sparql_text.index("WHERE")
    brace_start = sparql_text.index("{", start)
    depth = 0
    i = brace_start
    while i < len(sparql_text):
        if sparql_text[i] == "{":
            depth += 1
        elif sparql_text[i] == "}":
            depth -= 1
            if depth == 0:
                break
        i += 1
    return sparql_text[brace_start + 1:i]


def strip_filters(where_block):
    out = []
    i = 0
    while i < len(where_block):
        if where_block[i:i + 6] == "FILTER":
            j = where_block.index("(", i)
            depth = 1
            k = j + 1
            while k < len(where_block) and depth > 0:
                if where_block[k] == "(":
                    depth += 1
                elif where_block[k] == ")":
                    depth -= 1
                k += 1
            i = k
        else:
            out.append(where_block[i])
            i += 1
    return "".join(out)


_TERM = r'(?:\?[A-Za-z0-9_]+|ns:[A-Za-z_][A-Za-z0-9_.]*)'
_TRIPLE_RE = re.compile(rf'({_TERM})\s+(ns:[A-Za-z_][A-Za-z0-9_.]*)\s+({_TERM})')


def extract_triples(where_block):
    """Triples of a WHERE block, including those inside UNION branches.

    The reconstruction wraps a hop in "{ s p o }UNION{ o p s }" whenever the
    edge direction is unknown, and lists relation alternatives the same way.
    Splitting on " .\\n" and keeping only 3-token lines therefore dropped every
    such hop: the two branches end up on one line and yield six tokens. Those
    edges then counted as missing, which made EHR/GoldGED worst on the SIMPLEST
    questions -- they contain proportionally the most UNIONs -- and produced
    EHR 0.0 for queries that answer perfectly (F1 = 1.0). Matching the triple
    pattern directly is independent of braces, UNION and line breaks.
    """
    return [tuple(m) for m in _TRIPLE_RE.findall(strip_filters(where_block))]


def edge_multiset(triples):
    return Counter((node_label(s), p, node_label(o)) for s, p, o in triples)


def compute_ehr(gold_triples, pred_triples):
    gold_edges = edge_multiset(gold_triples)
    total = sum(gold_edges.values())
    if total == 0:
        return None
    pred_edges = edge_multiset(pred_triples)
    hit = sum((gold_edges & pred_edges).values())
    return hit / total


def build_triple_graph(triples):
    g = nx.DiGraph()
    for s, p, o in triples:
        g.add_node(s, label=node_label(s))
        g.add_node(o, label=node_label(o))
        g.add_edge(s, o, label=p)
    return g


GED_TIMEOUT_SEC = 2.0


def compute_gold_ged(gold_triples, pred_triples):
    g_gd = build_triple_graph(gold_triples)
    g_re = build_triple_graph(pred_triples)
    node_match = lambda a, b: a["label"] == b["label"]
    edge_match = lambda a, b: a["label"] == b["label"]
    start = time.time()
    best = None
    for cost in nx.optimize_graph_edit_distance(g_re, g_gd, node_match=node_match, edge_match=edge_match):
        best = cost
        if time.time() - start > GED_TIMEOUT_SEC:
            break
    return best


def reasoning_metrics(gold_where, reconstruct_sparql):
    """Convenience wrapper returning (ehr, gold_ged) for a lookup-pass item."""
    gold_triples = [tuple(t) for t in (gold_where or [])]
    pred_triples = extract_triples(extract_where_block(reconstruct_sparql))
    return compute_ehr(gold_triples, pred_triples), compute_gold_ged(gold_triples, pred_triples)


# ---------------------------------------------------------------- answer-set metrics
def eval_result(true_list, pred_list):
    true_set, pred_set = set(true_list), set(pred_list)
    inter = true_set & pred_set
    precision = len(inter) / len(pred_set) if pred_set else 0.0
    recall = len(inter) / len(true_set) if true_set else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    hit_at_1 = 1 if (len(pred_list) > 0 and pred_list[0] in true_set) else 0
    return precision, recall, f1, hit_at_1
