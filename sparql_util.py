import os

import requests

# The previous value pointed at a private host used for the experiments.  A
# local Freebase/Virtuoso service is the portable default; override it for a
# container or remote deployment without changing source code.
endpoint = os.environ.get("MEMQ_SPARQL_ENDPOINT", "http://localhost:3001/sparql")

def get_result(sql, ans):
    if "?" in ans:
        ans = ans.replace("?","")
    response = requests.get(endpoint, params={"query": sql}, headers={"Accept": "application/json"})
    results = response.json()
    answers = []
    for result in results["results"]["bindings"]:
        answers.append(result[ans]['value'].replace("http://rdf.freebase.com/ns/", "ns:"))
    return answers


NAME_QUERY = """
PREFIX ns: <http://rdf.freebase.com/ns/>
SELECT DISTINCT ?x
WHERE {
FILTER (!isLiteral(?x) OR lang(?x) = '' OR langMatches(lang(?x), 'en')) .
{entity} ns:type.object.name ?x
} 
"""
def get_friendly_name(mid):
    if mid[:3] != "ns:":
        mid = "ns:" + mid
    query = NAME_QUERY.replace('{entity}', mid)
    try:
        name = get_result(query, "x")[0]
        return name
    except:
        raise Exception(f"unable to get name for {mid}")
