import json
from sparql_util import get_result
from tqdm import tqdm
import networkx as nx


def get_unused_name(name_list, name):
    if name not in name_list:
        return name
    else:
        i = 0
        while name + str(i) in name_list:
            i += 1
        return name + str(i)


def process_dataset(dataset_name):
    input_file = f"output/{dataset_name}_train_graph.json"
    output_file = f"output/{dataset_name}_train_cvt_list.json"

    with open(input_file, "r") as f:
        train_graph = json.load(f)

    not_evaluable = []
    for d in tqdm(train_graph, desc=dataset_name):
        all_rel = d['all_rel']
        nodelist = []
        cvt_list = {}
        for n in d['nodeorder']:
            nodelist.append(n)
            cvt_list[n] = None
        all_cause = [f"FILTER (!isLiteral({d['AnsE']}) OR lang({d['AnsE']}) = '' OR langMatches(lang({d['AnsE']}), 'en'))"]

        for node in d['nodeorder']:
            if "ns:" in node:
                cvt_list[node] = False
                continue
            if not all_rel[node]['is_try']:
                if not all_rel[node]['reversed']:
                    if " UNION " in all_rel[node]['relation']:
                        relations = all_rel[node]['relation'].split(" UNION ")
                        assert len(relations) == 2
                        if " + " in relations[0] and " + " in relations[1]:
                            [sube1, sube2] = relations[0].split(" + ")
                            tmpname1 = get_unused_name(nodelist, "?tmp")
                            nodelist.append(tmpname1)
                            [sube3, sube4] = relations[1].split(" + ")
                            tmpname2 = get_unused_name(nodelist, "?tmp")
                            nodelist.append(tmpname2)
                            all_cause.append(f"{{ {all_rel[node]['father']} {sube1} {tmpname1} .\n {tmpname1} {sube2} {node} }}\nUNION\n{{ {all_rel[node]['father']} {sube3} {tmpname2} .\n {tmpname2} {sube4} {node} }}")
                        else:
                            all_cause.append(f"{{ {all_rel[node]['father']} {relations[0]} {node} }}\nUNION\n{{ {all_rel[node]['father']} {relations[1]} {node} }}")
                    else:
                        all_cause.append(f"{all_rel[node]['father']} {all_rel[node]['relation']} {node}")
                else:
                    if " UNION " in all_rel[node]['relation']:
                        relations = all_rel[node]['relation'].split(" UNION ")
                        assert len(relations) == 2
                        if " + " in relations[0] and " + " in relations[1]:
                            [sube1, sube2] = relations[0].split(" + ")
                            tmpname1 = get_unused_name(nodelist, "?tmp")
                            nodelist.append(tmpname1)
                            [sube3, sube4] = relations[1].split(" + ")
                            tmpname2 = get_unused_name(nodelist, "?tmp")
                            nodelist.append(tmpname2)
                            all_cause.append(f"{{ {node} {sube1} {tmpname1} .\n {tmpname1} {sube2} {all_rel[node]['father']} }}\nUNION\n{{ {node} {sube3} {tmpname2} .\n {tmpname2} {sube4} {all_rel[node]['father']} }}")
                        else:
                            all_cause.append(f"{{ {node} {relations[0]} {all_rel[node]['father']} . }}\nUNION\n{{ {node} {relations[1]} {all_rel[node]['father']} . }}")
                    else:
                        all_cause.append(f"{node} {all_rel[node]['relation']} {all_rel[node]['father']}")

                for f in all_rel[node]['filter']:
                    all_cause.append(f"FILTER({f})")
            else:
                tmpname = get_unused_name(nodelist, "?tmp")
                assert len(all_rel[node]['filter']) == 1 and not all_rel[node]['reversed']
                nodelist.append(tmpname)
                all_cause.append(f"FILTER(NOT EXISTS {{ {all_rel[node]['father']} {all_rel[node]['relation']} {tmpname} }} || EXISTS {{{all_rel[node]['father']} {all_rel[node]['relation']} {node} . FILTER({all_rel[node]['filter'][0]}) }})")

        where = " .\n".join(all_cause)
        orderby_cause = ""

        if d.get('order') is not None:
            if d['order']['start'] == 0:
                orderby_cause = f"ORDER BY {d['order']['order']}({d['order']['var']})\nLIMIT {d['order']['len']}"
            else:
                orderby_cause = f"ORDER BY {d['order']['order']}({d['order']['var']})\nLIMIT {d['order']['len']}\nOFFSET {d['order']['start']}"

        graph_sparql = f"PREFIX ns: <http://rdf.freebase.com/ns/>\nSELECT DISTINCT {d['AnsE']}\nWHERE \n{{\n{where}\n}}\n{orderby_cause}"
        d['graph_sparql'] = graph_sparql
        d['orderby'] = orderby_cause
        d['all_cause'] = all_cause

        try:
            graph_res = get_result(graph_sparql, d['AnsE'])
        except Exception:
            graph_res = []

        if len(graph_res) == 0:
            new_cause = []
            for x in all_cause:
                if "FILTER" not in x:
                    new_cause.append(x)
            all_cause = new_cause
            where = " .\n".join(new_cause)
            orderby_cause = ""

            graph_sparql = f"PREFIX ns: <http://rdf.freebase.com/ns/>\nSELECT DISTINCT {d['AnsE']}\nWHERE \n{{\n{where}\n}}\n{orderby_cause}"
            try:
                graph_res = get_result(graph_sparql, d['AnsE'])
            except Exception:
                graph_res = []

        if len(graph_res) == 0:
            print(f"{d['id']} not evaluable")
            d['cvt_list'] = None
            not_evaluable.append(d['id'])
            continue

        G = nx.Graph(d['G'])

        for node in G.nodes():
            if "ns:" in node:
                cvt_list[node] = False
                continue

            if all_rel[node]['relation'] in ["ns:film.actor.film", "ns:government.government_office_or_title.office_holders", "ns:organization.organization.leadership", "ns:government.political_appointer.appointees", "ns:education.educational_institution.students_graduates", "ns:film.film.release_date_s", "ns:film.film_character.portrayed_in_films", "ns:government.governmental_body.members", "ns:government.politician.government_positions_held", "ns:government.governmental_jurisdiction.governing_officials"]:
                if not all_rel[node]['reversed']:
                    if G.degree(node) == 1:
                        print(f"{d['id']}: CVT node in leaf, please check")
                    cvt_list[node] = True
                    cvt_list[all_rel[node]['father']] = False
                else:
                    cvt_list[node] = False
                    cvt_list[all_rel[node]['father']] = True
                continue
            elif all_rel[node]['relation'] in ["ns:film.performance.film", "ns:organization.leadership.person", "ns:organization.leadership.role", "ns:government.government_position_held.office_holder", "ns:government.government_position_held.from", "ns:education.education.student", "ns:government.government_position_held.governmental_body", "ns:film.performance.character", "ns:government.government_position_held.basic_title", "ns:government.government_position_held.district_represented", "ns:location.mailing_address.citytown", "ns:government.government_position_held.office_position_or_title", "ns:government.government_position_held.appointed_by"]:
                if all_rel[node]['reversed']:
                    if G.degree(node) == 1:
                        print(f"{d['id']}: CVT node in leaf, please check")
                    cvt_list[node] = True
                    cvt_list[all_rel[node]['father']] = False
                else:
                    cvt_list[node] = False
                    cvt_list[all_rel[node]['father']] = True
                continue
            if G.degree(node) == 1:
                cvt_list[node] = False
                continue
            else:
                check_cause = []
                for w in all_cause:
                    check_cause.append(w)
                check_cause.append(f"FILTER(EXISTS{{ {node} ns:type.object.name ?nodename }})")
                check_where = " .\n".join(check_cause)
                check_sparql = f"PREFIX ns: <http://rdf.freebase.com/ns/>\nSELECT DISTINCT {d['AnsE']}\nWHERE \n{{\n{check_where}\n}}\n{orderby_cause}"
                try:
                    check_result = get_result(check_sparql, d['AnsE'])
                except Exception:
                    check_result = []
                if len(check_result) > 0:
                    cvt_list[node] = False
                elif len(check_result) == 0 and len(graph_res) > 0:
                    cvt_list[node] = True
                else:
                    print(f"{d['id']} not expected check result for {node}")
        d['cvt_list'] = cvt_list

    with open(output_file, "w") as f:
        json.dump(train_graph, f)

    print(f"{dataset_name}: {len(not_evaluable)} not evaluable: {not_evaluable}")
    return train_graph


if __name__ == "__main__":
    process_dataset("webqsp")
    process_dataset("cwq")