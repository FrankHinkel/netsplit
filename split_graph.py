def read_graph_with_node_info(filename):
    import csv
    G = nx.Graph()
    with open(filename, newline='') as f:
        reader = csv.reader(f)
        header = next(reader)
        for row in reader:
            if len(row) < 2:
                continue
            source = row[0]
            target = row[1]
            G.add_node(source)
            G.add_node(target)
            # Infos für source (row[2]) sammeln
            if len(row) >= 3:
                if 'info' not in G.nodes[source]:
                    G.nodes[source]['info'] = []
                if row[2] and row[2] not in G.nodes[source]['info']:
                    G.nodes[source]['info'].append(row[2])
            # Infos für target (row[3]) sammeln
            if len(row) >= 4:
                if 'info' not in G.nodes[target]:
                    G.nodes[target]['info'] = []
                if row[3] and row[3] not in G.nodes[target]['info']:
                    G.nodes[target]['info'].append(row[3])
            G.add_edge(source, target)
    
    return G
import networkx as nx
from networkx.algorithms import community as nx_comm

def detect_communities(G, method="greedy"):
    """Ermittelt Communities (Cluster) im Graphen.

    method:
        greedy  - Clauset-Newman-Moore (greedy modularity)
        label   - Asynchrone Label Propagation

    Rückgabe: Liste von Mengen (Knoten pro Community)
    Zusätzlich wird ein Knotenattribut 'community' (int) gesetzt.
    """
    if G.number_of_nodes() == 0:
        return []
    if method == "label":
        comms = list(nx_comm.asyn_lpa_communities(G))
    else:  # default greedy
        comms = list(nx_comm.greedy_modularity_communities(G))
    # Attribut setzen
    for cid, nodes in enumerate(comms):
        for n in nodes:
            G.nodes[n]['community'] = cid
    return comms

def collapse_graph_by_communities(G, communities):
    """Fasst Communities zu Superknoten zusammen.

    Erzeugt einen neuen Graphen H:
        - jeder Community -> ein Knoten (id = community index)
        - Knoten-Attribut 'original_nodes': Liste der ursprünglichen Knoten
        - Knoten-Attribut 'size': Anzahl ursprünglicher Knoten
        - Kanten zwischen Communities erhalten Gewichte (weight = Anzahl Originalkanten, die zwischen den beiden Communities verlaufen)
    """
    H = nx.Graph()
    # Mapping Originalknoten -> Community ID
    node2comm = {}
    for cid, nodes in enumerate(communities):
        for n in nodes:
            node2comm[n] = cid
    # Superknoten anlegen
    for cid, nodes in enumerate(communities):
        H.add_node(cid, original_nodes=list(nodes), size=len(nodes))
    # Kanten aggregieren
    edge_weights = {}
    for u, v in G.edges():
        cu = node2comm.get(u)
        cv = node2comm.get(v)
        if cu is None or cv is None:
            continue
        if cu == cv:
            continue  # interne Kanten ignorieren
        if cu > cv:
            cu, cv = cv, cu  # sort für konsistente key
        edge_weights[(cu, cv)] = edge_weights.get((cu, cv), 0) + 1
    for (cu, cv), w in edge_weights.items():
        H.add_edge(cu, cv, weight=w)
    return H

# Beispielgraph erstellen
def create_example_graph():
    G = nx.Graph()
    G.add_edges_from([
        ('T001', 'T002'), ('T002', 'T003'), 
        ('T004', 'T005'), ('T006', 'T007'), ('T007', 'T008'), ('T008', 'T006')
    ])
    return G

# Zufälligen Graphen mit n Knoten erzeugen
def create_random_graph(n, p=0.3):
    """
    Erzeugt einen zufälligen ungerichteten Graphen mit n Knoten.
    p: Wahrscheinlichkeit für eine Kante zwischen zwei Knoten (default: 0.3)
    """
    return nx.erdos_renyi_graph(n, p)

    # Gleichmäßige Komponenten erzeugen
def create_balanced_components_graph(n, k, p=0.05):
        """
        Erzeugt einen Graphen mit k gleich großen, zufällig verbundenen Komponenten.
        Jede Komponente hat ca. n//k Knoten.
        """
        G = nx.Graph()
        nodes = list(range(n))
        groups = [nodes[i::k] for i in range(k)]
        for group in groups:
            H = nx.erdos_renyi_graph(len(group), p)
            mapping = dict(zip(H.nodes, group))
            H = nx.relabel_nodes(H, mapping)
            G = nx.compose(G, H)
        return G
def create_stochastic_block_graph(sizes, p_in=0.5, p_out=0.05):
    """
    Erzeugt einen Graphen mit mehreren Blöcken (Cluster) unterschiedlicher Größe.
    - sizes: Liste mit Knotenzahlen pro Block, z.B. [20, 30, 15]
    - p_in: Wahrscheinlichkeit für Kanten innerhalb eines Blocks
    - p_out: Wahrscheinlichkeit für Kanten zwischen Blöcken
    """
    n_blocks = len(sizes)
    p_matrix = [[p_in if i == j else p_out for j in range(n_blocks)] for i in range(n_blocks)]
    return nx.stochastic_block_model(sizes, p_matrix)
def split_graph_into_components(G):
    components = list(nx.connected_components(G))
    subgraphs = [G.subgraph(c).copy() for c in components]
    return subgraphs

def main():

    # Beispiel: Graph aus CSV mit Node-Infos einlesen
    G = read_graph_with_node_info('input_graph.csv')
    subgraphs = split_graph_into_components(G)


    for i, sg in enumerate(subgraphs):
        # Clustering-Koeffizienten berechnen
        clustering = nx.clustering(sg)
        avg_clustering = nx.average_clustering(sg)
        # Werte als Attribut bei jedem Knoten speichern
        for node in sg.nodes:
            sg.nodes[node]['clustering'] = clustering.get(node, 0.0)
            sg.nodes[node]['avg_clustering'] = avg_clustering
        central = find_central_node(sg)
        print(f"Subgraph {i+1}: Nodes = {list(sg.nodes())}")
        print(f"  Zentralster Knoten: {central}")
        print(f"  Durchschnittlicher Clustering-Koeffizient: {avg_clustering:.3f}")
        # Zusatzinfos für alle Knoten ausgeben
        for node in sg.nodes:
            info = sg.nodes[node]
            if info:
                print(f"    {node}: {info}")

    # OPTIONAL: Communities im größten Subgraphen erkennen und kollabieren
    # largest = max(subgraphs, key=lambda g: g.number_of_nodes()) if subgraphs else None
    # if largest:
    #     comms = detect_communities(largest, method="greedy")
    #     print(f"Größter Subgraph: {largest.number_of_nodes()} Knoten, Communities: {len(comms)}")
    #     H = collapse_graph_by_communities(largest, comms)
    #     print(f"Collapsed Graph: {H.number_of_nodes()} Superknoten, {H.number_of_edges()} Kanten")
    #     for n, data in H.nodes(data=True):
    #         print(f"  Superknoten {n}: size={data['size']}, original_nodes={data['original_nodes'][:10]}{'...' if data['size']>10 else ''}")




    # Die 100 größten Subgraphen in eine gemeinsame CSV exportieren
    top_100 = sorted(subgraphs, key=lambda sg: sg.number_of_nodes(), reverse=True)[:100]
    total_edges = sum(g.number_of_edges() for g in top_100)
    total_nodes = sum(g.number_of_nodes() for g in top_100)
    print(f"Exportiere {total_edges} Kanten aus {len(top_100)} Subgraphen mit insgesamt {total_nodes} Knoten in subgraphs_top100.csv")
    if total_edges == 0:
        print("Warnung: Es werden keine Kanten exportiert! Prüfe die Eingabedaten.")
    export_all_graphs_to_csv(top_100, "subgraphs_top100.csv")
    # Kollabierte Supergraphen (Communities) exportieren
    export_collapsed_subgraphs(top_100, "collapsed_nodes.csv", "collapsed_edges.csv", community_method="greedy")
    print("Collapsed Supergraphs in collapsed_nodes.csv und collapsed_edges.csv exportiert.")
    #plot_multiple_graphs_grid(top_100, n_cols=10)
    
# Exportiert mehrere Graphen in eine gemeinsame CSV (Kantenliste)
def export_all_graphs_to_csv(graphs, filename):
    import csv

    # Für vereinfachte Variante: nur 'source-info' und 'target-info' als eine Spalte je Knoten

    with open(filename, mode='w', newline='') as f:
        writer = csv.writer(f)
        header = [
            "source", "target", "order", "degree_source", "degree_target", "dist_to_center",
            "source_info", "target_info",
            "source_clustering", "target_clustering",
            "source_avg_clustering", "target_avg_clustering"
        ]
        writer.writerow(header)
        for idx, G in enumerate(graphs, 1):
            degrees = dict(G.degree())
            # Zentralknoten bestimmen
            try:
                center = nx.center(G)
                center_node = center[0] if center else None
            except nx.NetworkXError:
                center_node = None
            # Kürzeste Wege zum Zentrum berechnen
            if center_node is not None:
                lengths = nx.shortest_path_length(G, source=center_node)
            else:
                lengths = {}
            for u, v in G.edges():
                def format_node(n):
                    return str(n)
                deg_u = degrees.get(u, 0)
                deg_v = degrees.get(v, 0)
                dist_u = lengths.get(u, '')
                dist_v = lengths.get(v, '')
                # Knoten mit geringerer Entfernung zum Zentrum als source
                if dist_u != '' and dist_v != '':
                    if dist_u <= dist_v:
                        source, target = u, v
                        deg_source, deg_target = deg_u, deg_v
                        dist_to_center = dist_u
                    else:
                        source, target = v, u
                        deg_source, deg_target = deg_v, deg_u
                        dist_to_center = dist_v
                else:
                    # Fallback: wie bisher nach Grad
                    if deg_u >= deg_v:
                        source, target = u, v
                        deg_source, deg_target = deg_u, deg_v
                    else:
                        source, target = v, u
                        deg_source, deg_target = deg_v, deg_u
                    dist_to_center = dist_u if dist_u != '' else dist_v
                # Zusatzinfos für source und target sammeln
                source_info = G.nodes[source] if source in G.nodes else {}
                target_info = G.nodes[target] if target in G.nodes else {}
                row = [
                    format_node(source),
                    format_node(target),
                    idx,
                    deg_source,
                    deg_target,
                    dist_to_center,
                    '|'.join(source_info.get('info', [])),
                    '|'.join(target_info.get('info', [])),
                    f"{source_info.get('clustering', 0.0):.5f}",
                    f"{target_info.get('clustering', 0.0):.5f}",
                    f"{source_info.get('avg_clustering', 0.0):.5f}",
                    f"{target_info.get('avg_clustering', 0.0):.5f}"
                ]
                writer.writerow(row)

def export_collapsed_subgraphs(graphs, filename_nodes, filename_edges, community_method="greedy"):
    """Exportiert kollabierte Versionen der Subgraphen.

    Erstellt zwei CSV Dateien:
      - filename_nodes: supergraph_id, community_id, size, original_nodes ("|"-joined)
      - filename_edges: supergraph_id, source_community, target_community, weight
    """
    import csv
    total_supernodes = 0
    total_superedges = 0
    with open(filename_nodes, 'w', newline='') as fn, open(filename_edges, 'w', newline='') as fe:
        wn = csv.writer(fn)
        we = csv.writer(fe)
        wn.writerow(["supergraph_id", "community_id", "size", "original_nodes"])
        we.writerow(["supergraph_id", "source_community", "target_community", "weight"])
        for gid, G in enumerate(graphs, 1):
            if G.number_of_nodes() == 0:
                continue
            comms = detect_communities(G, method=community_method)
            H = collapse_graph_by_communities(G, comms)
            # Nodes schreiben
            for cid, data in H.nodes(data=True):
                orig_nodes = data.get('original_nodes', [])
                wn.writerow([
                    gid,
                    cid,
                    data.get('size', len(orig_nodes)),
                    '|'.join(map(str, orig_nodes))
                ])
            total_supernodes += H.number_of_nodes()
            # Edges schreiben
            for u, v, edata in H.edges(data=True):
                we.writerow([
                    gid,
                    u,
                    v,
                    edata.get('weight', 1)
                ])
            total_superedges += H.number_of_edges()
    print(f"Collapsed Export: {total_supernodes} Superknoten, {total_superedges} Superkanten aus {len(graphs)} Subgraphen")

# Exportiert einen Graphen als CSV (Kantenliste)
def export_graph_to_csv(G, filename, order=None):
    import csv
    with open(filename, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["source", "target", "order"])
        for u, v in G.edges():
            writer.writerow([u, v, order])

# Mehrere Subgraphen gemeinsam plotten

# Mehrere Subgraphen in einem Grid plotten
def plot_multiple_graphs_grid(graphs, n_cols=8):
    import matplotlib.pyplot as plt
    import numpy as np
    n = len(graphs)
    n_rows = int(np.ceil(n / n_cols))
    plt.figure(figsize=(n_cols * 3, n_rows * 3))
    pos_all = {}
    for idx, g in enumerate(graphs):
        row = idx // n_cols
        col = idx % n_cols
        # Layout für Subgraph
        pos = nx.spring_layout(g, seed=42)
        # Offset für Grid-Position
        offset = np.array([col * 5.0, -row * 5.0])
        for node in g.nodes:
            pos_all[node] = pos[node] + offset
    colors = plt.cm.tab20.colors
    for idx, g in enumerate(graphs):
        color = colors[idx % len(colors)]
        # Zentralknoten bestimmen
        try:
            center = nx.center(g)
        except nx.NetworkXError:
            center = []
        # Alle anderen Knoten
        other_nodes = [n for n in g.nodes if n not in center]
        # Zentralknoten als Vollpunkte
        if center:
            nx.draw_networkx_nodes(g, pos_all, nodelist=center, node_color='black', node_size=180, label=None)
        # Andere Knoten als Kreise (nur Rand)
        if other_nodes:
            nx.draw_networkx_nodes(g, pos_all, nodelist=other_nodes, node_color=[color], edgecolors='black', node_size=100, linewidths=1.5, label=f"Subgraph {idx+1} (|V|={g.number_of_nodes()})", node_shape='o')
        nx.draw_networkx_edges(g, pos_all, edgelist=g.edges, edge_color=[color])
    nx.draw_networkx_labels(nx.Graph(), pos_all, font_size=7)
    plt.legend(fontsize=8, loc='upper left', bbox_to_anchor=(0.85, 1))
    plt.title(f"Die {n} größten Subgraphen (Grid)")
    plt.axis('off')
    plt.tight_layout()
    plt.show()


# Zentralster Knoten eines Graphen
def find_central_node(G):
    # Knoten mit minimaler Exzentrizität (Graphzentrum)
    try:
        center = nx.center(G)
        return center[0] if center else None
    except nx.NetworkXError:
        return None

# Plot-Funktion
def plot_graph(G, title=None):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(8, 6))
    pos = nx.spring_layout(G, seed=42)
    nx.draw(G, pos, with_labels=True, node_color='skyblue', edge_color='gray', node_size=100)
    if title:
        plt.title(title)
    plt.show()

if __name__ == "__main__":
    main()
