import networkx as nx

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
    # Beispiel: Gleichmäßige Komponenten
    G = create_stochastic_block_graph([200,200,200], p_in=0.002, p_out=0.0021)
    #G = create_balanced_components_graph(100, 4, p=0.5)
    # G = create_random_graph(10000, p=0.0003)
    # G = create_example_graph()
    subgraphs = split_graph_into_components(G)

    for i, sg in enumerate(subgraphs):
        central = find_central_node(sg)
        print(f"Subgraph {i+1}: Nodes = {list(sg.nodes())}")
        print(f"  Zentralster Knoten: {central}")




    # Die 100 größten Subgraphen in eine gemeinsame CSV exportieren
    top_100 = sorted(subgraphs, key=lambda sg: sg.number_of_nodes(), reverse=True)[:100]
    export_all_graphs_to_csv(top_100, "subgraphs_top100.csv")
    #plot_multiple_graphs_grid(top_100, n_cols=10)
    
# Exportiert mehrere Graphen in eine gemeinsame CSV (Kantenliste)
def export_all_graphs_to_csv(graphs, filename):
    import csv
    with open(filename, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["source", "target", "order", "degree_source", "degree_target"])
        for idx, G in enumerate(graphs, 1):
            degrees = dict(G.degree())
            for u, v in G.edges():
                def format_node(n):
                    n_str = str(n)
                    if n_str.startswith('k'):
                        n_str = n_str[1:]
                    return f"T{str(n_str).zfill(12)}"
                writer.writerow([
                    format_node(u),
                    format_node(v),
                    idx,
                    degrees.get(u, 0),
                    degrees.get(v, 0)
                ])

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
