# NetworkX Funktionsübersicht & Referenz

## Grundlegende Graph-Typen
- `nx.Graph()` – Ungerichteter Graph
- `nx.DiGraph()` – Gerichteter Graph
- `nx.MultiGraph()` – Ungerichteter Multigraph (mehrere Kanten zwischen Knoten)
- `nx.MultiDiGraph()` – Gerichteter Multigraph

## Graph-Erstellung
- `G = nx.Graph()`
- `G.add_node(node)` / `G.add_nodes_from([n1, n2, ...])`
- `G.add_edge(u, v)` / `G.add_edges_from([(u1, v1), (u2, v2), ...])`

## Graph-Generatoren
- `nx.erdos_renyi_graph(n, p)` – Zufallsgraph
- `nx.barabasi_albert_graph(n, m)` – Skalenfreier Graph
- `nx.watts_strogatz_graph(n, k, p)` – Small-World-Graph
- `nx.stochastic_block_model(sizes, p_matrix)` – Blockmodell

## Komponenten & Pfade
- `nx.connected_components(G)` – Zusammenhangskomponenten (ungerichtet)
- `nx.strongly_connected_components(G)` – Starke Komponenten (gerichtet)
- `nx.shortest_path(G, source, target)` – Kürzester Pfad
- `nx.has_path(G, source, target)` – Existenz eines Pfads

## Zentralitätsmaße
- `nx.degree_centrality(G)` – Degree-Zentralität
- `nx.betweenness_centrality(G)` – Vermittlungszentralität
- `nx.closeness_centrality(G)` – Nähezentralität
- `nx.eigenvector_centrality(G)` – Eigenvektor-Zentralität
- `nx.center(G)` – Zentrum (Knoten mit minimaler Exzentrizität)

## Graph-Analyse
- `G.nodes` / `G.edges` – Knoten und Kanten
- `G.number_of_nodes()` / `G.number_of_edges()`
- `nx.is_connected(G)` – Ist der Graph zusammenhängend?
- `nx.diameter(G)` – Durchmesser
- `nx.clustering(G)` – Clustering-Koeffizient

## Visualisierung
- `nx.draw(G)` – Einfache Zeichnung (benötigt matplotlib)
- `nx.draw_networkx(G)` – Erweiterte Zeichnung

## Ein-/Ausgabe
- `nx.read_edgelist(path)` / `nx.write_edgelist(G, path)`
- `nx.read_gml(path)` / `nx.write_gml(G, path)`
- `nx.read_graphml(path)` / `nx.write_graphml(G, path)`

## Dokumentation
- Offizielle Dokumentation: https://networkx.org/documentation/stable/

---

*Dies ist eine kompakte Übersicht. Für Details siehe die offizielle Dokumentation!*
