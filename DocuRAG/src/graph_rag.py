from typing import List, Dict
import networkx as nx
import matplotlib.pyplot as plt

class GraphRAG:
    def __init__(self):
        self.graph = nx.Graph()

    def add_node(self, node_id: str, **attributes) -> None:
        self.graph.add_node(node_id, **attributes)

    def add_edge(self, source: str, target: str, **attributes) -> None:
        self.graph.add_edge(source, target, **attributes)

    def visualize(self) -> None:
        pos = nx.spring_layout(self.graph)
        nx.draw(self.graph, pos, with_labels=True, node_color='lightblue', node_size=2000, font_size=10, font_color='black', font_weight='bold')
        plt.title("Graph RAG Visualization")
        plt.show()

    def extract_relationships(self, data: List[Dict]) -> None:
        for item in data:
            node_id = item.get('id')
            related_ids = item.get('related_ids', [])
            self.add_node(node_id, **item)
            for related_id in related_ids:
                self.add_edge(node_id, related_id)

    def clear_graph(self) -> None:
        self.graph.clear()