from neo4j import GraphDatabase
import embedd_node, graph_embedd
import torch
import os
from dotenv import load_dotenv

load_dotenv()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_driver():
    uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    user = os.getenv("NEO4J_USER", "neo4j")
    password = os.getenv("NEO4J_PASSWORD")
    if not password:
        raise ValueError("NEO4J_PASSWORD not found in environment variables. Please set it in .env file")
    return GraphDatabase.driver(uri, auth=(user, password))


def main():
    driver = get_driver()
    try:
        with driver.session() as session:
            embedd_node.delete_old_embeddings(session)
            entities = embedd_node.get_entities_from_neo4j(session)
            if not entities:
                print("Không có entity nào để tạo embedding.")
                return

            combined_texts = [embedd_node.build_text_payload(name, value) for _, name, value in entities]
            embeddings = embedd_node.create_embeddings(combined_texts, device)
            embedd_node.save_embeddings_to_neo4j(session, entities, embeddings)

            graph_embedd.delete_old_graph_embeddings(session)
            nodes, edges = graph_embedd.get_graph_data_from_neo4j(session)
            if not nodes:
                print("Không có node nào trong Neo4j để tạo graph embedding.")
                return

            data, node2idx = graph_embedd.build_pyg_data(nodes, edges)
            data = data.to(device)

            in_channels = data.x.shape[1]
            hidden_channels = 128
            out_channels = 256
            num_layers = 3
            heads = 8
            dropout = 0.2
            num_epochs = 50
            lr = 0.001
            K = 2

            model = graph_embedd.NAGphormer(
                in_channels, hidden_channels, out_channels, K, num_layers, heads, dropout
            ).to(device)
            decoder = graph_embedd.Decoder(out_channels, in_channels).to(device)
            model, decoder = graph_embedd.train_model(model, decoder, data, num_epochs, device, lr=lr)

            model.eval()
            with torch.no_grad():
                num_nodes = data.num_nodes
                edge_index = data.edge_index.to(device)
                values = torch.ones(edge_index.size(1)).to(device)
                A = torch.sparse_coo_tensor(edge_index, values, (num_nodes, num_nodes)).to_dense()
                A = A + torch.eye(num_nodes).to(device)
                D = A.sum(dim=1)
                D_inv_sqrt = torch.diag(torch.pow(D, -0.5))
                A_hat = D_inv_sqrt @ A @ D_inv_sqrt

                graph_embeddings = model(data.x, A_hat)

            graph_embedd.save_graph_embeddings_to_neo4j(session, nodes, node2idx, graph_embeddings)
            print("Huấn luyện và lưu embedding hoàn thành")
    finally:
        driver.close()


if __name__ == "__main__":
    main()
