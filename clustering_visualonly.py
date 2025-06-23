import argparse
import json
import os
import numpy as np
import faiss
from tqdm import tqdm
from libKMCUDA import kmeans_cuda

def load_index_and_mapping(index_path, index_json_path):
    print("[INFO] Caricamento indice FAISS e mappatura immagini…")
    index = faiss.read_index(os.path.join(index_path, "knn.index"))
    with open(os.path.join(index_json_path, "knn.json")) as f:
        index_map = json.load(f)
    print(f"[INFO] {index.ntotal} vettori nel FAISS index")
    return index, index_map

def reconstruct_all_features(index):
    total = index.ntotal
    print(f"[INFO] Ricostruzione di {total} feature da FAISS…")
    feats = np.stack([index.reconstruct(i) for i in tqdm(range(total))])
    print(f"[DEBUG] Feature immagine shape = {feats.shape}")
    return feats

def save_cluster_metadata(labels, doc_ids, centroids, features, output_file="clusters_metadata.json"):
    print(f"[INFO] Salvataggio file {output_file}…")
    clusters = {}
    for i, lab in enumerate(labels):
        clusters.setdefault(int(lab), []).append(i)
    out = {}
    for cid, indices in clusters.items():
        center = centroids[cid]
        #  lasciamo l’ordine d’inserimento
        docs = [doc_ids[i] for i in indices]
        out[str(cid)] = {
            "centroid": center.tolist(),
            "documents": docs
        }
    with open(output_file, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[INFO] File salvato: {output_file}")

def cluster_features(features, n_clusters, seed=42, verbosity=1):
    # normalizza per metrica coseno
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    features_norm = features.astype(np.float32) / norms.astype(np.float32)
    print(f"[INFO] Lancio kmeans_cuda: k={n_clusters}, metric=cos…")
    centroids, assignments = kmeans_cuda(
        features_norm,
        n_clusters,
        tolerance=0.01,
        init="k-means++",
        yinyang_t=0.0,
        metric="cos",
        device=0,
        verbosity=verbosity
    )
    print(f"[DEBUG] centroids shape = {centroids.shape}")
    print(f"[DEBUG] assignments shape = {assignments.shape}")
    return assignments, centroids

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--index_path", required=True,
                        help="cartella contenente knn.index")
    parser.add_argument("--index_json_path", required=True,
                        help="cartella contenente knn.json")
    parser.add_argument("--n_clusters", type=int, default=5, #______________________---N clusters
                        help="numero di cluster da creare")
    args = parser.parse_args()

    # Carica indice e feature immagine
    idx, idx_map = load_index_and_mapping(args.index_path, args.index_json_path)
    img_feats = reconstruct_all_features(idx)  # (N, D)
    doc_ids = [entry[0] for entry in idx_map]

    # Clustering solo su feature visive
    labels, cents = cluster_features(img_feats, args.n_clusters)

    # Salvataggio metadata (senza ordinamento interno)
    save_cluster_metadata(labels, doc_ids, cents, img_feats)

    print("[DONE] Clustering visivo completato.")
