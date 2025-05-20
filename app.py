import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.cluster import AgglomerativeClustering
import unicodedata
import re
import networkx as nx
from pyvis.network import Network
import streamlit.components.v1 as components
import tempfile

st.set_page_config(page_title="ClusterIQ – BERT-powered Keyword Clustering", layout="wide")
st.title("🔍 ClusterIQ – Semantic Clustering with Sentence-BERT")
st.markdown("Upload file CSV từ khóa, công cụ sẽ phân cụm theo ngữ nghĩa (BERT), gán vai trò Pillar/Cluster và tính tiềm năng SEO.")

uploaded_file = st.file_uploader("📥 Upload file .csv chứa cột 'Keyword'", type="csv")

def calculate_kos(volume, difficulty, intent):
    intent_weight = {
        "transactional": 1.5,
        "commercial": 1.2,
        "informational": 1.0,
        "navigational": 0.8
    }
    try: volume = float(volume)
    except: volume = 0.0
    try: difficulty = float(difficulty)
    except: difficulty = 0.0
    weight = intent_weight.get(str(intent).lower().strip(), 1.0)
    return (volume * weight) / (difficulty + 1)

def classify_kos(score):
    if score > 100: return "🔥 Rất tiềm năng"
    elif score > 50: return "✅ Tiềm năng vừa"
    elif score > 20: return "⚠️ Tiềm năng thấp"
    else: return "❌ Không nên ưu tiên"

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    if "Primary Keyword" in df.columns:
        df["Keyword"] = df["Primary Keyword"]
    elif "Keyword" not in df.columns:
        st.error("⚠️ File cần có cột 'Keyword' hoặc 'Primary Keyword'")
        st.stop()

    st.info("📦 Đang tải Sentence-BERT model...")
    model = SentenceTransformer("all-mpnet-base-v2")
    embeddings = model.encode(df["Keyword"].dropna().tolist(), show_progress_bar=True)

    st.info("🔍 Đang phân cụm từ khóa...")
    clustering = AgglomerativeClustering(n_clusters=None, distance_threshold=1.2)
    df["Flexible Cluster ID"] = clustering.fit_predict(embeddings)

    st.info("🧱 Đang phân vai trò và tính điểm KOS...")
    records = []
    for cluster_id, group in df.groupby("Flexible Cluster ID"):
        if len(group) < 2:
            continue
        pillar_row = group.loc[group["Keyword"].apply(len).idxmin()]
        pillar_kw = pillar_row["Keyword"]

        for _, row in group.iterrows():
            role = "Pillar Page" if row["Keyword"] == pillar_kw else "Cluster Content"
            kos = calculate_kos(row.get("Volume", 0), row.get("Keyword Difficulty", 0), row.get("Intent", "informational"))
            rating = classify_kos(kos)

            records.append({
                "Flexible Cluster ID": cluster_id,
                "Vai trò": role,
                "Keyword": row["Keyword"],
                "Intent": row.get("Intent", ""),
                "Volume": row.get("Volume", ""),
                "Difficulty": row.get("Keyword Difficulty", ""),
                "KOS": round(kos, 1),
                "Mức độ tiềm năng": rating
            })

    result_df = pd.DataFrame(records)

    st.success("✅ Xử lý xong!")
    st.dataframe(result_df)

    csv = result_df.to_csv(index=False).encode("utf-8")
    st.download_button("📥 Tải file kết quả", data=csv, file_name="clusteriq_bert_result.csv", mime="text/csv")

    # Hiển thị sơ đồ cụm từ khóa dạng network graph
    st.subheader("🌐 Sơ đồ phân cụm từ khóa (Interactive Network)")
    G = nx.Graph()

    for cluster_id, group in result_df.groupby("Flexible Cluster ID"):
        if len(group) < 2:
            continue
        pillar = group[group["Vai trò"] == "Pillar Page"]["Keyword"].values[0]
        G.add_node(pillar, label=pillar, color="#2F80ED")
        for _, row in group.iterrows():
            if row["Vai trò"] == "Cluster Content":
                G.add_node(row["Keyword"], label=row["Keyword"], color="#56CC9D")
                G.add_edge(pillar, row["Keyword"])

    net = Network(height="600px", width="100%", bgcolor="#ffffff", font_color="black")
    net.from_nx(G)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".html") as tmp_file:
        net.save_graph(tmp_file.name)
        tmp_file.seek(0)
        components.html(tmp_file.read().decode("utf-8"), height=650, scrolling=True)
