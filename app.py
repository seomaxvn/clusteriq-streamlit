import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.cluster import AgglomerativeClustering
import unicodedata
import re

st.set_page_config(page_title="ClusterIQ – BERT-powered Keyword Clustering", layout="wide")
st.title("🔍 ClusterIQ – Semantic Clustering with Sentence-BERT")
st.markdown("Upload file CSV từ khóa, công cụ sẽ phân cụm theo ngữ nghĩa (BERT), gán vai trò Pillar/Cluster và tính tiềm năng SEO.")

uploaded_file = st.file_uploader("📥 Upload file .csv chứa cột 'Keyword'", type="csv")

def slugify(text):
    text = text.lower()
    text = unicodedata.normalize('NFD', text)
    text = text.encode('ascii', 'ignore').decode("utf-8")
    text = re.sub(r'[^\w\s-]', '', text)
    text = re.sub(r'[\s]+', '-', text)
    return '/' + text.strip('-')

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
        pillar_url = slugify(pillar_kw)

        for _, row in group.iterrows():
            role = "Pillar Page" if row["Keyword"] == pillar_kw else "Cluster Content"
            anchor = "" if role == "Pillar Page" else pillar_kw
            target = "" if role == "Pillar Page" else pillar_url
            kos = calculate_kos(row.get("Volume", 0), row.get("Keyword Difficulty", 0), row.get("Intent", "informational"))
            rating = classify_kos(kos)

            records.append({
                "Flexible Cluster ID": cluster_id,
                "Vai trò": role,
                "Keyword": row["Keyword"],
                "Suggested URL": slugify(row["Keyword"]),
                "Liên kết đến Pillar": target,
                "Anchor Text gợi ý": anchor,
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
