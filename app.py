
import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import torch
from transformers import AutoTokenizer, AutoModel
from tensorflow.keras.models import load_model
from sklearn.metrics.pairwise import cosine_similarity


# ============================
# 🔹 1. Chargement des données
# ============================

@st.cache_resource
def load_data():
    # Load the checkpoint data which has image_path and label columns
    df = pd.read_csv("checkpoints/data_with_images.csv")
    
    # Load SBERT embeddings
    embeddings_sbert = np.load("embeddings_sbert.npy")
    
    # Load CNN visual embeddings
    embeddings_visual = np.load("checkpoints/embeddings/cnn_embeddings.npy")
    
    # Note: data_cleaned.parquet has 1428 rows, embeddings_sbert has 1428 embeddings
    # But checkpoints/data_with_images.csv has only 900 rows (valid images)
    # And cnn_embeddings.npy has 895 embeddings
    
    # We need to match the indices properly
    # For now, we'll work with the valid image data (900 rows)
    # and use the first 895 visual embeddings (those that were successfully processed)
    
    # Add SBERT embeddings - match by index from the checkpoint
    # Get the indices from the checkpoint data
    df_full = pd.read_parquet("data_cleaned.parquet")
    
    # Create a mapping from product_code to sbert embedding
    sbert_dict = {df_full.iloc[i]['product_code']: embeddings_sbert[i] 
                  for i in range(len(df_full))}
    
    # Add SBERT embeddings to checkpoint data
    df["embedding_sbert"] = df["product_code"].map(sbert_dict)
    
    # Add visual embeddings (only for the first 895 that have them)
    df["embedding_visual"] = None
    df["embedding_visual"] = df["embedding_visual"].astype(object)
    for i in range(min(len(df), len(embeddings_visual))):
        df.at[i, "embedding_visual"] = embeddings_visual[i]
    
    # Filter out rows without visual embeddings
    df = df[df["embedding_visual"].notna()].copy()
    
    # Convert to proper numpy arrays
    df["embedding_sbert"] = df["embedding_sbert"].apply(lambda x: np.array(x) if x is not None else None)
    df["embedding_visual"] = df["embedding_visual"].apply(lambda x: np.array(x) if x is not None else None)
    
    return df


@st.cache_resource
def load_models():
    from tensorflow.keras.models import Sequential as KerasSequential
    
    # CNN - Charger le modèle complet
    cnn_model = load_model("models/cnn_model.keras", compile=False)
    
    # Créer le modèle d'embedding (extrait les features de la couche dense_1, pas la sortie de classification)
    # La couche dense_1 produit des embeddings de 256 dimensions
    # Pour un modèle Sequential, on crée un nouveau modèle avec les couches jusqu'à dense_1
    dense_1_index = None
    for i, layer in enumerate(cnn_model.layers):
        if layer.name == "dense_1":
            dense_1_index = i
            break
    
    if dense_1_index is None:
        raise ValueError("Couche 'dense_1' non trouvée dans le modèle CNN")
    
    # Créer le modèle d'embedding en incluant jusqu'à dense_1 (inclus)
    cnn_embedding_model = KerasSequential(cnn_model.layers[:dense_1_index + 1])

    # SBERT
    tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
    model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

    return cnn_embedding_model, tokenizer, model


data = load_data()
cnn_model, tokenizer, model = load_models()


# ==============================
# 🔹 2. Fonctions de recommandation
# ==============================

def preprocess_uploaded_image(uploaded_img, target_size=(224, 224)):
    """
    Prétraite une image téléchargée pour l'inference.
    - Convertit en RGB
    - Redimensionne à la taille cible
    - Normalise entre 0 et 1
    """
    # Convertir en RGB (au cas où l'image serait en RGBA, grayscale, etc.)
    img = uploaded_img.convert("RGB")
    
    # Redimensionner
    img = img.resize(target_size, Image.Resampling.LANCZOS)
    
    # Convertir en array numpy et normaliser
    img_array = np.array(img, dtype=np.float32) / 255.0
    
    # Ajouter la dimension batch
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array

def recommend_by_image(uploaded_img, data, cnn_model, top_k=10):
    """
    Recommande des produits similaires à partir d'une image téléchargée.
    L'image peut être de n'importe quelle taille et format.
    """
    # Prétraiter l'image téléchargée
    img = preprocess_uploaded_image(uploaded_img, target_size=(224, 224))

    # Embedding visuel (extrait les features de 256 dimensions)
    query_embedding = cnn_model.predict(img, verbose=0)[0]

    # Matrice des embeddings existants
    X = np.vstack(data["embedding_visual"].values)

    sims = cosine_similarity([query_embedding], X)[0]
    top_indices = sims.argsort()[::-1][:top_k]

    return data.iloc[top_indices], sims[top_indices]


def recommend_by_text(query_text, data, tokenizer, model, top_k=10):
    encoded = tokenizer(
        [query_text],
        padding=True,
        truncation=True,
        return_tensors="pt"
    )

    with torch.no_grad():
        output = model(**encoded)

    def mean_pooling(model_output, attention_mask):
        token_embeddings = model_output.last_hidden_state
        mask = attention_mask.unsqueeze(-1)
        mask = mask.expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * mask, dim=1) / mask.sum(dim=1)

    query_emb = mean_pooling(output, encoded["attention_mask"]).numpy()[0]

    X = np.vstack(data["embedding_sbert"].values)
    sims = cosine_similarity([query_emb], X)[0]
    top_indices = sims.argsort()[::-1][:top_k]

    return data.iloc[top_indices], sims[top_indices]


def recommend_combined(uploaded_img, query_text, data, cnn_model, tokenizer, model, alpha=0.5, top_k=10):
    df_img, sims_img = recommend_by_image(uploaded_img, data, cnn_model, top_k=len(data))
    df_txt, sims_txt = recommend_by_text(query_text, data, tokenizer, model, top_k=len(data))

    sims_img = (sims_img - sims_img.min()) / (sims_img.max() - sims_img.min())
    sims_txt = (sims_txt - sims_txt.min()) / (sims_txt.max() - sims_txt.min())

    combined = alpha * sims_img + (1 - alpha) * sims_txt

    top_indices = combined.argsort()[::-1][:top_k]
    return data.iloc[top_indices], combined[top_indices]


# ==============================
# 🔹 3. Interface Streamlit
# ==============================

st.set_page_config(page_title="Chanel Recommender", layout="wide")
st.title("✨ Plateforme de recommandation — Produits Chanel")


# Choix du mode
mode = st.sidebar.radio(
    "Mode de recherche",
    ["Recherche par image", "Recherche par texte", "Recherche combinée"]
)

st.sidebar.markdown("---")


# ==========================
# 🔹 Mode 1 — Recherche image
# ==========================

if mode == "Recherche par image":
    st.header("🔍 Recherche par image")

    uploaded_img = st.file_uploader("Uploader une image", type=["jpg", "png"])

    if uploaded_img is not None:
        img = Image.open(uploaded_img)
        st.image(img, caption="Image fournie", width=250)

        results, scores = recommend_by_image(img, data, cnn_model)

        st.subheader("Résultats")
        cols = st.columns(5)

        for i, (idx, row) in enumerate(results.head(10).iterrows()):
            with cols[i % 5]:
                st.image(row["imageurl"], width=150)
                st.write(row["title"])
                st.write(f"Score: {scores[i]:.3f}")


# ==========================
# 🔹 Mode 2 — Recherche texte
# ==========================

elif mode == "Recherche par texte":
    st.header("💬 Recherche textuelle")

    query = st.text_input("Décris un produit Chanel (ex: 'red matte lipstick')")

    if query:
        results, scores = recommend_by_text(query, data, tokenizer, model)

        st.subheader("Résultats")
        cols = st.columns(5)

        for i, (idx, row) in enumerate(results.head(10).iterrows()):
            with cols[i % 5]:
                st.image(row["imageurl"], width=150)
                st.write(row["title"])
                st.write(f"Score: {scores[i]:.3f}")


# ==========================
# 🔹 Mode 3 — Recherche combinée
# ==========================

elif mode == "Recherche combinée":
    st.header("🔗 Recherche combinée (image + texte)")

    uploaded_img = st.file_uploader("Uploader une image", type=["jpg","png"])
    query = st.text_input("Ajouter une description textuelle")

    alpha = st.slider("Poids de l'image vs texte", 0.0, 1.0, 0.5)

    if uploaded_img and query:
        img = Image.open(uploaded_img)

        results, scores = recommend_combined(img, query, data, cnn_model, tokenizer, model, alpha)

        st.subheader("Résultats")
        cols = st.columns(5)

        for i, (idx, row) in enumerate(results.head(10).iterrows()):
            with cols[i % 5]:
                st.image(row["imageurl"], width=150)
                st.write(row["title"])
                st.write(f"Score combiné: {scores[i]:.3f}")
