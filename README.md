# Projet 8INF919 (UQAC): Plateforme de recommandation des produits Chanel

L'objectif est de développer un système de recommandation multimodal pour des produits de luxe (Chanel). Le projet combine l'analyse de données, la vision par ordinateur (Computer Vision) et le traitement du langage naturel (NLP) pour comparer la pertinence des recommandations basées sur l'image et le texte.

## Contenu du projet

Le projet est divisé en 4 parties principales traitées dans les notebooks et l'application :

- **Partie 1 - Analyse et Préparation**: Exploration du dataset (prix, catégories), identification des biais, nettoyage des descriptions textuelles et prétraitement des images (redimensionnement, normalisation).

- **Partie 2 - Embeddings Visuels**: Extraction et comparaison de vecteurs caractéristiques via deux approches : un modèle CNN entraîné pour la classification et l'utilisation de modèles pré-entraînés (type CLIP ou ResNet). Visualisation des clusters par t-SNE/UMAP.

- **Partie 3 - Embeddings Textuels**: Traduction des descriptions, génération d'embeddings via des modèles NLP (BERT, Sentence-BERT) et analyse de la cohérence sémantique par rapport aux embeddings visuels.

- **Partie 4 - Plateforme de Recommandation**: Développement d'une interface interactive (Streamlit) permettant trois modes de recherche : par image similaire, par description textuelle, ou via une approche combinée (Image + Texte) avec pondération.

## Pré-requis et Installation

Pour exécuter le notebook, il faut 

1. Créer un environnement virtuel :

```bash
python3 -m venv venv
```

2. Activer l'environnement virtuel créé :

```bash
source venv/bin/activate # Sur Linux/Mac
venv\Scripts\activate # Sur Windows
```

3. Installer les dépendances :

```bash
pip install -r requirements.txt
```

4. Exécuter les cellules du notebook `Notebook.ipynb`.

## Auteurs

- BASKAR Arnold
- KRETZ Victor
- MORIN Alban
- NARESH PRABAHARAN Vinith