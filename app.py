import re
from pathlib import Path
from typing import List, Optional, Dict

import cv2
import numpy as np
import pandas as pd
import pytesseract
import streamlit as st
from pdf2image import convert_from_bytes

# ================== CONFIG ==================

# Sur Windows LOCAL, tu peux mettre un chemin vers Poppler :
# POPPLER_PATH = r"C:\tools\poppler-24.02.0\Library\bin"
# Sur Streamlit Cloud (Linux), on laisse None.
POPPLER_PATH: Optional[str] = None

# Sur Windows LOCAL, si Tesseract n'est pas dans le PATH, décommente :
# pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}

# ================== REGEX PATTERNS ==================

# ---- BL EXTERNE ----
BL_PATTERNS: List[re.Pattern] = [
    # Cas normal : "BL externe : 215230-316215"
    # ou "BL externe . 215230-316215", etc.
    re.compile(
        r"BL\W*externe\W*([0-9]{4,}[-/][0-9]{4,})",
        flags=re.IGNORECASE,
    ),
    # Petites erreurs d’OCR : BI / B1 / 8L, exterme, etc.
    re.compile(
        r"B[IL18]\W*exter[nm]e\W*([0-9A-Za-z\-_/]{5,})",
        flags=re.IGNORECASE,
    ),
]

# ---- ID CAMION ----
ID_CAMION_PATTERNS: List[re.Pattern] = [
    re.compile(
        r"ID\W*CAMI?ON[^\n]*?:\W*([A-Z0-9\-]+)",
        flags=re.IGNORECASE,
    ),
]

# ---- ID CHAUFFEUR ----
ID_CHAUFFEUR_PATTERNS: List[re.Pattern] = [
    re.compile(
        r"ID\W*CHAUFFEUR[^\n]*?:\W*([A-Z0-9 ]+?)(?=\s+Chauffeur\b|$)",
        flags=re.IGNORECASE,
    ),
]

# ================== OCR CORE ==================


def ocr_cv2_image(img: np.ndarray, lang: str = "eng+fra") -> str:
    """OCR sur une image cv2 (BGR)."""
    if img is None:
        raise ValueError("ocr_cv2_image received None image")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
    gray = cv2.bilateralFilter(gray, 9, 75, 75)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    config = r"--psm 6 --oem 3"
    text = pytesseract.image_to_string(thresh, config=config, lang=lang)
    return text


def ocr_image_uploaded(uploaded_file, lang: str = "eng+fra") -> str:
    """OCR pour un fichier image uploadé (Streamlit UploadedFile)."""
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    return ocr_cv2_image(img, lang=lang)


def ocr_pdf_uploaded(uploaded_file, lang: str = "eng+fra") -> List[str]:
    """OCR de toutes les pages d’un PDF uploadé. Retourne une liste de textes (une par page)."""
    pdf_bytes = uploaded_file.getvalue()
    pil_pages = convert_from_bytes(pdf_bytes, dpi=300, poppler_path=POPPLER_PATH)

    texts: List[str] = []
    for pil_img in pil_pages:
        img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        texts.append(ocr_cv2_image(img, lang=lang))
    return texts


# ================== EXTRACTORS ==================


def _best_candidate(candidates: List[str]) -> str:
    if not candidates:
        return ""

    def score(v: str) -> int:
        digit_count = sum(ch.isdigit() for ch in v)
        hyphen_bonus = 2 if "-" in v or "/" in v else 0
        return digit_count * 10 + hyphen_bonus + len(v)

    return max(candidates, key=score)


def extract_bl_externe(text: str) -> str:
    candidates: List[str] = []

    for pattern in BL_PATTERNS:
        for m in pattern.finditer(text):
            val = m.group(1).strip(" \t\r\n.,;:")
            if len(val) < 5:
                continue
            if not any(c.isdigit() for c in val):
                continue
            candidates.append(val)

    return _best_candidate(candidates)


def extract_id_camion(text: str) -> str:
    candidates: List[str] = []

    for pattern in ID_CAMION_PATTERNS:
        for m in pattern.finditer(text):
            val = m.group(1).strip(" \t\r\n.,;:")
            if len(val) < 3:
                continue
            if not any(c.isalnum() for c in val):
                continue
            candidates.append(val)

    if not candidates:
        return ""
    return max(candidates, key=len)


def extract_id_chauffeur(text: str) -> str:
    candidates: List[str] = []

    for pattern in ID_CHAUFFEUR_PATTERNS:
        for m in pattern.finditer(text):
            val = m.group(1).strip(" \t\r\n.,;:")
            if len(val) < 2:
                continue
            if not any(c.isalnum() for c in val):
                continue
            candidates.append(val)

    if not candidates:
        return ""
    return max(candidates, key=len)


# ================== TRAITEMENT DES FICHIERS ==================


def process_files(uploaded_files, lang: str = "eng+fra") -> pd.DataFrame:
    """
    Traite une liste de fichiers uploadés.
    Retourne un DataFrame avec colonnes :
    ['BL externe', 'ID chauffeur', 'ID camion'].

    Si pour une page/doc les 3 champs sont vides, on ne garde pas de ligne.
    """
    rows: List[Dict[str, str]] = []

    for file in uploaded_files:
        suffix = Path(file.name).suffix.lower()
        file.seek(0)  # reset le pointeur par sécurité

        if suffix == ".pdf":
            texts = ocr_pdf_uploaded(file, lang=lang)
        elif suffix in IMAGE_EXTS:
            texts = [ocr_image_uploaded(file, lang=lang)]
        else:
            continue  # type non géré (normalement filtré par file_uploader)

        for text in texts:
            bl_val = extract_bl_externe(text)
            id_camion = extract_id_camion(text)
            id_chauffeur = extract_id_chauffeur(text)

            if not (bl_val or id_camion or id_chauffeur):
                continue

            rows.append(
                {
                    "BL externe": bl_val,
                    "ID chauffeur": id_chauffeur,
                    "ID camion": id_camion,
                }
            )

    if not rows:
        return pd.DataFrame(columns=["BL externe", "ID chauffeur", "ID camion"])

    return pd.DataFrame(rows, columns=["BL externe", "ID chauffeur", "ID camion"])


# ================== STREAMLIT UI ==================


def inject_css():
    st.markdown(
        """
        <style>
        .main {
            background-color: #05070d;
        }
        .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
            max-width: 1100px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def main():
    st.set_page_config(
        page_title="Extraction BL / ID Camion / ID Chauffeur",
        page_icon="🧪",
        layout="wide",
    )

    inject_css()

    # Titre propre
    st.markdown("## ✅ Extraction des données de réception")

    st.markdown("#### Fichier(s) (PDF / Images)")
    st.markdown(
        "Glissez-déposez **plusieurs fichiers** (PDF, JPG, PNG, TIFF, …) "
        "ou cliquez sur *Browse files*. "
        "Depuis l’explorateur tu peux sélectionner tout le contenu d’un dossier et le déposer ici."
    )

    uploaded_files = st.file_uploader(
        "Fichiers à traiter",
        type=["pdf", "png", "jpg", "jpeg", "tif", "tiff", "bmp"],
        accept_multiple_files=True,
        key="file_uploader",
        help="Sélectionne tous les fichiers de ton dossier et glisse-les ici.",
    )

    st.markdown("<br>", unsafe_allow_html=True)

    col1, col2 = st.columns([1, 3])
    with col1:
        run = st.button("🚀 Lancer le traitement")

    if run:
        if not uploaded_files:
            st.warning("Ajoute au moins un fichier avant de lancer le traitement.")
        else:
            with st.spinner("Traitement OCR en cours..."):
                df_result = process_files(uploaded_files, lang="eng+fra")

            if df_result.empty:
                st.info(
                    "Aucune donnée trouvée (BL externe / ID chauffeur / ID camion)."
                )
            else:
                st.success(
                    f"Traitement terminé — {len(df_result)} ligne(s) extraite(s)."
                )
                st.markdown("#### Aperçu des résultats")
                st.dataframe(df_result, use_container_width=True)

                csv_bytes = df_result.to_csv(index=False).encode("utf-8-sig")
                st.download_button(
                    label="💾 Télécharger le CSV",
                    data=csv_bytes,
                    file_name="extraction_bl_id.csv",
                    mime="text/csv",
                )


if __name__ == "__main__":
    main()
