import os
import streamlit as st
import torch
import pandas as pd
from transformers import (BertTokenizer,BertForSequenceClassification)
import plotly.express as px
from transformers.utils import logging


logging.set_verbosity_error()

# ==========================
# PATHS
# ==========================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR,"models")

HISTORY_PATH = os.path.join(BASE_DIR,"outputs","history.json")


# ==========================
# PAGE SETTINGS
# ==========================

st.set_page_config(

    page_title="Toxic Comment Classifier",
    page_icon="🤖",
    layout="wide"
)


# ==========================
# CONFIG
# ==========================

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

MAX_LENGTH=128


LABEL_NAMES=[

    "toxic",
    "severe_toxic",
    "obscene",
    "threat",
    "insult",
    "identity_hate"
]


BEST_THRESHOLDS=[0.50,0.40,0.50,0.35,0.35,0.60]


# ==========================
# LOAD MODEL
# ==========================

@st.cache_resource
def load_model():

    tokenizer=(BertTokenizer.from_pretrained(MODEL_PATH))

    model=(BertForSequenceClassification.from_pretrained(MODEL_PATH))

    model.to(device)

    model.eval()

    return tokenizer,model

tokenizer,model=load_model()

# ==========================
# SIDEBAR
# ==========================

st.sidebar.title("⚙️ Settings")

# st.sidebar.markdown("---")

# selected_example = st.sidebar.selectbox(
#     "💬 Try examples",
#     [
#         "",
#         "Have a nice day",
#         "You are stupid",
#         "I hate you",
#         "I will kill you",
#         "You are an idiot"
#     ]
# )

st.sidebar.markdown("---")

st.sidebar.markdown("## 📊 Model info")
st.sidebar.info(
    """
    Fine-tuned BERT  
    Task: Multi-label Toxic Classification
    """
)

st.sidebar.markdown("---")

st.sidebar.markdown("## 🎯 Toxic classes")

LABEL_NAMES = [
    "toxic",
    "severe_toxic",
    "obscene",
    "threat",
    "insult",
    "identity_hate"
]

icons = {
    "toxic": "☠️",
    "severe_toxic": "🔥",
    "obscene": "🤬",
    "threat": "⚠️",
    "insult": "😡",
    "identity_hate": "🧠"
}

for c in LABEL_NAMES:
    st.sidebar.markdown(
        f"""
        <div style="
            padding:6px;
            border-radius:8px;
            margin-bottom:5px;
            background-color:#1f1f1f;
        ">
            {icons[c]} <b>{c}</b>
        </div>
        """,
        unsafe_allow_html=True
    )

st.sidebar.markdown("---")



# ==========================
# TITLE
# ==========================

st.title(
    "🤖 Toxic Comment Classifier"
)

st.write(

"""
Detect toxicity in comments using BERT.
"""
)



# ==========================
# INPUT
# ==========================

text=st.text_area(

    "Enter comment",

    # value=selected_example,

    height=150
)



# ==========================
# PREDICT
# ==========================

if st.button(
    "Analyze"
):

    if text.strip()=="":

        st.warning(
            "Please enter text"
        )

    else:

        with st.spinner(
            "Analyzing..."
        ):


            inputs=tokenizer(

                text,

                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=MAX_LENGTH
            )


            inputs={

                k:v.to(device)

                for k,v in inputs.items()
            }


            with torch.no_grad():

                outputs=model(
                    **inputs
                )


                probs=(
                    torch.sigmoid(
                        outputs.logits
                    )
                    .cpu()
                    .numpy()[0]
                )


        predictions=(
            probs >
            BEST_THRESHOLDS
        ).astype(int)



        detected=[]

        for label,pred in zip(

            LABEL_NAMES,
            predictions
        ):

            if pred==1:

                detected.append(
                    label
                )



        st.markdown("---")


        if len(
            detected
        )>0:

            st.error(
                "⚠ Toxic comment detected"
            )

            st.write(
                "Detected toxic classes:"
            )

            for x in detected:

                st.write(
                    f"• {x}"
                )

        else:

            st.success(
                "✅ Non-toxic comment"
            )



        # ==================
        # TABLE
        # ==================

        results=pd.DataFrame({

            "Class":
                LABEL_NAMES,

            "Probability":
                probs,

            "Threshold":
                BEST_THRESHOLDS
        })


        results[
            "Probability %"
        ]=(

            results[
                "Probability"
            ]*100

        ).round(2)


        st.subheader(
            "Prediction Details"
        )


        st.dataframe(

            results.sort_values(

                by="Probability %",
                ascending=False
            ),

            use_container_width=True
        )



        # ==================
        # BARS
        # ==================

        st.subheader(
            "Class probabilities"
        )


        fig=px.bar(

            results,

            x="Class",

            y="Probability %"
        )

        st.plotly_chart(

            fig,

            use_container_width=True
        )



        # ==================
        # PROGRESS BARS
        # ==================

        st.subheader(
            "Scores"
        )

        for _,row in results.iterrows():

            st.write(

                f"{row['Class']} : {row['Probability %']}%"
            )

            st.progress(
                float(
                    row[
                        "Probability"
                    ]
                )
            )