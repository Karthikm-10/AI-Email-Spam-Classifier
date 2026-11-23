
import streamlit as st
import joblib
import re

# -----------------------------
# PAGE CONFIGURATION
# -----------------------------

st.set_page_config(
    page_title="AI Email Spam Classifier",
    page_icon="📧",
    layout="centered"
)


# -----------------------------
# LOAD TRAINED MODEL
# -----------------------------

@st.cache_resource
def load_model():
    return joblib.load("models/spam_pipeline_comparison.pkl")


model = load_model()


# -----------------------------
# TEXT PREPROCESSING
# -----------------------------

def clean_text(text):

    text = text.lower()

    text = re.sub(
        r"http\S+|www\S+",
        " ",
        text
    )

    text = re.sub(
        r"\S+@\S+",
        " ",
        text
    )

    text = re.sub(
        r"[^a-zA-Z\s]",
        " ",
        text
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    ).strip()

    return text


# -----------------------------
# SIDEBAR
# -----------------------------

st.sidebar.title("📧 Project Information")

st.sidebar.write(
    "AI-powered email spam classification "
    "using Machine Learning."
)

st.sidebar.divider()

st.sidebar.subheader("Technologies")

st.sidebar.write(
    """
    - Python
    - Pandas
    - NumPy
    - Scikit-learn
    - TF-IDF
    - Logistic Regression
    - Streamlit
    """
)

st.sidebar.divider()

st.sidebar.caption(
    "Academic / Portfolio Machine Learning Project"
)


# -----------------------------
# MAIN APPLICATION
# -----------------------------

st.title("📧 AI Email Spam Classifier")

st.write(
    "Enter an email message below. "
    "Our machine learning model will classify "
    "it as Spam or Not Spam."
)

st.divider()


# -----------------------------
# EMAIL INPUT FORM
# -----------------------------

with st.form("email_form"):

    email_text = st.text_area(
        "Enter your email message",
        height=200,
        placeholder=(
            "Paste your email content here..."
        )
    )

    submitted = st.form_submit_button(
        "Analyze Email",
        type="primary",
        use_container_width=True
    )


# -----------------------------
# PREDICTION
# -----------------------------

if submitted:

    if not email_text.strip():

        st.warning(
            "Please enter an email message."
        )

    else:

        cleaned_email = clean_text(email_text)

        if not cleaned_email:

            st.warning(
                "Please enter meaningful email text."
            )

        else:

            prediction = model.predict(
                [cleaned_email]
            )[0]

            probabilities = model.predict_proba(
                [cleaned_email]
            )[0]

            probability_dict = dict(
                zip(model.classes_, probabilities)
            )

            spam_score = probability_dict["spam"]
            ham_score = probability_dict["ham"]

            st.divider()

            st.subheader("Classification Result")

            if prediction == "spam":

                st.error(
                    "🚨 This email is classified as SPAM."
                )

            else:

                st.success(
                    "✅ This email is classified as NOT SPAM."
                )

            st.subheader("Prediction Scores")

            col1, col2 = st.columns(2)

            with col1:

                st.metric(
                    "Spam score",
                    f"{spam_score * 100:.2f}%"
                )

                st.progress(float(spam_score))

            with col2:

                st.metric(
                    "Ham score",
                    f"{ham_score * 100:.2f}%"
                )

                st.progress(float(ham_score))

            st.caption(
                "These are model-estimated class probabilities, "
                "not guarantees that an email is safe."
            )


# -----------------------------
# FOOTER
# -----------------------------

st.divider()

st.caption(
    "Developed using Python, Scikit-learn, "
    "TF-IDF and Logistic Regression."
)
