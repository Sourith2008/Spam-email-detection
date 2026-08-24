import streamlit as st
import joblib
import math

model=joblib.load('models/best_model.pkl')
vectorizer=joblib.load('models/tfidf_vectorizer.pkl')

st.title("📧 Spam Email Detector")

st.write(
    "Enter an email below and the model will classify it "
    "as Spam or Not Spam."
)

email=st.text_area(
    "Enter an E-mail: "
)
if st.button("Detect spam"):
    if email.strip()=="":
        st.warning("Please enter an E-mail first.")
    else:
        email_vect=vectorizer.transform([email])
        prediction=model.predict(email_vect)
        score=model.decision_function(email_vect)[0]
        if prediction[0]==1:
            confidence=1/(1+math.exp(-score))
            st.error("🚨 SPAM DETECTED")
        else:
            confidence=1/(1+math.exp(-score))
            st.success("✅ NOT SPAM")
        st.write(f"Confidence: {confidence*100:.2f}%")