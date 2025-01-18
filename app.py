import streamlit as st
import pickle
import re
import nltk
from nltk.corpus import stopwords
from tensorflow.keras.preprocessing.sequence import pad_sequences
from nltk.stem.porter import PorterStemmer
nltk.download('stopwords')

# Load saved model and tokenizer
model = pickle.load(open("model.pkl", "rb"))
tokenizer = pickle.load(open("tokenizer.pkl", "rb"))

# Initialize preprocessing components
ps = PorterStemmer()
stop_words = set(stopwords.words('english'))

def preprocessing(content):
    content = re.sub('[^a-zA-Z]', ' ', content)
    content = content.lower()
    content = content.split()
    content = [ps.stem(word) for word in content if not word in stop_words]
    content = " ".join(content)
    return content

# Streamlit app setup
st.title("Fake News Detection")
st.write("This app detects if the input text is fake news or not.")

# Input text from user
user_input = st.text_area("Enter the news article text below:")

if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("Please enter some text for prediction.")
    else:
        # Preprocess the user input
        processed_text = preprocessing(user_input)
        
        # Convert the text into a format the model can understand (using tokenizer)
        input_data = tokenizer.texts_to_sequences([processed_text])
        maxlen=500
        padded_input = pad_sequences(input_data,padding = 'post', maxlen=maxlen)

    
        
        prediction = model.predict(padded_input)
        confidence = float(prediction[0])  # Extract confidence as a percentage
        classification = "Real News" if confidence > 0.5 else "Fake News"

        # Set the colors and emoji depending on sentiment
        if classification == "Real News":
            classification_color = "color:green; font-size:24px; font-weight:bold;"
            emoji = "✅"
        else:
            classification_color = "color:red; font-size:24px; font-weight:bold;"
            emoji = "❌"

        # Display the sentiment using a styled markdown element
        st.markdown(
            f"""
            <div style="text-align:center; padding:20px; border:2px solid {'green' if classification == 'Real News' else 'red'}; 
            border-radius:10px; background-color:#f9f9f9;">
                <p style="{classification_color}">{emoji} Predicted Classification: {classification}</p>
                <p style="font-size:18px;">Confidence Level: <span style="font-weight:bold; color: {'green' if classification == 'Real News' else 'red'}">{confidence * 100:.2f}%</span></p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Display a progress bar for the confidence level
        progress_text = f"Confidence: {confidence * 100:.2f}%"
        st.progress(confidence)