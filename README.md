
# Fake News Classification

A machine learning project to classify fake and real news articles using LSTM and Random Forest classifiers. This project also features an interactive web interface built with **Streamlit**.

## Watch the Video 📺
[![YouTube Video](https://img.shields.io/badge/YouTube-Watch%20Video-red?logo=youtube&logoColor=white&style=for-the-badge)](https://youtu.be/klZtUiRsb6w)

---

![Image](https://github.com/user-attachments/assets/8fa1d4ac-7316-48f2-abcc-22b56f160927)

## 🎯 Project Overview
The goal of this project is to classify whether a news article is fake or real based on the text content. The system uses two powerful models: 
- **LSTM (Long Short-Term Memory)**: An advanced model for handling sequences, like the text of an article.
- **Random Forest Classifier**: An ensemble method for robust predictions and classification.

The project also includes a **Streamlit web application** where users can input text and receive a fake/real prediction.

---

## 🚀 How to Run the Project

### 1. **Clone the Repository**
```bash
git clone https://github.com/DataScientist00/Fake-News-Classification.git
cd Fake-News-Classification
```

### 2. **Install Required Dependencies**
Create a Python virtual environment and install the dependencies from the `requirements.txt`:

```bash
pip install -r requirements.txt
```

### 3. **Run the Streamlit App**
Start the Streamlit app to use the fake news classifier:

```bash
streamlit run app.py
```

The application will be available at `http://localhost:8501`. You can input text and the model will predict if the article is fake or real.

---

## 🧑‍💻 Technologies Used
- **Languages**: Python
- **Libraries**:
  - **Machine Learning**: Keras (LSTM), Scikit-learn (Random Forest, GridSearchCV)
  - **Data Handling**: Pandas, NumPy
  - **Text Preprocessing**: Tokenization, Padding, Stop-word removal
  - **Web Framework**: Streamlit
  - **Visualization**: Matplotlib, Seaborn (optional)
- **Tools**: Jupyter Notebook (for development and exploration)

---

## 🧠 Model Overview

### 1. **LSTM Model**
   - **Input**: News article text
   - **Preprocessing**: Tokenization and padding of text
   - **Model**: A deep learning LSTM-based architecture for sequence classification

### 2. **Random Forest Classifier**
   - **Input**: Text converted into numerical vectors (e.g., TF-IDF or word embeddings)
   - **Model**: An ensemble method, Random Forest, with hyperparameter tuning using **GridSearchCV**

---

## 📊 Results

- **LSTM Model**: Achieved an accuracy of 96.88% 
- **Random Forest Model**: Achieved an accuracy of 89.93%

---

## 🖼 Screenshots

Here’s how the Streamlit app looks:

![Image](https://github.com/user-attachments/assets/5f664101-73c2-435a-9477-1300d3e12c59)

*Example: Input a news article and the app will classify it.*

---

## 🛠 Future Work
- Improve preprocessing techniques like implementing advanced embeddings (e.g., Word2Vec, GloVe) to represent text more effectively.
- Enhance LSTM architecture by exploring variations such as GRU or adding attention mechanisms.
- Further explore ensemble methods or other classification models such as CatBoost.
- Deploy the model for real-time predictions on the web (using Flask or Streamlit) or mobile apps.

---

## 🤝 Contributing

Contributions are always welcome! If you'd like to contribute:
1. Fork the repository
2. Create a branch for your changes
3. Submit a pull request

---

## 📞 Contact
For any questions or feedback, reach out to:
- **Email**: nikzmishra@gmail.com
- Youtube: [Channel](https://www.youtube.com/@NeuralArc00/videos)

---

**⭐ If you found this project helpful, please give it a star! ⭐**
