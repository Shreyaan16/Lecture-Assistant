# Lecture Assistant: Automated Lecture Summarizer & Quiz Generator

Lecture Assistant is an AI-powered tool designed to help students quickly grasp lecture content by generating concise summaries, FAQs, flashcards, reference materials, and practice exams. The application supports multiple document formats and leverages state-of-the-art NLP techniques for efficient processing.

## 🚀 Features

### 📂 Upload Documents
- Supports **PPT, PDF, XLSX, and TXT** formats for lecture materials.

### 🎯 Topic Selection
- Users can specify the lecture topic for more accurate content generation.

### 📜 Summarization
- Generates a concise summary using **Facebook BART CNN Large** and **Hugging Face** models.
- Allows users to provide feedback to refine the generated summary.

### ❓ FAQ Generation
- Automatically creates FAQs for **Easy, Medium, and Hard** difficulty levels.
- Provides answers and an option to regenerate FAQs if needed.

### 🃏 Flashcards
- Generates flashcards for different difficulty levels based on user selection.

### 🗨️ QNA Chatbot
- Has memory of previous prompts and answers to help the user clear his/her doubts

### 🔗 Reference Materials
- Fetches **YouTube videos** and **relevant reference links** using **LangChain tools**.

### 📝 Multiple-Correct Question (MCQ) Exam
- Generates an MCQ-based exam using **Gemini** based on the selected difficulty level.

## 🛠️ Tech Stack
- **NLP & NER**: spaCy
- **Summarization**: Facebook BART CNN Large & Hugging Face
- **Flashcards & Exam Questions**: Gemini
- **YouTube & Reference Links**: LangChain tools
- **Frontend**: Streamlit
- **Deployment**: Hugging Face Spaces

## 🌐 Live Demo
The application is deployed on Hugging Face Spaces:
👉 **[Lecture Assistant - Try it Here](https://huggingface.co/spaces/Shreyaan16/LectureAssistant)**

### ⚠️ Known Issues
- **YouTube video retrieval**: The deployed version on Hugging Face Spaces encounters a **cookie error** while fetching YouTube videos. Until fixed, you can run the application locally using Streamlit.

## 🏗️ Installation & Running Locally

### 📥 Clone the Repository
```bash
git clone https://github.com/Shreyaan16/Lecture-Assistant.git
cd Lecture-Assistant
```

### 📦 Install Dependencies
```bash
pip install -r requirements.txt
```

### 🚀 Run the App
```bash
streamlit run app.py
```

## 📽️ Video Demo
A video demonstration showcasing the tool's features is available [here](#). (Add link if available)

## 📜 License
This project should have been licensed under the MIT License but I am lazy.

---
Contributions and feedback are welcome! 🚀

