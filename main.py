import tkinter as tk
from tkinter import filedialog
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def calculate_tfidf_score():
    keyword = keyword_entry.get()
    file_path = file_path_label.cget("text")
    
    if not keyword or not file_path:
        result_label.config(text="Please enter a keyword and select a text file.")
        return

    with open(file_path, 'r', encoding='utf-8') as file:
        corpus = [file.read()]

    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform(corpus)
    
    keyword_vector = tfidf_vectorizer.transform([keyword])
    cosine_similarities = cosine_similarity(keyword_vector, tfidf_matrix)
    score = np.max(cosine_similarities)

    result_label.config(text=f"TF-IDF Score for '{keyword}': {score:.2f}")

def browse_file():
    file_path = filedialog.askopenfilename(filetypes=[("Text files", "*.txt")])
    file_path_label.config(text=file_path)

app = tk.Tk()
app.title("TF-IDF Calculator")
app.geometry("400x200")

title_label = tk.Label(app, text="TF-IDF Score Calculator", font=("Helvetica", 16))
title_label.pack(pady=10)

keyword_label = tk.Label(app, text="Enter a Keyword:")
keyword_label.pack()
keyword_entry = tk.Entry(app)
keyword_entry.pack()

browse_button = tk.Button(app, text="Browse Text File", command=browse_file)
browse_button.pack(pady=10)

file_path_label = tk.Label(app, text="", wraplength=200)
file_path_label.pack()

calculate_button = tk.Button(app, text="Calculate TF-IDF Score", command=calculate_tfidf_score)
calculate_button.pack(pady=10)

result_label = tk.Label(app, text="", font=("Helvetica", 14))
result_label.pack()

app.mainloop()
