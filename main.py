import tkinter as tk
from tkinter import filedialog
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

languages = {
    'en': {
        'title': 'TF-IDF Calculator',
        'enter_keyword': 'Enter a Keyword:',
        'browse_file': 'Browse Text File',
        'calculate_score': 'Calculate TF-IDF Score',
        'result_prefix': "TF-IDF Score for '{keyword}': {score:.2f}",
        'select_file': 'Please select a text file.',
    },
    'ja': {
        'title': 'TF-IDF 計算機',
        'enter_keyword': 'キーワードを入力:',
        'browse_file': 'テキストファイルを選択',
        'calculate_score': 'TF-IDF スコアを計算',
        'result_prefix': "'{keyword}' の TF-IDF スコア: {score:.2f}",
        'select_file': 'テキストファイルを選択してください。',
    },
    'ar': {
        'title': 'آلة حاسبة TF-IDF',
        'enter_keyword': 'أدخل كلمة مفتاحية:',
        'browse_file': 'تصفح ملف النص',
        'calculate_score': 'حساب درجة TF-IDF',
        'result_prefix': "نتيجة TF-IDF للكلمة '{keyword}': {score:.2f}",
        'select_file': 'الرجاء تحديد ملف نص.',
    },
}

current_language = 'en'  # Default to English

def change_language(lang):
    global current_language
    current_language = lang
    update_ui()

def update_ui():
    app.title(languages[current_language]['title'])
    keyword_label.config(text=languages[current_language]['enter_keyword'])
    browse_button.config(text=languages[current_language]['browse_file'])
    calculate_button.config(text=languages[current_language]['calculate_score'])

def calculate_tfidf_score():
    keyword = keyword_entry.get()
    file_path = file_path_label.cget("text")
    
    if not keyword or not file_path:
        result_label.config(text=languages[current_language]['select_file'])
        return

    with open(file_path, 'r', encoding='utf-8') as file:
        corpus = [file.read()]

    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform(corpus)
    
    keyword_vector = tfidf_vectorizer.transform([keyword])
    cosine_similarities = cosine_similarity(keyword_vector, tfidf_matrix)
    score = np.max(cosine_similarities)

    result_label.config(text=languages[current_language]['result_prefix'].format(keyword=keyword, score=score))

def browse_file():
    file_path = filedialog.askopenfilename(filetypes=[("Text files", "*.txt")])
    file_path_label.config(text=file_path)

app = tk.Tk()
app.title(languages[current_language]['title'])
app.geometry("460x300")
app.configure(bg="black")

title_label = tk.Label(app, text=languages[current_language]['title'], font=("Helvetica", 16), fg="white", bg="black")
title_label.pack(pady=10)

keyword_label = tk.Label(app, text=languages[current_language]['enter_keyword'], fg="white", bg="black")
keyword_label.pack()
keyword_entry = tk.Entry(app)
keyword_entry.pack()

browse_button = tk.Button(app, text=languages[current_language]['browse_file'], command=browse_file)
browse_button.pack(pady=10)

file_path_label = tk.Label(app, text="", wraplength=200, fg="white", bg="black")
file_path_label.pack()

calculate_button = tk.Button(app, text=languages[current_language]['calculate_score'], command=calculate_tfidf_score)
calculate_button.pack(pady=10)

result_label = tk.Label(app, text="", font=("Helvetica", 14), fg="white", bg="black")
result_label.pack()

english_button = tk.Button(app, text="English", command=lambda: change_language('en'))
english_button.pack(side="left")
japanese_button = tk.Button(app, text="日本語", command=lambda: change_language('ja'))
japanese_button.pack(side="left")
arabic_button = tk.Button(app, text="العربية", command=lambda: change_language('ar'))
arabic_button.pack(side="left")

update_ui()

app.mainloop()
#KiraiEEE
