import tkinter as tk
from tkinter import ttk
import re
from main import *
# 定义一个函数来检查新闻是否为真
def check_news():
    news_text = news_entry.get()
    if len(news_text.strip()) == 0:
        result_label.config(text="请输入新闻内容")
        return
    if answer(news_text)==0:
        result_label.config(text="这篇新闻可能是真的")
    else:
        result_label.config(text="这篇新闻可能是假的")

# 创建主窗口
root = tk.Tk()
root.title("新闻真假辨别")

# 创建一个标签和输入框
news_label = ttk.Label(root, text="请输入新闻内容:", font=("Arial", 14))
news_label.pack(pady=50)

news_entry = ttk.Entry(root, width=80, font=("Arial", 14))
news_entry.pack(pady=10)

# 创建一个按钮和结果标签
check_button = ttk.Button(root, text="检查新闻", command=check_news)
check_button.pack(pady=100)

result_label = ttk.Label(root, text="", font=("Arial", 14))
result_label.pack(pady=10)

root.mainloop()