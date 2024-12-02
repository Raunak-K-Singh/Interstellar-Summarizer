import tkinter as tk
from tkinter import ttk, messagebox
from summarizer import AdvancedCrossLingualSummarizer

class InterstellarSummarizerGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Interstellar Cross-Lingual Summarizer")
        self.root.geometry("800x600")
        self.root.configure(bg="#0d1117")  # Interstellar dark theme
        
        # Initialize summarizer
        self.summarizer = AdvancedCrossLingualSummarizer()
        
        # Title Label
        title = tk.Label(
            root,
            text="Interstellar Cross-Lingual Summarizer",
            bg="#0d1117",
            fg="#58a6ff",
            font=("Futura", 16, "bold")
        )
        title.pack(pady=10)
        
        # Input Section
        self.input_label = tk.Label(root, text="Input Text:", bg="#0d1117", fg="#c9d1d9")
        self.input_label.pack(anchor="w", padx=20)
        self.input_text = tk.Text(root, wrap="word", height=10, bg="#161b22", fg="#c9d1d9", insertbackground="#c9d1d9")
        self.input_text.pack(fill="both", padx=20, pady=5)
        
        # Language Dropdown Section
        lang_frame = tk.Frame(root, bg="#0d1117")
        lang_frame.pack(pady=10)
        
        self.source_lang_label = tk.Label(lang_frame, text="Source Language:", bg="#0d1117", fg="#c9d1d9")
        self.source_lang_label.grid(row=0, column=0, padx=10)
        self.source_lang_dropdown = ttk.Combobox(lang_frame, values=list(self.summarizer.language_codes.keys()), state="readonly")
        self.source_lang_dropdown.grid(row=0, column=1)
        self.source_lang_dropdown.set("en")  # Default
        
        self.target_lang_label = tk.Label(lang_frame, text="Target Language:", bg="#0d1117", fg="#c9d1d9")
        self.target_lang_label.grid(row=0, column=2, padx=10)
        self.target_lang_dropdown = ttk.Combobox(lang_frame, values=list(self.summarizer.language_codes.keys()), state="readonly")
        self.target_lang_dropdown.grid(row=0, column=3)
        self.target_lang_dropdown.set("en")  # Default
        
        # Strategy and Settings
        settings_frame = tk.Frame(root, bg="#0d1117")
        settings_frame.pack(pady=10)
        
        self.strategy_label = tk.Label(settings_frame, text="Strategy:", bg="#0d1117", fg="#c9d1d9")
        self.strategy_label.grid(row=0, column=0, padx=10)
        self.strategy_dropdown = ttk.Combobox(settings_frame, values=self.summarizer.summarization_strategies, state="readonly")
        self.strategy_dropdown.grid(row=0, column=1)
        self.strategy_dropdown.set("abstractive")  # Default
        
        self.max_length_label = tk.Label(settings_frame, text="Max Length:", bg="#0d1117", fg="#c9d1d9")
        self.max_length_label.grid(row=0, column=2, padx=10)
        self.max_length_spinbox = tk.Spinbox(settings_frame, from_=50, to=500, width=5)
        self.max_length_spinbox.grid(row=0, column=3)
        self.max_length_spinbox.delete(0, "end")
        self.max_length_spinbox.insert(0, "150")  # Default
        
        # Buttons
        button_frame = tk.Frame(root, bg="#0d1117")
        button_frame.pack(pady=10)
        
        summarize_button = tk.Button(button_frame, text="Summarize", bg="#21262d", fg="#58a6ff", command=self.perform_summarization)
        summarize_button.grid(row=0, column=0, padx=20)
        
        clear_button = tk.Button(button_frame, text="Clear", bg="#21262d", fg="#f85149", command=self.clear_all)
        clear_button.grid(row=0, column=1, padx=20)
        
        # Output Section
        self.output_label = tk.Label(root, text="Generated Summaries:", bg="#0d1117", fg="#c9d1d9")
        self.output_label.pack(anchor="w", padx=20)
        self.output_text = tk.Text(root, wrap="word", height=10, bg="#161b22", fg="#c9d1d9", insertbackground="#c9d1d9")
        self.output_text.pack(fill="both", padx=20, pady=5)
    
    def perform_summarization(self):
        """Perform summarization using the backend summarizer."""
        input_text = self.input_text.get("1.0", tk.END).strip()
        source_lang = self.source_lang_dropdown.get()
        target_lang = self.target_lang_dropdown.get()
        strategy = self.strategy_dropdown.get()
        max_length = int(self.max_length_spinbox.get())
        
        if not input_text:
            messagebox.showerror("Error", "Input text is required!")
            return
        
        try:
            summaries = self.summarizer.summarize(
                text=input_text,
                source_lang=source_lang,
                target_lang=target_lang,
                strategy=strategy,
                max_length=max_length
            )
            self.output_text.delete("1.0", tk.END)
            for idx, summary in enumerate(summaries, start=1):
                self.output_text.insert(tk.END, f"Summary {idx}:\n{summary}\n\n")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to summarize text: {e}")
    
    def clear_all(self):
        """Clear all inputs and outputs."""
        self.input_text.delete("1.0", tk.END)
        self.output_text.delete("1.0", tk.END)

if __name__ == "__main__":
    root = tk.Tk()
    app = InterstellarSummarizerGUI(root)
    root.mainloop()
