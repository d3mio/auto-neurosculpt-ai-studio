import tkinter as tk
from tkinter import ttk
import customtkinter as ctk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
import shap
from lime import lime_tabular
import pandas as pd
import threading

ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")

class NeuroSculptAIStudio:
    def __init__(self, root):
        self.root = root
        self.root.title("NeuroSculpt AI Studio")
        self.root.geometry("1200x800")
        self.root.minsize(1000, 700)
        
        self.create_widgets()
        self.setup_layout()
    
    def create_widgets(self):
        # Main frames
        self.top_frame = ctk.CTkFrame(self.root, corner_radius=10)
        self.left_frame = ctk.CTkFrame(self.root, corner_radius=10)
        self.center_frame = ctk.CTkFrame(self.root, corner_radius=10)
        self.right_frame = ctk.CTkFrame(self.root, corner_radius=10)
        
        # Top frame components
        self.title_label = ctk.CTkLabel(self.top_frame, text="NEUROSCULPT AI STUDIO", 
                                      font=ctk.CTkFont(size=20, weight="bold"))
        self.model_load_btn = ctk.CTkButton(self.top_frame, text="Load Model", command=self.load_model)
        self.status_label = ctk.CTkLabel(self.top_frame, text="Status: Ready", text_color="lightgreen")
        
        # Left frame - Model inspector
        self.model_tree = ttk.Treeview(self.left_frame, height=20)
        self.model_tree["columns"] = ("value",)
        self.model_tree.column("#0", width=200, minwidth=100)
        self.model_tree.column("value", width=150, minwidth=50)
        self.model_tree.heading("#0", text="Layer")
        self.model_tree.heading("value", text="Params")
        
        self.whatif_frame = ctk.CTkFrame(self.left_frame, corner_radius=5)
        self.whatif_label = ctk.CTkLabel(self.whatif_frame, text="What-If Analysis")
        self.feature_sliders = []
        for i in range(3):
            slider = ctk.CTkSlider(self.whatif_frame, from_=0, to=1, orientation="horizontal")
            slider.set(0.5)
            self.feature_sliders.append(slider)
        self.whatif_btn = ctk.CTkButton(self.whatif_frame, text="Run Analysis", command=self.run_whatif)
        
        # Center frame - Visualization
        self.tabview = ctk.CTkTabview(self.center_frame, width=600, height=500)
        self.tabview.add("Architecture")
        self.tabview.add("SHAP")
        self.tabview.add("LIME")
        self.tabview.add("Bias")
        
        self.figure = plt.Figure(figsize=(6, 4), dpi=100)
        self.ax = self.figure.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.figure, master=self.tabview.tab("Architecture"))
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Right frame - Console & Controls
        self.console = ctk.CTkTextbox(self.right_frame, width=300, height=200, font=("Consolas", 12))
        self.console.insert("0.0", "NeuroSculpt AI Studio initialized\n")
        
        self.interpret_btn = ctk.CTkButton(self.right_frame, text="Run SHAP Analysis", command=self.run_shap)
        self.lime_btn = ctk.CTkButton(self.right_frame, text="Run LIME Analysis", command=self.run_lime)
        self.bias_btn = ctk.CTkButton(self.right_frame, text="Check Bias", command=self.check_bias)
        
        self.progress = ctk.CTkProgressBar(self.right_frame, orientation="horizontal")
        self.progress.set(0)
    
    def setup_layout(self):
        self.top_frame.pack(side="top", fill="x", padx=10, pady=5)
        self.left_frame.pack(side="left", fill="y", padx=10, pady=5)
        self.center_frame.pack(side="left", fill="both", expand=True, padx=10, pady=5)
        self.right_frame.pack(side="right", fill="y", padx=10, pady=5)
        
        # Top frame layout
        self.title_label.pack(side="left", padx=20, pady=10)
        self.model_load_btn.pack(side="right", padx=20, pady=10)
        self.status_label.pack(side="right", padx=20, pady=10)
        
        # Left frame layout
        self.model_tree.pack(fill="both", expand=True, padx=5, pady=5)
        self.whatif_frame.pack(fill="x", padx=5, pady=10)
        self.whatif_label.pack(pady=5)
        for slider in self.feature_sliders:
            slider.pack(fill="x", padx=10, pady=2)
        self.whatif_btn.pack(pady=5)
        
        # Center frame layout
        self.tabview.pack(fill="both", expand=True, padx=5, pady=5)
        
        # Right frame layout
        self.console.pack(fill="x", padx=5, pady=5)
        self.interpret_btn.pack(fill="x", padx=5, pady=5)
        self.lime_btn.pack(fill="x", padx=5, pady=5)
        self.bias_btn.pack(fill="x", padx=5, pady=5)
        self.progress.pack(fill="x", padx=5, pady=15)
        
        # Sample visualization
        self.update_sample_plot()
    
    def update_sample_plot(self):
        self.ax.clear()
        x = np.linspace(0, 10, 100)
        y = np.sin(x)
        self.ax.plot(x, y)
        self.ax.set_title("Model Architecture Graph")
        self.canvas.draw()
    
    def load_model(self):
        self.console.insert("end", "Loading model...\n")
        self.progress.start()
        
        # Simulate model loading
        threading.Thread(target=self._simulate_model_load).start()
    
    def _simulate_model_load(self):
        import time
        for i in range(1, 6):
            time.sleep(0.5)
            self.progress.set(i/5)
        
        # Add sample data to tree
        layers = [
            ("Input", "784"),
            ("Dense_1", "128"),
            ("BatchNorm", "256"),
            ("Dropout", "0.2"),
            ("Output", "10")
        ]
        
        for layer in layers:
            self.model_tree.insert("", "end", text=layer[0], values=(layer[1],))
        
        self.status_label.configure(text="Status: Model Loaded")
        self.progress.stop()
        self.progress.set(0)
        self.console.insert("end", "Model loaded successfully!\n")
    
    def run_shap(self):
        self.console.insert("end", "Running SHAP analysis...\n")
        
        # Simulate SHAP
        plt.figure(figsize=(6,4))
        shap_values = np.random.rand(10, 5)
        features = [f"Feature {i}" for i in range(5)]
        shap.summary_plot(shap_values, plot_type="bar", feature_names=features, show=False)
        
        tab = self.tabview.tab("SHAP")
        for widget in tab.winfo_children():
            widget.destroy()
            
        figure = plt.gcf()
        canvas = FigureCanvasTkAgg(figure, master=tab)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        self.console.insert("end", "SHAP analysis completed\n")
    
    def run_lime(self):
        self.console.insert("end", "Running LIME analysis...\n")
        
        # Simulate LIME
        plt.figure(figsize=(6,4))
        features = [f"Feature {i}" for i in range(5)]
        values = np.random.rand(5)
        plt.barh(features, values)
        plt.title("LIME Feature Importance")
        
        tab = self.tabview.tab("LIME")
        for widget in tab.winfo_children():
            widget.destroy()
            
        figure = plt.gcf()
        canvas = FigureCanvasTkAgg(figure, master=tab)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        self.console.insert("end", "LIME analysis completed\n")
    
    def check_bias(self):
        self.console.insert("end", "Checking for bias...\n")
        
        # Simulate bias detection
        plt.figure(figsize=(6,4))
        groups = ["Group A", "Group B", "Group C"]
        values = [0.8, 0.4, 0.6]
        plt.bar(groups, values)
        plt.ylim(0, 1)
        plt.title("Fairness Analysis")
        
        tab = self.tabview.tab("Bias")
        for widget in tab.winfo_children():
            widget.destroy()
            
        figure = plt.gcf()
        canvas = FigureCanvasTkAgg(figure, master=tab)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        self.console.insert("end", "Bias detection completed\n")
    
    def run_whatif(self):
        values = [slider.get() for slider in self.feature_sliders]
        self.console.insert("end", f"Running what-if with values: {values}\n")
        
        # Update visualization
        self.ax.clear()
        x = np.linspace(0, 10, 100)
        y = np.sin(x * values[0]) * values[1] + values[2]
        self.ax.plot(x, y, color='red')
        self.ax.set_title("What-If Scenario")
        self.canvas.draw()

if __name__ == "__main__":
    root = ctk.CTk()
    app = NeuroSculptAIStudio(root)
    root.mainloop()