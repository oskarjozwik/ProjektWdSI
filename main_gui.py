import customtkinter as ctk
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import json
from datetime import datetime

#konfiguracja interfejsu
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

USED_FEATURES = ['habitat', 'population', 'cap-shape', 'cap-color', 'odor', 'gill-size', 'gill-color', 'stalk-shape', 'ring-number']
MAPPING = {
    'habitat': {'name': 'Siedlisko', 'values': {'u': 'miasto', 'g': 'trawy', 'm': 'łąki', 'd': 'las', 'p': 'ścieżki', 'w': 'odpady', 'l': 'liście'}},
    'population': {'name': 'Populacja', 'values': {'s': 'rozproszona', 'n': 'liczna', 'a': 'gromadna', 'v': 'kilka', 'y': 'pojedynczo', 'c': 'skupiona'}},
    'cap-shape': {'name': 'Kształt kapelusza', 'values': {'x': 'wypukły', 'b': 'dzonkowaty', 's': 'płaski', 'f': 'lejkowaty', 'k': 'stożkowaty', 'c': 'wklęsły'}},
    'cap-color': {'name': 'Kolor kapelusza', 'values': {'n': 'brązowy', 'y': 'żółty', 'w': 'biały', 'g': 'szary', 'e': 'czerwony', 'p': 'różowy', 'b': 'beżowy', 'u': 'fioletowy', 'c': 'cynamonowy', 'r': 'zielony'}},
    'odor': {'name': 'Zapach', 'values': {'p': 'ostry', 'a': 'anyżowy', 'l': 'migdałowy', 'n': 'brak', 'f': 'cuchnący', 'c': 'kreozotowy', 'y': 'rybi', 's': 'korzenny', 'm': 'pleśniowy'}},
    'gill-size': {'name': 'Rozmiar blaszek', 'values': {'n': 'wąskie', 'b': 'szerokie'}},
    'gill-color': {'name': 'Kolor blaszek', 'values': {'k': 'czarny', 'n': 'brązowy', 'g': 'szary', 'p': 'różowy', 'w': 'biały', 'h': 'czekoladowy', 'u': 'fioletowy', 'e': 'czerwony', 'b': 'płowy', 'r': 'zielony', 'y': 'żółty', 'o': 'pomarańczowy'}},
    'stalk-shape': {'name': 'Kształt trzonu', 'values': {'e': 'rozszerzający się', 't': 'zwężający się'}},
    'ring-number': {'name': 'Liczba pierścieni', 'values': {'o': 'jeden', 't': 'dwa', 'n': 'brak'}}
}

class MycoGuardApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("MycoGuard AI - System Ekspercki")
        self.geometry("1300x950")

        # Modele
        self.model = joblib.load('mushroom_model.pkl')
        self.encoders = joblib.load('encoders.pkl')

        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # sidebar
        self.sidebar = ctk.CTkFrame(self, width=220, corner_radius=0)
        self.sidebar.grid(row=0, column=0, sticky="nsew")
        ctk.CTkLabel(self.sidebar, text="MycoGuard AI", font=("Arial", 24, "bold")).pack(pady=30)
        
        ctk.CTkButton(self.sidebar, text="Ważność Cech", command=self.show_importance).pack(pady=10, padx=20)
        ctk.CTkButton(self.sidebar, text="Metryki Modelu", command=self.show_metrics).pack(pady=10, padx=20)
        ctk.CTkButton(self.sidebar, text="POMOC I ATLAS", fg_color="#d35400", hover_color="#e67e22", command=self.open_help).pack(pady=10, padx=20)
        self.btn_log = ctk.CTkButton(self.sidebar, text="Logi Systemowe", fg_color="#34495e", command=self.show_logs)
        self.btn_log.pack(pady=10, padx=20)

        # Main
        self.main_area = ctk.CTkFrame(self, fg_color="transparent")
        self.main_area.grid(row=0, column=1, sticky="nsew", padx=20, pady=20)
        self.main_area.grid_columnconfigure((0,1), weight=1)
        self.main_area.grid_rowconfigure(0, weight=3); self.main_area.grid_rowconfigure(1, weight=2)

        # Wejscie
        self.input_card = ctk.CTkScrollableFrame(self.main_area, label_text="Dane z terenu")
        self.input_card.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        self.dropdowns = {}
        for feat in USED_FEATURES:
            ctk.CTkLabel(self.input_card, text=MAPPING[feat]['name'], font=("Arial", 12, "bold")).pack(pady=(10,0))
            pol_to_code = {v: k for k, v in MAPPING[feat]['values'].items()}
            combo = ctk.CTkComboBox(self.input_card, values=list(pol_to_code.keys()), width=280)
            combo.set(list(pol_to_code.keys())[0]); combo.pack()
            self.dropdowns[feat] = (combo, pol_to_code)

        # Wyjscie
        self.res_card = ctk.CTkFrame(self.main_area)
        self.res_card.grid(row=0, column=1, sticky="nsew", padx=10, pady=10)
        ctk.CTkButton(self.res_card, text="URUCHOM ANALIZĘ AI", height=50, fg_color="#27ae60", command=self.analyze).pack(pady=15, padx=40, fill="x")
        
        self.v_frame = ctk.CTkFrame(self.res_card, fg_color="#34495e", height=120); self.v_frame.pack(pady=10, padx=30, fill="x")
        self.v_frame.pack_propagate(False)
        self.v_text = ctk.CTkLabel(self.v_frame, text="GOTOWY", font=("Arial", 28, "bold")); self.v_text.pack(expand=True)
        
        self.conf_lab = ctk.CTkLabel(self.res_card, text="Pewność: 0%"); self.conf_lab.pack()
        self.conf_bar = ctk.CTkProgressBar(self.res_card, width=300); self.conf_bar.set(0); self.conf_bar.pack(pady=5)

        # Kluczowe czynniki
        self.path_label = ctk.CTkLabel(self.res_card, text="Kluczowe czynniki werdyktu:", font=("Arial", 13, "bold"))
        self.path_label.pack(pady=(20, 5))
        self.path_text = ctk.CTkLabel(self.res_card, text="Analiza nie została jeszcze przeprowadzona.", wraplength=350, text_color="gray")
        self.path_text.pack(pady=5, padx=20)

        # BOTTOM
        self.bottom_panel = ctk.CTkFrame(self.main_area, fg_color="#1a1a1a")
        self.bottom_panel.grid(row=1, column=0, columnspan=2, sticky="nsew", padx=10, pady=10)
        self.log_box = ctk.CTkTextbox(self.bottom_panel, font=("Consolas", 12))
        self.log_box.pack(fill="both", expand=True)
        self.current_view = None

    def open_help(self):
        """Otwiera system oceny jadalności grzybów."""
        help_win = ctk.CTkToplevel(self)
        help_win.title("MycoGuard - Identyfikacja Grzybów Jadalnych")
        help_win.geometry("900x800")
        help_win.attributes("-topmost", True)

        tabs = ctk.CTkTabview(help_win)
        tabs.pack(fill="both", expand=True, padx=10, pady=10)
        tabs.add("Słownik Cech"); tabs.add("Atlas Grzybów Jadalnych"); tabs.add("Atlas Grzybów Trujących")

        # Tab 1: Slownik cech
        scroll_dict = ctk.CTkScrollableFrame(tabs.tab("Słownik Cech"))
        scroll_dict.pack(fill="both", expand=True)
        
        dict_data = [
            ("Kształt kapelusza", "Może być wypukły (typowe dla młodych okazów), dzwonkowaty, płaski lub lejkowaty. Kształt zmienia się wraz z wiekiem grzyba."),
            ("Zapach (Kluczowy predyktor)", "Najważniejsza cecha w modelu AI. Zapachy anyżowe i migdałowe to domena smacznych pieczarek i lejkówek. Zapach mączny, rybi lub cuchnący to sygnał ostrzegawczy."),
            ("Blaszki (Rozmiar i Kolor)", "Gęstość i rozmiar blaszek pomagają odróżnić np. gołąbki od muchomorów. Kolor blaszek zmienia się u wielu gatunków po dotknięciu."),
            ("Trzon i Pierścień", "Pierścień (ring) to pozostałość osłony. Jego obecność lub brak oraz kształt (wiszący, nieruchomy) pozwala wykluczyć wiele pomyłek."),
            ("Siedlisko i Populacja", "Niektóre grzyby rosną tylko w grupach (populacja: liczna), inne zawsze pojedynczo. Las liściasty kontra iglasty to również istotny filtr dla modelu.")
        ]
        
        for k, v in dict_data.items() if isinstance(dict_data, dict) else dict_data:
            ctk.CTkLabel(scroll_dict, text=f"• {k}", font=("Arial", 15, "bold"), text_color="#3498db").pack(anchor="w", pady=(10,0))
            ctk.CTkLabel(scroll_dict, text=v, wraplength=800, justify="left").pack(anchor="w", padx=20)

        # Tab 3: Atlas Jadalne
        scroll_edible = ctk.CTkScrollableFrame(tabs.tab("Atlas Grzybów Jadalnych"))
        scroll_edible.pack(fill="both", expand=True)
        
        edibles = [
            ("Borowik Szlachetny (Prawdziwek)", "Król lasów. Pękaty, jasnobrązowy trzon z siateczką. Kapelusz gładki, brązowy. Brak blaszek (ma rurki)."),
            ("Pieprznik Jadalny (Kurka)", "Cały żółty lub pomarańczowy. Ma charakterystyczne listewki zamiast blaszek. Pachnie lekko morelowo."),
            ("Czubajka Kania", "Duży kapelusz z ruchomym pierścieniem na długim, smukłym trzonie. Często mylona z muchomorem sromotnikowym!"),
            ("Maślak Zwyczajny", "Kapelusz bardzo lepki, ciemnobrązowy. Trzon z pierścieniem. Rośnie głównie pod sosnami.")
        ]
        for name, desc in edibles:
            ctk.CTkLabel(scroll_edible, text=name, font=("Arial", 15, "bold"), text_color="#2ecc71").pack(anchor="w", pady=(15,0))
            ctk.CTkLabel(scroll_edible, text=desc, wraplength=800, justify="left").pack(anchor="w", padx=20)

        #Tab 3: Atlas Trujacych
        scroll_toxic = ctk.CTkScrollableFrame(tabs.tab("Atlas Grzybów Trujących"))
        scroll_toxic.pack(fill="both", expand=True)
        
        toxics = [
            ("Muchomor Sromotnikowy (Zielonawy)", "Najbardziej zabójczy. Kapelusz oliwkowozielony, białe blaszki, wyraźna pochwa u podstawy trzonu. Mylo-niebezpieczny z kanią!"),
            ("Muchomor Czerwony", "Charakterystyczny czerwony kapelusz z białymi kropkami. Silnie toksyczny i halucynogenny."),
            ("Goryczak Żółciowy (Szatan)", "Często mylony z borowikiem. Bardzo gorzki (psuje całe danie). Rurki pod kapeluszem różowieją z wiekiem."),
            ("Piestrzenica Kasztanowata", "Kapelusz przypominający mózg, brązowy. Zawiera gyromitrynę – śmiertelną truciznę usuwaną (częściowo) przez gotowanie, ale nadal odradzana."),
            ("Lisówka Pomarańczowa (Fałszywa Kurka)", "Mylona z kurką. Ma gęstsze blaszki i cieńszy trzon. Powoduje silne dolegliwości żołądkowe.")
        ]
        for name, desc in toxics:
            ctk.CTkLabel(scroll_toxic, text=name, font=("Arial", 15, "bold"), text_color="#e74c3c").pack(anchor="w", pady=(15,0))
            ctk.CTkLabel(scroll_toxic, text=desc, wraplength=800, justify="left").pack(anchor="w", padx=20)    

    def analyze(self):
        input_data = {}
        for feat in USED_FEATURES:
            combo, pol_to_code = self.dropdowns[feat]
            code = pol_to_code[combo.get()]
            input_data[feat] = [self.encoders[feat].transform([code])[0]]

        df_i = pd.DataFrame(input_data)[USED_FEATURES]
        probs = self.model.predict_proba(df_input := df_i)[0]
        max_idx = np.argmax(probs); conf = probs[max_idx]
        res = self.encoders['class'].inverse_transform([self.model.classes_[max_idx]])[0]

        # Ścieżka decyzji, 3 najwazniejsze cechy
        importances = self.model.feature_importances_
        top_idx = np.argsort(importances)[-3:][::-1]
        factors = [MAPPING[USED_FEATURES[i]]['name'] for i in top_idx]
        
        self.v_text.configure(text="TRUJĄCY 💀" if res == 'p' else "JADALNY 🍴")
        self.v_frame.configure(fg_color="#c0392b" if res == 'p' else "#27ae60")
        self.conf_bar.set(conf); self.conf_lab.configure(text=f"Pewność: {conf*100:.1f}%")
        self.path_text.configure(text=f"Decyzja oparta głównie na cechach: {', '.join(factors)}.", text_color="white")
        self.log_box.insert("end", f"[{datetime.now().strftime('%H:%M:%S')}] Analiza zakończona: {res}\n")

    def show_logs(self):
        if self.current_view: self.current_view.destroy()
        self.log_box.pack(fill="both", expand=True)

    def prepare_view(self):
        self.log_box.pack_forget()
        if self.current_view: self.current_view.destroy()
        self.current_view = ctk.CTkFrame(self.bottom_panel, fg_color="#1a1a1a")
        self.current_view.pack(fill="both", expand=True)

    def show_importance(self):
        self.prepare_view()
        importances = self.model.feature_importances_
        indices = np.argsort(importances)
        fig, ax = plt.subplots(figsize=(10, 4), facecolor='#1a1a1a')
        ax.set_facecolor('#1a1a1a')
        ax.barh(range(len(indices)), importances[indices], color='#3498db')
        ax.set_yticks(range(len(indices)))
        ax.set_yticklabels([MAPPING[USED_FEATURES[i]]['name'] for i in indices], color='white')
        plt.tight_layout(); self.embed_plot(fig)

    def show_metrics(self):
        self.prepare_view()
        try:
            with open('model_stats.json', 'r') as f: stats = json.load(f)
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4), facecolor='#1a1a1a')
            cm = np.array(stats['confusion_matrix'])
            ax1.imshow(cm, cmap=plt.cm.Blues)
            for i in range(2):
                for j in range(2):
                    ax1.text(j, i, str(cm[i, j]), ha="center", va="center", color="orange", fontsize=16)
            ax1.set_xticks([0,1]); ax1.set_xticklabels(['Jadalny', 'Trujący'], color='white')
            ax1.set_yticks([0,1]); ax1.set_yticklabels(['Jadalny', 'Trujący'], color='white')
            ax2.pie(stats['data_dist'].values(), labels=['Trujące', 'Jadalne'], autopct='%1.1f%%', colors=['#e74c3c', '#2ecc71'], textprops={'color':"w"})
            plt.tight_layout(); self.embed_plot(fig)
        except: self.log_box.insert("end", "Brak pliku statystyk. Uruchom train_model.py!\n")

    def embed_plot(self, fig):
        canvas = FigureCanvasTkAgg(fig, master=self.current_view)
        canvas.draw(); canvas.get_tk_widget().pack(fill="both", expand=True)

if __name__ == "__main__":
    app = MycoGuardApp(); app.mainloop()