import random
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.formula.api import mixedlm

def generate_random_latin_square(n):
    """
    Erzeugt ein Lateinisches Quadrat (Größe n x n) mit Zufalls-Permutationen:
    Jede Zeile (Runde) und Spalte (Platz) enthält jeden Wert (Spielerindex) genau einmal.
    Anschließend werden die Symbole und Spalten zufällig permutiert.
    """
    # Standard-LQ über (i + j) mod n
    L = [[(i + j) % n for j in range(n)] for i in range(n)]
    
    # Spielerwerte (0..n-1) permutieren
    symbol_perm = list(range(n))
    random.shuffle(symbol_perm)
    for i in range(n):
        for j in range(n):
            L[i][j] = symbol_perm[L[i][j]]
    
    # Spalten (Plätze) permutieren
    col_perm = list(range(n))
    random.shuffle(col_perm)
    
    L_perm = []
    for i in range(n):
        row = [L[i][col] for col in col_perm]
        L_perm.append(row)
    
    return L_perm

def create_seating_plan(players):
    """
    Aus dem randomisierten Lateinischen Quadrat wird ein Sitzplan-DataFrame gebaut,
    in dem jeder Spieler pro Runde/Platz eindeutig zugeordnet ist.
    """
    n = len(players)
    latin_sq = generate_random_latin_square(n)
    
    plan_data = []
    for r in range(n):
        for seat in range(n):
            player_idx = latin_sq[r][seat]
            plan_data.append({
                "Runde": r+1,
                "Platz": seat+1,
                "Spieler": players[player_idx]
            })
    
    df = pd.DataFrame(plan_data)
    return df.sort_values(by=["Runde", "Platz"]).reset_index(drop=True)

def main():
    players = ["A", "B", "C", "D", "E", "F", "G"]
    n = len(players)
    
    # Gesamtdaten für alle gespielten Runden
    df_all = pd.DataFrame(columns=["Block", "Runde", "Spieler", "Platz", "Punkte"])
    
    block_count = 0
    while True:
        block_count += 1
        print(f"\n=== Starte Block {block_count} (je {n} Runden) ===")
        
        # Erzeuge Sitzplan für n Runden basierend auf Lateinischem Quadrat
        seating_plan = create_seating_plan(players)
        seating_plan["Block"] = block_count
        
        # Abarbeitung der Runden dieses Blocks
        for r in range(1, n+1):
            print(f"\n--- Runde {r} in Block {block_count} ---")
            
            # Zeige Sitzordnung für diese Runde
            round_plan = seating_plan[seating_plan["Runde"] == r]
            for idx, row in round_plan.iterrows():
                print(f"  Spieler {row['Spieler']} -> Platz {row['Platz']}")
            
            # Punkte abfragen
            punkte_dict = {}
            for idx, row in round_plan.iterrows():
                sp = row["Spieler"]
                pl = row["Platz"]
                eingabe = input(f"Punkte für Spieler {sp} (Platz {pl}): ")
                try:
                    punkte = float(eingabe)
                except ValueError:
                    punkte = 0.0
                punkte_dict[sp] = punkte
            
            # Daten in df_all anfügen
            for idx, row in round_plan.iterrows():
                sp = row["Spieler"]
                df_all = df_all.append({
                    "Block": block_count,
                    "Runde": r + (block_count - 1)*n,  # globale Rundennummer
                    "Spieler": sp,
                    "Platz": row["Platz"],
                    "Punkte": punkte_dict[sp]
                }, ignore_index=True)
            
            # Nach jeder Runde: Mixed-Effects-Korrektur (wenn >= 2 Runden insgesamt)
            if df_all["Runde"].nunique() >= 2:
                model = mixedlm("Punkte ~ C(Platz)", data=df_all, groups=df_all["Spieler"])
                result = model.fit()
                
                df_all["Punkte_vorhergesagt"] = result.fittedvalues
                df_all["Punkte_korr"] = df_all["Punkte"] - df_all["Punkte_vorhergesagt"]
            else:
                # Naive Korrektur, falls nur 1 Runde gespielt wurde
                df_all["Punkte_korr"] = df_all["Punkte"] - df_all.groupby("Platz")["Punkte"].transform("mean")
            
            # Zwischenstandsplot als Liniendiagramm mit kumulierten korrigierten Punkten
            df_agg = df_all.groupby(["Runde", "Spieler"], as_index=False)["Punkte_korr"].sum()
            df_agg["Punkte_korr_cum"] = df_agg.groupby("Spieler")["Punkte_korr"].cumsum()
            pivot_cum = df_agg.pivot(index="Runde", columns="Spieler", values="Punkte_korr_cum").fillna(0)
            
            plt.figure(figsize=(7, 4))
            pivot_cum.plot(marker="o", ax=plt.gca())
            plt.title(f"Kumulierte korrigierte Punkte nach globaler Runde {r + (block_count-1)*n}")
            plt.xlabel("Runde (global)")
            plt.ylabel("Korrigierte Punkte (kumuliert)")
            plt.legend(title="Spieler", bbox_to_anchor=(1.04, 1), loc="upper left")
            plt.tight_layout()
            plt.show()
        
        # Nach diesem Block noch einen Block spielen?
        weitermachen = input("Neuen Block spielen? (j/n): ").strip().lower()
        if weitermachen == "n":
            break
    
    # Abschließende Mixed-Effects-Analyse (wenn >= 2 Runden gespielt)
    if df_all["Runde"].nunique() >= 2:
        print("\n--- Endgültige Mixed-Effects-Analyse ---")
        final_model = mixedlm("Punkte ~ C(Platz)", data=df_all, groups=df_all["Spieler"]).fit()
        print(final_model.summary())
        
        df_all["Final_korr"] = df_all["Punkte"] - final_model.fittedvalues
        print("\nBeispielhafter Auszug:")
        print(df_all[["Block", "Runde", "Spieler", "Platz", "Punkte", "Final_korr"]].head(14))
    else:
        print("\nWeniger als 2 Runden gespielt – kein sinnvolles Mixed-Effects-Modell möglich.")
    
    print("\n--- Programm beendet ---")

if __name__ == "__main__":
    main()