import os
import numpy as np
import pandas as pd


def generate_realistic_synthetic_data(df_source, n_samples=1500, balance_target=True, noise_scale=0.05, random_state=42):
    """
    Lê uma base de referência (como o conjunto de treino do ifood_df.csv) e gera
    registros sintéticos estatisticamente fiéis à realidade do negócio.

    Características técnicas:
    - Preserva correlações multivariadas via reamostragem estratificada com perturbação gaussiana suave.
    - Mantém coerência proporcional entre variação de renda e gastos por categoria.
    - Garante exclusividade mútua em colunas One-Hot (Estado Civil e Escolaridade).
    - Recalcula colunas derivadas (MntTotal, MntRegularProds e AcceptedCmpOverall).
    """
    rng = np.random.default_rng(random_state)

    # 1. Amostragem base (estratificada para equilibrar Response=1 ou proporcional à base)
    if balance_target and "Response" in df_source.columns:
        n_pos = n_samples // 2
        n_neg = n_samples - n_pos

        df_pos = df_source[df_source["Response"] == 1]
        df_neg = df_source[df_source["Response"] == 0]

        idx_pos = rng.choice(df_pos.index, size=n_pos, replace=True)
        idx_neg = rng.choice(df_neg.index, size=n_neg, replace=True)

        synth_df = pd.concat([df_pos.loc[idx_pos], df_neg.loc[idx_neg]], ignore_index=True)
        synth_df = synth_df.sample(frac=1.0, random_state=random_state).reset_index(drop=True)
    else:
        idx = rng.choice(df_source.index, size=n_samples, replace=True)
        synth_df = df_source.loc[idx].reset_index(drop=True)

    # 2. Perturbação realista de Renda e Gastos (mantendo proporção econômica)
    if "Income" in synth_df.columns:
        income_std = df_source["Income"].std()
        income_shock = rng.normal(0, income_std * noise_scale, size=n_samples)
        old_income = synth_df["Income"].replace(0, 1)
        synth_df["Income"] = np.clip(
            synth_df["Income"] + income_shock,
            df_source["Income"].min(),
            df_source["Income"].max()
        ).round(2)
        income_ratio = synth_df["Income"] / old_income
    else:
        income_ratio = np.ones(n_samples)

    product_cols = [
        "MntWines", "MntFruits", "MntMeatProducts",
        "MntFishProducts", "MntSweetProducts", "MntGoldProds"
    ]
    for col in product_cols:
        if col in synth_df.columns:
            col_std = df_source[col].std()
            noise = rng.normal(0, col_std * noise_scale, size=n_samples)
            adjusted = (synth_df[col] * income_ratio) + noise
            synth_df[col] = np.clip(
                np.round(adjusted),
                df_source[col].min(),
                df_source[col].max()
            ).astype(int)

    # 3. Perturbação nas variáveis comportamentais e temporais
    behavior_cols = [
        "Recency", "Age", "Customer_Days",
        "NumDealsPurchases", "NumWebPurchases", "NumCatalogPurchases",
        "NumStorePurchases", "NumWebVisitsMonth"
    ]
    for col in behavior_cols:
        if col in synth_df.columns:
            col_std = df_source[col].std()
            noise = rng.normal(0, col_std * noise_scale, size=n_samples)
            synth_df[col] = np.clip(
                np.round(synth_df[col] + noise),
                df_source[col].min(),
                df_source[col].max()
            ).astype(int)

    # 4. Garante integridade das variáveis categóricas One-Hot (Estado Civil e Escolaridade)
    marital_cols = [c for c in synth_df.columns if c.startswith("marital_")]
    if marital_cols:
        # Garante que exatamente uma coluna marital_ seja 1 por linha
        probs_marital = df_source[marital_cols].mean().values
        probs_marital = probs_marital / probs_marital.sum()
        swap_mask = rng.random(n_samples) < 0.03  # 3% de variação demográfica
        if swap_mask.any():
            chosen = rng.choice(len(marital_cols), size=swap_mask.sum(), p=probs_marital)
            synth_df.loc[swap_mask, marital_cols] = 0
            for row_idx, col_idx in zip(synth_df.index[swap_mask], chosen):
                synth_df.iat[row_idx, synth_df.columns.get_loc(marital_cols[col_idx])] = 1

    education_cols = [c for c in synth_df.columns if c.startswith("education_")]
    if education_cols:
        probs_edu = df_source[education_cols].mean().values
        probs_edu = probs_edu / probs_edu.sum()
        swap_mask = rng.random(n_samples) < 0.03
        if swap_mask.any():
            chosen = rng.choice(len(education_cols), size=swap_mask.sum(), p=probs_edu)
            synth_df.loc[swap_mask, education_cols] = 0
            for row_idx, col_idx in zip(synth_df.index[swap_mask], chosen):
                synth_df.iat[row_idx, synth_df.columns.get_loc(education_cols[col_idx])] = 1

    # 5. Recálculo exato das colunas derivadas e constantes institucionais
    if all(c in synth_df.columns for c in product_cols):
        synth_df["MntTotal"] = synth_df[product_cols].sum(axis=1)
        synth_df["MntRegularProds"] = synth_df["MntTotal"] - synth_df["MntGoldProds"]

    cmp_cols = ["AcceptedCmp1", "AcceptedCmp2", "AcceptedCmp3", "AcceptedCmp4", "AcceptedCmp5"]
    if all(c in synth_df.columns for c in cmp_cols):
        synth_df["AcceptedCmpOverall"] = synth_df[cmp_cols].sum(axis=1)

    for const_col in ["Z_CostContact", "Z_Revenue"]:
        if const_col in df_source.columns:
            synth_df[const_col] = df_source[const_col].iloc[0]

    return synth_df


if __name__ == "__main__":
    # Execução independente para leitura do ifood_df.csv e inspeção da base gerada
    input_path = os.path.join("data", "ifood_df.csv")
    output_path = os.path.join("data", "ifood_synthetic_preview.csv")

    df_original = pd.read_csv(input_path)
    df_sintetico = generate_realistic_synthetic_data(df_original, n_samples=1500, balance_target=True)
    df_sintetico.to_csv(output_path, index=False)

    print("=" * 60)
    print("TARGET AI | GERADOR DE DADOS SINTÉTICOS REALISTAS")
    print("=" * 60)
    print(f"-> Dataset original lido: {input_path} ({len(df_original)} registros)")
    print(f"-> Base sintética salva:  {output_path} ({len(df_sintetico)} registros)")
    print("\nComparação de Médias (Real vs. Sintético):")
    cols_check = ["Income", "Recency", "MntWines", "MntMeatProducts", "MntTotal", "Response"]
    comp = pd.DataFrame({
        "Real_Mean": df_original[cols_check].mean().round(2),
        "Synth_Mean": df_sintetico[cols_check].mean().round(2)
    })
    print(comp)