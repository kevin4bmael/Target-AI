import os
import joblib
import pandas as pd
from src.evaluation_models import explain_prediction_with_shap
from src.llm_integration import generate_targeted_marketing_copy

def predict_new_customer(customer_df):
    """
    Carrega o modelo persistido em disco e realiza a inferência em tempo real
    para um novo cliente, aplicando o limiar econômico e gerando copy via LLM.
    """
    model_path = "models/propensity_rf_pipeline.joblib"
    
    if not os.path.exists(model_path):
        print(f"[ERRO] Modelo não encontrado em '{model_path}'. Execute 'python main.py' primeiro para treinar e salvar.")
        return

    print("="*60)
    print("TARGET AI | MODO DE INFERÊNCIA E DISPARO CONTEXTUAL")
    print("="*60)

    # 1. Carregamento instantâneo do pipeline treinado
    artifact = joblib.load(model_path)
    pipeline = artifact["pipeline"]
    t_star = artifact["economic_threshold"]
    feature_names = artifact["feature_names"]
    X_train_bg = artifact["X_train_background"]

    # 2. Predição de probabilidade (sem retreinar!)
    probabilidade = pipeline.predict_proba(customer_df)[0, 1]

    print(f"\n-> Probabilidade de Adesão Estimada: {probabilidade:.2%}")
    print(f"-> Limiar Econômico Exigido (t*):    {t_star:.2%}")

    # 3. Tomada de decisão financeira
    if probabilidade >= t_star:
        print("-> DECISÃO FINANCEIRA: [APROVADO] Alta propensão e retorno positivo esperado.")
        
        # Extrai explicabilidade e gera e-mail personalizado
        shap_evidences = explain_prediction_with_shap(pipeline, X_train_bg, customer_df, feature_names)
        
        print("\n--- Principais Fatores de Decisão (SHAP) ---")
        print(shap_evidences.head(5).to_string(index=False))
        
        print("\nGerando oferta personalizada com IA Generativa...")
        email_copy = generate_targeted_marketing_copy(shap_evidences.head(5))
        print("\n=== E-MAIL PRONTO PARA DISPARO ===")
        print(email_copy)
    else:
        print("-> DECISÃO FINANCEIRA: [RECUSADO] Baixa tendência de adesão. Campanha não enviada para evitar desperdício de verba.")

if __name__ == "__main__":
    # Simulação de entrada de um novo perfil de cliente (exemplo extraído da base)
    df_exemplo = pd.read_csv("data/ifood_df.csv").drop(columns=["Response"])
    
    # Você pode alterar qualquer índice (ex: iloc[[0]], iloc[[15]], iloc[[42]]) para testar perfis diferentes
    novo_cliente = df_exemplo.iloc[[15]].copy()
    
    predict_new_customer(novo_cliente)