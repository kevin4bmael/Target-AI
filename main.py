import os
import random
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from src.pipeline_data import get_train_test_data, create_modeling_pipeline
from src.synthetic_data import generate_realistic_synthetic_data
from src.evaluation_models import (
    evaluate_models_with_cv, 
    calculate_economic_threshold_and_costs, 
    explain_prediction_with_shap,
    plot_learning_curve
)
from src.llm_integration import generate_targeted_marketing_copy


def main():
    print("=" * 60)
    print("TARGET AI | MOTOR DE PROPENSÃO E CAMPANHAS INTELIGENTES")
    print("=" * 60)
    
    data_path = os.path.join("data", "ifood_df.csv")
    
    # 1. Carregamento, Particionamento Honesto (80/20) e Geração de Dados Sintéticos no Treino
    print("\n[1/5] Lendo dataset original (80/20) e gerando base sintética realista para o treino...")
    X_train_real, X_test, y_train_real, y_test, feature_names = get_train_test_data(data_path)
    
    # Reconstrói apenas a fatia de treino para gerar sintéticos sem vazamento de dados (Data Leakage)
    train_real_df = X_train_real.copy()
    train_real_df["Response"] = y_train_real.values
    
    synth_df = generate_realistic_synthetic_data(
        df_source=train_real_df,
        n_samples=1500,
        balance_target=True,
        noise_scale=0.05,
        random_state=42
    )
    
    # Une os registros reais de treino aos registros sintéticos gerados
    train_augmented_df = pd.concat([train_real_df, synth_df], ignore_index=True)
    X_train_val = train_augmented_df[list(feature_names)]
    y_train_val = train_augmented_df["Response"]
    
    print(f"-> Treino Real: {len(X_train_real)} | Sintéticos Gerados: {len(synth_df)} | Total Treino: {len(X_train_val)}")
    print(f"-> Amostras de Teste Real no Cofre (100% intocadas): {len(X_test)}")
    
    # 2. Validação Cruzada Rigorosa com RepeatedStratifiedKFold
    print("\n[2/5] Executando Validação Cruzada (RepeatedStratifiedKFold) com Pipelines...")
    summary_results, cv_scores, p_value_wilcoxon = evaluate_models_with_cv(X_train_val, y_train_val)
    
    print("\n--- Resultados Comparativos (Validação Cruzada) ---")
    for model_name, metrics in summary_results.items():
        print(f"\nModelo: {model_name}")
        for metric, val in metrics.items():
            print(f"  {metric}: {val}")
            
    print(f"\n[Teste Estatístico] P-valor do Teste de Wilcoxon Pareado: {p_value_wilcoxon:.5f}")
    if p_value_wilcoxon < 0.05:
        print("-> Conclusão: A diferença de desempenho entre os modelos é estatisticamente significativa ao nível de 5%.")
    else:
        print("-> Conclusão: A diferença de desempenho não é estatisticamente significativa ao nível de 5%.")
        
    # 3. Treinamento Final do Melhor Modelo (Random Forest), Curva de Aprendizagem e Persistência
    # Nota: Sem class_weight='balanced' no modelo final pois a base sintética já equilibrou as classes
    print("\n[3/5] Treinando o modelo final (Random Forest), gerando Curva de Aprendizagem e salvando artefato...")
    final_rf_pipeline = create_modeling_pipeline(
        RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42)
    )
    
    chart_path = plot_learning_curve(final_rf_pipeline, X_train_val, y_train_val)
    print(f"-> Gráfico da Curva de Aprendizagem salvo em: {chart_path}")
    
    final_rf_pipeline.fit(X_train_val, y_train_val)
    
    # 4. Cálculo do Limiar Econômico e Custo Operacional (avaliado no Teste 100% Real)
    print("\n[4/5] Calculando Limiar Econômico com base na Matriz de Custos (Teste Real)...")
    y_test_proba = final_rf_pipeline.predict_proba(X_test)[:, 1]
    t_star, expected_cost, conf_matrix = calculate_economic_threshold_and_costs(y_test, y_test_proba, c_fp=1800, c_fn=7500)
    print(f"-> Limiar Econômico Calculado (t*): {t_star:.3f}")
    print(f"-> Custo Operacional Total Esperado no Teste: ${expected_cost:,.2f}")
    print(f"-> Matriz de Confusão (TN, FP, FN, TP): {conf_matrix}")
    
    # Persistência do Modelo Treinado e Parâmetros de Produção
    os.makedirs("models", exist_ok=True)
    model_artifact = {
        "pipeline": final_rf_pipeline,
        "economic_threshold": t_star,
        "feature_names": list(feature_names),
        "X_train_background": X_train_val
    }
    model_path = os.path.join("models", "propensity_rf_pipeline.joblib")
    joblib.dump(model_artifact, model_path)
    print(f"-> Artefato de produção (Pipeline + Limiar Econômico) salvo em: {model_path}")
    
    # 5. Explicabilidade SHAP e Geração de Campanha Contextual via LLM
    print("\n[5/5] Extraindo Explicabilidade SHAP para Cliente Propenso e Gerando Copy Contextual via LLM...")
    propensity_indices = [i for i, p in enumerate(y_test_proba) if p >= t_star]
    
    if propensity_indices:
        # Sorteia dinamicamente um cliente propenso diferente a cada execução
        target_idx = random.choice(propensity_indices)
        single_customer_df = X_test.iloc[[target_idx]]
        
        print(f"-> Cliente selecionado do conjunto de teste (índice {target_idx}) | Propensão: {y_test_proba[target_idx]:.2%}")
        shap_evidences = explain_prediction_with_shap(final_rf_pipeline, X_train_val, single_customer_df, feature_names)
        
        print("\n--- Fatores de Influência (SHAP) para o Cliente Selecionado ---")
        print(shap_evidences.head(5).to_string(index=False))
        
        campaign_email = generate_targeted_marketing_copy(shap_evidences.head(5))
        print("\n=== E-MAIL PROMOCIONAL GERADO PELO LLM (COM BASE NO SHAP E HORÁRIO) ===")
        print(campaign_email)
    else:
        print("Nenhum cliente atingiu o limiar de propensão nesta amostra de teste.")


if __name__ == "__main__":
    main()