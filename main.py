import os
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from src.pipeline_data import get_train_test_data, create_modeling_pipeline
from src.evaluation_models import (
    evaluate_models_with_cv, 
    calculate_economic_threshold_and_costs, 
    explain_prediction_with_shap,
    plot_learning_curve
)
from src.llm_integration import generate_targeted_marketing_copy

def main():
    print("="*60)
    print("SISTEMA INTELIGENTE DE PROPENSITY MODELING - CARCDSI")
    print("="*60)
    
    data_path = 'data/ifood_df.csv'
    
    # 1. Carregamento e Particionamento Honesto (80/20)
    print("\n[1/5] Carregando e particionando os dados (80% Treino / 20% Teste Final)...")
    X_train_val, X_test, y_train_val, y_test, feature_names = get_train_test_data(data_path)
    print(f"-> Amostras de Treino/Validação: {len(X_train_val)} | Amostras de Teste no Cofre: {len(X_test)}")
    
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
    print("\n[3/5] Treinando o modelo final (Random Forest), gerando Curva de Aprendizagem e salvando artefato...")
    final_rf_pipeline = create_modeling_pipeline(
        RandomForestClassifier(n_estimators=100, max_depth=8, class_weight='balanced', random_state=42)
    )
    
    chart_path = plot_learning_curve(final_rf_pipeline, X_train_val, y_train_val)
    print(f"-> Gráfico da Curva de Aprendizagem salvo em: {chart_path}")
    
    final_rf_pipeline.fit(X_train_val, y_train_val)
    
    # 4. Cálculo do Limiar Econômico e Custo Operacional
    print("\n[4/5] Calculando Limiar Econômico com base na Matriz de Custos...")
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
        "X_train_background": X_train_val  # Referência para o explainer do SHAP
    }
    model_path = "models/propensity_rf_pipeline.joblib"
    joblib.dump(model_artifact, model_path)
    print(f"-> Artefato de produção (Pipeline + Limiar Econômico) salvo em: {model_path}")
    
    # 5. Explicabilidade SHAP e Geração de Campanha via LLM
    print("\n[5/5] Extraindo Explicabilidade SHAP para Cliente Propenso e Gerando Copy via LLM...")
    propensity_indices = [i for i, p in enumerate(y_test_proba) if p >= t_star]
    
    if propensity_indices:
        target_idx = propensity_indices[0]
        single_customer_df = X_test.iloc[[target_idx]]
        
        shap_evidences = explain_prediction_with_shap(final_rf_pipeline, X_train_val, single_customer_df, feature_names)
        
        print("\n--- Fatores de Influência (SHAP) para o Cliente Selecionado ---")
        print(shap_evidences.head(5).to_string(index=False))
        
        campaign_email = generate_targeted_marketing_copy(shap_evidences.head(5))
        print("\n=== E-MAIL PROMOCIONAL GERADO PELO LLM (COM BASE NO SHAP) ===")
        print(campaign_email)
    else:
        print("Nenhum cliente atingiu o limiar de propensão nesta amostra de teste.")

if __name__ == "__main__":
    main()