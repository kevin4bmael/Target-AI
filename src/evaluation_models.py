import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import (
    RepeatedStratifiedKFold, 
    cross_validate, 
    learning_curve, 
    StratifiedKFold
)
from sklearn.metrics import (
    precision_score, recall_score, f1_score, 
    roc_auc_score, average_precision_score, brier_score_loss, confusion_matrix
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from scipy.stats import wilcoxon
import shap
from src.pipeline_data import create_modeling_pipeline

def evaluate_models_with_cv(X, y):
    """
    Avalia a Regressão Logística (Baseline) e a Random Forest (Avançado)
    utilizando RepeatedStratifiedKFold e métricas de desempenho rigorosas.
    """
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=42)
    
    models = {
        "Regressão Logística (Baseline)": LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42),
        "Random Forest (Avançado)": RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
    }
    
    cv_detailed_scores = {}
    summary_results = {}
    
    scoring_metrics = ['roc_auc', 'average_precision', 'neg_brier_score']
    
    for name, model in models.items():
        pipeline = create_modeling_pipeline(model)
        
        scores = cross_validate(pipeline, X, y, cv=cv, scoring=scoring_metrics)
        
        cv_detailed_scores[name] = scores['test_roc_auc']
        brier_scores = -scores['test_neg_brier_score']
        
        summary_results[name] = {
            "ROC-AUC Médio": f"{scores['test_roc_auc'].mean():.3f} (±{scores['test_roc_auc'].std():.3f})",
            "PR-AUC (Avg Precision) Médio": f"{scores['test_average_precision'].mean():.3f} (±{scores['test_average_precision'].std():.3f})",
            "Brier Score Médio": f"{brier_scores.mean():.3f} (±{brier_scores.std():.3f})"
        }
        
    stat, p_value = wilcoxon(
        cv_detailed_scores["Random Forest (Avançado)"], 
        cv_detailed_scores["Regressão Logística (Baseline)"]
    )
    
    return summary_results, cv_detailed_scores, p_value

def calculate_economic_threshold_and_costs(y_true, y_proba, c_fp=1800, c_fn=7500):
    """
    Calcula o Limiar Econômico ótimo com base na matriz de custos financeiros
    (Custo de Falso Positivo vs Falso Negativo) e retorna a matriz de confusão econômica.
    """
    t_star = c_fp / (c_fp + c_fn)
    
    y_pred_econ = (y_proba >= t_star).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred_econ).ravel()
    
    expected_cost = (fp * c_fp) + (fn * c_fn)
    
    return t_star, expected_cost, (tn, fp, fn, tp)

def explain_prediction_with_shap(best_pipeline, X_train, X_sample, feature_names):
    """
    Calcula os valores SHAP para explicar a predição de um cliente propenso
    utilizando o modelo treinado.
    """
    scaler = best_pipeline.named_steps['scaler']
    classifier = best_pipeline.named_steps['classifier']
    
    X_train_scaled = scaler.transform(X_train)
    X_sample_scaled = scaler.transform(X_sample)
    
    explainer = shap.TreeExplainer(classifier, X_train_scaled)
    shap_values = explainer(X_sample_scaled)
    
    contributions = pd.DataFrame({
        'Feature': feature_names,
        'Valor_Cliente': X_sample.values[0],
        'SHAP_Value': shap_values.values[0] if len(shap_values.values.shape) == 2 else shap_values.values[0, :, 1]
    }).sort_values(by='SHAP_Value', key=abs, ascending=False)
    
    return contributions

def plot_learning_curve(pipeline, X, y, output_dir="reports", filename="curva_aprendizagem_rf.png"):
    """
    Gera e salva o gráfico da Curva de Aprendizagem (Treino vs. Validação)
    para diagnosticar overfitting, underfitting e maturidade do modelo.
    """
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, filename)
    
    # Remove o arquivo anterior se existir para evitar bloqueio de escrita no Windows
    if os.path.exists(output_path):
        try:
            os.remove(output_path)
        except PermissionError:
            pass
            
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    train_sizes = np.linspace(0.1, 1.0, 8)
    
    train_sizes_abs, train_scores, val_scores = learning_curve(
        pipeline,
        X,
        y,
        cv=cv,
        scoring="roc_auc",
        train_sizes=train_sizes,
        n_jobs=-1
    )
    
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    val_mean = np.mean(val_scores, axis=1)
    val_std = np.std(val_scores, axis=1)
    
    plt.figure(figsize=(9, 5.5))
    plt.title("Curva de Aprendizagem - Random Forest (ROC-AUC)", fontsize=13, fontweight="bold")
    plt.xlabel("Quantidade de Amostras de Treinamento", fontsize=11)
    plt.ylabel("Score ROC-AUC", fontsize=11)
    plt.grid(True, linestyle="--", alpha=0.6)
    
    plt.plot(train_sizes_abs, train_mean, "o-", color="#1f77b4", label="Treino (Score Empírico)")
    plt.fill_between(train_sizes_abs, train_mean - train_std, train_mean + train_std, alpha=0.15, color="#1f77b4")
    
    plt.plot(train_sizes_abs, val_mean, "s-", color="#2ca02c", label="Validação Cruzada (Generalização)")
    plt.fill_between(train_sizes_abs, val_mean - val_std, val_mean + val_std, alpha=0.15, color="#2ca02c")
    
    plt.ylim(0.5, 1.05)
    plt.legend(loc="lower right", fontsize=10)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    
    return output_path