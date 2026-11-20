# 🎯 TargetAI — Motor Inteligente de Propensity Modeling e Campanhas Contextuais

> **A oferta certa, no momento exato, para quem realmente vai comprar.**

O **TargetAI** é um motor inteligente de recomendação e disparo de campanhas (*Propensity Modeling*) desenvolvido de ponta a ponta para o setor de marketing digital, delivery e varejo. O sistema resolve o problema do envio genérico e indiscriminado de promoções — que gera desgaste da marca (*spam*) e desperdício de verba — ao cruzar dados demográficos e o histórico de consumo de cada cliente (renda, recência, frequência e gastos por categoria) para prever sua real probabilidade de adesão.

Para transformar dados brutos em conversão imediata, a plataforma combina **Engenharia de Dados Sintéticos**, **Machine Learning Supervisionado**, **Otimização Financeira (Limiar Econômico)**, **IA Explicável (SHAP)** e **IA Generativa Contextual (Google Gemini)**. O grande diferencial da ferramenta está na sua **inteligência temporal dinâmica**: antes de redigir a oferta, o sistema identifica automaticamente o dia da semana, o momento do mês e o horário exato da execução para alinhar o cupom à **refeição ou ocasião de consumo mais próxima** (como Almoço, Café da Tarde, Jantar, Sexta-feira à Noite ou economia de Fim de Mês), entregando mensagens curtas, bem-humoradas e com gatilhos de urgência altamente persuasivos.

---

## 🧠 Como o Sistema Funciona (Arquitetura)

O fluxo operacional do **TargetAI** é dividido em 5 pilares integrados:

1. **Engenharia e Aumento de Dados Sintéticos (`src/synthetic_data.py`):**
   - Combate o desbalanceamento natural das campanhas de marketing (onde apenas ~15% dos clientes respondem positivamente) gerando registros sintéticos estatisticamente fiéis à base original (`ifood_df.csv`).
   - Utiliza **reamostragem estratificada com perturbação gaussiana controlada** usando apenas `numpy` e `pandas`, preservando correlações econômicas reais (proporcionalidade entre renda e gastos por categoria), garantindo exclusividade mútua em variáveis *One-Hot* (estado civil e escolaridade) e recalculando colunas derivadas (`MntTotal`, `MntRegularProds` e `AcceptedCmpOverall`).
   - **Zero Data Leakage:** Os dados sintéticos são injetados **exclusivamente na fatia de treino (80%)**, mantendo os **20% de teste (`X_test`) 100% reais e intocados** para uma auditoria honesta.

2. **Previsão de Propensão (`src/pipeline_data.py` & `src/evaluation_models.py`):**
   - Encapsula `StandardScaler` e modelos classificadores em `Pipelines` do Scikit-Learn, avaliados via **RepeatedStratifiedKFold** e comparados estatisticamente pelo **Teste de Wilcoxon Pareado**.
   - Diagnostica a maturidade do aprendizado e a ausência de *overfitting* através da **Curva de Aprendizagem** (`reports/curva_aprendizagem_rf.png`).

3. **Decisão Financeira Orientada a Custos:**
   - Em vez de usar o corte padrão de 50%, calcula o **Limiar Econômico Ótimo ($t^*$)** ponderando o custo operacional de abordar um cliente não propenso (**Falso Positivo**) frente à perda de receita por ignorar um comprador real (**Falso Negativo**).

4. **Explicabilidade Matemática (XAI com SHAP):**
   - Seleciona dinamicamente clientes aprovados pelo limiar econômico e extrai via `TreeExplainer` quais atributos comportamentais e demográficos (ex.: recência de compra, gasto em vinhos, carnes, frutas ou compras via web/catálogo) impulsionaram a probabilidade de conversão daquele perfil específico.

5. **Geração Dinâmica de Ofertas Contextuais (`src/llm_integration.py`):**
   - **Sensor Temporal em Tempo Real:** Detecta automaticamente o dia da semana, dia do mês e hora exata do disparo para selecionar o gatilho contextual ideal (Almoço, Café da Tarde, Jantar, Sexta-feira à Noite ou Fim do Mês).
   - **Copywriting Hiperpersonalizado:** Cruza o contexto da refeição/horário com os fatores reais do SHAP e aciona o **Google Gemini** (com *exponential backoff* e *fallback* de modelos) para redigir uma campanha única, destacando o cupom logo no primeiro parágrafo e fechando com um *Call-to-Action* direto.

---

## 📁 Estrutura do Repositório

```text
target-ai/
├── data/
│   ├── ifood_df.csv                      # Base histórica original de comportamento de clientes
│   └── ifood_synthetic_preview.csv       # Amostra gerada para auditoria estatística (opcional)
├── models/
│   └── propensity_rf_pipeline.joblib     # Artefato persistido (Pipeline + Limiar Econômico)
├── reports/
│   └── curva_aprendizagem_rf.png         # Diagnóstico visual de generalização do modelo
├── src/
│   ├── __init__.py
│   ├── pipeline_data.py                  # Particionamento (80/20) e construção de Pipelines
│   ├── synthetic_data.py                 # Gerador de dados sintéticos realistas (NumPy/Pandas)
│   ├── evaluation_models.py              # Validação cruzada, Wilcoxon, Custos, SHAP e Gráficos
│   └── llm_integration.py                # Integração resiliente com Gemini e gatilhos temporais
├── main.py                               # Pipeline completo de treino, aumento de dados e persistência
├── predict_cliente.py                    # Inferência rápida em produção para novos clientes
├── .env                                  # Chave de API (não versionado no Git)
├── .gitignore
└── requirements.txt