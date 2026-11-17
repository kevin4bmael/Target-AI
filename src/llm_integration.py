import os
import time
from datetime import datetime
from google import genai
from dotenv import load_dotenv

load_dotenv()

def get_current_context_trigger():
    """
    Detecta automaticamente a data, dia da semana e horário atual
    para definir o gatilho contextual mais assertivo para a campanha.
    """
    agora = datetime.now()
    hora = agora.hour
    dia_semana = agora.weekday()  # 0 = Segunda, ..., 4 = Sexta, 5 = Sábado, 6 = Domingo
    dia_mes = agora.day

    dias_nomes = ["Segunda-feira", "Terça-feira", "Quarta-feira", "Quinta-feira", "Sexta-feira", "Sábado", "Domingo"]
    nome_dia = dias_nomes[dia_semana]
    horario_str = agora.strftime("%H:%M")

    # 1. Verifica gatilhos especiais de calendário primeiro
    if dia_mes >= 25:
        contexto = "Fim do Mês (foco em economia inteligente, custo-benefício e cupom para salvar o orçamento)"
    elif dia_semana == 4 and hora >= 18:
        contexto = "Sexta-feira à Noite ('Sextou!', clima de celebração, descanso e recompensa pela semana)"
    # 2. Verifica a janela de horário do dia
    elif 11 <= hora < 15:
        contexto = "Horário de Almoço (foco em rapidez, praticidade e matar a fome agora)"
    elif 15 <= hora < 18:
        contexto = "Café da Tarde (foco em pausa merecida, lanche rápido ou doce)"
    elif 18 <= hora <= 23:
        contexto = "Horário de Jantar (foco em conforto, praticidade para não precisar cozinhar hoje)"
    else:
        contexto = "Manhã / Madrugada (foco em garantir desconto antecipado para o almoço de hoje)"

    return f"{nome_dia}, {horario_str} (Dia {dia_mes}) -> Gatilho Ativo: {contexto}"

def generate_targeted_marketing_copy(shap_summary_df):
    """
    Usa o LLM (Gemini) integrando os fatos matemáticos do SHAP e o contexto
    temporal em tempo real (data/hora) para gerar uma copy hiperpersonalizada.
    """
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return "[AVISO] GEMINI_API_KEY não configurada no arquivo .env."
        
    client = genai.Client(api_key=api_key)
    
    # Captura o momento exato da execução
    contexto_temporal = get_current_context_trigger()
    print(f"-> Contexto Temporal Detectado: {contexto_temporal}")
    
    prompt = f"""
    Atue como um Copywriter Sênior especialista em Growth Marketing e psicologia do consumidor para delivery e varejo (estilo iFood).
    Nosso modelo preditivo de Machine Learning identificou um cliente com altíssima propensão de compra neste exato momento.
    
    CONTEXTO TEMPORAL ATUAL (MOMENTO DO DISPARO):
    {contexto_temporal}
    
    FATORES COMPORTAMENTAIS DO CLIENTE (EXTRAÍDOS VIA SHAP):
    {shap_summary_df.to_string(index=False)}
    
    TAREFA:
    Crie UMA ÚNICA campanha curta (e-mail/push) totalmente adaptada ao CONTEXTO TEMPORAL ATUAL acima e alinhada aos produtos/hábitos de maior impacto positivo revelados pelo SHAP (ex: carnes, frutas, vinhos, compra recente, etc.).

    REGRAS OBRIGATÓRIAS:
    - Tom de voz: Curto, leve, envolvente e bem-humorado.
    - Regra do 1º Parágrafo: Evidencie o CUPOM DE DESCONTO e o benefício financeiro logo na primeira frase/parágrafo liberado uso contido de emojis para reforçar a mensagem.
    - Gatilhos Mentais: Aplique explicitamente escassez, urgência ligada ao horário atual (ex: "válido só para este almoço", "expira em 2 horas") e recompensa imediata.
    - Personalização: Conecte sutilmente a oferta aos itens favoritos do cliente apontados pelo SHAP (sem citar termos técnicos como "SHAP", "variável" ou "modelo").
    - Tamanho: Máximo de 3 parágrafos curtos e diretos.
    - Fechamento: Termine com um Call-to-Action (CTA) curto, direto e magnético (máximo 4 palavras, ex: "Resgatar Cupom Agora", "Pedir com Desconto").
    """
    
    candidate_models = [
        'gemini-3.8-flash',
        'gemini-3.8-flash-lite'
    ]
    
    for model_name in candidate_models:
        for tentativa in range(1, 4):
            try:
                response = client.models.generate_content(
                    model=model_name,
                    contents=prompt,
                )
                return f"[Gerado via {model_name} | {contexto_temporal}]\n\n{response.text}"
            except Exception as e:
                erro_str = str(e)
                if "503" in erro_str or "UNAVAILABLE" in erro_str:
                    espera = tentativa * 3
                    print(f"-> {model_name} ocupado (503). Tentativa {tentativa}/3: aguardando {espera}s...")
                    time.sleep(espera)
                else:
                    print(f"-> Aviso ao usar {model_name}: {e}")
                    break

    return "[AVISO] Os servidores do Gemini estão temporariamente congestionados. Tente rodar novamente em alguns minutos."