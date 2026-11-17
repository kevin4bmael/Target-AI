import os
import time
from google import genai
from dotenv import load_dotenv

load_dotenv()

def generate_targeted_marketing_copy(shap_summary_df):
    """
    Usa o LLM (Gemini) integrando os fatos matemáticos do SHAP
    para gerar copies curtas, bem-humoradas e orientadas a gatilhos mentais
    para diferentes momentos de consumo e contextos.
    """
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return "[AVISO] GEMINI_API_KEY não configurada no arquivo .env."
        
    client = genai.Client(api_key=api_key)
    
    prompt = f"""
    Atue como um cientista de dados sênior e estrategista de marketing digital de varejo.
    O nosso modelo de Machine Learning (Random Forest) identificou que um cliente específico possui altíssima propensão de compra.
    Abaixo estão os fatores determinantes extraídos matematicamente pelo algoritmo SHAP que explicaram essa propensão:
    
    Abaixo estão os fatores comportamentais exatos extraídos via SHAP que explicam o perfil desse cliente:
    {shap_summary_df.to_string(index=False)}
    
    TAREFA:
   Com base nesses fatos exatos, crie um e-mail de marketing persuasivo, elegante oferecendo um benefício comercial ou desconto alinhado aos padrões de consumo revelados pelas variáveis de maior impacto positivo.
    REGRAS OBRIGATÓRIAS PARA TODAS AS 6 VARIAÇÕES:
    - Tom de voz: Curto, leve, informal e bem-humorado (use trocadilhos leves, linguagem próxima e emojis estratégicos, sem exageros).
    - Benefício Imediato no 1º Parágrafo: Destaque o cupom de desconto, frete grátis ou oferta logo na primeira frase/parágrafo. Não faça introduções longas.
    - Gatilhos Mentais: Aplique explicitamente escassez, urgência (ex: "válido pelas próximas 2 horas", "só hoje") e recompensa imediata.
    - Personalização: Conecte sutilmente a oferta aos produtos favoritos do cliente apontados pelo SHAP (sem citar termos técnicos como "SHAP" ou "variável").
    - Adaptação ao Contexto: Conecte a oferta diretamente ao momento do envio. Use estas referências de abordagem conforme o cenário:
       - *Almoço / Jantar:* Foco em praticidade, matar a fome rápido e não precisar cozinhar ou lavar louça.
       - *Café da tarde:* Foco em pausa merecida, aquele docinho, café ou lanche rápido para dar energia.
       - *Chuva:* Foco em conforto, ficar no sofá sem sair de casa enquanto o pedido chega quentinho.
       - *Sexta-feira à noite:* Foco em "sextou", celebração, pedir algo especial sem culpa e relaxar.
    - Tamanho: Máximo de 3 parágrafos bem curtos por cenário (ideal para leitura rápida na tela do celular).
    - Call-to-Action (CTA) Curto:** Termine com uma chamada para ação direta, imperativa e de no máximo 5 palavras (ex: "Resgate seu cupom agora!", "Pedir com desconto", "Aproveitar antes que acabe").
    """
    
    # Modelos ativos liberados para novas chaves de API
    candidate_models = [
        'gemini-3.8-flash',
        'gemini-3.8-flash-lite'
    ]
    
    for model_name in candidate_models:
        # Tenta até 3 vezes cada modelo em caso de pico de demanda (503)
        for tentativa in range(1, 4):
            try:
                response = client.models.generate_content(
                    model=model_name,
                    contents=prompt,
                )
                return f"[Gerado via {model_name}]\n\n{response.text}"
            except Exception as e:
                erro_str = str(e)
                if "503" in erro_str or "UNAVAILABLE" in erro_str:
                    espera = tentativa * 3  # Espera 3s, 6s, 9s...
                    print(f"-> {model_name} ocupado (503). Tentativa {tentativa}/3: aguardando {espera}s...")
                    time.sleep(espera)
                else:
                    print(f"-> Aviso ao usar {model_name}: {e}")
                    break  # Se for outro erro, pula direto para o próximo modelo

    return "[AVISO] Os servidores do Gemini estão temporariamente congestionados. Tente rodar novamente em alguns minutos."