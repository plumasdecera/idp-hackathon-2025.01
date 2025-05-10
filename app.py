from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from dotenv import load_dotenv
import os
import requests
from PyPDF2 import PdfReader
import json

load_dotenv()

app = Flask(__name__)
CORS(app)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={GEMINI_API_KEY}"

def extrair_texto_pdf(nome_arquivo):
    texto = ""
    try:
        with open(nome_arquivo, 'rb') as arquivo_pdf:
            leitor_pdf = PdfReader(arquivo_pdf)
            for pagina in leitor_pdf.pages:
                texto += pagina.extract_text() or ""
    except Exception as e:
        print(f"Erro ao processar {nome_arquivo}: {e}")
    return texto

def carregar_contexto_pdfs(diretorio):
    contexto = ""
    arquivos_pdf = [f for f in os.listdir(diretorio) if f.endswith(".pdf")]
    for arquivo in arquivos_pdf:
        caminho_arquivo = os.path.join(diretorio, arquivo)
        texto = extrair_texto_pdf(caminho_arquivo)
        if texto:
            contexto += f"\n--- Conteúdo de: {arquivo} ---\n{texto}\n"
    return contexto

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/perguntar', methods=['POST'])
def perguntar():
    data = request.get_json()
    pergunta = data.get('pergunta', '').strip()

    if not pergunta:
        return jsonify({'erro': 'Pergunta vazia.'}), 400

    path = "context"
    contexto_pdfs = carregar_contexto_pdfs(os.path.join(os.getcwd(), path))
    if not contexto_pdfs:
        return jsonify({'erro': 'Nenhum PDF encontrado ou erro ao processar os PDFs.'}), 500

    instrucao_base = """
        Purpose and Goals:
        * Act as a specialized search assistant for a college project.
        * Utilize the provided PDF files from the college to answer student queries.
        * Extract and synthesize relevant information to generate comprehensive and accurate responses.
        * Help students find specific details and understand concepts within the provided materials.

        Behaviors and Rules:

        1) Information Retrieval:
        a) When a student asks a question, search the provided PDF files for relevant information.
        b) Prioritize information that directly answers the question with full explanation.
        c) If the answer is present in multiple documents, synthesize the information into a single, concise, and coherent response. Do not repeat the same data in different words.

        2) Response Generation:
        a) Always generate a direct, clear, and complete answer in the "body" field using only the content from the PDFs.
        b) Do not respond with phrases like "see the calendar" or "consult the PDF". Instead, explain explicitly what the document states.
        c) Cite the relevant filename(s) in the "source" field only.
        d) Your full response must be a clean JSON (no markdown) structured as follows:
        - "title": a short descriptive summary of the answer.
        - "body": a detailed explanation, written clearly and informatively.
        - "source": the filename(s) of the PDF(s) used to construct the answer.

        e) The "title" should summarize the topic. The "body" must provide a full explanation and may include basic HTML tags to improve readability:
        - Use <strong> to emphasize important words, dates, or terms;
        - Use <ul><li> for structured lists;
        - Use <br> to break lines for better formatting;
        - Use <a href='#'> for references, if applicable;
        - Do not use markdown or code blocks (e.g., no ```json).
        
        f) Do not include generic introductions such as "the calendar contains the dates" or "according to the document...". Go straight to the concrete information and facts.
        g) If the answer is a list, use <ul><li> to format it. For example:
        <ul>
            <li>Item 1</li>
            <li>Item 2</li>
        </ul>

        
        i) If the answer is a number, use the format 1.234,56 (comma as decimal separator). For example: 1.234,56.
        j) If the answer is a percentage, use the format 1.234,56% (comma as decimal separator). For example: 1.234,56%.
        k) If the answer is a currency, use the format R$ 1.234,56 (comma as decimal separator). For example: R$ 1.234,56.
        l) If the answer is a URL, use the format https://www.example.com (without http:// or www.). For example: https://www.example.com.
        3) Limitations:
        a) If the information is not found in the PDFs, state clearly in the "body": "Essa informação não foi encontrada nos arquivos fornecidos."
        b) Do not create or infer information that is not explicitly present in the documents.
        c) Do not refer to external sources or websites beyond the provided PDF content.

        Overall Tone:
        * Be informative, helpful, and objective.
        * Use clear, academic, and professional language appropriate for students and faculty.
        """

    prompt_completo = f"{instrucao_base}\n\nContexto dos PDFs:\n{contexto_pdfs}\n\nPergunta: {pergunta}"

    try:
        response = requests.post(GEMINI_API_URL, json={
            "contents": [{"parts": [{"text": prompt_completo}]}]
        })

        result = response.json()

        if 'error' in result:
            code = result['error'].get('code')
            message = result['error'].get('message', 'Erro desconhecido.')
            print(f"[Gemini API] ❌ Erro {code}: {message}")
            return jsonify({'erro': f"Erro da API Gemini ({code}): {message}"}), 400

        resposta_texto = result.get('candidates', [{}])[0].get('content', {}).get('parts', [{}])[0].get('text', 'Sem resposta.')

        try:
            resposta_limpa = resposta_texto.strip().removeprefix("```json").removesuffix("```").strip()
            resposta_json = json.loads(resposta_limpa)
        except Exception as e:
            print("❌ Erro ao interpretar resposta como JSON:", e)
            resposta_json = {
                "title": "Resposta não formatada",
                "body": resposta_texto,
                "source": "Desconhecida"
            }
        dumped = json.dumps(resposta_json, indent=4, ensure_ascii=False)
        print(f"[Gemini API] ✅ Resposta: {dumped}")
        return jsonify({'resposta': resposta_json})

    except Exception as e:
        print("❌ Erro inesperado:", e)
        return jsonify({'erro': 'Erro inesperado ao consultar a API Gemini.'}), 500

if __name__ == '__main__':
    app.run(debug=True)
