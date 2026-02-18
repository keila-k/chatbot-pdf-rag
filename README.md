# Chatbot baseado em PDFs (RAG) — Embeddings + Busca Vetorial

Projeto para responder perguntas com base no conteúdo de arquivos PDF usando:
- Extração de texto de PDFs
- Quebra em trechos (chunks)
- Geração de embeddings
- Indexação e busca vetorial (FAISS)
- Interface de chat (Streamlit)

## Estrutura
- `inputs/`: sentenças exigidas pela DIO
- `pdfs/`: PDFs para indexação
- `src/`: scripts de indexação e app do chatbot
- `vectorstore/`: índice FAISS + metadados (gerado)


## Prints do Projeto
### Indexação dos PDFs
![VS Code terminal showing PDF indexing initialization with project folder structure and Python virtual environment activated](assets/prints/01 VS Code terminal PDF indexing.png)
![Streamlit Run](assets/prints/02 VS Code streamlit run src app.png)
![Criação do ReadMe](assets/prints/03 Git Bash README EOF)

### Chat (Streamlit)
![Indexar PDF](assets/prints/04 VS Code Indexar PDF.png)
![Streamlit Run](assets/prints/05 VS Code Streamlit Run.png)

### Localhost
![Abriu o Localhost com erro](assets/prints/06 Abriu o Localhost com o Chatbot.png)
![Abriu o Localhost corretamente](assets/prints/07 Abriu o Localhost com o Chatbot de modo correto agora.png)

### Evidências adicionais
![A pergunta do Exemplo](assets/prints/08 Chat com uma resposta sobre o Grapho de Djikstra -1)
![Parte da Resposta](assets/prints/09 Chat com uma resposta sobre o Grapho de Djikstra -2.png)
![Zoom out na Resposta](10 Chat com uma resposta sobre o Grapho de Djikstra -38.png)


### Explicação dos Prints
Neste exemplo 5 pdfs de artigos sobre o Algoritmo de Djikstra foram utilizados: 3 deles em português e 2 em espanhol.

Outros PDFs podem ser utilizados no lugar dos que foram consumidos neste exemplo.

## Mini-Relatório. Meus cometários específicos [por keila-k, 2026-02-18]. 
Rodei todo o proceso no meu computador, em localhost. É a primeira vez que faço RAG. O streamlit run conseguiu abrir o localhost mesmo sem vectorstore, mas o app parava de rodar quando tentava carregar vectorstore/index.faiss e vectorstore/chunks.json. Desconfiei da versão de Python utilizada, mas mantive a mais atual disponível, não troquei. O vectorstore/ ficava vazio porque a indexação não terminava (o VS Code sugeria MemoryError). 
Pesquisei como solucionar. 
A causa mais comum é: PDFs grandes + chunking ingênuo + embeddings em lote grande. 
Então substituí meu "src/indexar_pdfs.py" por uma versão incremental: 
- para limitar chunks por PDF, 
- limitar tamanho total de texto por PDF, 
- e adicionar FAISS em lotes e salvar o chunks.json no final sem estourar RAM. 
Depois indexei PDFs e conferi o Vectorstore. Rodei o Streamlit. Deu certo. Este é apenas um comentário pessoal se também for a sua primeira vez fazendo RAG e/ou se você enfrentou um problema similar. Eu perdi uma manhã de feriado solucionando. Criar foi relativamente rápido.

## Como rodar
1) Criar ambiente e instalar dependências
```bash
python -m venv .venv
source .venv/Scripts/activate
pip install -r requirements.txt

