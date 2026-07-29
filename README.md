# 🦎 Camaleão Pessoal

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white)
![Ollama](https://img.shields.io/badge/Ollama-local-000?logo=ollama&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-v3.2.0-orange)
![Cloud](https://img.shields.io/badge/Cloud-Zero-red)

> **Assistente pessoal com aprendizado contínuo, roteamento por energia e neurogênese automática.**
> 100% local. Seus dados nunca saem do seu PC.

---

## 🧬 O que faz o Camaleão?

| Feature | Descrição |
|---|---|
| 🦎 **Neurogênese** | Nichos nascem sozinhos quando você fala de temas novos |
| ⚡ **Roteamento por energia** | Cada nicho vota com a própria confiança (NLL) |
| 🧠 **Memória vetorial** | Lembra conversas relevantes de qualquer nicho |
| ❄️ **Hibernação** | Nichos inativos por 7 dias congelam automaticamente |
| 📄 **RAG por nicho** | Cada nicho indexa seus próprios documentos |
| 🌐 **Interface web** | Chat + painel de energia + diagnóstico visual |

---

## ⚡ Quick Start

```bash
# 1. Instala Ollama → https://ollama.ai

# 2. Baixa o modelo
ollama pull qwen2.5:3b

# 3. Clona e instala
git clone https://github.com/SEU_USER/camaleao-pessoal.git
cd camaleao-pessoal
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# 4. Roda
python camaleao.py
```

> **Windows?** Duplo-clique em `camaleao_rodar_caslu.bat` — ele cuida de tudo.

---

## 🎬 Demo

```
Voce: quero aprender python
   🦎 NEUROGENESE: Nicho 'Coding' nasceu!
   [Nicho: Coding | Conf: 0.82]
   Python é uma linguagem versátil...

Voce: como funciona um loop for?
   [Nicho: Coding | Conf: 0.91]  ← roteamento por energia
   O loop 'for' em Python itera sobre...

Voce: quero saber de futebol
   🦎 NEUROGENESE: Nicho 'Football' nasceu!
   [Nicho: Football | Conf: 0.75]

Voce: quem ganhou a copa de 94?
   [Nicho: Football | Conf: 0.88]  ← alternou automaticamente
   O Brasil venceu a Copa de 1994...
```

---

## 🏗️ Arquitetura

```
┌─────────────────────────────────────────────────────┐
│                    USUÁRIO                           │
│                  "quero aprender IA"                 │
└──────────────────────┬──────────────────────────────┘
                       ▼
┌──────────────────────────────────────────────────────┐
│              ⚡ ROTEAMENTO POR ENERGIA               │
│                                                      │
│  1. Keywords? ──→ match direto (instantâneo)         │
│  2. Energia  ──→ cada nicho gera 50 tokens           │
│                   menor NLL vence                    │
│  3. Fallback ──→ distância de embedding              │
└──────────────────────┬──────────────────────────────┘
                       ▼
┌──────────────────────────────────────────────────────┐
│                    🚪 GATE                           │
│                                                      │
│  • Surpresa extrema? ──→ rejeita                     │
│  • Energia forte?    ──→ bypass distância            │
│  • Confiança baixa?  ──→ rejeita                     │
└──────────────────────┬──────────────────────────────┘
                       ▼
┌──────────────────────────────────────────────────────┐
│              🧬 NOVIDADE + NEUROGÊNESE               │
│                                                      │
│  margem = NLL_nicho − NLL_base                       │
│  margem > 0.5 e NLL > 7.0 → NOVO                    │
│  3 prompts novos → nicho nasce automaticamente       │
└──────────────────────┬──────────────────────────────┘
                       ▼
┌──────────────────────────────────────────────────────┐
│                💬 RESPOSTA + MEMÓRIA                 │
│                                                      │
│  system prompt + contexto + RAG + memória vetorial   │
│  → Ollama (qwen2.5:3b local)                        │
│  → Salva embedding + NLL no nicho                    │
└──────────────────────────────────────────────────────┘
```

---

## 🖥️ Interface Web

Rode `/web` no terminal e acesse `http://localhost:8080`:

- 💬 Chat integrado com indicador de routing
- ⚡ Painel de energia (NLL por nicho em tempo real)
- 🦎 Criação de nichos pela interface
- 📊 Diagnóstico visual completo
- 🔍 Detalhes do nicho (duplo-clique)

---

## 📖 Uso

### Comandos

| Comando | Descrição |
|---|---|
| `/nicho <nome> "<desc>" [kw]` | Criar nicho manual |
| `/status` | Ver saúde do organismo |
| `/diagnostico` | Métricas completas |
| `/energia on\|off` | Toggle roteamento por energia |
| `/web [porta]` | Interface web |
| `/add_doc <nicho> <arquivo>` | Adicionar documento RAG |
| `/exportar` | Backup ZIP |
| `/importar <arquivo>` | Restaurar backup |
| `/bom` / `/ruim` | Feedback da última resposta |
| `/corrigir [nicho] <resposta>` | Corrigir última resposta |

### Configuração

Edite as variáveis no topo de `camaleao.py`:

```python
class Config:
    MODELO_BASE = "qwen2.5:3b"              # Modelo Ollama
    MODELO_EMBEDDING = "paraphrase-multilingual-MiniLM-L12-v2"
    
    USA_ROTEAMENTO_ENERGIA = True            # Roteamento por NLL
    ENERGIA_LIMIAR_MARGEM = 0.5             # Margem para novidade
    USA_NEUROGENESE = True                   # Criação automática de nichos
    NEURO_MIN_AMOSTRAS_NASCER = 3           # Mensagens para criar nicho
    NEURO_MAX_NICHOS = 10                   # Limite de nichos auto-gerados
    HIBERNACAO_DIAS = 7                     # Dias para hibernar nicho
```

---

## 🧪 Testes

```bash
python teste_camaleao.py
```

Valida: roteamento por energia, gate, feedback, neurogênese, proteção de nichos manuais.

---

## 📁 Estrutura

```
camaleao-pessoal/
├── camaleao.py              # Motor principal (2000+ linhas)
├── camaleao_web.html        # Interface web
├── teste_camaleao.py        # Suite de testes
├── camaleao_rodar_caslu.bat # Launcher Windows
├── requirements.txt         # Dependências
├── LICENSE                  # MIT
└── README.md                # Este arquivo
```

### Dados (gerados automaticamente, não versionar)

```
./nichos/      # JSONs de cada nicho
./memoria/     # Embeddings vetoriais
./dados/       # Dados auxiliares
```

---

## 🔬 Como funciona

### Roteamento por Energia

Diferente do roteamento por distância (que colapsa em espaços anisotrópicos), cada nicho gera 50 tokens e mede a velocidade. Menor NLL = maior confiança = vence.

### Neurogênese

Quando o melhor nicho tem NLL muito pior que o modelo base, o prompt é novo. Após 3 prompts novos, um nicho nasce automaticamente com nome gerado a partir das palavras-chave.

### Memória

Cada nicho mantém: protótipo (centroide), NLL lar (EMA), histórico, resumos periódicos. A memória vetorial global recupera conversas relevantes de qualquer nicho.

---

## 🤝 Contribuição

1. Fork
2. Branch (`git checkout -b feature/nova-coisa`)
3. Commit (`git commit -m 'add: nova coisa'`)
4. Push (`git push origin feature/nova-coisa`)
5. Pull Request

---

## 📄 Licença

MIT — veja [LICENSE](LICENSE).

---

## 🙏 Referências

- [Ollama](https://ollama.ai) — inferência local
- [Sentence Transformers](https://www.sbert.net/) — embeddings
- [Qwen 2.5](https://github.com/QwenLM/Qwen2.5) — modelo base

---

*"O organismo continua sendo a tese; os princípios são o que o organismo nos ensinou sobre os olhos com que ele vê."*
