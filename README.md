<p align="center">
  <img src="https://readme-typing-svg.demolab.com?font=Fira+Code&weight=700&size=42&pause=1000&color=00E676&center=true&vCenter=true&multiline=true&repeat=true&width=700&height=100&lines=%F0%9F%A6%8E+CAMALE%C3%83O+PESSOAL;Seus+dados.+Seu+PC.+Sua+IA." alt="Camaleão Pessoal" />
</p>

<p align="center">
  <img src="https://img.shields.io/badge/🧠_APRENDIZADO-CONTÍNUO-00E676?style=for-the-badge&labelColor=000" />
  <img src="https://img.shields.io/badge/⚡_ROTEAMENTO-POR_ENERGIA-FF6D00?style=for-the-badge&labelColor=000" />
  <img src="https://img.shields.io/badge/🧬_NEUROGÊNSE-AUTOMÁTICA-AA00FF?style=for-the-badge&labelColor=000" />
  <img src="https://img.shields.io/badge/☁️_CLOUD-ZERO-FF1744?style=for-the-badge&labelColor=000" />
</p>

<br>

> **Não é mais um chatbot. É um organismo digital que nasce, aprende, morre e renasce — como um camaleão trocando de pele.**

<br>

---

## 🧬 O QUE ISSO FAZ?

<table align="center">
<tr>
  <td width="33%" align="center">
    <h1>🦎</h1>
    <b>NEUROGÊNSE</b><br>
    <sub>Nichos nascem sozinhos quando você fala de algo novo. Sem config. Sem setup. Só conversa.</sub>
  </td>
  <td width="33%" align="center">
    <h1>⚡</h1>
    <b>ROTEAMENTO POR ENERGIA</b><br>
    <sub>Cada nicho "vota" com a própria confiança (NLL). O mais certo responde. Democracia neural.</sub>
  </td>
  <td width="33%" align="center">
    <h1>🧠</h1>
    <b>MEMÓRIA VETORIAL</b><br>
    <sub>Lembra conversas relevantes de qualquer nicho. Não é um banco de dados — é um cérebro.</sub>
  </td>
</tr>
<tr>
  <td width="33%" align="center">
    <h1>❄️</h1>
    <b>HIBERNAÇÃO</b><br>
    <sub>Nicho parou 7 dias? Congela. Libera RAM. Quando você volta, ele desperta.</sub>
  </td>
  <td width="33%" align="center">
    <h1>📄</h1>
    <b>RAG POR NICHO</b><br>
    <sub>Cada nicho indexa seus próprios documentos. Contexto isolado. Respostas precisas.</sub>
  </td>
  <td width="33%" align="center">
    <h1>🌐</h1>
    <b>INTERFACE WEB</b><br>
    <sub>Chat + painel de energia + diagnóstico visual. Tudo no browser. Tudo local.</sub>
  </td>
</tr>
</table>

<br>

---

## ⚡ QUICK START

```bash
# 1. Instala o cérebro → https://ollama.ai
ollama pull qwen2.5:3b

# 2. Clona o organismo
git clone https://github.com/Lucass25321/camaleao-pessoal.git
cd camaleao-pessoal

# 3. Ativa o habitat
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# 4. Desperta
python camaleao.py
```

> **Windows?** `camaleao_rodar_caslu.bat` — duplo clique. Ele cuida de tudo.

<br>

---

## 🎬 DEMO AO VIVO

```
╔══════════════════════════════════════════════════════════════╗
║  💬 quero aprender python                                    ║
║     🦎 NEUROGÊNSE: Nicho 'Coding' nasceu!                   ║
║     [Nicho: Coding │ Conf: 0.82]                            ║
║     Python é uma linguagem versátil...                       ║
║                                                              ║
║  💬 como funciona um loop for?                               ║
║     [Nicho: Coding │ Conf: 0.91]  ← roteamento por energia  ║
║     O loop 'for' em Python itera sobre...                    ║
║                                                              ║
║  💬 quero saber de futebol                                   ║
║     🦎 NEUROGÊNSE: Nicho 'Football' nasceu!                  ║
║     [Nicho: Football │ Conf: 0.75]                          ║
║                                                              ║
║  💬 quem ganhou a copa de 94?                                ║
║     [Nicho: Football │ Conf: 0.88]  ← alternou sozinho      ║
║     O Brasil venceu a Copa de 1994...                        ║
╚══════════════════════════════════════════════════════════════╝
```

<br>

---

## 🏗️ ARQUITETURA

```
         ┌──────────────────────────────────────┐
         │            💬 USUÁRIO                │
         │        "quero aprender IA"            │
         └──────────────┬───────────────────────┘
                        │
                        ▼
         ┌──────────────────────────────────────┐
         │        🔍 CLASSIFICADOR DE ENERGIA   │
         │      (NLL de cada nicho ativo)        │
         └──┬──────────┬──────────┬─────────────┘
            │          │          │
            ▼          ▼          ▼
      ┌──────────┐┌──────────┐┌──────────┐
      │ 🧠 Coding ││ 🧠 Saúde ││ 🧠 ...    │
      │ Conf: 0.9││ Conf: 0.3││ Conf: 0.1│
      └────┬─────┘└──────────┘└──────────┘
           │
           ▼
      ┌──────────────────────────────────────┐
      │        📚 RAG + MEMÓRIA VETORIAL     │
      │         (contexto do nicho)           │
      └──────────────┬───────────────────────┘
                     │
                     ▼
      ┌──────────────────────────────────────┐
      │           🤖 OLLAMA (local)          │
      │         qwen2.5:3b / 7b / etc        │
      └──────────────────────────────────────┘
```

<br>

---

## 🧪 COMO FUNCIONA POR DENTRO

| Mecanismo | O que acontece |
|---|---|
| **Embedding** | Cada mensagem vira um vetor (sentence-transformers) |
| **NLL Score** | Ollama calcula a "energia" — quão confiante cada nicho está |
| **Votação** | O nicho com maior confiança (menor NLL) responde |
| **Neurogênese** | Se nenhum nicho confia o bastante (> limiar), um novo nasce |
| **Hibernação** | Nichos sem uso por 7 dias congelam (desativam do roteamento) |
| **Reativação** | Mensagem similar ao nicho hibernado? Ele desperta automaticamente |

<br>

---

## 📊 STACK

<p align="center">
  <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/Ollama-000000?style=for-the-badge&logo=ollama&logoColor=white" />
  <img src="https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white" />
  <img src="https://img.shields.io/badge/ChromaDB-E63946?style=for-the-badge&logoColor=white" />
  <img src="https://img.shields.io/badge/Sentence--Transformers-FF6D00?style=for-the-badge&logoColor=white" />
  <img src="https://img.shields.io/badge/HTML5-E34F26?style=for-the-badge&logo=html5&logoColor=white" />
</p>

<br>

---

## 🤝 CONTRIBUIR

```bash
# Fork → Branch → Commit → PR
git checkout -b feature/nova-funcionalidade
git commit -m "feat: adiciona X"
git push origin feature/nova-funcionalidade
```

<br>

---

## 📜 LICENÇA

MIT — faça o que quiser. Só não venda como se fosse seu. 🦎

<br>

---

<p align="center">
  <b>⭐ Se achou útil, dá uma estrela. Ajuda o camaleão a crescer.</b>
</p>

<p align="center">
  <sub>feito com 🧬 por <a href="https://github.com/Lucass25321">Lucass25321</a></sub>
</p>
