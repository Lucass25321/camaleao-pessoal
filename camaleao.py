# -*- coding: utf-8 -*-
"""
Camaleao Pessoal v3.2.0
100% local. Zero cloud.
Arquitetura: roteamento por energia (NLL), neurogenese automatica, hibernacao, memoria longo prazo, RAG, web, transmutacao, nomes inteligentes
"""

import os
import json
import shutil
import zipfile
import glob
import re
import http.server
from http.server import SimpleHTTPRequestHandler
import socketserver
import webbrowser
import numpy as np
from datetime import datetime
from typing import List, Dict, Optional, Tuple
import ollama
from sentence_transformers import SentenceTransformer

# =============================================================================
# CONFIGURACAO
# =============================================================================

class Config:
    MODELO_BASE = "qwen2.5:3b"
    MODELO_EMBEDDING = "paraphrase-multilingual-MiniLM-L12-v2"
    PASTA_DADOS = "./dados"
    PASTA_NICHOS = "./nichos"
    PASTA_MEMORIA = "./memoria"

    GATE_LIMIAR_DISTANCIA = 5.0
    DISTANCIA_MAX_ROTEAMENTO = 6.0     # v3.1.5: 4.0 -> 6.0 (embeddings de 384d podem ter dist 5-6 mesmo semanticamente proximos)
    WARMUP_AMOSTRAS = 8

    USA_ENERGIA = True
    ENERGIA_MAX_NLL = 10.0

    USA_CONTEXTO = True
    CONTEXTO_DECAY = 0.7
    CONTEXTO_LIMIAR_MUDANCA = 0.3

    USA_NEUROGENESE = True
    NEURO_LIMIAR_NOVIDADE = 5.0

    USA_ROTEAMENTO_ENERGIA = True   # v3.2: roteamento por NLL (energia) em vez de distancia
    ENERGIA_LIMIAR_MARGEM = 0.5     # v3.2: margem NLL_nicho - NLL_base para novidade (aumentado: 0.3->0.5 para reduzir falsos positivos)
    NEURO_MIN_AMOSTRAS_NASCER = 3
    NEURO_MAX_NICHOS = 10
    NEURO_JANELA_EXPLORACAO = 2
    NEURO_LIMIAR_JANELA = 3.0
    NEURO_LIMIAR_FUSAO = 8.0

    HIBERNACAO_DIAS = 7
    HIBERNACAO_ATIVA = True
    USA_FEEDBACK_AUTO = False
    USA_FUSAO = False
    CONSOLIDACAO_LIMIAR_VAR = 3.0

    MEMORIA_RESUMO_A_CADA = 10
    MEMORIA_RESUMO_ATIVA = True

    DIAGNOSTICO_JANELA = 20

    GATE_LIMIAR_CONFIANCA_ENERGIA = 0.05
    GATE_LIMIAR_CONFIANCA_CLASSICO = 0.3

    def __init__(self):
        os.makedirs(self.PASTA_DADOS, exist_ok=True)
        os.makedirs(self.PASTA_NICHOS, exist_ok=True)
        os.makedirs(self.PASTA_MEMORIA, exist_ok=True)

# Stopwords comuns em portugues (usadas por neurogenese e criar_nicho)
STOPWORDS = {
    "o","a","os","as","um","uma","de","do","da","dos","das","em","no","na","nos","nas",
    "e","que","para","com","por","nao","eu","voce","ele","ela","eles","elas",
    "isso","isto","aquilo","meu","minha","seu","sua","gosto","vou","vai",
    "hoje","agora","quem","qual","quando","onde","como","porque","mas","mais",
    "muito","bem","ja","ainda","sobre","tambem","todo","toda","todos","todas",
    "esse","essa","esses","essas","aquele","aquela","estou","esta","sao","ser",
    "ter","foi","sobre","para","com","que","sao","este","este","outro","outra",
    "outros","outras","mesmo","mesma","proprio","propria","qualquer","nenhum",
    "nenhuma","algum","alguma","alguns","algumas","cada","quais","pois","menos",
    "sempre","nunca","antes","depois","durante","entre","sob","atraves","dentro",
    "fora","alem","acima","abaixo","perto","longe","aqui","ali","la","ca","assim",
    "tanto","tanta","tal","tais","pouco","pouca",
    # Cumprimentos e frases feitas
    "boa","tarde","bom","dia","noite","oi","ola","eai","fala","salve",
    # Verbos muito comuns
    "quer","quero","queria","gostaria","pode","preciso","precisa","vamos","fazer",
    "apreender","aprender","estudar","saber","entender","conhecer",
    # Artigos/preposicoes extras
    "uns","umas","ao","aos","pela","pelo","pelos","pelas",
    "deste","desse","nesse","naquele","desta","dessa","nessa","naquela",
    # Pronomes
    "me","te","se","nos","vos","lhe","lhes","mim","ti","si",
    # Adverbios comuns
    "apenas","tambem","mesmo","ainda","sempre","nunca","talvez",
    "depois","antes","agora","hoje","ontem","amanha","cedo",
    # Preenchimento
    "tipo","coisa","negocio","parada","mano","cara",
    "obrigado","obrigada","valeu","por favor","desculpa","desculpe","opa"
}

# =============================================================================
# MEMORIA VETORIAL LOCAL
# =============================================================================

class MemoriaVetorial:
    def __init__(self, pasta: str, dim: int = 384):
        self.pasta = pasta
        self.dim = dim
        self.embeddings = []
        self.textos = []
        self.metadados = []
        os.makedirs(self.pasta, exist_ok=True)
        self._carregar()

    def _carregar(self):
        path = os.path.join(self.pasta, "memoria.npz")
        if os.path.exists(path):
            try:
                data = np.load(path, allow_pickle=True)
                self.embeddings = list(data["embeddings"]) if "embeddings" in data else []
                self.textos = list(data["textos"]) if "textos" in data else []
                self.metadados = list(data["metadados"]) if "metadados" in data else []
            except Exception as e:
                print(f"   [AVISO] Memoria corrompida, recriando: {e}")
                self.embeddings = []
                self.textos = []
                self.metadados = []

    def salvar(self):
        os.makedirs(self.pasta, exist_ok=True)
        path = os.path.join(self.pasta, "memoria.npz")
        temp_path = os.path.join(self.pasta, "memoria_temp")
        try:
            np.savez(temp_path,
                     embeddings=np.array(self.embeddings) if self.embeddings else np.zeros((0, self.dim)),
                     textos=np.array(self.textos, dtype=object) if self.textos else np.array([]),
                     metadados=np.array(self.metadados, dtype=object) if self.metadados else np.array([]))
            shutil.move(temp_path + ".npz", path)
        except Exception as e:
            print(f"   [ERRO] Falha ao salvar memoria: {e}")

    def adicionar(self, texto: str, embedding: np.ndarray, metadado: Dict):
        self.embeddings.append(embedding)
        self.textos.append(texto)
        self.metadados.append(json.dumps(metadado))
        self.salvar()

    def buscar(self, embedding: np.ndarray, k: int = 5) -> List[Tuple[str, float, Dict]]:
        if not self.embeddings:
            return []
        embeddings = np.array(self.embeddings)
        norm_query = embedding / (np.linalg.norm(embedding) + 1e-10)
        norm_mem = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-10)
        scores = np.dot(norm_mem, norm_query)
        top_k = np.argsort(scores)[-k:][::-1]
        resultados = []
        for idx in top_k:
            resultados.append((self.textos[idx], float(scores[idx]), json.loads(self.metadados[idx])))
        return resultados

# =============================================================================
# RAG - DOCUMENTOS POR NICHO
# =============================================================================

class DocumentoNicho:
    def __init__(self, nicho_nome: str, config: Config):
        self.nicho_nome = nicho_nome
        self.config = config
        self.pasta_docs = os.path.join(config.PASTA_NICHOS, nicho_nome, "docs")
        self.pasta_index = os.path.join(config.PASTA_NICHOS, nicho_nome, "index")
        self.embeddings = []
        self.textos = []
        self.fontes = []
        os.makedirs(self.pasta_docs, exist_ok=True)
        os.makedirs(self.pasta_index, exist_ok=True)
        self._carregar_index()

    def _caminho_index(self):
        return os.path.join(self.pasta_index, "rag.npz")

    def _carregar_index(self):
        path = self._caminho_index()
        if os.path.exists(path):
            try:
                data = np.load(path, allow_pickle=True)
                self.embeddings = list(data["embeddings"]) if "embeddings" in data else []
                self.textos = list(data["textos"]) if "textos" in data else []
                self.fontes = list(data["fontes"]) if "fontes" in data else []
            except Exception as e:
                print(f"   [AVISO] Index RAG corrompido para {self.nicho_nome}: {e}")
                self.embeddings = []
                self.textos = []
                self.fontes = []

    def salvar_index(self):
        path = self._caminho_index()
        temp = path + ".tmp"
        try:
            np.savez(temp,
                     embeddings=np.array(self.embeddings) if self.embeddings else np.zeros((0, 384)),
                     textos=np.array(self.textos, dtype=object) if self.textos else np.array([]),
                     fontes=np.array(self.fontes, dtype=object) if self.fontes else np.array([]))
            shutil.move(temp, path)
        except Exception as e:
            print(f"   [ERRO] Falha ao salvar index RAG: {e}")

    def adicionar_documento(self, caminho: str, embedder) -> int:
        if not os.path.exists(caminho):
            return 0
        nome = os.path.basename(caminho)
        try:
            with open(caminho, "r", encoding="utf-8", errors="ignore") as f:
                texto = f.read()
        except Exception as e:
            print(f"   [ERRO] Nao consegui ler {caminho}: {e}")
            return 0
        chunks = self._chunkar(texto)
        for chunk in chunks:
            if len(chunk.strip()) < 20:
                continue
            emb = embedder.encode(chunk, convert_to_numpy=True)
            self.embeddings.append(emb)
            self.textos.append(chunk)
            self.fontes.append(nome)
        self.salvar_index()
        return len(chunks)

    def _chunkar(self, texto: str, tamanho=500, overlap=100) -> List[str]:
        paragrafos = [p.strip() for p in re.split(r"\n\s*\n", texto) if len(p.strip()) > 10]
        chunks = []
        for p in paragrafos:
            if len(p) <= tamanho:
                chunks.append(p)
            else:
                for i in range(0, len(p), tamanho - overlap):
                    chunk = p[i:i+tamanho]
                    if len(chunk) > 50:
                        chunks.append(chunk)
        return chunks

    def buscar(self, embedding: np.ndarray, k: int = 3) -> List[Tuple[str, str, float]]:
        if not self.embeddings:
            return []
        embeddings = np.array(self.embeddings)
        norm_query = embedding / (np.linalg.norm(embedding) + 1e-10)
        norm_mem = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-10)
        scores = np.dot(norm_mem, norm_query)
        top_k = np.argsort(scores)[-k:][::-1]
        resultados = []
        for idx in top_k:
            if scores[idx] > 0.3:
                resultados.append((self.textos[idx], self.fontes[idx], float(scores[idx])))
        return resultados

    def listar(self) -> List[str]:
        if not os.path.exists(self.pasta_docs):
            return []
        return [f for f in os.listdir(self.pasta_docs) if os.path.isfile(os.path.join(self.pasta_docs, f))]

    def limpar(self):
        self.embeddings = []
        self.textos = []
        self.fontes = []
        if os.path.exists(self.pasta_index):
            shutil.rmtree(self.pasta_index)
        if os.path.exists(self.pasta_docs):
            for f in os.listdir(self.pasta_docs):
                os.remove(os.path.join(self.pasta_docs, f))
        os.makedirs(self.pasta_docs, exist_ok=True)
        os.makedirs(self.pasta_index, exist_ok=True)

# =============================================================================
# NICHO
# =============================================================================

class Nicho:
    def __init__(self, nome: str, descricao: str, config: Config, palavras_chave: List[str] = None, auto_gerado: bool = False):
        self.nome = nome
        self.descricao = descricao
        self.config = config
        self.palavras_chave = palavras_chave or []
        self.auto_gerado = auto_gerado
        self.prototipo = None
        self.raio = None
        self.n_amostras = 0
        self.energia = 1.0
        self.historico = []
        self.ultima_atividade = datetime.now().isoformat()
        self.conversas_desde_resumo = 0
        self.resumo_longo = ""
        self.hibernado = False
        self.consolidado = False
        self.distancias_historico = []
        self.nll_historico = []    # v3.2: historico de NLLs neste nicho
        self.nll_lar = None         # v3.2: EMA da NLL no proprio dominio ("lar")
        os.makedirs(self.config.PASTA_NICHOS, exist_ok=True)
        self._carregar()

    def _caminho(self):
        return os.path.join(self.config.PASTA_NICHOS, f"{self.nome}.json")

    def _carregar(self):
        path = self._caminho()
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self.descricao = data.get("descricao", self.descricao)
                    self.palavras_chave = data.get("palavras_chave", self.palavras_chave)
                    self.auto_gerado = data.get("auto_gerado", self.auto_gerado)
                    self.prototipo = np.array(data["prototipo"]) if data.get("prototipo") else None
                    self.raio = data.get("raio")
                    self.n_amostras = data.get("n_amostras", 0)
                    self.energia = data.get("energia", 1.0)
                    self.historico = data.get("historico", [])
                    self.ultima_atividade = data.get("ultima_atividade", datetime.now().isoformat())
                    self.conversas_desde_resumo = data.get("conversas_desde_resumo", 0)
                    self.resumo_longo = data.get("resumo_longo", "")
                    self.hibernado = data.get("hibernado", False)
                    self.consolidado = data.get("consolidado", False)
                    self.distancias_historico = data.get("distancias_historico", [])
                    self.nll_historico = data.get("nll_historico", [])
                    self.nll_lar = data.get("nll_lar")
            except json.JSONDecodeError:
                print(f"   [AVISO] Arquivo {self.nome}.json corrompido, recriando nicho.")
                self.prototipo = None
                self.raio = None
                self.n_amostras = 0
                self.historico = []

    def salvar(self):
        os.makedirs(self.config.PASTA_NICHOS, exist_ok=True)
        data = {
            "nome": self.nome,
            "descricao": self.descricao,
            "palavras_chave": self.palavras_chave,
            "auto_gerado": self.auto_gerado,
            "prototipo": self.prototipo.tolist() if self.prototipo is not None else None,
            "raio": float(self.raio) if self.raio is not None else None,
            "n_amostras": int(self.n_amostras),
            "energia": float(self.energia),
            "historico": self.historico[-100:],
            "ultima_atividade": self.ultima_atividade,
            "conversas_desde_resumo": self.conversas_desde_resumo,
            "resumo_longo": self.resumo_longo,
            "hibernado": self.hibernado,
            "consolidado": self.consolidado,
            "distancias_historico": [float(d) for d in self.distancias_historico[-50:]],
            "nll_historico": [float(d) for d in self.nll_historico[-50:]],
            "nll_lar": float(self.nll_lar) if self.nll_lar is not None else None
        }
        path = self._caminho()
        temp_path = path + ".tmp"
        try:
            with open(temp_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            shutil.move(temp_path, path)
        except Exception as e:
            print(f"   [ERRO] Falha ao salvar nicho {self.nome}: {e}")

    def atualizar_prototipo(self, embedding: np.ndarray, forcar: bool = False):
        if self.nome == "geral":
            return
        if self.consolidado and not forcar:
            return
        if self.prototipo is None:
            self.prototipo = embedding.copy()
            self.raio = 0.0
        else:
            novo = (self.prototipo * self.n_amostras + embedding) / (self.n_amostras + 1)
            dist = np.linalg.norm(embedding - self.prototipo)
            self.raio = (self.raio * self.n_amostras + dist) / (self.n_amostras + 1)
            self.prototipo = novo
        if self.prototipo is not None:
            self.distancias_historico.append(float(np.linalg.norm(embedding - self.prototipo)))
            if len(self.distancias_historico) > 50:
                self.distancias_historico = self.distancias_historico[-40:]
        self.n_amostras += 1
        if not self.consolidado and self.n_amostras >= self.config.WARMUP_AMOSTRAS:
            if len(self.distancias_historico) >= 3:
                var = float(np.var(self.distancias_historico[-5:]))
                if var > self.config.CONSOLIDACAO_LIMIAR_VAR:
                    print(f"   🟡 CONSOLIDACAO ADIADA: Nicho '{self.nome}' tem variancia alta ({var:.2f}). Continua plástico.")
                    self.salvar()
                    return
            self.consolidado = True
            self.salvar()
            print(f"   🔒 CONSOLIDACAO: Nicho '{self.nome}' consolidado apos {self.n_amostras} amostras.")
        else:
            self.salvar()

    def atualizar_nll(self, nll: float):
        """Atualiza historico de NLL e EMA do 'lar' (media movel exponencial)."""
        self.nll_historico.append(float(nll))
        if len(self.nll_historico) > 50:
            self.nll_historico = self.nll_historico[-40:]
        # EMA do lar: alpha=0.3 para suavizar mas reagir a mudancas
        if self.nll_lar is None:
            self.nll_lar = nll
        else:
            self.nll_lar = 0.7 * self.nll_lar + 0.3 * nll
        self.salvar()

    def distancia(self, embedding: np.ndarray) -> float:
        if self.prototipo is None:
            return float("inf")
        if self.nome == "geral":
            return float("inf")
        return float(np.linalg.norm(embedding - self.prototipo))

    def em_warmup(self) -> bool:
        return self.n_amostras < self.config.WARMUP_AMOSTRAS

    def match_palavras_chave(self, texto: str) -> float:
        if not self.palavras_chave:
            return 0.0
        texto_lower = texto.lower()
        matches = sum(1 for palavra in self.palavras_chave if palavra.lower() in texto_lower)
        return matches / len(self.palavras_chave)

    def system_prompt(self) -> str:
        status = "[HIBERNADO]" if self.hibernado else ""
        resumo_str = f"\n\nResumo:\n{self.resumo_longo}" if self.resumo_longo else ""
        return f"""Voce e o assistente pessoal de Lucas. {status}
Nicho: {self.nome} — {self.descricao}.
Amostras: {self.n_amostras}.{resumo_str}
Responda de forma pessoal, contextual e honesta. Se nao souber, diga "nao sei"."""

# =============================================================================
# ORGANISMO CAMALEAO
# =============================================================================

class OrganismoCamaleao:
    def __init__(self):
        self.config = Config()
        print("Carregando modelo de embeddings (primeira vez pode demorar)...")
        self.embedder = SentenceTransformer(self.config.MODELO_EMBEDDING)
        self.memoria = MemoriaVetorial(self.config.PASTA_MEMORIA)
        self.nichos: Dict[str, Nicho] = {}
        self.nicho_ativo: Optional[str] = None
        self.ultimo_nicho: Optional[str] = None
        self.buffer_novidade: List[Tuple[str, np.ndarray]] = []
        self.episodio = []
        self.ultima_interacao = None
        self._nll_cache: Dict[str, float] = {}  # v3.2: cache de NLL por nicho (evita medir 2x)
        self.documentos: Dict[str, DocumentoNicho] = {}
        self.janela_exploracao = 0
        self._carregar_nichos()
        self._garantir_nicho_base()
        self._carregar_documentos()

    def _carregar_nichos(self):
        if not os.path.exists(self.config.PASTA_NICHOS):
            return
        for arquivo in os.listdir(self.config.PASTA_NICHOS):
            if arquivo.endswith(".json"):
                nome = arquivo[:-5]
                nicho = Nicho(nome, "", self.config)
                self.nichos[nome] = nicho
                print(f"   Nicho carregado: {nome} ({nicho.n_amostras} amostras)")
        self._limpar_duplicados()

    def _limpar_duplicados(self):
        if not self.config.USA_NEUROGENESE:
            return
        nichos_auto = [(n, nic) for n, nic in self.nichos.items() if nic.auto_gerado and nic.prototipo is not None]
        if len(nichos_auto) < 2:
            return
        fundidos = set()
        for i, (nome_a, nicho_a) in enumerate(nichos_auto):
            if nome_a in fundidos:
                continue
            for j, (nome_b, nicho_b) in enumerate(nichos_auto):
                if i >= j or nome_b in fundidos or nome_b == nome_a:
                    continue
                dist = float(np.linalg.norm(nicho_a.prototipo - nicho_b.prototipo))
                if dist < self.config.NEURO_LIMIAR_FUSAO * 1.2:
                    def tem_sufixo_num(s):
                        parts = s.rsplit("_", 1)
                        return len(parts) == 2 and parts[1].isdigit()
                    a_sem = not tem_sufixo_num(nome_a)
                    b_sem = not tem_sufixo_num(nome_b)
                    if a_sem and not b_sem:
                        alvo, vitima = nicho_a, nicho_b
                        nome_alvo, nome_vitima = nome_a, nome_b
                    elif b_sem and not a_sem:
                        alvo, vitima = nicho_b, nicho_a
                        nome_alvo, nome_vitima = nome_b, nome_a
                    else:
                        alvo = nicho_a if nicho_a.n_amostras >= nicho_b.n_amostras else nicho_b
                        vitima = nicho_b if alvo == nicho_a else nicho_a
                        nome_alvo = nome_a if alvo == nicho_a else nome_b
                        nome_vitima = nome_b if alvo == nicho_a else nome_a
                    print(f"\n   🧹 LIMPEZA: Fundindo '{nome_vitima}' -> '{nome_alvo}' (dist={dist:.2f})")
                    for _ in range(vitima.n_amostras):
                        alvo.atualizar_prototipo(vitima.prototipo)
                    for kw in vitima.palavras_chave:
                        if kw not in alvo.palavras_chave:
                            alvo.palavras_chave.append(kw)
                    alvo.historico.extend(vitima.historico)
                    path_remover = vitima._caminho()
                    if os.path.exists(path_remover):
                        os.remove(path_remover)
                    if nome_vitima in self.nichos:
                        del self.nichos[nome_vitima]
                    fundidos.add(nome_vitima)
                    alvo.salvar()
                    print(f"      Amostras finais: {alvo.n_amostras}")

    def _garantir_nicho_base(self):
        if "geral" not in self.nichos:
            self.nichos["geral"] = Nicho("geral", "Conversas gerais e diversas", self.config)
            self.nichos["geral"].salvar()
            print("   Nicho 'geral' criado")

    def _embedding(self, texto: str) -> np.ndarray:
        return self.embedder.encode(texto, convert_to_numpy=True)

    def _medir_nll(self, prompt: str, nicho: Nicho) -> float:
        """Mede 'energia' do nicho para este prompt.
        Gera ~50 tokens e mede tokens/sec MEDIO da geracao.
        """
        system = nicho.system_prompt()
        try:
            response = ollama.generate(
                model=self.config.MODELO_BASE,
                prompt=prompt,
                system=system,
                options={"temperature": 0.1, "num_predict": 50},
                stream=False
            )
            eval_count = response.get("eval_count", 0)
            eval_duration = response.get("eval_duration", 0)  # nanosec
            if eval_count >= 5 and eval_duration > 0:
                tokens_per_sec = eval_count / (eval_duration / 1e9)
                nll = max(0.5, 6.0 / (1.0 + tokens_per_sec / 30.0))
                return nll
            resposta = response.get("response", "")
            if not resposta or len(resposta) < 10:
                return self.config.ENERGIA_MAX_NLL
            return 4.0
        except Exception as e:
            print(f"   [ERRO] NLL nicho {nicho.nome}: {e}")
            return self.config.ENERGIA_MAX_NLL

    def _medir_nll_base(self, prompt: str) -> float:
        """Mede NLL do modelo base (sem system prompt de nicho)."""
        try:
            response = ollama.generate(
                model=self.config.MODELO_BASE,
                prompt=prompt,
                options={"temperature": 0.1, "num_predict": 50},
                stream=False
            )
            eval_count = response.get("eval_count", 0)
            eval_duration = response.get("eval_duration", 0)
            if eval_count >= 5 and eval_duration > 0:
                tokens_per_sec = eval_count / (eval_duration / 1e9)
                return max(0.5, 6.0 / (1.0 + tokens_per_sec / 30.0))
            return 4.0
        except Exception:
            return 4.0

    def roteamento(self, prompt: str, nlls_precalculados: Dict[str, float] = None) -> Tuple[str, float, Dict]:
        emb = self._embedding(prompt)
        # Fast path: palavras-chave
        scores_kw = {}
        for nome, nicho in self.nichos.items():
            scores_kw[nome] = nicho.match_palavras_chave(prompt)
        if scores_kw:
            melhor_kw = max(scores_kw, key=scores_kw.get)
            if scores_kw[melhor_kw] >= 0.3:
                return melhor_kw, 0.9, {"motivo": "palavras_chave", "score": scores_kw[melhor_kw]}

        # v3.2: Roteamento por energia (NLL)
        if self.config.USA_ROTEAMENTO_ENERGIA:
            nlls = {}
            for nome, nicho in self.nichos.items():
                if nicho.hibernado or nome == "geral":  # geral e fallback, nao compete
                    continue
                # Reusa NLL ja medido no _detectar_novidade (evita inconsistencia por cache)
                if nlls_precalculados and nome in nlls_precalculados:
                    nlls[nome] = nlls_precalculados[nome]
                else:
                    nlls[nome] = self._medir_nll(prompt, nicho)
            if nlls:
                nicho_vencedor = min(nlls, key=nlls.get)
                nll_vencedor = nlls[nicho_vencedor]
                sorted_nlls = sorted(nlls.values())
                if len(sorted_nlls) > 1:
                    # Confianca baseada na margem entre 1o e 2o lugar
                    margem = sorted_nlls[1] - sorted_nlls[0]
                    confianca = min(0.95, 0.5 + margem / 2.0)
                else:
                    # So 1 nicho: confianca baseada no NLL absoluto
                    # NLL < 2.0 = muito confiante, NLL > 5.0 = pouco confiante
                    confianca = max(0.3, min(0.9, 1.0 - nll_vencedor / 10.0))
                print(f"   [ENERGIA] NLLs: {', '.join(f'{n}={v:.2f}' for n,v in sorted(nlls.items(), key=lambda x: x[1]))} -> {nicho_vencedor} (conf={confianca:.2f})")
                return nicho_vencedor, confianca, {"nlls": nlls, "motivo": "energia"}
            # Se so tem geral, cai pro fallback
            return "geral", 0.5, {"motivo": "fallback_geral"}

        # Fallback: roteamento por distancia de embedding
        distancias = {}
        for nome, nicho in self.nichos.items():
            distancias[nome] = nicho.distancia(emb)
        distancias_validas = {k: v for k, v in distancias.items() if v != float("inf")}
        if distancias_validas:
            nicho_mais_proximo = min(distancias_validas, key=distancias_validas.get)
            dist_min = distancias_validas[nicho_mais_proximo]
            if dist_min > self.config.DISTANCIA_MAX_ROTEAMENTO:
                return "geral", 0.5, {"motivo": "fallback_distancia", "dist_min": dist_min}
            nicho = self.nichos[nicho_mais_proximo]
            if nicho.em_warmup():
                confianca = 0.8
            elif nicho.raio is not None and nicho.raio > 0:
                confianca = max(0, 1 - dist_min / (nicho.raio * 3))
            else:
                confianca = 0.5
            return nicho_mais_proximo, confianca, {"distancias": distancias, "motivo": "embedding"}
        return "geral", 0.5, {"motivo": "fallback_geral"}

    def _detectar_novidade(self, prompt: str, emb: np.ndarray) -> Tuple[bool, float, Dict[str, float]]:
        """Detecta novidade usando margem NLL (paper: NLL_nicho - NLL_base).
        Retorna (eh_novo, margem, nlls_medidos) — nlls_medidos para reusar no roteamento.
        """
        nlls_medidos = {}
        if not self.config.USA_NEUROGENESE:
            return False, 0.0, nlls_medidos

        if self.config.USA_ROTEAMENTO_ENERGIA:
            nll_base = self._medir_nll_base(prompt)
            nlls_nicho = {}
            for nome, nicho in self.nichos.items():
                if nicho.nome == "geral" or nicho.hibernado:
                    continue
                nll = self._medir_nll(prompt, nicho)
                nlls_nicho[nome] = nll
                nlls_medidos[nome] = nll
            if nlls_nicho:
                melhor_nll = min(nlls_nicho.values())
                margem = melhor_nll - nll_base
                if self.janela_exploracao > 0:
                    limiar_efetivo = self.config.ENERGIA_LIMIAR_MARGEM * 0.8
                    modo = "EXPLORACAO"
                else:
                    limiar_efetivo = self.config.ENERGIA_LIMIAR_MARGEM
                    modo = "normal"
                eh_novo = margem > limiar_efetivo
                # v3.2: Limiar absoluto — se o melhor nicho tem NLL razoavel (< 7.0),
                # nao e "novo" mesmo que a margem seja alta (ex: cumprimentos genericos)
                if eh_novo and melhor_nll < 7.0:
                    eh_novo = False
                    status = "familiar (NLL absoluto baixo)"
                else:
                    status = "NOVO" if eh_novo else "familiar"
                janela_str = f" [janela:{self.janela_exploracao}]" if self.janela_exploracao > 0 else ""
                print(f"   [NOVIDADE-ENERGIA] margem={margem:.3f} (melhor_nicho={melhor_nll:.3f} - base={nll_base:.3f}, limiar={limiar_efetivo}, modo={modo}){janela_str} -> {status}")
                return eh_novo, margem, nlls_medidos
            else:
                # Sem nichos para medir — se nao ha NENHUM nicho especializado,
                # e novidade (precisamos criar o primeiro!)
                nichos_especializados = [n for n in self.nichos if n != "geral"]
                if not nichos_especializados:
                    print(f"   [NOVIDADE-ENERGIA] Sem nichos especializados -> NOVO (bootstrap)")
                    return True, 999.0, nlls_medidos
                print(f"   [NOVIDADE-ENERGIA] Nichos existentes mas sem dados -> familiar")
                return False, 0.0, nlls_medidos

        # Fallback: novidade por distancia (so se energia desativada)
        distancias = []
        for nome, nicho in self.nichos.items():
            if nicho.nome == "geral":
                continue
            dist = nicho.distancia(emb)
            if dist != float("inf"):
                distancias.append(dist)
        if not distancias:
            return True, float("inf"), nlls_medidos
        dist_media = np.mean(distancias)
        if self.janela_exploracao > 0:
            limiar_efetivo = self.config.NEURO_LIMIAR_JANELA
            modo = "EXPLORACAO"
        else:
            limiar_efetivo = self.config.NEURO_LIMIAR_NOVIDADE
            modo = "normal"
        eh_novo = dist_media > limiar_efetivo
        status = "NOVO" if eh_novo else "familiar"
        janela_str = f" [janela:{self.janela_exploracao}]" if self.janela_exploracao > 0 else ""
        print(f"   [NOVIDADE] dist_media={dist_media:.2f} (limiar={limiar_efetivo}, modo={modo}){janela_str} -> {status}")
        return eh_novo, dist_media, nlls_medidos

    def _tentar_neurogenese(self) -> Optional[str]:
        if not self.config.USA_NEUROGENESE:
            return None
        if len(self.buffer_novidade) < self.config.NEURO_MIN_AMOSTRAS_NASCER:
            return None
        embeddings_buffer = np.array([emb for _, emb in self.buffer_novidade])
        prototipo_buffer = np.mean(embeddings_buffer, axis=0)
        nicho_mais_proximo = None
        dist_mais_proxima = float("inf")
        for nome, nicho in self.nichos.items():
            if nicho.prototipo is not None and nicho.auto_gerado:
                dist = float(np.linalg.norm(prototipo_buffer - nicho.prototipo))
                if dist < dist_mais_proxima:
                    dist_mais_proxima = dist
                    nicho_mais_proximo = nicho
        if nicho_mais_proximo and dist_mais_proxima < self.config.NEURO_LIMIAR_FUSAO:
            print(f"\n   🔗 FUSAO: Buffer fundido no nicho auto-gerado '{nicho_mais_proximo.nome}'!")
            print(f"      Distancia: {dist_mais_proxima:.2f} (limiar: {self.config.NEURO_LIMIAR_FUSAO})")
            for p_texto, p_emb in self.buffer_novidade:
                nicho_mais_proximo.atualizar_prototipo(p_emb)
            prompts = [p for p, _ in self.buffer_novidade]
            palavras_todas = []
            for p in prompts:
                palavras_todas.extend([t for t in p.lower().split() if len(t) > 2])
            from collections import Counter
            contagem = Counter(palavras_todas)
            novas_kw = [p for p, c in contagem.most_common(5) if p not in STOPWORDS and p not in nicho_mais_proximo.palavras_chave and len(p) > 2 and c >= 2]
            if novas_kw:
                nicho_mais_proximo.palavras_chave.extend(novas_kw)
                nicho_mais_proximo.salvar()
                print(f"      Novas palavras-chave: {', '.join(novas_kw)}")
            print(f"      Amostras totais: {nicho_mais_proximo.n_amostras}")
            nicho_fundido = nicho_mais_proximo
            nichos_removidos = []
            for nome_outro, nicho_outro in list(self.nichos.items()):
                if nome_outro == nicho_fundido.nome or nome_outro == "geral":
                    continue
                if not nicho_outro.auto_gerado:
                    continue
                if nicho_outro.prototipo is not None and nicho_fundido.prototipo is not None:
                    dist_cascata = float(np.linalg.norm(nicho_fundido.prototipo - nicho_outro.prototipo))
                    if dist_cascata < self.config.NEURO_LIMIAR_FUSAO * 1.2:
                        print(f"      🔗 CASCATA: Absorvendo nicho '{nome_outro}' (dist={dist_cascata:.2f})")
                        for _ in range(nicho_outro.n_amostras):
                            nicho_fundido.atualizar_prototipo(nicho_outro.prototipo)
                        for kw in nicho_outro.palavras_chave:
                            if kw not in nicho_fundido.palavras_chave:
                                nicho_fundido.palavras_chave.append(kw)
                        nicho_fundido.historico.extend(nicho_outro.historico)
                        path_remover = nicho_outro._caminho()
                        if os.path.exists(path_remover):
                            os.remove(path_remover)
                        del self.nichos[nome_outro]
                        nichos_removidos.append(nome_outro)
            if nichos_removidos:
                nicho_fundido.salvar()
                print(f"      Nichos absorvidos: {', '.join(nichos_removidos)}")
                print(f"      Amostras finais: {nicho_fundido.n_amostras}")
            self.buffer_novidade = []
            self.janela_exploracao = 0
            return nicho_fundido.nome
        nichos_auto = sum(1 for n in self.nichos.values() if n.auto_gerado)
        if nichos_auto >= self.config.NEURO_MAX_NICHOS:
            print(f"   [NEUROGENESE] Limite de nichos auto-gerados atingido ({self.config.NEURO_MAX_NICHOS})")
            return None
        prompts = [p for p, _ in self.buffer_novidade]
        palavras_todas = []
        bigramas_todos = []
        for p in prompts:
            tokens = [t for t in p.lower().split() if len(t) > 2]
            palavras_todas.extend(tokens)
            for i in range(len(tokens)-1):
                bigramas_todos.append(f"{tokens[i]}_{tokens[i+1]}")
        from collections import Counter
        contagem = Counter(palavras_todas)
        contagem_bigramas = Counter(bigramas_todos)
        # Palavras que aparecem pelo menos min_freq vezes
        # Buffer pequeno (bootstrap): c>=1; Buffer grande: c>=2
        min_freq = 1 if len(self.buffer_novidade) <= 4 else 2
        palavras_filtradas = [(p, c) for p, c in contagem.most_common(15) if p not in STOPWORDS and len(p) > 2 and c >= min_freq]
        bigramas_filtrados = [(b.replace("_", " "), c) for b, c in contagem_bigramas.most_common(5) if c >= 2]
        if bigramas_filtrados:
            palavras_filtradas = bigramas_filtrados + palavras_filtradas
        if not palavras_filtradas:
            return None
        nome_base = self._gerar_nome_nicho(palavras_filtradas, prompts)
        nome_nicho = nome_base
        suffix = 1
        while nome_nicho in self.nichos:
            nome_nicho = f"{nome_base}_{suffix}"
            suffix += 1
        palavras_chave = [p for p, _ in palavras_filtradas[:5]]
        descricao = f"Nicho auto-gerado a partir de {len(prompts)} prompts sobre: {', '.join(palavras_chave)}"
        nicho = Nicho(nome_nicho, descricao, self.config, palavras_chave, auto_gerado=True)
        for p_texto, p_emb in self.buffer_novidade:
            nicho.atualizar_prototipo(p_emb)
        self.nichos[nome_nicho] = nicho
        nicho.salvar()
        print(f"\n   🦎 NEUROGENESE: Nicho '{nome_nicho}' nasceu!")
        print(f"      Descricao: {descricao}")
        print(f"      Palavras-chave: {', '.join(palavras_chave)}")
        print(f"      Amostras: {nicho.n_amostras}")
        self.buffer_novidade = []
        self.janela_exploracao = 0
        return nome_nicho

    def gate(self, prompt: str, nicho_nome: str, confianca: float, usa_energia: bool, dist_media: float, nll_nicho: float = None) -> Tuple[bool, str]:
        nicho = self.nichos[nicho_nome]
        emb = self._embedding(prompt)
        dist_nicho = nicho.distancia(emb)

        # v3.2: Se roteamento por energia achou NLL excelente, confia na energia
        # NLL < 2.0 = modelo muito confiante naquele nicho → bypass de distancia
        energia_forte = (nll_nicho is not None and nll_nicho < 2.0
                         and nicho.nll_lar is not None and nll_nicho < nicho.nll_lar)

        if dist_nicho != float("inf") and not energia_forte:
            if nicho.raio is not None and nicho.raio > 0.5:
                if dist_nicho > nicho.raio * 4:
                    return False, f"surpresa_extrema (dist={dist_nicho:.1f} > 4x raio={nicho.raio:.1f})"
                if dist_nicho > self.config.GATE_LIMIAR_DISTANCIA * 1.5:
                    return False, "transmutacao_teto"
            else:
                if dist_nicho > 1.5:
                    return False, f"surpresa_extrema (dist={dist_nicho:.1f} > limiar_absoluto=1.5)"
                if dist_nicho > self.config.GATE_LIMIAR_DISTANCIA * 1.2:
                    return False, "transmutacao_teto"
        elif energia_forte:
            print(f"   [GATE] Bypass distancia por energia forte (NLL={nll_nicho:.2f} < lar={nicho.nll_lar:.2f})")

        if nicho.em_warmup():
            return True, "warmup"

        if nicho.distancias_historico and len(nicho.distancias_historico) >= 3:
            mu = float(np.mean(nicho.distancias_historico))
            sigma = float(np.std(nicho.distancias_historico)) + 1e-6
            z = abs(dist_nicho - mu) / sigma
            if z > 3.0:
                return False, f"surpresa_alta (z={z:.1f})"

        limiar = self.config.GATE_LIMIAR_CONFIANCA_ENERGIA if usa_energia else self.config.GATE_LIMIAR_CONFIANCA_CLASSICO
        if confianca < limiar:
            return False, "baixa_confianca_roteamento"

        distancias = [n.distancia(emb) for n in self.nichos.values() if not n.em_warmup() and n.nome != "geral"]
        if distancias:
            dist_media_gate = float(np.mean(distancias))
            if dist_media_gate > self.config.GATE_LIMIAR_DISTANCIA * 2:
                return False, "dominio_exotico"

        return True, "aprovado"

    def _contexto_episodio(self, nicho_nome: str, max_turnos: int = 4) -> str:
        """Formata os ultimos turnos da conversa no nicho ativo para injetar no system prompt."""
        turnos = [t for t in self.episodio if t[2] == nicho_nome]
        if not turnos:
            return ""
        recentes = turnos[-max_turnos:]
        linhas = ["\n--- Historico recente ---"]
        for user_msg, bot_msg, _ in recentes:
            linhas.append(f"User: {user_msg[:120]}")
            linhas.append(f"Bot: {bot_msg[:120]}")
        linhas.append("--- Fim do historico ---\n")
        return "\n".join(linhas)

    def conversar(self, prompt: str) -> Dict:
        self._verificar_hibernacao()
        resultado = {
            "prompt": prompt,
            "nicho": None,
            "gate_passou": False,
            "resposta": None,
            "motivo_abstencao": None,
            "memoria_relevante": [],
            "neurogenese": None
        }
        emb = self._embedding(prompt)
        eh_novo, dist_media, nlls_medidos = self._detectar_novidade(prompt, emb)
        dentro_janela = self.janela_exploracao > 0
        if self.config.USA_NEUROGENESE and (eh_novo or dentro_janela):
            self.buffer_novidade.append((prompt, emb))
            if eh_novo:
                self.janela_exploracao = self.config.NEURO_JANELA_EXPLORACAO
                print(f"   [NEUROGENESE] Novidade detectada (dist={dist_media:.2f}). Buffer: {len(self.buffer_novidade)}/{self.config.NEURO_MIN_AMOSTRAS_NASCER} | Janela: {self.janela_exploracao}")
            else:
                self.janela_exploracao -= 1
                print(f"   [NEUROGENESE] Janela (dist={dist_media:.2f}). Buffer: {len(self.buffer_novidade)}/{self.config.NEURO_MIN_AMOSTRAS_NASCER} | Restam: {self.janela_exploracao}")
            nicho_novo = self._tentar_neurogenese()
            if nicho_novo:
                resultado["neurogenese"] = nicho_novo
                nicho_nome = nicho_novo
                confianca = 0.9
                passou = True
                motivo = "neurogenese"
                resultado["nicho"] = nicho_nome
                resultado["confianca_roteamento"] = confianca
                memoria = self.memoria.buscar(emb, k=3)
                resultado["memoria_relevante"] = memoria
                nicho = self.nichos[nicho_nome]
                contexto_memoria = ""
                if memoria:
                    contexto_memoria = "\nMemorias relevantes:\n" + "\n".join([f"- {m[0][:200]}" for m in memoria])
                contexto_rag = self._contexto_rag(nicho_nome, emb)
                contexto_ep = self._contexto_episodio(nicho_nome)
                system = nicho.system_prompt() + contexto_memoria + contexto_rag + contexto_ep
                try:
                    response = ollama.generate(model=self.config.MODELO_BASE, prompt=prompt, system=system, options={"temperature": 0.7})
                    resposta = response["response"]
                except Exception as e:
                    resposta = f"Erro: {e}"
                resultado["resposta"] = resposta
                resultado["gate_passou"] = True
                resultado["motivo_abstencao"] = motivo
                nicho.historico.append({"data": datetime.now().isoformat(), "prompt": prompt, "resposta": resposta})
                # v3.2: Registra NLL no nicho para estatisticas de energia
                nll_nicho = self._medir_nll(prompt, nicho)
                nicho.atualizar_nll(nll_nicho)
                if len(resposta) > 20:
                    nicho.atualizar_prototipo(emb)
                else:
                    print(f"   [AVISO] Resposta curta ({len(resposta)} chars) — prototipo nao atualizado.")
                self.memoria.adicionar(prompt, emb, {"nicho": nicho_nome, "resposta": resposta[:500], "data": datetime.now().isoformat()})
                self.nicho_ativo = nicho_nome
                self.ultimo_nicho = nicho_nome
                nicho.ultima_atividade = datetime.now().isoformat()
                self._atualizar_memoria_longo_prazo(nicho)
                return resultado
        else:
            self.janela_exploracao = 0
        nicho_nome, confianca, info = self.roteamento(prompt, nlls_precalculados=nlls_medidos)
        resultado["nicho"] = nicho_nome
        resultado["confianca_roteamento"] = confianca
        usa_energia = self.config.USA_ENERGIA and len(self.nichos) > 1
        # v3.2: Passa NLL do roteamento por energia para o gate
        nll_nicho = info.get("nlls", {}).get(nicho_nome) if "nlls" in info else None
        passou, motivo = self.gate(prompt, nicho_nome, confianca, usa_energia, dist_media, nll_nicho=nll_nicho)
        resultado["gate_passou"] = passou
        resultado["motivo_abstencao"] = motivo
        if not passou and self.config.USA_NEUROGENESE and dist_media > self.config.NEURO_LIMIAR_NOVIDADE:
            self.buffer_novidade.append((prompt, emb))
            print(f"   [NEUROGENESE] Gate rejeitou, novidade (dist={dist_media:.2f}). Buffer: {len(self.buffer_novidade)}/{self.config.NEURO_MIN_AMOSTRAS_NASCER}")
            nicho_novo = self._tentar_neurogenese()
            if nicho_novo:
                resultado["neurogenese"] = nicho_novo
                nicho_nome = nicho_novo
                passou = True
                motivo = "neurogenese_fallback"
        if not passou:
            resultado["resposta"] = self._mensagem_abstencao(motivo)
            return resultado
        memoria = self.memoria.buscar(emb, k=3)
        resultado["memoria_relevante"] = memoria
        nicho = self.nichos[nicho_nome]
        contexto_memoria = ""
        if memoria:
            contexto_memoria = "\nMemorias relevantes:\n" + "\n".join([f"- {m[0][:200]}" for m in memoria])
        contexto_rag = self._contexto_rag(nicho_nome, emb)
        contexto_ep = self._contexto_episodio(nicho_nome)
        system = nicho.system_prompt() + contexto_memoria + contexto_rag + contexto_ep
        try:
            response = ollama.generate(model=self.config.MODELO_BASE, prompt=prompt, system=system, options={"temperature": 0.7})
            resposta = response["response"]
        except Exception as e:
            resposta = f"Erro: {e}"
        resultado["resposta"] = resposta
        nicho.historico.append({"data": datetime.now().isoformat(), "prompt": prompt, "resposta": resposta})
        # v3.2: Registra NLL no nicho (usa o que ja foi medido no roteamento por energia)
        if "nlls" in info and nicho_nome in info["nlls"]:
            nicho.atualizar_nll(info["nlls"][nicho_nome])
        else:
            nll_nicho = self._medir_nll(prompt, nicho)
            nicho.atualizar_nll(nll_nicho)
        if len(resposta) > 20:
            nicho.atualizar_prototipo(emb)
        else:
            print(f"   [AVISO] Resposta curta ({len(resposta)} chars) — prototipo nao atualizado.")
        self.memoria.adicionar(prompt, emb, {"nicho": nicho_nome, "resposta": resposta[:500], "data": datetime.now().isoformat()})
        self.nicho_ativo = nicho_nome
        self.ultimo_nicho = nicho_nome
        nicho.ultima_atividade = datetime.now().isoformat()
        self._atualizar_memoria_longo_prazo(nicho)
        self.ultima_interacao = {
            "prompt": prompt,
            "nicho": nicho_nome,
            "resposta": resposta,
            "embedding": emb,
            "confianca": confianca,
            "gate_passou": passou,
            "motivo": motivo
        }
        self.episodio.append((prompt, resposta, nicho_nome))
        if len(self.episodio) > 50:
            self.episodio = self.episodio[-40:]
        return resultado

    def _mensagem_abstencao(self, motivo: str) -> str:
        if motivo == "transmutacao_teto":
            return "[Camaleao: Isso parece pertencer a outro dominio. Nao vou arriscar uma resposta fora da minha area. Pode ser mais especifico?]"
        elif motivo == "baixa_confianca_roteamento":
            return "[Camaleao: Nao sei qual area da sua vida isso pertence. Pode me dar mais contexto?]"
        elif motivo.startswith("surpresa_alta"):
            return "[Camaleao: Esse prompt eh muito diferente do que costumo ver. Nao vou arriscar.]"
        elif motivo.startswith("surpresa_extrema"):
            return "[Camaleao: Isso esta completamente fora do meu dominio. Nao vou arriscar.]"
        elif motivo == "dominio_exotico":
            nichos = ", ".join(self.nichos.keys())
            return f"[Camaleao: Fora dos meus dominios ({nichos}). Quer criar um novo nicho?]"
        return "[Camaleao: Nao sei responder a isso ainda.]"

    def _verificar_hibernacao(self):
        if not self.config.HIBERNACAO_ATIVA:
            return
        agora = datetime.now()
        for nome, nicho in self.nichos.items():
            if nome == "geral" or nicho.hibernado:
                continue
            try:
                ultima = datetime.fromisoformat(nicho.ultima_atividade)
                dias_inativo = (agora - ultima).days
                if dias_inativo >= self.config.HIBERNACAO_DIAS:
                    nicho.hibernado = True
                    nicho.salvar()
                    print(f"   ❄️ HIBERNACAO: Nicho '{nome}' hibernado ({dias_inativo} dias inativo)")
            except:
                pass

    def _gerar_resumo_nicho(self, nicho: Nicho) -> str:
        if not nicho.historico or len(nicho.historico) < 3:
            return ""
        recentes = nicho.historico[-20:]
        texto_historico = "\n".join([f"User: {h['prompt']}\nBot: {h['resposta'][:200]}" for h in recentes])
        prompt_resumo = f"""Resuma em 3-4 frases curtas o que voce aprendeu sobre esta pessoa nestas conversas.
Foque em preferencias, fatos importantes e padroes. Seja especifico e factual.

Conversas:
{texto_historico}

Resumo:"""
        try:
            response = ollama.generate(model=self.config.MODELO_BASE, prompt=prompt_resumo, options={"temperature": 0.3})
            resumo = response["response"].strip()
            if resumo and len(resumo) > 20:
                return resumo
        except Exception as e:
            print(f"   [AVISO] Falha ao gerar resumo: {e}")
        return ""

    def _atualizar_memoria_longo_prazo(self, nicho: Nicho):
        if not self.config.MEMORIA_RESUMO_ATIVA:
            return
        nicho.conversas_desde_resumo += 1
        if nicho.conversas_desde_resumo >= self.config.MEMORIA_RESUMO_A_CADA:
            print(f"   📝 Gerando resumo para nicho '{nicho.nome}'...")
            resumo = self._gerar_resumo_nicho(nicho)
            if resumo:
                nicho.resumo_longo = resumo
                nicho.salvar()
                print(f"   ✅ Resumo: {resumo[:100]}...")
            nicho.conversas_desde_resumo = 0

    def _contexto_rag(self, nicho_nome: str, emb: np.ndarray) -> str:
        if nicho_nome not in self.documentos:
            return ""
        resultados = self.documentos[nicho_nome].buscar(emb, k=3)
        if not resultados:
            return ""
        contexto = "\n\nDocumentos relevantes:\n"
        for texto, fonte, score in resultados:
            contexto += f"\n[Fonte: {fonte} | Relevancia: {score:.2f}]\n{texto[:400]}\n"
        return contexto

    def _carregar_documentos(self):
        for nome in self.nichos:
            self.documentos[nome] = DocumentoNicho(nome, self.config)

    def adicionar_documento(self, nicho_nome: str, caminho: str) -> int:
        if nicho_nome not in self.nichos:
            print(f"   [ERRO] Nicho '{nicho_nome}' nao existe.")
            return 0
        if nicho_nome not in self.documentos:
            self.documentos[nicho_nome] = DocumentoNicho(nicho_nome, self.config)
        pasta_docs = self.documentos[nicho_nome].pasta_docs
        destino = os.path.join(pasta_docs, os.path.basename(caminho))
        if os.path.exists(caminho) and not os.path.exists(destino):
            shutil.copy2(caminho, destino)
            caminho = destino
        n_chunks = self.documentos[nicho_nome].adicionar_documento(caminho, self.embedder)
        print(f"   📄 Documento indexado: {os.path.basename(caminho)} ({n_chunks} chunks) -> nicho '{nicho_nome}'")
        return n_chunks

    def listar_documentos(self, nicho_nome: str) -> List[str]:
        if nicho_nome not in self.documentos:
            return []
        return self.documentos[nicho_nome].listar()

    def limpar_documentos(self, nicho_nome: str):
        if nicho_nome in self.documentos:
            self.documentos[nicho_nome].limpar()
            print(f"   🗑️ Documentos do nicho '{nicho_nome}' removidos.")

    def criar_nicho(self, nome: str, descricao: str, palavras_chave: str = ""):
        if nome in self.nichos:
            print(f"Nicho '{nome}' ja existe.")
            return
        kw_list = [p.strip() for p in palavras_chave.split(",") if p.strip()]
        if nome not in kw_list:
            kw_list.append(nome)
        # Extrair palavras da descricao como palavras-chave implicitas
        descricao_tokens = [t.lower() for t in re.findall(r"[a-zA-Zà-úÀ-Ú]+", descricao) if len(t) > 3]
        for token in descricao_tokens:
            if token not in STOPWORDS and token not in kw_list and len(token) > 3:
                kw_list.append(token)
                if len(kw_list) >= 8:
                    break
        self.nichos[nome] = Nicho(nome, descricao, self.config, kw_list)
        texto_inicial = f"{nome}. {descricao}. {', '.join(kw_list)}"
        try:
            emb_inicial = self._embedding(texto_inicial)
            self.nichos[nome].atualizar_prototipo(emb_inicial, forcar=True)
        except Exception as e:
            print(f"   [AVISO] Nao consegui gerar prototipo inicial: {e}")
        self.nichos[nome].salvar()
        print(f"Nicho criado: {nome} — {descricao}")
        if kw_list:
            print(f"   Palavras-chave: {', '.join(kw_list)}")

    def renomear_nicho(self, nome_antigo: str, nome_novo: str):
        if nome_antigo not in self.nichos:
            print(f"Nicho '{nome_antigo}' nao existe.")
            return
        if nome_novo in self.nichos:
            print(f"Nicho '{nome_novo}' ja existe.")
            return
        nicho = self.nichos[nome_antigo]
        nicho.nome = nome_novo
        path_antigo = os.path.join(self.config.PASTA_NICHOS, f"{nome_antigo}.json")
        if os.path.exists(path_antigo):
            os.remove(path_antigo)
        nicho.salvar()
        self.nichos[nome_novo] = nicho
        del self.nichos[nome_antigo]
        if self.ultimo_nicho == nome_antigo:
            self.ultimo_nicho = nome_novo
        if self.nicho_ativo == nome_antigo:
            self.nicho_ativo = nome_novo
        print(f"Nicho renomeado: '{nome_antigo}' -> '{nome_novo}'")

    def diagnostico(self):
        print("\n" + "="*60)
        print("DIAGNOSTICO DO CAMALEAO PESSOAL v3.2.0")
        print("="*60)
        total_conversas = sum(len(n.historico) for n in self.nichos.values())
        if total_conversas == 0:
            print("Sem conversas ainda para diagnosticar.")
            return
        print(f"\n📊 Estatisticas Gerais:")
        print(f"   Total de conversas: {total_conversas}")
        print(f"   Nichos: {len(self.nichos)} (auto-gerados: {sum(1 for n in self.nichos.values() if n.auto_gerado)})")
        print(f"   Memorias: {len(self.memoria.textos)}")
        print(f"   Buffer de novidade: {len(self.buffer_novidade)}")
        print(f"\n🦎 Nichos:")
        for nome, nicho in sorted(self.nichos.items(), key=lambda x: len(x[1].historico), reverse=True):
            status = "❄️ HIBERNADO" if nicho.hibernado else "🟢 ATIVO" if not nicho.em_warmup() else "🟡 WARMUP"
            dias = ""
            if not nicho.hibernado and nicho.ultima_atividade:
                try:
                    dias = f" (ultima: {(datetime.now() - datetime.fromisoformat(nicho.ultima_atividade)).days}d)"
                except:
                    pass
            cons_str = " | 🔒" if nicho.consolidado else ""
            resumo_str = " | 📝" if nicho.resumo_longo else ""
            var_str = ""
            if len(nicho.distancias_historico) >= 3:
                var = float(np.var(nicho.distancias_historico[-5:]))
                var_str = f" | var={var:.2f}"
            nll_str = ""
            if nicho.nll_lar is not None:
                nll_str = f" | NLL_lar={nicho.nll_lar:.2f}"
            print(f"   {status} {nome}: {len(nicho.historico)} conversas, {nicho.n_amostras} amostras{cons_str}{resumo_str}{var_str}{nll_str}{dias}")
            if nicho.resumo_longo:
                print(f"      📝 {nicho.resumo_longo[:80]}...")
        janela = self.config.DIAGNOSTICO_JANELA
        todas_conversas = []
        for nome, nicho in self.nichos.items():
            for h in nicho.historico:
                todas_conversas.append((h["data"], nome, h["prompt"]))
        if len(todas_conversas) >= 5:
            todas_conversas.sort(key=lambda x: x[0])
            recentes = todas_conversas[-janela:]
            from collections import Counter
            dist = Counter([c[1] for c in recentes])
            print(f"\n📈 Roteamento (ultimas {len(recentes)} mensagens):")
            for nicho_nome, count in dist.most_common():
                pct = count / len(recentes) * 100
                bar = "█" * int(pct / 5)
                print(f"   {nicho_nome:12s}: {bar} {count}/{len(recentes)} ({pct:.0f}%)")
        print("="*60)

    def diagnostico_energia(self):
        """Diagnostico especifico do roteamento por energia (NLL)."""
        print("\n" + "="*60)
        print("DIAGNOSTICO DE ENERGIA (Roteamento por NLL)")
        print("="*60)
        nichos_com_nll = [(n, nic) for n, nic in self.nichos.items() if nic.nll_historico]
        if not nichos_com_nll:
            print("Sem dados de NLL ainda. Faca algumas conversas primeiro.")
            return
        print(f"\n📊 Roteamento por energia: {'ATIVADO' if self.config.USA_ROTEAMENTO_ENERGIA else 'DESATIVADO'}")
        print(f"   Limiar de margem: {self.config.ENERGIA_LIMIAR_MARGEM}")
        print(f"\n⚡ Energia por nicho:")
        for nome, nicho in sorted(nichos_com_nll, key=lambda x: x[1].nll_lar or 999):
            nlls = nicho.nll_historico[-20:]
            mu = np.mean(nlls)
            sigma = np.std(nlls)
            lar = nicho.nll_lar or mu
            print(f"   {nome:15s}: lar={lar:.2f} | media={mu:.2f} | std={sigma:.2f} | amostras={len(nlls)}")
        # Mostra separabilidade
        if len(nichos_com_nll) >= 2:
            lars = [(n, nic.nll_lar) for n, nic in nichos_com_nll if nic.nll_lar is not None]
            if len(lars) >= 2:
                lars.sort(key=lambda x: x[1])
                menor, maior = lars[0], lars[-1]
                gap = maior[1] - menor[1]
                print(f"\n📐 Separabilidade:")
                print(f"   Melhor nicho: {menor[0]} (lar={menor[1]:.2f})")
                print(f"   Pior nicho:   {maior[0]} (lar={maior[1]:.2f})")
                print(f"   Gap: {gap:.2f} nats")
                if gap > 1.0:
                    print(f"   ✅ Nichos bem separados energeticamente")
                elif gap > 0.3:
                    print(f"   ⚠️ Separacao moderada")
                else:
                    print(f"   ❌ Nichos proximos — roteamento por energia pode ser ruidoso")
        print("="*60)

    def exportar(self, caminho_destino: str = None) -> str:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if caminho_destino is None:
            caminho_destino = f"camaleao_backup_{timestamp}.zip"
        for nicho in self.nichos.values():
            nicho.salvar()
        self.memoria.salvar()
        with zipfile.ZipFile(caminho_destino, "w", zipfile.ZIP_DEFLATED) as zf:
            for arquivo in glob.glob(os.path.join(self.config.PASTA_NICHOS, "*.json")):
                zf.write(arquivo, os.path.join("nichos", os.path.basename(arquivo)))
            for arquivo in glob.glob(os.path.join(self.config.PASTA_MEMORIA, "*")):
                zf.write(arquivo, os.path.join("memoria", os.path.basename(arquivo)))
            for arquivo in glob.glob(os.path.join(self.config.PASTA_DADOS, "*")):
                if os.path.isfile(arquivo):
                    zf.write(arquivo, os.path.join("dados", os.path.basename(arquivo)))
            config_export = {
                "version": "3.2.0",
                "data_export": timestamp,
                "nichos": list(self.nichos.keys()),
                "nichos_auto": [n for n, nic in self.nichos.items() if nic.auto_gerado],
                "total_memorias": len(self.memoria.textos)
            }
            zf.writestr("config.json", json.dumps(config_export, indent=2, ensure_ascii=False))
        tamanho = os.path.getsize(caminho_destino) / 1024
        print(f"\n   💾 BACKUP: '{caminho_destino}'")
        print(f"      Tamanho: {tamanho:.1f} KB")
        print(f"      Nichos: {len(self.nichos)}")
        print(f"      Memorias: {len(self.memoria.textos)}")
        return caminho_destino

    def importar(self, caminho_zip: str) -> bool:
        if not os.path.exists(caminho_zip):
            print(f"   [ERRO] Arquivo nao encontrado: {caminho_zip}")
            return False
        print(f"\n   📥 IMPORTANDO: '{caminho_zip}'")
        with zipfile.ZipFile(caminho_zip, "r") as zf:
            arquivos = zf.namelist()
            tem_nichos = any(a.startswith("nichos/") for a in arquivos)
            if not tem_nichos:
                print("   [ERRO] ZIP invalido: nenhum nicho encontrado")
                return False
            backup_auto = f"camaleao_auto_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip"
            self.exportar(backup_auto)
            print(f"   (Backup automatico: {backup_auto})")
            for pasta in [self.config.PASTA_NICHOS, self.config.PASTA_MEMORIA, self.config.PASTA_DADOS]:
                if os.path.exists(pasta):
                    shutil.rmtree(pasta)
                os.makedirs(pasta, exist_ok=True)
            zf.extractall(".")
        self.nichos = {}
        self._carregar_nichos()
        self._garantir_nicho_base()
        self.memoria = MemoriaVetorial(self.config.PASTA_MEMORIA)
        self.buffer_novidade = []
        self.janela_exploracao = 0
        print(f"   ✅ Importado! Nichos: {len(self.nichos)}, Memorias: {len(self.memoria.textos)}")
        return True

    def iniciar_web(self, porta=8080):
        html_path = os.path.join(os.path.dirname(__file__) if __file__ else ".", "camaleao_web.html")
        if not os.path.exists(html_path):
            self._gerar_html_web(html_path)
        organismo = self
        class CamaleaoHandler(SimpleHTTPRequestHandler):
            def do_GET(self):
                if self.path == "/" or self.path == "/camaleao_web.html":
                    self.path = "/camaleao_web.html"
                    return super().do_GET()
                elif self.path == "/api/status":
                    self._json_response({
                        "nichos": list(organismo.nichos.keys()),
                        "nicho_ativo": organismo.nicho_ativo,
                        "memorias": len(organismo.memoria.textos),
                        "neurogenese": organismo.config.USA_NEUROGENESE,
                        "energia": organismo.config.USA_ENERGIA,
                        "contexto": organismo.config.USA_CONTEXTO
                    })
                elif self.path == "/api/nichos":
                    dados = {}
                    for nome, nicho in organismo.nichos.items():
                        docs = organismo.documentos.get(nome, DocumentoNicho(nome, organismo.config)).listar() if hasattr(organismo, "documentos") and nome in organismo.documentos else []
                        dados[nome] = {
                            "amostras": nicho.n_amostras,
                            "historico": len(nicho.historico),
                            "hibernado": nicho.hibernado,
                            "auto": nicho.auto_gerado,
                            "resumo": nicho.resumo_longo[:100] if nicho.resumo_longo else "",
                            "documentos": docs,
                            "nll_lar": round(nicho.nll_lar, 3) if nicho.nll_lar is not None else None,
                            "nll_amostras": len(nicho.nll_historico)
                        }
                    self._json_response(dados)
                elif self.path == "/api/energia":
                    nichos_nll = {}
                    for nome, nicho in organismo.nichos.items():
                        if nicho.nll_historico:
                            nlls = nicho.nll_historico[-20:]
                            nichos_nll[nome] = {
                                "lar": round(nicho.nll_lar, 3) if nicho.nll_lar else None,
                                "media": round(float(np.mean(nlls)), 3),
                                "std": round(float(np.std(nlls)), 3),
                                "amostras": len(nlls)
                            }
                    self._json_response({
                        "roteamento_energia": organismo.config.USA_ROTEAMENTO_ENERGIA,
                        "limiar_margem": organismo.config.ENERGIA_LIMIAR_MARGEM,
                        "nichos": nichos_nll
                    })
                else:
                    self.send_error(404)
            def do_OPTIONS(self):
                self.send_response(200)
                self.send_header("Access-Control-Allow-Origin", "*")
                self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
                self.send_header("Access-Control-Allow-Headers", "Content-Type")
                self.end_headers()
            def do_POST(self):
                content_length = int(self.headers.get("Content-Length", 0))
                post_data = self.rfile.read(content_length).decode("utf-8")
                data = json.loads(post_data) if post_data else {}
                if self.path == "/api/conversar":
                    prompt = data.get("prompt", "")
                    if prompt:
                        resultado = organismo.conversar(prompt)
                        nicho_nome = resultado.get("nicho")
                        nicho_obj = organismo.nichos.get(nicho_nome)
                        self._json_response({
                            "nicho": nicho_nome,
                            "resposta": resultado.get("resposta"),
                            "confianca": resultado.get("confianca_roteamento", 0),
                            "gate_passou": resultado.get("gate_passou"),
                            "neurogenese": resultado.get("neurogenese"),
                            "memoria": len(resultado.get("memoria_relevante", [])),
                            "nll_lar": round(nicho_obj.nll_lar, 3) if nicho_obj and nicho_obj.nll_lar else None
                        })
                    else:
                        self._json_response({"erro": "prompt vazio"}, 400)
                elif self.path == "/api/diagnostico":
                    total = sum(len(n.historico) for n in organismo.nichos.values())
                    dist = {}
                    for nome, nicho in organismo.nichos.items():
                        dist[nome] = len(nicho.historico)
                    self._json_response({
                        "total_conversas": total,
                        "nichos_count": len(organismo.nichos),
                        "memorias": len(organismo.memoria.textos),
                        "distribuicao": dist
                    })
                elif self.path == "/api/criar_nicho":
                    nome = data.get("nome", "").strip()
                    desc = data.get("descricao", "").strip()
                    kw = data.get("palavras_chave", "").strip()
                    if not nome:
                        self._json_response({"erro": "Nome obrigatorio"}, 400)
                    elif nome in organismo.nichos:
                        self._json_response({"erro": f"Nicho '{nome}' ja existe"}, 400)
                    else:
                        organismo.criar_nicho(nome, desc, kw)
                        self._json_response({"ok": True, "nome": nome})
                elif self.path == "/api/exportar":
                    try:
                        caminho = organismo.exportar()
                        self._json_response({"ok": True, "caminho": caminho, "mensagem": f"Backup criado: {caminho}"})
                    except Exception as e:
                        self._json_response({"erro": str(e)}, 500)
                else:
                    self.send_error(404)
            def _json_response(self, data, status=200):
                self.send_response(status)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                self.wfile.write(json.dumps(data, ensure_ascii=False).encode("utf-8"))
            def log_message(self, format, *args):
                pass
        if os.path.exists(html_path):
            destino = os.path.abspath("camaleao_web.html")
            if os.path.abspath(html_path) != destino:
                shutil.copy2(html_path, destino)
        handler = CamaleaoHandler
        with socketserver.TCPServer(("", porta), handler) as httpd:
            print(f"\n   🌐 SERVIDOR WEB: http://localhost:{porta}")
            print("   Abrindo navegador...")
            webbrowser.open(f"http://localhost:{porta}")
            print("   Pressione Ctrl+C para parar.\n")
            try:
                httpd.serve_forever()
            except KeyboardInterrupt:
                print("\n   Servidor web encerrado.")

    # Palavras que nao sao stopwords mas nao devem virar nome de nicho
    PALAVRAS_GENERICAS = {
        "ajuda","ajudar","pergunta","perguntar","duvida","duvidas","problema","problemas",
        "questao","questoes","assunto","assuntos","topico","topicos","exemplo","exemplos",
        "explicar","explicacao","ensinar","ensino","informacao","informacoes",
        "resposta","responder","falar","conversar","conversa","pessoal","pessoa",
        "gostaria","quero","preciso","precisa","pode","podem","vamos","fazer",
        "algo","alguma","algumas","qualquer","nenhuma","outra","outro",
        "basico","basica","simples","facil","dificil","rapido","melhor","pior",
        "primeiro","segundo","terceiro","novo","nova","antigo","antiga",
        "grande","pequeno","alto","baixo","forte","fraco",
        "mental","gradual","gradualmente","total","totalmente","completo",
        "possivel","impossivel","necessario","importante","util","interessante",
        "realmente","certamente","provavelmente","normalmente","geralmente",
    }

    def _gerar_nome_nicho(self, palavras_filtradas, prompts):
        """Gera um nome de nicho legivel a partir das palavras filtradas."""
        TRADUCOES = {
            "cachorro": "dogs", "cachorros": "dogs", "golden": "dogs", "retriever": "dogs",
            "racao": "pet_food", "filhote": "puppies", "gato": "cats", "gatos": "cats",
            "futebol": "football", "copa": "football", "jogador": "football", "time": "football",
            "musica": "music", "musico": "music", "banda": "music", "album": "music",
            "jogo": "gaming", "jogos": "gaming", "video_game": "gaming", "game": "gaming",
            "comida": "food", "receita": "food", "restaurante": "food", "culinaria": "food",
            "saude": "health", "doenca": "health", "remedio": "health", "medico": "health",
            "programacao": "coding", "programa": "coding", "codigo": "coding", "python": "coding",
            "viagem": "travel", "hotel": "travel", "passeio": "travel", "ferias": "travel",
            "carro": "cars", "carros": "cars", "motor": "cars", "veiculo": "cars",
            "livro": "books", "livros": "books", "leitura": "books", "romance": "books",
            "filme": "movies", "filmes": "movies", "cinema": "movies", "serie": "movies",
            "esporte": "sports", "academia": "fitness", "treino": "fitness", "exercicio": "fitness",
            "trabalho": "work", "emprego": "work", "carreira": "work", "empresa": "work",
            "escola": "education", "faculdade": "education", "curso": "education", "estudo": "education",
            "dinheiro": "finance", "investimento": "finance", "banco": "finance", "economia": "finance",
            "casa": "home", "apartamento": "home", "decoracao": "home", "moveis": "home",
            "roupa": "fashion", "moda": "fashion", "tenis": "fashion", "sapato": "fashion",
            "politica": "politics", "governo": "politics", "eleicao": "politics",
            "tecnologia": "tech", "computador": "tech", "celular": "tech", "internet": "tech",
            "natureza": "nature", "animais": "animals", "plantas": "plants", "meio_ambiente": "nature",
            "namoro": "dating", "relacionamento": "relationships", "amor": "relationships",
            "matematica": "math", "matematico": "math", "conta": "math", "contas": "math",
            "probabilidade": "probability", "estatistica": "statistics",
            "calculo": "math", "calculos": "math", "derivada": "calculus", "integral": "calculus",
            "derivada": "calculus", "integral": "calculus", "equacao": "math",
            "caixa": "retail", "loja": "retail", "mercado": "retail", "supermercado": "retail",
            "ia": "ai", "inteligencia": "ai", "artificial": "ai", "machine_learning": "ai",
            "deep_learning": "ai", "rede": "ai", "neural": "ai", "llm": "ai", "gpt": "ai",
            "programacao": "coding", "python": "coding", "javascript": "coding", "codigo": "coding",
        }
        # Filtra palavras genericas e stopwords
        palavras_validas = [(p, c) for p, c in palavras_filtradas
                           if p.lower() not in STOPWORDS and p.lower() not in self.PALAVRAS_GENERICAS and len(p) > 2]
        if not palavras_validas:
            # Fallback: extrai substantivos dos prompts mais longos
            for p in sorted(prompts, key=len, reverse=True)[:3]:
                tokens = [t.lower() for t in re.findall(r"[a-zA-Zà-úÀ-Ú]{4,}", p)
                          if t.lower() not in STOPWORDS and t.lower() not in self.PALAVRAS_GENERICAS]
                if tokens:
                    palavras_validas = [(t, 1) for t in tokens[:2]]
                    break
        if not palavras_validas:
            # Ultimo fallback: numero sequencial
            return f"Topic{len(prompts)}"
        # Traduz e monta nome
        nomes = []
        for palavra, contagem in palavras_validas[:3]:
            p = palavra.lower().replace(" ", "_").replace("-", "_")
            if p in TRADUCOES:
                nomes.append(TRADUCOES[p])
            elif len(p) > 3:
                nomes.append(p)
        if not nomes:
            return f"Topic{len(prompts)}"
        # Title case
        nome = "_".join(nomes[:2])
        nome = nome.replace("_", "").title().replace(" ", "_")
        return nome[:20]

    def _gerar_html_web(self, path: str):
        html = """<!DOCTYPE html>
<html lang="pt-BR">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Camaleao Pessoal</title>
<style>
*{margin:0;padding:0;box-sizing:border-box;font-family:'Segoe UI',system-ui,sans-serif}
body{background:#0f172a;color:#e2e8f0;height:100vh;display:flex;overflow:hidden}
.sidebar{width:260px;background:#1e293b;border-right:1px solid #334155;display:flex;flex-direction:column}
.sidebar-header{padding:20px;border-bottom:1px solid #334155}
.sidebar-header h1{font-size:1.1rem;color:#10b981;display:flex;align-items:center;gap:8px}
.sidebar-header .version{font-size:.7rem;color:#64748b;margin-top:4px}
.nichos-list{flex:1;overflow-y:auto;padding:10px}
.nicho-item{padding:10px 12px;margin:4px 0;border-radius:8px;cursor:pointer;transition:.2s;display:flex;align-items:center;gap:8px;font-size:.85rem}
.nicho-item:hover{background:#334155}
.nicho-item.active{background:#10b98122;border:1px solid #10b981}
.nicho-item.hibernado{opacity:.5}
.nicho-item .badge{width:8px;height:8px;border-radius:50%;background:#10b981}
.nicho-item .badge.auto{background:#f59e0b}
.nicho-item .badge.hibernado{background:#64748b}
.sidebar-footer{padding:12px;border-top:1px solid #334155;font-size:.75rem;color:#64748b;text-align:center}
.main{flex:1;display:flex;flex-direction:column}
.chat-header{padding:16px 20px;border-bottom:1px solid #334155;display:flex;align-items:center;justify-content:space-between}
.chat-header .info{display:flex;align-items:center;gap:12px}
.chat-header .nicho-atual{padding:4px 12px;background:#10b98122;color:#10b981;border-radius:20px;font-size:.8rem;font-weight:600}
.chat-header .status{font-size:.75rem;color:#64748b}
.chat-messages{flex:1;overflow-y:auto;padding:20px;display:flex;flex-direction:column;gap:12px}
.message{max-width:75%;padding:12px 16px;border-radius:16px;font-size:.9rem;line-height:1.5;animation:fadeIn .3s}
.message.user{align-self:flex-end;background:#10b981;color:#fff;border-bottom-right-radius:4px}
.message.bot{align-self:flex-start;background:#1e293b;border:1px solid #334155;border-bottom-left-radius:4px}
.message .meta{font-size:.7rem;opacity:.6;margin-top:6px}
.message .neuro{color:#f59e0b;font-size:.75rem;margin-top:4px}
.chat-input{padding:16px 20px;border-top:1px solid #334155;display:flex;gap:10px}
.chat-input input{flex:1;padding:12px 16px;border-radius:24px;border:1px solid #334155;background:#1e293b;color:#e2e8f0;font-size:.9rem;outline:none}
.chat-input input:focus{border-color:#10b981}
.chat-input button{padding:12px 20px;border-radius:24px;border:none;background:#10b981;color:#fff;font-weight:600;cursor:pointer;transition:.2s}
.chat-input button:hover{background:#059669}
.chat-input button:disabled{opacity:.5;cursor:not-allowed}
.diagnostico-btn{padding:8px 14px;border-radius:8px;border:1px solid #334155;background:#1e293b;color:#94a3b8;font-size:.8rem;cursor:pointer}
.diagnostico-btn:hover{background:#334155}
.modal{display:none;position:fixed;top:0;left:0;width:100%;height:100%;background:#000000aa;z-index:100;justify-content:center;align-items:center}
.modal.active{display:flex}
.modal-content{background:#1e293b;border:1px solid #334155;border-radius:12px;width:500px;max-height:80vh;overflow-y:auto;padding:20px}
.modal-header{display:flex;justify-content:space-between;align-items:center;margin-bottom:16px}
.modal-header h2{font-size:1.1rem;color:#10b981}
.modal-close{cursor:pointer;font-size:1.5rem;color:#64748b}
.modal-close:hover{color:#e2e8f0}
.stat-row{display:flex;justify-content:space-between;padding:8px 0;border-bottom:1px solid #334155;font-size:.85rem}
.stat-bar{height:20px;background:#334155;border-radius:4px;overflow:hidden;margin:4px 0}
.stat-bar-fill{height:100%;background:#10b981;border-radius:4px;transition:width .5s}
@keyframes fadeIn{from{opacity:0;transform:translateY(10px)}to{opacity:1;transform:translateY(0)}}
.typing{display:flex;gap:4px;padding:12px 16px}
.typing span{width:8px;height:8px;background:#64748b;border-radius:50%;animation:typing 1.4s infinite}
.typing span:nth-child(2){animation-delay:.2s}
.typing span:nth-child(3){animation-delay:.4s}
@keyframes typing{0%,60%,100%{transform:translateY(0)}30%{transform:translateY(-10px)}}
</style>
</head>
<body>
<div class="sidebar">
  <div class="sidebar-header">
    <h1>🦎 Camaleao</h1>
    <div class="version">Pessoal v3.2.0 — 100% local</div>
  </div>
  <div class="nichos-list" id="nichosList"></div>
  <div class="sidebar-footer">
    <div id="footerInfo">Carregando...</div>
  </div>
</div>
<div class="main">
  <div class="chat-header">
    <div class="info">
      <div class="nicho-atual" id="nichoAtual">geral</div>
      <div class="status" id="statusInfo">Pronto</div>
    </div>
    <button class="diagnostico-btn" onclick="abrirDiagnostico()">📊 Diagnostico</button>
  </div>
  <div class="chat-messages" id="chatMessages"></div>
  <div class="chat-input">
    <input type="text" id="inputPrompt" placeholder="Digite sua mensagem..." onkeypress="if(event.key==='Enter')enviar()">
    <button id="btnEnviar" onclick="enviar()">Enviar</button>
  </div>
</div>
<div class="modal" id="modalDiagnostico">
  <div class="modal-content">
    <div class="modal-header">
      <h2>📊 Diagnostico do Organismo</h2>
      <span class="modal-close" onclick="fecharDiagnostico()">&times;</span>
    </div>
    <div id="conteudoDiagnostico"></div>
  </div>
</div>
<script>
let nichoAtual = 'geral';
let carregando = false;
async function carregarNichos() {
  try {
    const r = await fetch('/api/nichos');
    const dados = await r.json();
    const list = document.getElementById('nichosList');
    list.innerHTML = '';
    for (const [nome, info] of Object.entries(dados)) {
      const div = document.createElement('div');
      div.className = 'nicho-item' + (nome === nichoAtual ? ' active' : '') + (info.hibernado ? ' hibernado' : '');
      const badgeClass = info.hibernado ? 'hibernado' : info.auto ? 'auto' : '';
      div.innerHTML = `<span class="badge ${badgeClass}"></span><span>${nome}</span><span style="margin-left:auto;font-size:.7rem;color:#64748b">${info.historico}</span>`;
      div.onclick = () => { nichoAtual = nome; atualizarUI(); };
      list.appendChild(div);
    }
    document.getElementById('footerInfo').textContent = Object.keys(dados).length + ' nichos carregados';
  } catch(e) { console.error(e); }
}
function atualizarUI() {
  document.getElementById('nichoAtual').textContent = nichoAtual;
  carregarNichos();
}
function adicionarMensagem(texto, tipo, meta='', neuro='') {
  const div = document.createElement('div');
  div.className = 'message ' + tipo;
  let html = texto.replace(/\n/g, '<br>');
  if (meta) html += `<div class="meta">${meta}</div>`;
  if (neuro) html += `<div class="neuro">${neuro}</div>`;
  div.innerHTML = html;
  document.getElementById('chatMessages').appendChild(div);
  document.getElementById('chatMessages').scrollTop = 999999;
}
function mostrarTyping() {
  const div = document.createElement('div');
  div.className = 'message bot typing';
  div.id = 'typing';
  div.innerHTML = '<span></span><span></span><span></span>';
  document.getElementById('chatMessages').appendChild(div);
  document.getElementById('chatMessages').scrollTop = 999999;
}
function removerTyping() {
  const t = document.getElementById('typing');
  if (t) t.remove();
}
async function enviar() {
  const input = document.getElementById('inputPrompt');
  const prompt = input.value.trim();
  if (!prompt || carregando) return;
  input.value = '';
  carregando = true;
  document.getElementById('btnEnviar').disabled = true;
  adicionarMensagem(prompt, 'user');
  mostrarTyping();
  document.getElementById('statusInfo').textContent = 'Pensando...';
  try {
    const r = await fetch('/api/conversar', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({prompt})
    });
    const dados = await r.json();
    removerTyping();
    const meta = `Nicho: ${dados.nicho} | Conf: ${(dados.confianca||0).toFixed(2)}`;
    const neuro = dados.neurogenese ? `🦎 Neurogenese: ${dados.neurogenese}` : '';
    adicionarMensagem(dados.resposta, 'bot', meta, neuro);
    if (dados.nicho) nichoAtual = dados.nicho;
    carregarNichos();
  } catch(e) {
    removerTyping();
    adicionarMensagem('Erro de comunicacao com o organismo.', 'bot');
  }
  carregando = false;
  document.getElementById('btnEnviar').disabled = false;
  document.getElementById('statusInfo').textContent = 'Pronto';
}
async function abrirDiagnostico() {
  try {
    const r = await fetch('/api/diagnostico');
    const d = await r.json();
    let html = '';
    html += `<div class="stat-row"><span>Total de conversas</span><span><b>${d.total_conversas}</b></span></div>`;
    html += `<div class="stat-row"><span>Nichos</span><span><b>${d.nichos_count}</b></span></div>`;
    html += `<div class="stat-row"><span>Memorias</span><span><b>${d.memorias}</b></span></div>`;
    html += '<div style="margin-top:12px;font-size:.85rem;color:#94a3b8">Distribuicao de conversas:</div>';
    const total = d.total_conversas || 1;
    for (const [nome, count] of Object.entries(d.distribuicao)) {
      const pct = (count / total * 100).toFixed(0);
      html += `<div style="margin:6px 0;font-size:.8rem">${nome} (${count})</div>`;
      html += `<div class="stat-bar"><div class="stat-bar-fill" style="width:${pct}%"></div></div>`;
    }
    document.getElementById('conteudoDiagnostico').innerHTML = html;
    document.getElementById('modalDiagnostico').classList.add('active');
  } catch(e) { console.error(e); }
}
function fecharDiagnostico() {
  document.getElementById('modalDiagnostico').classList.remove('active');
}
carregarNichos();
setInterval(carregarNichos, 5000);
</script>
</body>
</html>"""
        with open(path, "w", encoding="utf-8") as f:
            f.write(html)
        print(f"   🌐 Interface web gerada: {path}")

    def status(self):
        print("\n" + "="*50)
        print("DIAGNOSTICO DO CAMALEAO PESSOAL")
        print("="*50)
        print(f"Nichos: {len(self.nichos)}")
        print(f"Roteamento: {'ENERGIA (NLL)' if self.config.USA_ENERGIA else 'CLASSICO'}")
        print(f"Contexto: {'ATIVO' if self.config.USA_CONTEXTO else 'INATIVO'}")
        print(f"Neurogenese: {'ATIVA' if self.config.USA_NEUROGENESE else 'INATIVA'}")
        if self.ultimo_nicho:
            print(f"Ultimo nicho: {self.ultimo_nicho}")
        print(f"Buffer de novidade: {len(self.buffer_novidade)}/{self.config.NEURO_MIN_AMOSTRAS_NASCER}")
        for nome, nicho in self.nichos.items():
            if nicho.hibernado:
                status = "HIBERNADO"
            elif nicho.em_warmup():
                status = "NOVO"
            elif nicho.energia > 0.7:
                status = "OK"
            else:
                status = "ATENCAO"
            auto = " [AUTO]" if nicho.auto_gerado else " [MANUAL]"
            raio_str = f"{nicho.raio:.3f}" if nicho.raio is not None else "N/A"
            kw_str = f" | KW: {nicho.palavras_chave}" if nicho.palavras_chave else ""
            cons_flag = " | 🔒" if nicho.consolidado else ""
            resumo_flag = " | 📝" if nicho.resumo_longo else ""
            print(f"  [{status}]{auto} {nome}: {nicho.n_amostras} amostras, raio={raio_str}{kw_str}{cons_flag}{resumo_flag}")
        print(f"Memorias: {len(self.memoria.textos)}")
        print("="*50)

    def feedback_bom(self):
        """Reforca o nicho da ultima interacao: puxa o prototipo para o embedding do prompt."""
        if not self.ultima_interacao:
            return {"erro": "Nenhuma interacao recente. Converse primeiro."}
        nicho_nome = self.ultima_interacao["nicho"]
        if nicho_nome == "geral":
            return {"erro": "Nicho 'geral' nao aprende com feedback."}
        emb = self.ultima_interacao["embedding"]
        nicho = self.nichos.get(nicho_nome)
        if not nicho:
            return {"erro": f"Nicho '{nicho_nome}' nao existe mais."}
        nicho.atualizar_prototipo(emb, forcar=True)
        nicho.energia = min(2.0, nicho.energia + 0.1)
        nicho.salvar()
        return {
            "nicho": nicho_nome,
            "energia": nicho.energia,
            "acao": "reforco_positivo",
            "amostras": nicho.n_amostras
        }

    def feedback_ruim(self, correcao: str = None):
        """
        Push negativo no nicho da ultima interacao: afasta o prototipo do embedding do prompt.
        Penaliza energia e aumenta o raio. Se correcao fornecida, armazena como memoria factual.
        """
        if not self.ultima_interacao:
            return {"erro": "Nenhuma interacao recente. Converse primeiro."}
        nicho_nome = self.ultima_interacao["nicho"]
        if nicho_nome == "geral":
            return {"erro": "Nicho 'geral' nao aprende com feedback."}
        emb = self.ultima_interacao["embedding"]
        nicho = self.nichos.get(nicho_nome)
        if not nicho:
            return {"erro": f"Nicho '{nicho_nome}' nao existe mais."}
        if nicho.prototipo is not None:
            vetor = emb - nicho.prototipo
            nicho.prototipo = nicho.prototipo - 0.1 * vetor
            if nicho.raio is not None:
                nicho.raio = max(0.5, nicho.raio * 1.1)
            else:
                nicho.raio = 1.0
        nicho.energia = max(0.1, nicho.energia - 0.15)
        if correcao:
            tokens = [t for t in correcao.lower().split() if len(t) > 2]
            for t in tokens:
                if t not in nicho.palavras_chave and len(nicho.palavras_chave) < 20:
                    nicho.palavras_chave.append(t)
            nicho.historico.append({
                "data": datetime.now().isoformat(),
                "prompt": self.ultima_interacao["prompt"],
                "resposta": self.ultima_interacao["resposta"],
                "correcao": correcao,
                "tipo": "correcao"
            })
        nicho.salvar()
        return {
            "nicho": nicho_nome,
            "energia": nicho.energia,
            "acao": "push_negativo",
            "amostras": nicho.n_amostras,
            "correcao": correcao
        }

    def testar_transmutacao(self, nicho_nome: str, prompt: str) -> Dict:
        """
        Simula se um prompt seria aceito/transmutado por um nicho especifico.
        Reporta distancia, gate, e qual nicho o roteamento real escolheria.
        """
        emb = self._embedding(prompt)
        nicho = self.nichos.get(nicho_nome)
        if not nicho:
            return {"erro": f"Nicho '{nicho_nome}' nao existe."}
        dist = nicho.distancia(emb)
        raio = nicho.raio if nicho.raio is not None else "N/A"
        em_warmup = nicho.em_warmup()
        confianca = 0.9 if not em_warmup else 0.5
        distancias_all = [n.distancia(emb) for n in self.nichos.values() if not n.em_warmup() and n.nome != "geral"]
        dist_media = float(np.mean(distancias_all)) if distancias_all else 0.0
        passou, motivo = self.gate(prompt, nicho_nome, confianca, False, dist_media)
        nicho_real, conf_real, info = self.roteamento(prompt)
        return {
            "nicho_alvo": nicho_nome,
            "distancia": dist,
            "raio": raio,
            "em_warmup": em_warmup,
            "gate_passou": passou,
            "motivo_gate": motivo,
            "nicho_roteado": nicho_real,
            "confianca_roteamento": conf_real,
            "prompt": prompt
        }

# =============================================================================
# INTERFACE
# =============================================================================

def main():
    print("\nIniciando Camaleao Pessoal...")
    print("   Carregando modelo de embeddings (primeira vez pode demorar)...")
    organismo = OrganismoCamaleao()

    print("\n" + "="*50)
    print("CAMALEAO PESSOAL v3.2.0")
    print("100% local. Seus dados nunca saem deste PC.")
    print("="*50)
    print("\n🦎 NEUROGENESE AUTOMATICA ATIVA!")
    print("Nichos nascem sozinhos quando detectam dominio novo.")
    print("Fale 2x sobre o mesmo tema novo e um nicho nasce.")
    print("="*50)
    print("\nComandos:")
    print('  /nicho <nome> "<descricao>" [palavras-chave]  - Criar novo nicho')
    print("  /renomear <antigo> <novo>                        - Renomear nicho")
    print("  /status                                          - Ver saude do organismo")
    print("  /diagnostico                                     - Metricas completas")
    print("  /energia on|off                                  - Roteamento por energia")
    print("  /contexto on|off                                 - Memoria de contexto")
    print("  /neuro on|off                                    - Neurogenese automatica")
    print("  /web [porta]                                     - Interface web")
    print("  /docs <nicho>                                    - Listar documentos")
    print("  /add_doc <nicho> <caminho>                       - Adicionar documento")
    print("  /exportar [caminho]                              - Backup")
    print("  /importar <caminho.zip>                          - Restaurar backup")
    print("  /bom                                             - Feedback positivo (ultimo nicho)")
    print("  /ruim                                            - Feedback negativo (ultimo nicho)")
    print("  /corrigir [nicho] <resposta certa>                - Corrige ultima resposta (ou nicho especifico)")
    print("  /testar_transmutacao <nicho> <prompt>            - Testa transmutacao")
    print("  /sair                                            - Encerrar")
    print("-"*50)

    while True:
        try:
            prompt = input("\nVoce: ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not prompt:
            continue

        if prompt == "/sair":
            print("Camaleao hibernando. Ate logo.")
            break

        elif prompt.startswith("/nicho "):
            partes = prompt[7:].strip()
            if '"' in partes:
                partes_split = partes.split('"')
                if len(partes_split) >= 2:
                    nome = partes_split[0].strip()
                    descricao = partes_split[1].strip()
                    palavras = partes_split[2].strip() if len(partes_split) > 2 else ""
                    organismo.criar_nicho(nome, descricao, palavras)
                else:
                    print('Uso: /nicho <nome> "<descricao>" [palavra1,palavra2,...]')
            else:
                tokens = partes.split(" ", 1)
                if len(tokens) >= 2:
                    organismo.criar_nicho(tokens[0], tokens[1], "")
                else:
                    print('Uso: /nicho <nome> "<descricao>" [palavras-chave]')

        elif prompt.startswith("/renomear "):
            partes = prompt[10:].strip().split()
            if len(partes) >= 2:
                organismo.renomear_nicho(partes[0], partes[1])
            else:
                print("Uso: /renomear <nome_antigo> <nome_novo>")

        elif prompt == "/status":
            organismo.status()

        elif prompt == "/diagnostico":
            organismo.diagnostico()

        elif prompt == "/web" or prompt.startswith("/web "):
            partes = prompt.split()
            porta = 8080
            if len(partes) > 1:
                try:
                    porta = int(partes[1])
                except:
                    pass
            organismo.iniciar_web(porta)

        elif prompt.startswith("/docs "):
            nicho = prompt[6:].strip()
            docs = organismo.listar_documentos(nicho)
            if docs:
                print(f"Documentos do nicho '{nicho}':")
                for d in docs:
                    print(f"  📄 {d}")
            else:
                print(f"Nenhum documento no nicho '{nicho}'.")

        elif prompt.startswith("/add_doc "):
            partes = prompt[9:].strip().split(None, 1)
            if len(partes) == 2:
                organismo.adicionar_documento(partes[0], partes[1])
            else:
                print("Uso: /add_doc <nicho> <caminho_do_arquivo>")

        elif prompt.startswith("/exportar"):
            partes = prompt[9:].strip()
            if partes:
                organismo.exportar(partes)
            else:
                organismo.exportar()

        elif prompt.startswith("/importar "):
            caminho = prompt[10:].strip()
            if caminho:
                organismo.importar(caminho)
            else:
                print("Uso: /importar <caminho.zip>")

        elif prompt == "/energia on":
            organismo.config.USA_ENERGIA = True
            print("[OK] Roteamento por ENERGIA ativado")

        elif prompt == "/energia off":
            organismo.config.USA_ENERGIA = False
            print("[OK] Roteamento CLASSICO ativado")

        elif prompt == "/contexto on":
            organismo.config.USA_CONTEXTO = True
            print("[OK] Memoria de CONTEXTO ativada")

        elif prompt == "/contexto off":
            organismo.config.USA_CONTEXTO = False
            print("[OK] Memoria de CONTEXTO desativada")

        elif prompt == "/neuro on":
            organismo.config.USA_NEUROGENESE = True
            print("[OK] NEUROGENESE automatica ativada")

        elif prompt == "/neuro off":
            organismo.config.USA_NEUROGENESE = False
            print("[OK] NEUROGENESE automatica desativada")

        elif prompt == "/bom":
            res = organismo.feedback_bom()
            if "erro" in res:
                print(f"[ERRO] {res['erro']}")
            else:
                print(f"[OK] Feedback BOM no nicho '{res['nicho']}'. Energia: {res['energia']:.2f} | Acao: {res['acao']} | Amostras: {res['amostras']}")

        elif prompt == "/ruim":
            res = organismo.feedback_ruim()
            if "erro" in res:
                print(f"[ERRO] {res['erro']}")
            else:
                print(f"[OK] Feedback RUIM no nicho '{res['nicho']}'. Energia: {res['energia']:.2f} | Acao: {res['acao']} | Amostras: {res['amostras']}")

        elif prompt.startswith("/corrigir "):
            resto = prompt[10:].strip()
            tokens = resto.split(None, 1)
            nicho_alvo = None
            correcao = resto
            if len(tokens) == 2 and tokens[0] in organismo.nichos and tokens[0] != "geral":
                nicho_alvo = tokens[0]
                correcao = tokens[1]
            if correcao:
                if nicho_alvo:
                    emb = organismo._embedding(correcao)
                    nicho = organismo.nichos[nicho_alvo]
                    nicho.atualizar_prototipo(emb, forcar=True)
                    nicho.energia = max(0.1, nicho.energia - 0.15)
                    tokens = [t for t in correcao.lower().split() if len(t) > 2]
                    for t in tokens:
                        if t not in nicho.palavras_chave and len(nicho.palavras_chave) < 20:
                            nicho.palavras_chave.append(t)
                    nicho.historico.append({
                        "data": datetime.now().isoformat(),
                        "prompt": f"[correcao manual] {correcao[:100]}",
                        "resposta": correcao,
                        "correcao": correcao,
                        "tipo": "correcao"
                    })
                    nicho.salvar()
                    print(f"[OK] Correcao aplicada no nicho '{nicho_alvo}'. Energia: {nicho.energia:.2f} | Amostras: {nicho.n_amostras}")
                    print(f"   Correcao armazenada: '{correcao[:100]}...'")
                else:
                    res = organismo.feedback_ruim(correcao)
                    if "erro" in res:
                        print(f"[ERRO] {res['erro']}")
                    else:
                        print(f"[OK] Correcao aplicada no nicho '{res['nicho']}'. Energia: {res['energia']:.2f} | Amostras: {res['amostras']}")
                        if res.get('correcao'):
                            print(f"   Correcao armazenada: '{res['correcao'][:100]}...'")
            else:
                print("Uso: /corrigir [nicho] <resposta certa>")

        elif prompt.startswith("/testar_transmutacao "):
            partes = prompt[21:].strip().split(None, 1)
            if len(partes) == 2:
                res = organismo.testar_transmutacao(partes[0], partes[1])
                if "erro" in res:
                    print(f"[ERRO] {res['erro']}")
                else:
                    print(f"\n[TESTE DE TRANSMUTACAO]")
                    print(f"  Prompt: '{res['prompt']}'")
                    print(f"  Nicho alvo: {res['nicho_alvo']}")
                    print(f"  Distancia: {res['distancia']:.3f} | Raio: {res['raio']}")
                    print(f"  Warmup: {'sim' if res['em_warmup'] else 'nao'}")
                    print(f"  Gate: {'PASSOU' if res['gate_passou'] else 'REJEITOU'} ({res['motivo_gate']})")
                    print(f"  Roteamento real escolheria: '{res['nicho_roteado']}' (conf={res['confianca_roteamento']:.2f})")
            else:
                print("Uso: /testar_transmutacao <nicho> <prompt>")

        else:
            print("\n   Camaleao pensando...")
            resultado = organismo.conversar(prompt)

            status = "WARMUP" if organismo.nichos[resultado['nicho']].em_warmup() else ""
            info_rota = ""
            if resultado.get('neurogenese'):
                info_rota = " [NEURO]"
            elif organismo.config.USA_CONTEXTO and organismo.ultimo_nicho == resultado['nicho']:
                info_rota = " [CONTEXTO]"
            elif organismo.config.USA_ENERGIA:
                info_rota = " [ENERGIA]"
            elif resultado.get('confianca_roteamento', 0) > 0.8:
                info_rota = " [KW]"

            print(f"\n[Nicho: {resultado['nicho']}{info_rota} {status} | Conf: {resultado.get('confianca_roteamento', 0):.2f}]")
            if resultado['memoria_relevante']:
                print(f"[Memoria: {len(resultado['memoria_relevante'])} itens]")
            if resultado.get('neurogenese'):
                print(f"[🦎 Neurogenese: nicho '{resultado['neurogenese']}' nasceu!]")
            print(f"\n{resultado['resposta']}")

if __name__ == "__main__":
    main()
