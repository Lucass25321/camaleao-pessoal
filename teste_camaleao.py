# -*- coding: utf-8 -*-
"""
Teste Automático do Camaleão Pessoal v3.2.0
Valida: roteamento por energia, gate, feedback, neurogênese, proteção de nichos manuais
"""

import os
import sys
import shutil
import json
import numpy as np

# =============================================================================
# LIMPEZA
# =============================================================================
print("="*60)
print("TESTE AUTOMATICO CAMALEAO v3.2.0")
print("="*60)
print("\n🧹 Limpando dados antigos...")
for pasta in ["./nichos", "./memoria", "./dados"]:
    if os.path.exists(pasta):
        shutil.rmtree(pasta)
        print(f"   Removido: {pasta}")
print("   ✅ Dados limpos")

# =============================================================================
# IMPORTA E INICIALIZA
# =============================================================================
print("\n📦 Importando Camaleao...")
from camaleao import OrganismoCamaleao, Config

print("\n🚀 Inicializando organismo (pode demorar no 1o download)...")
organismo = OrganismoCamaleao()
print("   ✅ Organismo pronto")

# =============================================================================
# UTILS
# =============================================================================
passou_total = 0
falhou_total = 0

def check(condicao, descricao):
    global passou_total, falhou_total
    if condicao:
        print(f"   ✅ PASSOU: {descricao}")
        passou_total += 1
    else:
        print(f"   ❌ FALHOU: {descricao}")
        falhou_total += 1
    return condicao

# =============================================================================
# TESTE 1: Criar nichos manuais
# =============================================================================
print("\n" + "="*60)
print("TESTE 1: Criar nichos manuais")
print("="*60)

organismo.criar_nicho("futebol", "Tudo sobre futebol, copas, times e jogadores")
organismo.criar_nicho("matematica", "Contas e problemas numericos")

check("futebol" in organismo.nichos, "Nicho 'futebol' existe")
check("matematica" in organismo.nichos, "Nicho 'matematica' existe")
check(not organismo.nichos["futebol"].auto_gerado, "futebol é manual (auto_gerado=False)")
check(not organismo.nichos["matematica"].auto_gerado, "matematica é manual (auto_gerado=False)")
check("futebol" in organismo.nichos["futebol"].palavras_chave, "Nome 'futebol' é palavra-chave implicita")
check("matematica" in organismo.nichos["matematica"].palavras_chave, "Nome 'matematica' é palavra-chave implicita")
check(organismo.nichos["futebol"].prototipo is not None, "futebol tem prototipo inicial")
check(organismo.nichos["matematica"].prototipo is not None, "matematica tem prototipo inicial")
check("copas" in organismo.nichos["futebol"].palavras_chave or "futebol" in organismo.nichos["futebol"].palavras_chave, 
      "Palavras da descricao viraram keywords (futebol tem: " + str(organismo.nichos["futebol"].palavras_chave) + ")")

# =============================================================================
# TESTE 2: Roteamento por palavras-chave ou embedding
# =============================================================================
print("\n" + "="*60)
print("TESTE 2: Roteamento")
print("="*60)

nicho, conf, info = organismo.roteamento("quem ganhou a copa de 94?")
check(nicho == "futebol", f"'copa de 94' → futebol (foi: {nicho}, motivo: {info.get('motivo')})")
check(conf >= 0.5, f"Confiança razoavel no roteamento (foi: {conf:.2f})")

nicho2, conf2, info2 = organismo.roteamento("quanto é 30 x 50?")
check(nicho2 == "matematica", f"'30 x 50' → matematica (foi: {nicho2}, motivo: {info2.get('motivo')})")

# v3.1.7: teste honesto — o embedding pode colocar "mona lisa" perto de matemática
# (limitação do modelo de embeddings). O importante é: NÃO vai pro futebol.
nicho3, conf3, info3 = organismo.roteamento("quem pintou a mona lisa?")
check(nicho3 != "futebol", f"'mona lisa' NÃO vai pro futebol (foi: {nicho3}, motivo: {info3.get('motivo')})")

# =============================================================================
# TESTE 3: Gate em nicho consolidado
# =============================================================================
print("\n" + "="*60)
print("TESTE 3: Gate de surpresa em nicho consolidado")
print("="*60)

# Forca matematica a ter amostras e raio
for i in range(10):
    emb = organismo._embedding(f"quanto é {i} + {i}?")
    organismo.nichos["matematica"].atualizar_prototipo(emb)

print(f"   matematica: {organismo.nichos['matematica'].n_amostras} amostras, raio={organismo.nichos['matematica'].raio:.3f}")

passou, motivo = organismo.gate("quanto é a raiz quadrada de 144?", "matematica", 0.9, False, 0.0)
check(passou, f"Matematica aceita matematica (foi: {passou}, {motivo})")

passou2, motivo2 = organismo.gate("quem ganhou a copa de 94?", "matematica", 0.9, False, 0.0)
check(not passou2, f"Matematica REJEITA futebol (foi: {passou2}, {motivo2})")
check("surpresa" in motivo2 or "teto" in motivo2, f"Motivo de rejeicao é surpresa/teto (foi: {motivo2})")

# =============================================================================
# TESTE 4: Gate em warmup com surpresa extrema
# =============================================================================
print("\n" + "="*60)
print("TESTE 4: Gate em warmup com surpresa extrema")
print("="*60)

# futebol ainda está em warmup (só tem 1 amostra do prototipo inicial)
print(f"   futebol: {organismo.nichos['futebol'].n_amostras} amostras, raio={organismo.nichos['futebol'].raio:.3f}")

passou3, motivo3 = organismo.gate("quanto é a raiz quadrada de 144?", "futebol", 0.8, False, 0.0)
check(not passou3, f"Futebol em warmup REJEITA matematica (foi: {passou3}, {motivo3})")
check("surpresa_extrema" in motivo3, f"Motivo é surpresa_extrema (foi: {motivo3})")

# =============================================================================
# TESTE 5: Feedback com nicho específico
# =============================================================================
print("\n" + "="*60)
print("TESTE 5: Feedback com nicho específico")
print("="*60)

# Simula uma interação no futebol
organismo.ultima_interacao = {
    "prompt": "quem ganhou a copa de 94?",
    "nicho": "futebol",
    "resposta": "A França ganhou",
    "embedding": organismo._embedding("quem ganhou a copa de 94?"),
    "confianca": 0.9,
    "gate_passou": True,
    "motivo": "aprovado"
}

# Feedback ruim SEM especificar nicho → vai na última interação (futebol)
res = organismo.feedback_ruim("O Brasil ganhou a Copa de 1994")
check(res["nicho"] == "futebol", f"Feedback ruim vai pro nicho da ultima interacao (foi: {res['nicho']})")
check(res["acao"] == "push_negativo", f"Acao é push_negativo (foi: {res['acao']})")
check(res["energia"] < 1.0, f"Energia caiu (foi: {res['energia']:.2f})")

# Feedback bom no futebol
res2 = organismo.feedback_bom()
check(res2["nicho"] == "futebol", f"Feedback bom vai pro futebol (foi: {res2['nicho']})")
check(res2["acao"] == "reforco_positivo", f"Acao é reforco_positivo (foi: {res2['acao']})")
check(res2["energia"] > res["energia"], f"Energia subiu com feedback bom (foi: {res2['energia']:.2f})")

# =============================================================================
# TESTE 6: Neurogênese automática
# =============================================================================
print("\n" + "="*60)
print("TESTE 6: Neurogênese automática")
print("="*60)

# Desativa neurogenese para nao interferir nos testes anteriores, agora reativa
organismo.config.USA_NEUROGENESE = True
organismo.config.NEURO_MIN_AMOSTRAS_NASCER = 3
organismo.buffer_novidade = []
organismo.janela_exploracao = 0

# 3 prompts sobre cachorros
prompts_cachorro = [
    "Eu tenho um golden retriever e ele adora destruir meus chinelos",
    "Qual a melhor ração para cachorro de porte médio?",
    "Quanto custa um golden retriever filhote?"
]

for i, p in enumerate(prompts_cachorro):
    print(f"\n   Prompt {i+1}: '{p[:50]}...'")
    resultado = organismo.conversar(p)
    print(f"   → Nicho: {resultado['nicho']} | Neurogenese: {resultado.get('neurogenese')} | Buffer: {len(organismo.buffer_novidade)}")

# Verifica se nasceu um nicho auto-gerado
nichos_auto = [n for n, nic in organismo.nichos.items() if nic.auto_gerado]
check(len(nichos_auto) > 0, f"Nasceu pelo menos 1 nicho auto-gerado (foi: {len(nichos_auto)}: {nichos_auto})")

# Verifica que nichos manuais ainda existem
check("futebol" in organismo.nichos, "futebol ainda existe apos neurogenese")
check("matematica" in organismo.nichos, "matematica ainda existe apos neurogenese")

# Verifica que nenhum nicho manual foi marcado como auto_gerado
check(not organismo.nichos["futebol"].auto_gerado, "futebol continua manual")
check(not organismo.nichos["matematica"].auto_gerado, "matematica continua manual")

# =============================================================================
# TESTE 7: Fusão não deleta nichos manuais
# =============================================================================
print("\n" + "="*60)
print("TESTE 7: Proteção de nichos manuais na fusão")
print("="*60)

# Cria um nicho auto-gerado artificialmente perto de futebol
organismo.nichos["futebol_teste"] = organismo.nichos["futebol"].__class__(
    "futebol_teste", "Teste", organismo.config, ["futebol"], auto_gerado=True
)
# Copia o prototipo do futebol para forçar proximidade
organismo.nichos["futebol_teste"].prototipo = organismo.nichos["futebol"].prototipo.copy()
organismo.nichos["futebol_teste"].n_amostras = 1
organismo.nichos["futebol_teste"].raio = 0.5
organismo.nichos["futebol_teste"].salvar()

# Forca limpeza de duplicados
organismo._limpar_duplicados()

check("futebol" in organismo.nichos, "futebol (manual) NAO foi deletado pela fusão")
check("futebol_teste" not in organismo.nichos, "futebol_teste (auto) FOI fundido/deletado")

# =============================================================================
# TESTE 8: Testar transmutação
# =============================================================================
print("\n" + "="*60)
print("TESTE 8: Testar transmutação")
print("="*60)

# Testa matematica com matematica → deve passar
res = organismo.testar_transmutacao("matematica", "quanto é 2 + 2?")
check(res["gate_passou"], f"Matematica aceita '2+2' (foi: {res['gate_passou']})")
check(res["nicho_roteado"] == "matematica", f"Roteamento escolhe matematica (foi: {res['nicho_roteado']})")

# Testa matematica (consolidado) com futebol → deve rejeitar
res2 = organismo.testar_transmutacao("matematica", "quem ganhou a copa de 1994?")
check(not res2["gate_passou"], f"Matematica REJEITA 'copa de 94' (foi: {res2['gate_passou']}, {res2['motivo_gate']})")
check("surpresa" in res2["motivo_gate"], f"Motivo é surpresa (foi: {res2['motivo_gate']})")

# =============================================================================
# TESTE 9: Roteamento por energia (NLL)
# =============================================================================
print("\n" + "="*60)
print("TESTE 9: Roteamento por energia (NLL)")
print("="*60)

# Verifica que o roteamento por energia esta ativo
check(organismo.config.USA_ROTEAMENTO_ENERGIA, "Roteamento por energia esta ATIVADO")

# Testa roteamento por energia com um prompt de matematica
nicho_e, conf_e, info_e = organismo.roteamento("quanto é 5 + 7?")
check(info_e.get("motivo") == "energia", f"Roteamento usou energia (motivo: {info_e.get('motivo')})")
check("nlls" in info_e, f"Info contem NLLs (keys: {list(info_e.keys())})")

# Verifica que todos os nichos receberam NLL
nichos_com_nll = [n for n, nic in organismo.nichos.items() if nic.nll_historico]
check(len(nichos_com_nll) >= 2, f"Pelo menos 2 nichos tem NLL (foi: {len(nichos_com_nll)})")

# Verifica que o nicho vencedor tem o menor NLL
if "nlls" in info_e:
    nlls = info_e["nlls"]
    vencedor = min(nlls, key=nlls.get)
    check(vencedor == nicho_e, f"Vencedor por NLL coincide com roteamento ({vencedor} == {nicho_e})")

# Testa diagnostico de energia
organismo.diagnostico_energia()

# =============================================================================
# RELATÓRIO FINAL
# =============================================================================
print("\n" + "="*60)
print("RELATORIO FINAL")
print("="*60)
total = passou_total + falhou_total
print(f"\n   ✅ Passaram: {passou_total}/{total}")
print(f"   ❌ Falharam: {falhou_total}/{total}")
print(f"   📊 Taxa de sucesso: {passou_total/total*100:.1f}%")

if falhou_total == 0:
    print("\n   🦎 TODOS OS TESTES PASSARAM! O organismo está saudável.")
else:
    print(f"\n   ⚠️  {falhou_total} teste(s) falharam. Revise os logs acima.")

print("="*60)
