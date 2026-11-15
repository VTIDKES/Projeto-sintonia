# -*- coding: utf-8 -*-
"""Componentes de interface compartilhados pelos modos do Sintonia (CTRL).

Seguem as diretrizes de identidade unificada do ecossistema SINTONIA
(análise RED-Eng): sidebar só de navegação, breadcrumbs, onboarding em
3 passos, cartões de erro didáticos, diagnóstico interpretativo e rodapé
de versão.
"""

import numpy as np
import control as ctrl
import streamlit as st

APP_VERSAO = "v1.1 — última atualização: 15/11/2026"

# (chave em st.session_state.modo_selecionado, rótulo)
ROTAS = [
    (None, "🏠 Início"),
    ("classico", "Modo Clássico"),
    ("canvas", "Modo Diagrama de Blocos"),
    ("circuitos", "Modo Simulação com Elementos (Beta)"),
    ("teoria", "📖 Teoria"),
]

ICONE_STATUS = {"Estável": "🟢 ●", "Marginal": "🟡 ▲", "Instável": "🔴 ■"}


def sidebar_navegacao(modo):
    """Sidebar com papel único: navegação global + alternador Básico/Avançado."""
    with st.sidebar:
        st.header("Navegação")
        for chave, rotulo in ROTAS:
            if st.button(rotulo, key=f"nav_{chave}", use_container_width=True,
                         type="primary" if chave == modo else "secondary"):
                st.session_state.modo_selecionado = chave
                st.rerun()
        st.toggle(
            "⚙ Modo Avançado", key="avancado",
            help="Básico: planta, ganho K, resposta no tempo e polos/zeros. "
                 "Avançado: controlador, sensor, espaço de estados, Bode, "
                 "Nyquist, LGR e erro estacionário.")
        st.markdown("---")


def breadcrumbs(modo):
    caminho = ["Início", "Sistemas de Controle (CTRL)"]
    if modo is not None:
        caminho.append(dict(ROTAS)[modo].replace("📖 ", ""))
    caminho[-1] = f"**{caminho[-1]}**"
    st.caption(" → ".join(caminho))


def rodape():
    st.markdown("---")
    st.caption(f"Projeto Sintonia · Sistemas de Controle (CTRL) · {APP_VERSAO}")


def onboarding(configurar, executar, interpretar):
    """Contrato fixo de 3 passos: Configurar → Executar → Interpretar."""
    with st.expander("📖 Como utilizar"):
        st.markdown(
            f"1. **Configurar** — {configurar}\n"
            f"2. **Executar** — {executar}\n"
            f"3. **Interpretar** — {interpretar}")


def cartao_erro(situacao, causa, acao):
    """Cartão de erro didático: Situação → Causa provável → Ação recomendada."""
    st.error(
        f"**Situação:** {situacao}\n\n"
        f"**Causa provável:** {causa}\n\n"
        f"**Ação recomendada:** {acao}")


def _fmt_polos(polos):
    if len(polos) == 0:
        return "nenhum"
    return ", ".join(
        f"{p.real:.3g}" if abs(p.imag) < 1e-9
        else f"{p.real:.3g} {'+' if p.imag >= 0 else '-'} {abs(p.imag):.3g}j"
        for p in polos)


def diagnosticar(sistema, malha_aberta=None):
    """Leitura qualitativa de estabilidade a partir da TransferFunction.

    `sistema` é a FT analisada (define o status pelos polos). `malha_aberta`
    é L(s), usada para as margens de ganho/fase; None quando não se conhece.
    Retorna dict com status, gm_db, pm e mensagens.
    """
    tol = 1e-6
    polos = ctrl.poles(sistema)
    instaveis = [p for p in polos if p.real > tol]
    no_eixo = [p for p in polos if abs(p.real) <= tol]

    if instaveis:
        status = "Instável"
        msgs = [f"Sistema instável: polo(s) no semiplano direito "
                f"(s = {_fmt_polos(instaveis)}). Reduza o ganho K ou adicione "
                f"um compensador em avanço de fase."]
    elif no_eixo:
        status = "Marginal"
        msgs = [f"Marginalmente estável: polo(s) sobre o eixo imaginário "
                f"(s = {_fmt_polos(no_eixo)}). A resposta não converge "
                f"(oscila sem amortecer ou cresce). Use realimentação ou "
                f"compensação para levar os polos ao semiplano esquerdo."]
    else:
        status = "Estável"
        msgs = ["Sistema estável: todos os polos estão no semiplano esquerdo."]

    gm_db = pm = None
    if malha_aberta is not None:
        gm, pm, _, _ = ctrl.margin(malha_aberta)
        gm_db = 20 * np.log10(gm) if np.isfinite(gm) and gm > 0 else np.inf
        if np.isnan(pm):
            pm = np.inf
        if pm <= 0 or gm_db < 0:
            msgs.append("Margem negativa: ao fechar a malha com esse ganho, "
                        "o sistema tende à instabilidade. Reduza o ganho ou "
                        "adicione compensação.")
        elif pm < 30:
            msgs.append("Estabilidade relativa baixa: margem de fase abaixo "
                        "de 30° indica resposta oscilatória. Considere "
                        "reduzir o ganho ou adicionar compensação.")
        if malha_aberta is not sistema:
            msgs.append(
                f"Efeito da realimentação: polos de malha aberta "
                f"({_fmt_polos(ctrl.poles(malha_aberta))}) → malha fechada "
                f"({_fmt_polos(polos)}).")

    return {"status": status, "gm_db": gm_db, "pm": pm, "mensagens": msgs}


def _fmt_margem(valor, unidade):
    return "∞" if np.isinf(valor) else f"{valor:.1f} {unidade}"


def cartao_diagnostico(sistema, malha_aberta=None):
    """Cartão interpretativo + selo de status na sidebar. Retorna o diagnóstico."""
    d = diagnosticar(sistema, malha_aberta)
    selo = f"{ICONE_STATUS[d['status']]} {d['status']}"
    if d["gm_db"] is not None:
        selo += (f" · MG {_fmt_margem(d['gm_db'], 'dB')}"
                 f" · MF {_fmt_margem(d['pm'], '°')}")
    caixa = {"Estável": st.success, "Marginal": st.warning,
             "Instável": st.error}[d["status"]]
    caixa(f"**🩺 Diagnóstico: {selo}**\n\n" + "\n\n".join(d["mensagens"]))
    st.sidebar.markdown(f"**Status do sistema**\n\n{selo}")
    return d


def demo():
    """Auto-teste: python ui_comum.py"""
    tf = ctrl.TransferFunction
    assert diagnosticar(tf([4], [1, 2, 4]))["status"] == "Estável"
    assert diagnosticar(tf([1], [1, 1, -2]))["status"] == "Instável"
    assert diagnosticar(tf([1], [1, 0, 4]))["status"] == "Marginal"
    # L(s) = 100/(s(s+1)(s+5)): margem negativa, malha fechada instável
    L = tf([100], [1, 6, 5, 0])
    d = diagnosticar(ctrl.feedback(L, 1), L)
    assert d["status"] == "Instável" and d["pm"] < 0
    assert any("Margem negativa" in m for m in d["mensagens"])
    # L(s) = 20/(s(s+1)): estável em MF, margem de fase baixa (~12,8°)
    L = tf([20], [1, 1, 0])
    d = diagnosticar(ctrl.feedback(L, 1), L)
    assert d["status"] == "Estável" and 0 < d["pm"] < 30
    assert any("abaixo de 30°" in m for m in d["mensagens"])
    print("ok")


if __name__ == "__main__":
    demo()
