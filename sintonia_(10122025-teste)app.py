# -*- coding: utf-8 -*-
"""
Sistema de Modelagem e Análise de Sistemas de Controle
Otimizado para Streamlit Cloud
"""

import streamlit as st
import pandas as pd
import numpy as np
import sympy as sp
from sympy.parsing.sympy_parser import parse_expr
import control as ctrl
from control import TransferFunction, margin, step_response, forced_response, root_locus
from scipy import signal
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# --- Editor visual (Simulink/Xcos) ---
try:
    from streamlit_flow import streamlit_flow
    from streamlit_flow.interfaces import StreamlitFlowNode, StreamlitFlowEdge
except Exception:
    streamlit_flow = None
    StreamlitFlowNode = None
    StreamlitFlowEdge = None

import uuid

# =====================================================
# CONFIGURAÇÕES E CONSTANTES
# =====================================================

# Opções de análise
ANALYSIS_OPTIONS = {
    "malha_aberta": ["Resposta no tempo", "Desempenho", "Diagrama de Polos e Zeros",
                    "Diagrama De Bode Magnitude", "Diagrama De Bode Fase", "Nyquist"],
    "malha_fechada": ["Resposta no tempo", "Desempenho", "Diagrama de Polos e Zeros",
                     "Diagrama De Bode Magnitude", "Diagrama De Bode Fase", "LGR"]
}

# Sinais de entrada
INPUT_SIGNALS = ['Degrau', 'Rampa', 'Senoidal', 'Impulso', 'Parabólica']

# =====================================================
# FUNÇÕES AUXILIARES
# =====================================================

def formatar_numero(valor):
    """Formata números para exibição amigável"""
    if np.isinf(valor):
        return '∞'
    elif np.isnan(valor):
        return '-'
    else:
        return f"{valor:.3f}"

# =====================================================
# FUNÇÕES DE TRANSFERÊNCIA
# =====================================================

def converter_para_tf(numerador_str, denominador_str):
    """Converte strings de numerador e denominador em uma função de transferência"""
    s = sp.Symbol('s')
    num = parse_expr(numerador_str.replace('^', '**'), local_dict={'s': s})
    den = parse_expr(denominador_str.replace('^', '**'), local_dict={'s': s})
    
    num, den = sp.fraction(sp.together(num / den))
    num_coeffs = [float(c) for c in sp.Poly(num, s).all_coeffs()]
    den_coeffs = [float(c) for c in sp.Poly(den, s).all_coeffs()]
    
    if den_coeffs and den_coeffs[0] != 1:
        fator = den_coeffs[0]
        num_coeffs = [c / fator for c in num_coeffs]
        den_coeffs = [c / fator for c in den_coeffs]
    
    return TransferFunction(num_coeffs, den_coeffs), (num, den)

def tipo_do_sistema(G):
    """Determina o tipo do sistema (número de integradores)"""
    G_min = ctrl.minreal(G, verbose=False)
    polos = ctrl.poles(G_min)
    tipo = sum(1 for p in polos if np.isclose(np.real_if_close(p), 0.0, atol=1e-3))
    return tipo

def constantes_de_erro(G):
    """Calcula as constantes de erro (Kp, Kv, Ka)"""
    s = ctrl.tf('s')
    G_min = ctrl.minreal(G, verbose=False)
    tipo = tipo_do_sistema(G_min)
    
    Kp = Kv = Ka = np.inf
    
    try:
        if tipo >= 0:
            Kp = ctrl.dcgain(G_min)
        if tipo >= 1:
            Kv = ctrl.dcgain(s * G_min)
        if tipo >= 2:
            Ka = ctrl.dcgain((s**2) * G_min)
    except:
        pass
    
    return tipo, Kp, Kv, Ka

def erros_estacionarios(tipo, Kp, Kv, Ka):
    """Calcula os erros estacionários para diferentes entradas"""
    resultados = []
    
    # Degrau (1/s)
    if tipo == 0:
        e_degrau = 1 / (1 + Kp) if np.isfinite(Kp) else 0
    else:
        e_degrau = 0
    resultados.append(("Degrau", e_degrau))
    
    # Rampa (1/s²)
    if tipo == 0:
        e_rampa = np.inf
    elif tipo == 1:
        e_rampa = 1 / Kv if np.isfinite(Kv) and Kv != 0 else np.inf
    else:
        e_rampa = 0
    resultados.append(("Rampa", e_rampa))
    
    # Parabólica (1/s³)
    if tipo <= 1:
        e_parab = np.inf
    elif tipo == 2:
        e_parab = 1 / Ka if np.isfinite(Ka) and Ka != 0 else np.inf
    else:
        e_parab = 0
    resultados.append(("Parabólica", e_parab))
    
    return resultados

# =====================================================
# PLOTAGEM E ANÁLISE
# =====================================================

def plot_resposta_tempo(sistema, entrada="Degrau", amplitude=1.0, freq=1.0):
    """Gera gráficos de resposta no tempo"""
    t = np.linspace(0, 10, 1000)
    
    if entrada == "Degrau":
        t, y = ctrl.step_response(sistema * amplitude, T=t)
        u = np.ones_like(t) * amplitude
    elif entrada == "Rampa":
        u = t * amplitude
        t, y, _ = forced_response(sistema, T=t, U=u)
    elif entrada == "Senoidal":
        u = amplitude * np.sin(2 * np.pi * freq * t)
        t, y, _ = forced_response(sistema, T=t, U=u)
    elif entrada == "Impulso":
        t, y = ctrl.impulse_response(sistema * amplitude, T=t)
        u = np.zeros_like(t)
        u[0] = amplitude * 100
    else:  # Parabólica
        u = (t**2) * amplitude
        t, y, _ = forced_response(sistema, T=t, U=u)
    
    fig = make_subplots(rows=2, cols=1, subplot_titles=("Entrada", "Saída"))
    
    fig.add_trace(go.Scatter(x=t, y=u, name="Entrada"), row=1, col=1)
    fig.add_trace(go.Scatter(x=t, y=y, name="Saída"), row=2, col=1)
    
    fig.update_layout(height=600, showlegend=True)
    fig.update_xaxes(title_text="Tempo (s)", row=2, col=1)
    
    return fig, t, y

def calcular_desempenho(t, y):
    """Calcula métricas de desempenho temporal"""
    y_final = y[-1]
    y_max = np.max(y)
    
    # Tempo de subida (10% a 90%)
    try:
        t_10 = t[np.where(y >= 0.1 * y_final)[0][0]]
        t_90 = t[np.where(y >= 0.9 * y_final)[0][0]]
        tempo_subida = t_90 - t_10
    except:
        tempo_subida = np.nan
    
    # Sobressinal
    sobressinal = ((y_max - y_final) / y_final) * 100 if y_final != 0 else np.nan
    
    # Tempo de acomodação (2%)
    try:
        idx_settle = np.where(np.abs(y - y_final) > 0.02 * np.abs(y_final))[0]
        tempo_acomodacao = t[idx_settle[-1]] if len(idx_settle) > 0 else 0
    except:
        tempo_acomodacao = np.nan
    
    # Tempo de pico
    tempo_pico = t[np.argmax(y)]
    
    return {
        "Valor Final": y_final,
        "Valor Máximo": y_max,
        "Tempo de Subida": tempo_subida,
        "Sobressinal (%)": sobressinal,
        "Tempo de Acomodação": tempo_acomodacao,
        "Tempo de Pico": tempo_pico
    }

def plot_polos_zeros(sistema):
    """Plota polos e zeros do sistema"""
    polos = ctrl.poles(sistema)
    zeros = ctrl.zeros(sistema)
    
    fig = go.Figure()
    
    if len(polos) > 0:
        fig.add_trace(go.Scatter(
            x=np.real(polos), y=np.imag(polos),
            mode='markers', name='Polos',
            marker=dict(symbol='x', size=12)
        ))
    
    if len(zeros) > 0:
        fig.add_trace(go.Scatter(
            x=np.real(zeros), y=np.imag(zeros),
            mode='markers', name='Zeros',
            marker=dict(symbol='circle-open', size=12)
        ))
    
    fig.add_hline(y=0, line_dash="dash", line_color="gray")
    fig.add_vline(x=0, line_dash="dash", line_color="gray")
    
    fig.update_layout(
        title="Diagrama de Polos e Zeros",
        xaxis_title="Parte Real",
        yaxis_title="Parte Imaginária",
        height=500
    )
    
    return fig

def plot_bode(sistema, tipo="magnitude"):
    """Plota diagrama de Bode"""
    w = np.logspace(-2, 2, 1000)
    mag, phase, omega = ctrl.bode(sistema, w, plot=False)
    
    if tipo == "magnitude":
        y = 20 * np.log10(mag)
        titulo = "Diagrama de Bode - Magnitude"
        y_label = "Magnitude (dB)"
    else:
        y = np.degrees(phase)
        titulo = "Diagrama de Bode - Fase"
        y_label = "Fase (graus)"
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=omega, y=y, mode='lines'))
    
    fig.update_layout(
        title=titulo,
        xaxis_title="Frequência (rad/s)",
        yaxis_title=y_label,
        xaxis_type="log",
        height=500
    )
    
    return fig

def plot_nyquist(sistema):
    """Plota diagrama de Nyquist"""
    w = np.logspace(-2, 2, 1000)
    real, imag, freq = ctrl.nyquist_plot(sistema, w, plot=False)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=real, y=imag, mode='lines', name='Nyquist'))
    fig.add_trace(go.Scatter(x=real, y=-imag, mode='lines', name='Nyquist (espelho)', line=dict(dash='dash')))
    
    fig.add_trace(go.Scatter(x=[-1], y=[0], mode='markers', name='-1+j0', marker=dict(color='red', size=10)))
    
    fig.add_hline(y=0, line_dash="dash", line_color="gray")
    fig.add_vline(x=0, line_dash="dash", line_color="gray")
    
    fig.update_layout(
        title="Diagrama de Nyquist",
        xaxis_title="Parte Real",
        yaxis_title="Parte Imaginária",
        height=500,
        showlegend=True
    )
    
    return fig

def plot_root_locus(sistema):
    """Plota lugar das raízes"""
    kvect = np.linspace(0, 100, 500)
    rlist, klist = root_locus(sistema, kvect=kvect, plot=False)
    
    fig = go.Figure()
    
    # Plotar trajetórias
    for i in range(rlist.shape[1]):
        fig.add_trace(go.Scatter(
            x=np.real(rlist[:, i]), y=np.imag(rlist[:, i]),
            mode='lines', name=f'Ramo {i+1}',
            showlegend=False
        ))
    
    # Plotar polos e zeros iniciais
    polos = ctrl.poles(sistema)
    zeros = ctrl.zeros(sistema)
    
    if len(polos) > 0:
        fig.add_trace(go.Scatter(
            x=np.real(polos), y=np.imag(polos),
            mode='markers', name='Polos',
            marker=dict(symbol='x', size=12, color='red')
        ))
    
    if len(zeros) > 0:
        fig.add_trace(go.Scatter(
            x=np.real(zeros), y=np.imag(zeros),
            mode='markers', name='Zeros',
            marker=dict(symbol='circle-open', size=12, color='blue')
        ))
    
    fig.add_hline(y=0, line_dash="dash", line_color="gray")
    fig.add_vline(x=0, line_dash="dash", line_color="gray")
    
    fig.update_layout(
        title="Lugar Geométrico das Raízes",
        xaxis_title="Parte Real",
        yaxis_title="Parte Imaginária",
        height=500
    )
    
    return fig

# =====================================================
# MALHA FECHADA
# =====================================================

def calcular_malha_fechada(planta, controlador=None, sensor=None):
    """Calcula o sistema em malha fechada com controlador e sensor opcionais"""
    if controlador is None:
        controlador = ctrl.tf([1], [1])
    if sensor is None:
        sensor = ctrl.tf([1], [1])
    
    # Sistema em malha fechada: Gc*Gp / (1 + Gc*Gp*H)
    return ctrl.feedback(controlador * planta, sensor)

# =====================================================
# INTERFACE DE BLOCOS
# =====================================================

def criar_sidebar():
    """Cria interface na sidebar para adicionar blocos"""
    st.sidebar.header("🧱 Editor de Blocos")
    
    nome = st.sidebar.text_input("Nome do bloco", value="G1")
    tipo = st.sidebar.selectbox("Tipo de bloco", ["Planta", "Controlador", "Sensor", "Outro"])
    
    numerador = st.sidebar.text_input("Numerador", value="1")
    denominador = st.sidebar.text_input("Denominador", value="s+1")
    
    if st.sidebar.button("➕ Adicionar Bloco"):
        try:
            tf, (num_sym, den_sym) = converter_para_tf(numerador, denominador)
            adicionar_bloco(nome, tipo, numerador, denominador)
            st.sidebar.success(f"Bloco {nome} adicionado!")
        except Exception as e:
            st.sidebar.error(f"Erro ao adicionar bloco: {e}")

# =====================================================
# GERENCIAMENTO DE BLOCOS
# =====================================================

def inicializar_blocos():
    """Inicializa o estado dos blocos se não existir"""
    if 'blocos' not in st.session_state:
        st.session_state.blocos = pd.DataFrame(columns=['nome', 'tipo', 'numerador', 'denominador', 'tf', 'tf_simbolico'])

def adicionar_bloco(nome, tipo, numerador, denominador):
    """Adiciona um novo bloco ao sistema"""
    try:
        tf, (num_sym, den_sym) = converter_para_tf(numerador, denominador)
        
        novo_bloco = pd.DataFrame([{
            'nome': nome,
            'tipo': tipo,
            'numerador': numerador,
            'denominador': denominador,
            'tf': tf,
            'tf_simbolico': (num_sym, den_sym)
        }])
        
        st.session_state.blocos = pd.concat([st.session_state.blocos, novo_bloco], ignore_index=True)
        return True, f"Bloco {nome} adicionado com sucesso!"
    except Exception as e:
        return False, f"Erro ao adicionar bloco: {e}"

def remover_bloco(nome):
    """Remove um bloco do sistema"""
    df = st.session_state.blocos
    if nome in df['nome'].values:
        st.session_state.blocos = df[df['nome'] != nome].reset_index(drop=True)
        return f"Bloco {nome} excluído."
    return "Bloco não encontrado."

def obter_bloco_por_tipo(tipo):
    """Obtém o primeiro bloco de um tipo específico"""
    df = st.session_state.blocos
    if any(df['tipo'] == tipo):
        return df[df['tipo'] == tipo].iloc[0]['tf']
    return None

# =====================================================
# EDITOR VISUAL (BETA) - Drag/Drop + Conexões
# Requer: pip install streamlit-flow-component
# =====================================================

def _ensure_flow_state():
    if "flow_state" not in st.session_state:
        st.session_state.flow_state = {"nodes": [], "edges": []}
    if "flow_edges" not in st.session_state:
        st.session_state.flow_edges = []
    if "flow_nodes_by_name" not in st.session_state:
        st.session_state.flow_nodes_by_name = {}  # nome_bloco -> node_id


def blocos_para_flow(df: pd.DataFrame):
    """Converte seus blocos (df) em nós do editor visual, preservando posições."""
    _ensure_flow_state()

    nodes = []
    edges = st.session_state.flow_state.get("edges", [])

    for i, row in df.iterrows():
        nome = str(row["nome"])
        tipo = str(row["tipo"])

        if nome not in st.session_state.flow_nodes_by_name:
            st.session_state.flow_nodes_by_name[nome] = f"node-{uuid.uuid4().hex[:8]}"

        node_id = st.session_state.flow_nodes_by_name[nome]

        # reaproveita posição anterior, se existir
        prev = None
        for n in st.session_state.flow_state.get("nodes", []):
            if isinstance(n, dict) and n.get("id") == node_id:
                prev = n
                break
            if StreamlitFlowNode is not None and hasattr(n, "id") and n.id == node_id:
                prev = {"position": n.position}
                break

        x = (prev["position"]["x"] if prev and "position" in prev else (50 + i * 220))
        y = (prev["position"]["y"] if prev and "position" in prev else 80)

        label = (
            f"**{nome}**\n"
            f"Tipo: `{tipo}`\n"
            f"Num: `{row['numerador']}`\n"
            f"Den: `{row['denominador']}`"
        )

        nodes.append(
            StreamlitFlowNode(
                id=node_id,
                pos=(x, y),
                data={"label": label, "nome": nome},
                node_type="default",
                source_position="right",
                target_position="left",
            )
        )

    valid_ids = {n.id for n in nodes}
    # flow_state pode guardar edges como dicts
    cleaned_edges = []
    for e in edges:
        if isinstance(e, dict):
            if e.get("source") in valid_ids and e.get("target") in valid_ids:
                cleaned_edges.append(e)
        else:
            if getattr(e, "source", None) in valid_ids and getattr(e, "target", None) in valid_ids:
                cleaned_edges.append(e)

    return nodes, cleaned_edges


def flow_para_edges(flow_return):
    """Extrai arestas do retorno do editor e salva em session_state."""
    edges_out = []
    for e in flow_return.get("edges", []):
        edges_out.append({"source": e["source"], "target": e["target"]})
    st.session_state.flow_edges = edges_out
    return edges_out


def montar_tf_em_serie_por_flow(df: pd.DataFrame, edges_simple: list):
    """Monta uma TF em série seguindo o 'tronco' principal do grafo (pipeline simples)."""
    if df.empty:
        return None

    nodeid_by_nome = st.session_state.get("flow_nodes_by_name", {}).copy()

    tf_by_node = {}
    for _, row in df.iterrows():
        nome = str(row["nome"])
        node_id = nodeid_by_nome.get(nome)
        if node_id:
            tf_by_node[node_id] = row["tf"]

    if not tf_by_node:
        return None

    out_map = {}
    in_deg = {nid: 0 for nid in tf_by_node.keys()}

    for e in edges_simple:
        s, t = e.get("source"), e.get("target")
        if s in tf_by_node and t in tf_by_node:
            out_map.setdefault(s, []).append(t)
            in_deg[t] = in_deg.get(t, 0) + 1

    starts = [nid for nid, deg in in_deg.items() if deg == 0]
    cur = starts[0] if starts else next(iter(tf_by_node.keys()))

    visited = set()
    serie = None

    while cur and cur in tf_by_node and cur not in visited:
        visited.add(cur)
        serie = tf_by_node[cur] if serie is None else (serie * tf_by_node[cur])
        nxts = out_map.get(cur, [])
        cur = nxts[0] if nxts else None

    return serie

# =====================================================
# APLICAÇÃO PRINCIPAL
# =====================================================

def main():
    st.set_page_config(page_title="Modelagem de Sistemas", layout="wide")
    st.title("📉 Modelagem e Análise de Sistemas de Controle")
    
    inicializar_blocos()
    
    if 'calculo_erro_habilitado' not in st.session_state:
        st.session_state.calculo_erro_habilitado = False
    
    # Sidebar
    criar_sidebar()
    
    # Abas principais
    tab1, tab2 = st.tabs(["📊 Análise", "⚙️ Configurações"])
    
    with tab2:
        st.header("⚙️ Configurações")
        
        st.subheader("🔧 Cálculo de Erro Estacionário")
        st.session_state.calculo_erro_habilitado = st.checkbox(
            "Habilitar Cálculo de Erro",
            value=st.session_state.calculo_erro_habilitado,
            help="Ativa o cálculo de erro estacionário para diferentes entradas"
        )
        
        st.info("💡 Use o botão 'Habilitar Cálculo de Erro' na aba ao lado esquerdo para ativar esta funcionalidade")
    
    with tab1:
        st.header("🧱 Blocos do Sistema")
        
        if st.session_state.blocos.empty:
            st.warning("Nenhum bloco adicionado. Use a sidebar para adicionar blocos.")
        else:
            st.dataframe(st.session_state.blocos[['nome', 'tipo', 'numerador', 'denominador']])
        
        # Opções de exclusão
        with st.sidebar:
            st.header("🗑️ Excluir Blocos")
            if not st.session_state.blocos.empty:
                excluir = st.selectbox("Selecionar", st.session_state.blocos['nome'])
                if st.button("❌ Excluir"):
                    mensagem = remover_bloco(excluir)
                    st.success(mensagem)
            else:
                st.info("Adicione blocos para poder excluir.")
        
        # =========================
        # Editor Visual (Simulink/Xcos) - BETA
        # =========================
        _ensure_flow_state()

        with st.expander("🧩 Editor Visual (arrasta, solta e conecta) — beta", expanded=True):
            if streamlit_flow is None:
                st.warning("Para usar o Editor Visual, instale: `pip install streamlit-flow-component`.")
            else:
                st.caption("Arraste os blocos no canvas e conecte (setas). As conexões ficam salvas na sessão.")
                df_blocos = st.session_state.blocos

                if df_blocos.empty:
                    st.info("Adicione blocos na sidebar primeiro para aparecerem aqui.")
                else:
                    nodes, edges = blocos_para_flow(df_blocos)

                    flow_ret = streamlit_flow(
                        "control_flow",
                        nodes=nodes,
                        edges=edges,
                        height=520,
                        fit_view=True,
                        allow_new_edges=True,
                        show_controls=True,
                        show_mini_map=True,
                        show_background=True,
                    )

                    if flow_ret is not None:
                        st.session_state.flow_state = flow_ret
                        flow_para_edges(flow_ret)

                    colA, colB = st.columns([1, 1])
                    with colA:
                        if st.button("🔁 Resetar layout do canvas"):
                            st.session_state.flow_state = {"nodes": [], "edges": []}
                            st.session_state.flow_edges = []
                            st.session_state.flow_nodes_by_name = {}
                            st.rerun()
                    with colB:
                        st.write(f"Conexões: **{len(st.session_state.get('flow_edges', []))}**")

        # Configuração da Simulação
        col1, col2 = st.columns([2, 1])
        
        with col2:
            st.subheader("🔍 Tipo de Sistema")
            tipo_malha = st.selectbox("Tipo:", ["Malha Aberta", "Malha Fechada"])
            usar_ganho = st.checkbox("Adicionar ganho K ajustável", value=False)
            
            if usar_ganho:
                K = st.slider(
                    "Ganho K",
                    min_value=0.1,
                    max_value=100.0,
                    value=1.0,
                    step=0.1
                )
            else:
                K = 1.0
            
            st.subheader("📥 Entrada")
            entrada = st.selectbox("Tipo de Entrada:", INPUT_SIGNALS)
            amplitude = st.number_input("Amplitude:", value=1.0)
            freq = st.number_input("Frequência (para senoidal):", value=1.0, min_value=0.1)
        
        with col1:
            st.subheader("📈 Análises Disponíveis")
            
            opcoes = ANALYSIS_OPTIONS["malha_aberta"] if tipo_malha == "Malha Aberta" else ANALYSIS_OPTIONS["malha_fechada"]
            analises = st.multiselect("Selecione análises:", opcoes, default=[opcoes[0]])
            
            if st.button("🚀 Executar Análise"):
                if st.session_state.blocos.empty:
                    st.error("Adicione pelo menos um bloco antes de analisar.")
                else:
                    # Obter blocos principais
                    planta = obter_bloco_por_tipo("Planta")
                    controlador = obter_bloco_por_tipo("Controlador")
                    sensor = obter_bloco_por_tipo("Sensor")
                    
                    if planta is None:
                        st.error("Adicione pelo menos uma Planta.")
                        return
                    
                    # Sistema com ganho
                    ganho_tf = ctrl.tf([K], [1])
                    
                    df = st.session_state.blocos
                    
                    # Calcular sistema
                    if tipo_malha == "Malha Aberta":
                        # Se tiver conexões no editor, monta uma série seguindo o fluxo
                        if st.session_state.get("flow_edges"):
                            serie = montar_tf_em_serie_por_flow(df, st.session_state.flow_edges)
                            if serie is None:
                                sistema = ganho_tf * planta
                            else:
                                sistema = ganho_tf * serie
                                st.info("🔗 Malha Aberta montada a partir do Editor Visual (série).")
                        else:
                            sistema = ganho_tf * planta

                        st.info(f"🔧 Sistema em Malha Aberta com K = {K:.2f}")
                    else:
                        planta_com_ganho = ganho_tf * planta
                        sistema = calcular_malha_fechada(planta_com_ganho, controlador, sensor)
                        st.info(f"🔧 Sistema em Malha Fechada com K = {K:.2f}")
                    
                    for analise in analises:
                        st.markdown(f"### 🔎 {analise}")
                        
                        if analise == "Resposta no tempo":
                            fig, t, y = plot_resposta_tempo(sistema, entrada, amplitude, freq)
                            st.plotly_chart(fig, use_container_width=True)
                        
                        elif analise == "Desempenho":
                            fig, t, y = plot_resposta_tempo(sistema, entrada, amplitude, freq)
                            desempenho = calcular_desempenho(t, y)
                            
                            st.subheader("📊 Métricas de Desempenho")
                            for k, v in desempenho.items():
                                st.write(f"**{k}:** {formatar_numero(v)}")
                            
                            if st.session_state.calculo_erro_habilitado:
                                st.subheader("🎯 Erro Estacionário")
                                tipo, Kp, Kv, Ka = constantes_de_erro(sistema)
                                st.write(f"**Tipo do Sistema:** {tipo}")
                                st.write(f"**Kp:** {formatar_numero(Kp)} | **Kv:** {formatar_numero(Kv)} | **Ka:** {formatar_numero(Ka)}")
                                
                                erros = erros_estacionarios(tipo, Kp, Kv, Ka)
                                df_erros = pd.DataFrame(erros, columns=["Entrada", "Erro Estacionário"])
                                df_erros["Erro Estacionário"] = df_erros["Erro Estacionário"].apply(formatar_numero)
                                st.table(df_erros)
                        
                        elif analise == "Diagrama de Polos e Zeros":
                            fig = plot_polos_zeros(sistema)
                            st.plotly_chart(fig, use_container_width=True)
                        
                        elif analise == "Diagrama De Bode Magnitude":
                            fig = plot_bode(sistema, "magnitude")
                            st.plotly_chart(fig, use_container_width=True)
                        
                        elif analise == "Diagrama De Bode Fase":
                            fig = plot_bode(sistema, "fase")
                            st.plotly_chart(fig, use_container_width=True)
                        
                        elif analise == "Nyquist":
                            fig = plot_nyquist(sistema)
                            st.plotly_chart(fig, use_container_width=True)
                        
                        elif analise == "LGR":
                            fig = plot_root_locus(sistema)
                            st.plotly_chart(fig, use_container_width=True)
                        
                        st.divider()

if __name__ == "__main__":
    main()
