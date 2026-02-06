# -*- coding: utf-8 -*-
"""
Sistema de Modelagem e Análise de Sistemas de Controle
Otimizado para Streamlit Cloud

✅ Alteração solicitada:
- Mantém o código original (lógica/estrutura) e adiciona uma NOVA ABA (tab)
  exclusiva para um Editor Visual tipo Xcos/Simulink (drag & drop + conexões),
  usando os mesmos blocos já existentes em st.session_state.blocos.

Dependência extra (Streamlit Cloud):
- streamlit-flow-component
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

# =====================================================
# EDITOR VISUAL (Simulink/Xcos) - Streamlit Flow
# =====================================================
# Requer no requirements.txt: streamlit-flow-component
try:
    from streamlit_flow import streamlit_flow
    from streamlit_flow.interfaces import StreamlitFlowNode
except Exception:
    streamlit_flow = None
    StreamlitFlowNode = None

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
        if tipo == 0:
            Kp = ctrl.dcgain(G_min)
        elif tipo == 1:
            Kv = ctrl.dcgain(s * G_min)
        elif tipo >= 2:
            Ka = ctrl.dcgain(s**2 * G_min)
    except Exception:
        pass

    if tipo == 0:
        Kv = Ka = np.inf
    elif tipo == 1:
        Kp = 0
        Ka = np.inf
    elif tipo >= 2:
        Kp = Kv = 0

    return tipo, Kp, Kv, Ka

def calcular_malha_fechada(planta, controlador=None, sensor=None):
    """Calcula a função de transferência de malha fechada"""
    if controlador is None:
        controlador = TransferFunction([1], [1])
    if sensor is None:
        sensor = TransferFunction([1], [1])

    G = controlador * planta
    H = sensor
    return ctrl.feedback(G, H)

# =====================================================
# ANÁLISE DE SISTEMAS
# =====================================================

def calcular_desempenho(tf):
    """Calcula métricas de desempenho do sistema"""
    den = tf.den[0][0]
    ordem = len(den) - 1
    polos = ctrl.poles(tf)
    gm, pm, wg, wp = margin(tf)
    gm_db = 20 * np.log10(gm) if gm != np.inf and gm > 0 else np.inf

    resultado = {
        'Margem de ganho': f"{formatar_numero(gm)} ({'∞' if gm == np.inf else f'{formatar_numero(gm_db)} dB'})",
        'Margem de fase': f"{formatar_numero(pm)}°",
        'Freq. cruz. fase': f"{formatar_numero(wg)} rad/s",
        'Freq. cruz. ganho': f"{formatar_numero(wp)} rad/s"
    }

    if ordem == 1:
        return _desempenho_ordem1(polos, resultado)
    elif ordem == 2:
        return _desempenho_ordem2(polos, resultado)
    elif ordem >= 3:
        return _desempenho_ordem_superior(polos, ordem, resultado)

def _desempenho_ordem1(polos, resultado):
    """Calcula desempenho para sistemas de 1ª ordem"""
    tau = -1 / polos[0].real
    resultado.update({
        'Tipo': '1ª Ordem',
        'Const. tempo (τ)': f"{formatar_numero(tau)} s",
        'Temp. subida (Tr)': f"{formatar_numero(2.2 * tau)} s",
        'Temp. acomodação (Ts)': f"{formatar_numero(4 * tau)} s",
        'Freq. natural (ωn)': f"{formatar_numero(1/tau)} rad/s",
        'Fator amortec. (ζ)': "1.0"
    })
    return resultado

def _desempenho_ordem2(polos, resultado):
    """Calcula desempenho para sistemas de 2ª ordem"""
    wn = np.sqrt(np.prod(np.abs(polos))).real
    zeta = -np.real(polos[0]) / wn
    wd = wn * np.sqrt(1 - zeta**2) if zeta < 1 else 0
    Mp = np.exp(-zeta * np.pi / np.sqrt(1 - zeta**2)) * 100 if zeta < 1 and zeta > 0 else 0
    Tr = (np.pi - np.arccos(zeta)) / wd if zeta < 1 and wd > 0 else float('inf')
    Tp = np.pi / wd if wd > 0 else float('inf')
    Ts = 4 / (zeta * wn) if zeta * wn > 0 else float('inf')

    resultado.update({
        'Tipo': '2ª Ordem',
        'Freq. natural (ωn)': f"{formatar_numero(wn)} rad/s",
        'Fator amortec. (ζ)': f"{formatar_numero(zeta)}",
        'Freq. amortec. (ωd)': f"{formatar_numero(wd)} rad/s",
        'Sobressinal (Mp)': f"{formatar_numero(Mp)}%",
        'Temp. subida (Tr)': f"{formatar_numero(Tr)} s",
        'Temp. pico (Tp)': f"{formatar_numero(Tp)} s",
        'Temp. acomodação (Ts)': f"{formatar_numero(Ts)} s"
    })
    return resultado

def _desempenho_ordem_superior(polos, ordem, resultado):
    """Calcula desempenho para sistemas de ordem superior"""
    polos_ordenados = sorted(polos, key=lambda p: np.real(p), reverse=True)
    polo_dominante = None
    par_dominante = None

    for i in range(len(polos_ordenados) - 1):
        p1, p2 = polos_ordenados[i], polos_ordenados[i+1]
        if np.isclose(p1.real, p2.real, atol=1e-2) and np.isclose(p1.imag, -p2.imag, atol=1e-2):
            par_dominante = (p1, p2)
            break

    if par_dominante:
        sigma = -np.real(par_dominante[0])
        omega_d = np.abs(np.imag(par_dominante[0]))
        wn = np.sqrt(sigma**2 + omega_d**2)
        zeta = sigma / wn if wn > 0 else 0
    else:
        polo_dominante = polos_ordenados[0]
        wn = np.abs(polo_dominante)
        zeta = -np.real(polo_dominante) / wn if wn != 0 else 0
        omega_d = wn * np.sqrt(1 - zeta**2) if zeta < 1 else 0

    Mp = np.exp(-zeta * np.pi / np.sqrt(1 - zeta**2)) * 100 if zeta < 1 and zeta > 0 else 0
    Tr = (np.pi - np.arccos(zeta)) / omega_d if zeta < 1 and omega_d > 0 else float('inf')
    Tp = np.pi / omega_d if omega_d > 0 else float('inf')
    Ts = 4 / (zeta * wn) if zeta * wn > 0 else float('inf')

    resultado.update({
        'Tipo': f'{ordem}ª Ordem (Par dominante)' if par_dominante else f'{ordem}ª Ordem (Polo dominante)',
        'Freq. natural (ωn)': f"{formatar_numero(wn)} rad/s",
        'Fator amortec. (ζ)': f"{formatar_numero(zeta)}",
        'Freq. amortec. (ωd)': f"{formatar_numero(omega_d)} rad/s",
        'Sobressinal (Mp)': f"{formatar_numero(Mp)} %",
        'Temp. subida (Tr)': f"{formatar_numero(Tr)} s",
        'Temp. pico (Tp)': f"{formatar_numero(Tp)} s",
        'Temp. acomodação (Ts)': f"{formatar_numero(Ts)} s",
        'Observação': 'Cálculo baseado no par dominante' if par_dominante else 'Cálculo baseado no polo dominante'
    })
    return resultado

def estimar_tempo_final_simulacao(tf):
    """Estima o tempo final para simulação baseado nos polos do sistema"""
    polos = ctrl.poles(tf)
    if len(polos) == 0:
        return 50.0
    if any(np.real(p) > 1e-6 for p in polos):
        return 20.0
    partes_reais_estaveis = [np.real(p) for p in polos if np.real(p) < -1e-6]
    if not partes_reais_estaveis:
        return 100.0
    sigma_dominante = max(partes_reais_estaveis)
    ts_estimado = 4 / abs(sigma_dominante)
    tempo_final = ts_estimado * 1.5
    return np.clip(tempo_final, a_min=10, a_max=500)

# =====================================================
# FUNÇÕES DE PLOTAGEM
# =====================================================

def configurar_linhas_interativas(fig):
    """Adiciona suporte para desenhar linhas horizontais e verticais em gráficos"""
    fig.update_layout(
        dragmode='zoom',
        newshape=dict(
            line=dict(color='green', width=2, dash='dash')
        ),
        modebar_add=[
            'drawline',
            'drawopenpath',
            'drawclosedpath',
            'drawcircle',
            'drawrect',
            'eraseshape'
        ]
    )
    return fig

def plot_polos_zeros(tf, fig=None):
    """Diagrama de Polos e Zeros interativo"""
    zeros = ctrl.zeros(tf)
    polos = ctrl.poles(tf)

    if fig is None:
        fig = go.Figure()

    if len(zeros) > 0:
        fig.add_trace(go.Scatter(
            x=np.real(zeros),
            y=np.imag(zeros),
            mode='markers',
            marker=dict(symbol='circle', size=12, color='blue'),
            name='Zeros',
            hovertemplate='Zero<br>Real: %{x:.3f}<br>Imaginário: %{y:.3f}<extra></extra>'
        ))

    if len(polos) > 0:
        fig.add_trace(go.Scatter(
            x=np.real(polos),
            y=np.imag(polos),
            mode='markers',
            marker=dict(symbol='x', size=12, color='red'),
            name='Polos',
            hovertemplate='Polo<br>Real: %{x:.3f}<br>Imaginário: %{y:.3f}<extra></extra>'
        ))

    fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.7)
    fig.add_vline(x=0, line_dash="dash", line_color="gray", opacity=0.7)

    fig.update_layout(
        title='Diagrama de Polos e Zeros (Interativo)',
        xaxis_title='Parte Real',
        yaxis_title='Parte Imaginária',
        showlegend=True,
        hovermode='closest'
    )

    fig = configurar_linhas_interativas(fig)
    return fig

def _gerar_sinal_entrada(entrada, t):
    """Gera sinais de entrada para simulação"""
    sinais = {
        'Degrau': np.ones_like(t),
        'Rampa': t,
        'Senoidal': np.sin(2*np.pi*t),
        'Impulso': np.concatenate([[1], np.zeros(len(t)-1)]),
        'Parabólica': t**2
    }
    return sinais[entrada]

def plot_resposta_temporal(sistema, entrada):
    """Resposta temporal interativa"""
    tempo_final = estimar_tempo_final_simulacao(sistema)
    t = np.linspace(0, tempo_final, 1000)
    u = _gerar_sinal_entrada(entrada, t)

    if entrada == 'Degrau':
        t_out, y = step_response(sistema, t)
    else:
        t_out, y, _ = forced_response(sistema, t, u, return_x=True)

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=t_out,
        y=u[:len(t_out)],
        mode='lines',
        line=dict(dash='dash', color='blue'),
        name='Entrada',
        hovertemplate='Tempo: %{x:.2f}s<br>Entrada: %{y:.3f}<extra></extra>'
    ))

    fig.add_trace(go.Scatter(
        x=t_out,
        y=y,
        mode='lines',
        line=dict(color='red'),
        name='Saída',
        hovertemplate='Tempo: %{x:.2f}s<br>Saída: %{y:.3f}<extra></extra>'
    ))

    fig.update_layout(
        title=f'Resposta Temporal - Entrada: {entrada}',
        xaxis_title='Tempo (s)',
        yaxis_title='Amplitude',
        showlegend=True,
        hovermode='x unified'
    )

    fig = configurar_linhas_interativas(fig)
    return fig, t_out, y

def plot_bode(sistema, tipo='both'):
    """Diagrama de Bode interativo"""
    numerator = sistema.num[0][0]
    denominator = sistema.den[0][0]
    sys = signal.TransferFunction(numerator, denominator)
    w = np.logspace(-3, 3, 1000)
    w, mag, phase = signal.bode(sys, w)

    if tipo == 'both':
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('Diagrama de Bode - Magnitude', 'Diagrama de Bode - Fase'),
            vertical_spacing=0.1
        )

        fig.add_trace(
            go.Scatter(
                x=w, y=mag,
                mode='lines',
                line=dict(color='blue', width=3),
                name='Magnitude',
                hovertemplate='Freq: %{x:.2f} rad/s<br>Magnitude: %{y:.2f} dB<extra></extra>',
                showlegend=False
            ),
            row=1, col=1
        )

        fig.add_trace(
            go.Scatter(
                x=w, y=phase,
                mode='lines',
                line=dict(color='red', width=3),
                name='Fase',
                hovertemplate='Freq: %{x:.2f} rad/s<br>Fase: %{y:.2f}°<extra></extra>',
                showlegend=False
            ),
            row=2, col=1
        )

        fig.update_xaxes(title_text="Frequência (rad/s)", type="log", row=1, col=1)
        fig.update_xaxes(title_text="Frequência (rad/s)", type="log", row=2, col=1)
        fig.update_yaxes(title_text="Magnitude (dB)", row=1, col=1)
        fig.update_yaxes(title_text="Fase (deg)", row=2, col=1)

        fig.update_layout(height=700, title_text="Diagrama de Bode")

    elif tipo == 'magnitude':
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=w, y=mag,
            mode='lines',
            line=dict(color='blue', width=3),
            name='Magnitude',
            hovertemplate='Freq: %{x:.2f} rad/s<br>Magnitude: %{y:.2f} dB<extra></extra>'
        ))
        fig.update_layout(
            title='Diagrama de Bode - Magnitude',
            xaxis_title="Frequência (rad/s)",
            yaxis_title="Magnitude (dB)",
            xaxis_type='log'
        )
    else:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=w, y=phase,
            mode='lines',
            line=dict(color='red', width=3),
            name='Fase',
            hovertemplate='Freq: %{x:.2f} rad/s<br>Fase: %{y:.2f}°<extra></extra>'
        ))
        fig.update_layout(
            title='Diagrama de Bode - Fase',
            xaxis_title="Frequência (rad/s)",
            yaxis_title="Fase (deg)",
            xaxis_type='log'
        )

    fig = configurar_linhas_interativas(fig)
    return fig

def plot_lgr(sistema):
    """Lugar Geométrico das Raízes interativo"""
    rlist, klist = root_locus(sistema, plot=False)

    fig = go.Figure()

    for i, r in enumerate(rlist.T):
        fig.add_trace(go.Scatter(
            x=np.real(r),
            y=np.imag(r),
            mode='lines',
            line=dict(color='blue', width=1),
            name=f'Ramo {i+1}',
            showlegend=False,
            hovertemplate='Real: %{x:.3f}<br>Imaginário: %{y:.3f}<extra></extra>'
        ))

    zeros = ctrl.zeros(sistema)
    polos = ctrl.poles(sistema)

    if len(zeros) > 0:
        fig.add_trace(go.Scatter(
            x=np.real(zeros),
            y=np.imag(zeros),
            mode='markers',
            marker=dict(symbol='circle', size=10, color='green'),
            name='Zeros',
            hovertemplate='Zero<br>Real: %{x:.3f}<br>Imaginário: %{y:.3f}<extra></extra>'
        ))

    if len(polos) > 0:
        fig.add_trace(go.Scatter(
            x=np.real(polos),
            y=np.imag(polos),
            mode='markers',
            marker=dict(symbol='x', size=12, color='red'),
            name='Polos',
            hovertemplate='Polo<br>Real: %{x:.3f}<br>Imaginário: %{y:.3f}<extra></extra>'
        ))

    fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.7)
    fig.add_vline(x=0, line_dash="dash", line_color="gray", opacity=0.7)

    fig.update_layout(
        title='Lugar Geométrico das Raízes (LGR)',
        xaxis_title='Parte Real',
        yaxis_title='Parte Imaginária',
        showlegend=True,
        hovermode='closest'
    )

    fig = configurar_linhas_interativas(fig)
    return fig

def plot_nyquist(sistema):
    """Diagrama de Nyquist interativo"""
    sistema_scipy = signal.TransferFunction(sistema.num[0][0], sistema.den[0][0])
    w = np.logspace(-2, 2, 1000)
    _, H = signal.freqresp(sistema_scipy, w)

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=H.real,
        y=H.imag,
        mode='lines',
        line=dict(color='blue', width=2),
        name='Nyquist',
        hovertemplate='Real: %{x:.3f}<br>Imaginário: %{y:.3f}<extra></extra>'
    ))

    fig.add_trace(go.Scatter(
        x=H.real,
        y=-H.imag,
        mode='lines',
        line=dict(dash='dash', color='gray', width=1),
        name='Reflexo simétrico',
        hovertemplate='Real: %{x:.3f}<br>Imaginário: %{y:.3f}<extra></extra>'
    ))

    fig.add_trace(go.Scatter(
        x=[-1],
        y=[0],
        mode='markers',
        marker=dict(symbol='circle', size=12, color='red'),
        name='Ponto crítico (-1,0)',
        hovertemplate='Ponto Crítico<br>Real: -1<br>Imaginário: 0<extra></extra>'
    ))

    fig.add_hline(y=0, line_color="black", line_width=1)
    fig.add_vline(x=0, line_color="black", line_width=1)

    fig.update_layout(
        title='Diagrama de Nyquist',
        xaxis_title='Parte Real',
        yaxis_title='Parte Imaginária',
        showlegend=True,
        hovermode='closest'
    )

    fig = configurar_linhas_interativas(fig)

    polos = ctrl.poles(sistema)
    polos_spd = sum(1 for p in polos if np.real(p) > 0)
    voltas = 0
    Z = polos_spd + voltas

    return fig, polos_spd, voltas, Z

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
        tf, tf_symb = converter_para_tf(numerador, denominador)
        novo = pd.DataFrame([{
            'nome': nome,
            'tipo': tipo,
            'numerador': numerador,
            'denominador': denominador,
            'tf': tf,
            'tf_simbolico': tf_symb
        }])
        st.session_state.blocos = pd.concat([st.session_state.blocos, novo], ignore_index=True)
        return True, f"Bloco {nome} adicionado."
    except Exception as e:
        return False, f"Erro na conversão: {e}"

def remover_bloco(nome):
    """Remove um bloco pelo nome"""
    st.session_state.blocos = st.session_state.blocos[st.session_state.blocos['nome'] != nome]
    return f"Bloco {nome} excluído."

def obter_bloco_por_tipo(tipo):
    """Obtém o primeiro bloco de um tipo específico"""
    df = st.session_state.blocos
    if any(df['tipo'] == tipo):
        return df[df['tipo'] == tipo].iloc[0]['tf']
    return None

# =====================================================
# EDITOR VISUAL - Funções (aba separada)
# =====================================================

def _ensure_flow_state():
    if "flow_state" not in st.session_state:
        st.session_state.flow_state = {"nodes": [], "edges": []}
    if "flow_edges" not in st.session_state:
        st.session_state.flow_edges = []
    if "flow_nodes_by_name" not in st.session_state:
        st.session_state.flow_nodes_by_name = {}  # nome_bloco -> node_id

def blocos_para_flow(df: pd.DataFrame):
    """Cria nós do editor visual a partir de st.session_state.blocos."""
    _ensure_flow_state()

    nodes = []
    edges = st.session_state.flow_state.get("edges", [])

    # cria/recupera ids por nome
    for i, row in df.iterrows():
        nome = str(row["nome"])
        tipo = str(row["tipo"])

        if nome not in st.session_state.flow_nodes_by_name:
            st.session_state.flow_nodes_by_name[nome] = f"node-{uuid.uuid4().hex[:8]}"
        node_id = st.session_state.flow_nodes_by_name[nome]

        # reaproveitar posição anterior
        prev_pos = None
        for n in st.session_state.flow_state.get("nodes", []):
            if isinstance(n, dict) and n.get("id") == node_id:
                prev_pos = n.get("position")
                break

        x = prev_pos["x"] if prev_pos else 60 + i * 220
        y = prev_pos["y"] if prev_pos else 100

        label = f"{nome}\n[{tipo}]\n{row['numerador']}/{row['denominador']}"

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

    cleaned_edges = []
    for e in edges:
        if isinstance(e, dict):
            if e.get("source") in valid_ids and e.get("target") in valid_ids:
                cleaned_edges.append(e)

    return nodes, cleaned_edges

def flow_para_edges(flow_return):
    """Salva edges simples em session_state.flow_edges"""
    edges_out = []
    for e in flow_return.get("edges", []):
        edges_out.append({"source": e["source"], "target": e["target"]})
    st.session_state.flow_edges = edges_out
    return edges_out

def montar_tf_em_serie_por_flow(df: pd.DataFrame, edges_simple: list):
    """
    Monta uma TF em série seguindo o 'tronco' principal do grafo:
    - encontra um nó com in-degree 0 e segue as conexões.
    """
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

def render_editor_visual_tab():
    """Aba separada: Editor Visual (drag/drop + conexões)"""
    st.subheader("🧩 Editor Visual de Blocos (tipo Xcos/Simulink)")

    if streamlit_flow is None or StreamlitFlowNode is None:
        st.warning("Editor Visual indisponível. No Streamlit Cloud, adicione `streamlit-flow-component` no requirements.txt.")
        st.code("pip install streamlit-flow-component")
        return

    _ensure_flow_state()
    df = st.session_state.blocos

    if df.empty:
        st.info("Adicione blocos na sidebar (Planta/Controlador/Sensor/Outro) para aparecerem aqui.")
        return

    st.caption("Arraste os blocos no canvas e conecte pelas alças (setas). As conexões ficam salvas para a simulação.")

    nodes, edges = blocos_para_flow(df)

    flow_ret = streamlit_flow(
        "control_flow",
        nodes=nodes,
        edges=edges,
        height=600,
        fit_view=True,
        allow_new_edges=True,
        show_controls=True,
        show_mini_map=True,
        show_background=True,
    )

    if flow_ret is not None:
        st.session_state.flow_state = flow_ret
        flow_para_edges(flow_ret)

    c1, c2, c3 = st.columns([1, 1, 2])
    with c1:
        if st.button("🔁 Resetar layout do canvas"):
            st.session_state.flow_state = {"nodes": [], "edges": []}
            st.session_state.flow_edges = []
            st.session_state.flow_nodes_by_name = {}
            st.rerun()
    with c2:
        st.metric("Conexões", len(st.session_state.get("flow_edges", [])))
    with c3:
        st.write("✅ O diagrama é usado automaticamente na **Malha Aberta** se existir ao menos 1 conexão (série).")

    with st.expander("📌 Ver conexões (debug)"):
        st.json(st.session_state.get("flow_edges", []))

# =====================================================
# APLICAÇÃO PRINCIPAL
# =====================================================

def main():
    st.set_page_config(page_title="Modelagem de Sistemas", layout="wide")
    st.title("📉 Modelagem e Análise de Sistemas de Controle")

    inicializar_blocos()

    if 'calculo_erro_habilitado' not in st.session_state:
        st.session_state.calculo_erro_habilitado = False

    if 'mostrar_ajuda' not in st.session_state:
        st.session_state.mostrar_ajuda = False

    # Sidebar (mantida original)
    with st.sidebar:
        st.header("🧱 Adicionar Blocos")
        nome = st.text_input("Nome", value="G1")
        tipo = st.selectbox("Tipo", ['Planta', 'Controlador', 'Sensor', 'Outro'])
        numerador = st.text_input("Numerador", placeholder="ex: 4*s")
        denominador = st.text_input("Denominador", placeholder="ex: s^2 + 2*s + 3")

        if st.button("➕ Adicionar"):
            sucesso, mensagem = adicionar_bloco(nome, tipo, numerador, denominador)
            if sucesso:
                st.success(mensagem)
            else:
                st.error(mensagem)

        if not st.session_state.blocos.empty:
            st.header("🗑️ Excluir Blocos")
            excluir = st.selectbox("Selecionar", st.session_state.blocos['nome'])
            if st.button("❌ Excluir"):
                mensagem = remover_bloco(excluir)
                st.success(mensagem)

        st.header("⚙️ Configurações")
        if st.button("🔢 Habilitar Cálculo de Erro" if not st.session_state.calculo_erro_habilitado else "❌ Desabilitar Cálculo de Erro"):
            st.session_state.calculo_erro_habilitado = not st.session_state.calculo_erro_habilitado
            st.rerun()

    # ✅ NOVA ESTRUTURA: abas (a simulação fica como estava, editor em aba separada)
    tab_sim, tab_editor = st.tabs(["📈 Simulação", "🧩 Editor Visual"])

    # =======================
    # TAB: SIMULAÇÃO (original)
    # =======================
    with tab_sim:
        # Cálculo de Erro (mantido original)
        if st.session_state.calculo_erro_habilitado:
            st.subheader("📊 Cálculo de Erro Estacionário")
            col1, col2 = st.columns(2)
            with col1:
                num_erro = st.text_input("Numerador", value="", key="num_erro")
            with col2:
                den_erro = st.text_input("Denominador", value="", key="den_erro")

            btn_col1, btn_col2 = st.columns(2)
            with btn_col1:
                if st.button("🔍 Calcular Erro Estacionário"):
                    try:
                        G, _ = converter_para_tf(num_erro, den_erro)
                        tipo_sis, Kp, Kv, Ka = constantes_de_erro(G)

                        df_res = pd.DataFrame([{"Tipo": tipo_sis, "Kp": Kp, "Kv": Kv, "Ka": Ka}])
                        st.subheader("📊 Resultado")
                        st.dataframe(
                            df_res.style.format({
                                "Kp": lambda x: formatar_numero(x),
                                "Kv": lambda x: formatar_numero(x),
                                "Ka": lambda x: formatar_numero(x)
                            }),
                            height=120,
                            use_container_width=True
                        )
                    except Exception as e:
                        st.error(f"Erro: {str(e)}")

            with btn_col2:
                if st.button("🗑️ Remover Planta", key="remover_planta"):
                    if not st.session_state.blocos.empty:
                        st.session_state.blocos = st.session_state.blocos[st.session_state.blocos['tipo'] != 'Planta']
                        st.success("Plantas removidas com sucesso!")
                    else:
                        st.warning("Nenhuma planta para remover")
        else:
            st.info("💡 Use o botão 'Habilitar Cálculo de Erro' na barra lateral para ativar esta funcionalidade")

        # Configuração da Simulação (mantida original)
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
                    step=0.1,
                    key="ganho_k_slider",
                    help="Ajuste o ganho K do sistema"
                )
                st.info(f"✅ Ganho K aplicado: {K:.2f}")
            else:
                K = 1.0

            st.subheader("📊 Análises desejadas")
            analise_opcoes = ANALYSIS_OPTIONS["malha_fechada" if tipo_malha == "Malha Fechada" else "malha_aberta"]
            analises = st.multiselect("Escolha:", analise_opcoes, default=analise_opcoes[0])
            entrada = st.selectbox("Sinal de Entrada", INPUT_SIGNALS)

        # Simulação Principal (mantida original, com "gancho" opcional do editor só para malha aberta)
        with col1:
            st.subheader("📈 Resultados da Simulação")

            col_sim, col_ajuda = st.columns([2, 1])

            with col_sim:
                if st.button("▶️ Executar Simulação", use_container_width=True):
                    try:
                        df = st.session_state.blocos
                        if df.empty:
                            st.warning("Adicione blocos primeiro.")
                            st.stop()

                        planta = obter_bloco_por_tipo('Planta')
                        controlador = obter_bloco_por_tipo('Controlador')
                        sensor = obter_bloco_por_tipo('Sensor')

                        if planta is None:
                            st.error("Adicione pelo menos um bloco do tipo Planta.")
                            st.stop()

                        ganho_tf = TransferFunction([K], [1])

                        # ✅ Mantém lógica original; se existir diagrama (conexões), usa na Malha Aberta (série)
                        if tipo_malha == "Malha Aberta":
                            sistema_base = planta
                            if st.session_state.get("flow_edges"):
                                serie = montar_tf_em_serie_por_flow(df, st.session_state.flow_edges)
                                if serie is not None:
                                    sistema_base = serie
                                    st.info("🔗 Malha Aberta montada pelo Editor Visual (série).")

                            sistema = ganho_tf * sistema_base
                            st.info(f"🔧 Sistema em Malha Aberta com K = {K:.2f}")
                        else:
                            planta_com_ganho = ganho_tf * planta
                            sistema = calcular_malha_fechada(planta_com_ganho, controlador, sensor)
                            st.info(f"🔧 Sistema em Malha Fechada com K = {K:.2f}")

                        for analise in analises:
                            st.markdown(f"### 🔎 {analise}")

                            if analise == 'Resposta no tempo':
                                fig, t_out, y = plot_resposta_temporal(sistema, entrada)
                                st.plotly_chart(fig, use_container_width=True)
                                st.caption("💡 Use as ferramentas de desenho na barra superior")

                            elif analise == 'Desempenho':
                                desempenho = calcular_desempenho(sistema)
                                for chave, valor in desempenho.items():
                                    st.markdown(f"**{chave}:** {valor}")

                            elif analise == 'Diagrama De Bode Magnitude':
                                fig = plot_bode(sistema, 'magnitude')
                                st.plotly_chart(fig, use_container_width=True)

                            elif analise == 'Diagrama De Bode Fase':
                                fig = plot_bode(sistema, 'fase')
                                st.plotly_chart(fig, use_container_width=True)

                            elif analise == 'Diagrama de Polos e Zeros':
                                fig = plot_polos_zeros(sistema)
                                st.plotly_chart(fig, use_container_width=True)

                            elif analise == 'LGR':
                                fig = plot_lgr(sistema)
                                st.plotly_chart(fig, use_container_width=True)

                            elif analise == 'Nyquist':
                                fig, polos_spd, voltas, Z = plot_nyquist(sistema)
                                st.markdown(f"**Polos no semiplano direito (P):** {polos_spd}")
                                st.markdown(f"**Voltas no -1 (N):** {voltas}")
                                st.markdown(f"**Z = N + P = {Z} → {'✅ Estável' if Z == 0 else '❌ Instável'}**")
                                st.plotly_chart(fig, use_container_width=True)

                    except Exception as e:
                        st.error(f"Erro durante a simulação: {e}")
                        import traceback
                        st.code(traceback.format_exc())

            with col_ajuda:
                if st.button("❓ Ajuda", use_container_width=True):
                    st.session_state.mostrar_ajuda = True

        # Modal de Ajuda (mantido original)
        if st.session_state.mostrar_ajuda:
            with st.container():
                st.markdown("---")
                st.subheader("🎯 Guia de Uso - Sistema de Análise de Controle")

                col_guide1, col_guide2 = st.columns(2)

                with col_guide1:
                    st.markdown("### 📋 Passo a Passo")
                    st.markdown("""
                    1. **🧱 Adicionar Blocos**: Na barra lateral, adicione pelo menos um bloco do tipo **Planta**
                    2. **🔧 Configurar**: Escolha o tipo de sistema (Malha Aberta/Fechada)
                    3. **📊 Selecionar Análises**: Escolha quais gráficos e análises deseja ver
                    4. **🎛️ Ajustar Parâmetros**: Use o ganho K se necessário
                    5. **▶️ Executar**: Clique em "Executar Simulação" para ver os resultados
                    6. **✏️ Desenhar**: Use as ferramentas de desenho nos gráficos
                    """)

                    st.markdown("### 🔍 Exemplos de Funções de Transferência")
                    st.markdown("""
                    - **1ª Ordem**: `1/(s+1)`
                    - **2ª Ordem**: `1/(s^2 + 2*s + 1)`
                    - **Integrador**: `1/s`
                    - **Sistema com zero**: `(s+1)/(s^2 + 2*s + 2)`
                    """)

                with col_guide2:
                    st.markdown("### 🔍 Tipos de Análise")
                    st.markdown("""
                    - **Resposta no tempo**: Comportamento temporal do sistema
                    - **Desempenho**: Métricas como sobressinal, tempo de acomodação
                    - **Diagrama de Bode**: Resposta em frequência
                    - **Polos e Zeros**: Estabilidade do sistema
                    - **LGR**: Lugar Geométrico das Raízes
                    - **Nyquist**: Análise de estabilidade
                    """)

                    st.markdown("### ⚠️ Dicas Importantes")
                    st.markdown("""
                    - Sempre comece adicionando uma **Planta**
                    - Use **Controlador** para sistemas em malha fechada
                    - O **Sensor** é opcional (padrão = 1)
                    - Verifique se a função de transferência está correta
                    """)

                if st.button("✅ Entendi, Fechar Ajuda"):
                    st.session_state.mostrar_ajuda = False
                    st.rerun()

    # =======================
    # TAB: EDITOR VISUAL
    # =======================
    with tab_editor:
        render_editor_visual_tab()

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 💡 Dica Rápida")
    st.sidebar.info("""
    Comece adicionando uma **Planta** e
    execute a simulação para ver os
    resultados básicos.

    🧩 **Editor Visual**: Use a aba "Editor Visual"
    para montar a série em Malha Aberta.
    """)

if __name__ == "__main__":
    main()
