# -*- coding: utf-8 -*-
"""
Sistema de Modelagem e Análise de Sistemas de Controle
Com Editor Visual de Diagrama de Blocos (estilo Xcos)
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
import streamlit.components.v1 as components
import json
import os
from collections import deque

# =====================================================
# CONFIGURAÇÕES E CONSTANTES
# =====================================================

ANALYSIS_OPTIONS = {
    "malha_aberta": ["Resposta no tempo", "Desempenho", "Diagrama de Polos e Zeros",
                    "Diagrama De Bode Magnitude", "Diagrama De Bode Fase", "Nyquist"],
    "malha_fechada": ["Resposta no tempo", "Desempenho", "Diagrama de Polos e Zeros",
                     "Diagrama De Bode Magnitude", "Diagrama De Bode Fase", "LGR"]
}

INPUT_SIGNALS = ['Degrau', 'Rampa', 'Senoidal', 'Impulso', 'Parabólica']

# =====================================================
# COMPONENTE VISUAL (bidirecional)
# =====================================================

_VISUAL_EDITOR_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "visual_blocks_frontend")
visual_blocks_editor = components.declare_component("visual_blocks_editor", path=_VISUAL_EDITOR_DIR)

def editor_visual_component(model_json="", key=None, height=700):
    """Componente bidirecional do editor visual de blocos.
    Envia o modelo atual para o JS e recebe o modelo atualizado."""
    component_value = visual_blocks_editor(model=model_json, key=key, default=None, height=height)
    return component_value

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

def format_poly(coeffs):
    """Formata coeficientes de polinômio como string LaTeX"""
    s = sp.Symbol('s')
    if len(coeffs) == 0:
        return "0"
    poly = sum(float(c) * s**(len(coeffs)-1-i) for i, c in enumerate(coeffs))
    return sp.latex(sp.nsimplify(poly, rational=False))

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
# GERENCIAMENTO DE BLOCOS (modo clássico)
# =====================================================

def inicializar_blocos():
    """Inicializa o estado dos blocos se não existir"""
    if 'blocos' not in st.session_state:
        st.session_state.blocos = pd.DataFrame(columns=['nome', 'tipo', 'numerador', 'denominador', 'tf', 'tf_simbolico'])
    if 'visual_model' not in st.session_state:
        st.session_state.visual_model = {"nodes": [], "edges": []}

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
# PROCESSADOR DE DIAGRAMA VISUAL
# =====================================================

def node_to_tf(node):
    """Converte um nó do diagrama visual em TransferFunction"""
    t = node.get('type', '')
    p = node.get('params', {})

    if t == 'tf':
        tf_obj, _ = converter_para_tf(p.get('num', '1'), p.get('den', '1'))
        return tf_obj
    elif t == 'gain':
        k = float(p.get('k', '1'))
        return TransferFunction([k], [1])
    elif t == 'int':
        return TransferFunction([1], [1, 0])
    elif t == 'der':
        return TransferFunction([1, 0], [1])
    elif t == 'pid':
        kp = float(p.get('kp', '1'))
        ki = float(p.get('ki', '0'))
        kd = float(p.get('kd', '0'))
        if ki == 0:
            return TransferFunction([kd, kp], [1])
        return TransferFunction([kd, kp, ki], [1, 0])
    elif t in ('input', 'output', 'branch'):
        return TransferFunction([1], [1])
    elif t == 'sat':
        return TransferFunction([1], [1])
    else:
        return TransferFunction([1], [1])

def _build_adjacency(nodes_dict, edges):
    """Constrói listas de adjacência do grafo"""
    adj_out = {nid: [] for nid in nodes_dict}
    adj_in = {nid: [] for nid in nodes_dict}
    for e in edges:
        src, dst = e.get('src'), e.get('dst')
        if src in nodes_dict and dst in nodes_dict:
            adj_out[src].append((dst, e.get('dstPort', 'in0'), e))
            adj_in[dst].append((src, e.get('dstPort', 'in0'), e))
    return adj_out, adj_in

def _topological_sort(nodes_dict, edges):
    """Ordenação topológica do grafo (retorna None se há ciclo)"""
    in_degree = {nid: 0 for nid in nodes_dict}
    adj = {nid: [] for nid in nodes_dict}
    for e in edges:
        src, dst = e.get('src'), e.get('dst')
        if src in nodes_dict and dst in nodes_dict:
            adj[src].append(dst)
            in_degree[dst] += 1

    queue = deque([nid for nid, deg in in_degree.items() if deg == 0])
    result = []
    while queue:
        nid = queue.popleft()
        result.append(nid)
        for neighbor in adj[nid]:
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)

    if len(result) != len(nodes_dict):
        return None  # Ciclo detectado
    return result

def _trace_forward_path(start_id, end_id, adj_out, nodes_dict, node_tfs, exclude_types=None):
    """Traça caminho de start_id a end_id multiplicando TFs.
    Retorna (TransferFunction, [path_ids]) ou (None, [])"""
    if exclude_types is None:
        exclude_types = set()

    queue = deque([(start_id, [start_id])])
    visited = {start_id}

    while queue:
        current, path = queue.popleft()
        if current == end_id:
            result_tf = TransferFunction([1], [1])
            for nid in path:
                ntype = nodes_dict[nid].get('type', '')
                if ntype not in ('input', 'output', 'branch', 'sum') and ntype not in exclude_types:
                    result_tf = result_tf * node_tfs.get(nid, TransferFunction([1], [1]))
            return result_tf, path

        for (neighbor, port, edge) in adj_out.get(current, []):
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, path + [neighbor]))

    return None, []

def _detect_canonical_feedback(model, nodes_dict, edges, adj_out, adj_in, node_tfs):
    """Detecta o padrão canônico de malha fechada:
    Input -> Sum -> [G forward] -> Output
               ^                     |
               +------ [H] <--------+
    Retorna (G_forward, H_feedback, sign) ou None"""

    input_nodes = [n for n in nodes_dict.values() if n.get('type') == 'input']
    output_nodes = [n for n in nodes_dict.values() if n.get('type') == 'output']
    sum_nodes = [n for n in nodes_dict.values() if n.get('type') == 'sum']

    if not sum_nodes:
        return None
    if len(sum_nodes) != 1:
        return None  # Só tratamos 1 somador por enquanto

    sum_node = sum_nodes[0]
    sum_id = sum_node['id']
    signs = sum_node.get('params', {}).get('signs', '+ -').strip().split()

    # Encontrar saída do somador -> caminho direto até output
    sum_outputs = adj_out.get(sum_id, [])
    if not sum_outputs:
        return None

    # Determinar nó de destino (output ou último nó)
    end_id = None
    if output_nodes:
        end_id = output_nodes[0]['id']
    else:
        # Sem nó output explícito: encontrar nó terminal (sem saídas)
        for nid in nodes_dict:
            if nodes_dict[nid].get('type') not in ('input', 'sum', 'branch'):
                if not adj_out.get(nid, []) or all(
                    nodes_dict.get(dst, {}).get('type') == 'sum' for dst, _, _ in adj_out.get(nid, [])
                ):
                    end_id = nid
        if end_id is None:
            return None

    # Traçar caminho direto: Sum -> ... -> Output
    G_forward, forward_path = _trace_forward_path(sum_id, end_id, adj_out, nodes_dict, node_tfs)
    if G_forward is None:
        return None

    # Procurar caminho de realimentação: algum nó no forward_path envia sinal de volta ao Sum
    # Verificar entradas do somador que não vêm do input
    sum_inputs = adj_in.get(sum_id, [])

    feedback_tf = None
    feedback_sign = -1

    for idx, (src_id, port, edge) in enumerate(sum_inputs):
        # Se a fonte desta entrada é um nó no caminho direto ou alcançável a partir dele,
        # é o caminho de realimentação
        src_node = nodes_dict.get(src_id, {})
        if src_node.get('type') == 'input':
            continue  # Entrada de referência, não feedback

        # É um nó de feedback: traçar o caminho de feedback
        # Encontrar de onde vem: do branch/output até este src
        # O feedback pode ser um único bloco ou uma cadeia

        # Verificar se o src é um branch ou está no forward path
        # Caso simples: src é um branch que vem do forward path
        if src_id in node_tfs and src_node.get('type') not in ('input', 'sum'):
            feedback_tf = node_tfs.get(src_id, TransferFunction([1], [1]))
            if src_node.get('type') in ('branch',):
                feedback_tf = TransferFunction([1], [1])

        # Determinar o sinal desta entrada do somador
        # Mapear port para index
        port_str = port if port else 'in0'
        try:
            port_idx = int(port_str.replace('in', ''))
        except (ValueError, AttributeError):
            port_idx = idx

        if port_idx < len(signs):
            feedback_sign = -1 if signs[port_idx] == '-' else 1

    if feedback_tf is None:
        # Sem realimentação encontrada - pode ser que o feedback é unitário
        # Verificar se há alguma conexão de volta ao somador que não seja do input
        has_non_input_feedback = False
        for (src_id, port, edge) in sum_inputs:
            if nodes_dict.get(src_id, {}).get('type') != 'input':
                has_non_input_feedback = True
                # Realimentação unitária (direto de um branch)
                feedback_tf = TransferFunction([1], [1])
                break

        if not has_non_input_feedback:
            # Sem realimentação - é um sistema em malha aberta passando pelo somador
            return None

    return G_forward, feedback_tf, feedback_sign

def processar_diagrama_visual(model):
    """Processa o diagrama visual e calcula o sistema equivalente.

    Retorna:
        (G_open_loop, G_closed_loop, details_dict) ou (None, None, "mensagem de erro")
    """
    if not model or not model.get('nodes'):
        return None, None, "Nenhum bloco no diagrama. Adicione blocos e conecte-os."

    nodes = model['nodes']
    edges = model.get('edges', [])

    # Filtrar nós válidos
    nodes_dict = {n['id']: n for n in nodes}

    # Converter todos os nós para TFs
    node_tfs = {}
    for nid, node in nodes_dict.items():
        try:
            tf_val = node_to_tf(node)
            if tf_val is not None:
                node_tfs[nid] = tf_val
        except Exception as e:
            return None, None, f"Erro ao converter bloco '{node.get('label', nid)}': {e}"

    # Construir adjacência
    adj_out, adj_in = _build_adjacency(nodes_dict, edges)

    # Contar blocos reais (excluindo input/output/branch)
    blocos_reais = [n for n in nodes if n.get('type') not in ('input', 'output', 'branch', 'sum')]

    if not blocos_reais:
        return None, None, "Nenhum bloco de transferência no diagrama."

    # ---- Estratégia 1: Detectar malha fechada canônica ----
    result = _detect_canonical_feedback(model, nodes_dict, edges, adj_out, adj_in, node_tfs)
    if result is not None:
        G_forward, H_feedback, fb_sign = result
        G_open = G_forward
        try:
            G_closed = ctrl.feedback(G_forward, H_feedback, sign=fb_sign)
        except Exception:
            G_closed = G_forward

        desc = f"Malha fechada detectada ({len(blocos_reais)} blocos)"
        if fb_sign == -1:
            desc += " - realimentação negativa"
        else:
            desc += " - realimentação positiva"

        return G_open, G_closed, {
            'forward': G_forward,
            'feedback': H_feedback,
            'open_loop': G_open,
            'closed_loop': G_closed,
            'description': desc,
            'tipo': 'malha_fechada'
        }

    # ---- Estratégia 2: Série simples (sem feedback) ----
    topo_order = _topological_sort(nodes_dict, edges)
    if topo_order is not None:
        # Sem ciclo - multiplicar TFs na ordem topológica
        result_tf = TransferFunction([1], [1])
        count = 0
        for nid in topo_order:
            ntype = nodes_dict[nid].get('type', '')
            if ntype not in ('input', 'output', 'branch', 'sum') and nid in node_tfs:
                result_tf = result_tf * node_tfs[nid]
                count += 1

        if count == 0:
            return None, None, "Nenhum bloco de transferência processável."

        G_open = result_tf
        try:
            G_closed = ctrl.feedback(result_tf, 1)
        except Exception:
            G_closed = result_tf

        return G_open, G_closed, {
            'forward': result_tf,
            'feedback': None,
            'open_loop': G_open,
            'closed_loop': G_closed,
            'description': f"Sistema em série com {count} blocos (sem realimentação)",
            'tipo': 'malha_aberta'
        }

    # ---- Estratégia 3: Fallback - multiplicar todos ----
    result_tf = TransferFunction([1], [1])
    count = 0
    for node in nodes:
        ntype = node.get('type', '')
        if ntype not in ('input', 'output', 'branch', 'sum') and node['id'] in node_tfs:
            result_tf = result_tf * node_tfs[node['id']]
            count += 1

    if count == 0:
        return None, None, "Não foi possível processar o diagrama."

    G_open = result_tf
    try:
        G_closed = ctrl.feedback(result_tf, 1)
    except Exception:
        G_closed = result_tf

    return G_open, G_closed, {
        'forward': result_tf,
        'feedback': None,
        'open_loop': G_open,
        'closed_loop': G_closed,
        'description': f"Sistema com {count} blocos (topologia com ciclo - aproximação em série)",
        'tipo': 'malha_aberta'
    }

# =====================================================
# FUNÇÕES DE EXIBIÇÃO
# =====================================================

def mostrar_tf_equivalente(G_open, G_closed, details):
    """Exibe as funções de transferência equivalentes e informações do sistema"""
    if isinstance(details, str):
        st.error(details)
        return

    desc = details.get('description', '')
    if desc:
        st.info(f"**Topologia detectada:** {desc}")

    col_tf1, col_tf2 = st.columns(2)

    with col_tf1:
        st.markdown("**G(s) - Malha Aberta:**")
        if G_open is not None:
            try:
                num_str = format_poly(G_open.num[0][0])
                den_str = format_poly(G_open.den[0][0])
                st.latex(f"G(s) = \\frac{{{num_str}}}{{{den_str}}}")
            except Exception:
                st.code(str(G_open))

    with col_tf2:
        st.markdown("**T(s) - Malha Fechada:**")
        if G_closed is not None:
            try:
                num_str = format_poly(G_closed.num[0][0])
                den_str = format_poly(G_closed.den[0][0])
                st.latex(f"T(s) = \\frac{{{num_str}}}{{{den_str}}}")
            except Exception:
                st.code(str(G_closed))

    # Informações do sistema
    if G_open is not None:
        try:
            tipo, Kp, Kv, Ka = constantes_de_erro(G_open)
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Tipo do Sistema", str(tipo))
            with col2:
                st.metric("Kp", formatar_numero(Kp))
            with col3:
                st.metric("Kv", formatar_numero(Kv))
            with col4:
                st.metric("Ka", formatar_numero(Ka))
        except Exception:
            pass

def executar_analises(sistema, analises, entrada):
    """Executa e exibe todas as análises selecionadas para um sistema.
    Função compartilhada entre modo clássico e visual."""
    for analise in analises:
        st.markdown(f"### {analise}")

        if analise == 'Resposta no tempo':
            fig, t_out, y = plot_resposta_temporal(sistema, entrada)
            st.plotly_chart(fig, use_container_width=True)

        elif analise == 'Desempenho':
            desempenho = calcular_desempenho(sistema)
            if desempenho:
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
            st.markdown(f"**Polos SPD (P):** {polos_spd}")
            st.markdown(f"**Voltas (N):** {voltas}")
            st.markdown(f"**Z = {Z} → {'Estável' if Z == 0 else 'Instável'}**")
            st.plotly_chart(fig, use_container_width=True)

# =====================================================
# APLICAÇÃO PRINCIPAL
# =====================================================

def main():
    st.set_page_config(page_title="Modelagem de Sistemas", layout="wide")
    st.title("Modelagem e Análise de Sistemas de Controle")

    inicializar_blocos()

    if 'calculo_erro_habilitado' not in st.session_state:
        st.session_state.calculo_erro_habilitado = False

    if 'mostrar_ajuda' not in st.session_state:
        st.session_state.mostrar_ajuda = False

    if 'modo_editor' not in st.session_state:
        st.session_state.modo_editor = 'classico'

    # Seletor de modo
    st.sidebar.header("Modo de Trabalho")
    modo = st.sidebar.radio(
        "Escolha o modo:",
        ['Clássico (Lista)', 'Editor Visual (Xcos)'],
        index=0 if st.session_state.modo_editor == 'classico' else 1,
        help="Clássico: adicionar blocos por formulário\nEditor Visual: arrastar e conectar blocos graficamente"
    )

    if modo == 'Editor Visual (Xcos)':
        st.session_state.modo_editor = 'visual'
    else:
        st.session_state.modo_editor = 'classico'

    # =====================================================
    # MODO EDITOR VISUAL
    # =====================================================
    if st.session_state.modo_editor == 'visual':
        st.subheader("Editor Visual de Diagrama de Blocos")

        # Renderizar componente bidirecional
        current_model_json = json.dumps(st.session_state.visual_model)
        result = editor_visual_component(model_json=current_model_json, key="visual_editor")

        # Receber estado atualizado do JS
        if result is not None:
            try:
                updated_model = json.loads(result) if isinstance(result, str) else result
                if isinstance(updated_model, dict) and 'nodes' in updated_model:
                    st.session_state.visual_model = updated_model
            except (json.JSONDecodeError, TypeError):
                pass

        # Sidebar: configurações de análise para modo visual
        with st.sidebar:
            st.markdown("---")
            st.header("Configurações de Análise")

            tipo_malha_visual = st.selectbox(
                "Tipo de Análise:",
                ["Malha Aberta", "Malha Fechada"],
                key="tipo_malha_visual"
            )

            usar_ganho_visual = st.checkbox(
                "Adicionar ganho K ajustável",
                value=False,
                key="ganho_visual_check"
            )

            if usar_ganho_visual:
                K_visual = st.slider(
                    "Ganho K", 0.1, 100.0, 1.0, 0.1,
                    key="ganho_k_visual"
                )
            else:
                K_visual = 1.0

            analise_opcoes_visual = ANALYSIS_OPTIONS[
                "malha_fechada" if tipo_malha_visual == "Malha Fechada" else "malha_aberta"
            ]
            analises_visual = st.multiselect(
                "Análises:",
                analise_opcoes_visual,
                default=[analise_opcoes_visual[0]],
                key="analises_visual"
            )

            entrada_visual = st.selectbox(
                "Sinal de Entrada",
                INPUT_SIGNALS,
                key="entrada_visual"
            )

            st.markdown("---")
            st.header("Estatísticas")
            model = st.session_state.visual_model
            st.metric("Blocos", len(model.get('nodes', [])))
            st.metric("Conexões", len(model.get('edges', [])))

        # Botões de ação
        col1, col2 = st.columns([1, 1])
        with col1:
            processar = st.button("Processar Diagrama", type="primary", use_container_width=True)
        with col2:
            if st.button("Exportar Diagrama", use_container_width=True):
                diagrama_json = json.dumps(st.session_state.visual_model, indent=2)
                st.download_button(
                    label="Baixar JSON",
                    data=diagrama_json,
                    file_name="diagrama_blocos.json",
                    mime="application/json"
                )

        # Processar e analisar
        if processar:
            try:
                model = st.session_state.visual_model
                G_open, G_closed, details = processar_diagrama_visual(model)

                if G_open is None:
                    error_msg = details if isinstance(details, str) else "Erro ao processar diagrama."
                    st.error(error_msg)
                else:
                    st.success("Diagrama processado com sucesso!")

                    # Mostrar TF equivalente
                    mostrar_tf_equivalente(G_open, G_closed, details)

                    st.markdown("---")

                    # Selecionar sistema para análise
                    ganho_tf = TransferFunction([K_visual], [1])

                    if tipo_malha_visual == "Malha Aberta":
                        sistema = ganho_tf * G_open
                    else:
                        if isinstance(details, dict) and details.get('tipo') == 'malha_fechada':
                            # Já é malha fechada, aplicar ganho
                            G_fwd = details.get('forward', G_open)
                            H_fb = details.get('feedback', TransferFunction([1], [1]))
                            try:
                                sistema = ctrl.feedback(ganho_tf * G_fwd, H_fb)
                            except Exception:
                                sistema = G_closed
                        else:
                            # Sistema sem feedback detectado - usar feedback unitário
                            try:
                                sistema = ctrl.feedback(ganho_tf * G_open, 1)
                            except Exception:
                                sistema = G_closed

                    # Executar análises selecionadas
                    executar_analises(sistema, analises_visual, entrada_visual)

            except Exception as e:
                st.error(f"Erro durante o processamento: {e}")

        return

    # =====================================================
    # MODO CLÁSSICO (código original)
    # =====================================================

    with st.sidebar:
        st.header("Adicionar Blocos")
        nome = st.text_input("Nome", value="G1")
        tipo = st.selectbox("Tipo", ['Planta', 'Controlador', 'Sensor', 'Outro'])
        numerador = st.text_input("Numerador", placeholder="ex: 4*s")
        denominador = st.text_input("Denominador", placeholder="ex: s^2 + 2*s + 3")

        if st.button("Adicionar"):
            sucesso, mensagem = adicionar_bloco(nome, tipo, numerador, denominador)
            if sucesso:
                st.success(mensagem)
            else:
                st.error(mensagem)

        if not st.session_state.blocos.empty:
            st.header("Excluir Blocos")
            excluir = st.selectbox("Selecionar", st.session_state.blocos['nome'])
            if st.button("Excluir"):
                mensagem = remover_bloco(excluir)
                st.success(mensagem)

        st.header("Configurações")
        if st.button("Habilitar Cálculo de Erro" if not st.session_state.calculo_erro_habilitado else "Desabilitar Cálculo de Erro"):
            st.session_state.calculo_erro_habilitado = not st.session_state.calculo_erro_habilitado
            st.rerun()

    if st.session_state.calculo_erro_habilitado:
        st.subheader("Cálculo de Erro Estacionário")
        col1, col2 = st.columns(2)
        with col1:
            num_erro = st.text_input("Numerador", value="", key="num_erro")
        with col2:
            den_erro = st.text_input("Denominador", value="", key="den_erro")

        btn_col1, btn_col2 = st.columns(2)
        with btn_col1:
            if st.button("Calcular Erro Estacionário"):
                try:
                    G, _ = converter_para_tf(num_erro, den_erro)
                    tipo, Kp, Kv, Ka = constantes_de_erro(G)

                    df = pd.DataFrame([{"Tipo": tipo, "Kp": Kp, "Kv": Kv, "Ka": Ka}])
                    st.subheader("Resultado")
                    st.dataframe(
                        df.style.format({
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
            if st.button("Remover Planta", key="remover_planta"):
                if not st.session_state.blocos.empty:
                    st.session_state.blocos = st.session_state.blocos[st.session_state.blocos['tipo'] != 'Planta']
                    st.success("Plantas removidas!")
                else:
                    st.warning("Nenhuma planta para remover")
    else:
        st.info("Use o botão 'Habilitar Cálculo de Erro' na barra lateral")

    col1, col2 = st.columns([2, 1])

    with col2:
        st.subheader("Tipo de Sistema")
        tipo_malha = st.selectbox("Tipo:", ["Malha Aberta", "Malha Fechada"])
        usar_ganho = st.checkbox("Adicionar ganho K ajustável", value=False)

        if usar_ganho:
            K = st.slider("Ganho K", 0.1, 100.0, 1.0, 0.1)
            st.info(f"Ganho K: {K:.2f}")
        else:
            K = 1.0

        st.subheader("Análises")
        analise_opcoes = ANALYSIS_OPTIONS["malha_fechada" if tipo_malha == "Malha Fechada" else "malha_aberta"]
        analises = st.multiselect("Escolha:", analise_opcoes, default=analise_opcoes[0])
        entrada = st.selectbox("Sinal de Entrada", INPUT_SIGNALS)

    with col1:
        st.subheader("Resultados")

        if st.button("Executar Simulação", use_container_width=True):
            try:
                df = st.session_state.blocos
                if df.empty:
                    st.warning("Adicione blocos primeiro.")
                    st.stop()

                planta = obter_bloco_por_tipo('Planta')
                controlador = obter_bloco_por_tipo('Controlador')
                sensor = obter_bloco_por_tipo('Sensor')

                if planta is None:
                    st.error("Adicione pelo menos uma Planta.")
                    st.stop()

                ganho_tf = TransferFunction([K], [1])

                if tipo_malha == "Malha Aberta":
                    sistema = ganho_tf * planta
                    st.info(f"Sistema em Malha Aberta com K = {K:.2f}")
                else:
                    planta_com_ganho = ganho_tf * planta
                    sistema = calcular_malha_fechada(planta_com_ganho, controlador, sensor)
                    st.info(f"Sistema em Malha Fechada com K = {K:.2f}")

                executar_analises(sistema, analises, entrada)

            except Exception as e:
                st.error(f"Erro durante a simulação: {e}")

    st.sidebar.markdown("---")
    st.sidebar.markdown("### Dica")
    st.sidebar.info("Experimente o **Editor Visual** para construir sistemas graficamente!")

if __name__ == "__main__":
    main()
