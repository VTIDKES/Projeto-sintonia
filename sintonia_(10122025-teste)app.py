# -*- coding: utf-8 -*-
"""
Sistema de Modelagem e Análise de Sistemas de Controle

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
    if 'visual_blocos' not in st.session_state:
        st.session_state.visual_blocos = []
    if 'visual_conexoes' not in st.session_state:
        st.session_state.visual_conexoes = []
    if 'visual_counter' not in st.session_state:
        st.session_state.visual_counter = 1
    if 'visual_sistema_json' not in st.session_state:
        st.session_state.visual_sistema_json = ""

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
# EDITOR VISUAL DE DIAGRAMA DE BLOCOS
# =====================================================

def criar_editor_visual_html():
    """Cria o editor visual de diagrama de blocos com drag-and-drop completo"""
    
    blocos_init = json.dumps(st.session_state.visual_blocos)
    conexoes_init = json.dumps(st.session_state.visual_conexoes)
    counter_init = st.session_state.visual_counter

    html_code = """
    <!DOCTYPE html>
    <html lang="pt-BR">
    <head>
    <meta charset="UTF-8">
    <style>
    * { margin:0; padding:0; box-sizing:border-box; }
    body {
        font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
        background: #0f1117;
        color: #e0e0e0;
        overflow: hidden;
    }

    /* ── TOOLBAR ── */
    .toolbar {
        background: #1a1d29;
        border-bottom: 1px solid #2d3044;
        padding: 8px 14px;
        display: flex;
        align-items: center;
        gap: 6px;
        flex-wrap: wrap;
        min-height: 48px;
    }
    .toolbar .sep {
        width: 1px; height: 28px; background: #2d3044; margin: 0 4px;
    }
    .tbtn {
        display: inline-flex; align-items: center; gap: 5px;
        background: #252839; color: #c8cad8;
        border: 1px solid #363a50; border-radius: 6px;
        padding: 6px 12px; font-size: 12px;
        cursor: pointer; transition: all .15s;
        white-space: nowrap;
    }
    .tbtn:hover { background: #2f3349; border-color: #5568d3; color: #fff; }
    .tbtn.primary { background: #4a5acf; border-color: #5568d3; color: #fff; }
    .tbtn.primary:hover { background: #5b6be0; }
    .tbtn.danger { background: #4a2030; border-color: #8b3050; color: #ff8fa3; }
    .tbtn.danger:hover { background: #5a2840; }
    .tbtn .ico { font-size: 14px; }

    .toolbar-label {
        font-size: 11px; color: #6b7094; margin-right: 2px;
        text-transform: uppercase; letter-spacing: .5px;
    }

    /* ── CANVAS ── */
    #canvas-wrap {
        position: relative;
        width: 100%;
        height: 540px;
        background:
            radial-gradient(circle at 50% 50%, #141722 0%, #0f1117 100%);
        overflow: hidden;
        cursor: default;
    }
    #canvas-wrap::before {
        content: '';
        position: absolute; inset: 0;
        background-image:
            radial-gradient(circle, #1e2235 1px, transparent 1px);
        background-size: 24px 24px;
        opacity: .5;
        pointer-events: none;
    }

    /* ── SVG CONNECTIONS ── */
    #svg-connections {
        position: absolute; inset: 0;
        width: 100%; height: 100%;
        pointer-events: none;
        z-index: 1;
    }
    #svg-connections path {
        fill: none;
        stroke-width: 2.5;
        pointer-events: visibleStroke;
        cursor: pointer;
    }
    #svg-connections path:hover {
        stroke-width: 4;
    }
    .conn-line { stroke: #5568d3; }
    .conn-feedback { stroke: #e8a035; stroke-dasharray: 8 4; }
    .conn-temp { stroke: #88dd55; stroke-dasharray: 4 4; opacity: .7; }

    /* ── BLOCKS ── */
    .block {
        position: absolute;
        z-index: 10;
        border-radius: 10px;
        cursor: grab;
        user-select: none;
        transition: box-shadow .15s;
        min-width: 130px;
    }
    .block:active { cursor: grabbing; }
    .block.selected {
        box-shadow: 0 0 0 2px #fbbf24, 0 0 24px rgba(251,191,36,.25) !important;
    }
    .block-inner {
        padding: 10px 14px;
        border-radius: 10px;
        position: relative;
    }

    /* Block type styles */
    .block-tf .block-inner {
        background: linear-gradient(135deg, #1e3a5f 0%, #1a2744 100%);
        border: 1px solid #2a5a8f;
        box-shadow: 0 4px 16px rgba(30,58,95,.4);
    }
    .block-gain .block-inner {
        background: linear-gradient(135deg, #2d1f4e 0%, #1f1635 100%);
        border: 1px solid #5a3d8f;
        box-shadow: 0 4px 16px rgba(90,61,143,.3);
    }
    .block-sum .block-inner {
        background: linear-gradient(135deg, #1f4a3a 0%, #152e28 100%);
        border: 1px solid #2d8a6a;
        box-shadow: 0 4px 16px rgba(45,138,106,.3);
        min-width: 60px;
        width: 60px; height: 60px;
        border-radius: 50%;
        display: flex; align-items: center; justify-content: center;
    }
    .block-int .block-inner {
        background: linear-gradient(135deg, #4a3a1f 0%, #352e15 100%);
        border: 1px solid #8a7a2d;
        box-shadow: 0 4px 16px rgba(138,122,45,.3);
    }
    .block-sensor .block-inner {
        background: linear-gradient(135deg, #4a1f2d 0%, #351520 100%);
        border: 1px solid #8a2d4a;
        box-shadow: 0 4px 16px rgba(138,45,74,.3);
    }
    .block-input .block-inner {
        background: linear-gradient(135deg, #1a3a1a 0%, #0f250f 100%);
        border: 1px solid #2d8a2d;
        box-shadow: 0 4px 16px rgba(45,138,45,.3);
        min-width: 80px;
    }
    .block-output .block-inner {
        background: linear-gradient(135deg, #3a1a1a 0%, #250f0f 100%);
        border: 1px solid #8a2d2d;
        box-shadow: 0 4px 16px rgba(138,45,45,.3);
        min-width: 80px;
    }

    .block-label {
        font-size: 10px;
        text-transform: uppercase;
        letter-spacing: .8px;
        opacity: .6;
        margin-bottom: 3px;
    }
    .block-name {
        font-weight: 700;
        font-size: 14px;
        color: #fff;
    }
    .block-tf-display {
        font-family: 'Courier New', monospace;
        font-size: 11px;
        margin-top: 5px;
        padding: 4px 8px;
        background: rgba(0,0,0,.3);
        border-radius: 4px;
        color: #a0b8d8;
        text-align: center;
    }
    .block-tf-display .tf-num {
        border-bottom: 1px solid #5580aa;
        padding-bottom: 2px;
        margin-bottom: 2px;
    }

    /* ── PORTS ── */
    .port {
        position: absolute;
        width: 14px; height: 14px;
        border-radius: 50%;
        cursor: crosshair;
        z-index: 20;
        transition: all .15s;
    }
    .port::after {
        content: '';
        position: absolute; inset: 3px;
        border-radius: 50%;
        background: currentColor;
    }
    .port-in {
        left: -7px; top: 50%; transform: translateY(-50%);
        border: 2px solid #44bb88;
        color: #44bb88;
        background: #0f1117;
    }
    .port-out {
        right: -7px; top: 50%; transform: translateY(-50%);
        border: 2px solid #5588dd;
        color: #5588dd;
        background: #0f1117;
    }
    .port:hover {
        transform: translateY(-50%) scale(1.5);
        box-shadow: 0 0 10px currentColor;
    }
    .port.connecting {
        transform: translateY(-50%) scale(1.4);
        box-shadow: 0 0 12px #fbbf24;
        border-color: #fbbf24;
        color: #fbbf24;
    }

    /* Sum block ports */
    .block-sum .port-in {
        left: -7px; top: 50%; transform: translateY(-50%);
    }
    .block-sum .port-in.port-in-2 {
        left: 50%; top: calc(100% + 1px);
        transform: translateX(-50%) translateY(-50%);
    }
    .block-sum .port-out {
        right: -7px; top: 50%; transform: translateY(-50%);
    }
    .block-sum .port-in.port-in-2:hover {
        transform: translateX(-50%) translateY(-50%) scale(1.5);
    }
    .block-sum .port-in.port-in-2.connecting {
        transform: translateX(-50%) translateY(-50%) scale(1.4);
    }

    /* ── SUM SIGN LABELS ── */
    .sum-sign {
        position: absolute;
        font-size: 14px; font-weight: 700;
        color: #60ddaa;
        pointer-events: none;
    }
    .sum-sign.sign-left { left: -18px; top: 50%; transform: translateY(-50%); }
    .sum-sign.sign-bottom { left: 50%; bottom: -20px; transform: translateX(-50%); }

    /* ── CONTEXT MENU ── */
    .ctx-menu {
        position: absolute; z-index: 1000;
        background: #1e2235;
        border: 1px solid #363a50;
        border-radius: 8px;
        padding: 4px;
        box-shadow: 0 8px 32px rgba(0,0,0,.6);
        min-width: 180px;
        display: none;
    }
    .ctx-item {
        padding: 7px 12px;
        font-size: 12px;
        cursor: pointer;
        border-radius: 5px;
        display: flex; align-items: center; gap: 8px;
        color: #c0c4d8;
        transition: background .1s;
    }
    .ctx-item:hover { background: #2a2f48; color: #fff; }
    .ctx-item.ctx-danger { color: #ff8fa3; }
    .ctx-item.ctx-danger:hover { background: #3a1a25; }
    .ctx-sep { height: 1px; background: #2d3044; margin: 3px 8px; }

    /* ── MODAL ── */
    .modal-overlay {
        position: fixed; inset: 0; z-index: 2000;
        background: rgba(0,0,0,.6);
        display: none; align-items: center; justify-content: center;
    }
    .modal-overlay.active { display: flex; }
    .modal-box {
        background: #1a1d29;
        border: 1px solid #363a50;
        border-radius: 12px;
        padding: 24px;
        min-width: 340px;
        box-shadow: 0 20px 60px rgba(0,0,0,.8);
    }
    .modal-title {
        font-size: 16px; font-weight: 700; color: #fff;
        margin-bottom: 16px;
    }
    .modal-field {
        margin-bottom: 12px;
    }
    .modal-field label {
        display: block; font-size: 11px;
        text-transform: uppercase; letter-spacing: .5px;
        color: #6b7094; margin-bottom: 4px;
    }
    .modal-field input, .modal-field select {
        width: 100%; padding: 8px 10px;
        background: #252839; border: 1px solid #363a50;
        border-radius: 6px; color: #e0e0e0;
        font-size: 13px; outline: none;
    }
    .modal-field input:focus, .modal-field select:focus {
        border-color: #5568d3;
    }
    .modal-actions {
        display: flex; gap: 8px; justify-content: flex-end;
        margin-top: 18px;
    }

    /* ── STATUS BAR ── */
    .status-bar {
        background: #1a1d29;
        border-top: 1px solid #2d3044;
        padding: 5px 14px;
        display: flex;
        align-items: center;
        justify-content: space-between;
        font-size: 11px;
        color: #6b7094;
    }
    .status-bar .tag {
        background: #252839;
        border: 1px solid #2d3044;
        border-radius: 4px;
        padding: 1px 8px;
        margin-left: 8px;
    }

    /* ── OUTPUT BOX ── */
    #output-box {
        display: none;
        position: absolute;
        bottom: 40px; left: 14px; right: 14px;
        background: #141722;
        border: 1px solid #2d3044;
        border-radius: 8px;
        padding: 12px 16px;
        font-family: 'Courier New', monospace;
        font-size: 12px;
        color: #88ddaa;
        max-height: 120px;
        overflow-y: auto;
        z-index: 100;
    }
    </style>
    </head>
    <body>

    <!-- TOOLBAR -->
    <div class="toolbar">
        <span class="toolbar-label">Inserir:</span>
        <button class="tbtn" onclick="openAddBlock('tf')">
            <span class="ico">𝑓</span> Função Transferência
        </button>
        <button class="tbtn" onclick="openAddBlock('gain')">
            <span class="ico">K</span> Ganho
        </button>
        <button class="tbtn" onclick="addSumBlock()">
            <span class="ico">Σ</span> Somador
        </button>
        <button class="tbtn" onclick="openAddBlock('int')">
            <span class="ico">∫</span> Integrador
        </button>
        <button class="tbtn" onclick="openAddBlock('sensor')">
            <span class="ico">H</span> Sensor
        </button>
        <div class="sep"></div>
        <button class="tbtn" onclick="addIOBlock('input')">
            <span class="ico">▶</span> Entrada R(s)
        </button>
        <button class="tbtn" onclick="addIOBlock('output')">
            <span class="ico">◼</span> Saída Y(s)
        </button>
        <div class="sep"></div>
        <button class="tbtn danger" onclick="deleteSelected()">
            <span class="ico">✕</span> Remover
        </button>
        <button class="tbtn danger" onclick="clearAll()">
            <span class="ico">↺</span> Limpar
        </button>
        <div class="sep"></div>
        <button class="tbtn primary" onclick="exportSystem()">
            <span class="ico">⚡</span> Exportar Sistema
        </button>
    </div>

    <!-- CANVAS -->
    <div id="canvas-wrap"
         oncontextmenu="return false;"
         onclick="onCanvasClick(event)">
        <svg id="svg-connections">
            <defs>
                <marker id="arrow" markerWidth="8" markerHeight="8"
                        refX="7" refY="3" orient="auto">
                    <polygon points="0 0, 8 3, 0 6" fill="#5568d3"/>
                </marker>
                <marker id="arrow-fb" markerWidth="8" markerHeight="8"
                        refX="7" refY="3" orient="auto">
                    <polygon points="0 0, 8 3, 0 6" fill="#e8a035"/>
                </marker>
                <marker id="arrow-temp" markerWidth="8" markerHeight="8"
                        refX="7" refY="3" orient="auto">
                    <polygon points="0 0, 8 3, 0 6" fill="#88dd55"/>
                </marker>
            </defs>
        </svg>
        <div id="output-box"></div>
    </div>

    <!-- CONTEXT MENU -->
    <div class="ctx-menu" id="ctx-menu">
        <div class="ctx-item" onclick="editBlock()">✏️ Editar</div>
        <div class="ctx-item" onclick="duplicateBlock()">📋 Duplicar</div>
        <div class="ctx-sep"></div>
        <div class="ctx-item" onclick="toggleFeedback()">🔄 Marcar conexão como feedback</div>
        <div class="ctx-sep"></div>
        <div class="ctx-item ctx-danger" onclick="deleteSelected()">🗑️ Remover</div>
    </div>

    <!-- MODAL -->
    <div class="modal-overlay" id="modal-overlay">
        <div class="modal-box">
            <div class="modal-title" id="modal-title">Adicionar Bloco</div>
            <div id="modal-body"></div>
            <div class="modal-actions">
                <button class="tbtn" onclick="closeModal()">Cancelar</button>
                <button class="tbtn primary" onclick="confirmModal()">Confirmar</button>
            </div>
        </div>
    </div>

    <!-- STATUS BAR -->
    <div class="status-bar">
        <div>
            Blocos: <span class="tag" id="st-blocks">0</span>
            Conexões: <span class="tag" id="st-conns">0</span>
        </div>
        <div id="st-hint">Clique direito nos blocos para opções</div>
    </div>

    <script>
    // ═══════════════════════════════════════
    //  STATE
    // ═══════════════════════════════════════
    let blocks = """ + blocos_init + """;
    let connections = """ + conexoes_init + """;
    let idCounter = """ + str(counter_init) + """;
    let selectedId = null;
    let dragging = null;
    let dragOff = {x:0, y:0};
    let connectingPort = null; // {blockId, portType, portIndex, el}
    let tempLine = null;
    let modalCallback = null;
    let modalType = null;

    // ═══════════════════════════════════════
    //  INIT
    // ═══════════════════════════════════════
    function init() {
        blocks.forEach(b => renderBlock(b));
        drawConnections();
        updateStatus();
    }

    // ═══════════════════════════════════════
    //  BLOCK RENDERING
    // ═══════════════════════════════════════
    function renderBlock(b) {
        const el = document.createElement('div');
        el.className = 'block block-' + b.type;
        el.id = 'block-' + b.id;
        el.style.left = b.x + 'px';
        el.style.top = b.y + 'px';

        let inner = '';
        if (b.type === 'sum') {
            inner = `<div class="block-inner">
                <div class="block-name" style="font-size:22px;">Σ</div>
                <div class="port port-in" data-block="${b.id}" data-port="in" data-idx="0"></div>
                <div class="port port-in port-in-2" data-block="${b.id}" data-port="in" data-idx="1"></div>
                <div class="port port-out" data-block="${b.id}" data-port="out" data-idx="0"></div>
                <div class="sum-sign sign-left">${b.sign1 || '+'}</div>
                <div class="sum-sign sign-bottom">${b.sign2 || '−'}</div>
            </div>`;
        } else if (b.type === 'input') {
            inner = `<div class="block-inner">
                <div class="block-label">ENTRADA</div>
                <div class="block-name">${b.name || 'R(s)'}</div>
                <div class="port port-out" data-block="${b.id}" data-port="out" data-idx="0"></div>
            </div>`;
        } else if (b.type === 'output') {
            inner = `<div class="block-inner">
                <div class="block-label">SAÍDA</div>
                <div class="block-name">${b.name || 'Y(s)'}</div>
                <div class="port port-in" data-block="${b.id}" data-port="in" data-idx="0"></div>
            </div>`;
        } else {
            const labels = {tf:'FUNÇÃO TRANSF.', gain:'GANHO', int:'INTEGRADOR', sensor:'SENSOR'};
            let tfHtml = '';
            if (b.num && b.den) {
                tfHtml = `<div class="block-tf-display">
                    <div class="tf-num">${b.num}</div>
                    <div class="tf-den">${b.den}</div>
                </div>`;
            } else if (b.value !== undefined) {
                tfHtml = `<div class="block-tf-display">${b.value}</div>`;
            }
            inner = `<div class="block-inner">
                <div class="block-label">${labels[b.type] || b.type}</div>
                <div class="block-name">${b.name}</div>
                ${tfHtml}
                <div class="port port-in" data-block="${b.id}" data-port="in" data-idx="0"></div>
                <div class="port port-out" data-block="${b.id}" data-port="out" data-idx="0"></div>
            </div>`;
        }

        el.innerHTML = inner;
        document.getElementById('canvas-wrap').appendChild(el);

        // Events
        el.addEventListener('mousedown', e => startDrag(e, b.id));
        el.addEventListener('contextmenu', e => showCtx(e, b.id));

        el.querySelectorAll('.port').forEach(p => {
            p.addEventListener('mousedown', e => {
                e.stopPropagation();
                startConnect(e, p);
            });
        });
    }

    // ═══════════════════════════════════════
    //  DRAG & DROP
    // ═══════════════════════════════════════
    function startDrag(e, id) {
        if (e.target.classList.contains('port')) return;
        if (e.button === 2) return;
        e.preventDefault();
        selectBlock(id);
        dragging = id;
        const el = document.getElementById('block-' + id);
        const rect = el.getBoundingClientRect();
        dragOff.x = e.clientX - rect.left;
        dragOff.y = e.clientY - rect.top;
    }

    document.addEventListener('mousemove', e => {
        if (dragging !== null) {
            const wrap = document.getElementById('canvas-wrap').getBoundingClientRect();
            const el = document.getElementById('block-' + dragging);
            let x = e.clientX - wrap.left - dragOff.x;
            let y = e.clientY - wrap.top - dragOff.y;
            x = Math.max(0, Math.min(x, wrap.width - el.offsetWidth));
            y = Math.max(0, Math.min(y, wrap.height - el.offsetHeight));
            el.style.left = x + 'px';
            el.style.top = y + 'px';
            const b = blocks.find(bl => bl.id === dragging);
            if (b) { b.x = x; b.y = y; }
            drawConnections();
        }
        if (connectingPort && tempLine) {
            const wrap = document.getElementById('canvas-wrap').getBoundingClientRect();
            const mx = e.clientX - wrap.left;
            const my = e.clientY - wrap.top;
            const start = getPortPos(connectingPort.blockId, connectingPort.portType, connectingPort.portIndex);
            const d = makePath(start.x, start.y, mx, my);
            tempLine.setAttribute('d', d);
        }
    });

    document.addEventListener('mouseup', e => {
        if (dragging !== null) {
            dragging = null;
            saveState();
        }
        if (connectingPort) {
            // Check if mouse is over a port
            const target = document.elementFromPoint(e.clientX, e.clientY);
            if (target && target.classList.contains('port')) {
                finishConnect(target);
            } else {
                cancelConnect();
            }
        }
    });

    // ═══════════════════════════════════════
    //  CONNECTIONS
    // ═══════════════════════════════════════
    function startConnect(e, portEl) {
        e.stopPropagation();
        e.preventDefault();
        const blockId = parseInt(portEl.dataset.block);
        const portType = portEl.dataset.port;
        const portIdx = parseInt(portEl.dataset.idx || 0);

        connectingPort = {blockId, portType, portIndex: portIdx, el: portEl};
        portEl.classList.add('connecting');

        // Create temp line
        const svg = document.getElementById('svg-connections');
        tempLine = document.createElementNS('http://www.w3.org/2000/svg', 'path');
        tempLine.classList.add('conn-temp');
        tempLine.setAttribute('marker-end', 'url(#arrow-temp)');
        svg.appendChild(tempLine);
    }

    function finishConnect(targetEl) {
        const tBlockId = parseInt(targetEl.dataset.block);
        const tPortType = targetEl.dataset.port;
        const tPortIdx = parseInt(targetEl.dataset.idx || 0);

        if (connectingPort.blockId === tBlockId) {
            cancelConnect();
            return;
        }

        let fromId, fromPort, fromIdx, toId, toPort, toIdx;

        if (connectingPort.portType === 'out' && tPortType === 'in') {
            fromId = connectingPort.blockId;
            fromPort = 'out';
            fromIdx = connectingPort.portIndex;
            toId = tBlockId;
            toPort = 'in';
            toIdx = tPortIdx;
        } else if (connectingPort.portType === 'in' && tPortType === 'out') {
            fromId = tBlockId;
            fromPort = 'out';
            fromIdx = tPortIdx;
            toId = connectingPort.blockId;
            toPort = 'in';
            toIdx = connectingPort.portIndex;
        } else {
            cancelConnect();
            return;
        }

        // Check duplicate
        const dup = connections.find(c =>
            c.from === fromId && c.fromIdx === fromIdx &&
            c.to === toId && c.toIdx === toIdx
        );
        if (!dup) {
            connections.push({
                from: fromId, fromPort, fromIdx,
                to: toId, toPort, toIdx,
                feedback: false
            });
        }

        cancelConnect();
        drawConnections();
        saveState();
    }

    function cancelConnect() {
        if (connectingPort) {
            connectingPort.el.classList.remove('connecting');
            connectingPort = null;
        }
        if (tempLine) {
            tempLine.remove();
            tempLine = null;
        }
    }

    function getPortPos(blockId, portType, portIdx) {
        const el = document.getElementById('block-' + blockId);
        const wrap = document.getElementById('canvas-wrap').getBoundingClientRect();
        if (!el) return {x:0, y:0};

        const selector = `.port[data-block="${blockId}"][data-port="${portType}"]` +
                          (portIdx > 0 ? `.port-in-2` : `:not(.port-in-2)`);
        let portEl = el.querySelector(selector);
        if (!portEl) {
            portEl = el.querySelector(`.port[data-block="${blockId}"][data-port="${portType}"]`);
        }
        if (!portEl) {
            const rect = el.getBoundingClientRect();
            return {
                x: rect.left + (portType === 'out' ? rect.width : 0) - wrap.left,
                y: rect.top + rect.height/2 - wrap.top
            };
        }

        const pr = portEl.getBoundingClientRect();
        return {
            x: pr.left + pr.width/2 - wrap.left,
            y: pr.top + pr.height/2 - wrap.top
        };
    }

    function makePath(x1, y1, x2, y2) {
        const dx = Math.abs(x2 - x1);
        const cp = Math.max(dx * 0.4, 40);
        return `M ${x1} ${y1} C ${x1+cp} ${y1}, ${x2-cp} ${y2}, ${x2} ${y2}`;
    }

    function drawConnections() {
        const svg = document.getElementById('svg-connections');
        // Keep defs, remove paths
        svg.querySelectorAll('path').forEach(p => p.remove());

        connections.forEach((c, ci) => {
            const from = getPortPos(c.from, 'out', c.fromIdx || 0);
            const to = getPortPos(c.to, 'in', c.toIdx || 0);

            const path = document.createElementNS('http://www.w3.org/2000/svg', 'path');

            if (c.feedback) {
                // Route feedback: go down then back
                const midY = Math.max(from.y, to.y) + 60;
                const d = `M ${from.x} ${from.y} L ${from.x+30} ${from.y}
                           Q ${from.x+30} ${midY} ${(from.x+to.x)/2} ${midY}
                           Q ${to.x-30} ${midY} ${to.x-30} ${to.y}
                           L ${to.x} ${to.y}`;
                path.setAttribute('d', d);
                path.classList.add('conn-feedback');
                path.setAttribute('marker-end', 'url(#arrow-fb)');
            } else {
                path.setAttribute('d', makePath(from.x, from.y, to.x, to.y));
                path.classList.add('conn-line');
                path.setAttribute('marker-end', 'url(#arrow)');
            }

            path.style.pointerEvents = 'visibleStroke';
            path.addEventListener('contextmenu', e => {
                e.preventDefault();
                e.stopPropagation();
                selectedConnIdx = ci;
                showConnCtx(e, ci);
            });

            svg.appendChild(path);
        });
        updateStatus();
    }

    let selectedConnIdx = null;

    function showConnCtx(e, ci) {
        selectedConnIdx = ci;
        const menu = document.getElementById('ctx-menu');
        menu.innerHTML = `
            <div class="ctx-item" onclick="toggleConnFeedback(${ci})">🔄 ${connections[ci].feedback ? 'Conexão normal' : 'Marcar como feedback'}</div>
            <div class="ctx-sep"></div>
            <div class="ctx-item ctx-danger" onclick="deleteConn(${ci})">🗑️ Remover conexão</div>
        `;
        menu.style.display = 'block';
        menu.style.left = e.clientX + 'px';
        menu.style.top = e.clientY + 'px';
    }

    function toggleConnFeedback(ci) {
        connections[ci].feedback = !connections[ci].feedback;
        drawConnections();
        hideCtx();
        saveState();
    }

    function deleteConn(ci) {
        connections.splice(ci, 1);
        drawConnections();
        hideCtx();
        saveState();
    }

    // ═══════════════════════════════════════
    //  SELECTION
    // ═══════════════════════════════════════
    function selectBlock(id) {
        document.querySelectorAll('.block').forEach(b => b.classList.remove('selected'));
        selectedId = id;
        const el = document.getElementById('block-' + id);
        if (el) el.classList.add('selected');
    }

    function onCanvasClick(e) {
        if (e.target.id === 'canvas-wrap' || e.target.id === 'svg-connections') {
            document.querySelectorAll('.block').forEach(b => b.classList.remove('selected'));
            selectedId = null;
        }
        hideCtx();
    }

    // ═══════════════════════════════════════
    //  CONTEXT MENU
    // ═══════════════════════════════════════
    function showCtx(e, id) {
        e.preventDefault();
        e.stopPropagation();
        selectBlock(id);
        const menu = document.getElementById('ctx-menu');
        const b = blocks.find(bl => bl.id === id);
        let items = '';
        if (b && b.type !== 'input' && b.type !== 'output') {
            items += `<div class="ctx-item" onclick="editBlock()">✏️ Editar</div>`;
        }
        items += `<div class="ctx-item" onclick="duplicateBlock()">📋 Duplicar</div>`;
        items += `<div class="ctx-sep"></div>`;
        items += `<div class="ctx-item ctx-danger" onclick="deleteSelected()">🗑️ Remover</div>`;
        menu.innerHTML = items;
        menu.style.display = 'block';
        menu.style.left = e.clientX + 'px';
        menu.style.top = e.clientY + 'px';
    }

    function hideCtx() {
        document.getElementById('ctx-menu').style.display = 'none';
    }
    document.addEventListener('click', e => {
        if (!e.target.closest('.ctx-menu')) hideCtx();
    });

    // ═══════════════════════════════════════
    //  BLOCK OPERATIONS
    // ═══════════════════════════════════════
    function openAddBlock(type) {
        modalType = type;
        const title = {tf:'Função de Transferência', gain:'Ganho', int:'Integrador', sensor:'Sensor'}[type];
        document.getElementById('modal-title').textContent = 'Adicionar ' + title;

        let body = '';
        body += `<div class="modal-field"><label>Nome</label>
                  <input id="m-name" value="${type === 'int' ? '1/s' : (type === 'sensor' ? 'H(s)' : (type === 'gain' ? 'K' : 'G' + idCounter))}"></div>`;

        if (type === 'tf' || type === 'sensor') {
            body += `<div class="modal-field"><label>Numerador (ex: s+1, 4*s)</label>
                      <input id="m-num" value="1" placeholder="1"></div>`;
            body += `<div class="modal-field"><label>Denominador (ex: s^2+2*s+1)</label>
                      <input id="m-den" value="s+1" placeholder="s+1"></div>`;
        } else if (type === 'gain') {
            body += `<div class="modal-field"><label>Valor do ganho K</label>
                      <input id="m-val" value="1" type="number" step="any"></div>`;
        }

        document.getElementById('modal-body').innerHTML = body;
        document.getElementById('modal-overlay').classList.add('active');

        modalCallback = () => {
            const name = document.getElementById('m-name').value || type;
            const wrap = document.getElementById('canvas-wrap').getBoundingClientRect();

            const b = {
                id: idCounter++,
                type: type,
                name: name,
                x: 80 + Math.random() * (wrap.width - 250),
                y: 60 + Math.random() * (wrap.height - 180)
            };

            if (type === 'tf' || type === 'sensor') {
                b.num = document.getElementById('m-num').value || '1';
                b.den = document.getElementById('m-den').value || '1';
            } else if (type === 'gain') {
                b.value = document.getElementById('m-val').value || '1';
                b.num = b.value;
                b.den = '1';
            } else if (type === 'int') {
                b.num = '1';
                b.den = 's';
            }

            blocks.push(b);
            renderBlock(b);
            saveState();
        };
    }

    function addSumBlock() {
        const wrap = document.getElementById('canvas-wrap').getBoundingClientRect();
        const b = {
            id: idCounter++,
            type: 'sum',
            name: 'Σ',
            x: 80 + Math.random() * (wrap.width - 200),
            y: 60 + Math.random() * (wrap.height - 180),
            sign1: '+',
            sign2: '−'
        };
        blocks.push(b);
        renderBlock(b);
        saveState();
    }

    function addIOBlock(ioType) {
        const wrap = document.getElementById('canvas-wrap').getBoundingClientRect();
        const b = {
            id: idCounter++,
            type: ioType,
            name: ioType === 'input' ? 'R(s)' : 'Y(s)',
            x: ioType === 'input' ? 30 : wrap.width - 140,
            y: wrap.height / 2 - 30
        };
        blocks.push(b);
        renderBlock(b);
        saveState();
    }

    function editBlock() {
        hideCtx();
        if (selectedId === null) return;
        const b = blocks.find(bl => bl.id === selectedId);
        if (!b || b.type === 'input' || b.type === 'output') return;

        modalType = 'edit';
        document.getElementById('modal-title').textContent = 'Editar ' + b.name;

        let body = `<div class="modal-field"><label>Nome</label>
                     <input id="m-name" value="${b.name}"></div>`;

        if (b.type === 'tf' || b.type === 'sensor') {
            body += `<div class="modal-field"><label>Numerador</label>
                      <input id="m-num" value="${b.num || '1'}"></div>`;
            body += `<div class="modal-field"><label>Denominador</label>
                      <input id="m-den" value="${b.den || '1'}"></div>`;
        } else if (b.type === 'gain') {
            body += `<div class="modal-field"><label>Valor K</label>
                      <input id="m-val" value="${b.value || '1'}" type="number" step="any"></div>`;
        } else if (b.type === 'sum') {
            body += `<div class="modal-field"><label>Sinal entrada principal</label>
                      <select id="m-s1"><option ${b.sign1==='+' ? 'selected' : ''}>+</option>
                      <option ${b.sign1==='−' ? 'selected' : ''}>−</option></select></div>`;
            body += `<div class="modal-field"><label>Sinal entrada feedback</label>
                      <select id="m-s2"><option ${b.sign2==='+' ? 'selected' : ''}>+</option>
                      <option ${b.sign2==='−' ? 'selected' : ''}>−</option></select></div>`;
        }

        document.getElementById('modal-body').innerHTML = body;
        document.getElementById('modal-overlay').classList.add('active');

        modalCallback = () => {
            b.name = document.getElementById('m-name').value || b.name;

            if (b.type === 'tf' || b.type === 'sensor') {
                b.num = document.getElementById('m-num').value || '1';
                b.den = document.getElementById('m-den').value || '1';
            } else if (b.type === 'gain') {
                b.value = document.getElementById('m-val').value || '1';
                b.num = b.value;
                b.den = '1';
            } else if (b.type === 'sum') {
                b.sign1 = document.getElementById('m-s1').value;
                b.sign2 = document.getElementById('m-s2').value;
            }

            // Re-render
            const el = document.getElementById('block-' + b.id);
            if (el) el.remove();
            renderBlock(b);
            drawConnections();
            saveState();
        };
    }

    function duplicateBlock() {
        hideCtx();
        if (selectedId === null) return;
        const orig = blocks.find(bl => bl.id === selectedId);
        if (!orig) return;
        const b = {...orig, id: idCounter++, x: orig.x + 30, y: orig.y + 30};
        blocks.push(b);
        renderBlock(b);
        saveState();
    }

    function deleteSelected() {
        hideCtx();
        if (selectedId === null) return;
        const el = document.getElementById('block-' + selectedId);
        if (el) el.remove();
        blocks = blocks.filter(b => b.id !== selectedId);
        connections = connections.filter(c => c.from !== selectedId && c.to !== selectedId);
        selectedId = null;
        drawConnections();
        saveState();
    }

    function clearAll() {
        if (!confirm('Limpar todo o diagrama?')) return;
        blocks.forEach(b => {
            const el = document.getElementById('block-' + b.id);
            if (el) el.remove();
        });
        blocks = [];
        connections = [];
        selectedId = null;
        drawConnections();
        saveState();
    }

    // ═══════════════════════════════════════
    //  MODAL
    // ═══════════════════════════════════════
    function closeModal() {
        document.getElementById('modal-overlay').classList.remove('active');
        modalCallback = null;
    }

    function confirmModal() {
        if (modalCallback) modalCallback();
        closeModal();
    }

    // Enter key in modal
    document.getElementById('modal-overlay').addEventListener('keydown', e => {
        if (e.key === 'Enter') confirmModal();
        if (e.key === 'Escape') closeModal();
    });

    // ═══════════════════════════════════════
    //  EXPORT SYSTEM
    // ═══════════════════════════════════════
    function exportSystem() {
        const data = {blocks, connections};
        const json = JSON.stringify(data);

        // Send to Streamlit
        const outputBox = document.getElementById('output-box');
        outputBox.style.display = 'block';
        outputBox.textContent = '✅ Sistema exportado! Clique em "Processar" no Streamlit.';

        // Use Streamlit component communication
        if (window.parent) {
            window.parent.postMessage({
                type: 'streamlit:setComponentValue',
                value: json
            }, '*');
        }

        // Also store in a hidden element for Streamlit to read
        let hiddenEl = document.getElementById('export-data');
        if (!hiddenEl) {
            hiddenEl = document.createElement('div');
            hiddenEl.id = 'export-data';
            hiddenEl.style.display = 'none';
            document.body.appendChild(hiddenEl);
        }
        hiddenEl.textContent = json;

        setTimeout(() => { outputBox.style.display = 'none'; }, 3000);
    }

    // ═══════════════════════════════════════
    //  STATUS
    // ═══════════════════════════════════════
    function updateStatus() {
        document.getElementById('st-blocks').textContent = blocks.length;
        document.getElementById('st-conns').textContent = connections.length;
    }

    function saveState() {
        updateStatus();
        // Send state to parent
        const data = {blocks, connections, counter: idCounter};
        if (window.parent) {
            window.parent.postMessage({
                type: 'diagram_state',
                data: JSON.stringify(data)
            }, '*');
        }
    }

    // ═══════════════════════════════════════
    //  KEYBOARD
    // ═══════════════════════════════════════
    document.addEventListener('keydown', e => {
        if (e.key === 'Delete' || e.key === 'Backspace') {
            if (selectedId !== null && !document.getElementById('modal-overlay').classList.contains('active')) {
                deleteSelected();
            }
        }
        if (e.key === 'Escape') {
            cancelConnect();
            closeModal();
        }
    });

    // Init
    init();
    </script>
    </body>
    </html>
    """
    return html_code


def processar_diagrama_visual():
    """Processa os blocos do editor visual e calcula o sistema equivalente"""
    blocos = st.session_state.visual_blocos
    conexoes = st.session_state.visual_conexoes

    if not blocos:
        return None, None, "Nenhum bloco no diagrama. Adicione blocos no editor visual."

    try:
        # Build transfer functions for each block
        tfs = {}
        for b in blocos:
            bid = b['id']
            btype = b.get('type', '')

            if btype in ('tf', 'sensor'):
                num_str = b.get('num', '1')
                den_str = b.get('den', '1')
                tf, _ = converter_para_tf(num_str, den_str)
                tfs[bid] = tf
            elif btype == 'gain':
                val = float(b.get('value', 1))
                tfs[bid] = TransferFunction([val], [1])
            elif btype == 'int':
                tfs[bid] = TransferFunction([1], [1, 0])
            elif btype == 'sum':
                tfs[bid] = 'sum'
            elif btype in ('input', 'output'):
                tfs[bid] = TransferFunction([1], [1])

        # Strategy: Try to detect standard feedback topology
        # Look for: Input -> Sum -> [Forward path blocks] -> Output
        #                     ^                              |
        #                     |--- [Feedback path] <---------|

        # Find forward path (non-feedback connections)
        fwd_conns = [c for c in conexoes if not c.get('feedback', False)]
        fb_conns = [c for c in conexoes if c.get('feedback', False)]

        # Build adjacency from forward connections
        adj = {}
        for c in fwd_conns:
            adj.setdefault(c['from'], []).append(c['to'])

        # Try to find a linear path through forward connections
        def find_path(start_candidates):
            for start in start_candidates:
                visited = set()
                path = [start]
                visited.add(start)
                current = start
                while current in adj:
                    nexts = [n for n in adj[current] if n not in visited]
                    if not nexts:
                        break
                    current = nexts[0]
                    path.append(current)
                    visited.add(current)
                if len(path) > 1:
                    return path
            return []

        # Find input blocks
        input_blocks = [b['id'] for b in blocos if b.get('type') == 'input']
        sum_blocks = [b['id'] for b in blocos if b.get('type') == 'sum']

        path = find_path(input_blocks if input_blocks else sum_blocks if sum_blocks else [blocos[0]['id']])

        if not path:
            # Fallback: multiply all non-sum, non-IO blocks in series
            series_tfs = [tfs[b['id']] for b in blocos
                         if b.get('type') not in ('sum', 'input', 'output') and b['id'] in tfs]
            if not series_tfs:
                return None, None, "Não foi possível calcular o sistema."

            G_open = series_tfs[0]
            for t in series_tfs[1:]:
                G_open = G_open * t
            return G_open, ctrl.feedback(G_open, TransferFunction([1], [1])), \
                   f"Sistema série: {len(series_tfs)} blocos"

        # Collect forward-path TFs (skip sum, input, output)
        fwd_tfs = []
        for bid in path:
            if bid in tfs and tfs[bid] != 'sum' and \
               not any(b['id'] == bid and b.get('type') in ('input', 'output') for b in blocos):
                fwd_tfs.append(tfs[bid])

        if not fwd_tfs:
            return None, None, "Nenhuma função de transferência encontrada no caminho direto."

        # Compute open-loop forward path
        G_forward = fwd_tfs[0]
        for t in fwd_tfs[1:]:
            G_forward = G_forward * t

        # Compute feedback path
        if fb_conns:
            fb_path_ids = set()
            for c in fb_conns:
                fb_path_ids.add(c['from'])
                fb_path_ids.add(c['to'])

            fb_tfs = []
            for bid in fb_path_ids:
                if bid in tfs and tfs[bid] != 'sum' and \
                   not any(b['id'] == bid and b.get('type') in ('input', 'output', 'sum') for b in blocos):
                    fb_tfs.append(tfs[bid])

            if fb_tfs:
                H = fb_tfs[0]
                for t in fb_tfs[1:]:
                    H = H * t
            else:
                H = TransferFunction([1], [1])

            G_closed = ctrl.feedback(G_forward, H)
            info = f"Malha aberta: G(s) = {G_forward}\nFeedback: H(s) = {H}\nMalha fechada: T(s) = {G_closed}"
        else:
            H = TransferFunction([1], [1])
            G_closed = ctrl.feedback(G_forward, H)
            info = f"G(s) = {G_forward} (feedback unitário)"

        return G_forward, G_closed, info

    except Exception as e:
        return None, None, f"Erro ao processar: {str(e)}"


# =====================================================
# COMPONENTE VISUAL (components.html)
# =====================================================

_VISUAL_HTML_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "visual_blocks_frontend", "index.html"
)

def _load_visual_editor_html():
    """Carrega o HTML do editor visual (com cache-bust)"""
    import time
    with open(_VISUAL_HTML_PATH, 'r', encoding='utf-8') as f:
        html = f.read()
    # Adiciona timestamp para forçar reload no navegador
    html = html.replace('</head>', f'<!-- v{int(time.time())} -->\n</head>')
    return html


# =====================================================
# SOLVER DE DIAGRAMA DE BLOCOS (SIGNAL FLOW)
# =====================================================

def _get_block_tf_sympy(node, s):
    """Retorna expressão sympy da TF de um bloco"""
    p = node.get('params', {})
    t = node.get('type', '')

    if t in ('tf', 'sensor'):
        num_str = p.get('num', '1')
        den_str = p.get('den', '1')
        num_expr = parse_expr(num_str.replace('^', '**'), local_dict={'s': s})
        den_expr = parse_expr(den_str.replace('^', '**'), local_dict={'s': s})
        return num_expr / den_expr
    elif t == 'gain':
        return sp.sympify(p.get('k', '1'))
    elif t == 'int':
        return 1 / s
    elif t == 'der':
        return s
    elif t == 'pid':
        kp = sp.sympify(p.get('kp', '1'))
        ki = sp.sympify(p.get('ki', '0'))
        kd = sp.sympify(p.get('kd', '0'))
        return kp + ki / s + kd * s
    else:
        return sp.Integer(1)


def resolver_diagrama_para_tf(nodes, edges):
    """
    Resolve o diagrama de blocos para obter a função de transferência equivalente.

    Usa análise de fluxo de sinais:
    - Cada nó tem uma variável de sinal de saída
    - Monta sistema de equações lineares
    - Resolve para Y(s)/R(s)
    """
    if not nodes or not edges:
        return None, "Diagrama vazio ou sem conexões."

    s = sp.Symbol('s')
    R_sym = sp.Symbol('R')

    # Encontrar entrada e saída
    input_nodes = [n for n in nodes if n.get('type') == 'input']
    output_nodes = [n for n in nodes if n.get('type') == 'output']

    if not input_nodes:
        return None, "Adicione um bloco de Entrada R(s)."
    if not output_nodes:
        return None, "Adicione um bloco de Saída Y(s)."

    input_id = input_nodes[0]['id']
    output_id = output_nodes[0]['id']

    # Variáveis simbólicas para sinal de saída de cada nó
    sig = {}
    for n in nodes:
        sig[n['id']] = sp.Symbol(f"x_{n['id'].replace('-','_')}")

    # Montar equações
    eqs = []

    for n in nodes:
        nid = n['id']
        ntype = n.get('type', '')
        params = n.get('params', {})

        # Arestas chegando neste nó
        incoming = [e for e in edges if e.get('dst') == nid]

        if ntype == 'input':
            eqs.append(sp.Eq(sig[nid], R_sym))
            continue

        if not incoming:
            eqs.append(sp.Eq(sig[nid], 0))
            continue

        if ntype == 'output':
            eqs.append(sp.Eq(sig[nid], sig[incoming[0]['src']]))

        elif ntype == 'sum':
            signs_str = params.get('signs', '+ -').strip().split()
            expr = sp.Integer(0)
            for edge in incoming:
                dst_port = edge.get('dstPort', 'in0')
                port_idx = 0
                try:
                    port_idx = int(dst_port.replace('in', ''))
                except (ValueError, AttributeError):
                    pass
                sign = signs_str[port_idx] if port_idx < len(signs_str) else '+'
                src_signal = sig[edge['src']]
                if sign == '-':
                    expr -= src_signal
                else:
                    expr += src_signal
            eqs.append(sp.Eq(sig[nid], expr))

        elif ntype == 'branch':
            eqs.append(sp.Eq(sig[nid], sig[incoming[0]['src']]))

        else:
            # tf, gain, int, der, pid, sensor
            tf_expr = _get_block_tf_sympy(n, s)
            eqs.append(sp.Eq(sig[nid], tf_expr * sig[incoming[0]['src']]))

    # Resolver sistema (incluir TODAS variáveis, inclusive entrada)
    unknowns = list(sig.values())

    try:
        sol = sp.solve(eqs, unknowns, dict=True)
        if not sol:
            return None, "Não foi possível resolver o sistema de equações."

        output_expr = sol[0][sig[output_id]]
        tf_ratio = sp.simplify(output_expr / R_sym)

        # Converter para coeficientes polinomiais
        tf_together = sp.together(tf_ratio)
        num_s, den_s = sp.fraction(tf_together)

        num_poly = sp.Poly(sp.expand(num_s), s)
        den_poly = sp.Poly(sp.expand(den_s), s)

        num_coeffs = [float(c) for c in num_poly.all_coeffs()]
        den_coeffs = [float(c) for c in den_poly.all_coeffs()]

        # Normalizar
        if den_coeffs and den_coeffs[0] != 0:
            factor = den_coeffs[0]
            num_coeffs = [c / factor for c in num_coeffs]
            den_coeffs = [c / factor for c in den_coeffs]

        tf_obj = TransferFunction(num_coeffs, den_coeffs)
        return tf_obj, None

    except Exception as e:
        return None, f"Erro ao resolver o diagrama: {str(e)}"


# =====================================================
# APLICAÇÃO PRINCIPAL
# =====================================================

def main():
    st.set_page_config(page_title="Modelagem de Sistemas", layout="wide")
    st.title("📉 Modelagem e Análise de Sistemas de Controle")

    inicializar_blocos()

    if 'calculo_erro_habilitado' not in st.session_state:
        st.session_state.calculo_erro_habilitado = False
    if 'modo_editor' not in st.session_state:
        st.session_state.modo_editor = 'classico'

    # ── Sidebar: Modo de trabalho ──
    st.sidebar.header("🎛️ Modo de Trabalho")
    modo = st.sidebar.radio(
        "Escolha o modo:",
        ['Clássico (Lista)', 'Editor Visual / Diagrama de Blocos'],
        index=0 if st.session_state.modo_editor == 'classico' else 1
    )
    st.session_state.modo_editor = 'visual' if 'Visual' in modo else 'classico'

    # ══════════════════════════════════════════
    #  MODO EDITOR VISUAL (Diagrama de Blocos)
    # ══════════════════════════════════════════
    if st.session_state.modo_editor == 'visual':

        # Editor visual completo com cálculos integrados em JavaScript
        html_content = _load_visual_editor_html()
        components.html(html_content, height=1200, scrolling=True)

        # ── Sidebar info ──
        with st.sidebar:
            st.markdown("### Como usar")
            st.markdown("""
            1. **Adicione** blocos pela barra superior
            2. **Arraste** para posicionar
            3. **Conecte**: porta azul (saída) → porta verde (entrada)
            4. **Edite** parâmetros no painel lateral
            5. Clique **CALCULAR** para ver resultados
            """)

        return

    # ══════════════════════════════════════════
    #  MODO CLÁSSICO (original preservado)
    # ══════════════════════════════════════════

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
                    tipo, Kp, Kv, Ka = constantes_de_erro(G)

                    df = pd.DataFrame([{"Tipo": tipo, "Kp": Kp, "Kv": Kv, "Ka": Ka}])
                    st.subheader("📊 Resultado")
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
            if st.button("🗑️ Remover Planta", key="remover_planta"):
                if not st.session_state.blocos.empty:
                    st.session_state.blocos = st.session_state.blocos[st.session_state.blocos['tipo'] != 'Planta']
                    st.success("Plantas removidas!")
                else:
                    st.warning("Nenhuma planta para remover")
    else:
        st.info("💡 Use o botão 'Habilitar Cálculo de Erro' na barra lateral")

    col1, col2 = st.columns([2, 1])

    with col2:
        st.subheader("🔍 Tipo de Sistema")
        tipo_malha = st.selectbox("Tipo:", ["Malha Aberta", "Malha Fechada"])
        usar_ganho = st.checkbox("Adicionar ganho K ajustável", value=False)

        if usar_ganho:
            K = st.slider("Ganho K", 0.1, 100.0, 1.0, 0.1)
            st.info(f"✅ Ganho K: {K:.2f}")
        else:
            K = 1.0

        st.subheader("📊 Análises")
        analise_opcoes = ANALYSIS_OPTIONS["malha_fechada" if tipo_malha == "Malha Fechada" else "malha_aberta"]
        analises = st.multiselect("Escolha:", analise_opcoes, default=analise_opcoes[0])
        entrada = st.selectbox("Sinal de Entrada", INPUT_SIGNALS)

    with col1:
        st.subheader("📈 Resultados")

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
                    st.error("Adicione pelo menos uma Planta.")
                    st.stop()

                ganho_tf = TransferFunction([K], [1])

                if tipo_malha == "Malha Aberta":
                    sistema = ganho_tf * planta
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
                        st.markdown(f"**Polos SPD (P):** {polos_spd}")
                        st.markdown(f"**Voltas (N):** {voltas}")
                        st.markdown(f"**Z = {Z} → {'✅ Estável' if Z == 0 else '❌ Instável'}**")
                        st.plotly_chart(fig, use_container_width=True)

            except Exception as e:
                st.error(f"Erro durante a simulação: {e}")

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 💡 Dica")
    st.sidebar.info("Experimente o **Editor Visual** para construir sistemas graficamente!")


if __name__ == "__main__":
    main()


