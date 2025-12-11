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
    if 'diagrama_blocos' not in st.session_state:
        st.session_state.diagrama_blocos = {
            'blocos': [],
            'conexoes': []
        }
    if 'bloco_contador' not in st.session_state:
        st.session_state.bloco_contador = 1

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
# EDITOR DE DIAGRAMA DE BLOCOS
# =====================================================

def criar_diagrama_blocos_html():
    """Cria o editor visual de diagrama de blocos"""
    blocos_data = json.dumps(st.session_state.diagrama_blocos['blocos'])
    conexoes_data = json.dumps(st.session_state.diagrama_blocos['conexoes'])
    
    html_code = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <style>
            body {{
                margin: 0;
                padding: 0;
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                overflow: hidden;
            }}
            #canvas-container {{
                width: 100%;
                height: 600px;
                background: linear-gradient(#f0f0f0 1px, transparent 1px),
                            linear-gradient(90deg, #f0f0f0 1px, transparent 1px);
                background-size: 20px 20px;
                position: relative;
                border: 2px solid #ddd;
                cursor: crosshair;
            }}
            .bloco {{
                position: absolute;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 15px;
                border-radius: 8px;
                cursor: move;
                box-shadow: 0 4px 6px rgba(0,0,0,0.2);
                min-width: 120px;
                text-align: center;
                user-select: none;
                transition: transform 0.1s, box-shadow 0.1s;
            }}
            .bloco:hover {{
                transform: scale(1.05);
                box-shadow: 0 6px 12px rgba(0,0,0,0.3);
            }}
            .bloco.selecionado {{
                border: 3px solid #ffd700;
                box-shadow: 0 0 20px rgba(255,215,0,0.5);
            }}
            .bloco-tipo {{
                font-size: 10px;
                opacity: 0.8;
                margin-bottom: 5px;
            }}
            .bloco-nome {{
                font-weight: bold;
                font-size: 14px;
                margin-bottom: 5px;
            }}
            .bloco-tf {{
                font-size: 11px;
                font-family: 'Courier New', monospace;
                background: rgba(0,0,0,0.2);
                padding: 5px;
                border-radius: 4px;
                margin-top: 5px;
            }}
            .porta {{
                width: 12px;
                height: 12px;
                background: #4CAF50;
                border: 2px solid white;
                border-radius: 50%;
                position: absolute;
                cursor: pointer;
                transition: all 0.2s;
            }}
            .porta:hover {{
                background: #45a049;
                transform: scale(1.3);
            }}
            .porta-entrada {{
                left: -6px;
                top: 50%;
                transform: translateY(-50%);
            }}
            .porta-saida {{
                right: -6px;
                top: 50%;
                transform: translateY(-50%);
            }}
            .toolbar {{
                background: #333;
                color: white;
                padding: 10px;
                display: flex;
                gap: 10px;
                flex-wrap: wrap;
            }}
            .btn {{
                background: #667eea;
                color: white;
                border: none;
                padding: 8px 15px;
                border-radius: 5px;
                cursor: pointer;
                font-size: 13px;
                transition: background 0.3s;
            }}
            .btn:hover {{
                background: #5568d3;
            }}
            .btn-danger {{
                background: #e74c3c;
            }}
            .btn-danger:hover {{
                background: #c0392b;
            }}
            svg {{
                position: absolute;
                top: 0;
                left: 0;
                width: 100%;
                height: 100%;
                pointer-events: none;
            }}
            #info-panel {{
                position: absolute;
                top: 10px;
                right: 10px;
                background: rgba(255,255,255,0.95);
                padding: 15px;
                border-radius: 8px;
                box-shadow: 0 4px 6px rgba(0,0,0,0.1);
                max-width: 250px;
                font-size: 12px;
            }}
        </style>
    </head>
    <body>
        <div class="toolbar">
            <button class="btn" onclick="adicionarBlocoTransferencia()">➕ Função Transferência</button>
            <button class="btn" onclick="adicionarBlocoSomador()">⊕ Somador</button>
            <button class="btn" onclick="adicionarBlocoGanho()">📊 Ganho</button>
            <button class="btn" onclick="adicionarBlocoIntegrador()">∫ Integrador</button>
            <button class="btn btn-danger" onclick="removerSelecionado()">🗑️ Remover</button>
            <button class="btn btn-danger" onclick="limparDiagrama()">🔄 Limpar</button>
        </div>
        
        <div id="canvas-container">
            <svg id="conexoes-svg"></svg>
            <div id="info-panel">
                <strong>📊 Instruções:</strong>
                <div>• Clique nos botões para adicionar</div>
                <div>• Arraste blocos para mover</div>
                <div>• Clique nas portas verdes para conectar</div>
                <div>• Clique no bloco para selecionar</div>
            </div>
        </div>

        <script>
            let blocos = {blocos_data};
            let conexoes = {conexoes_data};
            let blocoIdCounter = {st.session_state.bloco_contador};
            let blocoSelecionado = null;
            let portaSelecionada = null;
            let arrastandoBloco = null;
            let offsetX = 0, offsetY = 0;

            function adicionarBloco(tipo, config) {{
                const container = document.getElementById('canvas-container');
                const bloco = document.createElement('div');
                bloco.className = 'bloco';
                bloco.id = 'bloco-' + blocoIdCounter;
                
                const blocoData = {{
                    id: blocoIdCounter,
                    tipo: tipo,
                    x: 100 + Math.random() * 200,
                    y: 100 + Math.random() * 200,
                    config: config
                }};
                
                blocos.push(blocoData);
                
                bloco.style.left = blocoData.x + 'px';
                bloco.style.top = blocoData.y + 'px';
                
                let nomeDisplay = config.nome || tipo;
                let tfDisplay = config.tf || '';
                
                bloco.innerHTML = `
                    <div class="bloco-tipo">${{tipo}}</div>
                    <div class="bloco-nome">${{nomeDisplay}}</div>
                    ${{tfDisplay ? '<div class="bloco-tf">' + tfDisplay + '</div>' : ''}}
                    <div class="porta porta-entrada" data-bloco="${{blocoIdCounter}}" data-tipo="entrada"></div>
                    <div class="porta porta-saida" data-bloco="${{blocoIdCounter}}" data-tipo="saida"></div>
                `;
                
                container.appendChild(bloco);
                
                bloco.addEventListener('mousedown', iniciarArrastar);
                bloco.addEventListener('click', selecionarBloco);
                
                const portas = bloco.querySelectorAll('.porta');
                portas.forEach(porta => {{
                    porta.addEventListener('click', clickPorta);
                }});
                
                blocoIdCounter++;
                salvarEstado();
            }}

            function adicionarBlocoTransferencia() {{
                const num = prompt('Numerador (ex: 1, s+1):', '1');
                if (num === null) return;
                const den = prompt('Denominador (ex: s+1, s^2+2*s+1):', 's+1');
                if (den === null) return;
                adicionarBloco('Transferência', {{
                    nome: 'G' + blocoIdCounter,
                    numerador: num,
                    denominador: den,
                    tf: num + ' / ' + den
                }});
            }}

            function adicionarBlocoSomador() {{
                adicionarBloco('Somador', {{nome: 'Σ' + blocoIdCounter}});
            }}

            function adicionarBlocoGanho() {{
                const ganho = prompt('Valor do ganho K:', '1');
                if (ganho === null) return;
                adicionarBloco('Ganho', {{nome: 'K=' + ganho, valor: ganho, tf: ganho}});
            }}

            function adicionarBlocoIntegrador() {{
                adicionarBloco('Integrador', {{nome: '∫', tf: '1/s'}});
            }}

            function iniciarArrastar(e) {{
                if (e.target.classList.contains('porta')) return;
                e.stopPropagation();
                arrastandoBloco = e.currentTarget;
                const rect = arrastandoBloco.getBoundingClientRect();
                const container = document.getElementById('canvas-container').getBoundingClientRect();
                offsetX = e.clientX - rect.left;
                offsetY = e.clientY - rect.top;
                
                document.addEventListener('mousemove', arrastar);
                document.addEventListener('mouseup', pararArrastar);
            }}

            function arrastar(e) {{
                if (arrastandoBloco) {{
                    const container = document.getElementById('canvas-container').getBoundingClientRect();
                    let x = e.clientX - container.left - offsetX;
                    let y = e.clientY - container.top - offsetY;
                    
                    x = Math.max(0, Math.min(x, container.width - arrastandoBloco.offsetWidth));
                    y = Math.max(0, Math.min(y, container.height - arrastandoBloco.offsetHeight));
                    
                    arrastandoBloco.style.left = x + 'px';
                    arrastandoBloco.style.top = y + 'px';
                    
                    const blocoId = parseInt(arrastandoBloco.id.split('-')[1]);
                    const bloco = blocos.find(b => b.id === blocoId);
                    if (bloco) {{
                        bloco.x = x;
                        bloco.y = y;
                    }}
                    
                    redesenharConexoes();
                }}
            }}

            function pararArrastar() {{
                arrastandoBloco = null;
                document.removeEventListener('mousemove', arrastar);
                document.removeEventListener('mouseup', pararArrastar);
                salvarEstado();
            }}

            function selecionarBloco(e) {{
                if (e.target.classList.contains('porta')) return;
                e.stopPropagation();
                
                document.querySelectorAll('.bloco').forEach(b => b.classList.remove('selecionado'));
                e.currentTarget.classList.add('selecionado');
                blocoSelecionado = parseInt(e.currentTarget.id.split('-')[1]);
            }}

            function clickPorta(e) {{
                e.stopPropagation();
                const blocoId = parseInt(e.target.dataset.bloco);
                const tipoPorta = e.target.dataset.tipo;
                
                if (!portaSelecionada) {{
                    portaSelecionada = {{blocoId, tipo: tipoPorta}};
                    e.target.style.background = '#FFC107';
                }} else {{
                    if (portaSelecionada.tipo === 'saida' && tipoPorta === 'entrada') {{
                        conexoes.push({{
                            origem: portaSelecionada.blocoId,
                            destino: blocoId
                        }});
                        redesenharConexoes();
                    }} else if (portaSelecionada.tipo === 'entrada' && tipoPorta === 'saida') {{
                        conexoes.push({{
                            origem: blocoId,
                            destino: portaSelecionada.blocoId
                        }});
                        redesenharConexoes();
                    }} else {{
                        alert('Conecte saída → entrada');
                    }}
                    
                    document.querySelectorAll('.porta').forEach(p => p.style.background = '#4CAF50');
                    portaSelecionada = null;
                    salvarEstado();
                }}
            }}

            function redesenharConexoes() {{
                const svg = document.getElementById('conexoes-svg');
                svg.innerHTML = '';
                
                conexoes.forEach(conexao => {{
                    const blocoOrigem = document.getElementById('bloco-' + conexao.origem);
                    const blocoDestino = document.getElementById('bloco-' + conexao.destino);
                    
                    if (blocoOrigem && blocoDestino) {{
                        const rectOrigem = blocoOrigem.getBoundingClientRect();
                        const rectDestino = blocoDestino.getBoundingClientRect();
                        const container = document.getElementById('canvas-container').getBoundingClientRect();
                        
                        const x1 = rectOrigem.right - container.left;
                        const y1 = rectOrigem.top + rectOrigem.height/2 - container.top;
                        const x2 = rectDestino.left - container.left;
                        const y2 = rectDestino.top + rectDestino.height/2 - container.top;
                        
                        const path = document.createElementNS('http://www.w3.org/2000/svg', 'path');
                        const dx = x2 - x1;
                        const curva = Math.abs(dx) / 2;
                        const d = `M ${{x1}} ${{y1}} C ${{x1 + curva}} ${{y1}}, ${{x2 - curva}} ${{y2}}, ${{x2}} ${{y2}}`;
                        
                        path.setAttribute('d', d);
                        path.setAttribute('stroke', '#667eea');
                        path.setAttribute('stroke-width', '3');
                        path.setAttribute('fill', 'none');
                        path.setAttribute('marker-end', 'url(#arrowhead)');
                        
                        svg.appendChild(path);
                    }}
                }});
                
                const defs = document.createElementNS('http://www.w3.org/2000/svg', 'defs');
                const marker = document.createElementNS('http://www.w3.org/2000/svg', 'marker');
                marker.setAttribute('id', 'arrowhead');
                marker.setAttribute('markerWidth', '10');
                marker.setAttribute('markerHeight', '10');
                marker.setAttribute('refX', '9');
                marker.setAttribute('refY', '3');
                marker.setAttribute('orient', 'auto');
                const polygon = document.createElementNS('http://www.w3.org/2000/svg', 'polygon');
                polygon.setAttribute('points', '0 0, 10 3, 0 6');
                polygon.setAttribute('fill', '#667eea');
                marker.appendChild(polygon);
                defs.appendChild(marker);
                svg.appendChild(defs);
            }}

            function removerSelecionado() {{
                if (blocoSelecionado !== null) {{
                    const blocoEl = document.getElementById('bloco-' + blocoSelecionado);
                    if (blocoEl) {{
                        blocoEl.remove();
                        blocos = blocos.filter(b => b.id !== blocoSelecionado);
                        conexoes = conexoes.filter(c => c.origem !== blocoSelecionado && c.destino !== blocoSelecionado);
                        redesenharConexoes();
                        blocoSelecionado = null;
                        salvarEstado();
                    }}
                }} else {{
                    alert('Selecione um bloco primeiro!');
                }}
            }}

            function limparDiagrama() {{
                if (confirm('Deseja limpar todo o diagrama?')) {{
                    blocos = [];
                    conexoes = [];
                    blocoSelecionado = null;
                    document.getElementById('canvas-container').innerHTML = '<svg id="conexoes-svg"></svg><div id="info-panel"><strong>📊 Instruções:</strong><div>• Clique nos botões para adicionar</div><div>• Arraste blocos para mover</div><div>• Clique nas portas verdes para conectar</div><div>• Clique no bloco para selecionar</div></div>';
                    salvarEstado();
                }}
            }}

            function salvarEstado() {{
                window.parent.postMessage({{
                    type: 'salvar_diagrama',
                    blocos: blocos,
                    conexoes: conexoes,
                    contador: blocoIdCounter
                }}, '*');
            }}

            // Inicializar blocos existentes
            blocos.forEach(blocoData => {{
                const container = document.getElementById('canvas-container');
                const bloco = document.createElement('div');
                bloco.className = 'bloco';
                bloco.id = 'bloco-' + blocoData.id;
                bloco.style.left = blocoData.x + 'px';
                bloco.style.top = blocoData.y + 'px';
                
                let nomeDisplay = blocoData.config.nome || blocoData.tipo;
                let tfDisplay = blocoData.config.tf || '';
                
                bloco.innerHTML = `
                    <div class="bloco-tipo">${{blocoData.tipo}}</div>
                    <div class="bloco-nome">${{nomeDisplay}}</div>
                    ${{tfDisplay ? '<div class="bloco-tf">' + tfDisplay + '</div>' : ''}}
                    <div class="porta porta-entrada" data-bloco="${{blocoData.id}}" data-tipo="entrada"></div>
                    <div class="porta porta-saida" data-bloco="${{blocoData.id}}" data-tipo="saida"></div>
                `;
                
                container.appendChild(bloco);
                bloco.addEventListener('mousedown', iniciarArrastar);
                bloco.addEventListener('click', selecionarBloco);
                
                const portas = bloco.querySelectorAll('.porta');
                portas.forEach(porta => {{
                    porta.addEventListener('click', clickPorta);
                }});
            }});

            redesenharConexoes();
        </script>
    </body>
    </html>
    """
    return html_code

def processar_diagrama_blocos():
    """Processa o diagrama de blocos e calcula o sistema equivalente"""
    blocos_diagrama = st.session_state.diagrama_blocos['blocos']
    
    if not blocos_diagrama:
        return None, "Nenhum bloco no diagrama"
    
    try:
        tfs = {}
        for bloco in blocos_diagrama:
            bloco_id = bloco['id']
            config = bloco['config']
            
            if bloco['tipo'] == 'Transferência':
                tf, _ = converter_para_tf(config['numerador'], config['denominador'])
                tfs[bloco_id] = tf
            elif bloco['tipo'] == 'Ganho':
                tfs[bloco_id] = TransferFunction([float(config['valor'])], [1])
            elif bloco['tipo'] == 'Integrador':
                tfs[bloco_id] = TransferFunction([1], [1, 0])
            elif bloco['tipo'] == 'Somador':
                tfs[bloco_id] = TransferFunction([1], [1])
        
        if len(blocos_diagrama) == 1:
            return tfs[blocos_diagrama[0]['id']], "Sistema de bloco único"
        
        sistema_final = tfs[blocos_diagrama[0]['id']]
        for i in range(1, len(blocos_diagrama)):
            sistema_final = sistema_final * tfs[blocos_diagrama[i]['id']]
        
        return sistema_final, f"Sistema com {len(blocos_diagrama)} blocos em série"
    
    except Exception as e:
        return None, f"Erro ao processar diagrama: {str(e)}"

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
    
    if 'modo_editor' not in st.session_state:
        st.session_state.modo_editor = 'classico'
    
    # Seletor de modo
    st.sidebar.header("🎛️ Modo de Trabalho")
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
        st.subheader("🎨 Editor Visual de Diagrama de Blocos")
        
        # Formulário para Função Transferência (se ativado)
        if 'show_tf_form' in st.session_state and st.session_state.show_tf_form:
            with st.form("form_tf"):
                st.markdown("**➕ Nova Função de Transferência**")
                num_tf = st.text_input("Numerador:", "1")
                den_tf = st.text_input("Denominador:", "s+1")
                col_sub1, col_sub2 = st.columns(2)
                with col_sub1:
                    submitted = st.form_submit_button("✅ Adicionar", use_container_width=True)
                with col_sub2:
                    cancelled = st.form_submit_button("❌ Cancelar", use_container_width=True)
                
                if submitted:
                    try:
                        tf, _ = converter_para_tf(num_tf, den_tf)
                        novo_bloco = {
                            'id': st.session_state.bloco_contador,
                            'tipo': 'Transferência',
                            'x': 150 + (len(st.session_state.diagrama_blocos['blocos']) * 200) % 600,
                            'y': 200 + (len(st.session_state.diagrama_blocos['blocos']) * 50) % 300,
                            'config': {
                                'nome': f'G{st.session_state.bloco_contador}',
                                'numerador': num_tf,
                                'denominador': den_tf,
                                'tf': f'{num_tf} / {den_tf}'
                            }
                        }
                        st.session_state.diagrama_blocos['blocos'].append(novo_bloco)
                        st.session_state.bloco_contador += 1
                        st.session_state.show_tf_form = False
                        st.success("✅ Função de Transferência adicionada!")
                        st.rerun()
                    except Exception as e:
                        st.error(f"❌ Erro: {str(e)}")
                
                if cancelled:
                    st.session_state.show_tf_form = False
                    st.rerun()
            st.markdown("---")
        
        # Formulário para Ganho (se ativado)
        if 'show_gain_form' in st.session_state and st.session_state.show_gain_form:
            with st.form("form_ganho"):
                st.markdown("**📊 Novo Ganho**")
                ganho_val = st.number_input("Valor do ganho K:", value=1.0, step=0.1)
                col_sub1, col_sub2 = st.columns(2)
                with col_sub1:
                    submitted = st.form_submit_button("✅ Adicionar", use_container_width=True)
                with col_sub2:
                    cancelled = st.form_submit_button("❌ Cancelar", use_container_width=True)
                
                if submitted:
                    novo_bloco = {
                        'id': st.session_state.bloco_contador,
                        'tipo': 'Ganho',
                        'x': 150 + (len(st.session_state.diagrama_blocos['blocos']) * 200) % 600,
                        'y': 200 + (len(st.session_state.diagrama_blocos['blocos']) * 50) % 300,
                        'config': {
                            'nome': f'K={ganho_val}',
                            'valor': str(ganho_val),
                            'tf': str(ganho_val)
                        }
                    }
                    st.session_state.diagrama_blocos['blocos'].append(novo_bloco)
                    st.session_state.bloco_contador += 1
                    st.session_state.show_gain_form = False
                    st.success("✅ Ganho adicionado!")
                    st.rerun()
                
                if cancelled:
                    st.session_state.show_gain_form = False
                    st.rerun()
            st.markdown("---")
        
        # Visualização do diagrama
        html_editor = criar_diagrama_blocos_html()
        components.html(html_editor, height=700, scrolling=False)
        
        st.markdown("---")
        
        # Painel de controle unificado
        col_add, col_list, col_action = st.columns([1, 2, 1])
        
        with col_add:
            st.markdown("### 🧱 Adicionar")
            if st.button("➕ Função Transferência", use_container_width=True, key="btn_add_tf"):
                st.session_state.show_tf_form = True
                st.rerun()
            
            if st.button("⊕ Somador", use_container_width=True, key="btn_add_sum"):
                novo_bloco = {
                    'id': st.session_state.bloco_contador,
                    'tipo': 'Somador',
                    'x': 150 + (len(st.session_state.diagrama_blocos['blocos']) * 200) % 600,
                    'y': 200 + (len(st.session_state.diagrama_blocos['blocos']) * 50) % 300,
                    'config': {'nome': f'Σ{st.session_state.bloco_contador}'}
                }
                st.session_state.diagrama_blocos['blocos'].append(novo_bloco)
                st.session_state.bloco_contador += 1
                st.success("✅ Somador adicionado!")
                st.rerun()
            
            if st.button("📊 Ganho", use_container_width=True, key="btn_add_gain"):
                st.session_state.show_gain_form = True
                st.rerun()
            
            if st.button("∫ Integrador", use_container_width=True, key="btn_add_int"):
                novo_bloco = {
                    'id': st.session_state.bloco_contador,
                    'tipo': 'Integrador',
                    'x': 150 + (len(st.session_state.diagrama_blocos['blocos']) * 200) % 600,
                    'y': 200 + (len(st.session_state.diagrama_blocos['blocos']) * 50) % 300,
                    'config': {'nome': '∫', 'tf': '1/s'}
                }
                st.session_state.diagrama_blocos['blocos'].append(novo_bloco)
                st.session_state.bloco_contador += 1
                st.success("✅ Integrador adicionado!")
                st.rerun()
        
        with col_list:
            st.markdown("### 🗂️ Blocos no Diagrama")
            if st.session_state.diagrama_blocos['blocos']:
                for bloco in st.session_state.diagrama_blocos['blocos']:
                    col_info, col_del = st.columns([5, 1])
                    with col_info:
                        tf_info = f" - {bloco['config'].get('tf', '')}" if bloco['config'].get('tf') else ""
                        st.text(f"🔹 {bloco['tipo']}: {bloco['config']['nome']}{tf_info}")
                    with col_del:
                        if st.button("🗑️", key=f"del_{bloco['id']}"):
                            st.session_state.diagrama_blocos['blocos'] = [
                                b for b in st.session_state.diagrama_blocos['blocos'] if b['id'] != bloco['id']
                            ]
                            st.session_state.diagrama_blocos['conexoes'] = [
                                c for c in st.session_state.diagrama_blocos['conexoes'] 
                                if c['origem'] != bloco['id'] and c['destino'] != bloco['id']
                            ]
                            st.success(f"✅ Bloco removido!")
                            st.rerun()
            else:
                st.info("➕ Nenhum bloco adicionado. Use os botões ao lado.")
        
        with col_action:
            st.markdown("### ⚙️ Ações")
            
            if st.button("⚡ Processar Sistema", type="primary", use_container_width=True):
                sistema, msg = processar_diagrama_blocos()
                if sistema:
                    st.success(msg)
                    
                    # Mostrar análise em expander
                    with st.expander("📊 Ver Análise Completa", expanded=True):
                        desempenho = calcular_desempenho(sistema)
                        st.markdown("**Métricas:**")
                        for chave, valor in desempenho.items():
                            st.markdown(f"- **{chave}:** {valor}")
                        
                        tab1, tab2, tab3 = st.tabs(["📈 Temporal", "📊 Bode", "🎯 Polos/Zeros"])
                        
                        with tab1:
                            fig, t, y = plot_resposta_temporal(sistema, 'Degrau')
                            st.plotly_chart(fig, use_container_width=True)
                        
                        with tab2:
                            fig = plot_bode(sistema, 'both')
                            st.plotly_chart(fig, use_container_width=True)
                        
                        with tab3:
                            fig = plot_polos_zeros(sistema)
                            st.plotly_chart(fig, use_container_width=True)
                else:
                    st.error(msg)
            
            if st.button("🔄 Limpar Tudo", use_container_width=True):
                if st.session_state.diagrama_blocos['blocos']:
                    st.session_state.diagrama_blocos = {'blocos': [], 'conexoes': []}
                    st.session_state.bloco_contador = 1
                    st.success("✅ Diagrama limpo!")
                    st.rerun()
                else:
                    st.warning("Diagrama já está vazio!")
            
            if st.button("💾 Exportar JSON", use_container_width=True):
                if st.session_state.diagrama_blocos['blocos']:
                    diagrama_json = json.dumps(st.session_state.diagrama_blocos, indent=2)
                    st.download_button(
                        label="📥 Baixar",
                        data=diagrama_json,
                        file_name="diagrama_blocos.json",
                        mime="application/json",
                        use_container_width=True
                    )
                else:
                    st.warning("Nenhum bloco para exportar!")
        
        # Estatísticas na sidebar
        st.sidebar.markdown("---")
        st.sidebar.markdown("### 📊 Estatísticas do Diagrama")
        st.sidebar.metric("Total de Blocos", len(st.session_state.diagrama_blocos['blocos']))
        st.sidebar.metric("Conexões", len(st.session_state.diagrama_blocos['conexoes']))
        
        st.sidebar.markdown("---")
        st.sidebar.markdown("### 📖 Instruções")
        st.sidebar.info("""
        **Modo Editor Visual:**
        
        1. ➕ Use os botões para adicionar blocos
        2. 🖱️ Arraste blocos no canvas
        3. 🔗 Clique nas portas verdes para conectar
        4. ⚡ Processe para analisar o sistema
        5. 🗑️ Remova blocos individualmente
        """)
        
        return
    
    # =====================================================
    # MODO CLÁSSICO (código original)
    # =====================================================
    
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
    st.sidebar.info("""
    Experimente o **Editor Visual**
    para construir sistemas graficamente!
    """)

if __name__ == "__main__":
    main()
