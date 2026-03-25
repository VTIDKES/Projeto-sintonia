# -*- coding: utf-8 -*-
"""
Sistema de Modelagem e Analise de Sistemas de Controle v2.0
Refatorado com: tela inicial, espaco de estados, modal de blocos,
logica corrigida de serie/paralelo/feedback, simplificacao automatica.
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
import re

# ══════════════════════════════════════════════════
# CONSTANTES
# ══════════════════════════════════════════════════

ANALYSIS_OPTIONS = {
    "malha_aberta": [
        "Resposta no tempo", "Desempenho", "Diagrama de Polos e Zeros",
        "Diagrama De Bode Magnitude", "Diagrama De Bode Fase", "Nyquist"
    ],
    "malha_fechada": [
        "Resposta no tempo", "Desempenho", "Diagrama de Polos e Zeros",
        "Diagrama De Bode Magnitude", "Diagrama De Bode Fase", "LGR"
    ],
}

INPUT_SIGNALS = ['Degrau', 'Rampa', 'Senoidal', 'Impulso', 'Parabolica']

BLOCK_TYPES = {
    'Planta': {'icon': 'G(s)', 'desc': 'Funcao de transferencia da planta'},
    'Controlador': {'icon': 'C(s)', 'desc': 'Controlador (PID, Lead-Lag, etc.)'},
    'Sensor': {'icon': 'H(s)', 'desc': 'Sensor na malha de realimentacao'},
    'Atuador': {'icon': 'A(s)', 'desc': 'Atuador do sistema'},
    'Pre-filtro': {'icon': 'F(s)', 'desc': 'Filtro antes do somador'},
    'Perturbacao': {'icon': 'D(s)', 'desc': 'Perturbacao/disturbio'},
}

CONNECTION_TYPES = ['Serie', 'Paralelo', 'Realimentacao Negativa', 'Realimentacao Positiva']

# ══════════════════════════════════════════════════
# INICIALIZACAO DO SESSION STATE
# ══════════════════════════════════════════════════

def inicializar_estado():
    defaults = {
        'modo_selecionado': None,
        'blocos': pd.DataFrame(columns=[
            'nome', 'tipo', 'representacao', 'numerador', 'denominador',
            'A', 'B', 'C', 'D', 'tf', 'tf_simbolico'
        ]),
        'conexoes': [],
        'calculo_erro_habilitado': False,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


# ══════════════════════════════════════════════════
# FUNCOES UTILITARIAS
# ══════════════════════════════════════════════════

def formatar_numero(valor):
    if np.isinf(valor):
        return '∞'
    if np.isnan(valor):
        return '-'
    return f"{valor:.3f}"


def parse_matrix(text):
    """Converte texto para matriz numpy. Aceita formatos:
       '[[1,2],[3,4]]'  ou  '1 2; 3 4'  ou  '1,2;3,4'
    """
    text = text.strip()
    if text.startswith('['):
        try:
            return np.array(json.loads(text), dtype=float)
        except Exception:
            text = text.replace('[', '').replace(']', '')
    rows = [r.strip() for r in text.split(';') if r.strip()]
    matrix = []
    for row in rows:
        vals = re.split(r'[,\s]+', row.strip())
        matrix.append([float(v) for v in vals if v])
    return np.array(matrix, dtype=float)


# ══════════════════════════════════════════════════
# FUNCOES DE TRANSFERENCIA E ESPACO DE ESTADOS
# ══════════════════════════════════════════════════

def converter_para_tf(numerador_str, denominador_str):
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


def converter_ss_para_tf(A_str, B_str, C_str, D_str):
    """Converte espaco de estados (A,B,C,D) para funcao de transferencia.
    T(s) = C(sI - A)^(-1)B + D
    """
    A = parse_matrix(A_str)
    B = parse_matrix(B_str)
    C = parse_matrix(C_str)
    D = parse_matrix(D_str)
    n = A.shape[0]
    if A.shape != (n, n):
        raise ValueError(f"Matriz A deve ser quadrada ({n}x{n})")
    if B.shape[0] != n:
        raise ValueError(f"Matriz B deve ter {n} linhas")
    if C.shape[1] != n:
        raise ValueError(f"Matriz C deve ter {n} colunas")
    ss_sys = ctrl.ss(A, B, C, D)
    tf_sys = ctrl.tf(ss_sys)
    s = sp.Symbol('s')
    num_coeffs = list(tf_sys.num[0][0])
    den_coeffs = list(tf_sys.den[0][0])
    num_sym = sum(c * s**i for i, c in enumerate(reversed(num_coeffs)))
    den_sym = sum(c * s**i for i, c in enumerate(reversed(den_coeffs)))
    return tf_sys, (num_sym, den_sym), ss_sys


def tipo_do_sistema(G):
    G_min = ctrl.minreal(G, verbose=False)
    polos = ctrl.poles(G_min)
    return sum(1 for p in polos if np.isclose(np.real_if_close(p), 0.0, atol=1e-3))


def constantes_de_erro(G):
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


# ══════════════════════════════════════════════════
# INTERCONEXAO DE BLOCOS (SERIE, PARALELO, FEEDBACK)
# ══════════════════════════════════════════════════

def blocos_em_serie(tf_list):
    """Serie: G_total = G1 * G2 * ... * Gn"""
    resultado = tf_list[0]
    for tf in tf_list[1:]:
        resultado = ctrl.series(resultado, tf)[2]
    return resultado


def blocos_em_paralelo(tf_list):
    """Paralelo: G_total = G1 + G2 + ... + Gn"""
    resultado = tf_list[0]
    for tf in tf_list[1:]:
        resultado = ctrl.parallel(resultado, tf)[2]
    return resultado


def realimentacao(G, H=None, positiva=False):
    """Feedback: T = G / (1 +/- G*H)
    Negativa (padrao): T = G / (1 + G*H)
    Positiva: T = G / (1 - G*H)
    """
    if H is None:
        H = TransferFunction([1], [1])
    sign = 1 if positiva else -1
    return ctrl.feedback(G, H, sign=sign)


def simplificar_diagrama(blocos_df, conexoes):
    """Simplifica um diagrama de blocos baseado nas conexoes definidas.
    Retorna a funcao de transferencia equivalente.
    """
    if blocos_df.empty:
        raise ValueError("Nenhum bloco definido.")

    tf_map = {}
    for _, row in blocos_df.iterrows():
        tf_map[row['nome']] = row['tf']

    if not conexoes:
        tfs = list(tf_map.values())
        if len(tfs) == 1:
            return tfs[0]
        return blocos_em_serie(tfs)

    resultado = None
    for con in conexoes:
        tipo_con = con['tipo']
        nomes = con['blocos']
        tfs = [tf_map[n] for n in nomes if n in tf_map]

        if len(tfs) < 2:
            if len(tfs) == 1:
                resultado = tfs[0] if resultado is None else ctrl.series(resultado, tfs[0])[2]
            continue

        if tipo_con == 'Serie':
            parcial = blocos_em_serie(tfs)
        elif tipo_con == 'Paralelo':
            parcial = blocos_em_paralelo(tfs)
        elif tipo_con == 'Realimentacao Negativa':
            G = tfs[0]
            H = tfs[1] if len(tfs) > 1 else TransferFunction([1], [1])
            parcial = realimentacao(G, H, positiva=False)
        elif tipo_con == 'Realimentacao Positiva':
            G = tfs[0]
            H = tfs[1] if len(tfs) > 1 else TransferFunction([1], [1])
            parcial = realimentacao(G, H, positiva=True)
        else:
            parcial = blocos_em_serie(tfs)

        if resultado is None:
            resultado = parcial
        else:
            resultado = ctrl.series(resultado, parcial)[2]

    if resultado is None:
        tfs = list(tf_map.values())
        resultado = blocos_em_serie(tfs) if len(tfs) > 1 else tfs[0]

    return ctrl.minreal(resultado, verbose=False)


def calcular_malha_fechada(planta, controlador=None, sensor=None):
    if controlador is None:
        controlador = TransferFunction([1], [1])
    if sensor is None:
        sensor = TransferFunction([1], [1])
    G = controlador * planta
    H = sensor
    return ctrl.feedback(G, H)


# ══════════════════════════════════════════════════
# ANALISE DE SISTEMAS
# ══════════════════════════════════════════════════

def calcular_desempenho(tf_sys):
    den = tf_sys.den[0][0]
    ordem = len(den) - 1
    polos = ctrl.poles(tf_sys)
    gm, pm, wg, wp = margin(tf_sys)
    gm_db = 20 * np.log10(gm) if gm != np.inf and gm > 0 else np.inf
    resultado = {
        'Margem de ganho': f"{formatar_numero(gm)} ({'∞' if gm == np.inf else f'{formatar_numero(gm_db)} dB'})",
        'Margem de fase': f"{formatar_numero(pm)}°",
        'Freq. cruz. fase': f"{formatar_numero(wg)} rad/s",
        'Freq. cruz. ganho': f"{formatar_numero(wp)} rad/s",
    }
    if ordem == 1:
        return _desempenho_ordem1(polos, resultado)
    elif ordem == 2:
        return _desempenho_ordem2(polos, resultado)
    else:
        return _desempenho_ordem_superior(polos, ordem, resultado)


def _desempenho_ordem1(polos, resultado):
    tau = -1 / polos[0].real
    resultado.update({
        'Tipo': '1a Ordem',
        'Const. tempo (t)': f"{formatar_numero(tau)} s",
        'Temp. subida (Tr)': f"{formatar_numero(2.2 * tau)} s",
        'Temp. acomodacao (Ts)': f"{formatar_numero(4 * tau)} s",
        'Freq. natural (wn)': f"{formatar_numero(1/tau)} rad/s",
        'Fator amortec. (z)': "1.0",
    })
    return resultado


def _desempenho_ordem2(polos, resultado):
    wn = np.sqrt(np.prod(np.abs(polos))).real
    zeta = -np.real(polos[0]) / wn
    wd = wn * np.sqrt(1 - zeta**2) if zeta < 1 else 0
    Mp = np.exp(-zeta * np.pi / np.sqrt(1 - zeta**2)) * 100 if 0 < zeta < 1 else 0
    Tr = (np.pi - np.arccos(zeta)) / wd if zeta < 1 and wd > 0 else float('inf')
    Tp = np.pi / wd if wd > 0 else float('inf')
    Ts = 4 / (zeta * wn) if zeta * wn > 0 else float('inf')
    resultado.update({
        'Tipo': '2a Ordem',
        'Freq. natural (wn)': f"{formatar_numero(wn)} rad/s",
        'Fator amortec. (z)': f"{formatar_numero(zeta)}",
        'Freq. amortec. (wd)': f"{formatar_numero(wd)} rad/s",
        'Sobressinal (Mp)': f"{formatar_numero(Mp)}%",
        'Temp. subida (Tr)': f"{formatar_numero(Tr)} s",
        'Temp. pico (Tp)': f"{formatar_numero(Tp)} s",
        'Temp. acomodacao (Ts)': f"{formatar_numero(Ts)} s",
    })
    return resultado


def _desempenho_ordem_superior(polos, ordem, resultado):
    polos_ordenados = sorted(polos, key=lambda p: np.real(p), reverse=True)
    par_dominante = None
    for i in range(len(polos_ordenados) - 1):
        p1, p2 = polos_ordenados[i], polos_ordenados[i + 1]
        if (np.isclose(p1.real, p2.real, atol=1e-2)
                and np.isclose(p1.imag, -p2.imag, atol=1e-2)):
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
    Mp = np.exp(-zeta * np.pi / np.sqrt(1 - zeta**2)) * 100 if 0 < zeta < 1 else 0
    Tr = (np.pi - np.arccos(zeta)) / omega_d if zeta < 1 and omega_d > 0 else float('inf')
    Tp = np.pi / omega_d if omega_d > 0 else float('inf')
    Ts = 4 / (zeta * wn) if zeta * wn > 0 else float('inf')
    label = f'{ordem}a Ordem (Par dominante)' if par_dominante else f'{ordem}a Ordem (Polo dominante)'
    resultado.update({
        'Tipo': label,
        'Freq. natural (wn)': f"{formatar_numero(wn)} rad/s",
        'Fator amortec. (z)': f"{formatar_numero(zeta)}",
        'Freq. amortec. (wd)': f"{formatar_numero(omega_d)} rad/s",
        'Sobressinal (Mp)': f"{formatar_numero(Mp)} %",
        'Temp. subida (Tr)': f"{formatar_numero(Tr)} s",
        'Temp. pico (Tp)': f"{formatar_numero(Tp)} s",
        'Temp. acomodacao (Ts)': f"{formatar_numero(Ts)} s",
    })
    return resultado


def estimar_tempo_final_simulacao(tf_sys):
    polos = ctrl.poles(tf_sys)
    if len(polos) == 0:
        return 50.0
    if any(np.real(p) > 1e-6 for p in polos):
        return 20.0
    partes_reais_estaveis = [np.real(p) for p in polos if np.real(p) < -1e-6]
    if not partes_reais_estaveis:
        return 100.0
    sigma_dominante = max(partes_reais_estaveis)
    ts_estimado = 4 / abs(sigma_dominante)
    return np.clip(ts_estimado * 1.5, a_min=10, a_max=500)


# ══════════════════════════════════════════════════
# FUNCOES DE PLOTAGEM
# ══════════════════════════════════════════════════

def configurar_linhas_interativas(fig):
    fig.update_layout(
        dragmode='zoom',
        newshape=dict(line=dict(color='green', width=2, dash='dash')),
        modebar_add=['drawline', 'drawopenpath', 'drawclosedpath',
                     'drawcircle', 'drawrect', 'eraseshape'],
    )
    return fig


def plot_polos_zeros(tf_sys, fig=None):
    zeros = ctrl.zeros(tf_sys)
    polos = ctrl.poles(tf_sys)
    if fig is None:
        fig = go.Figure()
    if len(zeros) > 0:
        fig.add_trace(go.Scatter(
            x=np.real(zeros), y=np.imag(zeros), mode='markers',
            marker=dict(symbol='circle', size=12, color='blue'), name='Zeros',
            hovertemplate='Zero<br>Real: %{x:.3f}<br>Imaginario: %{y:.3f}<extra></extra>'))
    if len(polos) > 0:
        fig.add_trace(go.Scatter(
            x=np.real(polos), y=np.imag(polos), mode='markers',
            marker=dict(symbol='x', size=12, color='red'), name='Polos',
            hovertemplate='Polo<br>Real: %{x:.3f}<br>Imaginario: %{y:.3f}<extra></extra>'))
    fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.7)
    fig.add_vline(x=0, line_dash="dash", line_color="gray", opacity=0.7)
    fig.update_layout(
        title='Diagrama de Polos e Zeros', xaxis_title='Parte Real',
        yaxis_title='Parte Imaginaria', showlegend=True, hovermode='closest')
    return configurar_linhas_interativas(fig)


def _gerar_sinal_entrada(entrada, t):
    sinais = {
        'Degrau': np.ones_like(t),
        'Rampa': t,
        'Senoidal': np.sin(2 * np.pi * t),
        'Impulso': np.concatenate([[1], np.zeros(len(t) - 1)]),
        'Parabolica': t**2,
    }
    return sinais[entrada]


def plot_resposta_temporal(sistema, entrada):
    tempo_final = estimar_tempo_final_simulacao(sistema)
    t = np.linspace(0, tempo_final, 1000)
    u = _gerar_sinal_entrada(entrada, t)
    if entrada == 'Degrau':
        t_out, y = step_response(sistema, t)
    else:
        t_out, y, _ = forced_response(sistema, t, u, return_x=True)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=t_out, y=u[:len(t_out)], mode='lines',
        line=dict(dash='dash', color='blue'), name='Entrada',
        hovertemplate='Tempo: %{x:.2f}s<br>Entrada: %{y:.3f}<extra></extra>'))
    fig.add_trace(go.Scatter(
        x=t_out, y=y, mode='lines', line=dict(color='red'), name='Saida',
        hovertemplate='Tempo: %{x:.2f}s<br>Saida: %{y:.3f}<extra></extra>'))
    fig.update_layout(
        title=f'Resposta Temporal - Entrada: {entrada}',
        xaxis_title='Tempo (s)', yaxis_title='Amplitude',
        showlegend=True, hovermode='x unified')
    return configurar_linhas_interativas(fig), t_out, y


def plot_bode(sistema, tipo='both'):
    numerator = sistema.num[0][0]
    denominator = sistema.den[0][0]
    sys_scipy = signal.TransferFunction(numerator, denominator)
    w = np.logspace(-3, 3, 1000)
    w, mag, phase = signal.bode(sys_scipy, w)
    if tipo == 'both':
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('Bode - Magnitude', 'Bode - Fase'),
            vertical_spacing=0.1)
        fig.add_trace(go.Scatter(
            x=w, y=mag, mode='lines', line=dict(color='blue', width=3),
            name='Magnitude', showlegend=False), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=w, y=phase, mode='lines', line=dict(color='red', width=3),
            name='Fase', showlegend=False), row=2, col=1)
        fig.update_xaxes(title_text="Frequencia (rad/s)", type="log", row=1, col=1)
        fig.update_xaxes(title_text="Frequencia (rad/s)", type="log", row=2, col=1)
        fig.update_yaxes(title_text="Magnitude (dB)", row=1, col=1)
        fig.update_yaxes(title_text="Fase (deg)", row=2, col=1)
        fig.update_layout(height=700, title_text="Diagrama de Bode")
    elif tipo == 'magnitude':
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=w, y=mag, mode='lines', line=dict(color='blue', width=3),
            name='Magnitude'))
        fig.update_layout(
            title='Bode - Magnitude', xaxis_title="Frequencia (rad/s)",
            yaxis_title="Magnitude (dB)", xaxis_type='log')
    else:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=w, y=phase, mode='lines', line=dict(color='red', width=3),
            name='Fase'))
        fig.update_layout(
            title='Bode - Fase', xaxis_title="Frequencia (rad/s)",
            yaxis_title="Fase (deg)", xaxis_type='log')
    return configurar_linhas_interativas(fig)


def plot_lgr(sistema):
    rlist, klist = root_locus(sistema, plot=False)
    fig = go.Figure()
    for i, r in enumerate(rlist.T):
        fig.add_trace(go.Scatter(
            x=np.real(r), y=np.imag(r), mode='lines',
            line=dict(color='blue', width=1),
            name=f'Ramo {i+1}', showlegend=False))
    zeros = ctrl.zeros(sistema)
    polos = ctrl.poles(sistema)
    if len(zeros) > 0:
        fig.add_trace(go.Scatter(
            x=np.real(zeros), y=np.imag(zeros), mode='markers',
            marker=dict(symbol='circle', size=10, color='green'), name='Zeros'))
    if len(polos) > 0:
        fig.add_trace(go.Scatter(
            x=np.real(polos), y=np.imag(polos), mode='markers',
            marker=dict(symbol='x', size=12, color='red'), name='Polos'))
    fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.7)
    fig.add_vline(x=0, line_dash="dash", line_color="gray", opacity=0.7)
    fig.update_layout(
        title='Lugar Geometrico das Raizes (LGR)',
        xaxis_title='Parte Real', yaxis_title='Parte Imaginaria',
        showlegend=True, hovermode='closest')
    return configurar_linhas_interativas(fig)


def plot_nyquist(sistema):
    sistema_scipy = signal.TransferFunction(sistema.num[0][0], sistema.den[0][0])
    w = np.logspace(-2, 2, 1000)
    _, H = signal.freqresp(sistema_scipy, w)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=H.real, y=H.imag, mode='lines',
        line=dict(color='blue', width=2), name='Nyquist'))
    fig.add_trace(go.Scatter(
        x=H.real, y=-H.imag, mode='lines',
        line=dict(dash='dash', color='gray', width=1), name='Reflexo simetrico'))
    fig.add_trace(go.Scatter(
        x=[-1], y=[0], mode='markers',
        marker=dict(symbol='circle', size=12, color='red'),
        name='Ponto critico (-1,0)'))
    fig.add_hline(y=0, line_color="black", line_width=1)
    fig.add_vline(x=0, line_color="black", line_width=1)
    fig.update_layout(
        title='Diagrama de Nyquist', xaxis_title='Parte Real',
        yaxis_title='Parte Imaginaria', showlegend=True, hovermode='closest')
    fig = configurar_linhas_interativas(fig)
    polos = ctrl.poles(sistema)
    polos_spd = sum(1 for p in polos if np.real(p) > 0)
    voltas = 0
    Z = polos_spd + voltas
    return fig, polos_spd, voltas, Z


# ══════════════════════════════════════════════════
# GERENCIAMENTO DE BLOCOS
# ══════════════════════════════════════════════════

def adicionar_bloco(nome, tipo, representacao, numerador='', denominador='',
                    A_str='', B_str='', C_str='', D_str=''):
    try:
        if representacao == 'Funcao de Transferencia':
            tf_obj, tf_symb = converter_para_tf(numerador, denominador)
            ss_sys = None
        else:
            tf_obj, tf_symb, ss_sys = converter_ss_para_tf(A_str, B_str, C_str, D_str)
            numerador = str(list(tf_obj.num[0][0]))
            denominador = str(list(tf_obj.den[0][0]))

        novo = pd.DataFrame([{
            'nome': nome, 'tipo': tipo, 'representacao': representacao,
            'numerador': numerador, 'denominador': denominador,
            'A': A_str, 'B': B_str, 'C': C_str, 'D': D_str,
            'tf': tf_obj, 'tf_simbolico': tf_symb,
        }])
        st.session_state.blocos = pd.concat(
            [st.session_state.blocos, novo], ignore_index=True)
        return True, f"Bloco '{nome}' adicionado com sucesso."
    except Exception as e:
        return False, f"Erro: {e}"


def remover_bloco(nome):
    st.session_state.blocos = st.session_state.blocos[
        st.session_state.blocos['nome'] != nome]
    st.session_state.conexoes = [
        c for c in st.session_state.conexoes if nome not in c['blocos']]
    return f"Bloco '{nome}' removido."


def obter_bloco_por_tipo(tipo):
    df = st.session_state.blocos
    if any(df['tipo'] == tipo):
        return df[df['tipo'] == tipo].iloc[0]['tf']
    return None


def obter_bloco_por_nome(nome):
    df = st.session_state.blocos
    match = df[df['nome'] == nome]
    if not match.empty:
        return match.iloc[0]['tf']
    return None


# ══════════════════════════════════════════════════
# EXECUCAO DE ANALISES
# ══════════════════════════════════════════════════

def executar_analises(sistema, analises, entrada, tipo_malha):
    for analise in analises:
        st.markdown(f"### {analise}")
        try:
            if analise == 'Resposta no tempo':
                fig, t_out, y = plot_resposta_temporal(sistema, entrada)
                st.plotly_chart(fig, use_container_width=True)
            elif analise == 'Desempenho':
                desempenho = calcular_desempenho(sistema)
                cols = st.columns(3)
                items = list(desempenho.items())
                for i, (chave, valor) in enumerate(items):
                    with cols[i % 3]:
                        st.metric(label=chave, value=valor)
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
                c1, c2, c3 = st.columns(3)
                c1.metric("Polos SPD (P)", polos_spd)
                c2.metric("Voltas (N)", voltas)
                c3.metric("Z = P + N", f"{Z} ({'Estavel' if Z == 0 else 'Instavel'})")
                st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Erro em '{analise}': {e}")


# ══════════════════════════════════════════════════
# EDITOR VISUAL HTML (MODO CANVAS)
# ══════════════════════════════════════════════════

def _load_visual_editor_html():
    return r'''<!DOCTYPE html>
<html lang="pt-BR"><head><meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1,maximum-scale=1,user-scalable=no">
<style>
:root{--bg:#0e1117;--sf:#1a1d2e;--sf2:#252840;--bd:#333654;--tx:#e0e4f0;--txm:#8890b0;
--acc:#5b6be0;--grn:#34d399;--red:#f87171;--yel:#fbbf24;--blu:#60a5fa;--pur:#a78bfa;--pnk:#f472b6}
*{margin:0;padding:0;box-sizing:border-box}
html,body{height:auto;min-height:100%;font-family:system-ui,sans-serif;background:var(--bg);color:var(--tx);overflow-y:auto}
.app{display:flex;flex-direction:column;min-height:100%}
/* Modal Overlay */
.modal-overlay{display:none;position:fixed;inset:0;background:rgba(0,0,0,.6);z-index:1000;align-items:center;justify-content:center}
.modal-overlay.vis{display:flex}
.modal{background:var(--sf);border:1px solid var(--bd);border-radius:16px;width:90%;max-width:520px;max-height:80vh;overflow-y:auto;box-shadow:0 20px 60px rgba(0,0,0,.5)}
.modal-hdr{display:flex;align-items:center;justify-content:space-between;padding:16px 20px;border-bottom:1px solid var(--bd)}
.modal-hdr h3{font-size:16px;color:var(--grn)}
.modal-close{background:none;border:none;color:var(--txm);font-size:22px;cursor:pointer;padding:4px 8px}
.modal-close:hover{color:var(--red)}
.modal-body{padding:16px 20px}
.block-grid{display:grid;grid-template-columns:1fr 1fr;gap:10px}
.block-option{background:var(--sf2);border:1px solid var(--bd);border-radius:10px;padding:14px;cursor:pointer;transition:all .15s;text-align:center}
.block-option:hover{border-color:var(--acc);background:#2f3349;transform:scale(1.03)}
.block-option .bo-icon{font-size:22px;font-weight:700;margin-bottom:4px}
.block-option .bo-label{font-size:12px;font-weight:600;color:var(--tx)}
.block-option .bo-desc{font-size:10px;color:var(--txm);margin-top:2px}
.cat-label{font-size:10px;text-transform:uppercase;letter-spacing:1px;color:var(--acc);margin:12px 0 6px;font-weight:700}
.cat-label:first-child{margin-top:0}
/* Config sub-panel inside modal */
.cfg-panel{display:none;padding:12px 0 0}
.cfg-panel.vis{display:block}
.cfg-panel h4{font-size:12px;color:var(--txm);margin-bottom:8px}
.cfg-row{display:grid;grid-template-columns:1fr 1fr;gap:8px;margin-bottom:8px}
.cfg-row label{font-size:10px;text-transform:uppercase;color:var(--txm);display:block;margin-bottom:3px}
.cfg-row input{width:100%;padding:8px;background:var(--sf2);border:1px solid var(--bd);border-radius:6px;color:var(--tx);font-size:13px;outline:none}
.cfg-row input:focus{border-color:var(--acc)}
.cfg-btn{background:var(--acc);border:none;color:#fff;font-weight:700;font-size:13px;padding:10px 24px;border-radius:8px;cursor:pointer;width:100%;margin-top:8px}
.cfg-btn:hover{background:#6e7df0}
.toolbar{background:var(--sf);border-bottom:1px solid var(--bd);padding:6px 10px;display:flex;align-items:center;gap:5px;flex-wrap:wrap;min-height:44px;z-index:50;position:sticky;top:0}
.toolbar .sep{width:1px;height:26px;background:var(--bd);margin:0 3px}
.toolbar .lbl{font-size:10px;color:var(--txm);text-transform:uppercase;letter-spacing:.5px;margin-right:2px;white-space:nowrap}
.tb{display:inline-flex;align-items:center;gap:4px;background:var(--sf2);color:#c8cad8;border:1px solid var(--bd);
border-radius:6px;padding:6px 10px;font-size:11px;cursor:pointer;white-space:nowrap;transition:all .15s;min-height:34px;touch-action:manipulation}
.tb:hover,.tb:active{background:#2f3349;border-color:var(--acc);color:#fff}
.tb-add{background:#16382a;border-color:#2d8a55;color:var(--grn);font-weight:700;font-size:13px;padding:8px 16px}
.tb-add:hover{background:#1e4a36}
.workspace{display:flex;height:460px;min-height:320px;flex-shrink:0}
.canvas-wrap{flex:1;position:relative;overflow:hidden;background:radial-gradient(circle,#141722,#0e1117);touch-action:none}
.canvas-wrap::before{content:"";position:absolute;inset:0;background-image:radial-gradient(circle,#1e2235 1px,transparent 1px);background-size:24px 24px;opacity:.45;pointer-events:none}
.canvas{position:absolute;inset:0}
.wires{position:absolute;inset:0;width:100%;height:100%;pointer-events:none;z-index:2}
.wires path{pointer-events:visibleStroke;cursor:pointer}
.block{position:absolute;z-index:10;border-radius:10px;cursor:grab;user-select:none;min-width:120px;box-shadow:0 4px 16px rgba(0,0,0,.35)}
.block:active{cursor:grabbing}
.block.sel{box-shadow:0 0 0 2px var(--yel),0 0 20px rgba(251,191,36,.2)!important}
.block-header{font-size:9px;text-transform:uppercase;letter-spacing:.7px;padding:6px 10px 2px;opacity:.6}
.block-body{padding:4px 10px 8px;font-weight:700;font-size:13px;color:#fff}
.block-tf-disp{font-family:monospace;font-size:10px;margin-top:3px;padding:3px 6px;background:rgba(0,0,0,.3);border-radius:4px;color:#a0b8d8;text-align:center}
.block-tf-disp .tf-num{border-bottom:1px solid rgba(255,255,255,.2);padding-bottom:2px;margin-bottom:2px}
.block-tf{background:linear-gradient(135deg,#1e3a5f,#152844);border:1px solid #2a5a8f}
.block-gain{background:linear-gradient(135deg,#2d1f4e,#1f1635);border:1px solid #5a3d8f}
.block-int{background:linear-gradient(135deg,#3d3a1a,#2a2812);border:1px solid #8a7a2d}
.block-der{background:linear-gradient(135deg,#2a2a2a,#1e1e1e);border:1px solid #555}
.block-pid{background:linear-gradient(135deg,#2a1f4e,#1a1535);border:1px solid #5548a0}
.block-sensor{background:linear-gradient(135deg,#4a1a2d,#351020);border:1px solid #8a2d50}
.block-actuator{background:linear-gradient(135deg,#1a3a4a,#102535);border:1px solid #2d708a}
.block-input{background:linear-gradient(135deg,#1a3d1a,#0f250f);border:1px solid #2d8a2d;border-radius:20px}
.block-output{background:linear-gradient(135deg,#3d1a1a,#250f0f);border:1px solid #8a2d2d;border-radius:20px}
.block-sum{background:linear-gradient(135deg,#1a3d3a,#122a28);border:2px solid #2d8a70;border-radius:50%;min-width:56px;width:56px;height:56px;display:flex;align-items:center;justify-content:center}
.block-sum .block-header{display:none}.block-sum .block-body{padding:0;font-size:20px;text-align:center}
.block-branch{background:#3b82f6;border:2px solid #2563eb;border-radius:50%;min-width:20px;width:20px;height:20px}
.block-branch .block-header,.block-branch .block-body{display:none}
.port{position:absolute;width:14px;height:14px;border-radius:50%;cursor:crosshair;z-index:20;transition:transform .12s}
.port::after{content:"";position:absolute;inset:3px;border-radius:50%;background:currentColor}
.port-in{border:2px solid var(--grn);color:var(--grn);background:var(--bg)}
.port-out{border:2px solid var(--blu);color:var(--blu);background:var(--bg)}
.port:hover,.port:active{transform:scale(1.5)!important;box-shadow:0 0 10px currentColor}
.port.active{border-color:var(--yel);color:var(--yel);transform:scale(1.4)!important}
.sign-label{position:absolute;font-size:11px;font-weight:700;color:var(--grn);pointer-events:none}
.panel{width:240px;background:var(--sf);border-left:1px solid var(--bd);display:flex;flex-direction:column;overflow-y:auto;z-index:30;font-size:12px}
.panel-section{padding:10px 12px;border-bottom:1px solid var(--bd)}
.panel-section h4{font-size:11px;text-transform:uppercase;letter-spacing:.5px;color:var(--txm);margin-bottom:8px}
.pg{margin-bottom:8px}
.pg label{display:block;font-size:10px;text-transform:uppercase;color:var(--txm);margin-bottom:3px}
.pg input{width:100%;padding:6px 8px;background:var(--sf2);border:1px solid var(--bd);border-radius:5px;color:var(--tx);font-size:12px;outline:none}
.pg input:focus{border-color:var(--acc)}
.hint{font-size:11px;color:var(--txm);line-height:1.5}
.hint b{color:var(--tx)}
.mode-bar{display:flex;gap:0;background:var(--sf);border-top:1px solid var(--bd)}
.mode-btn{flex:1;padding:12px;background:none;border:none;border-bottom:3px solid transparent;color:var(--txm);font-size:14px;font-weight:700;cursor:pointer;transition:all .15s;touch-action:manipulation;letter-spacing:.3px}
.mode-btn.active{color:var(--grn);border-bottom-color:var(--grn);background:rgba(52,211,153,.06)}
.mode-btn:hover{color:var(--tx)}
.manual-sec{background:var(--sf);padding:20px;border-top:1px solid var(--bd);display:none}
.manual-sec.vis{display:block}
.manual-sec h4{font-size:12px;text-transform:uppercase;color:var(--acc);margin-bottom:14px;letter-spacing:.5px}
.man-tabs{display:flex;gap:8px;margin-bottom:16px;flex-wrap:wrap}
.man-tab{padding:8px 16px;background:var(--sf2);border:1px solid var(--bd);border-radius:8px;color:var(--txm);font-size:12px;font-weight:600;cursor:pointer;transition:all .15s;touch-action:manipulation}
.man-tab.active{background:var(--acc);border-color:var(--acc);color:#fff}
.man-tab:hover:not(.active){border-color:var(--acc);color:var(--tx)}
.man-row{display:grid;grid-template-columns:1fr 1fr;gap:14px;margin-bottom:12px}
.man-row .pg input{font-size:14px;padding:10px 12px}
.man-row .pg label{font-size:11px;margin-bottom:4px}
.man-hint{font-size:11px;color:var(--txm);margin-top:6px;line-height:1.6}
.man-hint code{background:var(--sf2);padding:1px 5px;border-radius:3px;color:var(--blu);font-size:11px}
.calc-bar{background:var(--sf);border-top:1px solid var(--bd);border-bottom:2px solid #22c55e;padding:12px 20px;text-align:center}
.calc-bar button{background:#16a34a;border:2px solid #22c55e;color:#fff;font-weight:700;font-size:16px;padding:14px 40px;border-radius:10px;cursor:pointer;letter-spacing:1px;box-shadow:0 0 20px rgba(34,197,94,.3);touch-action:manipulation}
.calc-bar button:hover{background:#22c55e;transform:scale(1.03)}
.results{background:var(--sf);padding:0;display:none}
.results.vis{display:block}
.results-hdr{padding:16px 20px 10px;display:flex;align-items:center;justify-content:space-between}
.results-hdr h3{font-size:16px;color:var(--grn)}
.rbody{padding:0 20px 20px}
.rcard{background:var(--sf2);border:1px solid var(--bd);border-radius:10px;padding:16px;margin-bottom:16px}
.rcard h4{font-size:12px;text-transform:uppercase;color:var(--acc);margin-bottom:10px}
.tf-disp{font-family:monospace;text-align:center;padding:12px;background:rgba(0,0,0,.3);border-radius:6px;font-size:14px;line-height:1.8}
.mgrid{display:grid;grid-template-columns:repeat(auto-fit,minmax(130px,1fr));gap:10px}
.mbox{background:rgba(0,0,0,.2);border-radius:8px;padding:10px;text-align:center}
.mbox .ml{font-size:10px;text-transform:uppercase;color:var(--txm);margin-bottom:4px}
.mbox .mv{font-size:16px;font-weight:700;color:var(--grn)}
.ebox{background:#3a1520;border:1px solid #8a3050;color:#ff8fa3;padding:12px;border-radius:8px;font-size:13px}
.pzl{font-family:monospace;font-size:12px;line-height:1.8}
@media(max-width:768px){
.workspace{flex-direction:column;height:350px}
.panel{width:100%;max-height:150px;border-left:none;border-top:1px solid var(--bd)}
.toolbar{padding:4px 6px;gap:3px}.tb{padding:5px 7px;font-size:10px;min-height:30px}
.calc-bar button{font-size:14px;padding:10px 24px;width:100%}
.mgrid{grid-template-columns:repeat(2,1fr)}
.man-row{grid-template-columns:1fr}.man-tabs{flex-wrap:wrap}
.block-grid{grid-template-columns:1fr}
}
</style></head><body><div class="app">

<!-- MODAL DE ADICIONAR BLOCO -->
<div class="modal-overlay" id="addModal">
<div class="modal">
<div class="modal-hdr"><h3>Adicionar Bloco</h3><button class="modal-close" onclick="closeModal()">&times;</button></div>
<div class="modal-body">
<div id="modalGrid">
<div class="cat-label">Sinais</div>
<div class="block-grid">
<div class="block-option" onclick="pickBlock('input')"><div class="bo-icon" style="color:var(--grn)">R(s)</div><div class="bo-label">Entrada</div><div class="bo-desc">Sinal de referencia</div></div>
<div class="block-option" onclick="pickBlock('output')"><div class="bo-icon" style="color:var(--red)">Y(s)</div><div class="bo-label">Saida</div><div class="bo-desc">Sinal de saida</div></div>
</div>
<div class="cat-label">Blocos de Transferencia</div>
<div class="block-grid">
<div class="block-option" onclick="pickBlock('tf')"><div class="bo-icon" style="color:var(--blu)">G(s)</div><div class="bo-label">Planta</div><div class="bo-desc">Funcao de transferencia</div></div>
<div class="block-option" onclick="pickBlock('gain')"><div class="bo-icon" style="color:var(--pur)">K</div><div class="bo-label">Ganho</div><div class="bo-desc">Ganho constante</div></div>
<div class="block-option" onclick="pickBlock('pid')"><div class="bo-icon" style="color:var(--pur)">PID</div><div class="bo-label">Controlador PID</div><div class="bo-desc">Kp + Ki/s + Kd*s</div></div>
<div class="block-option" onclick="pickBlock('int')"><div class="bo-icon" style="color:var(--yel)">1/s</div><div class="bo-label">Integrador</div><div class="bo-desc">Integracao pura</div></div>
<div class="block-option" onclick="pickBlock('der')"><div class="bo-icon" style="color:#aaa">s</div><div class="bo-label">Derivador</div><div class="bo-desc">Derivacao pura</div></div>
<div class="block-option" onclick="pickBlock('actuator')"><div class="bo-icon" style="color:var(--blu)">A(s)</div><div class="bo-label">Atuador</div><div class="bo-desc">Dinamica do atuador</div></div>
</div>
<div class="cat-label">Realimentacao</div>
<div class="block-grid">
<div class="block-option" onclick="pickBlock('sensor')"><div class="bo-icon" style="color:var(--pnk)">H(s)</div><div class="bo-label">Sensor</div><div class="bo-desc">Malha de realimentacao</div></div>
<div class="block-option" onclick="pickBlock('sum')"><div class="bo-icon" style="color:var(--grn)">&Sigma;</div><div class="bo-label">Somador</div><div class="bo-desc">Soma/subtracao de sinais</div></div>
<div class="block-option" onclick="pickBlock('branch')"><div class="bo-icon" style="color:var(--blu)">&bull;</div><div class="bo-label">Ramificacao</div><div class="bo-desc">Divide sinal em dois</div></div>
</div>
</div>
<!-- Config panel after picking -->
<div class="cfg-panel" id="cfgPanel">
<h4 id="cfgTitle">Configurar bloco</h4>
<div id="cfgFields"></div>
<button class="cfg-btn" onclick="confirmAdd()">Adicionar ao Diagrama</button>
<button class="cfg-btn" style="background:var(--sf2);margin-top:4px;border:1px solid var(--bd)" onclick="backToGrid()">Voltar</button>
</div>
</div></div></div>

<div class="toolbar" id="diag-toolbar">
<button class="tb tb-add" onclick="openModal()">+ Adicionar Bloco</button>
<div class="sep"></div>
<span class="lbl">Rapido:</span>
<button class="tb" style="background:#162038;border-color:#2d558a;color:var(--blu)" data-add="tf">G(s)</button>
<button class="tb" style="background:#201638;border-color:#5a2d8a;color:var(--pur)" data-add="gain">K</button>
<button class="tb" data-add="sum">&Sigma;</button>
<button class="tb" style="background:#381628;border-color:#8a2d5a;color:var(--pnk)" data-add="sensor">H(s)</button>
<div class="sep"></div>
<button class="tb" style="background:#3a1520;border-color:#8a3050;color:#ff8fa3" id="btnDel">Del</button>
<button class="tb" style="background:#3a1520;border-color:#8a3050;color:#ff8fa3" id="btnClear">Limpar</button>
<button class="tb" id="btnAuto">Auto</button>
</div>
<div class="workspace" id="diag-workspace">
<div class="canvas-wrap" id="cw"><div class="canvas" id="cv"></div>
<svg class="wires" id="wires"><defs>
<marker id="ar" markerWidth="8" markerHeight="8" refX="7" refY="3" orient="auto"><polygon points="0 0,8 3,0 6" fill="#5b6be0"/></marker>
<marker id="ar2" markerWidth="8" markerHeight="8" refX="7" refY="3" orient="auto"><polygon points="0 0,8 3,0 6" fill="#e8a035"/></marker>
</defs></svg></div>
<div class="panel" id="panel">
<div class="panel-section"><h4>Sistema</h4>
<div style="display:grid;grid-template-columns:1fr auto;gap:4px 8px;font-size:12px">
<span style="color:var(--txm)">Blocos:</span><span id="pB" style="font-weight:600">0</span>
<span style="color:var(--txm)">Conexoes:</span><span id="pE" style="font-weight:600">0</span>
</div></div>
<div class="panel-section"><h4>Parametros</h4><div id="pA"><span class="hint">Selecione um bloco.</span></div></div>
<div class="panel-section"><h4>Dicas</h4><div class="hint">
<b>Conectar:</b> porta <span style="color:var(--blu)">azul</span> &rarr; porta <span style="color:var(--grn)">verde</span><br>
<b>Calcular:</b> botao verde abaixo<br><b>Mobile:</b> toque e arraste!
</div></div></div></div>

<div class="mode-bar">
<button class="mode-btn active" id="modeDiag" onclick="setMode('diag')">&#9638; Diagrama de Blocos</button>
<button class="mode-btn" id="modeMan" onclick="setMode('manual')">&#9998; Entrada Manual</button>
</div>

<div class="manual-sec" id="manualSec">
<div class="man-tabs">
<button class="man-tab active" id="subDirect" onclick="setSubMode('direct')">T(s) Direta</button>
<button class="man-tab" id="subClosed" onclick="setSubMode('closed')">Malha Fechada G/(1+GH)</button>
<button class="man-tab" id="subOpen" onclick="setSubMode('open')">Malha Aberta G*H</button>
<button class="man-tab" id="subSS" onclick="setSubMode('ss')">Espaco de Estados</button>
</div>
<div id="manDirect">
<h4>Funcao de Transferencia T(s) = Num / Den</h4>
<div class="man-row">
<div class="pg"><label>Numerador</label><input id="manNum" value="1" placeholder="ex: s+1"></div>
<div class="pg"><label>Denominador</label><input id="manDen" value="s^2+2s+1" placeholder="ex: s^2+2s+1"></div>
</div>
<div class="man-hint">Formato: <code>s^2+3s+1</code> ou <code>2s^3+s+5</code>. Use <code>^</code> para potencias.</div>
</div>
<div id="manClosed" style="display:none">
<h4>Malha Fechada: T(s) = G(s) / (1 + G(s)&middot;H(s))</h4>
<div class="man-row">
<div class="pg"><label>G(s) Numerador</label><input id="manGN" value="10" placeholder="ex: 10"></div>
<div class="pg"><label>G(s) Denominador</label><input id="manGD" value="s^2+3s+1" placeholder="ex: s^2+3s+1"></div>
</div>
<div class="man-row">
<div class="pg"><label>H(s) Numerador</label><input id="manHN" value="1" placeholder="ex: 1"></div>
<div class="pg"><label>H(s) Denominador</label><input id="manHD" value="1" placeholder="ex: s+1"></div>
</div>
<div class="man-hint">Calcula <code>T(s) = G/(1+GH)</code> com realimentacao unitaria quando H=1.</div>
</div>
<div id="manOpen" style="display:none">
<h4>Malha Aberta: L(s) = G(s) &middot; H(s)</h4>
<div class="man-row">
<div class="pg"><label>G(s) Numerador</label><input id="manOGN" value="10" placeholder="ex: 10"></div>
<div class="pg"><label>G(s) Denominador</label><input id="manOGD" value="s^2+3s+1" placeholder="ex: s^2+3s+1"></div>
</div>
<div class="man-row">
<div class="pg"><label>H(s) Numerador</label><input id="manOHN" value="1" placeholder="ex: 1"></div>
<div class="pg"><label>H(s) Denominador</label><input id="manOHD" value="1" placeholder="ex: 1"></div>
</div>
<div class="man-hint">Analisa a funcao de transferencia de malha aberta <code>L(s) = G*H</code>.</div>
</div>
<div id="manSS" style="display:none">
<h4>Espaco de Estados: dx/dt = Ax + Bu, y = Cx + Du</h4>
<div class="man-row">
<div class="pg"><label>Matriz A (nxn)</label><input id="manA" value="0 1; -2 -3" placeholder="0 1; -2 -3"></div>
<div class="pg"><label>Matriz B (nx1)</label><input id="manB" value="0; 1" placeholder="0; 1"></div>
</div>
<div class="man-row">
<div class="pg"><label>Matriz C (1xn)</label><input id="manC" value="1 0" placeholder="1 0"></div>
<div class="pg"><label>Matriz D (1x1)</label><input id="manD" value="0" placeholder="0"></div>
</div>
<div class="man-hint">Use <code>;</code> para separar linhas. Ex: <code>0 1; -2 -3</code> = matriz 2x2. Converte para T(s) = C(sI-A)<sup>-1</sup>B + D</div>
</div>
</div>

<div class="calc-bar"><button id="btnCalcMain" onclick="onCalc()">&#9654; CALCULAR DIAGRAMA</button></div>

<div class="results" id="res"><div class="results-hdr"><h3>Resultados</h3>
<button onclick="document.getElementById('res').classList.remove('vis')" style="background:none;border:1px solid var(--bd);color:var(--txm);border-radius:6px;padding:4px 10px;cursor:pointer;font-size:12px">Fechar</button>
</div><div class="rbody" id="rb"></div></div></div>

<script>
/* ===== MODAL ===== */
var pendingType=null;
function openModal(){document.getElementById("addModal").classList.add("vis");document.getElementById("modalGrid").style.display="block";document.getElementById("cfgPanel").classList.remove("vis");pendingType=null}
function closeModal(){document.getElementById("addModal").classList.remove("vis")}
function backToGrid(){document.getElementById("modalGrid").style.display="block";document.getElementById("cfgPanel").classList.remove("vis")}
function pickBlock(t){
  var simple=["input","output","int","der","branch"];
  if(simple.indexOf(t)>=0){addB(t);closeModal();return}
  pendingType=t;
  document.getElementById("modalGrid").style.display="none";
  var cp=document.getElementById("cfgPanel");cp.classList.add("vis");
  var cf=document.getElementById("cfgFields");var h="";
  var titles={tf:"Planta G(s)",gain:"Ganho K",pid:"Controlador PID",sensor:"Sensor H(s)",sum:"Somador",actuator:"Atuador A(s)"};
  document.getElementById("cfgTitle").textContent="Configurar: "+(titles[t]||t);
  if(t==="tf"||t==="sensor"||t==="actuator"){
    h+='<div class="cfg-row"><div><label>Numerador</label><input id="cfgNum" value="1" placeholder="ex: s+1"></div>';
    h+='<div><label>Denominador</label><input id="cfgDen" value="s+1" placeholder="ex: s^2+2s+1"></div></div>'}
  else if(t==="gain"){h+='<div class="cfg-row"><div><label>Ganho K</label><input id="cfgK" value="1" placeholder="ex: 10"></div><div></div></div>'}
  else if(t==="pid"){
    h+='<div class="cfg-row"><div><label>Kp</label><input id="cfgKp" value="1"></div><div><label>Ki</label><input id="cfgKi" value="0"></div></div>';
    h+='<div class="cfg-row"><div><label>Kd</label><input id="cfgKd" value="0"></div><div></div></div>'}
  else if(t==="sum"){h+='<div class="cfg-row"><div><label>Sinais (+ ou -)</label><input id="cfgSigns" value="+ -" placeholder="+ - +"></div><div></div></div>'}
  cf.innerHTML=h}
function confirmAdd(){
  if(!pendingType)return;
  var t=pendingType,p={};
  if(t==="tf"||t==="sensor"||t==="actuator"){p.num=(document.getElementById("cfgNum")||{}).value||"1";p.den=(document.getElementById("cfgDen")||{}).value||"1"}
  else if(t==="gain"){p.k=(document.getElementById("cfgK")||{}).value||"1"}
  else if(t==="pid"){p.kp=(document.getElementById("cfgKp")||{}).value||"1";p.ki=(document.getElementById("cfgKi")||{}).value||"0";p.kd=(document.getElementById("cfgKd")||{}).value||"0"}
  else if(t==="sum"){p.signs=(document.getElementById("cfgSigns")||{}).value||"+ -"}
  addBP(t,p);closeModal()}

/* ===== POLY MATH ===== */
function pTrim(p){var a=p.slice();while(a.length>1&&Math.abs(a[a.length-1])<1e-14)a.pop();return a.length?a:[0]}
function pAdd(a,b){var r=[];for(var i=0;i<Math.max(a.length,b.length);i++)r.push((a[i]||0)+(b[i]||0));return pTrim(r)}
function pSub(a,b){var r=[];for(var i=0;i<Math.max(a.length,b.length);i++)r.push((a[i]||0)-(b[i]||0));return pTrim(r)}
function pMul(a,b){var r=[];for(var i=0;i<a.length+b.length-1;i++)r.push(0);for(var i=0;i<a.length;i++)for(var j=0;j<b.length;j++)r[i+j]+=a[i]*b[j];return pTrim(r)}
function pScl(a,k){return a.map(function(c){return c*k})}
function pEv(p,x){var r=0;for(var i=p.length-1;i>=0;i--)r=r*x+p[i];return r}
function cMul(a,b){return{r:a.r*b.r-a.i*b.i,i:a.r*b.i+a.i*b.r}}
function cDiv(a,b){var d=b.r*b.r+b.i*b.i;return d<1e-30?{r:0,i:0}:{r:(a.r*b.r+a.i*b.i)/d,i:(a.i*b.r-a.r*b.i)/d}}
function cAbs(a){return Math.sqrt(a.r*a.r+a.i*a.i)}
function cEvP(p,z){var r={r:0,i:0};for(var i=p.length-1;i>=0;i--){r=cMul(r,z);r.r+=p[i]}return r}

/* ===== PF (poly fraction) ===== */
function pfC(c){return{n:[c],d:[1]}}
function pfZ(a){var t=pTrim(a.n);return t.length===1&&Math.abs(t[0])<1e-14}
function pfAdd(a,b){return{n:pAdd(pMul(a.n,b.d),pMul(b.n,a.d)),d:pMul(a.d,b.d)}}
function pfSub(a,b){return{n:pSub(pMul(a.n,b.d),pMul(b.n,a.d)),d:pMul(a.d,b.d)}}
function pfMul(a,b){return{n:pMul(a.n,b.n),d:pMul(a.d,b.d)}}
function pfDiv(a,b){return{n:pMul(a.n,b.d),d:pMul(a.d,b.n)}}

/* ===== POLY SIMPLIFY ===== */
function polyRem(a,b){a=a.slice();b=pTrim(b);if(b.length>a.length)return pTrim(a);for(var i=a.length-1;i>=b.length-1;i--){var c=a[i]/b[b.length-1];for(var j=0;j<b.length;j++)a[j+i-b.length+1]-=c*b[j]}var r=a.slice(0,b.length-1).map(function(v){return Math.abs(v)<1e-8?0:v});return pTrim(r)}
function polyGCD(a,b){a=pTrim(a);b=pTrim(b);while(b.length>1||(b.length===1&&Math.abs(b[0])>1e-8)){var r=polyRem(a,b);a=b;b=r}var lc=a[a.length-1];if(Math.abs(lc)>1e-14)a=a.map(function(c){return c/lc});return pTrim(a)}
function polyDivQ(a,b){a=a.slice();b=pTrim(b);if(b.length===1)return pTrim(a.map(function(c){return c/b[0]}));if(b.length>a.length)return[0];var q=new Array(a.length-b.length+1);for(var i=a.length-1;i>=b.length-1;i--){var c=a[i]/b[b.length-1];q[i-b.length+1]=c;for(var j=0;j<b.length;j++)a[j+i-b.length+1]-=c*b[j]}return pTrim(q)}
function pfReduce(pf){var n=pTrim(pf.n),d=pTrim(pf.d);var g=polyGCD(n,d);if(g.length<=1)return{n:n,d:d};return{n:pTrim(polyDivQ(n,g)),d:pTrim(polyDivQ(d,g))}}

/* ===== PARSER ===== */
function parseP(s){
  s=(s||"").replace(/\s+/g,"");if(!s)return[0];
  if(!/s/i.test(s)){var v=parseFloat(s);return[isNaN(v)?0:v]}
  var n="";for(var i=0;i<s.length;i++){if(s[i]==="-"&&i>0&&s[i-1]!=="^"&&s[i-1]!=="+"&&s[i-1]!=="-"&&s[i-1]!=="*")n+="+";n+=s[i]}
  var ts=n.split("+").filter(function(t){return t}),co={};
  ts.forEach(function(t){t=t.replace(/\*/g,"");var m;
    if(m=t.match(/^([+-]?\d*\.?\d*)s\^(\d+)$/)){var c=m[1]===""||m[1]==="+"?1:m[1]==="-"?-1:parseFloat(m[1]);co[parseInt(m[2])]=(co[parseInt(m[2])]||0)+c}
    else if(m=t.match(/^([+-]?\d*\.?\d*)s$/)){var c=m[1]===""||m[1]==="+"?1:m[1]==="-"?-1:parseFloat(m[1]);co[1]=(co[1]||0)+c}
    else{var c=parseFloat(t);if(!isNaN(c))co[0]=(co[0]||0)+c}});
  var ks=Object.keys(co).map(Number);if(!ks.length)return[0];var mx=Math.max.apply(null,ks),r=[];
  for(var i=0;i<=mx;i++)r.push(co[i]||0);return pTrim(r)}

/* ===== SS PARSER (State Space) ===== */
function parseMat(s){
  s=(s||"").trim();
  var rows=s.split(";");
  return rows.map(function(r){return r.trim().split(/[\s,]+/).map(Number)})}
function ssToTF(As,Bs,Cs,Ds){
  /* For SISO: T(s) = C(sI-A)^-1 B + D computed symbolically via det/adj */
  var A=parseMat(As),B=parseMat(Bs),C=parseMat(Cs),D=parseMat(Ds);
  var n=A.length;
  if(n===0)return{n:[D[0]?D[0][0]:0],d:[1]};
  /* Build characteristic polynomial det(sI - A) using Fadeev-LeVerrier */
  /* For small n (1-4), this is efficient enough */
  var M=[];for(var i=0;i<n;i++){M.push([]);for(var j=0;j<n;j++)M[i].push(i===j?[A[i][j]*-1,1]:[-A[i][j]])} /* sI - A as poly entries */
  /* For n<=3, compute directly; for larger, use approximation */
  if(n===1){
    var a00=A[0][0],b0=B[0][0],c0=C[0][0],d0=D[0]?D[0][0]:0;
    return{n:pAdd([d0*(-a00)+c0*b0],[d0]),d:[-a00,1]}}
  if(n===2){
    var tr=A[0][0]+A[1][1],dt=A[0][0]*A[1][1]-A[0][1]*A[1][0];
    var den=[dt,-tr,1]; /* s^2 - tr*s + det */
    /* adj(sI-A) * B then C * that */
    var adj00=[-A[1][1],1],adj01=[A[0][1]],adj10=[A[1][0]],adj11=[-A[0][0],1];
    var v0=pAdd(pScl(adj00,B[0][0]),pScl(adj01,B[1][0]));
    var v1=pAdd(pScl(adj10,B[0][0]),pScl(adj11,B[1][0]));
    var num=pAdd(pScl(v0,C[0][0]),pScl(v1,C[0].length>1?C[0][1]:0));
    var d0v=D[0]?D[0][0]:0;
    if(Math.abs(d0v)>1e-14)num=pAdd(num,pMul([d0v],den));
    return pfReduce({n:num,d:den})}
  /* General case: numerical via companion form */
  /* Use Fadeev-LeVerrier for char poly, then numerical adj */
  var cp=[1];var Mk=[];for(var i=0;i<n;i++){Mk.push([]);for(var j=0;j<n;j++)Mk[i].push(0)}
  for(var k=1;k<=n;k++){
    var prev=Mk;var AM=[];
    for(var i=0;i<n;i++){AM.push([]);for(var j=0;j<n;j++){var s2=0;for(var l=0;l<n;l++)s2+=A[i][l]*prev[l][j];AM[i].push(s2)}}
    if(k===1)for(var i=0;i<n;i++)for(var j=0;j<n;j++)AM[i][j]+=(i===j?1:0);
    var tk=0;for(var i=0;i<n;i++)tk+=AM[i][i];
    var ck=-tk/k;cp.unshift(ck);
    Mk=[];for(var i=0;i<n;i++){Mk.push([]);for(var j=0;j<n;j++)Mk[i].push(AM[i][j]+(i===j?ck:0))}}
  /* cp is den coeffs [c0,c1,...,1] ascending */
  var den=cp;
  /* num: evaluate C*adj(sI-A)*B numerically at n+1 points, fit poly */
  var pts=[];for(var k=0;k<=n;k++){
    var sv=k*10+1;
    var sIA=[];for(var i=0;i<n;i++){sIA.push([]);for(var j=0;j<n;j++)sIA[i].push((i===j?sv:0)-A[i][j])}
    /* invert sIA by Gauss-Jordan */
    var aug=[];for(var i=0;i<n;i++){aug.push([]);for(var j=0;j<2*n;j++)aug[i].push(j<n?sIA[i][j]:(i===j-n?1:0))}
    for(var c2=0;c2<n;c2++){var mx2=c2;for(var r=c2+1;r<n;r++)if(Math.abs(aug[r][c2])>Math.abs(aug[mx2][c2]))mx2=r;
      var tmp=aug[c2];aug[c2]=aug[mx2];aug[mx2]=tmp;var piv=aug[c2][c2];if(Math.abs(piv)<1e-30)continue;
      for(var j=0;j<2*n;j++)aug[c2][j]/=piv;
      for(var r=0;r<n;r++){if(r===c2)continue;var f=aug[r][c2];for(var j=0;j<2*n;j++)aug[r][j]-=f*aug[c2][j]}}
    var inv=[];for(var i=0;i<n;i++){inv.push([]);for(var j=0;j<n;j++)inv[i].push(aug[i][j+n])}
    /* C * inv * B */
    var CiB=0;for(var i=0;i<(C[0]||[]).length;i++)for(var j=0;j<(B[0]||[]).length;j++){var v3=0;for(var l=0;l<n;l++)v3+=inv[i][l]*B[l][j];CiB+=C[0][i]*v3}
    var d0v=D[0]?D[0][0]:0;
    pts.push({s:sv,v:CiB+d0v*pEv(den,sv)})}
  /* Lagrange interpolation for numerator poly (degree <= n) */
  var numC=[];for(var i=0;i<=n;i++){
    var li=1,yi=pts[i].v;for(var j=0;j<=n;j++){if(j===i)continue;li*=(pts[i].s-pts[j].s)}
    numC.push(yi/li)}
  /* Build num poly from pts using Newton form... simpler: Vandermonde */
  var V=[];for(var i=0;i<=n;i++){V.push([]);for(var j=0;j<=n;j++)V[i].push(Math.pow(pts[i].s,j))}
  var rhs=pts.map(function(p){return p.v});
  /* Solve Vandermonde system */
  for(var c2=0;c2<=n;c2++){var mx2=c2;for(var r=c2+1;r<=n;r++)if(Math.abs(V[r][c2])>Math.abs(V[mx2][c2]))mx2=r;
    var tmp=V[c2];V[c2]=V[mx2];V[mx2]=tmp;var tmp2=rhs[c2];rhs[c2]=rhs[mx2];rhs[mx2]=tmp2;
    var piv=V[c2][c2];if(Math.abs(piv)<1e-30)continue;for(var j=c2;j<=n;j++)V[c2][j]/=piv;rhs[c2]/=piv;
    for(var r=0;r<=n;r++){if(r===c2)continue;var f=V[r][c2];for(var j=c2;j<=n;j++)V[r][j]-=f*V[c2][j];rhs[r]-=f*rhs[c2]}}
  var num=rhs.map(function(v){return Math.abs(v)<1e-10?0:v});
  return pfReduce({n:pTrim(num),d:pTrim(den)})}

/* ===== FORMAT ===== */
function fN(n){if(Math.abs(n-Math.round(n))<1e-8)return Math.round(n).toString();return n.toFixed(4).replace(/0+$/,"").replace(/\.$/,"")}
function fP(c){c=pTrim(c);var ts=[];for(var i=c.length-1;i>=0;i--){var v=c[i];if(Math.abs(v)<1e-10)continue;var t;
  if(i===0)t=fN(v);else if(i===1){t=Math.abs(v-1)<1e-10?"s":Math.abs(v+1)<1e-10?"-s":fN(v)+"s"}
  else{t=Math.abs(v-1)<1e-10?"s^"+i:Math.abs(v+1)<1e-10?"-s^"+i:fN(v)+"s^"+i}ts.push(t)}
  if(!ts.length)return"0";var s=ts[0];for(var i=1;i<ts.length;i++)s+=ts[i][0]==="-"?" - "+ts[i].slice(1):" + "+ts[i];return s}

/* ===== BLOCK TF ===== */
function bTF(nd){var p=nd.params||{},t=nd.type;
  if(t==="tf"||t==="sensor"||t==="actuator")return{n:parseP(p.num||"1"),d:parseP(p.den||"1")};
  if(t==="gain")return pfC(parseFloat(p.k)||1);
  if(t==="int")return{n:[1],d:[0,1]};if(t==="der")return{n:[0,1],d:[1]};
  if(t==="pid"){var kp=parseFloat(p.kp)||0,ki=parseFloat(p.ki)||0,kd=parseFloat(p.kd)||0;
    if(ki===0&&kd===0)return pfC(kp||1);if(ki===0)return{n:[kp,kd],d:[1]};return{n:[ki,kp,kd],d:[0,1]}}
  return pfC(1)}

/* ===== SOLVER (Gaussian elimination on signal-flow graph) ===== */
function solve(nodes,edges){
  if(!nodes.length)return{e:"Adicione blocos."};
  var inp=null,out=null;nodes.forEach(function(n){if(n.type==="input")inp=n;if(n.type==="output")out=n});
  if(!inp)return{e:"Adicione Entrada R(s)."};if(!out)return{e:"Adicione Saida Y(s)."};if(!edges.length)return{e:"Conecte os blocos."};
  var N=nodes.length,ix={};nodes.forEach(function(n,i){ix[n.id]=i});
  /* Build system of equations: for each node i, X_i = sum(contributions) */
  var A=[],b=[];for(var i=0;i<N;i++){A.push([]);for(var j=0;j<N;j++)A[i].push(pfC(0));b.push(pfC(0))}
  for(var i=0;i<N;i++){var nd=nodes[i];A[i][i]=pfC(1);
    if(nd.type==="input"){b[i]=pfC(1);continue}
    var inc=edges.filter(function(e){return e.dst===nd.id});
    if(nd.type==="sum"){
      /* Summing junction: output = sum of signed inputs */
      var sg=((nd.params||{}).signs||"+ -").trim().split(/\s+/);
      inc.forEach(function(e){var si=ix[e.src];if(si===undefined)return;
        var pi=parseInt((e.dstPort||"in0").replace("in",""))||0;
        var sign=sg[pi]==="-"?-1:1;
        A[i][si]=pfSub(A[i][si],pfC(sign))})
    } else {
      /* Regular block: output = TF * input */
      var tf=bTF(nd);
      inc.forEach(function(e){var si=ix[e.src];if(si===undefined)return;
        A[i][si]=pfSub(A[i][si],tf)})
    }
  }
  /* Gaussian elimination with partial pivoting */
  for(var c=0;c<N;c++){var pv=-1;for(var r=c;r<N;r++)if(!pfZ(A[r][c])){pv=r;break}
    if(pv<0)return{e:"Sistema singular - verifique as conexoes."};
    if(pv!==c){var t=A[c];A[c]=A[pv];A[pv]=t;var t2=b[c];b[c]=b[pv];b[pv]=t2}
    for(var r=c+1;r<N;r++){if(pfZ(A[r][c]))continue;var f=pfDiv(A[r][c],A[c][c]);
      for(var j=c;j<N;j++)A[r][j]=pfSub(A[r][j],pfMul(f,A[c][j]));b[r]=pfSub(b[r],pfMul(f,b[c]))}}
  /* Back substitution */
  var x=[];for(var i=0;i<N;i++)x.push(pfC(0));
  for(var i=N-1;i>=0;i--){var s=b[i];for(var j=i+1;j<N;j++)s=pfSub(s,pfMul(A[i][j],x[j]));x[i]=pfDiv(s,A[i][i])}
  var oi=ix[out.id],tf={n:pTrim(x[oi].n),d:pTrim(x[oi].d)};
  /* Normalize */
  if(tf.d[tf.d.length-1]<0){tf.n=pScl(tf.n,-1);tf.d=pScl(tf.d,-1)}
  var lc=tf.d[tf.d.length-1];if(Math.abs(lc)>1e-14&&Math.abs(lc-1)>1e-10){tf.n=pScl(tf.n,1/lc);tf.d=pScl(tf.d,1/lc)}
  tf=pfReduce(tf);return{tf:tf}}

/* ===== ROOTS ===== */
function roots(poly){poly=pTrim(poly);var n=poly.length-1;if(n<=0)return[];
  var lc=poly[n],p=poly.map(function(c){return c/lc});
  if(n===1)return[{r:-p[0],i:0}];
  if(n===2){var bb=p[1],cc=p[0],d=bb*bb-4*cc;
    if(d>=0)return[{r:(-bb+Math.sqrt(d))/2,i:0},{r:(-bb-Math.sqrt(d))/2,i:0}];
    return[{r:-bb/2,i:Math.sqrt(-d)/2},{r:-bb/2,i:-Math.sqrt(-d)/2}]}
  var rs=[];for(var i=0;i<n;i++){var a=2*Math.PI*i/n+.4;rs.push({r:(1+.5*i/n)*Math.cos(a),i:(1+.5*i/n)*Math.sin(a)})}
  for(var it=0;it<1500;it++){var mx=0;for(var i=0;i<n;i++){var v=cEvP(p,rs[i]),pr={r:1,i:0};
    for(var j=0;j<n;j++)if(j!==i)pr=cMul(pr,{r:rs[i].r-rs[j].r,i:rs[i].i-rs[j].i});
    if(cAbs(pr)<1e-30)continue;var dl=cDiv(v,pr);rs[i].r-=dl.r;rs[i].i-=dl.i;mx=Math.max(mx,cAbs(dl))}if(mx<1e-12)break}
  rs.forEach(function(r){if(Math.abs(r.i)<1e-8)r.i=0});return rs}

/* ===== STEP/FORCED RESPONSE (Stehfest) ===== */
function fact(n){var r=1;for(var i=2;i<=n;i++)r*=i;return r}
var SN=14,SV=[];(function(){for(var i=1;i<=SN;i++){var s=0,k0=Math.floor((i+1)/2),k1=Math.min(i,SN/2);
  for(var k=k0;k<=k1;k++)s+=Math.pow(k,SN/2)*fact(2*k)/(fact(SN/2-k)*fact(k)*fact(k-1)*fact(i-k)*fact(2*k-i));
  SV.push(Math.pow(-1,SN/2+i)*s)}})();
function stepResp(tf,tMax,nP){var ln2=Math.LN2,ts=[],ys=[];
  for(var i=0;i<nP;i++){var t=(i+1)*tMax/nP;ts.push(t);var s=0;
    for(var j=0;j<SN;j++){var sv=(j+1)*ln2/t,nv=pEv(tf.n,sv),dv=pEv(tf.d,sv)*sv;if(Math.abs(dv)<1e-30)continue;s+=SV[j]*nv/dv}
    ys.push(s*ln2/t)}return{t:ts,y:ys}}
function forceResp(tf,sig,tMax,nP){var ln2=Math.LN2,ts=[],ys=[],us=[];var om=2*Math.PI;
  for(var i=0;i<nP;i++){var t=(i+1)*tMax/nP;ts.push(t);var s=0;
    for(var j=0;j<SN;j++){var sv=(j+1)*ln2/t,nv=pEv(tf.n,sv),dv=pEv(tf.d,sv);
      if(Math.abs(dv)<1e-30)continue;
      var uv;if(sig==='rampa')uv=1/(sv*sv);else if(sig==='impulso')uv=1;
      else if(sig==='parabolica')uv=2/(sv*sv*sv);else if(sig==='senoidal')uv=om/(sv*sv+om*om);
      else uv=1/sv;
      s+=SV[j]*nv*uv/dv}
    ys.push(s*ln2/t);
    if(sig==='rampa')us.push(t);else if(sig==='senoidal')us.push(Math.sin(om*t));
    else if(sig==='impulso')us.push(i===0?nP/tMax:0);else if(sig==='parabolica')us.push(t*t);
    else us.push(1)}
  return{t:ts,y:ys,u:us}}

/* ===== NYQUIST ===== */
function nyq(tf,wMin,wMax,nP){var re=[],im=[];var lm=Math.log10(wMin),lx=Math.log10(wMax);
  for(var i=0;i<nP;i++){var w=Math.pow(10,lm+i*(lx-lm)/(nP-1));
    var jw={r:0,i:w},nc=cEvP(tf.n,jw),dc=cEvP(tf.d,jw),T=cDiv(nc,dc);re.push(T.r);im.push(T.i)}
  return{re:re,im:im}}

/* ===== LGR ===== */
function lgr(tf,nK){var kMax=200,branches=[];var nt=pTrim(tf.n),dt=pTrim(tf.d);var np=dt.length-1;
  if(np<=0)return branches;
  for(var i=0;i<np;i++)branches.push({re:[],im:[]});var prev=null;
  for(var ki=0;ki<nK;ki++){var k=ki*kMax/(nK-1);
    var cp=pAdd(dt,pScl(nt,k));var rs=roots(cp);
    if(prev&&prev.length===rs.length){var used=[],matched=[];
      for(var i=0;i<prev.length;i++){var best=-1,bd=Infinity;
        for(var j=0;j<rs.length;j++){if(used.indexOf(j)>=0)continue;
          var d=Math.sqrt(Math.pow(rs[j].r-prev[i].r,2)+Math.pow(rs[j].i-prev[i].i,2));
          if(d<bd){bd=d;best=j}}
        used.push(best);matched.push(rs[best])}rs=matched}
    for(var i=0;i<Math.min(rs.length,np);i++){branches[i].re.push(rs[i].r);branches[i].im.push(rs[i].i)}
    prev=rs.slice(0,np)}
  return branches}

function chartLGR(id,branches,tf){var c=document.getElementById(id);if(!c)return;
  var w=c.width=c.parentElement.clientWidth||500,h=c.height=300,ctx=c.getContext("2d");
  var mg={l:55,r:15,t:15,b:35},pw=w-mg.l-mg.r,ph=h-mg.t-mg.b;
  var allX=[],allY=[];
  branches.forEach(function(br){allX=allX.concat(br.re);allY=allY.concat(br.im)});
  if(!allX.length)return;
  var x0=Math.min.apply(null,allX),x1=Math.max.apply(null,allX);
  var y0=Math.min.apply(null,allY),y1=Math.max.apply(null,allY);
  var xp=(x1-x0)*.15||1,yp=(y1-y0)*.15||1;x0-=xp;x1+=xp;y0-=yp;y1+=yp;
  function mX(x){return mg.l+(x-x0)/(x1-x0)*pw}
  function mY(y){return mg.t+ph-(y-y0)/(y1-y0)*ph}
  ctx.fillStyle="#0e1117";ctx.fillRect(0,0,w,h);
  ctx.strokeStyle="#252840";ctx.lineWidth=.5;
  for(var i=0;i<=5;i++){var gy=mg.t+ph*i/5;ctx.beginPath();ctx.moveTo(mg.l,gy);ctx.lineTo(w-mg.r,gy);ctx.stroke();
    ctx.fillStyle="#8890b0";ctx.font="10px system-ui";ctx.textAlign="right";ctx.fillText(fN(y1-(y1-y0)*i/5),mg.l-5,gy+4)}
  if(y0<=0&&y1>=0){ctx.strokeStyle="#555";ctx.lineWidth=1;ctx.beginPath();ctx.moveTo(mg.l,mY(0));ctx.lineTo(w-mg.r,mY(0));ctx.stroke()}
  if(x0<=0&&x1>=0){ctx.strokeStyle="#555";ctx.lineWidth=1;ctx.beginPath();ctx.moveTo(mX(0),mg.t);ctx.lineTo(mX(0),mg.t+ph);ctx.stroke()}
  var cols=["#5b6be0","#60a5fa","#a78bfa","#f472b6","#fbbf24"];
  branches.forEach(function(br,bi){ctx.strokeStyle=cols[bi%cols.length];ctx.lineWidth=1.5;ctx.beginPath();var st=false;
    for(var i=0;i<br.re.length;i++){var px=mX(br.re[i]),py=mY(br.im[i]);
      if(!st){ctx.moveTo(px,py);st=true}else ctx.lineTo(px,py)}ctx.stroke()});
  var ps=roots(tf.d);ctx.strokeStyle="#ff4444";ctx.lineWidth=2;
  ps.forEach(function(p){var px=mX(p.r),py=mY(p.i);ctx.beginPath();ctx.moveTo(px-5,py-5);ctx.lineTo(px+5,py+5);ctx.stroke();
    ctx.beginPath();ctx.moveTo(px+5,py-5);ctx.lineTo(px-5,py+5);ctx.stroke()});
  var zs=roots(tf.n);ctx.strokeStyle="#44ff44";ctx.lineWidth=2;
  zs.forEach(function(z){ctx.beginPath();ctx.arc(mX(z.r),mY(z.i),5,0,2*Math.PI);ctx.stroke()});
  ctx.fillStyle="#8890b0";ctx.font="11px system-ui";ctx.textAlign="center";ctx.fillText("Parte Real",mg.l+pw/2,h-5);
  ctx.save();ctx.translate(12,mg.t+ph/2);ctx.rotate(-Math.PI/2);ctx.fillText("Parte Imaginaria",0,0);ctx.restore()}

/* ===== BODE ===== */
function bode(tf,wMin,wMax,nP){var fs=[],ms=[],ps=[],lm=Math.log10(wMin),lx=Math.log10(wMax);
  for(var i=0;i<nP;i++){var w=Math.pow(10,lm+i*(lx-lm)/(nP-1));fs.push(w);
    var jw={r:0,i:w},nc=cEvP(tf.n,jw),dc=cEvP(tf.d,jw),T=cDiv(nc,dc),mg=cAbs(T);
    ms.push(mg>1e-30?20*Math.log10(mg):-600);ps.push(Math.atan2(T.i,T.r)*180/Math.PI)}
  for(var i=1;i<ps.length;i++){while(ps[i]-ps[i-1]>180)ps[i]-=360;while(ps[i]-ps[i-1]<-180)ps[i]+=360}
  return{w:fs,m:ms,p:ps}}

/* ===== AUTO RANGES ===== */
function autoT(tf){var ps=roots(tf.d);if(!ps.length)return 10;var m=Infinity;
  ps.forEach(function(p){if(Math.abs(p.r)>1e-6)m=Math.min(m,Math.abs(p.r))});return m===Infinity?10:Math.min(100,Math.max(2,7/m))}
function autoW(tf){var ps=roots(tf.d).concat(roots(tf.n)),fs=ps.map(function(p){return Math.sqrt(p.r*p.r+p.i*p.i)}).filter(function(f){return f>1e-6});
  if(!fs.length)return{a:.01,b:1000};return{a:Math.max(.001,Math.min.apply(null,fs)/20),b:Math.min(1e5,Math.max.apply(null,fs)*20)}}

/* ===== PERF ===== */
function perf(t,y){if(!y.length)return{};var l=y.slice(Math.floor(y.length*.9)),yf=l.reduce(function(a,b){return a+b},0)/l.length;
  var ym=Math.max.apply(null,y),os=Math.abs(yf)>1e-6?Math.max(0,(ym-yf)/Math.abs(yf)*100):0;
  var t10=null,t90=null;if(yf>0)for(var i=0;i<y.length;i++){if(t10===null&&y[i]>=yf*.1)t10=t[i];if(t90===null&&y[i]>=yf*.9){t90=t[i];break}}
  var tr=t10!==null&&t90!==null?t90-t10:NaN,ts2=NaN;
  if(Math.abs(yf)>1e-6)for(var i=y.length-1;i>=0;i--)if(Math.abs(y[i]-yf)>.02*Math.abs(yf)){ts2=i<y.length-1?t[i+1]:t[i];break}
  return{"Valor Final":fN(yf),"Sobressinal":fN(os)+"%","T. Subida":isNaN(tr)?"N/A":fN(tr)+"s","T. Acomod.":isNaN(ts2)?"N/A":fN(ts2)+"s","Pico":fN(ym)}}

/* ===== CANVAS CHARTS ===== */
function chart(id,xD,yD,xL,yL,col,logX){var c=document.getElementById(id);if(!c)return;
  var w=c.width=c.parentElement.clientWidth||500,h=c.height=280,ctx=c.getContext("2d");
  var mg={l:55,r:15,t:15,b:35},pw=w-mg.l-mg.r,ph=h-mg.t-mg.b;
  var vY=yD.filter(function(v){return isFinite(v)});if(!vY.length)return;
  var xA=logX?xD.map(function(x){return Math.log10(x)}):xD;
  var x0=Math.min.apply(null,xA),x1=Math.max.apply(null,xA),y0=Math.min.apply(null,vY),y1=Math.max.apply(null,vY);
  var yp=(y1-y0)*.1||1;y0-=yp;y1+=yp;
  function mX(x){return mg.l+((logX?Math.log10(x):x)-x0)/(x1-x0)*pw}
  function mY(y){return mg.t+ph-(y-y0)/(y1-y0)*ph}
  ctx.fillStyle="#0e1117";ctx.fillRect(0,0,w,h);
  ctx.strokeStyle="#252840";ctx.lineWidth=.5;
  for(var i=0;i<=5;i++){var gy=mg.t+ph*i/5;ctx.beginPath();ctx.moveTo(mg.l,gy);ctx.lineTo(w-mg.r,gy);ctx.stroke();
    ctx.fillStyle="#8890b0";ctx.font="10px system-ui";ctx.textAlign="right";ctx.fillText(fN(y1-(y1-y0)*i/5),mg.l-5,gy+4)}
  ctx.strokeStyle=col;ctx.lineWidth=2;ctx.beginPath();var st=false;
  for(var i=0;i<xD.length;i++){if(!isFinite(yD[i])){st=false;continue}
    var px=mX(xD[i]),py=Math.max(mg.t,Math.min(mg.t+ph,mY(yD[i])));if(!st){ctx.moveTo(px,py);st=true}else ctx.lineTo(px,py)}ctx.stroke();
  ctx.fillStyle="#8890b0";ctx.font="11px system-ui";ctx.textAlign="center";ctx.fillText(xL,mg.l+pw/2,h-5);
  ctx.save();ctx.translate(12,mg.t+ph/2);ctx.rotate(-Math.PI/2);ctx.fillText(yL,0,0);ctx.restore()}

function chart2(id,xD,y1,y2,xL,yL){var c=document.getElementById(id);if(!c)return;
  var w=c.width=c.parentElement.clientWidth||500,h=c.height=280,ctx=c.getContext("2d");
  var mg={l:55,r:15,t:15,b:35},pw=w-mg.l-mg.r,ph=h-mg.t-mg.b;
  var all=y1.concat(y2).filter(function(v){return isFinite(v)});if(!all.length)return;
  var x0=Math.min.apply(null,xD),x1=Math.max.apply(null,xD);
  var y0=Math.min.apply(null,all),ym=Math.max.apply(null,all);
  var yp=(ym-y0)*.1||1;y0-=yp;ym+=yp;
  function mX(x){return mg.l+(x-x0)/(x1-x0)*pw}
  function mY(y){return mg.t+ph-(y-y0)/(ym-y0)*ph}
  ctx.fillStyle="#0e1117";ctx.fillRect(0,0,w,h);
  ctx.strokeStyle="#252840";ctx.lineWidth=.5;
  for(var i=0;i<=5;i++){var gy=mg.t+ph*i/5;ctx.beginPath();ctx.moveTo(mg.l,gy);ctx.lineTo(w-mg.r,gy);ctx.stroke();
    ctx.fillStyle="#8890b0";ctx.font="10px system-ui";ctx.textAlign="right";ctx.fillText(fN(ym-(ym-y0)*i/5),mg.l-5,gy+4)}
  ctx.strokeStyle="#60a5fa";ctx.lineWidth=1.5;ctx.setLineDash([6,4]);ctx.beginPath();var st=false;
  for(var i=0;i<xD.length;i++){if(!isFinite(y1[i])){st=false;continue}
    var px=mX(xD[i]),py=Math.max(mg.t,Math.min(mg.t+ph,mY(y1[i])));if(!st){ctx.moveTo(px,py);st=true}else ctx.lineTo(px,py)}ctx.stroke();
  ctx.strokeStyle="#f44";ctx.lineWidth=2;ctx.setLineDash([]);ctx.beginPath();st=false;
  for(var i=0;i<xD.length;i++){if(!isFinite(y2[i])){st=false;continue}
    var px=mX(xD[i]),py=Math.max(mg.t,Math.min(mg.t+ph,mY(y2[i])));if(!st){ctx.moveTo(px,py);st=true}else ctx.lineTo(px,py)}ctx.stroke();
  ctx.fillStyle="#60a5fa";ctx.fillRect(mg.l+10,mg.t+8,20,3);ctx.fillStyle="#8890b0";ctx.font="10px system-ui";ctx.textAlign="left";ctx.fillText("Entrada",mg.l+35,mg.t+12);
  ctx.fillStyle="#f44";ctx.fillRect(mg.l+10,mg.t+22,20,3);ctx.fillText("Saida",mg.l+35,mg.t+26);
  ctx.fillStyle="#8890b0";ctx.font="11px system-ui";ctx.textAlign="center";ctx.fillText(xL,mg.l+pw/2,h-5);
  ctx.save();ctx.translate(12,mg.t+ph/2);ctx.rotate(-Math.PI/2);ctx.fillText(yL,0,0);ctx.restore()}

function chartXY(id,xD,yD,xL,yL){var c=document.getElementById(id);if(!c)return;
  var w=c.width=c.parentElement.clientWidth||500,h=c.height=380,ctx=c.getContext("2d");
  var mg={l:55,r:15,t:15,b:45},pw=w-mg.l-mg.r,ph=h-mg.t-mg.b;
  var vX=xD.filter(function(v){return isFinite(v)}),vY=yD.filter(function(v){return isFinite(v)});
  if(!vX.length||!vY.length)return;
  var sX=vX.slice().sort(function(a,b){return a-b}),sY=vY.slice().sort(function(a,b){return a-b});
  var lo=Math.floor(sX.length*.03),hi=Math.min(sX.length-1,Math.floor(sX.length*.97));
  var loY=Math.floor(sY.length*.03),hiY=Math.min(sY.length-1,Math.floor(sY.length*.97));
  var x0=Math.min(sX[lo],-1.5),x1=Math.max(sX[hi],0.5);
  var y0=sY[loY],y1=sY[hiY];
  var xp=(x1-x0)*.15||1,yp=(y1-y0)*.15||1;x0-=xp;x1+=xp;y0-=yp;y1+=yp;
  function mX(x){return mg.l+(x-x0)/(x1-x0)*pw}
  function mY(y){return mg.t+ph-(y-y0)/(y1-y0)*ph}
  ctx.fillStyle="#0e1117";ctx.fillRect(0,0,w,h);
  ctx.strokeStyle="#252840";ctx.lineWidth=.5;
  for(var i=0;i<=5;i++){var gy=mg.t+ph*i/5;ctx.beginPath();ctx.moveTo(mg.l,gy);ctx.lineTo(w-mg.r,gy);ctx.stroke();
    ctx.fillStyle="#8890b0";ctx.font="10px system-ui";ctx.textAlign="right";ctx.fillText(fN(y1-(y1-y0)*i/5),mg.l-5,gy+4);
    var gx=mg.l+pw*i/5;ctx.beginPath();ctx.moveTo(gx,mg.t);ctx.lineTo(gx,mg.t+ph);ctx.stroke();
    ctx.textAlign="center";ctx.fillText(fN(x0+(x1-x0)*i/5),gx,mg.t+ph+14)}
  if(y0<=0&&y1>=0){var ay=mY(0);ctx.strokeStyle="#555";ctx.lineWidth=1;ctx.beginPath();ctx.moveTo(mg.l,ay);ctx.lineTo(w-mg.r,ay);ctx.stroke()}
  if(x0<=0&&x1>=0){var ax=mX(0);ctx.strokeStyle="#555";ctx.lineWidth=1;ctx.beginPath();ctx.moveTo(ax,mg.t);ctx.lineTo(ax,mg.t+ph);ctx.stroke()}
  ctx.strokeStyle="#5b6be0";ctx.lineWidth=2;ctx.beginPath();var st=false;
  for(var i=0;i<xD.length;i++){if(!isFinite(xD[i])||!isFinite(yD[i])){st=false;continue}
    var px=Math.max(mg.l,Math.min(mg.l+pw,mX(xD[i]))),py=Math.max(mg.t,Math.min(mg.t+ph,mY(yD[i])));
    if(!st){ctx.moveTo(px,py);st=true}else ctx.lineTo(px,py)}ctx.stroke();
  ctx.strokeStyle="#888";ctx.lineWidth=1;ctx.setLineDash([4,4]);ctx.beginPath();st=false;
  for(var i=0;i<xD.length;i++){if(!isFinite(xD[i])||!isFinite(yD[i])){st=false;continue}
    var px=Math.max(mg.l,Math.min(mg.l+pw,mX(xD[i]))),py=Math.max(mg.t,Math.min(mg.t+ph,mY(-yD[i])));
    if(!st){ctx.moveTo(px,py);st=true}else ctx.lineTo(px,py)}ctx.stroke();ctx.setLineDash([]);
  ctx.fillStyle="#ff4444";ctx.beginPath();ctx.arc(mX(-1),mY(0),6,0,2*Math.PI);ctx.fill();
  ctx.fillStyle="#fff";ctx.font="10px system-ui";ctx.textAlign="left";ctx.fillText("-1",mX(-1)+8,mY(0)+4);
  ctx.fillStyle="#8890b0";ctx.font="11px system-ui";ctx.textAlign="center";ctx.fillText(xL,mg.l+pw/2,h-5);
  ctx.save();ctx.translate(12,mg.t+ph/2);ctx.rotate(-Math.PI/2);ctx.fillText(yL,0,0);ctx.restore()}

/* ===== RESULTS ===== */
function fC(c){if(Math.abs(c.i)<1e-8)return fN(c.r);return fN(c.r)+(c.i>=0?" + ":" - ")+fN(Math.abs(c.i))+"j"}
function esc(s){return s.replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;")}

var curSig='degrau';
var curMalha='aberta';
var selAn={tempo:true,desemp:true,pz:true,bm:true,bp:true,nyqst:true,lgr:true};
var _lastTF=null;

function reRender(){if(_lastTF)showRes(_lastTF)}

function showRes(tf){
  _lastTF=tf;
  var rd=document.getElementById("res"),rb=document.getElementById("rb");
  rd.classList.add("vis");
  var ns=fP(tf.n),ds=fP(tf.d);
  var ps=roots(tf.d),zs=roots(tf.n),stb=ps.every(function(p){return p.r<1e-6});
  var tM=autoT(tf);
  var sr=forceResp(tf,curSig,tM,400);
  var pf=perf(sr.t,sr.y);
  var wr=autoW(tf),bd=bode(tf,wr.a,wr.b,400);
  var nqd=nyq(tf,wr.a,wr.b,400);
  var lgrData=lgr(tf,300);
  var sigNomes={degrau:'Degrau',rampa:'Rampa',senoidal:'Senoidal',impulso:'Impulso',parabolica:'Parabolica'};
  var h='';
  var ss='background:var(--sf2);border:1px solid var(--bd);border-radius:6px;color:var(--tx);padding:5px 10px;font-size:12px';
  h+='<div class="rcard"><h4>Configuracoes de Analise</h4>';
  h+='<div style="display:flex;flex-wrap:wrap;align-items:center;gap:12px;margin-bottom:12px">';
  h+='<div style="display:flex;align-items:center;gap:6px"><span style="font-size:12px;color:var(--txm);font-weight:600">Tipo de Malha:</span>';
  h+='<select onchange="curMalha=this.value;reRender()" style="'+ss+';font-weight:600">';
  h+='<option value="aberta"'+(curMalha==='aberta'?' selected':'')+'>Malha Aberta</option>';
  h+='<option value="fechada"'+(curMalha==='fechada'?' selected':'')+'>Malha Fechada</option>';
  h+='</select></div>';
  h+='<div style="display:flex;align-items:center;gap:6px"><span style="font-size:12px;color:var(--txm)">Sinal:</span>';
  h+='<select onchange="curSig=this.value;reRender()" style="'+ss+'">';
  ['degrau','rampa','senoidal','impulso','parabolica'].forEach(function(s){
    h+='<option value="'+s+'"'+(curSig===s?' selected':'')+'>'+sigNomes[s]+'</option>'});
  h+='</select></div></div>';
  var chks;
  if(curMalha==='aberta'){
    chks=[['tempo','Resposta no tempo'],['desemp','Desempenho'],['pz','Diagrama de Polos e Zeros'],['bm','Diagrama De Bode Magnitude'],['bp','Diagrama De Bode Fase'],['nyqst','Nyquist']];
  }else{
    chks=[['tempo','Resposta no tempo'],['desemp','Desempenho'],['pz','Diagrama de Polos e Zeros'],['bm','Diagrama De Bode Magnitude'],['bp','Diagrama De Bode Fase'],['lgr','LGR']];
  }
  h+='<div style="display:flex;flex-wrap:wrap;gap:8px 16px">';
  chks.forEach(function(c){h+='<label style="font-size:12px;cursor:pointer;display:flex;align-items:center;gap:4px"><input type="checkbox" '+(selAn[c[0]]?'checked':'')+' onchange="selAn.'+c[0]+'=this.checked;reRender()"> '+c[1]+'</label>'});
  h+='</div></div>';
  h+='<div class="rcard"><h4>T(s)</h4><div class="tf-disp"><div style="display:inline-block;text-align:center"><div style="padding:0 8px">'+esc(ns)+'</div><div style="border-top:2px solid var(--acc);padding:4px 8px 0">'+esc(ds)+'</div></div></div></div>';
  if(selAn.tempo){
    h+='<div class="rcard"><h4>Resposta no Tempo - '+sigNomes[curSig]+'</h4><div><canvas id="cStep"></canvas></div></div>';}
  if(selAn.desemp){
    h+='<div class="rcard"><h4>Desempenho</h4><div class="mgrid">';
    Object.keys(pf).forEach(function(k){h+='<div class="mbox"><div class="ml">'+k+'</div><div class="mv">'+pf[k]+'</div></div>'});
    h+='</div></div>';}
  if(selAn.pz){
    h+='<div class="rcard"><h4>Diagrama de Polos e Zeros</h4><div style="display:flex;gap:20px;flex-wrap:wrap"><div><b style="color:var(--red)">Polos:</b><div class="pzl">';
    ps.forEach(function(p){h+='<div style="color:var(--red)">'+fC(p)+'</div>'});if(!ps.length)h+="<div>-</div>";
    h+='</div></div><div><b style="color:var(--blu)">Zeros:</b><div class="pzl">';
    zs.forEach(function(z){h+='<div style="color:var(--blu)">'+fC(z)+'</div>'});if(!zs.length)h+="<div>-</div>";
    h+='</div></div></div><div style="margin-top:8px;padding:6px 10px;border-radius:6px;font-weight:700;font-size:13px;'+(stb?'background:#16382a;color:#34d399">ESTAVEL':'background:#3a1520;color:#f87171">INSTAVEL')+'</div></div>';}
  if(selAn.bm){
    h+='<div class="rcard"><h4>Diagrama De Bode - Magnitude</h4><div><canvas id="cBM"></canvas></div></div>';}
  if(selAn.bp){
    h+='<div class="rcard"><h4>Diagrama De Bode - Fase</h4><div><canvas id="cBP"></canvas></div></div>';}
  if(curMalha==='aberta'&&selAn.nyqst){
    h+='<div class="rcard"><h4>Nyquist</h4><div><canvas id="cNyq"></canvas></div>';
    var nps=roots(tf.d),npsd=0;nps.forEach(function(p){if(p.r>1e-6)npsd++});
    h+='<div style="margin-top:8px;font-size:12px"><b>Polos SPD (P):</b> '+npsd+' | <b>Z = P + N:</b> '+(npsd===0?'Estavel':'Instavel')+'</div></div>';}
  if(curMalha==='fechada'&&selAn.lgr){
    h+='<div class="rcard"><h4>Lugar Geometrico das Raizes (LGR)</h4><div><canvas id="cLGR"></canvas></div>';
    h+='<div style="margin-top:8px;font-size:11px;color:var(--txm)">Polos (X vermelho) | Zeros (O verde) | K: 0 a 200</div></div>';}
  rb.innerHTML=h;
  if(selAn.tempo)chart2("cStep",sr.t,sr.u,sr.y,"Tempo (s)","Amplitude");
  if(selAn.bm)chart("cBM",bd.w,bd.m,"w (rad/s)","dB","#60a5fa",true);
  if(selAn.bp)chart("cBP",bd.w,bd.p,"w (rad/s)","graus","#f472b6",true);
  if(curMalha==='aberta'&&selAn.nyqst)chartXY("cNyq",nqd.re,nqd.im,"Parte Real","Parte Imaginaria");
  if(curMalha==='fechada'&&selAn.lgr)chartLGR("cLGR",lgrData,tf);
  rd.scrollIntoView({behavior:"smooth"})}

/* ===== MODE SWITCHING ===== */
var curMode="diag",curSubMode="direct";
function setMode(m){curMode=m;
  document.getElementById("modeDiag").classList.toggle("active",m==="diag");
  document.getElementById("modeMan").classList.toggle("active",m==="manual");
  document.getElementById("diag-workspace").style.display=m==="diag"?"flex":"none";
  document.getElementById("diag-toolbar").style.display=m==="diag"?"flex":"none";
  document.getElementById("manualSec").classList.toggle("vis",m==="manual");
  document.getElementById("btnCalcMain").innerHTML=m==="diag"?"&#9654; CALCULAR DIAGRAMA":"&#9654; CALCULAR T(s)"}
function setSubMode(m){curSubMode=m;
  ["subDirect","subClosed","subOpen","subSS"].forEach(function(id){document.getElementById(id).classList.toggle("active",id==="sub"+m.charAt(0).toUpperCase()+m.slice(1))});
  document.getElementById("subDirect").classList.toggle("active",m==="direct");
  document.getElementById("subClosed").classList.toggle("active",m==="closed");
  document.getElementById("subOpen").classList.toggle("active",m==="open");
  document.getElementById("subSS").classList.toggle("active",m==="ss");
  document.getElementById("manDirect").style.display=m==="direct"?"block":"none";
  document.getElementById("manClosed").style.display=m==="closed"?"block":"none";
  document.getElementById("manOpen").style.display=m==="open"?"block":"none";
  document.getElementById("manSS").style.display=m==="ss"?"block":"none"}

function onCalc(){
  if(curMode==="manual"){
    var tf;
    if(curSubMode==="direct"){
      tf={n:parseP(document.getElementById("manNum").value),d:parseP(document.getElementById("manDen").value)};
    } else if(curSubMode==="closed"){
      var gn=parseP(document.getElementById("manGN").value),gd=parseP(document.getElementById("manGD").value);
      var hn=parseP(document.getElementById("manHN").value),hd=parseP(document.getElementById("manHD").value);
      /* T(s) = G/(1+GH) = (Gn*Hd) / (Gd*Hd + Gn*Hn) */
      tf={n:pMul(gn,hd),d:pAdd(pMul(gd,hd),pMul(gn,hn))};
    } else if(curSubMode==="open"){
      var gn=parseP(document.getElementById("manOGN").value),gd=parseP(document.getElementById("manOGD").value);
      var hn=parseP(document.getElementById("manOHN").value),hd=parseP(document.getElementById("manOHD").value);
      /* L(s) = G*H = (Gn*Hn) / (Gd*Hd) */
      tf={n:pMul(gn,hn),d:pMul(gd,hd)};
    } else if(curSubMode==="ss"){
      /* State Space */
      try{
        tf=ssToTF(
          document.getElementById("manA").value,
          document.getElementById("manB").value,
          document.getElementById("manC").value,
          document.getElementById("manD").value);
      }catch(e){
        var rd=document.getElementById("res"),rb=document.getElementById("rb");
        rd.classList.add("vis");rb.innerHTML='<div class="ebox">Erro no espaco de estados: '+esc(String(e))+'</div>';
        rd.scrollIntoView({behavior:"smooth"});return}
    }
    var lc=tf.d[tf.d.length-1];if(Math.abs(lc)>1e-14&&Math.abs(lc-1)>1e-10){tf.n=pScl(tf.n,1/lc);tf.d=pScl(tf.d,1/lc)}
    tf=pfReduce(tf);showRes(tf);return;
  }
  var r=solve(model.nodes,model.edges);
  if(r.e){var rd=document.getElementById("res"),rb=document.getElementById("rb");rd.classList.add("vis");rb.innerHTML='<div class="ebox">'+esc(r.e)+'</div>';rd.scrollIntoView({behavior:"smooth"});return}
  showRes(r.tf)}

/* ===== EDITOR STATE ===== */
var model={nodes:[],edges:[]},selId=null,dragSt=null,conSt=null;
var cw=document.getElementById("cw"),cv=document.getElementById("cv"),wSvg=document.getElementById("wires");
function nxtId(){var m=0;model.nodes.forEach(function(n){var v=parseInt(n.id.replace("n",""))||0;if(v>m)m=v});return"n"+(m+1)}
function ptr(e){if(e.touches&&e.touches.length)return{x:e.touches[0].clientX,y:e.touches[0].clientY};if(e.changedTouches&&e.changedTouches.length)return{x:e.changedTouches[0].clientX,y:e.changedTouches[0].clientY};return{x:e.clientX,y:e.clientY}}
var BL={tf:"Planta",gain:"Ganho",sum:"Somador",int:"Integrador",der:"Derivador",pid:"PID",sensor:"Sensor",actuator:"Atuador",input:"Entrada",output:"Saida",branch:"Ramificacao"};
function dPar(t){if(t==="tf")return{num:"1",den:"s+1"};if(t==="gain")return{k:"1"};if(t==="sum")return{signs:"+ -"};if(t==="pid")return{kp:"1",ki:"0",kd:"0"};if(t==="sensor"||t==="actuator")return{num:"1",den:"1"};if(t==="input")return{label:"R(s)"};if(t==="output")return{label:"Y(s)"};return{}}
function gPC(t,p){if(t==="input")return{i:[],o:[{id:"out0"}]};if(t==="output")return{i:[{id:"in0"}],o:[]};if(t==="branch")return{i:[{id:"in0"}],o:[{id:"out0"},{id:"out1"}]};
  if(t==="sum"){var sg=(p&&p.signs?p.signs:"+ -").trim().split(/\s+/);return{i:sg.map(function(s,i){return{id:"in"+i,sign:s}}),o:[{id:"out0"}]}}return{i:[{id:"in0"}],o:[{id:"out0"}]}}
function bTxt(n){var p=n.params||{};if(n.type==="tf"||n.type==="actuator")return'<div class="block-tf-disp"><div class="tf-num">'+(p.num||"1")+'</div><div>'+(p.den||"1")+'</div></div>';
  if(n.type==="gain")return"K="+(p.k||"1");if(n.type==="pid")return'<div class="block-tf-disp">Kp='+(p.kp||"0")+" Ki="+(p.ki||"0")+" Kd="+(p.kd||"0")+'</div>';
  if(n.type==="sensor")return'<div class="block-tf-disp"><div class="tf-num">'+(p.num||"1")+'</div><div>'+(p.den||"1")+'</div></div>';
  if(n.type==="input")return p.label||"R(s)";if(n.type==="output")return p.label||"Y(s)";if(n.type==="sum")return"\u03a3";if(n.type==="int")return"1/s";if(n.type==="der")return"s";return""}
/* addB: quick add (toolbar) */
function addB(t){var r=cw.getBoundingClientRect(),n={id:nxtId(),type:t,x:40+Math.random()*(r.width-200),y:40+Math.random()*(r.height-140),params:dPar(t)};
  if(t==="input"){n.x=30;n.y=r.height/2-30}if(t==="output"){n.x=r.width-160;n.y=r.height/2-30}model.nodes.push(n);render();setSel(n.id)}
/* addBP: add with params (modal) */
function addBP(t,p){var r=cw.getBoundingClientRect(),n={id:nxtId(),type:t,x:40+Math.random()*(r.width-200),y:40+Math.random()*(r.height-140),params:Object.assign(dPar(t),p)};
  if(t==="input"){n.x=30;n.y=r.height/2-30}if(t==="output"){n.x=r.width-160;n.y=r.height/2-30}model.nodes.push(n);render();setSel(n.id)}
function delSel(){if(!selId)return;model.edges=model.edges.filter(function(e){return e.src!==selId&&e.dst!==selId});model.nodes=model.nodes.filter(function(n){return n.id!==selId});selId=null;conSt=null;render()}
function clrAll(){model={nodes:[],edges:[]};selId=null;conSt=null;render();document.getElementById("res").classList.remove("vis")}
function autoLay(){if(!model.nodes.length)return;var lv={},aj={};model.edges.forEach(function(e){if(!aj[e.src])aj[e.src]=[];aj[e.src].push(e.dst)});
  var ins=model.nodes.filter(function(n){return n.type==="input"}).map(function(n){return n.id});if(!ins.length)ins=[model.nodes[0].id];
  var q=ins.map(function(id){return{id:id,l:0}}),vis={};ins.forEach(function(id){vis[id]=true});
  while(q.length){var c=q.shift();lv[c.id]=c.l;(aj[c.id]||[]).forEach(function(n){if(!vis[n]){vis[n]=true;q.push({id:n,l:c.l+1})}})}
  model.nodes.forEach(function(n){if(!(n.id in lv))lv[n.id]=3});var bl={};Object.keys(lv).forEach(function(id){var l=lv[id];if(!bl[l])bl[l]=[];bl[l].push(id)});
  Object.keys(bl).forEach(function(l){bl[l].forEach(function(id,i){var n=model.nodes.find(function(nd){return nd.id===id});if(n){n.x=50+parseInt(l)*170;n.y=50+i*100}})});render()}
function setSel(id){selId=id;document.querySelectorAll(".block").forEach(function(el){el.classList.toggle("sel",el.dataset.id===id)});updP()}
function pSty(t,k,i){if(t==="sum"){if(k==="in"&&i===0)return"left:-7px;top:50%;transform:translateY(-50%)";if(k==="in"&&i===1)return"left:50%;bottom:-7px;transform:translateX(-50%)";if(k==="in"&&i>=2)return"left:-7px;top:"+(8+i*14)+"px";return"right:-7px;top:50%;transform:translateY(-50%)"}
  if(t==="branch"){if(k==="in")return"left:-7px;top:50%;transform:translateY(-50%)";if(k==="out"&&i===0)return"right:-7px;top:50%;transform:translateY(-50%)";return"left:50%;bottom:-7px;transform:translateX(-50%)"}
  if(t==="input"&&k==="out")return"right:-7px;top:50%;transform:translateY(-50%)";if(t==="output"&&k==="in")return"left:-7px;top:50%;transform:translateY(-50%)";
  return k==="in"?"left:-7px;top:50%;transform:translateY(-50%)":"right:-7px;top:50%;transform:translateY(-50%)"}
function render(){cv.querySelectorAll(".block").forEach(function(el){el.remove()});
  model.nodes.forEach(function(n){var el=document.createElement("div");el.className="block block-"+n.type;el.dataset.id=n.id;el.style.left=n.x+"px";el.style.top=n.y+"px";
    var pc=gPC(n.type,n.params),h="";
    if(n.type==="sum")h='<div class="block-body">\u03a3</div>';else if(n.type==="branch")h="";
    else h='<div class="block-header">'+BL[n.type]+'</div><div class="block-body">'+bTxt(n)+'</div>';
    pc.i.forEach(function(p,i){h+='<div class="port port-in" data-port="'+p.id+'" data-nid="'+n.id+'" data-kind="in" style="'+pSty(n.type,"in",i)+'"></div>';
      if(n.type==="sum"&&p.sign)h+='<span class="sign-label" style="'+(i===0?"left:-20px;top:50%;transform:translateY(-50%)":i===1?"left:50%;bottom:-18px;transform:translateX(-50%)":"left:-20px;top:"+(8+i*14)+"px")+'">'+p.sign+'</span>'});
    pc.o.forEach(function(p,i){h+='<div class="port port-out" data-port="'+p.id+'" data-nid="'+n.id+'" data-kind="out" style="'+pSty(n.type,"out",i)+'"></div>'});
    el.innerHTML=h;cv.appendChild(el);
    el.addEventListener("mousedown",function(ev){onBD(ev,n.id)});el.addEventListener("touchstart",function(ev){onBD(ev,n.id)},{passive:false});
    el.querySelectorAll(".port").forEach(function(pl){pl.addEventListener("mousedown",function(ev){ev.stopPropagation();onPC(ev,pl)});
      pl.addEventListener("touchstart",function(ev){ev.stopPropagation();ev.preventDefault();onPC(ev,pl)},{passive:false})})});
  document.querySelectorAll(".block").forEach(function(el){el.classList.toggle("sel",el.dataset.id===selId)});rW();updP()}
function rW(){wSvg.querySelectorAll("path").forEach(function(p){p.remove()});var wr=cw.getBoundingClientRect();wSvg.setAttribute("width",wr.width);wSvg.setAttribute("height",wr.height);
  model.edges.forEach(function(e,idx){var a=gPP(e.src,e.srcPort||"out0"),b=gPP(e.dst,e.dstPort||"in0");if(!a||!b)return;
    var p=document.createElementNS("http://www.w3.org/2000/svg","path"),dx=Math.max(50,Math.abs(b.x-a.x)*.4);
    if(b.x<a.x-20){var mY=Math.max(a.y,b.y)+60;p.setAttribute("d","M "+a.x+" "+a.y+" C "+(a.x+40)+" "+a.y+","+(a.x+40)+" "+mY+","+((a.x+b.x)/2)+" "+mY+" S "+(b.x-40)+" "+b.y+","+b.x+" "+b.y);
      p.setAttribute("stroke","#e8a035");p.setAttribute("stroke-dasharray","8 4");p.setAttribute("marker-end","url(#ar2)")}
    else{p.setAttribute("d","M "+a.x+" "+a.y+" C "+(a.x+dx)+" "+a.y+","+(b.x-dx)+" "+b.y+","+b.x+" "+b.y);p.setAttribute("stroke","#5b6be0");p.setAttribute("marker-end","url(#ar)")}
    p.setAttribute("fill","none");p.setAttribute("stroke-width","2.5");p.style.pointerEvents="visibleStroke";p.style.cursor="pointer";
    p.addEventListener("click",function(){model.edges.splice(idx,1);render()});p.addEventListener("touchend",function(ev){ev.preventDefault();model.edges.splice(idx,1);render()});
    wSvg.appendChild(p)})}
function gPP(nid,pid){var bl=document.querySelector('.block[data-id="'+nid+'"]');if(!bl)return null;var pl=bl.querySelector('.port[data-port="'+pid+'"]');
  if(!pl){pl=bl.querySelector(".port.port-"+(pid.indexOf("out")===0?"out":"in"))}if(!pl)return null;var cr=cw.getBoundingClientRect(),pr=pl.getBoundingClientRect();
  return{x:pr.left+pr.width/2-cr.left,y:pr.top+pr.height/2-cr.top}}
function onBD(e,id){if(e.target.classList.contains("port"))return;if(e.button===2)return;e.preventDefault();setSel(id);var p=ptr(e),el=document.querySelector('.block[data-id="'+id+'"]');if(!el)return;var r=el.getBoundingClientRect();dragSt={id:id,ox:p.x-r.left,oy:p.y-r.top}}
function onPM(e){if(!dragSt)return;e.preventDefault();var p=ptr(e),wr=cw.getBoundingClientRect(),el=document.querySelector('.block[data-id="'+dragSt.id+'"]');if(!el)return;
  var nx=Math.max(0,Math.min(wr.width-el.offsetWidth,p.x-wr.left-dragSt.ox)),ny=Math.max(0,Math.min(wr.height-el.offsetHeight,p.y-wr.top-dragSt.oy));
  el.style.left=nx+"px";el.style.top=ny+"px";var nd=model.nodes.find(function(n){return n.id===dragSt.id});if(nd){nd.x=nx;nd.y=ny}rW()}
document.addEventListener("mousemove",onPM);document.addEventListener("mouseup",function(){dragSt=null});
document.addEventListener("touchmove",onPM,{passive:false});document.addEventListener("touchend",function(){dragSt=null});
cv.addEventListener("mousedown",function(e){if(e.target===cv){conSt=null;document.querySelectorAll(".port.active").forEach(function(p){p.classList.remove("active")});setSel(null)}});
cv.addEventListener("touchstart",function(e){if(e.target===cv){conSt=null;document.querySelectorAll(".port.active").forEach(function(p){p.classList.remove("active")});setSel(null)}},{passive:false});
function onPC(e,pl){e.preventDefault();var nid=pl.dataset.nid,pid=pl.dataset.port,kind=pl.dataset.kind;setSel(nid);
  if(conSt){if(conSt.nid===nid){conSt=null;document.querySelectorAll(".port.active").forEach(function(p){p.classList.remove("active")});return}
    var sI,sP,dI,dP;if(conSt.kind==="out"&&kind==="in"){sI=conSt.nid;sP=conSt.pid;dI=nid;dP=pid}
    else if(conSt.kind==="in"&&kind==="out"){sI=nid;sP=pid;dI=conSt.nid;dP=conSt.pid}
    else{conSt=null;document.querySelectorAll(".port.active").forEach(function(p){p.classList.remove("active")});return}
    if(!model.edges.some(function(ed){return ed.src===sI&&ed.srcPort===sP&&ed.dst===dI&&ed.dstPort===dP}))
      model.edges.push({id:"e"+Math.random().toString(36).slice(2),src:sI,srcPort:sP,dst:dI,dstPort:dP});
    conSt=null;document.querySelectorAll(".port.active").forEach(function(p){p.classList.remove("active")});render()}
  else{conSt={nid:nid,pid:pid,kind:kind};pl.classList.add("active")}}
function updP(){document.getElementById("pB").textContent=model.nodes.length;document.getElementById("pE").textContent=model.edges.length;
  var pa=document.getElementById("pA");if(!selId){pa.innerHTML='<span class="hint">Selecione um bloco.</span>';return}
  var nd=model.nodes.find(function(n){return n.id===selId});if(!nd){pa.innerHTML="";return}var p=nd.params||{};
  var h='<div style="font-size:11px;color:var(--txm);margin-bottom:6px">'+BL[nd.type]+' <b style="color:var(--tx)">'+nd.id+'</b></div>';
  if(nd.type==="tf"||nd.type==="sensor"||nd.type==="actuator"){h+=pI("Num","num",p.num||"1");h+=pI("Den","den",p.den||"1")}
  else if(nd.type==="gain")h+=pI("K","k",p.k||"1");
  else if(nd.type==="sum")h+=pI("Sinais","signs",p.signs||"+ -");
  else if(nd.type==="pid"){h+=pI("Kp","kp",p.kp||"1");h+=pI("Ki","ki",p.ki||"0");h+=pI("Kd","kd",p.kd||"0")}
  else if(nd.type==="input"||nd.type==="output")h+=pI("Label","label",p.label||"");
  pa.innerHTML=h;pa.querySelectorAll("input[data-key]").forEach(function(inp){inp.addEventListener("input",function(){nd.params[inp.dataset.key]=inp.value;
    if(inp.dataset.key==="signs")render();else{var bl=document.querySelector('.block[data-id="'+nd.id+'"] .block-body');if(bl)bl.innerHTML=bTxt(nd)}})})}
function pI(l,k,v){return'<div class="pg"><label>'+l+'</label><input data-key="'+k+'" value="'+esc(String(v))+'"></div>'}
document.querySelectorAll(".tb[data-add]").forEach(function(b){b.addEventListener("click",function(){addB(b.dataset.add)})});
document.getElementById("btnDel").addEventListener("click",delSel);document.getElementById("btnClear").addEventListener("click",clrAll);document.getElementById("btnAuto").addEventListener("click",autoLay);
document.addEventListener("keydown",function(e){if(e.target.tagName==="INPUT")return;if(e.key==="Delete"||e.key==="Backspace")delSel();if(e.key==="Escape"){conSt=null;closeModal();document.querySelectorAll(".port.active").forEach(function(p){p.classList.remove("active")})}});
render();new ResizeObserver(function(){rW()}).observe(cw);
</script></body></html>'''


# ══════════════════════════════════════════════════
# TELA INICIAL (SELECAO DE MODO)
# ══════════════════════════════════════════════════

def tela_inicial():
    st.markdown("""
    <style>
    .mode-card {
        background: linear-gradient(135deg, #1a1d2e, #252840);
        border: 2px solid #333654;
        border-radius: 16px;
        padding: 28px 24px;
        text-align: center;
        cursor: pointer;
        transition: all 0.2s;
        height: 100%;
    }
    .mode-card:hover {
        border-color: #5b6be0;
        transform: translateY(-2px);
        box-shadow: 0 8px 30px rgba(91,107,224,0.2);
    }
    .mode-icon { font-size: 42px; margin-bottom: 12px; }
    .mode-title { font-size: 20px; font-weight: 700; margin-bottom: 8px; color: #e0e4f0; }
    .mode-desc { font-size: 14px; color: #8890b0; line-height: 1.6; }
    .welcome-title {
        text-align: center; font-size: 32px; font-weight: 700;
        margin-bottom: 8px; color: #e0e4f0;
    }
    .welcome-sub {
        text-align: center; font-size: 16px; color: #8890b0;
        margin-bottom: 40px;
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown("")
    st.markdown("")
    st.markdown('<div class="welcome-title">Sistema de Modelagem e Analise de Controle</div>',
                unsafe_allow_html=True)
    st.markdown('<div class="welcome-sub">Escolha o modo de trabalho para comecar</div>',
                unsafe_allow_html=True)

col1, col2, = st.columns(3, gap="large")

with col2:
        st.markdown("""
        <div class="mode-card">
            <div class="mode-icon">&#128200;</div>
            <div class="mode-title">Modo Classico</div>
            <div class="mode-desc">
                Insira Planta, Controlador e Sensor por funcao de
                transferencia. Analise em malha aberta ou fechada
                com ganho K ajustavel.
            </div>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Selecionar Modo Classico", use_container_width=True, type="primary"):
            st.session_state.modo_selecionado = 'classico'
            st.rerun()

with col2:
        st.markdown("""
        <div class="mode-card">
            <div class="mode-icon">&#127912;</div>
            <div class="mode-title">Modo Canvas</div>
            <div class="mode-desc">
                Editor visual de diagrama de blocos. Arraste, conecte
                e calcule a FT equivalente graficamente.
                Entrada manual e espaco de estados integrados.
            </div>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Selecionar Modo Canvas", use_container_width=True, type="primary"):
            st.session_state.modo_selecionado = 'canvas'
            st.rerun()


# ══════════════════════════════════════════════════
# MODO CLASSICO
# ══════════════════════════════════════════════════

def modo_classico():
    st.title("Modo Classico - Funcao de Transferencia")

    with st.sidebar:
        st.header("Navegacao")
        if st.button("Voltar a Tela Inicial"):
            st.session_state.modo_selecionado = None
            st.rerun()

        st.markdown("---")
        st.header("Adicionar Blocos")

        nome = st.text_input("Nome", value="G1")
        tipo = st.selectbox("Tipo", ['Planta', 'Controlador', 'Sensor', 'Atuador'])
        numerador = st.text_input("Numerador", placeholder="ex: 4*s")
        denominador = st.text_input("Denominador", placeholder="ex: s^2 + 2*s + 3")

        if st.button("Adicionar", type="primary", use_container_width=True):
            ok, msg = adicionar_bloco(nome, tipo, 'Funcao de Transferencia',
                                      numerador, denominador)
            if ok:
                st.success(msg)
                st.rerun()
            else:
                st.error(msg)

        if not st.session_state.blocos.empty:
            st.markdown("---")
            st.header("Remover Blocos")
            excluir = st.selectbox("Selecionar", st.session_state.blocos['nome'])
            if st.button("Excluir"):
                remover_bloco(excluir)
                st.rerun()

        st.markdown("---")
        st.header("Configuracoes")
        if st.button(
            "Habilitar Calculo de Erro"
            if not st.session_state.calculo_erro_habilitado
            else "Desabilitar Calculo de Erro"
        ):
            st.session_state.calculo_erro_habilitado = (
                not st.session_state.calculo_erro_habilitado)
            st.rerun()

    # Calculo de erro
    if st.session_state.calculo_erro_habilitado:
        st.subheader("Calculo de Erro Estacionario")
        col1, col2 = st.columns(2)
        with col1:
            num_erro = st.text_input("Numerador", value="", key="num_erro")
        with col2:
            den_erro = st.text_input("Denominador", value="", key="den_erro")

        if st.button("Calcular Erro Estacionario"):
            try:
                G, _ = converter_para_tf(num_erro, den_erro)
                tipo, Kp, Kv, Ka = constantes_de_erro(G)
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Tipo", tipo)
                c2.metric("Kp", formatar_numero(Kp))
                c3.metric("Kv", formatar_numero(Kv))
                c4.metric("Ka", formatar_numero(Ka))
            except Exception as e:
                st.error(f"Erro: {e}")

    # Painel de analise
    col1, col2 = st.columns([2, 1])

    with col2:
        st.subheader("Configuracao")
        tipo_malha = st.selectbox("Tipo de Sistema", ["Malha Aberta", "Malha Fechada"])
        usar_ganho = st.checkbox("Adicionar ganho K ajustavel", value=False)
        K = st.slider("Ganho K", 0.1, 100.0, 1.0, 0.1) if usar_ganho else 1.0

        st.subheader("Analises")
        chave = "malha_fechada" if tipo_malha == "Malha Fechada" else "malha_aberta"
        analise_opcoes = ANALYSIS_OPTIONS[chave]
        analises = st.multiselect("Escolha:", analise_opcoes, default=[analise_opcoes[0]])
        entrada = st.selectbox("Sinal de Entrada", INPUT_SIGNALS)

    with col1:
        st.subheader("Resultados")

        if st.button("Executar Simulacao", use_container_width=True, type="primary"):
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
                    if controlador is not None:
                        sistema = controlador * sistema
                    st.info(f"Malha Aberta | K = {K:.2f}")
                else:
                    planta_com_ganho = ganho_tf * planta
                    sistema = calcular_malha_fechada(planta_com_ganho, controlador, sensor)
                    st.info(f"Malha Fechada | K = {K:.2f}")

                # Mostrar FT equivalente
                num_str = _tf_to_str(sistema.num[0][0])
                den_str = _tf_to_str(sistema.den[0][0])
                st.latex(f"T(s) = \\frac{{{num_str}}}{{{den_str}}}")

                executar_analises(sistema, analises, entrada, tipo_malha)

            except Exception as e:
                st.error(f"Erro durante a simulacao: {e}")


# ══════════════════════════════════════════════════
# MODO CANVAS
# ══════════════════════════════════════════════════

def modo_canvas():
    st.title("Modo Canvas - Editor Visual")

    with st.sidebar:
        st.header("Navegacao")
        if st.button("Voltar a Tela Inicial"):
            st.session_state.modo_selecionado = None
            st.rerun()

        st.markdown("---")
        st.markdown("### Como usar")
        st.markdown("""
        **Diagrama de Blocos:**
        1. Clique **+ Adicionar Bloco** ou use os botoes rapidos
        2. Arraste para posicionar
        3. Conecte: porta azul (saida) -> porta verde (entrada)
        4. Edite parametros no painel lateral
        5. Clique **CALCULAR** para ver resultados

        **Entrada Manual:**
        1. Clique em "Entrada Manual"
        2. Escolha: T(s) Direta, Malha Fechada, Malha Aberta ou Espaco de Estados
        3. Preencha os campos
        4. Clique **CALCULAR T(s)**
        """)

    html_content = _load_visual_editor_html()
    components.html(html_content, height=1200, scrolling=True)


# ══════════════════════════════════════════════════
# FUNCOES AUXILIARES DE FORMATACAO
# ══════════════════════════════════════════════════

def _tf_to_str(coeffs):
    """Converte coeficientes para string LaTeX de polinomio."""
    n = len(coeffs) - 1
    terms = []
    for i, c in enumerate(coeffs):
        power = n - i
        c_val = float(c)
        if abs(c_val) < 1e-10:
            continue
        if power == 0:
            terms.append(f"{c_val:.4g}")
        elif power == 1:
            if abs(c_val - 1) < 1e-10:
                terms.append("s")
            elif abs(c_val + 1) < 1e-10:
                terms.append("-s")
            else:
                terms.append(f"{c_val:.4g}s")
        else:
            if abs(c_val - 1) < 1e-10:
                terms.append(f"s^{{{power}}}")
            elif abs(c_val + 1) < 1e-10:
                terms.append(f"-s^{{{power}}}")
            else:
                terms.append(f"{c_val:.4g}s^{{{power}}}")
    if not terms:
        return "0"
    result = terms[0]
    for t in terms[1:]:
        if t.startswith('-'):
            result += f" - {t[1:]}"
        else:
            result += f" + {t}"
    return result


def _format_complex_list(arr):
    """Formata lista de numeros complexos para exibicao."""
    if len(arr) == 0:
        return "Nenhum"
    parts = []
    for v in arr:
        if np.isreal(v):
            parts.append(f"{np.real(v):.4f}")
        else:
            parts.append(f"{np.real(v):.4f} {'+' if np.imag(v) >= 0 else '-'} {abs(np.imag(v)):.4f}j")
    return ", ".join(parts)


# ══════════════════════════════════════════════════
# APLICACAO PRINCIPAL
# ══════════════════════════════════════════════════

def main():
    st.set_page_config(
        page_title="Modelagem de Sistemas de Controle",
        page_icon="&#9881;",
        layout="wide",
    )
    inicializar_estado()

    if st.session_state.modo_selecionado is None:
        tela_inicial()
    elif st.session_state.modo_selecionado == 'lista':
        modo_lista()
    elif st.session_state.modo_selecionado == 'classico':
        modo_classico()
    elif st.session_state.modo_selecionado == 'canvas':
        modo_canvas()


if __name__ == "__main__":
    main()
