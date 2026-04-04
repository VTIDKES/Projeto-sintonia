# -*- coding: utf-8 -*-
"""
Sistema de Modelagem e Análise de Sistemas de Controle v2.0
Refatorado com: tela inicial, espaço de estados, modal de blocos,
lógica corrigida de série/paralelo/feedback, simplificação automática.
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

INPUT_SIGNALS = ['Degrau', 'Rampa', 'Senoidal', 'Impulso', 'Parabólica']

BLOCK_TYPES = {
    'Planta': {'icon': 'G(s)', 'desc': 'Função de transferência da planta'},
    'Controlador': {'icon': 'C(s)', 'desc': 'Controlador (PID, Lead-Lag, etc.)'},
    'Sensor': {'icon': 'H(s)', 'desc': 'Sensor na malha de realimentação'},
    'Atuador': {'icon': 'A(s)', 'desc': 'Atuador do sistema'},
    'Pre-filtro': {'icon': 'F(s)', 'desc': 'Filtro antes do somador'},
    'Perturbação': {'icon': 'D(s)', 'desc': 'Perturbação/distúrbio'},
}

CONNECTION_TYPES = ['Série', 'Paralelo', 'Realimentação Negativa', 'Realimentação Positiva']

# ══════════════════════════════════════════════════
# INICIALIZAÇÃO DO SESSION STATE
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
        'representacao_classico': 'Função de Transferência',
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


# ══════════════════════════════════════════════════
# FUNÇÕES UTILITÁRIAS
# ══════════════════════════════════════════════════

def formatar_numero(valor):
    if np.isinf(valor):
        return '∞'
    if np.isnan(valor):
        return '-'
    return f"{valor:.3f}"


def _parse_value(v):
    """Parse a single value: numbers, math expressions, or expressions with 's'."""
    v = v.strip()
    if not v:
        return sp.Integer(0)
    try:
        return sp.Rational(v) if '/' in v else sp.Float(float(v))
    except (ValueError, TypeError):
        s = sp.Symbol('s')
        return parse_expr(v.replace('^', '**'), local_dict={'s': s})


def parse_matrix(text):
    """Parse matrix text. Returns (matrix, has_s).
    matrix is np.array (float) if purely numeric, or sp.Matrix if contains 's'.
    has_s indicates whether any entry contains the variable s."""
    text = text.strip()
    if text.startswith('['):
        try:
            return np.array(json.loads(text), dtype=float), False
        except Exception:
            text = text.replace('[', '').replace(']', '')
    rows = [r.strip() for r in text.split(';') if r.strip()]
    matrix = []
    has_s = False
    for row in rows:
        vals = re.split(r'[,\s]+', row.strip())
        parsed_row = []
        for v in vals:
            if not v:
                continue
            pv = _parse_value(v)
            if isinstance(pv, sp.Basic) and pv.free_symbols:
                has_s = True
            parsed_row.append(pv)
        matrix.append(parsed_row)
    if has_s:
        return sp.Matrix(matrix), True
    # All numeric — convert to numpy float
    float_matrix = []
    for row in matrix:
        float_matrix.append([float(v.evalf()) if isinstance(v, sp.Basic) else float(v) for v in row])
    return np.array(float_matrix, dtype=float), False


# ══════════════════════════════════════════════════
# FUNÇÕES DE TRANSFERÊNCIA E ESPAÇO DE ESTADOS
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
    A_mat, a_s = parse_matrix(A_str)
    B_mat, b_s = parse_matrix(B_str)
    C_mat, c_s = parse_matrix(C_str)
    D_mat, d_s = parse_matrix(D_str)

    has_symbolic = a_s or b_s or c_s or d_s

    s = sp.Symbol('s')

    if has_symbolic:
        # ── Caminho simbólico: G(s) = C(sI − A)⁻¹B + D ──
        A = sp.Matrix(A_mat) if not isinstance(A_mat, sp.Matrix) else A_mat
        B = sp.Matrix(B_mat) if not isinstance(B_mat, sp.Matrix) else B_mat
        C = sp.Matrix(C_mat) if not isinstance(C_mat, sp.Matrix) else C_mat
        D = sp.Matrix(D_mat) if not isinstance(D_mat, sp.Matrix) else D_mat

        n = A.shape[0]
        if A.shape[0] != A.shape[1]:
            raise ValueError(f"Matriz A deve ser quadrada ({n}x{n})")
        if B.shape[0] != n:
            B = B.T
        if C.shape[1] != n:
            C = C.T
        if B.shape[0] != n:
            raise ValueError(f"Matriz B deve ter {n} linhas")
        if C.shape[1] != n:
            raise ValueError(f"Matriz C deve ter {n} colunas")

        sI_A = s * sp.eye(n) - A
        G_matrix = C * sI_A.inv() * B + D
        tf_expr = sp.simplify(G_matrix[0, 0])
        num_sym, den_sym = sp.fraction(sp.together(tf_expr))

        num_poly = sp.Poly(sp.expand(num_sym), s)
        den_poly = sp.Poly(sp.expand(den_sym), s)
        num_coeffs = [float(c) for c in num_poly.all_coeffs()]
        den_coeffs = [float(c) for c in den_poly.all_coeffs()]

        if den_coeffs and den_coeffs[0] != 1:
            fator = den_coeffs[0]
            num_coeffs = [c / fator for c in num_coeffs]
            den_coeffs = [c / fator for c in den_coeffs]

        tf_sys = TransferFunction(num_coeffs, den_coeffs)
        return {
            "tipo": "SISO",
            "tf": tf_sys,
            "simbolico": tf_expr,
            "num": num_sym,
            "den": den_sym,
            "ss": None
        }

    # ── Caminho numérico (original) ──
    A = np.atleast_2d(A_mat).astype(float)
    B = np.atleast_2d(B_mat).astype(float)
    C = np.atleast_2d(C_mat).astype(float)
    D = np.atleast_2d(D_mat).astype(float)

    n = A.shape[0]
    if A.shape != (n, n):
        raise ValueError(f"Matriz A deve ser quadrada ({n}x{n})")
    if B.shape[0] != n and B.shape[1] == n:
        B = B.T
    if C.shape[1] != n and C.shape[0] == n:
        C = C.T
    if B.shape[0] != n:
        raise ValueError(f"Matriz B deve ter {n} linhas")
    if C.shape[1] != n:
        raise ValueError(f"Matriz C deve ter {n} colunas")

    ss_sys = ctrl.ss(A, B, C, D)
    tf_sys = ctrl.tf(ss_sys)

    if tf_sys.ninputs == 1 and tf_sys.noutputs == 1:
        num_coeffs = list(tf_sys.num[0][0])
        den_coeffs = list(tf_sys.den[0][0])
        num_sym = sum(c * s**k for k, c in enumerate(reversed(num_coeffs)))
        den_sym = sum(c * s**k for k, c in enumerate(reversed(den_coeffs)))
        return {
            "tipo": "SISO",
            "tf": tf_sys,
            "simbolico": num_sym / den_sym,
            "num": num_sym,
            "den": den_sym,
            "ss": ss_sys
        }
    else:
        G = []
        for i in range(tf_sys.noutputs):
            linha = []
            for j in range(tf_sys.ninputs):
                num_coeffs = list(tf_sys.num[i][j])
                den_coeffs = list(tf_sys.den[i][j])
                num_sym = sum(c * s**k for k, c in enumerate(reversed(num_coeffs)))
                den_sym = sum(c * s**k for k, c in enumerate(reversed(den_coeffs)))
                linha.append(num_sym / den_sym)
            G.append(linha)
        return {
            "tipo": "MIMO",
            "tf": tf_sys,
            "G": G,
            "ss": ss_sys
        }


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
# INTERCONEXÃO DE BLOCOS
# ══════════════════════════════════════════════════

def blocos_em_serie(tf_list):
    resultado = tf_list[0]
    for tf in tf_list[1:]:
        resultado = resultado * tf
    return resultado


def blocos_em_paralelo(tf_list):
    resultado = tf_list[0]
    for tf in tf_list[1:]:
        resultado = resultado + tf
    return resultado


def realimentacao(G, H=None, positiva=False):
    if H is None:
        H = TransferFunction([1], [1])
    sign = 1 if positiva else -1
    return ctrl.feedback(G, H, sign=sign)


def simplificar_diagrama(blocos_df, conexoes):
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
                resultado = tfs[0] if resultado is None else resultado * tfs[0]
            continue

        if tipo_con == 'Série':
            parcial = blocos_em_serie(tfs)
        elif tipo_con == 'Paralelo':
            parcial = blocos_em_paralelo(tfs)
        elif tipo_con == 'Realimentação Negativa':
            G = tfs[0]
            H = tfs[1] if len(tfs) > 1 else TransferFunction([1], [1])
            parcial = realimentacao(G, H, positiva=False)
        elif tipo_con == 'Realimentação Positiva':
            G = tfs[0]
            H = tfs[1] if len(tfs) > 1 else TransferFunction([1], [1])
            parcial = realimentacao(G, H, positiva=True)
        else:
            parcial = blocos_em_serie(tfs)

        if resultado is None:
            resultado = parcial
        else:
            resultado = resultado * parcial

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
# ANÁLISE DE SISTEMAS
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
        'Tipo': '1ª Ordem',
        'Const. tempo (t)': f"{formatar_numero(tau)} s",
        'Temp. subida (Tr)': f"{formatar_numero(2.2 * tau)} s",
        'Temp. acomodação (Ts)': f"{formatar_numero(4 * tau)} s",
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
        'Tipo': '2ª Ordem',
        'Freq. natural (wn)': f"{formatar_numero(wn)} rad/s",
        'Fator amortec. (z)': f"{formatar_numero(zeta)}",
        'Freq. amortec. (wd)': f"{formatar_numero(wd)} rad/s",
        'Sobressinal (Mp)': f"{formatar_numero(Mp)}%",
        'Temp. subida (Tr)': f"{formatar_numero(Tr)} s",
        'Temp. pico (Tp)': f"{formatar_numero(Tp)} s",
        'Temp. acomodação (Ts)': f"{formatar_numero(Ts)} s",
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
    label = f'{ordem}ª Ordem (Par dominante)' if par_dominante else f'{ordem}ª Ordem (Polo dominante)'
    resultado.update({
        'Tipo': label,
        'Freq. natural (wn)': f"{formatar_numero(wn)} rad/s",
        'Fator amortec. (z)': f"{formatar_numero(zeta)}",
        'Freq. amortec. (wd)': f"{formatar_numero(omega_d)} rad/s",
        'Sobressinal (Mp)': f"{formatar_numero(Mp)} %",
        'Temp. subida (Tr)': f"{formatar_numero(Tr)} s",
        'Temp. pico (Tp)': f"{formatar_numero(Tp)} s",
        'Temp. acomodação (Ts)': f"{formatar_numero(Ts)} s",
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
    return np.clip(ts_estimado * 1.2, a_min=5, a_max=60)


# ══════════════════════════════════════════════════
# FUNÇÕES DE PLOTAGEM
# ══════════════════════════════════════════════════

PLOTLY_DARK = dict(
    paper_bgcolor='#0e1117',
    plot_bgcolor='#1a1d2e',
    font=dict(color='#e0e4f0', family='system-ui, sans-serif'),
    xaxis=dict(gridcolor='#333654', zerolinecolor='#444870', linecolor='#333654'),
    yaxis=dict(gridcolor='#333654', zerolinecolor='#444870', linecolor='#333654'),
    legend=dict(bgcolor='rgba(26,29,46,0.8)', bordercolor='#333654', borderwidth=1),
    margin=dict(l=60, r=30, t=50, b=50),
)

PLOTLY_LIGHT = dict(
    paper_bgcolor='white',
    plot_bgcolor='white',
    font=dict(color='black', family='system-ui, sans-serif'),
    xaxis=dict(gridcolor='#d9d9d9', zerolinecolor='#cccccc', linecolor='#000000'),
    yaxis=dict(gridcolor='#d9d9d9', zerolinecolor='#cccccc', linecolor='#000000'),
    legend=dict(bgcolor='rgba(255,255,255,0.85)', bordercolor='#cccccc', borderwidth=1),
    margin=dict(l=60, r=30, t=50, b=50),
)


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
            hovertemplate='Zero<br>Real: %{x:.3f}<br>Imaginário: %{y:.3f}<extra></extra>'))
    if len(polos) > 0:
        fig.add_trace(go.Scatter(
            x=np.real(polos), y=np.imag(polos), mode='markers',
            marker=dict(symbol='x', size=12, color='red'), name='Polos',
            hovertemplate='Polo<br>Real: %{x:.3f}<br>Imaginário: %{y:.3f}<extra></extra>'))
    fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.7)
    fig.add_vline(x=0, line_dash="dash", line_color="gray", opacity=0.7)
    fig.update_layout(
        title='Diagrama de Polos e Zeros',
        xaxis_title='Parte Real',
        yaxis_title='Parte Imaginaria',
        showlegend=True,
        hovermode='closest',
        template='plotly_white',
        **PLOTLY_LIGHT
    )
    return configurar_linhas_interativas(fig)


def _gerar_sinal_entrada(entrada, t):
    sinais = {
        'Degrau': np.ones_like(t),
        'Rampa': t,
        'Senoidal': np.sin(2 * np.pi * t),
        'Impulso': np.concatenate([[1], np.zeros(len(t) - 1)]),
        'Parabólica': t**2,
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
        x=t_out, y=y, mode='lines', line=dict(color='red'), name='Saída',
        hovertemplate='Tempo: %{x:.2f}s<br>Saída: %{y:.3f}<extra></extra>'))
    fig.update_layout(
        title=f'Resposta Temporal - Entrada: {entrada}',
        xaxis_title='Tempo (s)',
        yaxis_title='Amplitude',
        showlegend=True,
        hovermode='x unified',
        template='plotly_white',
        **PLOTLY_LIGHT
    )
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
        fig.update_xaxes(title_text="Frequência (rad/s)", type="log", row=1, col=1)
        fig.update_xaxes(title_text="Frequência (rad/s)", type="log", row=2, col=1)
        fig.update_yaxes(title_text="Magnitude (dB)", row=1, col=1)
        fig.update_yaxes(title_text="Fase (deg)", row=2, col=1)
        fig.update_layout(height=700, title_text="Diagrama de Bode")
    elif tipo == 'magnitude':
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=w, y=mag, mode='lines', line=dict(color='blue', width=3),
            name='Magnitude'))
        fig.update_layout(
            title='Bode - Magnitude', xaxis_title="Frequência (rad/s)",
            yaxis_title="Magnitude (dB)", xaxis_type='log')
    else:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=w, y=phase, mode='lines', line=dict(color='red', width=3),
            name='Fase'))
        fig.update_layout(
            title='Bode - Fase', xaxis_title="Frequência (rad/s)",
            yaxis_title="Fase (deg)", xaxis_type='log')
    fig.update_layout(template='plotly_white', **PLOTLY_LIGHT)
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
        title='Lugar Geométrico das Raízes (LGR)',
        xaxis_title='Parte Real',
        yaxis_title='Parte Imaginaria',
        showlegend=True,
        hovermode='closest',
        template='plotly_white',
        **PLOTLY_LIGHT
    )
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
        line=dict(dash='dash', color='gray', width=1), name='Reflexo simétrico'))
    fig.add_trace(go.Scatter(
        x=[-1], y=[0], mode='markers',
        marker=dict(symbol='circle', size=12, color='red'),
        name='Ponto crítico (-1,0)'))
    fig.add_hline(y=0, line_color="black", line_width=1)
    fig.add_vline(x=0, line_color="black", line_width=1)
    fig.update_layout(
        title='Diagrama de Nyquist',
        xaxis_title='Parte Real',
        yaxis_title='Parte Imaginaria',
        showlegend=True,
        hovermode='closest',
        template='plotly_white',
        **PLOTLY_LIGHT
    )
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
        if representacao == 'Função de Transferência':
            tf_obj, tf_symb = converter_para_tf(numerador, denominador)
            ss_sys = None
        else:
            A_mat, _ = parse_matrix(A_str)
            if A_mat.shape[0] > 4:
                return False, "Erro: dimensão máxima permitida é 4x4."
            resultado = converter_ss_para_tf(A_str, B_str, C_str, D_str)
            tf_obj  = resultado["tf"]
            tf_symb = resultado.get("simbolico", resultado.get("G", None))
            ss_sys  = resultado.get("ss", None)
            numerador   = " ".join(f"{v:.10g}" for v in tf_obj.num[0][0])
            denominador = " ".join(f"{v:.10g}" for v in tf_obj.den[0][0])

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


def calcular_operacao_entre_sistemas(nome_resultado, nome_g1, operacao, nome_g2,
                                      nome_g2_unit=False):
    try:
        G1 = obter_bloco_por_nome(nome_g1)
        if G1 is None:
            return False, f"Bloco '{nome_g1}' não encontrado.", None

        if nome_g2_unit:
            G2 = TransferFunction([1], [1])
        else:
            G2 = obter_bloco_por_nome(nome_g2)
            if G2 is None:
                return False, f"Bloco '{nome_g2}' não encontrado.", None

        if operacao == 'Série':
            resultado = G1 * G2
        elif operacao == 'Paralelo':
            resultado = G1 + G2
        elif operacao == 'Realimentação Negativa':
            resultado = ctrl.feedback(G1, G2, sign=-1)
        elif operacao == 'Realimentação Positiva':
            resultado = ctrl.feedback(G1, G2, sign=+1)
        else:
            return False, f"Operação '{operacao}' desconhecida.", None

        resultado = ctrl.minreal(resultado, verbose=False)

        if any(st.session_state.blocos['nome'] == nome_resultado):
            return False, f"Nome '{nome_resultado}' já existe. Escolha outro.", None

        num_str = str(list(resultado.num[0][0]))
        den_str = str(list(resultado.den[0][0]))
        ok, msg = adicionar_bloco(
            nome_resultado, 'Planta', 'Função de Transferência',
            num_str, den_str
        )
        if not ok:
            return False, msg, None

        return True, f"✅ {nome_resultado} criado com sucesso.", resultado

    except Exception as e:
        return False, f"Erro ao calcular operação: {e}", None


# ══════════════════════════════════════════════════
# EXECUÇÃO DE ANÁLISES
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
                c3.metric("Z = P + N", f"{Z} ({'Estável' if Z == 0 else 'Instável'})")
                st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Erro em '{analise}': {e}")


# ══════════════════════════════════════════════════
# TELA INICIAL
# ══════════════════════════════════════════════════

def tela_inicial():
    st.markdown("""
    <style>
    .mode-card {
        background: linear-gradient(135deg, #1a1d2e, #252840);
        border: 2px solid #333654;
        border-radius: 16px;
        overflow: hidden;
        text-align: center;
        cursor: pointer;
        transition: all 0.25s ease;
        display: flex;
        flex-direction: column;
        min-height: 350px;
        user-select: none;
    }
    .mode-card-body {
        padding: 28px 24px 24px;
        display: flex;
        flex-direction: column;
        align-items: center;
        flex: 1;
    }
    .mode-card-classic  { border-top: 4px solid #5b6be0; }
    .mode-card-diagram  { border-top: 4px solid #34d399; }
    .mode-card-classic:hover {
        border-color: #5b6be0;
        transform: translateY(-3px);
        box-shadow: 0 14px 40px rgba(91,107,224,0.3);
        background: linear-gradient(135deg, #1e2238, #2b2f52);
    }
    .mode-card-diagram:hover {
        border-color: #34d399;
        transform: translateY(-3px);
        box-shadow: 0 14px 40px rgba(52,211,153,0.2);
        background: linear-gradient(135deg, #1a2230, #1e2e2a);
    }
    .mode-icon {
        width: 100%;
        max-width: 230px;
        margin: 0 auto 20px;
        display: flex;
        align-items: center;
        justify-content: center;
    }
    .mode-title { font-size: 20px; font-weight: 700; margin-bottom: 10px; color: #e0e4f0; }
    .mode-desc  { font-size: 14px; color: #8890b0; line-height: 1.7; flex: 1; }
    .mode-hint  { margin-top: 14px; font-size: 12px; font-weight: 600; letter-spacing: .4px; }
    .mode-hint-classic { color: #5b6be0; }
    .mode-hint-diagram { color: #34d399; }
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
    st.markdown('<div class="welcome-title">Sistema de Modelagem e Análise de Controle</div>',
                unsafe_allow_html=True)
    st.markdown('<div class="welcome-sub">Escolha o modo de trabalho para começar</div>',
                unsafe_allow_html=True)

    col1, col2 = st.columns(2, gap="large")

    with col1:
        st.markdown("""
        <div class="mode-card mode-card-classic">
            <div class="mode-card-body">
                <div class="mode-icon">
                    <svg viewBox="0 0 240 80" xmlns="http://www.w3.org/2000/svg" width="100%">
                      <line x1="20" y1="10" x2="20" y2="72" stroke="#555" stroke-width="1.2"/>
                      <line x1="20" y1="72" x2="230" y2="72" stroke="#555" stroke-width="1.2"/>
                      <polygon points="20,8 17,14 23,14" fill="#555"/>
                      <polygon points="232,72 226,69 226,75" fill="#555"/>
                      <line x1="20" y1="17" x2="230" y2="17" stroke="#444" stroke-width="1" stroke-dasharray="5,3"/>
                      <text x="233" y="75" fill="#8890b0" font-size="10" font-family="monospace">t</text>
                      <text x="5" y="15" fill="#8890b0" font-size="10" font-family="monospace">y</text>
                      <polyline points="20,72 30,54 40,42 50,34 60,28 70,24 95,20 120,18 170,17 225,17"
                                fill="none" stroke="#5b6be0" stroke-width="2.5"
                                stroke-linecap="round" stroke-linejoin="round"/>
                      <circle cx="225" cy="17" r="3.5" fill="#5b6be0"/>
                    </svg>
                </div>
                <div class="mode-title">Modo Clássico</div>
                <div class="mode-desc">
                    Insira Planta, Controlador e Sensor por função de
                    transferência ou espaço de estados. Análise em
                    malha aberta ou fechada com ganho K ajustável.
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Entrar no Modo Clássico", key="btn_classico", type="primary", use_container_width=True):
            st.session_state.modo_selecionado = 'classico'
            st.rerun()

    with col2:
        st.markdown("""
        <div class="mode-card mode-card-diagram">
            <div class="mode-card-body">
                <div class="mode-icon">
                    <svg viewBox="0 0 240 80" xmlns="http://www.w3.org/2000/svg" width="100%">
                      <defs>
                        <marker id="arw-diag" markerWidth="8" markerHeight="6" refX="7" refY="3" orient="auto">
                          <polygon points="0 0, 8 3, 0 6" fill="#34d399"/>
                        </marker>
                      </defs>
                      <text x="3" y="45" fill="#8890b0" font-size="12" font-family="monospace" font-weight="bold">R(s)</text>
                      <line x1="35" y1="40" x2="58" y2="40" stroke="#34d399" stroke-width="2" marker-end="url(#arw-diag)"/>
                      <rect x="60" y="18" width="120" height="44" fill="#1a1d2e" stroke="#34d399" stroke-width="1.5" rx="5"/>
                      <text x="120" y="35" fill="#e0e4f0" font-size="13" font-family="monospace" text-anchor="middle">4</text>
                      <line x1="73" y1="40" x2="167" y2="40" stroke="#e0e4f0" stroke-width="1" opacity="0.35"/>
                      <text x="120" y="55" fill="#e0e4f0" font-size="13" font-family="monospace" text-anchor="middle">s + 4</text>
                      <line x1="180" y1="40" x2="203" y2="40" stroke="#34d399" stroke-width="2" marker-end="url(#arw-diag)"/>
                      <text x="206" y="45" fill="#8890b0" font-size="12" font-family="monospace" font-weight="bold">Y(s)</text>
                    </svg>
                </div>
                <div class="mode-title">Modo Diagrama de Blocos</div>
                <div class="mode-desc">
                    Editor visual de diagrama de blocos. Arraste, conecte
                    e calcule a FT equivalente graficamente.
                    Entrada manual e espaço de estados integrados.
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Entrar no Modo Diagrama de Blocos", key="btn_canvas", type="primary", use_container_width=True):
            st.session_state.modo_selecionado = 'canvas'
            st.rerun()


# ══════════════════════════════════════════════════
# WIDGET DE ESPAÇO DE ESTADOS (BONITO)
# ══════════════════════════════════════════════════

def _render_ss_widget(uid, key_prefix):
    """
    Renderiza o editor bonito de Espaço de Estados com seletor de dimensão
    e grade de inputs nativos do Streamlit. Retorna (A_str, B_str, C_str, D_str).
    """
    dim_key = f"{key_prefix}_ss_dim"
    if dim_key not in st.session_state:
        st.session_state[dim_key] = 2

    # ── Seletor de dimensão ──
    st.markdown("""
    <style>
    .ss-section-title {
        font-size: 13px; font-weight: 700; letter-spacing: .5px;
        text-transform: uppercase; color: #8890b0; margin: 10px 0 6px;
    }
    .ss-dim-label {
        font-size: 12px; color: #8890b0; margin-right: 6px; font-weight: 600;
    }
    .ss-matrix-label {
        font-size: 11px; font-weight: 700; color: #ef4444;
        text-transform: uppercase; letter-spacing: .5px;
        padding: 4px 10px; background: rgba(239,68,68,.1);
        border-left: 3px solid #ef4444; border-radius: 0 4px 4px 0;
        margin-bottom: 6px; display: inline-block;
    }
    .ss-matrix-sub {
        font-size: 9px; color: #64748b; font-weight: 400;
        text-transform: none; letter-spacing: 0; margin-left: 4px;
    }
    .ss-info-box {
        background: linear-gradient(135deg, #1a1d2e, #1e2340);
        border: 1px solid #333654; border-radius: 10px;
        padding: 12px 16px; margin: 8px 0 14px;
        font-family: monospace; font-size: 12px; color: #8890b0;
        line-height: 1.7;
    }
    .ss-info-box b { color: #e0e4f0; }
    .ss-info-box span { color: #5b6be0; }
    .ss-eq { color: #34d399; font-weight: 700; }
    </style>
    """, unsafe_allow_html=True)

    st.markdown('<div class="ss-section-title">Dimensão do Sistema</div>', unsafe_allow_html=True)

    dcols = st.columns([1,1,1,1,4])
    for i, n in enumerate([1,2,3,4]):
        with dcols[i]:
            is_active = st.session_state[dim_key] == n
            if st.button(
                f"**{n}×{n}**" if is_active else f"{n}×{n}",
                key=f"{key_prefix}_dim_{n}",
                type="primary" if is_active else "secondary",
                use_container_width=True
            ):
                st.session_state[dim_key] = n
                st.rerun()

    _n = st.session_state[dim_key]

    # Info box com equações
    st.markdown(f"""
    <div class="ss-info-box">
        <span class="ss-eq">ẋ = Ax + Bu</span> &nbsp;|&nbsp; <span class="ss-eq">y = Cx + Du</span><br>
        <b>A</b>: {_n}×{_n} &nbsp;·&nbsp; <b>B</b>: {_n}×1 &nbsp;·&nbsp; <b>C</b>: 1×{_n} &nbsp;·&nbsp; <b>D</b>: 1×1
        &nbsp;→&nbsp; <b>T(s)</b> = C(sI−A)⁻¹B + D
    </div>
    """, unsafe_allow_html=True)

    # ── Matriz A ──
    st.markdown('<div class="ss-matrix-label">A <span class="ss-matrix-sub">(nxn) — Dinâmica</span></div>', unsafe_allow_html=True)
    A_vals = []
    for i in range(_n):
        cols = st.columns(_n)
        row = []
        for j in range(_n):
            default = "1" if i == j else "0"
            val = cols[j].text_input(
                f"a[{i+1},{j+1}]",
                value=default,
                key=f"{key_prefix}_A_{i}_{j}",
                label_visibility="collapsed"
            )
            row.append(val.strip() if val.strip() else "0")
        A_vals.append(row)
    A_str = "; ".join(" ".join(row) for row in A_vals)

    st.markdown("<div style='margin: 8px 0'></div>", unsafe_allow_html=True)

    # ── Matrizes B e C lado a lado ──
    col_b, col_c = st.columns(2)

    with col_b:
        st.markdown('<div class="ss-matrix-label">B <span class="ss-matrix-sub">(nx1) — Entrada</span></div>', unsafe_allow_html=True)
        B_vals = []
        for i in range(_n):
            val = st.text_input(
                f"b[{i+1},1]",
                value="0",
                key=f"{key_prefix}_B_{i}",
                label_visibility="collapsed"
            )
            B_vals.append(val.strip() if val.strip() else "0")
        B_str = "; ".join(B_vals)

    with col_c:
        st.markdown('<div class="ss-matrix-label">C <span class="ss-matrix-sub">(1xn) — Saída</span></div>', unsafe_allow_html=True)
        C_vals = []
        c_cols = st.columns(_n)
        for j in range(_n):
            default = "1" if j == 0 else "0"
            val = c_cols[j].text_input(
                f"c[1,{j+1}]",
                value=default,
                key=f"{key_prefix}_C_{j}",
                label_visibility="collapsed"
            )
            C_vals.append(val.strip() if val.strip() else "0")
        C_str = " ".join(C_vals)

    st.markdown("<div style='margin: 8px 0'></div>", unsafe_allow_html=True)

    # ── Matriz D ──
    st.markdown('<div class="ss-matrix-label">D <span class="ss-matrix-sub">(1x1) — Transmissão direta</span></div>', unsafe_allow_html=True)
    D_val = st.text_input(
        "d[1,1]",
        value="0",
        key=f"{key_prefix}_D",
        label_visibility="collapsed"
    )
    D_str = D_val.strip() if D_val.strip() else "0"

    # Preview da TF convertida
    try:
        res = converter_ss_para_tf(A_str, B_str, C_str, D_str)
        tf_preview = res["tf"]
        num_p = _tf_to_str(tf_preview.num[0][0])
        den_p = _tf_to_str(tf_preview.den[0][0])
        st.markdown(f"""
        <div style="background:rgba(52,211,153,.06);border:1px solid rgba(52,211,153,.25);
             border-radius:8px;padding:10px 14px;margin-top:10px;text-align:center">
            <div style="font-size:10px;color:#34d399;font-weight:700;text-transform:uppercase;
                 letter-spacing:.5px;margin-bottom:6px">T(s) equivalente</div>
            <div style="font-family:monospace;font-size:13px;color:#e0e4f0">
                <div style="border-bottom:1px solid rgba(255,255,255,.2);padding-bottom:4px;margin-bottom:4px">{num_p}</div>
                <div>{den_p}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    except Exception:
        pass

    return A_str, B_str, C_str, D_str


# ══════════════════════════════════════════════════
# MODO CLÁSSICO
# ══════════════════════════════════════════════════

def modo_classico():
    st.title("Modo Clássico — Função de Transferência")

    with st.sidebar:
        st.header("Navegação")
        if st.button("← Voltar à Tela Inicial", use_container_width=True):
            st.session_state.modo_selecionado = None
            st.rerun()

        st.markdown("---")

        # ── Planta (primeira TF) ──
        tem_planta = not st.session_state.blocos.empty
        if not tem_planta:
            st.header("Planta G(s)")
            st.caption("Insira a função de transferência principal.")
        else:
            st.header("Adicionar Função de Transferência")

        nome = st.text_input(
            "Nome",
            value=f"G{len(st.session_state.blocos)+1}",
            key="cl_nome")

        representacao = st.radio(
            "Representação",
            ['Função de Transferência', 'Espaço de Estados'],
            horizontal=True,
            key="representacao_classico"
        )

        if representacao == 'Função de Transferência':
            numerador   = st.text_input("Numerador",   placeholder="ex: 4",     key="cl_num")
            denominador = st.text_input("Denominador", placeholder="ex: s^2+2s+3", key="cl_den")
            A_str = B_str = C_str = D_str = ''
        else:
            A_str, B_str, C_str, D_str = _render_ss_widget("sidebar", "cl_sidebar")
            numerador = denominador = ''

        # Se já existe planta, pedir operação
        if tem_planta:
            operacao_nova = st.selectbox(
                "Operação com o sistema atual",
                CONNECTION_TYPES,
                key="cl_op_tipo")
            if operacao_nova in ['Realimentação Negativa', 'Realimentação Positiva']:
                st.caption("A nova TF será usada como H(s) na realimentação.")

        if st.button(
                "Definir Planta" if not tem_planta else "Adicionar e Aplicar",
                type="primary", use_container_width=True):
            tipo_bloco = 'Planta' if not tem_planta else 'Planta'
            ok, msg = adicionar_bloco(nome, tipo_bloco, representacao,
                                      numerador, denominador,
                                      A_str, B_str, C_str, D_str)
            if ok:
                if tem_planta:
                    # Auto-aplicar operação com o bloco anterior
                    nomes_existentes = list(st.session_state.blocos['nome'])
                    # O bloco recém-adicionado é o último
                    bloco_novo = nomes_existentes[-1]
                    # O sistema acumulado até agora: usar todos os anteriores
                    blocos_anteriores = nomes_existentes[:-1]
                    if len(blocos_anteriores) >= 1:
                        bloco_base = blocos_anteriores[-1]
                        st.session_state.conexoes.append({
                            'tipo': operacao_nova,
                            'blocos': [bloco_base, bloco_novo],
                        })
                st.success(msg)
                st.rerun()
            else:
                st.error(msg)

        # ── Lista simples de TFs no sistema ──
        if not st.session_state.blocos.empty:
            st.markdown("---")
            st.subheader("Sistema Atual")

            simbolos = {'Série': ' → ', 'Paralelo': ' || ',
                        'Realimentação Negativa': ' -fb ',
                        'Realimentação Positiva': ' +fb '}

            for idx, row in st.session_state.blocos.iterrows():
                tf_obj = row['tf']
                num_s = _tf_to_str(tf_obj.num[0][0])
                den_s = _tf_to_str(tf_obj.den[0][0])
                # Encontrar operação associada
                op_label = ""
                if idx > 0:
                    for con in st.session_state.conexoes:
                        if row['nome'] in con['blocos'] and con['blocos'][-1] == row['nome']:
                            op_label = f" [{con['tipo']}]"
                            break
                st.text(f"{row['nome']}: ({num_s}) / ({den_s}){op_label}")
                if st.button(f"Remover {row['nome']}", key=f"rm_cl_{idx}",
                             use_container_width=True):
                    remover_bloco(row['nome'])
                    st.rerun()

        st.markdown("---")
        st.header("Configurações")
        lbl_erro = ("Desabilitar Cálculo de Erro"
                    if st.session_state.calculo_erro_habilitado
                    else "Habilitar Cálculo de Erro")
        if st.button(lbl_erro, use_container_width=True):
            st.session_state.calculo_erro_habilitado = (
                not st.session_state.calculo_erro_habilitado)
            st.rerun()

    # ── Área principal ──

    # Cálculo de erro
    if st.session_state.calculo_erro_habilitado:
        st.subheader("Cálculo de Erro Estacionário")
        c1, c2 = st.columns(2)
        with c1:
            num_erro = st.text_input("Numerador G(s)", value="", key="num_erro")
        with c2:
            den_erro = st.text_input("Denominador G(s)", value="", key="den_erro")

        if st.button("Calcular Erro Estacionário"):
            try:
                G, _ = converter_para_tf(num_erro, den_erro)
                tipo_s, Kp, Kv, Ka = constantes_de_erro(G)
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Tipo", tipo_s)
                c2.metric("Kp", formatar_numero(Kp))
                c3.metric("Kv", formatar_numero(Kv))
                c4.metric("Ka", formatar_numero(Ka))
            except Exception as e:
                st.error(f"Erro: {e}")
        st.markdown("---")

    # ── Configuração de análise ──
    col1, col2 = st.columns([2, 1])

    with col2:
        st.subheader("Configuração")
        tipo_malha = st.selectbox("Tipo de Sistema", ["Malha Aberta", "Malha Fechada"])
        usar_ganho = st.checkbox("Adicionar ganho K ajustável", value=False)
        K = st.slider("Ganho K", 0.1, 100.0, 1.0, 0.1) if usar_ganho else 1.0

        st.subheader("Análises")
        chave = "malha_fechada" if tipo_malha == "Malha Fechada" else "malha_aberta"
        analise_opcoes = ANALYSIS_OPTIONS[chave]
        analises = st.multiselect("Escolha:", analise_opcoes, default=[analise_opcoes[0]])
        entrada = st.selectbox("Sinal de Entrada", INPUT_SIGNALS)

    with col1:
        st.subheader("Resultados")

        if st.button("Executar Simulação", use_container_width=True, type="primary"):
            try:
                df = st.session_state.blocos
                if df.empty:
                    st.warning("Adicione a planta primeiro.")
                    st.stop()

                ganho_tf = TransferFunction([K], [1])

                if st.session_state.conexoes:
                    sistema = simplificar_diagrama(
                        st.session_state.blocos, st.session_state.conexoes)
                    if usar_ganho and K != 1.0:
                        sistema = ganho_tf * sistema
                    if tipo_malha == "Malha Fechada" and not any(
                            c['tipo'].startswith('Realimentação')
                            for c in st.session_state.conexoes):
                        sistema = ctrl.feedback(sistema, TransferFunction([1], [1]))
                    label_extra = " (com conexões)"
                else:
                    planta      = obter_bloco_por_tipo('Planta')
                    controlador = obter_bloco_por_tipo('Controlador')
                    sensor      = obter_bloco_por_tipo('Sensor')

                    if planta is None:
                        st.error("Adicione pelo menos uma Planta.")
                        st.stop()

                    if tipo_malha == "Malha Aberta":
                        sistema = ganho_tf * planta
                        if controlador is not None:
                            sistema = controlador * sistema
                    else:
                        planta_com_ganho = ganho_tf * planta
                        sistema = calcular_malha_fechada(
                            planta_com_ganho, controlador, sensor)
                    label_extra = ""

                sistema = ctrl.minreal(sistema, verbose=False)
                st.info(f"{tipo_malha} | K = {K:.2f}{label_extra}")

                num_str = _tf_to_str(sistema.num[0][0])
                den_str = _tf_to_str(sistema.den[0][0])
                st.latex(f"T(s) = \\frac{{{num_str}}}{{{den_str}}}")

                executar_analises(sistema, analises, entrada, tipo_malha)

            except Exception as e:
                st.error(f"Erro durante a simulação: {e}")


# ══════════════════════════════════════════════════
# EDITOR VISUAL HTML (CANVAS)
# ══════════════════════════════════════════════════

def _load_visual_editor_html():
    return r'''<!DOCTYPE html>
<html lang="pt-BR"><head><meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1,maximum-scale=1,user-scalable=no">
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>
:root{--bg:#0e1117;--sf:#1a1d2e;--sf2:#252840;--bd:#333654;--tx:#e0e4f0;--txm:#8890b0;
--acc:#5b6be0;--grn:#34d399;--red:#f87171;--yel:#fbbf24;--blu:#60a5fa;--pur:#a78bfa;--pnk:#f472b6}
*{margin:0;padding:0;box-sizing:border-box}
html,body{height:auto;min-height:100%;font-family:system-ui,sans-serif;background:var(--bg);color:var(--tx);overflow-y:auto}
.app{display:flex;flex-direction:column;min-height:100%}
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
.block-ss{background:linear-gradient(135deg,#1a3d2a,#0f2518);border:1px solid #2d8a50}
.block-input{background:linear-gradient(135deg,#1a3d1a,#0f250f);border:1px solid #2d8a2d;border-radius:20px}
.block-output{background:linear-gradient(135deg,#3d1a1a,#250f0f);border:1px solid #8a2d2d;border-radius:20px}
.block-sum{background:linear-gradient(135deg,#1a3d3a,#122a28);border:2px solid #2d8a70;border-radius:50%;min-width:56px;width:56px;height:56px;display:flex;align-items:center;justify-content:center}
.block-sum .block-header{display:none}.block-sum .block-body{padding:0;font-size:20px;text-align:center}
.block-branch{background:#3b82f6;border:2px solid #2563eb;border-radius:50%;min-width:56px;width:56px;height:56px}
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
.conn-item{display:flex;align-items:center;gap:8px;padding:10px 12px;margin-bottom:5px;background:var(--sf2);border:2px solid var(--bd);border-radius:8px;cursor:pointer;font-size:13px;transition:all .15s;user-select:none}
.conn-item:hover{border-color:var(--acc);background:#2f3349}
.conn-item.conn-sel{border-color:#5b6be0;background:#2a2f4a;box-shadow:0 0 8px rgba(91,107,224,.3)}
.conn-item .conn-chk{font-size:16px;color:var(--txm);min-width:18px;text-align:center}
.conn-item.conn-sel .conn-chk{color:#5b6be0}
</style></head><body><div class="app">

<div class="modal-overlay" id="addModal">
<div class="modal">
<div class="modal-hdr"><h3>Adicionar Bloco</h3><button class="modal-close" onclick="closeModal()">&times;</button></div>
<div class="modal-body">
<div id="modalGrid">
<div class="cat-label">Sinais</div>
<div class="block-grid">
<div class="block-option" onclick="pickBlock('input')"><div class="bo-icon" style="color:var(--grn)">R(s)</div><div class="bo-label">Entrada</div><div class="bo-desc">Sinal de referência</div></div>
<div class="block-option" onclick="pickBlock('output')"><div class="bo-icon" style="color:var(--red)">Y(s)</div><div class="bo-label">Saída</div><div class="bo-desc">Sinal de saída</div></div>
</div>
<div class="cat-label">Blocos de Transferência</div>
<div class="block-grid">
<div class="block-option" onclick="pickBlock('tf')"><div class="bo-icon" style="color:var(--blu)">G(s)</div><div class="bo-label">Planta</div><div class="bo-desc">Função de transferência</div></div>
<div class="block-option" onclick="pickBlock('ss')"><div class="bo-icon" style="color:var(--grn)">SS</div><div class="bo-label">Espaço de Estados</div><div class="bo-desc">Matrizes A,B,C,D livres</div></div>
<div class="block-option" onclick="pickBlock('gain')"><div class="bo-icon" style="color:var(--pur)">K</div><div class="bo-label">Ganho</div><div class="bo-desc">Ganho constante</div></div>
<div class="block-option" onclick="pickBlock('pid')"><div class="bo-icon" style="color:var(--pur)">PID</div><div class="bo-label">Controlador PID</div><div class="bo-desc">Kp + Ki/s + Kd*s</div></div>
<div class="block-option" onclick="pickBlock('int')"><div class="bo-icon" style="color:var(--yel)">1/s</div><div class="bo-label">Integrador</div><div class="bo-desc">Integração pura</div></div>
<div class="block-option" onclick="pickBlock('der')"><div class="bo-icon" style="color:#aaa">s</div><div class="bo-label">Derivador</div><div class="bo-desc">Derivação pura</div></div>
<div class="block-option" onclick="pickBlock('actuator')"><div class="bo-icon" style="color:var(--blu)">A(s)</div><div class="bo-label">Atuador</div><div class="bo-desc">Dinâmica do atuador</div></div>
</div>
<div class="cat-label">Realimentação</div>
<div class="block-grid">
<div class="block-option" onclick="pickBlock('sensor')"><div class="bo-icon" style="color:var(--pnk)">H(s)</div><div class="bo-label">Sensor</div><div class="bo-desc">Malha de realimentação</div></div>
<div class="block-option" onclick="pickBlock('sum')"><div class="bo-icon" style="color:var(--grn)">&Sigma;</div><div class="bo-label">Somador</div><div class="bo-desc">Soma/subtração de sinais</div></div>
<div class="block-option" onclick="pickBlock('branch')"><div class="bo-icon" style="color:var(--blu)">&bull;</div><div class="bo-label">Ramificação</div><div class="bo-desc">Divide sinal em dois</div></div>
</div>
</div>
<div class="cfg-panel" id="cfgPanel">
<h4 id="cfgTitle">Configurar bloco</h4>
<div id="cfgFields"></div>
<button class="cfg-btn" onclick="confirmAdd()">Adicionar ao Diagrama</button>
<button class="cfg-btn" style="background:var(--sf2);margin-top:4px;border:1px solid var(--bd)" onclick="backToGrid()">Voltar</button>
</div>
</div></div></div>

<div class="modal-overlay" id="connModal">
<div class="modal">
<div class="modal-hdr"><h3 id="connModalTitle">Conectar Blocos</h3><button class="modal-close" onclick="closeConnModal()">&times;</button></div>
<div class="modal-body">
<p style="font-size:12px;color:var(--txm);margin-bottom:12px" id="connModalHint">Selecione os blocos na ordem desejada.</p>
<div id="connBlockList" style="max-height:280px;overflow-y:auto;padding:4px"></div>
<p id="connSelCount" style="font-size:11px;color:var(--txm);margin:8px 0 4px;text-align:center">0 blocos selecionados</p>
<div style="margin-top:8px;display:flex;gap:8px">
<button class="cfg-btn" style="padding:12px 28px;font-size:14px" onclick="applyConn()">&#10003; Aplicar Conexão</button>
<button class="cfg-btn" style="background:var(--sf2);border:1px solid var(--bd);padding:12px 20px" onclick="closeConnModal()">Cancelar</button>
</div>
</div></div></div>

<div class="toolbar" id="diag-toolbar">
<button class="tb tb-add" onclick="openModal()">+ Adicionar Bloco</button>
<div class="sep"></div>
<span class="lbl">Rápido:</span>
<button class="tb" style="background:#162038;border-color:#2d558a;color:var(--blu)" data-add="tf">G(s)</button>
<button class="tb" style="background:#1a3d2a;border-color:#2d8a50;color:var(--grn)" onclick="pickBlock('ss')">SS</button>
<button class="tb" style="background:#201638;border-color:#5a2d8a;color:var(--pur)" data-add="gain">K</button>
<button class="tb" data-add="sum">&Sigma;</button>
<button class="tb" style="background:#381628;border-color:#8a2d5a;color:var(--pnk)" data-add="sensor">H(s)</button>
<div class="sep"></div>
<span class="lbl">Conexão:</span>
<button class="tb" style="background:#0e2a1a;border-color:#2d8a55;color:#34d399" id="btnSérie" onclick="openConnModal('serie')">Série</button>
<button class="tb" style="background:#1a1040;border-color:#6d5acd;color:#a78bfa" id="btnParalelo" onclick="openConnModal('paralelo')">Paralelo</button>
<button class="tb" style="background:#2a1020;border-color:#8a2d50;color:#f472b6" id="btnFbNeg" onclick="openConnModal('fb_neg')">FB -</button>
<button class="tb" style="background:#2a2010;border-color:#8a7a2d;color:#fbbf24" id="btnFbPos" onclick="openConnModal('fb_pos')">FB +</button>
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
<span style="color:var(--txm)">Conexões:</span><span id="pE" style="font-weight:600">0</span>
</div></div>
<div class="panel-section"><h4>Parâmetros</h4><div id="pA"><span class="hint">Selecione um bloco.</span></div></div>
<div class="panel-section"><h4>Dicas</h4><div class="hint">
<b>Conectar:</b> porta <span style="color:var(--blu)">azul</span> &rarr; porta <span style="color:var(--grn)">verde</span><br>
<b>Calcular:</b> botão verde abaixo<br><b>Mobile:</b> toque e arraste!
</div></div></div></div>

<div class="mode-bar">
<button class="mode-btn active" id="modeDiag" onclick="setMode('diag')">&#9638; Diagrama de Blocos</button>
<button class="mode-btn" id="modeMan" onclick="setMode('manual')">&#9998; Entrada Manual</button>
</div>

<div class="manual-sec" id="manualSec">
<div class="man-tabs">
<button class="man-tab active" id="subDirect" onclick="setSubMode('direct')">T(s) Direta</button>
<button class="man-tab" id="subClosed" onclick="setSubMode('closed')">Malha Fechada</button>
<button class="man-tab" id="subOpen" onclick="setSubMode('open')">Malha Aberta G*H</button>
<button class="man-tab" id="subSS" onclick="setSubMode('ss')">Espaço de Estados</button>
</div>
<div id="manDirect">
<h4>Função de Transferência T(s) = Num / Den</h4>
<div class="man-row">
<div class="pg"><label>Numerador</label><input id="manNum" value="1" placeholder="ex: s+1"></div>
<div class="pg"><label>Denominador</label><input id="manDen" value="s^2+2s+1" placeholder="ex: s^2+2s+1"></div>
</div>
<div class="man-hint">Formato: <code>s^2+3s+1</code> ou <code>2s^3+s+5</code>. Use <code>^</code> para potências.</div>
</div>
<div id="manClosed" style="display:none">
<div class="man-row" style="margin-bottom:8px">
<div class="pg"><label>Tipo de Realimentação</label>
<select id="manFbType" style="width:100%;padding:8px;background:var(--sf2);border:1px solid var(--bd);border-radius:6px;color:var(--tx);font-size:13px" onchange="updateFbLabel()">
<option value="neg">Negativa: T(s) = G / (1 + G&middot;H)</option>
<option value="pos">Positiva: T(s) = G / (1 - G&middot;H)</option>
</select></div>
</div>
<h4 id="manClosedLabel">Malha Fechada: T(s) = G(s) / (1 + G(s)&middot;H(s))</h4>
<div class="man-row">
<div class="pg"><label>G(s) Numerador</label><input id="manGN" value="10" placeholder="ex: 10"></div>
<div class="pg"><label>G(s) Denominador</label><input id="manGD" value="s^2+3s+1" placeholder="ex: s^2+3s+1"></div>
</div>
<div class="man-row">
<div class="pg"><label>H(s) Numerador</label><input id="manHN" value="1" placeholder="ex: 1"></div>
<div class="pg"><label>H(s) Denominador</label><input id="manHD" value="1" placeholder="ex: s+1"></div>
</div>
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
</div>
<div id="manSS" style="display:none">
<h4>Espaço de Estados: dx/dt = Ax + Bu, y = Cx + Du</h4>
<div class="man-row">
<div class="pg"><label>Matriz A (nxn)</label><input id="manA" value="0 1; -2 -3" placeholder="0 1; -2 -3"></div>
<div class="pg"><label>Matriz B (nx1)</label><input id="manB" value="0; 1" placeholder="0; 1"></div>
</div>
<div class="man-row">
<div class="pg"><label>Matriz C (1xn)</label><input id="manC" value="1 0" placeholder="1 0"></div>
<div class="pg"><label>Matriz D (1x1)</label><input id="manD" value="0" placeholder="0"></div>
</div>
<div class="man-hint">Use <code>;</code> para separar linhas. Ex: <code>0 1; -2 -3</code> = matriz 2x2.</div>
</div>
</div>

<div class="calc-bar"><button id="btnCalcMain" onclick="onCalc()">&#9654; CALCULAR DIAGRAMA</button></div>

<div class="results" id="res"><div class="results-hdr"><h3>Resultados</h3>
<button onclick="document.getElementById('res').classList.remove('vis')" style="background:none;border:1px solid var(--bd);color:var(--txm);border-radius:6px;padding:4px 10px;cursor:pointer;font-size:12px">Fechar</button>
</div><div class="rbody" id="rb"></div></div></div>

<script>
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
  var titles={tf:"Planta G(s)",ss:"Espaço de Estados",gain:"Ganho K",pid:"Controlador PID",sensor:"Sensor H(s)",sum:"Somador",actuator:"Atuador A(s)"};
  document.getElementById("cfgTitle").textContent="Configurar: "+(titles[t]||t);
  if(t==="tf"||t==="sensor"||t==="actuator"){
    h+='<div class="cfg-row"><div><label>Numerador</label><input id="cfgNum" value="1" placeholder="ex: s+1"></div>';
    h+='<div><label>Denominador</label><input id="cfgDen" value="s+1" placeholder="ex: s^2+2s+1"></div></div>'}
  else if(t==="ss"){
    h+='<div class="cfg-row"><div><label>Matriz A (nxn)</label><input id="cfgSSA" value="0 1; -2 -3" placeholder="0 1; -2 -3"></div>';
    h+='<div><label>Matriz B (nx1)</label><input id="cfgSSB" value="0; 1" placeholder="0; 1"></div></div>';
    h+='<div class="cfg-row"><div><label>Matriz C (1xn)</label><input id="cfgSSC" value="1 0" placeholder="1 0"></div>';
    h+='<div><label>Matriz D (1x1)</label><input id="cfgSSD" value="0" placeholder="0"></div></div>'}
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
  else if(t==="ss"){p.ssA=(document.getElementById("cfgSSA")||{}).value||"0";p.ssB=(document.getElementById("cfgSSB")||{}).value||"0";p.ssC=(document.getElementById("cfgSSC")||{}).value||"1";p.ssD=(document.getElementById("cfgSSD")||{}).value||"0"}
  else if(t==="gain"){p.k=(document.getElementById("cfgK")||{}).value||"1"}
  else if(t==="pid"){p.kp=(document.getElementById("cfgKp")||{}).value||"1";p.ki=(document.getElementById("cfgKi")||{}).value||"0";p.kd=(document.getElementById("cfgKd")||{}).value||"0"}
  else if(t==="sum"){p.signs=(document.getElementById("cfgSigns")||{}).value||"+ -"}
  addBP(t,p);closeModal()}

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
function pfC(c){return{n:[c],d:[1]}}
function pfZ(a){var t=pTrim(a.n);return t.length===1&&Math.abs(t[0])<1e-14}
function pfAdd(a,b){return pfReduce({n:pAdd(pMul(a.n,b.d),pMul(b.n,a.d)),d:pMul(a.d,b.d)})}
function pfSub(a,b){return pfReduce({n:pSub(pMul(a.n,b.d),pMul(b.n,a.d)),d:pMul(a.d,b.d)})}
function pfMul(a,b){return pfReduce({n:pMul(a.n,b.n),d:pMul(a.d,b.d)})}
function pfDiv(a,b){return pfReduce({n:pMul(a.n,b.d),d:pMul(a.d,b.n)})}
function polyRem(a,b){a=a.slice();b=pTrim(b);if(b.length>a.length)return pTrim(a);var mx=0;for(var i=0;i<a.length;i++)mx=Math.max(mx,Math.abs(a[i]));for(var i=a.length-1;i>=b.length-1;i--){var c=a[i]/b[b.length-1];for(var j=0;j<b.length;j++)a[j+i-b.length+1]-=c*b[j]}var tol=1e-10*Math.max(1,mx);var r=a.slice(0,b.length-1).map(function(v){return Math.abs(v)<tol?0:v});return pTrim(r)}
function polyGCD(a,b){a=pTrim(a);b=pTrim(b);if(a.length<b.length){var t=a;a=b;b=t}for(var it=0;it<50;it++){b=pTrim(b);if(b.length<=1){var mx2=0;for(var i=0;i<a.length;i++)mx2=Math.max(mx2,Math.abs(a[i]));if(Math.abs(b[0])<1e-8*Math.max(1,mx2))break;return[1]}var r=polyRem(a,b);var mr=0;for(var i=0;i<r.length;i++)mr=Math.max(mr,Math.abs(r[i]));var mb=0;for(var i=0;i<b.length;i++)mb=Math.max(mb,Math.abs(b[i]));if(mr<1e-8*Math.max(1,mb)){a=b;break}a=b;b=r}a=pTrim(a);var lc=a[a.length-1];if(Math.abs(lc)>1e-14)a=a.map(function(c){return c/lc});return pTrim(a)}
function polyDivQ(a,b){a=a.slice();b=pTrim(b);if(b.length===1)return pTrim(a.map(function(c){return c/b[0]}));if(b.length>a.length)return[0];var q=new Array(a.length-b.length+1);for(var i=a.length-1;i>=b.length-1;i--){var c=a[i]/b[b.length-1];q[i-b.length+1]=c;for(var j=0;j<b.length;j++)a[j+i-b.length+1]-=c*b[j]}return pTrim(q)}
function pfReduce(pf){var n=pTrim(pf.n),d=pTrim(pf.d);var g=polyGCD(n,d);if(g.length<=1)return{n:n,d:d};return{n:pTrim(polyDivQ(n,g)),d:pTrim(polyDivQ(d,g))}}
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
function parseMat(s){s=(s||"").trim();var rows=s.split(";");return rows.map(function(r){return r.trim().split(/[\s,]+/).map(Number)})}
function ssToTF(As,Bs,Cs,Ds){
  var A=parseMat(As),B=parseMat(Bs),C=parseMat(Cs),D=parseMat(Ds);
  var n=A.length;
  if(n===0)return{n:[D[0]?D[0][0]:0],d:[1]};
  if(n===1){var a00=A[0][0],b0=B[0][0],c0=C[0][0],d0=D[0]?D[0][0]:0;return{n:pAdd([d0*(-a00)+c0*b0],[d0]),d:[-a00,1]}}
  if(n===2){var tr=A[0][0]+A[1][1],dt=A[0][0]*A[1][1]-A[0][1]*A[1][0];var den=[dt,-tr,1];var adj00=[-A[1][1],1],adj01=[A[0][1]],adj10=[A[1][0]],adj11=[-A[0][0],1];var v0=pAdd(pScl(adj00,B[0][0]),pScl(adj01,B[1][0]));var v1=pAdd(pScl(adj10,B[0][0]),pScl(adj11,B[1][0]));var num=pAdd(pScl(v0,C[0][0]),pScl(v1,C[0].length>1?C[0][1]:0));var d0v=D[0]?D[0][0]:0;if(Math.abs(d0v)>1e-14)num=pAdd(num,pMul([d0v],den));return pfReduce({n:num,d:den})}
  var cp=[1];var Mk=[];for(var i=0;i<n;i++){Mk.push([]);for(var j=0;j<n;j++)Mk[i].push(0)}
  for(var k=1;k<=n;k++){var prev=Mk;var AM=[];for(var i=0;i<n;i++){AM.push([]);for(var j=0;j<n;j++){var s2=0;for(var l=0;l<n;l++)s2+=A[i][l]*prev[l][j];AM[i].push(s2)}}if(k===1)for(var i=0;i<n;i++)for(var j=0;j<n;j++)AM[i][j]+=(i===j?1:0);var tk=0;for(var i=0;i<n;i++)tk+=AM[i][i];var ck=-tk/k;cp.unshift(ck);Mk=[];for(var i=0;i<n;i++){Mk.push([]);for(var j=0;j<n;j++)Mk[i].push(AM[i][j]+(i===j?ck:0))}}
  var den=cp;var pts=[];for(var k=0;k<=n;k++){var sv=k*10+1;var sIA=[];for(var i=0;i<n;i++){sIA.push([]);for(var j=0;j<n;j++)sIA[i].push((i===j?sv:0)-A[i][j])}var aug=[];for(var i=0;i<n;i++){aug.push([]);for(var j=0;j<2*n;j++)aug[i].push(j<n?sIA[i][j]:(i===j-n?1:0))}for(var c2=0;c2<n;c2++){var mx2=c2;for(var r=c2+1;r<n;r++)if(Math.abs(aug[r][c2])>Math.abs(aug[mx2][c2]))mx2=r;var tmp=aug[c2];aug[c2]=aug[mx2];aug[mx2]=tmp;var piv=aug[c2][c2];if(Math.abs(piv)<1e-30)continue;for(var j=0;j<2*n;j++)aug[c2][j]/=piv;for(var r=0;r<n;r++){if(r===c2)continue;var f=aug[r][c2];for(var j=0;j<2*n;j++)aug[r][j]-=f*aug[c2][j]}}var inv=[];for(var i=0;i<n;i++){inv.push([]);for(var j=0;j<n;j++)inv[i].push(aug[i][j+n])}var CiB=0;for(var i=0;i<(C[0]||[]).length;i++)for(var j=0;j<(B[0]||[]).length;j++){var v3=0;for(var l=0;l<n;l++)v3+=inv[i][l]*B[l][j];CiB+=C[0][i]*v3}var d0v=D[0]?D[0][0]:0;pts.push({s:sv,v:CiB+d0v*pEv(den,sv)})}
  var V=[];for(var i=0;i<=n;i++){V.push([]);for(var j=0;j<=n;j++)V[i].push(Math.pow(pts[i].s,j))}var rhs=pts.map(function(p){return p.v});
  for(var c2=0;c2<=n;c2++){var mx2=c2;for(var r=c2+1;r<=n;r++)if(Math.abs(V[r][c2])>Math.abs(V[mx2][c2]))mx2=r;var tmp=V[c2];V[c2]=V[mx2];V[mx2]=tmp;var tmp2=rhs[c2];rhs[c2]=rhs[mx2];rhs[mx2]=tmp2;var piv=V[c2][c2];if(Math.abs(piv)<1e-30)continue;for(var j=c2;j<=n;j++)V[c2][j]/=piv;rhs[c2]/=piv;for(var r=0;r<=n;r++){if(r===c2)continue;var f=V[r][c2];for(var j=c2;j<=n;j++)V[r][j]-=f*V[c2][j];rhs[r]-=f*rhs[c2]}}
  var num=rhs.map(function(v){return Math.abs(v)<1e-10?0:v});return pfReduce({n:pTrim(num),d:pTrim(den)})}
function fN(n){if(Math.abs(n-Math.round(n))<1e-8)return Math.round(n).toString();return n.toFixed(4).replace(/0+$/,"").replace(/\.$/,"")}
function fP(c){c=pTrim(c);var ts=[];for(var i=c.length-1;i>=0;i--){var v=c[i];if(Math.abs(v)<1e-10)continue;var t;if(i===0)t=fN(v);else if(i===1){t=Math.abs(v-1)<1e-10?"s":Math.abs(v+1)<1e-10?"-s":fN(v)+"s"}else{t=Math.abs(v-1)<1e-10?"s^"+i:Math.abs(v+1)<1e-10?"-s^"+i:fN(v)+"s^"+i}ts.push(t)}if(!ts.length)return"0";var s=ts[0];for(var i=1;i<ts.length;i++)s+=ts[i][0]==="-"?" - "+ts[i].slice(1):" + "+ts[i];return s}
function bTF(nd){var p=nd.params||{},t=nd.type;
  if(t==="tf"||t==="sensor"||t==="actuator")return{n:parseP(p.num||"1"),d:parseP(p.den||"1")};
  if(t==="ss")return ssToTF(p.ssA||"0",p.ssB||"0",p.ssC||"1",p.ssD||"0");
  if(t==="gain")return pfC(parseFloat(p.k)||1);
  if(t==="int")return{n:[1],d:[0,1]};if(t==="der")return{n:[0,1],d:[1]};
  if(t==="pid"){var kp=parseFloat(p.kp)||0,ki=parseFloat(p.ki)||0,kd=parseFloat(p.kd)||0;if(Math.abs(ki)<1e-14&&Math.abs(kd)<1e-14)return pfC(kp||1);if(Math.abs(ki)<1e-14)return{n:[kp,kd],d:[1]};if(Math.abs(kd)<1e-14)return{n:[ki,kp],d:[0,1]};return{n:[ki,kp,kd],d:[0,1]}}
  return pfC(1)}
function solve(nodes,edges){
  if(!nodes.length)return{e:"Adicione blocos."};
  var inp=null,out=null;nodes.forEach(function(n){if(n.type==="input")inp=n;if(n.type==="output")out=n});
  if(!inp)return{e:"Adicione Entrada R(s)."};if(!out)return{e:"Adicione Saída Y(s)."};if(!edges.length)return{e:"Conecte os blocos."};
  var N=nodes.length,ix={};nodes.forEach(function(n,i){ix[n.id]=i});
  var A=[],b=[];for(var i=0;i<N;i++){A.push([]);for(var j=0;j<N;j++)A[i].push(pfC(0));b.push(pfC(0))}
  for(var i=0;i<N;i++){var nd=nodes[i];A[i][i]=pfC(1);if(nd.type==="input"){b[i]=pfC(1);continue}var inc=edges.filter(function(e){return e.dst===nd.id});
    if(nd.type==="sum"){var sg=((nd.params||{}).signs||"+ -").trim().split(/\s+/);inc.forEach(function(e){var si=ix[e.src];if(si===undefined)return;var pi=parseInt((e.dstPort||"in0").replace("in",""))||0;var sign=sg[pi]==="-"?-1:1;A[i][si]=pfSub(A[i][si],pfC(sign))})}
    else{var tf=bTF(nd);inc.forEach(function(e){var si=ix[e.src];if(si===undefined)return;A[i][si]=pfSub(A[i][si],tf)})}}
  for(var c=0;c<N;c++){var pv=-1;for(var r=c;r<N;r++)if(!pfZ(A[r][c])){pv=r;break}if(pv<0)return{e:"Sistema singular - verifique as conexões."};if(pv!==c){var t=A[c];A[c]=A[pv];A[pv]=t;var t2=b[c];b[c]=b[pv];b[pv]=t2}for(var r=c+1;r<N;r++){if(pfZ(A[r][c]))continue;var f=pfDiv(A[r][c],A[c][c]);for(var j=c;j<N;j++)A[r][j]=pfSub(A[r][j],pfMul(f,A[c][j]));b[r]=pfSub(b[r],pfMul(f,b[c]))}}
  var x=[];for(var i=0;i<N;i++)x.push(pfC(0));for(var i=N-1;i>=0;i--){var s=b[i];for(var j=i+1;j<N;j++)s=pfSub(s,pfMul(A[i][j],x[j]));x[i]=pfDiv(s,A[i][i])}
  var oi=ix[out.id],tf={n:pTrim(x[oi].n),d:pTrim(x[oi].d)};
  if(tf.d[tf.d.length-1]<0){tf.n=pScl(tf.n,-1);tf.d=pScl(tf.d,-1)}var lc=tf.d[tf.d.length-1];if(Math.abs(lc)>1e-14&&Math.abs(lc-1)>1e-10){tf.n=pScl(tf.n,1/lc);tf.d=pScl(tf.d,1/lc)}
  tf=pfReduce(tf);return{tf:tf}}
function roots(poly){poly=pTrim(poly);var n=poly.length-1;if(n<=0)return[];var lc=poly[n],p=poly.map(function(c){return c/lc});if(n===1)return[{r:-p[0],i:0}];if(n===2){var bb=p[1],cc=p[0],d=bb*bb-4*cc;if(d>=0)return[{r:(-bb+Math.sqrt(d))/2,i:0},{r:(-bb-Math.sqrt(d))/2,i:0}];return[{r:-bb/2,i:Math.sqrt(-d)/2},{r:-bb/2,i:-Math.sqrt(-d)/2}]}var rs=[];for(var i=0;i<n;i++){var a=2*Math.PI*i/n+.4;rs.push({r:(1+.5*i/n)*Math.cos(a),i:(1+.5*i/n)*Math.sin(a)})}for(var it=0;it<1500;it++){var mx=0;for(var i=0;i<n;i++){var v=cEvP(p,rs[i]),pr={r:1,i:0};for(var j=0;j<n;j++)if(j!==i)pr=cMul(pr,{r:rs[i].r-rs[j].r,i:rs[i].i-rs[j].i});if(cAbs(pr)<1e-30)continue;var dl=cDiv(v,pr);rs[i].r-=dl.r;rs[i].i-=dl.i;mx=Math.max(mx,cAbs(dl))}if(mx<1e-12)break}rs.forEach(function(r){if(Math.abs(r.i)<1e-8)r.i=0});return rs}
var SN=14,SV=[];(function(){for(var i=1;i<=SN;i++){var s=0,k0=Math.floor((i+1)/2),k1=Math.min(i,SN/2);for(var k=k0;k<=k1;k++){var f=function(n){var r=1;for(var i=2;i<=n;i++)r*=i;return r};s+=Math.pow(k,SN/2)*f(2*k)/(f(SN/2-k)*f(k)*f(k-1)*f(i-k)*f(2*k-i))}SV.push(Math.pow(-1,SN/2+i)*s)}})();
function stepResp(tf,tMax,nP){var ln2=Math.LN2,ts=[],ys=[];for(var i=0;i<nP;i++){var t=(i+1)*tMax/nP;ts.push(t);var s=0;for(var j=0;j<SN;j++){var sv=(j+1)*ln2/t,nv=pEv(tf.n,sv),dv=pEv(tf.d,sv)*sv;if(Math.abs(dv)<1e-30)continue;s+=SV[j]*nv/dv}ys.push(s*ln2/t)}return{t:ts,y:ys}}
function forceResp(tf,sig,tMax,nP){var ln2=Math.LN2,ts=[],ys=[],us=[];var om=2*Math.PI;for(var i=0;i<nP;i++){var t=(i+1)*tMax/nP;ts.push(t);var s=0;for(var j=0;j<SN;j++){var sv=(j+1)*ln2/t,nv=pEv(tf.n,sv),dv=pEv(tf.d,sv);if(Math.abs(dv)<1e-30)continue;var uv;if(sig==='rampa')uv=1/(sv*sv);else if(sig==='impulso')uv=1;else if(sig==='parabolica')uv=2/(sv*sv*sv);else if(sig==='senoidal')uv=om/(sv*sv+om*om);else uv=1/sv;var contrib=SV[j]*nv*uv/dv;if(isFinite(contrib))s+=contrib}var yv=s*ln2/t;ys.push(isFinite(yv)?yv:0);if(sig==='rampa')us.push(t);else if(sig==='senoidal')us.push(Math.sin(om*t));else if(sig==='impulso')us.push(i===0?nP/tMax:0);else if(sig==='parabolica')us.push(t*t);else us.push(1)}return{t:ts,y:ys,u:us}}
function nyq(tf,wMin,wMax,nP){var re=[],im=[];var lm=Math.log10(wMin),lx=Math.log10(wMax);for(var i=0;i<nP;i++){var w=Math.pow(10,lm+i*(lx-lm)/(nP-1));var jw={r:0,i:w},nc=cEvP(tf.n,jw),dc=cEvP(tf.d,jw),T=cDiv(nc,dc);re.push(T.r);im.push(T.i)}return{re:re,im:im}}
function lgr(tf,nK){var kMax=200,branches=[];var nt=pTrim(tf.n),dt=pTrim(tf.d);var np=dt.length-1;if(np<=0)return branches;for(var i=0;i<np;i++)branches.push({re:[],im:[]});var prev=null;for(var ki=0;ki<nK;ki++){var k=ki*kMax/(nK-1);var cp=pAdd(dt,pScl(nt,k));var rs=roots(cp);if(prev&&prev.length===rs.length){var used=[],matched=[];for(var i=0;i<prev.length;i++){var best=-1,bd=Infinity;for(var j=0;j<rs.length;j++){if(used.indexOf(j)>=0)continue;var d=Math.sqrt(Math.pow(rs[j].r-prev[i].r,2)+Math.pow(rs[j].i-prev[i].i,2));if(d<bd){bd=d;best=j}}used.push(best);matched.push(rs[best])}rs=matched}for(var i=0;i<Math.min(rs.length,np);i++){branches[i].re.push(rs[i].r);branches[i].im.push(rs[i].i)}prev=rs.slice(0,np)}return branches}
var _PT={paper_bgcolor:'#0e1117',plot_bgcolor:'#0e1117',font:{color:'#8890b0',family:'system-ui,sans-serif',size:12},margin:{l:55,r:15,t:35,b:45},dragmode:'zoom',modebar:{add:['drawline','drawopenpath','eraseshape']},newshape:{line:{color:'#34d399',width:2,dash:'dash'}}};
function _pAx(title,extra){var o={title:title,gridcolor:'#252840',zerolinecolor:'#555',linecolor:'#333654'};if(extra)for(var k in extra)o[k]=extra[k];return o}
function chartLGR(id,branches,tf){var el=document.getElementById(id);if(!el)return;var traces=[],cols=['#5b6be0','#60a5fa','#a78bfa','#f472b6','#fbbf24'];branches.forEach(function(br,bi){traces.push({x:br.re,y:br.im,mode:'lines',line:{color:cols[bi%cols.length],width:1.5},name:'Ramo '+(bi+1),showlegend:false,hovertemplate:'Real: %{x:.3f}<br>Imag: %{y:.3f}<extra></extra>'})});var ps2=roots(tf.d);if(ps2.length)traces.push({x:ps2.map(function(p){return p.r}),y:ps2.map(function(p){return p.i}),mode:'markers',marker:{color:'#ff4444',size:12,symbol:'x'},name:'Polos',hovertemplate:'Polo<br>Real: %{x:.3f}<br>Imag: %{y:.3f}<extra></extra>'});var zs2=roots(tf.n);if(zs2.length)traces.push({x:zs2.map(function(z){return z.r}),y:zs2.map(function(z){return z.i}),mode:'markers',marker:{color:'#44ff44',size:10,symbol:'circle-open',line:{width:2,color:'#44ff44'}},name:'Zeros',hovertemplate:'Zero<br>Real: %{x:.3f}<br>Imag: %{y:.3f}<extra></extra>'});var lay=Object.assign({},_PT,{xaxis:_pAx('Parte Real'),yaxis:_pAx('Parte Imaginaria'),height:380,showlegend:true,legend:{x:0.01,y:0.99,bgcolor:'rgba(0,0,0,0.3)',font:{color:'#e0e4f0'}},hovermode:'closest',shapes:[{type:'line',x0:0,x1:0,y0:-1e6,y1:1e6,line:{color:'#555',width:1,dash:'dash'}},{type:'line',x0:-1e6,x1:1e6,y0:0,y1:0,line:{color:'#555',width:1,dash:'dash'}}]});Plotly.newPlot(el,traces,lay,{responsive:true})}
function chartPZ(id,ps,zs){var el=document.getElementById(id);if(!el)return;var traces=[];if(zs.length)traces.push({x:zs.map(function(z){return z.r}),y:zs.map(function(z){return z.i}),mode:'markers',marker:{color:'#60a5fa',size:12,symbol:'circle-open',line:{width:2,color:'#60a5fa'}},name:'Zeros',hovertemplate:'Zero<br>Real: %{x:.3f}<br>Imag: %{y:.3f}<extra></extra>'});if(ps.length)traces.push({x:ps.map(function(p){return p.r}),y:ps.map(function(p){return p.i}),mode:'markers',marker:{color:'#ff4444',size:12,symbol:'x'},name:'Polos',hovertemplate:'Polo<br>Real: %{x:.3f}<br>Imag: %{y:.3f}<extra></extra>'});var lay=Object.assign({},_PT,{xaxis:_pAx('Parte Real'),yaxis:_pAx('Parte Imaginaria'),height:350,showlegend:true,legend:{x:0.01,y:0.99,bgcolor:'rgba(0,0,0,0.3)',font:{color:'#e0e4f0'}},hovermode:'closest',shapes:[{type:'line',x0:0,x1:0,y0:-1e6,y1:1e6,line:{color:'#555',width:1,dash:'dash'}},{type:'line',x0:-1e6,x1:1e6,y0:0,y1:0,line:{color:'#555',width:1,dash:'dash'}}]});Plotly.newPlot(el,traces,lay,{responsive:true})}
function bode(tf,wMin,wMax,nP){var fs=[],ms=[],ps=[],lm=Math.log10(wMin),lx=Math.log10(wMax);for(var i=0;i<nP;i++){var w=Math.pow(10,lm+i*(lx-lm)/(nP-1));fs.push(w);var jw={r:0,i:w},nc=cEvP(tf.n,jw),dc=cEvP(tf.d,jw),T=cDiv(nc,dc),mg=cAbs(T);ms.push(mg>1e-30?20*Math.log10(mg):-600);ps.push(Math.atan2(T.i,T.r)*180/Math.PI)}for(var i=1;i<ps.length;i++){while(ps[i]-ps[i-1]>180)ps[i]-=360;while(ps[i]-ps[i-1]<-180)ps[i]+=360}return{w:fs,m:ms,p:ps}}
function autoT(tf){var ps=roots(tf.d);if(!ps.length)return 20;var temInstável=ps.some(function(p){return p.r>1e-4});if(temInstável)return 15;var temZero=ps.some(function(p){return Math.abs(p.r)<1e-6&&Math.abs(p.i)<1e-6});if(temZero)return 50;var m=Infinity;ps.forEach(function(p){if(Math.abs(p.r)>1e-6)m=Math.min(m,Math.abs(p.r))});return m===Infinity?20:Math.min(200,Math.max(5,8/m))}
function autoW(tf){var ps=roots(tf.d).concat(roots(tf.n));var fs=ps.map(function(p){return Math.sqrt(p.r*p.r+p.i*p.i)}).filter(function(f){return f>1e-6});if(!fs.length)return{a:.01,b:1000};var wMin=Math.max(1e-3,Math.min.apply(null,fs)/100);var wMax=Math.min(1e5,Math.max.apply(null,fs)*100);if(Math.log10(wMax/wMin)<4)wMax=wMin*1e4;return{a:wMin,b:wMax}}
function perf(t,y){if(!y.length)return{};var l=y.slice(Math.floor(y.length*.9)),yf=l.reduce(function(a,b){return a+b},0)/l.length;var ym=Math.max.apply(null,y),os=Math.abs(yf)>1e-6?Math.max(0,(ym-yf)/Math.abs(yf)*100):0;var t10=null,t90=null;if(yf>0)for(var i=0;i<y.length;i++){if(t10===null&&y[i]>=yf*.1)t10=t[i];if(t90===null&&y[i]>=yf*.9){t90=t[i];break}}var tr=t10!==null&&t90!==null?t90-t10:NaN,ts2=NaN;if(Math.abs(yf)>1e-6)for(var i=y.length-1;i>=0;i--)if(Math.abs(y[i]-yf)>.02*Math.abs(yf)){ts2=i<y.length-1?t[i+1]:t[i];break}return{"Valor Final":fN(yf),"Sobressinal":fN(os)+"%","T. Subida":isNaN(tr)?"N/A":fN(tr)+"s","T. Acomod.":isNaN(ts2)?"N/A":fN(ts2)+"s","Pico":fN(ym)}}
function fC(c){if(Math.abs(c.i)<1e-8)return fN(c.r);return fN(c.r)+(c.i>=0?" + ":" - ")+fN(Math.abs(c.i))+"j"}
function esc(s){return s.replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;")}
var curSig='degrau',curMalha='aberta',curFbSign='neg';
var selAn={tempo:true,desemp:true,pz:true,bm:true,bp:true,nyqst:true,lgr:true};
var _lastTF=null;
function reRender(){if(_lastTF)showRes(_lastTF)}
function calcGM(bd){for(var i=1;i<bd.p.length;i++){if(bd.p[i-1]>-180&&bd.p[i]<=-180){var w=(bd.p[i-1]+180)/(bd.p[i-1]-bd.p[i]);var mag=bd.m[i-1]+w*(bd.m[i]-bd.m[i-1]);return -mag}}return Infinity}
function calcPM(bd){for(var i=1;i<bd.m.length;i++){if(bd.m[i-1]>=0&&bd.m[i]<0){var w=bd.m[i-1]/(bd.m[i-1]-bd.m[i]);var ph=bd.p[i-1]+w*(bd.p[i]-bd.p[i-1]);return ph+180}}return Infinity}
function showRes(tf){
  _lastTF=tf;var rd=document.getElementById("res"),rb=document.getElementById("rb");rd.classList.add("vis");
  var tfAnálise=tf;var ftmfErro=null;
  if(curMalha==='fechada'){try{var ftmfNum=pTrim(tf.n.slice());var ftmfDen=curFbSign==='pos'?pTrim(pSub(tf.d,tf.n)):pTrim(pAdd(tf.d,tf.n));var lc2=ftmfDen[ftmfDen.length-1];if(Math.abs(lc2)<1e-10){ftmfErro="Denominador da FTMF degenerado.";tfAnálise=tf;}else{var reduced=pfReduce({n:ftmfNum,d:ftmfDen});if(Math.abs(lc2-1)>1e-10){reduced.n=pScl(reduced.n,1/lc2);reduced.d=pScl(reduced.d,1/lc2)}tfAnálise=reduced;}}catch(err){ftmfErro="Erro ao calcular FTMF: "+String(err);tfAnálise=tf;}}
  var ns=fP(tfAnálise.n),ds=fP(tfAnálise.d);var ps=roots(tfAnálise.d),zs=roots(tfAnálise.n),stb=ps.every(function(p){return p.r<1e-6});var tM=autoT(tfAnálise);var sr=forceResp(tfAnálise,curSig,tM,400);var pf=perf(sr.t,sr.y);var wr=autoW(tfAnálise),bd=bode(tfAnálise,wr.a,wr.b,600);var nqd=nyq(tfAnálise,wr.a,wr.b,400);var lgrData=lgr(tf,300);
  var sigNomes={degrau:'Degrau',rampa:'Rampa',senoidal:'Senoidal',impulso:'Impulso',parabolica:'Parabólica'};
  var h='';var ss='background:var(--sf2);border:1px solid var(--bd);border-radius:6px;color:var(--tx);padding:5px 10px;font-size:12px';
  h+='<div class="rcard"><h4>Configurações de Análise</h4>';
  h+='<div style="display:flex;flex-wrap:wrap;align-items:center;gap:12px;margin-bottom:12px">';
  h+='<div style="display:flex;align-items:center;gap:6px"><span style="font-size:12px;color:var(--txm);font-weight:600">Tipo de Malha:</span>';
  h+='<select onchange="curMalha=this.value;reRender()" style="'+ss+';font-weight:600">';
  h+='<option value="aberta"'+(curMalha==='aberta'?' selected':'')+'>Malha Aberta</option>';
  h+='<option value="fechada"'+(curMalha==='fechada'?' selected':'')+'>Malha Fechada</option>';
  h+='</select></div>';
  if(curMalha==='fechada'){h+='<div style="display:flex;align-items:center;gap:6px"><span style="font-size:12px;color:var(--txm);font-weight:600">Realimentação:</span>';h+='<select onchange="curFbSign=this.value;reRender()" style="'+ss+';font-weight:600">';h+='<option value="neg"'+(curFbSign==='neg'?' selected':'')+'>Negativa G/(1+GH)</option>';h+='<option value="pos"'+(curFbSign==='pos'?' selected':'')+'>Positiva G/(1-GH)</option>';h+='</select></div>';}
  h+='<div style="display:flex;align-items:center;gap:6px"><span style="font-size:12px;color:var(--txm)">Sinal:</span>';
  h+='<select onchange="curSig=this.value;reRender()" style="'+ss+'">';
  ['degrau','rampa','senoidal','impulso','parabolica'].forEach(function(s){h+='<option value="'+s+'"'+(curSig===s?' selected':'')+'>'+sigNomes[s]+'</option>'});
  h+='</select></div></div>';
  var chks;
  if(curMalha==='aberta'){chks=[['tempo','Resposta no tempo'],['desemp','Desempenho'],['pz','Polos e Zeros'],['bm','Bode Magnitude'],['bp','Bode Fase']];}
  else{chks=[['tempo','Resposta no tempo'],['desemp','Desempenho'],['pz','Polos e Zeros'],['bm','Bode Magnitude'],['bp','Bode Fase'],['nyqst','Nyquist'],['lgr','LGR']];}
  h+='<div style="display:flex;flex-wrap:wrap;gap:8px 16px">';
  chks.forEach(function(c){h+='<label style="font-size:12px;cursor:pointer;display:flex;align-items:center;gap:4px"><input type="checkbox" '+(selAn[c[0]]?'checked':'')+' onchange="selAn.'+c[0]+'=this.checked;reRender()"> '+c[1]+'</label>'});
  h+='</div></div>';
  var tLabel=curMalha==='fechada'?'T(s) - Malha Fechada':'G(s) - Malha Aberta';
  h+='<div class="rcard"><h4>'+tLabel+'</h4>';
  if(ftmfErro)h+='<div style="color:#fbbf24;font-size:11px;margin-bottom:6px">⚠ '+esc(ftmfErro)+'</div>';
  h+='<div class="tf-disp"><div style="display:inline-block;text-align:center"><div style="padding:0 8px">'+esc(ns)+'</div><div style="border-top:2px solid var(--acc);padding:4px 8px 0">'+esc(ds)+'</div></div></div></div>';
  if(selAn.tempo){h+='<div class="rcard"><h4>Resposta no Tempo - '+sigNomes[curSig]+'</h4><div id="cStep" style="width:100%;height:320px"></div></div>';}
  if(selAn.desemp){h+='<div class="rcard"><h4>Desempenho</h4><div class="mgrid">';Object.keys(pf).forEach(function(k){h+='<div class="mbox"><div class="ml">'+k+'</div><div class="mv">'+pf[k]+'</div></div>'});h+='</div></div>';}
  if(selAn.pz){h+='<div class="rcard"><h4>Diagrama de Polos e Zeros</h4><div id="cPZ" style="width:100%;height:360px"></div>';h+='<div style="display:flex;gap:20px;flex-wrap:wrap;margin-top:8px"><div><b style="color:var(--red)">Polos:</b><div class="pzl">';ps.forEach(function(p){h+='<div style="color:var(--red)">'+fC(p)+'</div>'});if(!ps.length)h+="<div>-</div>";h+='</div></div><div><b style="color:var(--blu)">Zeros:</b><div class="pzl">';zs.forEach(function(z){h+='<div style="color:var(--blu)">'+fC(z)+'</div>'});if(!zs.length)h+="<div>-</div>";h+='</div></div></div>';h+='<div style="margin-top:8px;padding:6px 10px;border-radius:6px;font-weight:700;font-size:13px;'+(stb?'background:#16382a;color:#34d399">ESTAVEL':'background:#3a1520;color:#f87171">INSTAVEL')+'</div></div>';}
  if(selAn.bm){h+='<div class="rcard"><h4>Bode - Magnitude</h4><div id="cBM" style="width:100%;height:320px"></div></div>';}
  if(selAn.bp){h+='<div class="rcard"><h4>Bode - Fase</h4>';var bdMA=bode(tf,wr.a,wr.b,600);var gm=calcGM(bdMA),pm=calcPM(bdMA);h+='<div style="display:flex;gap:16px;margin-bottom:6px;flex-wrap:wrap"><span style="font-size:12px;color:var(--txm)">Margem de Ganho: <b style="color:'+(gm>0?'#34d399':'#f87171')+'">'+fN(gm)+' dB</b></span><span style="font-size:12px;color:var(--txm)">Margem de Fase: <b style="color:'+(pm>0?'#34d399':'#f87171')+'">'+fN(pm)+'°</b></span></div><div id="cBP" style="width:100%;height:320px"></div></div>';}
  if(curMalha==='aberta'&&selAn.nyqst){h+='<div class="rcard"><h4>Nyquist</h4><div id="cNyq" style="width:100%;height:380px"></div>';var nps=roots(tf.d),npsd=0;nps.forEach(function(p){if(p.r>1e-6)npsd++});h+='<div style="margin-top:8px;font-size:12px"><b>Polos SPD (P):</b> '+npsd+' | <b>Z = P + N:</b> '+(npsd===0?'Estável':'Instável')+'</div></div>';}
  if(curMalha==='fechada'&&selAn.lgr){h+='<div class="rcard"><h4>LGR</h4><div id="cLGR" style="width:100%;height:380px"></div></div>';}
  rb.innerHTML=h;
  setTimeout(function(){
    function chart2(id,xD,y1,y2,xL,yL){var el=document.getElementById(id);if(!el)return;Plotly.newPlot(el,[{x:xD,y:y1,mode:'lines',line:{color:'#60a5fa',width:1.5,dash:'dash'},name:'Entrada'},{x:xD,y:y2,mode:'lines',line:{color:'#f44336',width:2},name:'Saída'}],Object.assign({},_PT,{xaxis:_pAx(xL),yaxis:_pAx(yL),height:300,showlegend:true,hovermode:'x unified'}),{responsive:true})}
    function chart(id,xD,yD,xL,yL,col,logX){var el=document.getElementById(id);if(!el)return;Plotly.newPlot(el,[{x:xD,y:yD,mode:'lines',line:{color:col,width:2.5}}],Object.assign({},_PT,{xaxis:_pAx(xL,{type:logX?'log':'linear'}),yaxis:_pAx(yL),height:300,showlegend:false,hovermode:'x unified'}),{responsive:true})}
    function chartXY(id,xD,yD,xL,yL){var el=document.getElementById(id);if(!el)return;Plotly.newPlot(el,[{x:xD,y:yD,mode:'lines',line:{color:'#5b6be0',width:2},name:'Nyquist'},{x:xD,y:yD.map(function(v){return -v}),mode:'lines',line:{color:'#888',width:1,dash:'dash'},name:'Reflexo'},{x:[-1],y:[0],mode:'markers',marker:{color:'#ff4444',size:10,symbol:'circle'},name:'(-1,0)'}],Object.assign({},_PT,{xaxis:_pAx(xL),yaxis:_pAx(yL,{scaleanchor:'x'}),height:380,showlegend:true,hovermode:'closest'}),{responsive:true})}
    try{if(selAn.tempo)chart2("cStep",sr.t,sr.u,sr.y,"Tempo (s)","Amplitude");}catch(e){}
    try{if(selAn.pz)chartPZ("cPZ",ps,zs);}catch(e){}
    try{if(selAn.bm)chart("cBM",bd.w,bd.m,"w (rad/s)","dB","#60a5fa",true);}catch(e){}
    try{if(selAn.bp)chart("cBP",bd.w,bd.p,"w (rad/s)","graus","#f472b6",true);}catch(e){}
    try{if(curMalha==='aberta'&&selAn.nyqst)chartXY("cNyq",nqd.re,nqd.im,"Parte Real","Parte Imaginaria");}catch(e){}
    try{if(curMalha==='fechada'&&selAn.lgr)chartLGR("cLGR",lgrData,tf);}catch(e){}
  },150);rd.scrollIntoView({behavior:"smooth"})}
var curMode="diag",curSubMode="direct";
function updateFbLabel(){var sel=document.getElementById("manFbType");var lbl=document.getElementById("manClosedLabel");if(sel&&lbl){if(sel.value==="pos"){lbl.innerHTML="Malha Fechada: T(s) = G(s) / (1 - G(s)&middot;H(s))";}else{lbl.innerHTML="Malha Fechada: T(s) = G(s) / (1 + G(s)&middot;H(s))";}} }
function setMode(m){curMode=m;document.getElementById("modeDiag").classList.toggle("active",m==="diag");document.getElementById("modeMan").classList.toggle("active",m==="manual");document.getElementById("diag-workspace").style.display=m==="diag"?"flex":"none";document.getElementById("diag-toolbar").style.display=m==="diag"?"flex":"none";document.getElementById("manualSec").classList.toggle("vis",m==="manual");document.getElementById("btnCalcMain").innerHTML=m==="diag"?"&#9654; CALCULAR DIAGRAMA":"&#9654; CALCULAR T(s)"}
function setSubMode(m){curSubMode=m;["Direct","Closed","Open","SS"].forEach(function(n){document.getElementById("sub"+n).classList.toggle("active","sub"+n==="sub"+m.charAt(0).toUpperCase()+m.slice(1))});document.getElementById("manDirect").style.display=m==="direct"?"block":"none";document.getElementById("manClosed").style.display=m==="closed"?"block":"none";document.getElementById("manOpen").style.display=m==="open"?"block":"none";document.getElementById("manSS").style.display=m==="ss"?"block":"none"}
function onCalc(){
  if(curMode==="manual"){var tf;
    if(curSubMode==="direct"){tf={n:parseP(document.getElementById("manNum").value),d:parseP(document.getElementById("manDen").value)};}
    else if(curSubMode==="closed"){var gn=parseP(document.getElementById("manGN").value),gd=parseP(document.getElementById("manGD").value),hn=parseP(document.getElementById("manHN").value),hd=parseP(document.getElementById("manHD").value);var fbType=(document.getElementById("manFbType")||{}).value||"neg";if(fbType==="pos"){tf={n:pMul(gn,hd),d:pSub(pMul(gd,hd),pMul(gn,hn))};}else{tf={n:pMul(gn,hd),d:pAdd(pMul(gd,hd),pMul(gn,hn))};}}
    else if(curSubMode==="open"){var gn=parseP(document.getElementById("manOGN").value),gd=parseP(document.getElementById("manOGD").value),hn=parseP(document.getElementById("manOHN").value),hd=parseP(document.getElementById("manOHD").value);tf={n:pMul(gn,hn),d:pMul(gd,hd)};}
    else if(curSubMode==="ss"){try{tf=ssToTF(document.getElementById("manA").value,document.getElementById("manB").value,document.getElementById("manC").value,document.getElementById("manD").value);}catch(e){var rd=document.getElementById("res"),rb=document.getElementById("rb");rd.classList.add("vis");rb.innerHTML='<div class="ebox">Erro: '+esc(String(e))+'</div>';rd.scrollIntoView({behavior:"smooth"});return}}
    var lc=tf.d[tf.d.length-1];if(Math.abs(lc)>1e-14&&Math.abs(lc-1)>1e-10){tf.n=pScl(tf.n,1/lc);tf.d=pScl(tf.d,1/lc)}tf=pfReduce(tf);showRes(tf);return;}
  var r=solve(model.nodes,model.edges);
  if(r.e){var rd=document.getElementById("res"),rb=document.getElementById("rb");rd.classList.add("vis");rb.innerHTML='<div class="ebox">'+esc(r.e)+'</div>';rd.scrollIntoView({behavior:"smooth"});return}
  showRes(r.tf)}
var model={nodes:[],edges:[]},selId=null,dragSt=null,conSt=null;
var cw=document.getElementById("cw"),cv=document.getElementById("cv"),wSvg=document.getElementById("wires");
function nxtId(){var m=0;model.nodes.forEach(function(n){var v=parseInt(n.id.replace("n",""))||0;if(v>m)m=v});return"n"+(m+1)}
function ptr(e){if(e.touches&&e.touches.length)return{x:e.touches[0].clientX,y:e.touches[0].clientY};if(e.changedTouches&&e.changedTouches.length)return{x:e.changedTouches[0].clientX,y:e.changedTouches[0].clientY};return{x:e.clientX,y:e.clientY}}
var BL={tf:"Planta",ss:"Espaço de Estados",gain:"Ganho",sum:"Somador",int:"Integrador",der:"Derivador",pid:"PID",sensor:"Sensor",actuator:"Atuador",input:"Entrada",output:"Saída",branch:"Ramificação"};
function dPar(t){if(t==="tf")return{num:"1",den:"s+1"};if(t==="ss")return{ssA:"0 1; -2 -3",ssB:"0; 1",ssC:"1 0",ssD:"0"};if(t==="gain")return{k:"1"};if(t==="sum")return{signs:"+ -"};if(t==="pid")return{kp:"1",ki:"0",kd:"0"};if(t==="sensor"||t==="actuator")return{num:"1",den:"1"};if(t==="input")return{label:"R(s)"};if(t==="output")return{label:"Y(s)"};return{}}
function gPC(t,p){if(t==="input")return{i:[],o:[{id:"out0"}]};if(t==="output")return{i:[{id:"in0"}],o:[]};if(t==="branch")return{i:[{id:"in0"}],o:[{id:"out0"},{id:"out1"}]};if(t==="sum"){var sg=(p&&p.signs?p.signs:"+ -").trim().split(/\s+/);return{i:sg.map(function(s,i){return{id:"in"+i,sign:s}}),o:[{id:"out0"}]}}return{i:[{id:"in0"}],o:[{id:"out0"}]}}
function bTxt(n){var p=n.params||{};if(n.type==="tf"||n.type==="actuator")return'<div class="block-tf-disp"><div class="tf-num">'+(p.num||"1")+'</div><div>'+(p.den||"1")+'</div></div>';if(n.type==="ss"){var tf=ssToTF(p.ssA||"0",p.ssB||"0",p.ssC||"1",p.ssD||"0");return'<div class="block-tf-disp"><div class="tf-num">'+fP(tf.n)+'</div><div>'+fP(tf.d)+'</div></div>'}if(n.type==="gain")return"K="+(p.k||"1");if(n.type==="pid"){var _tf=bTF(n);return'<div class="block-tf-disp"><div class="tf-num">'+fP(_tf.n)+'</div><div>'+fP(_tf.d)+'</div></div>';}if(n.type==="sensor")return'<div class="block-tf-disp"><div class="tf-num">'+(p.num||"1")+'</div><div>'+(p.den||"1")+'</div></div>';if(n.type==="input")return p.label||"R(s)";if(n.type==="output")return p.label||"Y(s)";if(n.type==="sum")return"\u03a3";if(n.type==="int")return"1/s";if(n.type==="der")return"s";return""}
function addB(t){var r=cw.getBoundingClientRect(),n={id:nxtId(),type:t,x:40+Math.random()*(r.width-200),y:40+Math.random()*(r.height-140),params:dPar(t)};if(t==="input"){n.x=30;n.y=r.height/2-30}if(t==="output"){n.x=r.width-160;n.y=r.height/2-30}model.nodes.push(n);render();setSel(n.id)}
function addBP(t,p){var r=cw.getBoundingClientRect(),n={id:nxtId(),type:t,x:40+Math.random()*(r.width-200),y:40+Math.random()*(r.height-140),params:Object.assign(dPar(t),p)};if(t==="input"){n.x=30;n.y=r.height/2-30}if(t==="output"){n.x=r.width-160;n.y=r.height/2-30}model.nodes.push(n);render();setSel(n.id)}
function delSel(){if(!selId)return;model.edges=model.edges.filter(function(e){return e.src!==selId&&e.dst!==selId});model.nodes=model.nodes.filter(function(n){return n.id!==selId});selId=null;conSt=null;render()}
function clrAll(){model={nodes:[],edges:[]};selId=null;conSt=null;render();document.getElementById("res").classList.remove("vis")}
function autoLay(){if(!model.nodes.length)return;var lv={},aj={};model.edges.forEach(function(e){if(!aj[e.src])aj[e.src]=[];aj[e.src].push(e.dst)});var ins=model.nodes.filter(function(n){return n.type==="input"}).map(function(n){return n.id});if(!ins.length)ins=[model.nodes[0].id];var q=ins.map(function(id){return{id:id,l:0}}),vis={};ins.forEach(function(id){vis[id]=true});while(q.length){var c=q.shift();lv[c.id]=c.l;(aj[c.id]||[]).forEach(function(n){if(!vis[n]){vis[n]=true;q.push({id:n,l:c.l+1})}})}model.nodes.forEach(function(n){if(!(n.id in lv))lv[n.id]=3});var bl={};Object.keys(lv).forEach(function(id){var l=lv[id];if(!bl[l])bl[l]=[];bl[l].push(id)});Object.keys(bl).forEach(function(l){bl[l].forEach(function(id,i){var n=model.nodes.find(function(nd){return nd.id===id});if(n){n.x=50+parseInt(l)*170;n.y=50+i*100}})});render()}
function setSel(id){selId=id;document.querySelectorAll(".block").forEach(function(el){el.classList.toggle("sel",el.dataset.id===id)});updP()}
function pSty(t,k,i){if(t==="sum"){if(k==="in"&&i===0)return"left:-7px;top:50%;transform:translateY(-50%)";if(k==="in"&&i===1)return"left:50%;bottom:-7px;transform:translateX(-50%)";if(k==="in"&&i>=2)return"left:-7px;top:"+(8+i*14)+"px";return"right:-7px;top:50%;transform:translateY(-50%)"}if(t==="branch"){if(k==="in")return"left:-7px;top:50%;transform:translateY(-50%)";if(k==="out"&&i===0)return"right:-7px;top:50%;transform:translateY(-50%)";return"left:50%;bottom:-7px;transform:translateX(-50%)"}if(t==="input"&&k==="out")return"right:-7px;top:50%;transform:translateY(-50%)";if(t==="output"&&k==="in")return"left:-7px;top:50%;transform:translateY(-50%)";return k==="in"?"left:-7px;top:50%;transform:translateY(-50%)":"right:-7px;top:50%;transform:translateY(-50%)"}
function render(){cv.querySelectorAll(".block").forEach(function(el){el.remove()});model.nodes.forEach(function(n){var el=document.createElement("div");el.className="block block-"+n.type;el.dataset.id=n.id;el.style.left=n.x+"px";el.style.top=n.y+"px";var pc=gPC(n.type,n.params),h="";if(n.type==="sum")h='<div class="block-body">\u03a3</div>';else if(n.type==="branch")h="";else h='<div class="block-header">'+BL[n.type]+'</div><div class="block-body">'+bTxt(n)+'</div>';pc.i.forEach(function(p,i){h+='<div class="port port-in" data-port="'+p.id+'" data-nid="'+n.id+'" data-kind="in" style="'+pSty(n.type,"in",i)+'"></div>';if(n.type==="sum"&&p.sign)h+='<span class="sign-label" style="'+(i===0?"left:-20px;top:50%;transform:translateY(-50%)":i===1?"left:50%;bottom:-18px;transform:translateX(-50%)":"left:-20px;top:"+(8+i*14)+"px")+'">'+p.sign+'</span>'});pc.o.forEach(function(p,i){h+='<div class="port port-out" data-port="'+p.id+'" data-nid="'+n.id+'" data-kind="out" style="'+pSty(n.type,"out",i)+'"></div>'});el.innerHTML=h;cv.appendChild(el);el.addEventListener("mousedown",function(ev){onBD(ev,n.id)});el.addEventListener("touchstart",function(ev){onBD(ev,n.id)},{passive:false});el.querySelectorAll(".port").forEach(function(pl){pl.addEventListener("mousedown",function(ev){ev.stopPropagation();onPC(ev,pl)});pl.addEventListener("touchstart",function(ev){ev.stopPropagation();ev.preventDefault();onPC(ev,pl)},{passive:false})})});document.querySelectorAll(".block").forEach(function(el){el.classList.toggle("sel",el.dataset.id===selId)});rW();updP()}
function rW(){wSvg.querySelectorAll("path").forEach(function(p){p.remove()});var wr=cw.getBoundingClientRect();wSvg.setAttribute("width",wr.width);wSvg.setAttribute("height",wr.height);model.edges.forEach(function(e,idx){var a=gPP(e.src,e.srcPort||"out0"),b=gPP(e.dst,e.dstPort||"in0");if(!a||!b)return;var p=document.createElementNS("http://www.w3.org/2000/svg","path"),dx=Math.max(50,Math.abs(b.x-a.x)*.4);if(b.x<a.x-20){var mY=Math.max(a.y,b.y)+60;p.setAttribute("d","M "+a.x+" "+a.y+" C "+(a.x+40)+" "+a.y+","+(a.x+40)+" "+mY+","+((a.x+b.x)/2)+" "+mY+" S "+(b.x-40)+" "+b.y+","+b.x+" "+b.y);p.setAttribute("stroke","#e8a035");p.setAttribute("stroke-dasharray","8 4");p.setAttribute("marker-end","url(#ar2)")}else{p.setAttribute("d","M "+a.x+" "+a.y+" C "+(a.x+dx)+" "+a.y+","+(b.x-dx)+" "+b.y+","+b.x+" "+b.y);p.setAttribute("stroke","#5b6be0");p.setAttribute("marker-end","url(#ar)")}p.setAttribute("fill","none");p.setAttribute("stroke-width","2.5");p.style.pointerEvents="visibleStroke";p.style.cursor="pointer";p.addEventListener("click",function(){model.edges.splice(idx,1);render()});p.addEventListener("touchend",function(ev){ev.preventDefault();model.edges.splice(idx,1);render()});wSvg.appendChild(p)})}
function gPP(nid,pid){var bl=document.querySelector('.block[data-id="'+nid+'"]');if(!bl)return null;var pl=bl.querySelector('.port[data-port="'+pid+'"]');if(!pl){pl=bl.querySelector(".port.port-"+(pid.indexOf("out")===0?"out":"in"))}if(!pl)return null;var cr=cw.getBoundingClientRect(),pr=pl.getBoundingClientRect();return{x:pr.left+pr.width/2-cr.left,y:pr.top+pr.height/2-cr.top}}
function onBD(e,id){if(e.target.classList.contains("port"))return;if(e.button===2)return;e.preventDefault();setSel(id);var p=ptr(e),el=document.querySelector('.block[data-id="'+id+'"]');if(!el)return;var r=el.getBoundingClientRect();dragSt={id:id,ox:p.x-r.left,oy:p.y-r.top}}
function onPM(e){if(!dragSt)return;e.preventDefault();var p=ptr(e),wr=cw.getBoundingClientRect(),el=document.querySelector('.block[data-id="'+dragSt.id+'"]');if(!el)return;var nx=Math.max(0,Math.min(wr.width-el.offsetWidth,p.x-wr.left-dragSt.ox)),ny=Math.max(0,Math.min(wr.height-el.offsetHeight,p.y-wr.top-dragSt.oy));el.style.left=nx+"px";el.style.top=ny+"px";var nd=model.nodes.find(function(n){return n.id===dragSt.id});if(nd){nd.x=nx;nd.y=ny}rW()}
document.addEventListener("mousemove",onPM);document.addEventListener("mouseup",function(){dragSt=null});document.addEventListener("touchmove",onPM,{passive:false});document.addEventListener("touchend",function(){dragSt=null});
cv.addEventListener("mousedown",function(e){if(e.target===cv){conSt=null;document.querySelectorAll(".port.active").forEach(function(p){p.classList.remove("active")});setSel(null)}});
cv.addEventListener("touchstart",function(e){if(e.target===cv){conSt=null;document.querySelectorAll(".port.active").forEach(function(p){p.classList.remove("active")});setSel(null)}},{passive:false});
function onPC(e,pl){e.preventDefault();var nid=pl.dataset.nid,pid=pl.dataset.port,kind=pl.dataset.kind;setSel(nid);if(conSt){if(conSt.nid===nid){conSt=null;document.querySelectorAll(".port.active").forEach(function(p){p.classList.remove("active")});return}var sI,sP,dI,dP;if(conSt.kind==="out"&&kind==="in"){sI=conSt.nid;sP=conSt.pid;dI=nid;dP=pid}else if(conSt.kind==="in"&&kind==="out"){sI=nid;sP=pid;dI=conSt.nid;dP=conSt.pid}else{conSt=null;document.querySelectorAll(".port.active").forEach(function(p){p.classList.remove("active")});return}if(!model.edges.some(function(ed){return ed.src===sI&&ed.srcPort===sP&&ed.dst===dI&&ed.dstPort===dP}))model.edges.push({id:"e"+Math.random().toString(36).slice(2),src:sI,srcPort:sP,dst:dI,dstPort:dP});conSt=null;document.querySelectorAll(".port.active").forEach(function(p){p.classList.remove("active")});render()}else{conSt={nid:nid,pid:pid,kind:kind};pl.classList.add("active")}}
function updP(){document.getElementById("pB").textContent=model.nodes.length;document.getElementById("pE").textContent=model.edges.length;var pa=document.getElementById("pA");if(!selId){pa.innerHTML='<span class="hint">Selecione um bloco.</span>';return}var nd=model.nodes.find(function(n){return n.id===selId});if(!nd){pa.innerHTML="";return}var p=nd.params||{};var h='<div style="font-size:11px;color:var(--txm);margin-bottom:6px">'+BL[nd.type]+' <b style="color:var(--tx)">'+nd.id+'</b></div>';if(nd.type==="tf"||nd.type==="sensor"||nd.type==="actuator"){h+=pI("Num","num",p.num||"1");h+=pI("Den","den",p.den||"1")}else if(nd.type==="ss"){h+=pI("Matriz A","ssA",p.ssA||"0 1; -2 -3");h+=pI("Matriz B","ssB",p.ssB||"0; 1");h+=pI("Matriz C","ssC",p.ssC||"1 0");h+=pI("Matriz D","ssD",p.ssD||"0")}else if(nd.type==="gain")h+=pI("K","k",p.k||"1");else if(nd.type==="sum")h+=pI("Sinais","signs",p.signs||"+ -");else if(nd.type==="pid"){h+=pI("Kp","kp",p.kp||"1");h+=pI("Ki","ki",p.ki||"0");h+=pI("Kd","kd",p.kd||"0")}else if(nd.type==="input"||nd.type==="output")h+=pI("Label","label",p.label||"");pa.innerHTML=h;pa.querySelectorAll("input[data-key]").forEach(function(inp){inp.addEventListener("input",function(){nd.params[inp.dataset.key]=inp.value;if(inp.dataset.key==="signs")render();else{var bl=document.querySelector('.block[data-id="'+nd.id+'"] .block-body');if(bl)bl.innerHTML=bTxt(nd)}})})}
function pI(l,k,v){return'<div class="pg"><label>'+l+'</label><input data-key="'+k+'" value="'+esc(String(v))+'"></div>'}
var connType=null,connSel=[];
function openConnModal(tipo){connType=tipo;connSel=[];var titles={serie:'Série',paralelo:'Paralelo',fb_neg:'Realimentação Negativa',fb_pos:'Realimentação Positiva'};var hints={serie:'Selecione 2+ blocos na ordem da cadeia.',paralelo:'Selecione 2+ blocos para somar.',fb_neg:'Selecione 2 blocos: G(s) e H(s).',fb_pos:'Selecione 2 blocos: G(s) e H(s).'};document.getElementById('connModalTitle').textContent=titles[tipo]||'Conexão';document.getElementById('connModalHint').textContent=hints[tipo]||'';var avail=model.nodes.filter(function(n){return['tf','ss','gain','pid','sensor','actuator','int','der'].indexOf(n.type)>=0});var cl=document.getElementById('connBlockList');var h='';if(!avail.length){h='<p style="color:#ef4444;font-size:13px;padding:12px">Nenhum bloco disponivel.</p>'}else{avail.forEach(function(n){var lbl=BL[n.type]+' ('+n.id+')';var det='';if(n.params){if(n.params.num)det=n.params.num+'/'+n.params.den;else if(n.params.k)det='K='+n.params.k;else if(n.params.ssA)det='SS'}h+='<div class="conn-item" data-connid="'+n.id+'" onclick="toggleConnSel(this,\''+n.id+'\')">';h+='<span class="conn-chk">&#9744;</span> ';h+='<span style="font-weight:600;color:var(--tx)">'+esc(lbl)+'</span>';if(det)h+='<span style="color:var(--txm);font-size:10px;margin-left:auto">'+esc(det)+'</span>';h+='</div>'})}cl.innerHTML=h;document.getElementById('connModal').classList.add('vis')}
function closeConnModal(){document.getElementById('connModal').classList.remove('vis');connType=null;connSel=[]}
function toggleConnSel(el,nid){var idx=connSel.indexOf(nid);var chk=el.querySelector('.conn-chk');if(idx>=0){connSel.splice(idx,1);el.classList.remove('conn-sel');if(chk)chk.innerHTML='&#9744;'}else{if((connType==='fb_neg'||connType==='fb_pos')&&connSel.length>=2){alert('Feedback: maximo 2 blocos.');return}connSel.push(nid);el.classList.add('conn-sel');if(chk)chk.innerHTML='&#9745;'}var cnt=document.getElementById('connSelCount');if(cnt)cnt.textContent=connSel.length+' bloco'+(connSel.length!==1?'s':'')+' selecionado'+(connSel.length!==1?'s':'')}
function applyConn(){if(connSel.length<2){alert('Selecione pelo menos 2 blocos.');return}if(connType==='serie')buildSérie(connSel.slice());else if(connType==='paralelo')buildParalelo(connSel.slice());else if(connType==='fb_neg')buildFeedback(connSel.slice(),false);else if(connType==='fb_pos')buildFeedback(connSel.slice(),true);closeConnModal();render()}
function mkEid(){return 'e'+Date.now().toString(36)+Math.random().toString(36).slice(2,6)}
function buildSérie(ids){for(var i=0;i<ids.length-1;i++){var a=ids[i],b=ids[i+1];var already=model.edges.some(function(e){return e.src===a&&e.dst===b});if(!already)model.edges.push({id:mkEid(),src:a,srcPort:'out0',dst:b,dstPort:'in0'})}var baseX=80,baseY=180;ids.forEach(function(id,i){var nd=model.nodes.find(function(n){return n.id===id});if(nd){nd.x=baseX+i*200;nd.y=baseY}})}
function buildParalelo(ids){var nb=ids.length;var baseX=60,baseY=80;var brIds=[];for(var i=0;i<nb-1;i++){var br={id:nxtId(),type:'branch',x:baseX,y:baseY+50+i*50,params:{}};model.nodes.push(br);brIds.push(br.id);if(i>0)model.edges.push({id:mkEid(),src:brIds[i-1],srcPort:'out0',dst:br.id,dstPort:'in0'})}var signs='';for(var i=0;i<nb;i++)signs+=(i>0?' ':'')+'+';var smX=baseX+420,smY=baseY+(nb-1)*55;var sm={id:nxtId(),type:'sum',x:smX,y:smY,params:{signs:signs}};model.nodes.push(sm);ids.forEach(function(id,i){var nd=model.nodes.find(function(n){return n.id===id});if(nd){nd.x=baseX+200;nd.y=baseY+i*120}if(i===0){model.edges.push({id:mkEid(),src:brIds[0],srcPort:'out1',dst:id,dstPort:'in0'})}else if(i<nb-1){model.edges.push({id:mkEid(),src:brIds[i],srcPort:'out1',dst:id,dstPort:'in0'})}else{model.edges.push({id:mkEid(),src:brIds[nb-2],srcPort:'out0',dst:id,dstPort:'in0'})}model.edges.push({id:mkEid(),src:id,srcPort:'out0',dst:sm.id,dstPort:'in'+i})})}
function buildFeedback(ids,positive){
  if(ids.length<2)return;
  var gId=ids[0],hId=ids[1];
  var gNd=model.nodes.find(function(n){return n.id===gId});
  var hNd=model.nodes.find(function(n){return n.id===hId});
  if(!gNd||!hNd)return;

  var baseX=Math.max(60,(gNd.x||80)-140),baseY=(gNd.y||160);
  var signStr=positive?'+ +':'+ -';
  var sm={id:nxtId(),type:'sum',x:baseX,y:baseY+15,params:{signs:signStr}};
  var br={id:nxtId(),type:'branch',x:baseX+380,y:baseY+15,params:{}};
  model.nodes.push(sm);model.nodes.push(br);

  var incomingMain=model.edges.filter(function(e){
    return e.dst===gId && (e.dstPort||'in0')==='in0' && e.src!==hId;
  });
  var outgoingMain=model.edges.filter(function(e){
    return e.src===gId && (e.srcPort||'out0')==='out0' && e.dst!==hId;
  });

  incomingMain.forEach(function(e){
    e.dst=sm.id;
    e.dstPort='in0';
  });

  outgoingMain.forEach(function(e){
    e.src=br.id;
    e.srcPort='out0';
  });

  model.edges = model.edges.filter(function(e){
    var isOldFbToG = e.dst===gId && e.src===hId;
    var isOldGToH = e.src===gId && e.dst===hId;
    return !(isOldFbToG || isOldGToH);
  });

  if(!incomingMain.length){
    var inp=model.nodes.find(function(n){return n.type==='input'});
    if(inp){
      model.edges.push({id:mkEid(),src:inp.id,srcPort:'out0',dst:sm.id,dstPort:'in0'});
    }
  }

  model.edges.push({id:mkEid(),src:sm.id,srcPort:'out0',dst:gId,dstPort:'in0'});
  model.edges.push({id:mkEid(),src:gId,srcPort:'out0',dst:br.id,dstPort:'in0'});
  model.edges.push({id:mkEid(),src:br.id,srcPort:'out1',dst:hId,dstPort:'in0'});
  model.edges.push({id:mkEid(),src:hId,srcPort:'out0',dst:sm.id,dstPort:'in1'});

  if(!outgoingMain.length){
    var out=model.nodes.find(function(n){return n.type==='output'});
    if(out){
      model.edges.push({id:mkEid(),src:br.id,srcPort:'out0',dst:out.id,dstPort:'in0'});
    }
  }

  gNd.x=baseX+180; gNd.y=baseY;
  hNd.x=baseX+220; hNd.y=baseY+160;
}
document.querySelectorAll(".tb[data-add]").forEach(function(b){b.addEventListener("click",function(){addB(b.dataset.add)})});
document.getElementById("btnDel").addEventListener("click",delSel);document.getElementById("btnClear").addEventListener("click",clrAll);document.getElementById("btnAuto").addEventListener("click",autoLay);
document.addEventListener("keydown",function(e){if(e.target.tagName==="INPUT")return;if(e.key==="Delete"||e.key==="Backspace")delSel();if(e.key==="Escape"){conSt=null;closeModal();closeConnModal();document.querySelectorAll(".port.active").forEach(function(p){p.classList.remove("active")})}});
render();new ResizeObserver(function(){rW()}).observe(cw);
var _initData=__INITIAL_BLOCKS__;
if(_initData){if(_initData.nodes&&_initData.edges){_initData.nodes.forEach(function(n){n.params=Object.assign(dPar(n.type),n.params||{});model.nodes.push(n)});_initData.edges.forEach(function(e){model.edges.push(e)});render();}else if(_initData.length>0){_initData.forEach(function(b){addBP(b.type,b.params)});setTimeout(autoLay,200);}}
</script></body></html>'''


# ══════════════════════════════════════════════════
# MODO CANVAS
# ══════════════════════════════════════════════════

def _tf_to_str(coeffs):
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


def _coeffs_to_poly_str(coeffs):
    n = len(coeffs) - 1
    terms = []
    for i, c in enumerate(coeffs):
        power = n - i
        c_val = float(c)
        if abs(c_val) < 1e-10:
            continue
        if power == 0:
            terms.append(f"{c_val:g}")
        elif power == 1:
            if abs(c_val - 1) < 1e-10:
                terms.append("s")
            elif abs(c_val + 1) < 1e-10:
                terms.append("-s")
            else:
                terms.append(f"{c_val:g}*s")
        else:
            if abs(c_val - 1) < 1e-10:
                terms.append(f"s^{power}")
            elif abs(c_val + 1) < 1e-10:
                terms.append(f"-s^{power}")
            else:
                terms.append(f"{c_val:g}*s^{power}")
    if not terms:
        return "0"
    result = terms[0]
    for t in terms[1:]:
        result += t if t.startswith('-') else '+' + t
    return result


def _format_complex_list(arr):
    if len(arr) == 0:
        return "Nenhum"
    parts = []
    for v in arr:
        if np.isreal(v):
            parts.append(f"{np.real(v):.4f}")
        else:
            parts.append(f"{np.real(v):.4f} {'+' if np.imag(v) >= 0 else '-'} {abs(np.imag(v)):.4f}j")
    return ", ".join(parts)


def _build_canvas_model(blocos_df, conexoes):
    tipo_map = {
        'Planta': 'tf', 'Controlador': 'tf', 'Sensor': 'sensor',
        'Atuador': 'actuator', 'Pre-filtro': 'tf', 'Perturbação': 'tf',
    }
    nodes = []
    edges = []
    nid = 0

    def mk_id():
        nonlocal nid
        nid += 1
        return f"n{nid}"

    if not conexoes:
        inp_id = mk_id()
        nodes.append({'id': inp_id, 'type': 'input', 'x': 30, 'y': 200, 'params': {'label': 'R(s)'}})
        for i, (_, row) in enumerate(blocos_df.iterrows()):
            tf_obj = row['tf']
            bt = tipo_map.get(row['tipo'], 'tf')
            b_id = mk_id()
            nodes.append({
                'id': b_id, 'type': bt,
                'x': 200 + i * 180, 'y': 180,
                'params': {'num': _coeffs_to_poly_str(tf_obj.num[0][0]),
                           'den': _coeffs_to_poly_str(tf_obj.den[0][0])}
            })
        out_id = mk_id()
        nodes.append({'id': out_id, 'type': 'output',
                      'x': 200 + len(blocos_df) * 180, 'y': 200,
                      'params': {'label': 'Y(s)'}})
        return {'nodes': nodes, 'edges': edges}

    bloco_map = {}
    for _, row in blocos_df.iterrows():
        tf_obj = row['tf']
        bt = tipo_map.get(row['tipo'], 'tf')
        bloco_map[row['nome']] = {
            'type': bt,
            'params': {'num': _coeffs_to_poly_str(tf_obj.num[0][0]),
                       'den': _coeffs_to_poly_str(tf_obj.den[0][0])}
        }

    inp_id = mk_id()
    nodes.append({'id': inp_id, 'type': 'input', 'x': 30, 'y': 200, 'params': {'label': 'R(s)'}})

    prev_out_id = inp_id
    prev_out_port = 'out0'
    x_offset = 200

    for con in conexoes:
        tipo_con = con['tipo']
        nomes = [n for n in con['blocos'] if n in bloco_map]
        if len(nomes) < 2:
            continue

        if tipo_con == 'Série':
            for i, nome in enumerate(nomes):
                b = bloco_map[nome]
                b_id = mk_id()
                nodes.append({'id': b_id, 'type': b['type'],
                              'x': x_offset + i * 180, 'y': 180,
                              'params': dict(b['params'])})
                edges.append({'id': f'e{len(edges)}', 'src': prev_out_id,
                              'srcPort': prev_out_port, 'dst': b_id, 'dstPort': 'in0'})
                prev_out_id = b_id
                prev_out_port = 'out0'
            x_offset += len(nomes) * 180

        elif tipo_con == 'Paralelo':
            nb = len(nomes)
            signs = ' '.join(['+'] * nb)
            sum_id = mk_id()
            nodes.append({'id': sum_id, 'type': 'sum',
                          'x': x_offset + 360, 'y': 210,
                          'params': {'signs': signs}})
            branch_ids = []
            cur_src_id = prev_out_id
            cur_src_port = prev_out_port
            for i in range(nb - 1):
                br_id = mk_id()
                nodes.append({'id': br_id, 'type': 'branch',
                              'x': x_offset + i * 30, 'y': 160 + i * 50,
                              'params': {}})
                edges.append({'id': f'e{len(edges)}', 'src': cur_src_id,
                              'srcPort': cur_src_port, 'dst': br_id, 'dstPort': 'in0'})
                branch_ids.append(br_id)
                cur_src_id = br_id
                cur_src_port = 'out0'
            for i, nome in enumerate(nomes):
                b = bloco_map[nome]
                b_id = mk_id()
                nodes.append({'id': b_id, 'type': b['type'],
                              'x': x_offset + 160, 'y': 100 + i * 120,
                              'params': dict(b['params'])})
                if i < nb - 1:
                    edges.append({'id': f'e{len(edges)}', 'src': branch_ids[i],
                                  'srcPort': 'out1', 'dst': b_id, 'dstPort': 'in0'})
                else:
                    edges.append({'id': f'e{len(edges)}', 'src': branch_ids[-1],
                                  'srcPort': 'out0', 'dst': b_id, 'dstPort': 'in0'})
                edges.append({'id': f'e{len(edges)}', 'src': b_id,
                              'srcPort': 'out0', 'dst': sum_id, 'dstPort': f'in{i}'})
            prev_out_id = sum_id
            prev_out_port = 'out0'
            x_offset += 520

        elif tipo_con.startswith('Realimentação'):
            is_pos = tipo_con == 'Realimentação Positiva'
            sign_str = '+ +' if is_pos else '+ -'
            sum_id = mk_id()
            nodes.append({'id': sum_id, 'type': 'sum',
                          'x': x_offset, 'y': 200,
                          'params': {'signs': sign_str}})
            edges.append({'id': f'e{len(edges)}', 'src': prev_out_id,
                          'srcPort': prev_out_port, 'dst': sum_id, 'dstPort': 'in0'})
            g_data = bloco_map[nomes[0]]
            g_id = mk_id()
            nodes.append({'id': g_id, 'type': g_data['type'],
                          'x': x_offset + 160, 'y': 180,
                          'params': dict(g_data['params'])})
            edges.append({'id': f'e{len(edges)}', 'src': sum_id,
                          'srcPort': 'out0', 'dst': g_id, 'dstPort': 'in0'})
            branch_id = mk_id()
            nodes.append({'id': branch_id, 'type': 'branch',
                          'x': x_offset + 340, 'y': 210,
                          'params': {}})
            edges.append({'id': f'e{len(edges)}', 'src': g_id,
                          'srcPort': 'out0', 'dst': branch_id, 'dstPort': 'in0'})
            if len(nomes) > 1:
                h_data = bloco_map[nomes[1]]
                h_id = mk_id()
                nodes.append({'id': h_id, 'type': h_data['type'],
                              'x': x_offset + 200, 'y': 340,
                              'params': dict(h_data['params'])})
                edges.append({'id': f'e{len(edges)}', 'src': branch_id,
                              'srcPort': 'out1', 'dst': h_id, 'dstPort': 'in0'})
                edges.append({'id': f'e{len(edges)}', 'src': h_id,
                              'srcPort': 'out0', 'dst': sum_id, 'dstPort': 'in1'})
            prev_out_id = branch_id
            prev_out_port = 'out0'
            x_offset += 420

    out_id = mk_id()
    nodes.append({'id': out_id, 'type': 'output',
                  'x': x_offset, 'y': 200, 'params': {'label': 'Y(s)'}})
    edges.append({'id': f'e{len(edges)}', 'src': prev_out_id,
                  'srcPort': prev_out_port, 'dst': out_id, 'dstPort': 'in0'})

    return {'nodes': nodes, 'edges': edges}


def modo_canvas():
    st.title("Modo Diagrama de Blocos")

    with st.sidebar:
        st.header("Navegação")
        if st.button("← Voltar à Tela Inicial"):
            st.session_state.modo_selecionado = None
            st.rerun()

        st.markdown("---")
        st.markdown("### Como usar")
        st.markdown("""
        **Diagrama de Blocos:**
        1. Clique **+ Adicionar Bloco** ou use os botões rápidos
        2. Arraste para posicionar
        3. Conecte: porta azul (saída) → porta verde (entrada)
        4. Edite parâmetros no painel lateral
        5. Clique **CALCULAR** para ver resultados

        **Entrada Manual:**
        1. Clique em "Entrada Manual"
        2. Escolha: T(s) Direta, Malha Fechada, Malha Aberta ou Espaço de Estados
        3. Preencha os campos
        4. Clique **CALCULAR T(s)**
        """)

    html_content = _load_visual_editor_html()

    if not st.session_state.blocos.empty:
        canvas_model = _build_canvas_model(
            st.session_state.blocos, st.session_state.conexoes)
        html_content = html_content.replace(
            '__INITIAL_BLOCKS__', json.dumps(canvas_model))
    else:
        html_content = html_content.replace('__INITIAL_BLOCKS__', '[]')

    components.html(html_content, height=2800, scrolling=True)


# ══════════════════════════════════════════════════
# APLICAÇÃO PRINCIPAL
# ══════════════════════════════════════════════════

def main():
    st.set_page_config(
        page_title="Modelagem de Sistemas de Controle",
        page_icon="⚙️",
        layout="wide",
    )
    inicializar_estado()

    if st.session_state.modo_selecionado is None:
        tela_inicial()
    elif st.session_state.modo_selecionado == 'classico':
        modo_classico()
    elif st.session_state.modo_selecionado == 'canvas':
        modo_canvas()


if __name__ == "__main__":
    main()
