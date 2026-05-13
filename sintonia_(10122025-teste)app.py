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
from sympy.parsing.sympy_parser import (
    parse_expr,
    standard_transformations,
    implicit_multiplication_application,
    convert_xor,
)
import control as ctrl
from control import TransferFunction, margin, step_response, forced_response, root_locus
from scipy import signal
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import re
from modo_diagrama_blocos import modo_canvas
from modo_guia_estudos import render_guia_popup

TRANSFORMATIONS = standard_transformations + (
    implicit_multiplication_application,
    convert_xor,
)

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


def normalizar_expressao(expr: str) -> str:
    return (
        expr.strip()
        .replace("²", "^2")
        .replace("³", "^3")
        .replace("−", "-")
        .replace("–", "-")
        .replace("·", "*")
    )


def _parse_value(v):
    """Parse a single value: numbers, math expressions, or expressions with 's'."""
    v = normalizar_expressao(v)
    if not v:
        return sp.Integer(0)
    try:
        return sp.Rational(v) if '/' in v else sp.Float(float(v))
    except (ValueError, TypeError):
        s = sp.Symbol('s')
        return parse_expr(v, local_dict={'s': s}, transformations=TRANSFORMATIONS)


def _grau_polinomio(coeffs):
    coeffs = np.asarray(coeffs, dtype=float)
    nz = np.flatnonzero(np.abs(coeffs) > 1e-12)
    if len(nz) == 0:
        return -np.inf
    return len(coeffs) - nz[0] - 1


def _tf_eh_propria(tf_sys):
    num_grau = _grau_polinomio(tf_sys.num[0][0])
    den_grau = _grau_polinomio(tf_sys.den[0][0])
    return num_grau <= den_grau


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
    num = parse_expr(
        normalizar_expressao(numerador_str),
        local_dict={'s': s},
        transformations=TRANSFORMATIONS
    )
    den = parse_expr(
        normalizar_expressao(denominador_str),
        local_dict={'s': s},
        transformations=TRANSFORMATIONS
    )
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
    return ctrl.minreal(resultado, verbose=False)


def blocos_em_paralelo(tf_list):
    resultado = tf_list[0]
    for tf in tf_list[1:]:
        resultado = resultado + tf
    return ctrl.minreal(resultado, verbose=False)


def realimentacao(G, H=None, positiva=False):
    """
    negativa -> G / (1 + G.H)
    positiva -> G / (1 - G.H)
    """
    if H is None:
        H = TransferFunction([1], [1])
    if positiva:
        return ctrl.minreal(ctrl.feedback(G, H, sign=1), verbose=False)
    return ctrl.minreal(ctrl.feedback(G, H, sign=-1), verbose=False)


def simplificar_diagrama(blocos_df, conexoes):
    if blocos_df.empty:
        raise ValueError("Nenhum bloco definido.")

    tf_map = {row['nome']: row['tf'] for _, row in blocos_df.iterrows()}

    if not conexoes:
        tfs = [tf_map[row['nome']] for _, row in blocos_df.iterrows()]
        if len(tfs) == 1:
            return ctrl.minreal(tfs[0], verbose=False)
        return blocos_em_serie(tfs)

    resultado = None

    for con in conexoes:
        tipo_con = con['tipo']
        nomes = con['blocos']

        if not nomes:
            continue

        tfs = [tf_map[n] for n in nomes if n in tf_map]
        if not tfs:
            continue

        if tipo_con == 'Série':
            parcial = blocos_em_serie(tfs)

        elif tipo_con == 'Paralelo':
            if len(tfs) < 2:
                raise ValueError("Conexão em paralelo precisa de pelo menos dois blocos.")
            parcial = blocos_em_paralelo(tfs)

        elif tipo_con == 'Realimentação Negativa':
            if len(tfs) < 2:
                raise ValueError("Realimentação Negativa precisa de G e H.")
            parcial = realimentacao(tfs[0], tfs[1], positiva=False)

        elif tipo_con == 'Realimentação Positiva':
            if len(tfs) < 2:
                raise ValueError("Realimentação Positiva precisa de G e H.")
            parcial = realimentacao(tfs[0], tfs[1], positiva=True)

        else:
            raise ValueError(f"Tipo de conexão desconhecido: {tipo_con}")

        if resultado is None:
            resultado = parcial
        else:
            resultado = ctrl.minreal(resultado * parcial, verbose=False)

    if resultado is None:
        tfs = [tf_map[row['nome']] for _, row in blocos_df.iterrows()]
        resultado = tfs[0] if len(tfs) == 1 else blocos_em_serie(tfs)

    return ctrl.minreal(resultado, verbose=False)


def calcular_malha_fechada(planta, controlador=None, sensor=None, positiva=False):
    if controlador is None:
        controlador = TransferFunction([1], [1])
    if sensor is None:
        sensor = TransferFunction([1], [1])
    G = controlador * planta
    H = sensor
    return realimentacao(G, H, positiva=positiva)


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
    return np.clip(ts_estimado * 1.5, a_min=10, a_max=500)


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
        title='Diagrama de Polos e Zeros', xaxis_title='Parte Real',
        yaxis_title='Parte Imaginaria', showlegend=True, hovermode='closest')
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
    sistema = ctrl.minreal(sistema, verbose=False)

    if not _tf_eh_propria(sistema):
        raise ValueError(
            "A função de transferência é imprópria "
            "(grau do numerador maior que o do denominador). "
            "A resposta no tempo não pode ser calculada diretamente. "
            "Use uma FT própria ou adicione a dinâmica física faltante."
        )

    polos = ctrl.poles(sistema)
    tempo_final = estimar_tempo_final_simulacao(sistema)

    if len(polos) > 0:
        reais_estaveis = [abs(np.real(p)) for p in polos if np.real(p) < 0]
        if reais_estaveis:
            polo_mais_rapido = max(reais_estaveis)
            if polo_mais_rapido > 50:
                tempo_final = min(0.1, tempo_final)

    t = np.linspace(0, tempo_final, 3000)
    u = _gerar_sinal_entrada(entrada, t)

    if entrada == 'Degrau':
        t_out, y = step_response(sistema, T=t)
    else:
        t_out, y = forced_response(sistema, T=t, U=u)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=t_out, y=y, mode='lines',
        line=dict(color='blue'), name='Saída',
        hovertemplate='Tempo: %{x:.5f}s<br>Saída: %{y:.6f}<extra></extra>'))
    fig.update_layout(
        title='Resposta ao Degrau' if entrada == 'Degrau' else f'Resposta Temporal - {entrada}',
        xaxis_title='Tempo (s)', yaxis_title='Saída',
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
        line=dict(dash='dash', color='gray', width=1), name='Reflexo simétrico'))
    fig.add_trace(go.Scatter(
        x=[-1], y=[0], mode='markers',
        marker=dict(symbol='circle', size=12, color='red'),
        name='Ponto crítico (-1,0)'))
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
        if representacao == 'Função de Transferência':
            tf_obj, tf_symb = converter_para_tf(numerador, denominador)
            ss_sys = None
        else:
            A_mat, _ = parse_matrix(A_str)
            if A_mat.shape[0] > 4:
                return False, "Erro: dimensão máxima permitida é 4x4."
            resultado = converter_ss_para_tf(A_str, B_str, C_str, D_str)
            tf_obj = resultado["tf"]
            tf_symb = resultado.get("simbolico", resultado.get("G", None))
            ss_sys = resultado.get("ss", None)
            numerador = " ".join(f"{v:.10g}" for v in tf_obj.num[0][0])
            denominador = " ".join(f"{v:.10g}" for v in tf_obj.den[0][0])

        if not st.session_state.blocos.empty and any(st.session_state.blocos['nome'] == nome):
            return False, f"Erro: já existe um bloco com o nome '{nome}'."

        novo = pd.DataFrame([{
            'nome': nome, 'tipo': tipo, 'representacao': representacao,
            'numerador': numerador, 'denominador': denominador,
            'A': A_str, 'B': B_str, 'C': C_str, 'D': D_str,
            'tf': ctrl.minreal(tf_obj, verbose=False), 'tf_simbolico': tf_symb,
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

    render_guia_popup("Consultar Guia de Estudos")

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
        render_guia_popup("Consultar Guia")

        st.markdown("---")

        tem_planta = not st.session_state.blocos.empty

        if not tem_planta:
            st.header("Planta G(s)")
            st.caption("Insira a função de transferência principal.")
            tipo_bloco_escolhido = 'Planta'
        else:
            st.header("Adicionar Função de Transferência")
            tipo_bloco_escolhido = st.selectbox(
                "Tipo do bloco",
                ['Planta', 'Controlador', 'Sensor', 'Atuador', 'Pre-filtro', 'Perturbação'],
                key="cl_tipo_bloco"
            )

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
            numerador = st.text_input("Numerador", placeholder="ex: 4", key="cl_num")
            denominador = st.text_input("Denominador", placeholder="ex: s^2+2s+3", key="cl_den")
            A_str = B_str = C_str = D_str = ''
        else:
            A_str, B_str, C_str, D_str = _render_ss_widget("sidebar", "cl_sidebar")
            numerador = denominador = ''

        operacao_nova = None
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
            ok, msg = adicionar_bloco(nome, tipo_bloco_escolhido, representacao,
                                      numerador, denominador,
                                      A_str, B_str, C_str, D_str)
            if ok:
                if tem_planta and operacao_nova is not None:
                    nomes_existentes = list(st.session_state.blocos['nome'])
                    bloco_novo = nomes_existentes[-1]
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

        if not st.session_state.blocos.empty:
            st.markdown("---")
            st.subheader("Sistema Atual")

            for idx, row in st.session_state.blocos.iterrows():
                tf_obj = row['tf']
                num_s = _tf_to_str(tf_obj.num[0][0])
                den_s = _tf_to_str(tf_obj.den[0][0])
                op_label = ""
                if idx > 0:
                    for con in st.session_state.conexoes:
                        if row['nome'] in con['blocos'] and con['blocos'][-1] == row['nome']:
                            op_label = f" [{con['tipo']}]"
                            break
                st.text(f"{row['nome']} ({row['tipo']}): ({num_s}) / ({den_s}){op_label}")
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
                        sistema = ctrl.minreal(ganho_tf * sistema, verbose=False)
                    label_extra = " (com conexões)"
                else:
                    planta = obter_bloco_por_tipo('Planta')
                    controlador = obter_bloco_por_tipo('Controlador')
                    sensor = obter_bloco_por_tipo('Sensor')

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
                            planta_com_ganho, controlador, sensor, positiva=False)
                    label_extra = ""

                sistema = ctrl.minreal(sistema, verbose=False)
                st.info(f"{tipo_malha} | K = {K:.2f}{label_extra}")

                num_str = _tf_to_str(sistema.num[0][0])
                den_str = _tf_to_str(sistema.den[0][0])
                st.latex(f"T(s) = \\frac{{{num_str}}}{{{den_str}}}")

                try:
                    ganho_dc = ctrl.dcgain(sistema)
                    st.metric("Ganho DC", f"{ganho_dc:.6f}")
                except Exception:
                    pass

                executar_analises(sistema, analises, entrada, tipo_malha)

            except Exception as e:
                st.error(f"Erro durante a simulação: {e}")


# ══════════════════════════════════════════════════
# EDITOR VISUAL HTML (CANVAS)
# ══════════════════════════════════════════════════

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
