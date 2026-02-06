# -*- coding: utf-8 -*-
"""
Sistema de Modelagem e Análise de Sistemas de Controle
Otimizado para Streamlit Cloud

✅ Reformulação para funcionar no Streamlit Cloud SEM depender de streamlit-flow-component.
Motivo: em alguns deploys o streamlit-flow-component não instala/importa no Cloud.

✅ O que foi adicionado:
- Uma NOVA ABA "🧩 Editor Visual" com drag & drop usando streamlit-sortables.
- O usuário arrasta os blocos para definir a ORDEM da SÉRIE (estilo "cadeia" Simulink).
- A Simulação em "Malha Aberta" usa automaticamente essa ordem se estiver definida.

Dependência extra (Streamlit Cloud):
- streamlit-sortables  (pip install streamlit-sortables)
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
# EDITOR VISUAL (drag & drop) - streamlit-sortables
# =====================================================
SORTABLES_AVAILABLE = True
SORTABLES_IMPORT_ERROR = None

try:
    from streamlit_sortables import sort_items
except Exception as e:
    SORTABLES_AVAILABLE = False
    SORTABLES_IMPORT_ERROR = str(e)
    sort_items = None

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
    G_min = ctrl.minreal(G, verbose=False)
    polos = ctrl.poles(G_min)
    tipo = sum(1 for p in polos if np.isclose(np.real_if_close(p), 0.0, atol=1e-3))
    return tipo

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

def calcular_malha_fechada(planta, controlador=None, sensor=None):
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
    polos_ordenados = sorted(polos, key=lambda p: np.real(p), reverse=True)
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
    fig.update_layout(
        dragmode='zoom',
        newshape=dict(line=dict(color='green', width=2, dash='dash')),
        modebar_add=['drawline','drawopenpath','drawclosedpath','drawcircle','drawrect','eraseshape']
    )
    return fig

def plot_polos_zeros(tf, fig=None):
    zeros = ctrl.zeros(tf)
    polos = ctrl.poles(tf)
    if fig is None:
        fig = go.Figure()

    if len(zeros) > 0:
        fig.add_trace(go.Scatter(
            x=np.real(zeros), y=np.imag(zeros),
            mode='markers', marker=dict(symbol='circle', size=12, color='blue'),
            name='Zeros',
            hovertemplate='Zero<br>Real: %{x:.3f}<br>Imaginário: %{y:.3f}<extra></extra>'
        ))

    if len(polos) > 0:
        fig.add_trace(go.Scatter(
            x=np.real(polos), y=np.imag(polos),
            mode='markers', marker=dict(symbol='x', size=12, color='red'),
            name='Polos',
            hovertemplate='Polo<br>Real: %{x:.3f}<br>Imaginário: %{y:.3f}<extra></extra>'
        ))

    fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.7)
    fig.add_vline(x=0, line_dash="dash", line_color="gray", opacity=0.7)

    fig.update_layout(
        title='Diagrama de Polos e Zeros (Interativo)',
        xaxis_title='Parte Real', yaxis_title='Parte Imaginária',
        showlegend=True, hovermode='closest'
    )
    return configurar_linhas_interativas(fig)

def _gerar_sinal_entrada(entrada, t):
    sinais = {
        'Degrau': np.ones_like(t),
        'Rampa': t,
        'Senoidal': np.sin(2*np.pi*t),
        'Impulso': np.concatenate([[1], np.zeros(len(t)-1)]),
        'Parabólica': t**2
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
    fig.add_trace(go.Scatter(x=t_out, y=u[:len(t_out)], mode='lines',
                             line=dict(dash='dash', color='blue'),
                             name='Entrada',
                             hovertemplate='Tempo: %{x:.2f}s<br>Entrada: %{y:.3f}<extra></extra>'))
    fig.add_trace(go.Scatter(x=t_out, y=y, mode='lines',
                             line=dict(color='red'),
                             name='Saída',
                             hovertemplate='Tempo: %{x:.2f}s<br>Saída: %{y:.3f}<extra></extra>'))
    fig.update_layout(
        title=f'Resposta Temporal - Entrada: {entrada}',
        xaxis_title='Tempo (s)', yaxis_title='Amplitude',
        showlegend=True, hovermode='x unified'
    )
    return configurar_linhas_interativas(fig), t_out, y

def plot_bode(sistema, tipo='both'):
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
        fig.add_trace(go.Scatter(x=w, y=mag, mode='lines', name='Magnitude', showlegend=False), row=1, col=1)
        fig.add_trace(go.Scatter(x=w, y=phase, mode='lines', name='Fase', showlegend=False), row=2, col=1)
        fig.update_xaxes(title_text="Frequência (rad/s)", type="log", row=1, col=1)
        fig.update_xaxes(title_text="Frequência (rad/s)", type="log", row=2, col=1)
        fig.update_yaxes(title_text="Magnitude (dB)", row=1, col=1)
        fig.update_yaxes(title_text="Fase (deg)", row=2, col=1)
        fig.update_layout(height=700, title_text="Diagrama de Bode")
        return configurar_linhas_interativas(fig)

    if tipo == 'magnitude':
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=w, y=mag, mode='lines', name='Magnitude'))
        fig.update_layout(title='Diagrama de Bode - Magnitude', xaxis_title="Frequência (rad/s)",
                          yaxis_title="Magnitude (dB)", xaxis_type='log')
        return configurar_linhas_interativas(fig)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=w, y=phase, mode='lines', name='Fase'))
    fig.update_layout(title='Diagrama de Bode - Fase', xaxis_title="Frequência (rad/s)",
                      yaxis_title="Fase (deg)", xaxis_type='log')
    return configurar_linhas_interativas(fig)

def plot_lgr(sistema):
    rlist, klist = root_locus(sistema, plot=False)
    fig = go.Figure()
    for r in rlist.T:
        fig.add_trace(go.Scatter(x=np.real(r), y=np.imag(r), mode='lines', showlegend=False))
    zeros = ctrl.zeros(sistema)
    polos = ctrl.poles(sistema)
    if len(zeros) > 0:
        fig.add_trace(go.Scatter(x=np.real(zeros), y=np.imag(zeros), mode='markers',
                                 marker=dict(symbol='circle', size=10, color='green'), name='Zeros'))
    if len(polos) > 0:
        fig.add_trace(go.Scatter(x=np.real(polos), y=np.imag(polos), mode='markers',
                                 marker=dict(symbol='x', size=12, color='red'), name='Polos'))
    fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.7)
    fig.add_vline(x=0, line_dash="dash", line_color="gray", opacity=0.7)
    fig.update_layout(title='Lugar Geométrico das Raízes (LGR)', xaxis_title='Parte Real',
                      yaxis_title='Parte Imaginária', showlegend=True, hovermode='closest')
    return configurar_linhas_interativas(fig)

def plot_nyquist(sistema):
    sistema_scipy = signal.TransferFunction(sistema.num[0][0], sistema.den[0][0])
    w = np.logspace(-2, 2, 1000)
    _, H = signal.freqresp(sistema_scipy, w)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=H.real, y=H.imag, mode='lines', name='Nyquist'))
    fig.add_trace(go.Scatter(x=H.real, y=-H.imag, mode='lines', name='Reflexo', line=dict(dash='dash')))
    fig.add_trace(go.Scatter(x=[-1], y=[0], mode='markers', name='(-1,0)', marker=dict(size=12, color='red')))
    fig.add_hline(y=0, line_color="black", line_width=1)
    fig.add_vline(x=0, line_color="black", line_width=1)
    fig.update_layout(title='Diagrama de Nyquist', xaxis_title='Parte Real', yaxis_title='Parte Imaginária',
                      showlegend=True, hovermode='closest')
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
    if 'blocos' not in st.session_state:
        st.session_state.blocos = pd.DataFrame(columns=['nome', 'tipo', 'numerador', 'denominador', 'tf', 'tf_simbolico'])
    if 'serie_order' not in st.session_state:
        st.session_state.serie_order = []

def adicionar_bloco(nome, tipo, numerador, denominador):
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
    st.session_state.blocos = st.session_state.blocos[st.session_state.blocos['nome'] != nome]
    st.session_state.serie_order = [x for x in st.session_state.serie_order if x != nome]
    return f"Bloco {nome} excluído."

def obter_bloco_por_tipo(tipo):
    df = st.session_state.blocos
    if any(df['tipo'] == tipo):
        return df[df['tipo'] == tipo].iloc[0]['tf']
    return None

def montar_serie_pela_ordem(df: pd.DataFrame, ordem_nomes: list):
    if df.empty or not ordem_nomes:
        return None
    mapa = {row['nome']: row['tf'] for _, row in df.iterrows()}
    serie = None
    for nome in ordem_nomes:
        tf = mapa.get(nome)
        if tf is None:
            continue
        serie = tf if serie is None else (serie * tf)
    return serie

# =====================================================
# TAB: EDITOR VISUAL (drag&drop)
# =====================================================


def _plot_diagrama_serie(ordem_nomes, df):
    """Gera um diagrama simples de blocos em série usando Plotly (sem dependências extras)."""
    if not ordem_nomes:
        fig = go.Figure()
        fig.update_layout(
            title="Diagrama (vazio)",
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            height=260,
            margin=dict(l=10, r=10, t=40, b=10),
        )
        return fig

    fig = go.Figure()
    fig.update_layout(
        title="Diagrama de Blocos (Série)",
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        height=260,
        margin=dict(l=10, r=10, t=40, b=10),
    )

    # Mapas rápidos
    row_by_nome = {str(r["nome"]): r for _, r in df.iterrows()}

    # Geometria
    start_x = 0.05
    y = 0.5
    box_w = 0.16
    box_h = 0.22
    gap = 0.06

    # Entrada e saída
    fig.add_annotation(x=0.01, y=y, text="R(s)", showarrow=False, font=dict(size=14))
    fig.add_annotation(x=0.99, y=y, text="Y(s)", showarrow=False, font=dict(size=14))

    # Setas
    def arrow(x0, x1):
        fig.add_annotation(
            x=x1, y=y, ax=x0, ay=y,
            xref="paper", yref="paper", axref="paper", ayref="paper",
            showarrow=True, arrowhead=3, arrowsize=1, arrowwidth=2
        )

    x = start_x
    # seta de entrada para o primeiro bloco
    arrow(0.02, x)

    for i, nome in enumerate(ordem_nomes):
        r = row_by_nome.get(nome)
        label = nome if r is None else f"{nome}\n{r['tipo']}"
        # caixa
        fig.add_shape(
            type="rect",
            xref="paper", yref="paper",
            x0=x, y0=y - box_h/2,
            x1=x + box_w, y1=y + box_h/2,
            line=dict(width=2),
            fillcolor="rgba(240,240,240,0.9)",
        )
        fig.add_annotation(
            x=x + box_w/2, y=y,
            xref="paper", yref="paper",
            text=label.replace("\n", "<br>"),
            showarrow=False,
            font=dict(size=12),
        )

        # seta para o próximo
        next_x = x + box_w + gap
        if i < len(ordem_nomes) - 1:
            arrow(x + box_w, next_x)
        else:
            # seta do último bloco para a saída
            arrow(x + box_w, 0.98)

        x = next_x

    return fig

def editar_bloco_inline(nome_bloco: str):
    """Editor rápido inline do bloco (mantém bloco original, só altera num/den)."""
    df = st.session_state.blocos
    if df.empty or nome_bloco not in df["nome"].astype(str).tolist():
        return

    idx = df.index[df["nome"].astype(str) == str(nome_bloco)][0]
    row = df.loc[idx]

    with st.form(f"form_edit_{nome_bloco}", clear_on_submit=False):
        st.write(f"✏️ Editando **{nome_bloco}** ({row['tipo']})")
        c1, c2 = st.columns(2)
        with c1:
            num = st.text_input("Numerador", value=str(row["numerador"]))
        with c2:
            den = st.text_input("Denominador", value=str(row["denominador"]))
        ok = st.form_submit_button("Salvar alteração")
        if ok:
            try:
                tf, tf_symb = converter_para_tf(num, den)
                st.session_state.blocos.loc[idx, "numerador"] = num
                st.session_state.blocos.loc[idx, "denominador"] = den
                st.session_state.blocos.loc[idx, "tf"] = tf
                st.session_state.blocos.loc[idx, "tf_simbolico"] = tf_symb
                st.success("✅ Bloco atualizado.")
            except Exception as e:
                st.error(f"Erro ao atualizar TF: {e}")

def render_tab_editor_visual():
    st.subheader("🧩 Editor Visual (mais dinâmico) — Montagem em Série")

    df = st.session_state.blocos
    if df.empty:
        st.info("Adicione blocos na barra lateral para aparecerem aqui.")
        return

    # Inicializa ordem (se necessário) e mantém sincronizado
    nomes = [str(row['nome']) for _, row in df.iterrows()]
    if not st.session_state.serie_order:
        st.session_state.serie_order = nomes.copy()
    else:
        atual = [n for n in st.session_state.serie_order if n in nomes]
        for n in nomes:
            if n not in atual:
                atual.append(n)
        st.session_state.serie_order = atual

    # Layout do editor: Paleta + Montagem + Preview
    left, right = st.columns([1.05, 1.95], gap="large")

    with left:
        st.markdown("### 🎛️ Paleta de blocos")
        st.caption("Clique para editar. Você também pode remover blocos aqui.")

        # Lista rápida
        for _, row in df.iterrows():
            nome = str(row["nome"])
            tipo = str(row["tipo"])
            with st.expander(f"🔹 {nome}  ·  {tipo}", expanded=False):
                st.code(f"G(s) = ({row['numerador']}) / ({row['denominador']})")
                b1, b2 = st.columns(2)
                with b1:
                    if st.button("✏️ Editar", key=f"btn_edit_{nome}"):
                        st.session_state["_edit_target"] = nome
                with b2:
                    if st.button("🗑️ Remover", key=f"btn_rm_{nome}"):
                        remover_bloco(nome)
                        st.rerun()

        if st.session_state.get("_edit_target"):
            st.markdown("---")
            editar_bloco_inline(st.session_state["_edit_target"])

    with right:
        st.markdown("### 🧱 Montagem (drag & drop)")
        st.caption("Arraste para definir a ordem da série. A **Malha Aberta** usará essa ordem automaticamente.")

        if not SORTABLES_AVAILABLE:
            st.error("O componente de drag&drop não carregou (streamlit-sortables).")
            st.write("Coloque isto no requirements.txt:")
            st.code("streamlit-sortables")
            st.write("Detalhe do erro de import:")
            st.code(SORTABLES_IMPORT_ERROR or "Sem detalhes.")
            return

        label_by_nome = {
            str(row['nome']): f"{row['nome']}  |  {row['tipo']}  |  {row['numerador']}/{row['denominador']}"
            for _, row in df.iterrows()
        }
        ordered_labels = [label_by_nome[n] for n in st.session_state.serie_order if n in label_by_nome]

        new_labels = sort_items(ordered_labels, direction="vertical")

        new_order = []
        for lab in new_labels:
            nome = lab.split("|")[0].strip()
            if nome in nomes and nome not in new_order:
                new_order.append(nome)

        st.session_state.serie_order = new_order

        c1, c2, c3 = st.columns([1, 1, 2])
        with c1:
            if st.button("🔁 Resetar ordem"):
                st.session_state.serie_order = nomes.copy()
                st.rerun()
        with c2:
            st.metric("Blocos na série", len(st.session_state.serie_order))
        with c3:
            st.write("**Ordem atual:**")
            st.write(" → ".join(st.session_state.serie_order) if st.session_state.serie_order else "(vazia)")

        st.markdown("### 👀 Preview do diagrama")
        fig_diag = _plot_diagrama_serie(st.session_state.serie_order, df)
        st.plotly_chart(fig_diag, use_container_width=True)

        st.markdown("### 🧮 Função equivalente (série)")
        serie = montar_serie_pela_ordem(df, st.session_state.serie_order)
        if serie is None:
            st.info("Defina a ordem (ou adicione blocos) para gerar a função equivalente.")
        else:
            try:
                # Mostra como TF e resumo
                st.code(f"{serie}")
                st.caption("Obs.: essa composição é aplicada automaticamente quando você escolhe **Malha Aberta** na aba Simulação.")
            except Exception:
                st.write(serie)


def main():
    st.set_page_config(page_title="Modelagem de Sistemas", layout="wide")
    st.title("📉 Modelagem e Análise de Sistemas de Controle")

    inicializar_blocos()

    if 'calculo_erro_habilitado' not in st.session_state:
        st.session_state.calculo_erro_habilitado = False

    if 'mostrar_ajuda' not in st.session_state:
        st.session_state.mostrar_ajuda = False

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

    tab_sim, tab_editor = st.tabs(["📈 Simulação", "🧩 Editor Visual"])

    with tab_editor:
        render_tab_editor_visual()

    with tab_sim:
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
                        st.session_state.serie_order = [n for n in st.session_state.serie_order if n in st.session_state.blocos['nome'].tolist()]
                        st.success("Plantas removidas com sucesso!")
                    else:
                        st.warning("Nenhuma planta para remover")
        else:
            st.info("💡 Use o botão 'Habilitar Cálculo de Erro' na barra lateral para ativar esta funcionalidade")

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

                        if tipo_malha == "Malha Aberta":
                            serie = montar_serie_pela_ordem(df, st.session_state.get("serie_order", []))
                            if serie is not None:
                                sistema = ganho_tf * serie
                                st.info("🧩 Usando ordem do Editor Visual (série) na Malha Aberta.")
                            else:
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
                    6. **🧩 Editor Visual**: Use a aba Editor Visual para arrastar a ordem da série (Malha Aberta)
                    """)

                with col_guide2:
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

    st.sidebar.markdown("---")
    st.sidebar.info("""
    🧩 Use a aba **Editor Visual** para montar uma cadeia em série (Malha Aberta)
    com **drag & drop**.
    """)

if __name__ == "__main__":
    main()
