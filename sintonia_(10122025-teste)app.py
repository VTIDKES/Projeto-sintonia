import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import control as ctrl


# ============================================================
# Configuração Streamlit
# ============================================================
st.set_page_config(page_title="XCOS SISO - Controle Clássico", layout="wide")

st.title("XCOS SISO (Malha Fechada) - Controle Clássico")
st.caption("Modelo padrão: R(s) → Σ → C(s) → G(s) → Y(s) com feedback Y(s) → B(s) → Σ")


# ============================================================
# Funções auxiliares
# ============================================================
def parse_poly(text):
    """
    Converte string "1, 2, 3" -> [1,2,3]
    """
    vals = []
    for x in text.split(","):
        x = x.strip()
        if x != "":
            vals.append(float(x))
    return vals

def make_tf(num_str, den_str):
    num = parse_poly(num_str)
    den = parse_poly(den_str)
    if len(num) == 0 or len(den) == 0:
        raise ValueError("Numerador ou denominador vazio.")
    return ctrl.tf(num, den)

def bode_plot(G, wmin=1e-2, wmax=1e2):
    w = np.logspace(np.log10(wmin), np.log10(wmax), 800)
    mag, phase, omega = ctrl.bode_plot(G, w, plot=False)

    mag = np.array(mag).squeeze()
    phase = np.array(phase).squeeze()
    omega = np.array(omega).squeeze()

    mag_db = 20 * np.log10(np.maximum(mag, 1e-300))

    fig1 = plt.figure()
    plt.semilogx(omega, mag_db)
    plt.grid(True, which="both")
    plt.xlabel("ω (rad/s)")
    plt.ylabel("|G(jω)| (dB)")
    plt.title("Bode - Magnitude")

    fig2 = plt.figure()
    plt.semilogx(omega, np.degrees(phase))
    plt.grid(True, which="both")
    plt.xlabel("ω (rad/s)")
    plt.ylabel("Fase (graus)")
    plt.title("Bode - Fase")

    return fig1, fig2

def pz_plot(G):
    poles = np.array(ctrl.poles(G), dtype=complex).flatten()
    zeros = np.array(ctrl.zeros(G), dtype=complex).flatten()

    fig = plt.figure()
    ax = plt.gca()

    if zeros.size > 0:
        ax.scatter(np.real(zeros), np.imag(zeros), marker="o", label="Zeros")
    if poles.size > 0:
        ax.scatter(np.real(poles), np.imag(poles), marker="x", label="Polos")

    ax.axhline(0, linewidth=1)
    ax.axvline(0, linewidth=1)
    ax.grid(True)
    ax.set_xlabel("Parte Real")
    ax.set_ylabel("Parte Imaginária")
    ax.legend()
    ax.set_title("Plano-s (Polos e Zeros)")

    return fig, poles, zeros

def time_response(G, kind, tmax, amp, freq_hz, duty, custom_expr=None):
    t = np.linspace(0, float(tmax), 2500)

    if kind == "Degrau":
        tout, y = ctrl.step_response(G, T=t)
        u = amp * np.ones_like(tout)

    elif kind == "Impulso":
        tout, y = ctrl.impulse_response(G, T=t)
        u = None

    elif kind == "Rampa":
        u = amp * t
        tout, y, _ = ctrl.forced_response(G, T=t, U=u)

    elif kind == "Seno":
        u = amp * np.sin(2*np.pi*freq_hz*t)
        tout, y, _ = ctrl.forced_response(G, T=t, U=u)

    elif kind == "Pulso":
        period = 1.0 / max(freq_hz, 1e-9)
        u = amp * (((t % period) / period) < duty).astype(float)
        tout, y, _ = ctrl.forced_response(G, T=t, U=u)

    elif kind == "Custom":
        safe_globals = {"np": np}
        safe_locals = {"t": t}
        u = eval(custom_expr, safe_globals, safe_locals)
        u = np.array(u, dtype=float).flatten()

        if u.shape[0] != t.shape[0]:
            raise ValueError("u(t) deve ter o mesmo tamanho do vetor t.")

        tout, y, _ = ctrl.forced_response(G, T=t, U=u)

    else:
        raise ValueError("Tipo de entrada inválida.")

    fig = plt.figure()
    plt.plot(tout, y, label="y(t)")
    plt.grid(True)
    plt.xlabel("t (s)")
    plt.ylabel("Saída")
    plt.title(f"Resposta no tempo - {kind}")

    if u is not None:
        plt.plot(tout, u, "--", label="u(t)")
        plt.legend()

    return fig


# ============================================================
# Sidebar: Configuração do Sistema
# ============================================================
st.sidebar.header("Configuração do Sistema")

usar_controlador = st.sidebar.checkbox("Incluir Controlador C(s)", value=True)
usar_feedback = st.sidebar.checkbox("Incluir Feedback B(s)", value=True)

tipo_feedback = st.sidebar.selectbox("Tipo de Feedback", ["Negativo (-)", "Positivo (+)"], index=0)

st.sidebar.divider()

st.sidebar.subheader("Planta G(s)")
G_num = st.sidebar.text_input("Numerador G(s)", value="1")
G_den = st.sidebar.text_input("Denominador G(s)", value="1, 1")

st.sidebar.divider()

if usar_controlador:
    st.sidebar.subheader("Controlador C(s)")
    C_num = st.sidebar.text_input("Numerador C(s)", value="1")
    C_den = st.sidebar.text_input("Denominador C(s)", value="1")
else:
    C_num, C_den = "1", "1"

st.sidebar.divider()

if usar_feedback:
    st.sidebar.subheader("Sensor/Feedback B(s)")
    B_num = st.sidebar.text_input("Numerador B(s)", value="1")
    B_den = st.sidebar.text_input("Denominador B(s)", value="1")
else:
    B_num, B_den = "1", "1"

st.sidebar.divider()

st.sidebar.subheader("Entrada e Simulação")

entrada = st.sidebar.selectbox(
    "Tipo de entrada",
    ["Degrau", "Impulso", "Rampa", "Seno", "Pulso", "Custom"],
    index=0
)

tmax = st.sidebar.number_input("Tempo máximo (s)", min_value=0.1, value=10.0, step=0.5)
amp = st.sidebar.number_input("Amplitude", value=1.0, step=0.1)

freq_hz = 1.0
duty = 0.5
custom_expr = "np.ones_like(t)"

if entrada in ["Seno", "Pulso"]:
    freq_hz = st.sidebar.number_input("Frequência (Hz)", min_value=0.0001, value=1.0, step=0.1)

if entrada == "Pulso":
    duty = st.sidebar.slider("Duty Cycle", min_value=0.05, max_value=0.95, value=0.5, step=0.05)

if entrada == "Custom":
    custom_expr = st.sidebar.text_area("u(t) =", value="2*np.sin(2*np.pi*1*t)")

st.sidebar.divider()

st.sidebar.subheader("Bode")
wmin = st.sidebar.number_input("ω min", min_value=1e-6, value=1e-2, format="%.6f")
wmax = st.sidebar.number_input("ω max", min_value=1e-6, value=1e2, format="%.6f")


# ============================================================
# Construção do sistema
# ============================================================
try:
    Gs = make_tf(G_num, G_den)
    Cs = make_tf(C_num, C_den)
    Bs = make_tf(B_num, B_den)

    # Malha aberta
    L = ctrl.series(Cs, Gs)

    # Feedback
    if usar_feedback:
        if tipo_feedback == "Negativo (-)":
            T = ctrl.feedback(L, Bs, sign=-1)   # padrão de controle
        else:
            T = ctrl.feedback(L, Bs, sign=+1)
    else:
        T = L

except Exception as e:
    st.error(f"Erro na criação do sistema: {e}")
    st.stop()


# ============================================================
# Mostrar diagrama (texto)
# ============================================================
st.subheader("Estrutura do Sistema (Modelo Padrão)")

st.latex(r"R(s) \rightarrow \Sigma \rightarrow C(s) \rightarrow G(s) \rightarrow Y(s)")
st.latex(r"Y(s) \rightarrow B(s) \rightarrow \Sigma")

if tipo_feedback == "Negativo (-)":
    st.write("🔁 Feedback configurado como: **NEGATIVO**")
else:
    st.write("🔁 Feedback configurado como: **POSITIVO**")


# ============================================================
# Resultados
# ============================================================
col1, col2 = st.columns(2)

with col1:
    st.subheader("Funções de Transferência")

    st.write("### Planta G(s)")
    st.code(str(Gs), language="text")

    if usar_controlador:
        st.write("### Controlador C(s)")
        st.code(str(Cs), language="text")

    if usar_feedback:
        st.write("### Feedback B(s)")
        st.code(str(Bs), language="text")

    st.write("### Malha Aberta L(s) = C(s)G(s)")
    st.code(str(L), language="text")

    st.write("### Malha Fechada T(s)")
    st.code(str(T), language="text")

with col2:
    st.subheader("Polos e Zeros da Malha Fechada")

    fig_pz, poles, zeros = pz_plot(T)
    st.pyplot(fig_pz, clear_figure=True)

    st.write("### Polos")
    if poles.size == 0:
        st.write("Nenhum")
    else:
        st.code("\n".join([str(p) for p in poles]), language="text")

    st.write("### Zeros")
    if zeros.size == 0:
        st.write("Nenhum")
    else:
        st.code("\n".join([str(z) for z in zeros]), language="text")


st.divider()

# ============================================================
# Resposta no tempo
# ============================================================
st.subheader("Resposta no Tempo")

try:
    fig_time = time_response(
        T,
        kind=entrada,
        tmax=tmax,
        amp=amp,
        freq_hz=freq_hz,
        duty=duty,
        custom_expr=custom_expr
    )
    st.pyplot(fig_time, clear_figure=True)
except Exception as e:
    st.error(f"Erro na resposta no tempo: {e}")


st.divider()

# ============================================================
# Bode
# ============================================================
st.subheader("Diagramas de Bode (Malha Fechada)")

try:
    fig_mag, fig_phase = bode_plot(T, wmin=wmin, wmax=wmax)
    st.pyplot(fig_mag, clear_figure=True)
    st.pyplot(fig_phase, clear_figure=True)
except Exception as e:
    st.error(f"Erro no Bode: {e}")


st.divider()

# ============================================================
# Informação final
# ============================================================
st.info(
    "Este app implementa o padrão clássico SISO: "
    "R(s) → Σ → C(s) → G(s) → Y(s) com feedback Y(s) → B(s) → Σ. "
    "Ideal para modelagem de sistemas de controle no estilo XCOS/Simulink."
)
