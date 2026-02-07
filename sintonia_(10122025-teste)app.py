import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

from uuid import uuid4
from collections import defaultdict, deque

# Diagrama/flow (React Flow via Streamlit component)
from streamlit_flow import streamlit_flow
from streamlit_flow.elements import StreamlitFlowNode, StreamlitFlowEdge
from streamlit_flow.state import StreamlitFlowState

# Controle clássico
import control as ctrl


# ============================================================
# UI
# ============================================================
st.set_page_config(page_title="Mini-XCOS (Streamlit)", layout="wide")

st.title("Mini-XCOS no Streamlit (SISO em série)")
st.caption(
    "Arraste/conecte blocos (INPUT → ... → OUTPUT). "
    "O app interpreta uma cadeia em série (sem bifurcações) e simula: tempo, Bode, polos e zeros."
)


# ============================================================
# Helpers do Flow
# ============================================================
def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid4()}"

def default_node(node_type: str, x: int, y: int, label: str, params: dict):
    return StreamlitFlowNode(
        id=new_id("node"),
        pos=(x, y),
        data={"label": label, "params": params, "block_type": node_type},
        node_type="default",
        source_position="right",
        target_position="left",
    )

def get_nodes_edges(state: StreamlitFlowState):
    nodes = state.nodes or []
    edges = state.edges or []
    nd = {n.id: n for n in nodes}
    ed = [(e.source, e.target) for e in edges]
    return nd, ed

def find_unique_path(edges, start, end):
    """
    Acha UM caminho start->end em grafo dirigido.
    Se houver 0 ou >1 caminhos, retorna None.
    """
    g = defaultdict(list)
    for s, t in edges:
        g[s].append(t)

    q = deque([(start, [start])])
    found_paths = []

    while q:
        u, path = q.popleft()
        if u == end:
            found_paths.append(path)
            if len(found_paths) > 1:
                return None
            continue

        for v in g[u]:
            if v in path:
                continue
            q.append((v, path + [v]))

    if len(found_paths) == 1:
        return found_paths[0]
    return None


# ============================================================
# Blocos -> Função de Transferência
# ============================================================
def block_to_tf(node: StreamlitFlowNode):
    data = node.data or {}
    btype = data.get("block_type", "GAIN")
    params = data.get("params", {})

    if btype in ("INPUT", "OUTPUT"):
        return ctrl.tf([1], [1])

    if btype == "GAIN":
        k = float(params.get("k", 1.0))
        return ctrl.tf([k], [1])

    if btype == "TF":
        num_s = str(params.get("num", "1"))
        den_s = str(params.get("den", "1, 1"))

        def parse_poly(s):
            vals = []
            for x in s.split(","):
                x = x.strip()
                if x != "":
                    vals.append(float(x))
            return vals

        num = parse_poly(num_s)
        den = parse_poly(den_s)

        if len(num) == 0 or len(den) == 0:
            raise ValueError("TF inválida: num/den vazios.")

        return ctrl.tf(num, den)

    # fallback
    return ctrl.tf([1], [1])

def series_tf(nodes_in_path, node_map):
    G = ctrl.tf([1], [1])
    for nid in nodes_in_path:
        G = ctrl.series(G, block_to_tf(node_map[nid]))
    return G


# ============================================================
# Plots e análises
# ============================================================
def bode_plot(G, wmin=1e-2, wmax=1e2):
    w = np.logspace(np.log10(wmin), np.log10(wmax), 600)
    mag, phase, omega = ctrl.bode_plot(G, w, plot=False)

    # mag pode vir como array 3D em algumas versões; achatamos
    mag = np.array(mag).squeeze()
    phase = np.array(phase).squeeze()
    omega = np.array(omega).squeeze()

    mag_db = 20 * np.log10(np.maximum(mag, 1e-300))

    fig1 = plt.figure()
    plt.semilogx(omega, mag_db)
    plt.xlabel("ω (rad/s)")
    plt.ylabel("|G(jω)| (dB)")
    plt.grid(True, which="both")
    plt.title("Bode — Magnitude")

    fig2 = plt.figure()
    plt.semilogx(omega, np.degrees(phase))
    plt.xlabel("ω (rad/s)")
    plt.ylabel("∠G(jω) (graus)")
    plt.grid(True, which="both")
    plt.title("Bode — Fase")

    return fig1, fig2

def pz_data(G):
    # Compatível com várias versões do python-control
    try:
        poles = ctrl.poles(G)
    except Exception:
        poles = ctrl.pole(G)

    try:
        zeros = ctrl.zeros(G)
    except Exception:
        zeros = ctrl.zero(G)

    poles = np.array(poles, dtype=complex).flatten()
    zeros = np.array(zeros, dtype=complex).flatten()
    return poles, zeros

def pz_plot(poles, zeros):
    fig = plt.figure()
    ax = plt.gca()

    if zeros.size > 0:
        ax.scatter(np.real(zeros), np.imag(zeros), marker="o", label="Zeros")
    if poles.size > 0:
        ax.scatter(np.real(poles), np.imag(poles), marker="x", label="Polos")

    ax.axhline(0, linewidth=1)
    ax.axvline(0, linewidth=1)
    ax.set_xlabel("Re")
    ax.set_ylabel("Im")
    ax.grid(True, which="both")
    ax.legend()
    ax.set_title("Plano-s (Polos e Zeros)")
    return fig

def time_response(G, kind="Degrau", tmax=10.0, n=2000, amp=1.0, freq_hz=1.0, duty=0.5):
    """
    Resposta no tempo:
      - Degrau, Impulso, Rampa, Seno, Pulso
    Retorna figura.
    """
    t = np.linspace(0, float(tmax), int(n))

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
        u = amp * np.sin(2 * np.pi * float(freq_hz) * t)
        tout, y, _ = ctrl.forced_response(G, T=t, U=u)

    elif kind == "Pulso":
        # Pulso quadrado (freq em Hz) com duty cycle
        period = 1.0 / max(float(freq_hz), 1e-9)
        u = amp * (((t % period) / period) < float(duty)).astype(float)
        tout, y, _ = ctrl.forced_response(G, T=t, U=u)

    else:
        raise ValueError("kind inválido.")

    fig = plt.figure()
    plt.plot(tout, y, label="y(t)")
    plt.xlabel("t (s)")
    plt.ylabel("Saída")
    plt.grid(True)

    if u is not None:
        plt.plot(tout, u, "--", label="u(t)")
        plt.legend()

    plt.title(f"Resposta no tempo — {kind}")
    return fig


# ============================================================
# Inicializa estado do diagrama
# ============================================================
if "flow_state" not in st.session_state:
    nodes = []

    n_in = StreamlitFlowNode(
        id="IN",
        pos=(50, 120),
        data={"label": "INPUT", "params": {}, "block_type": "INPUT"},
        node_type="input",
        source_position="right",
        target_position="left",
    )
    n_out = StreamlitFlowNode(
        id="OUT",
        pos=(900, 120),
        data={"label": "OUTPUT", "params": {}, "block_type": "OUTPUT"},
        node_type="output",
        source_position="right",
        target_position="left",
    )
    nodes.extend([n_in, n_out])

    g0 = default_node("GAIN", 380, 120, "GAIN (k=2)", {"k": 2.0})
    nodes.append(g0)

    edges = [
        StreamlitFlowEdge(id="e1", source="IN", target=g0.id, animated=True),
        StreamlitFlowEdge(id="e2", source=g0.id, target="OUT", animated=True),
    ]

    st.session_state.flow_state = StreamlitFlowState(key="flow", nodes=nodes, edges=edges)


# ============================================================
# Sidebar
# ============================================================
with st.sidebar:
    st.subheader("Blocos")
    st.write("Adicione blocos e conecte no diagrama (cadeia em série).")

    c1, c2 = st.columns(2)
    with c1:
        if st.button("➕ GAIN"):
            st.session_state.flow_state.nodes.append(
                default_node("GAIN", 380, 260, "GAIN (k=1)", {"k": 1.0})
            )
            st.rerun()

    with c2:
        if st.button("➕ TF"):
            st.session_state.flow_state.nodes.append(
                default_node("TF", 380, 400, "TF [1]/[1,1]", {"num": "1", "den": "1, 1"})
            )
            st.rerun()

    st.divider()
    st.subheader("Editar bloco (pelo ID)")
    st.caption("Use o painel 'Estado (debug)' para ver o ID do nó (ou copie e cole).")

    edit_id = st.text_input("ID do nó", value="")
    if edit_id:
        node_map, _ = get_nodes_edges(st.session_state.flow_state)
        if edit_id in node_map:
            n = node_map[edit_id]
            data = n.data or {}
            btype = data.get("block_type", "GAIN")
            st.write(f"Tipo: **{btype}**")
            params = data.get("params", {})

            if btype == "GAIN":
                k = st.number_input("k", value=float(params.get("k", 1.0)))
                n.data["params"] = {"k": float(k)}
                n.data["label"] = f"GAIN (k={k:g})"
                if st.button("Salvar"):
                    st.rerun()

            elif btype == "TF":
                num = st.text_input("num (ex: 1, 2)", value=str(params.get("num", "1")))
                den = st.text_input("den (ex: 1, 2, 1)", value=str(params.get("den", "1, 1")))
                n.data["params"] = {"num": num, "den": den}
                n.data["label"] = f"TF [{num}] / [{den}]"
                if st.button("Salvar"):
                    st.rerun()
            else:
                st.info("INPUT/OUTPUT não são editáveis.")
        else:
            st.warning("ID não encontrado no diagrama.")

    st.divider()
    st.subheader("Simulação")

    entrada = st.selectbox(
        "Entrada para resposta no tempo",
        ["Degrau", "Impulso", "Rampa", "Seno", "Pulso", "Custom"],
        index=0,
    )

    tmax = st.number_input("Tempo máx (s)", min_value=0.1, value=10.0, step=0.5)
    amp = st.number_input("Amplitude", value=1.0, step=0.1)

    freq_hz = 1.0
    duty = 0.5
    custom_expr = "np.ones_like(t)"

    if entrada in ["Seno", "Pulso"]:
        freq_hz = st.number_input("Frequência (Hz)", min_value=0.0001, value=1.0, step=0.1)
    if entrada == "Pulso":
        duty = st.slider("Duty cycle", min_value=0.05, max_value=0.95, value=0.5, step=0.05)
    if entrada == "Custom":
        st.caption("Escreva u(t) usando numpy. Variável disponível: t (array). Ex: 2*np.sin(2*np.pi*1*t)")
        custom_expr = st.text_area("u(t) =", value="2*np.sin(2*np.pi*1*t)")

    st.divider()
    st.subheader("Bode")
    wmin = st.number_input("Bode ω min", min_value=1e-6, value=1e-2, format="%.6f")
    wmax = st.number_input("Bode ω max", min_value=1e-6, value=1e2, format="%.6f")


# ============================================================
# Layout principal
# ============================================================
left, right = st.columns([1.25, 1])

with left:
    st.subheader("Diagrama")
    st.session_state.flow_state = streamlit_flow("flow", st.session_state.flow_state)

    with st.expander("Estado (debug)"):
        st.json(
            {
                "nodes": [
                    {
                        "id": n.id,
                        "label": (n.data or {}).get("label"),
                        "block_type": (n.data or {}).get("block_type"),
                    }
                    for n in st.session_state.flow_state.nodes
                ],
                "edges": [
                    {"source": e.source, "target": e.target}
                    for e in st.session_state.flow_state.edges
                ],
            }
        )

with right:
    st.subheader("Resultados")

    node_map, edge_list = get_nodes_edges(st.session_state.flow_state)

    if "IN" not in node_map or "OUT" not in node_map:
        st.error("Faltou INPUT/OUTPUT no diagrama.")
        st.stop()

    path = find_unique_path(edge_list, "IN", "OUT")
    if path is None:
        st.error(
            "Não consegui achar um caminho ÚNICO IN → OUT.\n\n"
            "✅ Deixe apenas uma cadeia em série (sem bifurcações/paralelo/loops)."
        )
        st.stop()

    middle = [nid for nid in path if nid not in ("IN", "OUT")]

    try:
        G = series_tf(middle, node_map)
        st.write("**Função de Transferência equivalente:**")
        st.code(str(G), language="text")
    except Exception as e:
        st.error(f"Erro montando a função de transferência: {e}")
        st.stop()

    # -------------------------
    # Resposta no tempo (inclui degrau)
    # -------------------------
    try:
        if entrada == "Custom":
            t = np.linspace(0, float(tmax), 2000)

            # Segurança mínima: só permite np e t
            safe_globals = {"np": np}
            safe_locals = {"t": t}

            u = eval(custom_expr, safe_globals, safe_locals)
            u = np.array(u, dtype=float).flatten()

            if u.shape[0] != t.shape[0]:
                raise ValueError("u(t) deve ter o mesmo tamanho de t.")

            tout, y, _ = ctrl.forced_response(G, T=t, U=u)

            fig_time = plt.figure()
            plt.plot(tout, y, label="y(t)")
            plt.plot(tout, u, "--", label="u(t)")
            plt.xlabel("t (s)")
            plt.ylabel("Saída")
            plt.grid(True)
            plt.legend()
            plt.title("Resposta no tempo — Custom")
            st.pyplot(fig_time, clear_figure=True)

        else:
            fig_time = time_response(G, kind=entrada, tmax=tmax, amp=amp, freq_hz=freq_hz, duty=duty)
            st.pyplot(fig_time, clear_figure=True)

    except Exception as e:
        st.error(f"Erro na resposta no tempo: {e}")
        st.stop()

    # -------------------------
    # Bode
    # -------------------------
    try:
        fig_mag, fig_phase = bode_plot(G, wmin=wmin, wmax=wmax)
        st.pyplot(fig_mag, clear_figure=True)
        st.pyplot(fig_phase, clear_figure=True)
    except Exception as e:
        st.error(f"Erro no Bode: {e}")

    # -------------------------
    # Polos e Zeros
    # -------------------------
    try:
        poles, zeros = pz_data(G)

        cA, cB = st.columns(2)
        with cA:
            st.write("**Polos:**")
            if poles.size == 0:
                st.write("Nenhum")
            else:
                st.code("\n".join([str(p) for p in poles]), language="text")

        with cB:
            st.write("**Zeros:**")
            if zeros.size == 0:
                st.write("Nenhum")
            else:
                st.code("\n".join([str(z) for z in zeros]), language="text")

        fig_pz = pz_plot(poles, zeros)
        st.pyplot(fig_pz, clear_figure=True)

    except Exception as e:
        st.error(f"Erro em polos/zeros: {e}")
