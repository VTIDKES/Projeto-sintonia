import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

from uuid import uuid4
from collections import defaultdict, deque

from streamlit_flow import streamlit_flow
from streamlit_flow.elements import StreamlitFlowNode, StreamlitFlowEdge
from streamlit_flow.state import StreamlitFlowState

import control as ctrl


# ============================================================
# UI
# ============================================================
st.set_page_config(page_title="XCOS SISO (Desenhável)", layout="wide")
st.title("XCOS SISO (Desenhável) — Streamlit Cloud")
st.caption(
    "Monte seu diagrama com blocos e conexões (SISO). "
    "O app interpreta 1 somador (SUM) e calcula: FT, tempo, Bode, polos e zeros."
)


# ============================================================
# Compatibilidade com versões diferentes do streamlit-flow-component
# ============================================================
def make_flow_state(nodes, edges, key="flow"):
    try:
        return StreamlitFlowState(key=key, nodes=nodes, edges=edges)
    except TypeError:
        try:
            return StreamlitFlowState(nodes, edges, key=key)
        except TypeError:
            s = StreamlitFlowState(nodes, edges)
            try:
                s.key = key
            except Exception:
                pass
            return s

def run_flow(key, state):
    try:
        return streamlit_flow(key, state)
    except TypeError:
        # versões antigas: streamlit_flow(key, nodes, edges)
        return streamlit_flow(key, state.nodes, state.edges)


# ============================================================
# Blocos disponíveis (tipo parecido com seu React)
# ============================================================
BLOCK_LIBRARY = [
    {"type": "IN", "name": "R(s) - Referência"},
    {"type": "SUM", "name": "∑ - Somador"},
    {"type": "P", "name": "P - Proporcional"},
    {"type": "PID", "name": "PID"},
    {"type": "PLANT_1", "name": "Planta 1ª Ordem"},
    {"type": "PLANT_2", "name": "Planta 2ª Ordem"},
    {"type": "FB", "name": "B(s) - Realimentação"},
    {"type": "OUT", "name": "Y(s) - Saída"},
    {"type": "DISPLAY", "name": "📊 Display"},
]


# ============================================================
# Helpers do canvas
# ============================================================
def new_id(prefix="n"):
    return f"{prefix}_{uuid4()}"

def mk_node(block_type, x, y, label, params=None, node_type="default"):
    return StreamlitFlowNode(
        id=new_id("n"),
        pos=(x, y),
        data={"label": label, "block_type": block_type, "params": params or {}},
        node_type=node_type,
        source_position="right",
        target_position="left",
    )

def get_nodes_edges(state):
    nodes = getattr(state, "nodes", None) or []
    edges = getattr(state, "edges", None) or []
    node_map = {n.id: n for n in nodes}
    edge_list = [(e.source, e.target) for e in edges]
    return node_map, edge_list


# ============================================================
# Grafo: achar caminho único (SISO)
# ============================================================
def find_unique_path(edges, start, end):
    g = defaultdict(list)
    for s, t in edges:
        g[s].append(t)

    q = deque([(start, [start])])
    found = []

    while q:
        u, path = q.popleft()
        if u == end:
            found.append(path)
            if len(found) > 1:
                return None
            continue

        for v in g[u]:
            if v in path:
                continue
            q.append((v, path + [v]))

    return found[0] if len(found) == 1 else None


# ============================================================
# Modelagem: bloco -> Função de Transferência
# ============================================================
def tf_from_params(num, den):
    def parse_poly(text):
        vals = []
        for x in str(text).split(","):
            x = x.strip()
            if x != "":
                vals.append(float(x))
        return vals

    n = parse_poly(num)
    d = parse_poly(den)
    if len(n) == 0 or len(d) == 0:
        raise ValueError("num/den inválidos.")
    return ctrl.tf(n, d)

def block_to_tf(node: StreamlitFlowNode):
    d = node.data or {}
    t = d.get("block_type", "")
    p = d.get("params", {}) or {}

    # blocos sem dinâmica
    if t in ("IN", "OUT", "SUM", "DISPLAY"):
        return ctrl.tf([1], [1])

    # P: ganho proporcional
    if t == "P":
        kp = float(p.get("kp", 1.0))
        return ctrl.tf([kp], [1])

    # PID: (Kd*s^2 + Kp*s + Ki)/s
    if t == "PID":
        kp = float(p.get("kp", 1.0))
        ki = float(p.get("ki", 0.0))
        kd = float(p.get("kd", 0.0))
        return ctrl.tf([kd, kp, ki], [1.0, 0.0])

    # Planta 1ª ordem: K/(tau*s + 1)
    if t == "PLANT_1":
        K = float(p.get("K", 1.0))
        tau = float(p.get("tau", 1.0))
        return ctrl.tf([K], [tau, 1.0])

    # Planta 2ª ordem: K*wn^2 / (s^2 + 2*zeta*wn*s + wn^2)
    if t == "PLANT_2":
        K = float(p.get("K", 1.0))
        wn = float(p.get("wn", 2.0))
        zeta = float(p.get("zeta", 0.7))
        return ctrl.tf([K * wn * wn], [1.0, 2.0 * zeta * wn, wn * wn])

    # Feedback genérico B(s) como TF configurável
    if t == "FB":
        num = p.get("num", "1")
        den = p.get("den", "1")
        return tf_from_params(num, den)

    # TF genérica (se você quiser adicionar depois)
    if t == "TF":
        num = p.get("num", "1")
        den = p.get("den", "1, 1")
        return tf_from_params(num, den)

    return ctrl.tf([1], [1])

def product_tf(path_ids, node_map, ignore=("IN", "OUT", "SUM", "DISPLAY")):
    G = ctrl.tf([1], [1])
    for nid in path_ids:
        bt = (node_map[nid].data or {}).get("block_type", "")
        if bt in ignore:
            continue
        G = ctrl.series(G, block_to_tf(node_map[nid]))
    return G


# ============================================================
# Análises: Bode, PZ, tempo
# ============================================================
def bode_plot(G, wmin=1e-2, wmax=1e2):
    w = np.logspace(np.log10(wmin), np.log10(wmax), 800)
    mag, phase, omega = ctrl.bode_plot(G, w, plot=False)
    mag = np.array(mag).squeeze()
    phase = np.array(phase).squeeze()
    omega = np.array(omega).squeeze()
    mag_db = 20 * np.log10(np.maximum(mag, 1e-300))

    f1 = plt.figure()
    plt.semilogx(omega, mag_db)
    plt.grid(True, which="both")
    plt.xlabel("ω (rad/s)")
    plt.ylabel("|G(jω)| (dB)")
    plt.title("Bode — Magnitude")

    f2 = plt.figure()
    plt.semilogx(omega, np.degrees(phase))
    plt.grid(True, which="both")
    plt.xlabel("ω (rad/s)")
    plt.ylabel("Fase (graus)")
    plt.title("Bode — Fase")
    return f1, f2

def pz_plot(G):
    poles = np.array(ctrl.poles(G), dtype=complex).flatten()
    zeros = np.array(ctrl.zeros(G), dtype=complex).flatten()

    fig = plt.figure()
    ax = plt.gca()
    if zeros.size:
        ax.scatter(np.real(zeros), np.imag(zeros), marker="o", label="Zeros")
    if poles.size:
        ax.scatter(np.real(poles), np.imag(poles), marker="x", label="Polos")
    ax.axhline(0, linewidth=1)
    ax.axvline(0, linewidth=1)
    ax.grid(True)
    ax.set_xlabel("Re")
    ax.set_ylabel("Im")
    ax.legend()
    ax.set_title("Plano-s (Polos e Zeros)")
    return fig, poles, zeros

def time_response(G, kind, tmax, amp, freq_hz, duty, custom_expr):
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
            raise ValueError("u(t) precisa ter o mesmo tamanho de t.")
        tout, y, _ = ctrl.forced_response(G, T=t, U=u)
    else:
        raise ValueError("Entrada inválida.")

    fig = plt.figure()
    plt.plot(tout, y, label="y(t)")
    plt.grid(True)
    plt.xlabel("t (s)")
    plt.ylabel("Saída")
    plt.title(f"Resposta no tempo — {kind}")
    if u is not None:
        plt.plot(tout, u, "--", label="u(t)")
        plt.legend()
    return fig


# ============================================================
# Estado inicial (template padrão)
# ============================================================
if "flow_state" not in st.session_state:
    nodes = []

    # Fixos (obrigatórios)
    n_in = StreamlitFlowNode(
        id="IN",
        pos=(60, 180),
        data={"label": "IN (R)", "block_type": "IN", "params": {}},
        node_type="input",
        source_position="right",
        target_position="left",
    )
    n_sum = StreamlitFlowNode(
        id="SUM",
        pos=(240, 180),
        data={"label": "SUM (−)", "block_type": "SUM", "params": {"sign": -1}},
        node_type="default",
        source_position="right",
        target_position="left",
    )
    n_out = StreamlitFlowNode(
        id="OUT",
        pos=(980, 180),
        data={"label": "OUT (Y)", "block_type": "OUT", "params": {}},
        node_type="output",
        source_position="right",
        target_position="left",
    )

    nodes.extend([n_in, n_sum, n_out])

    # Um bloco no direto + um no feedback pra começar
    plant = mk_node("PLANT_1", 560, 160, "Planta 1ª (K, tau)", {"K": 1.0, "tau": 1.0})
    fb = mk_node("FB", 560, 360, "Feedback B(s)", {"num": "1", "den": "1"})
    nodes.extend([plant, fb])

    edges = [
        StreamlitFlowEdge(id="e1", source="IN", target="SUM", animated=True),
        StreamlitFlowEdge(id="e2", source="SUM", target=plant.id, animated=True),
        StreamlitFlowEdge(id="e3", source=plant.id, target="OUT", animated=True),
        StreamlitFlowEdge(id="e4", source="OUT", target=fb.id, animated=True),
        StreamlitFlowEdge(id="e5", source=fb.id, target="SUM", animated=True),
    ]

    st.session_state.flow_state = make_flow_state(nodes, edges, key="flow")


# ============================================================
# Sidebar: adicionar blocos + editar parâmetros
# ============================================================
with st.sidebar:
    st.header("📦 Blocos (Adicionar)")

    c1, c2 = st.columns(2)
    with c1:
        if st.button("➕ P"):
            st.session_state.flow_state.nodes.append(
                mk_node("P", 420, 40, "P (Kp)", {"kp": 1.0})
            )
            st.rerun()
    with c2:
        if st.button("➕ PID"):
            st.session_state.flow_state.nodes.append(
                mk_node("PID", 560, 40, "PID (Kp,Ki,Kd)", {"kp": 1.0, "ki": 0.0, "kd": 0.0})
            )
            st.rerun()

    c3, c4 = st.columns(2)
    with c3:
        if st.button("➕ Planta 1ª"):
            st.session_state.flow_state.nodes.append(
                mk_node("PLANT_1", 420, 120, "Planta 1ª (K,tau)", {"K": 1.0, "tau": 1.0})
            )
            st.rerun()
    with c4:
        if st.button("➕ Planta 2ª"):
            st.session_state.flow_state.nodes.append(
                mk_node("PLANT_2", 560, 120, "Planta 2ª (K,wn,ζ)", {"K": 1.0, "wn": 2.0, "zeta": 0.7})
            )
            st.rerun()

    c5, c6 = st.columns(2)
    with c5:
        if st.button("➕ Feedback B(s)"):
            st.session_state.flow_state.nodes.append(
                mk_node("FB", 700, 120, "B(s) TF", {"num": "1", "den": "1"})
            )
            st.rerun()
    with c6:
        if st.button("↺ Reset template"):
            st.session_state.pop("flow_state", None)
            st.rerun()

    st.divider()
    st.header("✏️ Editar bloco (por ID)")
    st.caption("Abra o Debug no app e copie o ID do nó. IN/SUM/OUT são fixos.")

    edit_id = st.text_input("ID do bloco", value="SUM")
    node_map, _ = get_nodes_edges(st.session_state.flow_state)

    if edit_id and edit_id in node_map:
        n = node_map[edit_id]
        d = n.data or {}
        bt = d.get("block_type")
        params = d.get("params", {}) or {}
        st.write(f"Tipo: **{bt}**")

        if bt == "P":
            kp = st.number_input("Kp", value=float(params.get("kp", 1.0)))
            n.data["params"] = {"kp": float(kp)}
            n.data["label"] = f"P (Kp={kp:g})"
            if st.button("Salvar P"):
                st.rerun()

        elif bt == "PID":
            kp = st.number_input("Kp", value=float(params.get("kp", 1.0)))
            ki = st.number_input("Ki", value=float(params.get("ki", 0.0)))
            kd = st.number_input("Kd", value=float(params.get("kd", 0.0)))
            n.data["params"] = {"kp": float(kp), "ki": float(ki), "kd": float(kd)}
            n.data["label"] = f"PID (Kp={kp:g}, Ki={ki:g}, Kd={kd:g})"
            if st.button("Salvar PID"):
                st.rerun()

        elif bt == "PLANT_1":
            K = st.number_input("K", value=float(params.get("K", 1.0)))
            tau = st.number_input("tau", value=float(params.get("tau", 1.0)))
            n.data["params"] = {"K": float(K), "tau": float(tau)}
            n.data["label"] = f"Planta 1ª (K={K:g}, tau={tau:g})"
            if st.button("Salvar Planta 1ª"):
                st.rerun()

        elif bt == "PLANT_2":
            K = st.number_input("K", value=float(params.get("K", 1.0)))
            wn = st.number_input("wn", value=float(params.get("wn", 2.0)))
            zeta = st.number_input("zeta", value=float(params.get("zeta", 0.7)))
            n.data["params"] = {"K": float(K), "wn": float(wn), "zeta": float(zeta)}
            n.data["label"] = f"Planta 2ª (K={K:g}, wn={wn:g}, ζ={zeta:g})"
            if st.button("Salvar Planta 2ª"):
                st.rerun()

        elif bt == "FB":
            num = st.text_input("num (ex: 1, 2)", value=str(params.get("num", "1")))
            den = st.text_input("den (ex: 1, 2, 1)", value=str(params.get("den", "1")))
            n.data["params"] = {"num": num, "den": den}
            n.data["label"] = f"B(s) [{num}]/[{den}]"
            if st.button("Salvar Feedback"):
                st.rerun()

        elif bt == "SUM":
            opt = st.selectbox("Sinal do feedback", ["Negativo (-)", "Positivo (+)"], index=0)
            sgn = -1 if opt.startswith("Negativo") else +1
            n.data["params"] = {"sign": int(sgn)}
            n.data["label"] = f"SUM ({'−' if sgn == -1 else '+'})"
            if st.button("Salvar SUM"):
                st.rerun()

        else:
            st.info("Esse tipo não tem parâmetros editáveis aqui.")
    else:
        st.warning("ID não encontrado (veja no Debug).")

    st.divider()
    st.header("▶️ Simulação")

    entrada = st.selectbox("Entrada", ["Degrau", "Impulso", "Rampa", "Seno", "Pulso", "Custom"], index=0)
    tmax = st.number_input("Tempo máx (s)", min_value=0.1, value=10.0, step=0.5)
    amp = st.number_input("Amplitude", value=1.0, step=0.1)

    freq_hz = 1.0
    duty = 0.5
    custom_expr = "np.ones_like(t)"

    if entrada in ("Seno", "Pulso"):
        freq_hz = st.number_input("Frequência (Hz)", min_value=0.0001, value=1.0, step=0.1)
    if entrada == "Pulso":
        duty = st.slider("Duty", 0.05, 0.95, 0.5, 0.05)
    if entrada == "Custom":
        custom_expr = st.text_area("u(t) =", value="2*np.sin(2*np.pi*1*t)")

    st.divider()
    st.header("📉 Bode")
    wmin = st.number_input("ω min", min_value=1e-6, value=1e-2, format="%.6f")
    wmax = st.number_input("ω max", min_value=1e-6, value=1e2, format="%.6f")


# ============================================================
# Canvas + Resultados
# ============================================================
left, right = st.columns([1.25, 1])

with left:
    st.subheader("🧩 Canvas (arraste e conecte)")
    st.session_state.flow_state = run_flow("flow", st.session_state.flow_state)

    with st.expander("Debug (IDs e conexões)"):
        nm, el = get_nodes_edges(st.session_state.flow_state)
        st.json({
            "nodes": [{"id": n.id, "type": (n.data or {}).get("block_type"), "label": (n.data or {}).get("label")} for n in nm.values()],
            "edges": [{"source": s, "target": t} for (s, t) in el],
        })

with right:
    st.subheader("📌 Resultados (SISO)")

    node_map, edges = get_nodes_edges(st.session_state.flow_state)

    # Nós obrigatórios
    for must in ("IN", "SUM", "OUT"):
        if must not in node_map:
            st.error(f"Faltou o nó obrigatório: {must}. Use Reset template.")
            st.stop()

    # Caminho direto SUM -> OUT
    fwd_path = find_unique_path(edges, "SUM", "OUT")
    if fwd_path is None:
        st.error("Preciso de um caminho ÚNICO SUM → OUT (sem bifurcação/2 caminhos).")
        st.stop()

    # Feedback OUT -> SUM (opcional)
    fbk_path = find_unique_path(edges, "OUT", "SUM")
    has_feedback = fbk_path is not None and len(fbk_path) >= 2

    try:
        Gf = product_tf(fwd_path, node_map)
        H = product_tf(fbk_path, node_map) if has_feedback else ctrl.tf([1], [1])

        sign = int(((node_map["SUM"].data or {}).get("params", {}) or {}).get("sign", -1))

        # Malha aberta e fechada
        L = ctrl.series(Gf, H)
        T = ctrl.feedback(Gf, H, sign=sign) if has_feedback else Gf

        st.write("**Caminho direto (SUM → OUT):**")
        st.code(" -> ".join(fwd_path), language="text")

        st.write("**Caminho feedback (OUT → SUM):**")
        st.code(" -> ".join(fbk_path) if has_feedback else "Sem feedback", language="text")

        st.write("### G_forward(s)")
        st.code(str(Gf), language="text")

        st.write("### H_feedback(s)")
        st.code(str(H), language="text")

        st.write("### Malha aberta L(s)=G·H")
        st.code(str(L), language="text")

        st.write("### Malha fechada T(s)")
        st.code(str(T), language="text")

    except Exception as e:
        st.error(f"Erro montando o sistema: {e}")
        st.stop()

    st.divider()

    # Tempo
    try:
        fig_time = time_response(T, entrada, tmax, amp, freq_hz, duty, custom_expr)
        st.pyplot(fig_time, clear_figure=True)
    except Exception as e:
        st.error(f"Erro na resposta no tempo: {e}")

    st.divider()

    # Bode
    try:
        fig_mag, fig_phase = bode_plot(T, wmin=wmin, wmax=wmax)
        st.pyplot(fig_mag, clear_figure=True)
        st.pyplot(fig_phase, clear_figure=True)
    except Exception as e:
        st.error(f"Erro no Bode: {e}")

    st.divider()

    # Polos e zeros
    try:
        fig_pz, poles, zeros = pz_plot(T)
        st.pyplot(fig_pz, clear_figure=True)

        st.write("**Polos:**")
        st.code("\n".join([str(p) for p in poles]) if poles.size else "Nenhum", language="text")

        st.write("**Zeros:**")
        st.code("\n".join([str(z) for z in zeros]) if zeros.size else "Nenhum", language="text")
    except Exception as e:
        st.error(f"Erro em polos/zeros: {e}")
