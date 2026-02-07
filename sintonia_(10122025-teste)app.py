import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Rectangle
from scipy import signal
import plotly.graph_objects as go
import plotly.express as px
from scipy.integrate import odeint
import warnings
warnings.filterwarnings('ignore')

# ==================== CONFIGURAÇÃO PÁGINA ====================
st.set_page_config(
    page_title="Simulador XCOS - Controle Clássico",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main {
        padding: 2rem;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
    }
    h1 {
        color: #1f77b4;
        text-align: center;
        border-bottom: 3px solid #1f77b4;
        padding-bottom: 1rem;
    }
    h2 {
        color: #ff7f0e;
        margin-top: 1.5rem;
    }
    .bloco {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# ==================== FUNÇÕES DE TRANSFERÊNCIA ====================

class SistemaControle:
    """Classe para representar sistemas de controle"""
    
    def __init__(self, num, den, nome=""):
        self.num = np.array(num, dtype=float)
        self.den = np.array(den, dtype=float)
        self.nome = nome
        self.tf = signal.TransferFunction(self.num, self.den)
        self.polos = self.tf.poles
        self.zeros = self.tf.zeros
    
    def __mul__(self, outro):
        """Multiplicação de funções de transferência em série"""
        num = np.convolve(self.num, outro.num)
        den = np.convolve(self.den, outro.den)
        return SistemaControle(num, den, f"({self.nome})×({outro.nome})")
    
    def ganho_dc(self):
        """Ganho em DC (frequência 0)"""
        if self.den[0] == 0 or len(self.den) == 0:
            return 0
        return self.num[0] / self.den[0]
    
    def resposta_degrau(self, t):
        """Calcula resposta ao degrau unitário"""
        t_sim, y = signal.step(self.tf, T=t)
        return t_sim, y
    
    def resposta_rampa(self, t):
        """Calcula resposta à rampa unitária"""
        # Rampa é degrau dividido por s, então multiplica por 1/s
        num_rampa = np.convolve(self.num, [1])
        den_rampa = np.convolve(self.den, [1, 0])
        tf_rampa = signal.TransferFunction(num_rampa, den_rampa)
        t_sim, y = signal.step(tf_rampa, T=t)
        return t_sim, y
    
    def diagrama_bode(self, w=None):
        """Calcula diagrama de Bode"""
        if w is None:
            w = np.logspace(-2, 2, 1000)
        w_real, mag, fase = signal.bode(self.tf, w)
        return w_real, mag, fase
    
    def diagrama_nyquist(self, w=None):
        """Calcula diagrama de Nyquist"""
        if w is None:
            w = np.logspace(-2, 2, 1000)
        w_real, H = signal.freqs(self.num, self.den, w)
        return H.real, H.imag
    
    def diagrama_polos_zeros(self):
        """Retorna polos e zeros"""
        return self.polos, self.zeros

# ==================== FUNÇÕES DE DESENHO ====================

def desenhar_diagrama_blocos_malha_aberta(fig, ax):
    """Desenha diagrama de blocos em malha aberta"""
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 3)
    ax.axis('off')
    
    # Entradas/Saídas
    ax.arrow(0.2, 1.5, 0.6, 0, head_width=0.15, head_length=0.1, fc='black', ec='black')
    ax.text(0.1, 1.7, 'R(s)', fontsize=12, fontweight='bold', ha='right')
    
    # Bloco C(s)
    rect_c = FancyBboxPatch((1.2, 1.2), 1.2, 0.6, boxstyle="round,pad=0.05", 
                            edgecolor='blue', facecolor='lightblue', linewidth=2)
    ax.add_patch(rect_c)
    ax.text(1.8, 1.5, 'C(s)', fontsize=11, fontweight='bold', ha='center', va='center')
    ax.arrow(1.8, 1.05, 0, -0.2, head_width=0.1, head_length=0.08, fc='black', ec='black')
    ax.text(1.95, 0.8, 'E(s)', fontsize=10, ha='left')
    
    # Conexão C(s) -> G(s)
    ax.arrow(2.4, 1.5, 0.6, 0, head_width=0.15, head_length=0.1, fc='black', ec='black')
    ax.text(2.7, 1.7, 'U(s)', fontsize=10, ha='center')
    
    # Bloco G(s)
    rect_g = FancyBboxPatch((3.2, 1.2), 1.2, 0.6, boxstyle="round,pad=0.05", 
                            edgecolor='green', facecolor='lightgreen', linewidth=2)
    ax.add_patch(rect_g)
    ax.text(3.8, 1.5, 'G(s)', fontsize=11, fontweight='bold', ha='center', va='center')
    
    # Saída
    ax.arrow(4.4, 1.5, 0.8, 0, head_width=0.15, head_length=0.1, fc='black', ec='black')
    ax.text(5.5, 1.7, 'Y(s)', fontsize=12, fontweight='bold', ha='center')
    
    # Função de transferência total
    ax.text(3, 0.2, 'H(s) = C(s) × G(s)', fontsize=12, fontweight='bold', 
            ha='center', bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))

def desenhar_diagrama_blocos_malha_fechada(fig, ax, tipo_realimentacao="-"):
    """Desenha diagrama de blocos em malha fechada"""
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 4)
    ax.axis('off')
    
    # Entrada
    ax.arrow(0.2, 2.5, 0.4, 0, head_width=0.15, head_length=0.1, fc='black', ec='black')
    ax.text(0.1, 2.7, 'R(s)', fontsize=12, fontweight='bold', ha='right')
    
    # Somador
    circle_sum = Circle((0.8, 2.5), 0.2, color='yellow', ec='black', linewidth=2)
    ax.add_patch(circle_sum)
    ax.text(0.8, 2.5, '∑', fontsize=14, fontweight='bold', ha='center', va='center')
    sinal = '-' if tipo_realimentacao == "-" else '+'
    ax.text(0.75, 2.15, sinal, fontsize=12, fontweight='bold', ha='center')
    
    # Bloco C(s)
    rect_c = FancyBboxPatch((1.3, 2.25), 1.0, 0.5, boxstyle="round,pad=0.03", 
                            edgecolor='blue', facecolor='lightblue', linewidth=2)
    ax.add_patch(rect_c)
    ax.text(1.8, 2.5, 'C(s)', fontsize=11, fontweight='bold', ha='center', va='center')
    ax.arrow(1.0, 2.5, 0.25, 0, head_width=0.12, head_length=0.08, fc='black', ec='black')
    ax.text(1.15, 2.75, 'E(s)', fontsize=10, ha='center')
    
    # Conexão C(s) -> G(s)
    ax.arrow(2.3, 2.5, 0.5, 0, head_width=0.12, head_length=0.08, fc='black', ec='black')
    ax.text(2.55, 2.75, 'U(s)', fontsize=10, ha='center')
    
    # Bloco G(s)
    rect_g = FancyBboxPatch((2.95, 2.25), 1.0, 0.5, boxstyle="round,pad=0.03", 
                            edgecolor='green', facecolor='lightgreen', linewidth=2)
    ax.add_patch(rect_g)
    ax.text(3.45, 2.5, 'G(s)', fontsize=11, fontweight='bold', ha='center', va='center')
    
    # Saída Y(s)
    ax.arrow(3.95, 2.5, 0.8, 0, head_width=0.12, head_length=0.08, fc='black', ec='black')
    ax.text(5.0, 2.75, 'Y(s)', fontsize=12, fontweight='bold', ha='center')
    
    # Realimentação - Bloco B(s)
    ax.arrow(5.0, 2.5, 0, -0.8, head_width=0.1, head_length=0.08, fc='black', ec='black')
    rect_b = FancyBboxPatch((4.6, 0.5), 0.8, 0.5, boxstyle="round,pad=0.03", 
                            edgecolor='red', facecolor='lightcoral', linewidth=2)
    ax.add_patch(rect_b)
    ax.text(5.0, 0.75, 'B(s)', fontsize=11, fontweight='bold', ha='center', va='center')
    
    # Conexão para somador
    ax.arrow(5.0, 0.5, 0, -0.5, head_width=0.1, head_length=0.08, fc='black', ec='black')
    ax.plot([5.0, 0.6], [0.0, 0.0], 'k-', linewidth=1)
    ax.plot([0.6, 0.6], [0.0, 2.3], 'k-', linewidth=1)
    
    # Legenda de fórmula
    ax.text(3, 3.8, 'Malha Fechada: H(s) = C(s)×G(s) / [1 ± B(s)×C(s)×G(s)]', 
            fontsize=11, fontweight='bold', ha='center',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

# ==================== INTERFACE PRINCIPAL ====================

st.title("⚙️ Simulador XCOS - Sistemas de Controle Clássico")
st.markdown("Simulação de sistemas em **malha aberta** e **malha fechada** com análise detalhada")
st.markdown("---")

# Tabs principais
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Malha Aberta",
    "🔁 Malha Fechada",
    "📈 Análise de Frequência",
    "📋 Teórico"
])

# ==================== TAB 1: MALHA ABERTA ====================

with tab1:
    st.header("Simulação em Malha Aberta")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🎛️ Configurar Controlador C(s)")
        
        tipo_controlador = st.selectbox(
            "Tipo de Controlador:",
            ["Proporcional (P)", "Integral (I)", "Derivativo (D)", "PID", "Lag-Lead", "Customizado"],
            key="ma_controlador"
        )
        
        if tipo_controlador == "Proporcional (P)":
            Kp_ma = st.slider("Ganho Kp:", 0.1, 10.0, 1.0, 0.1)
            C_ma = SistemaControle([Kp_ma], [1], f"C(s) = {Kp_ma}")
        
        elif tipo_controlador == "Integral (I)":
            Ki_ma = st.slider("Ganho Ki:", 0.1, 10.0, 1.0, 0.1)
            C_ma = SistemaControle([Ki_ma], [1, 0], f"C(s) = {Ki_ma}/s")
        
        elif tipo_controlador == "Derivativo (D)":
            Kd_ma = st.slider("Ganho Kd:", 0.1, 10.0, 1.0, 0.1)
            C_ma = SistemaControle([Kd_ma, 0], [1], f"C(s) = {Kd_ma}s")
        
        elif tipo_controlador == "PID":
            Kp_pid = st.slider("Kp:", 0.1, 10.0, 1.0, 0.1, key="pid_kp_ma")
            Ki_pid = st.slider("Ki:", 0.1, 10.0, 0.5, 0.1, key="pid_ki_ma")
            Kd_pid = st.slider("Kd:", 0.1, 10.0, 0.5, 0.1, key="pid_kd_ma")
            # PID: C(s) = Kp + Ki/s + Kd*s
            num_pid = np.convolve([Kd_pid, Kp_pid, Ki_pid], [1])
            C_ma = SistemaControle(num_pid, [1, 0], f"C(s) = {Kd_pid}s² + {Kp_pid}s + {Ki_pid}) / s")
        
        elif tipo_controlador == "Lag-Lead":
            tau1_ll = st.slider("τ₁ (Lag):", 0.1, 10.0, 1.0, 0.1)
            tau2_ll = st.slider("τ₂ (Lead):", 0.1, 10.0, 0.5, 0.1)
            num_ll = np.convolve([tau1_ll, 1], [tau2_ll, 1])
            den_ll = np.convolve([tau1_ll, 1], [1])
            C_ma = SistemaControle(num_ll, den_ll, f"C(s) = ({tau1_ll}s+1)({tau2_ll}s+1) / ({tau1_ll}s+1)")
        
        else:  # Customizado
            num_str = st.text_input("Numerador (espaço separado):", "1")
            den_str = st.text_input("Denominador (espaço separado):", "1")
            try:
                C_num = [float(x) for x in num_str.split()]
                C_den = [float(x) for x in den_str.split()]
                C_ma = SistemaControle(C_num, C_den, f"C(s) = {num_str}/{den_str}")
            except:
                st.error("Formato inválido!")
                C_ma = SistemaControle([1], [1], "C(s) = 1")
    
    with col2:
        st.subheader("🔧 Configurar Processo G(s)")
        
        tipo_processo = st.selectbox(
            "Tipo de Processo:",
            ["Primeira Ordem", "Segunda Ordem", "Integradora", "Customizado"],
            key="ma_processo"
        )
        
        if tipo_processo == "Primeira Ordem":
            tau_1o = st.slider("Constante de Tempo τ:", 0.1, 5.0, 1.0, 0.1)
            K_1o = st.slider("Ganho Estático K:", 0.1, 10.0, 1.0, 0.1)
            G_ma = SistemaControle([K_1o], [tau_1o, 1], f"G(s) = {K_1o}/({tau_1o}s+1)")
        
        elif tipo_processo == "Segunda Ordem":
            wn_2o = st.slider("Frequência Natural ωn:", 0.1, 10.0, 2.0, 0.1)
            zeta_2o = st.slider("Amortecimento ζ:", 0.1, 2.0, 0.7, 0.1)
            K_2o = st.slider("Ganho Estático K:", 0.1, 10.0, 1.0, 0.1)
            G_ma = SistemaControle([K_2o * wn_2o**2], [1, 2*zeta_2o*wn_2o, wn_2o**2], 
                                  f"G(s) = {K_2o*wn_2o**2} / (s² + {2*zeta_2o*wn_2o:.2f}s + {wn_2o**2:.2f})")
        
        elif tipo_processo == "Integradora":
            K_int = st.slider("Ganho K:", 0.1, 10.0, 1.0, 0.1)
            G_ma = SistemaControle([K_int], [1, 0], f"G(s) = {K_int}/s")
        
        else:  # Customizado
            num_str_g = st.text_input("Numerador G(s):", "1")
            den_str_g = st.text_input("Denominador G(s):", "1 1")
            try:
                G_num = [float(x) for x in num_str_g.split()]
                G_den = [float(x) for x in den_str_g.split()]
                G_ma = SistemaControle(G_num, G_den, f"G(s) = {num_str_g}/{den_str_g}")
            except:
                st.error("Formato inválido!")
                G_ma = SistemaControle([1], [1, 1], "G(s) = 1/(s+1)")
    
    # Cálculos malha aberta
    H_ma = C_ma * G_ma
    
    st.markdown("---")
    
    # Desenhar diagrama
    st.subheader("📐 Diagrama de Blocos")
    fig_diagrama_ma, ax_diagrama_ma = plt.subplots(figsize=(12, 3))
    desenhar_diagrama_blocos_malha_aberta(fig_diagrama_ma, ax_diagrama_ma)
    st.pyplot(fig_diagrama_ma, use_container_width=True)
    
    # Função de transferência
    col_info1, col_info2 = st.columns(2)
    with col_info1:
        st.markdown("### 📝 Função de Transferência")
        st.latex(f"H(s) = C(s) \\times G(s)")
        st.info(f"**Polos:** {np.round(H_ma.polos, 4)}")
        st.info(f"**Zeros:** {np.round(H_ma.zeros, 4)}")
        st.info(f"**Ganho DC:** {np.round(H_ma.ganho_dc(), 4)}")
    
    with col_info2:
        # Análise de estabilidade
        st.markdown("### 🔍 Análise de Estabilidade")
        polos_reais = np.real(H_ma.polos)
        if np.all(polos_reais < 0):
            st.success("✅ Sistema ESTÁVEL (todos os polos no semiplano esquerdo)")
        else:
            st.error("❌ Sistema INSTÁVEL (polos no semiplano direito)")
    
    # Respostas temporais
    st.markdown("---")
    st.subheader("⏱️ Respostas Temporais")
    
    t_max = st.slider("Tempo máximo de simulação (s):", 1, 50, 10, key="ma_t_max")
    t_ma = np.linspace(0, t_max, 1000)
    
    col_resp1, col_resp2 = st.columns(2)
    
    with col_resp1:
        st.markdown("#### Resposta ao Degrau Unitário")
        t_step, y_step = H_ma.resposta_degrau(t_ma)
        
        fig_step, ax_step = plt.subplots(figsize=(10, 5))
        ax_step.plot(t_step, y_step, 'b-', linewidth=2, label='y(t)')
        ax_step.axhline(y=H_ma.ganho_dc(), color='r', linestyle='--', label='Ganho DC')
        ax_step.set_xlabel('Tempo (s)', fontsize=11)
        ax_step.set_ylabel('Amplitude', fontsize=11)
        ax_step.set_title('Resposta ao Degrau Unitário - Malha Aberta', fontsize=12, fontweight='bold')
        ax_step.grid(True, alpha=0.3)
        ax_step.legend()
        st.pyplot(fig_step, use_container_width=True)
    
    with col_resp2:
        st.markdown("#### Resposta à Rampa Unitária")
        try:
            t_ramp, y_ramp = H_ma.resposta_rampa(t_ma)
            
            fig_ramp, ax_ramp = plt.subplots(figsize=(10, 5))
            ax_ramp.plot(t_ramp, y_ramp, 'g-', linewidth=2, label='y(t)')
            ax_ramp.plot(t_ramp, t_ramp, 'r--', label='Entrada (rampa)', alpha=0.7)
            ax_ramp.set_xlabel('Tempo (s)', fontsize=11)
            ax_ramp.set_ylabel('Amplitude', fontsize=11)
            ax_ramp.set_title('Resposta à Rampa Unitária - Malha Aberta', fontsize=12, fontweight='bold')
            ax_ramp.grid(True, alpha=0.3)
            ax_ramp.legend()
            ax_ramp.set_ylim([0, t_max])
            st.pyplot(fig_ramp, use_container_width=True)
        except:
            st.warning("⚠️ Impossível calcular resposta à rampa para este sistema")
    
    # Diagrama de polos e zeros
    st.markdown("---")
    st.subheader("🎯 Diagrama de Polos e Zeros")
    
    fig_pz, ax_pz = plt.subplots(figsize=(8, 8))
    
    polos, zeros = H_ma.diagrama_polos_zeros()
    
    # Plot dos polos
    ax_pz.scatter(np.real(polos), np.imag(polos), marker='x', s=200, c='red', linewidth=3, label='Polos')
    # Plot dos zeros
    ax_pz.scatter(np.real(zeros), np.imag(zeros), marker='o', s=200, c='blue', facecolors='none', 
                 linewidth=2, label='Zeros')
    
    # Eixo imaginário
    ax_pz.axhline(y=0, color='k', linewidth=0.5)
    ax_pz.axvline(x=0, color='k', linewidth=1.5)
    
    # Região de estabilidade
    ax_pz.fill_between([-10, 0], -10, 10, alpha=0.1, color='green', label='Região Estável')
    
    ax_pz.set_xlabel('Eixo Real', fontsize=11)
    ax_pz.set_ylabel('Eixo Imaginário', fontsize=11)
    ax_pz.set_title('Diagrama de Polos e Zeros - Malha Aberta', fontsize=12, fontweight='bold')
    ax_pz.grid(True, alpha=0.3)
    ax_pz.legend(loc='upper right')
    ax_pz.set_xlim([-5, 1])
    ax_pz.set_ylim([-5, 5])
    
    st.pyplot(fig_pz, use_container_width=True)

# ==================== TAB 2: MALHA FECHADA ====================

with tab2:
    st.header("Simulação em Malha Fechada")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("🎛️ Controlador C(s)")
        
        tipo_controlador_mf = st.selectbox(
            "Tipo:",
            ["Proporcional (P)", "Integral (I)", "Derivativo (D)", "PID", "Customizado"],
            key="mf_controlador"
        )
        
        if tipo_controlador_mf == "Proporcional (P)":
            Kp_mf = st.slider("Ganho Kp:", 0.1, 20.0, 2.0, 0.1, key="kp_mf")
            C_mf = SistemaControle([Kp_mf], [1], f"C(s) = {Kp_mf}")
        
        elif tipo_controlador_mf == "Integral (I)":
            Ki_mf = st.slider("Ganho Ki:", 0.1, 10.0, 1.0, 0.1, key="ki_mf")
            C_mf = SistemaControle([Ki_mf], [1, 0], f"C(s) = {Ki_mf}/s")
        
        elif tipo_controlador_mf == "Derivativo (D)":
            Kd_mf = st.slider("Ganho Kd:", 0.1, 10.0, 1.0, 0.1, key="kd_mf")
            C_mf = SistemaControle([Kd_mf, 0], [1], f"C(s) = {Kd_mf}s")
        
        elif tipo_controlador_mf == "PID":
            Kp_pid_mf = st.slider("Kp:", 0.1, 10.0, 1.0, 0.1, key="kp_pid_mf")
            Ki_pid_mf = st.slider("Ki:", 0.1, 10.0, 0.5, 0.1, key="ki_pid_mf")
            Kd_pid_mf = st.slider("Kd:", 0.1, 10.0, 0.5, 0.1, key="kd_pid_mf")
            num_pid_mf = np.convolve([Kd_pid_mf, Kp_pid_mf, Ki_pid_mf], [1])
            C_mf = SistemaControle(num_pid_mf, [1, 0], f"PID")
        
        else:
            num_str_c = st.text_input("Numerador:", "1", key="num_c_mf")
            den_str_c = st.text_input("Denominador:", "1", key="den_c_mf")
            try:
                C_num = [float(x) for x in num_str_c.split()]
                C_den = [float(x) for x in den_str_c.split()]
                C_mf = SistemaControle(C_num, C_den, "C(s)")
            except:
                C_mf = SistemaControle([1], [1], "C(s) = 1")
    
    with col2:
        st.subheader("🔧 Processo G(s)")
        
        tipo_processo_mf = st.selectbox(
            "Tipo:",
            ["Primeira Ordem", "Segunda Ordem", "Integradora"],
            key="mf_processo"
        )
        
        if tipo_processo_mf == "Primeira Ordem":
            tau_mf = st.slider("τ:", 0.1, 5.0, 1.0, 0.1, key="tau_mf")
            K_mf = st.slider("K:", 0.1, 10.0, 1.0, 0.1, key="k_mf")
            G_mf = SistemaControle([K_mf], [tau_mf, 1], "G(s)")
        
        elif tipo_processo_mf == "Segunda Ordem":
            wn_mf = st.slider("ωn:", 0.5, 10.0, 2.0, 0.1, key="wn_mf")
            zeta_mf = st.slider("ζ:", 0.1, 2.0, 0.7, 0.1, key="zeta_mf")
            K_mf = st.slider("K:", 0.1, 10.0, 1.0, 0.1, key="k_mf_2o")
            G_mf = SistemaControle([K_mf * wn_mf**2], [1, 2*zeta_mf*wn_mf, wn_mf**2], "G(s)")
        
        else:
            K_int_mf = st.slider("K:", 0.1, 10.0, 1.0, 0.1, key="k_int_mf")
            G_mf = SistemaControle([K_int_mf], [1, 0], "G(s)")
    
    with col3:
        st.subheader("🔄 Realimentação B(s)")
        
        tipo_realimentacao = st.radio(
            "Tipo de Realimentação:",
            ["-", "+"],
            key="tipo_realim"
        )
        
        B_mf_type = st.selectbox(
            "B(s):",
            ["Unitária (B=1)", "Proporcional", "Customizado"],
            key="b_mf"
        )
        
        if B_mf_type == "Unitária (B=1)":
            B_mf = SistemaControle([1], [1], "B(s) = 1")
        elif B_mf_type == "Proporcional":
            K_b = st.slider("Ganho:", 0.1, 5.0, 1.0, 0.1, key="kb_mf")
            B_mf = SistemaControle([K_b], [1], f"B(s) = {K_b}")
        else:
            num_str_b = st.text_input("Numerador B:", "1", key="num_b_mf")
            den_str_b = st.text_input("Denominador B:", "1", key="den_b_mf")
            try:
                B_num = [float(x) for x in num_str_b.split()]
                B_den = [float(x) for x in den_str_b.split()]
                B_mf = SistemaControle(B_num, B_den, "B(s)")
            except:
                B_mf = SistemaControle([1], [1], "B(s) = 1")
    
    # Cálculo malha fechada
    st.markdown("---")
    
    # H(s) = C(s)*G(s) / [1 ± B(s)*C(s)*G(s)]
    CG = C_mf * G_mf
    BCG = B_mf * CG
    
    if tipo_realimentacao == "-":
        # H(s) = C(s)*G(s) / [1 + B(s)*C(s)*G(s)]
        num_mf = CG.num
        den_mf = np.polyadd(np.array([1]), BCG.num)
    else:
        # H(s) = C(s)*G(s) / [1 - B(s)*C(s)*G(s)]
        num_mf = CG.num
        den_mf = np.polyadd(np.array([1]), -BCG.num)
    
    H_mf = SistemaControle(num_mf, den_mf, "H(s)")
    
    # Desenhar diagrama
    st.subheader("📐 Diagrama de Blocos")
    fig_diagrama_mf, ax_diagrama_mf = plt.subplots(figsize=(12, 4))
    desenhar_diagrama_blocos_malha_fechada(fig_diagrama_mf, ax_diagrama_mf, tipo_realimentacao)
    st.pyplot(fig_diagrama_mf, use_container_width=True)
    
    # Análise
    col_anl1, col_anl2 = st.columns(2)
    
    with col_anl1:
        st.markdown("### 📊 Função de Transferência")
        st.latex(f"H(s) = \\frac{{C(s)G(s)}}{{1 {tipo_realimentacao} B(s)C(s)G(s)}}")
        st.info(f"**Polos:** {np.round(H_mf.polos, 4)}")
        st.info(f"**Zeros:** {np.round(H_mf.zeros, 4)}")
        st.info(f"**Ganho DC:** {np.round(H_mf.ganho_dc(), 4)}")
    
    with col_anl2:
        st.markdown("### 🔍 Estabilidade")
        polos_reais_mf = np.real(H_mf.polos)
        if np.all(polos_reais_mf < 0):
            st.success("✅ Sistema ESTÁVEL (todos os polos no semiplano esquerdo)")
        else:
            st.error("❌ Sistema INSTÁVEL")
    
    # Respostas
    st.markdown("---")
    st.subheader("⏱️ Respostas Temporais")
    
    t_max_mf = st.slider("Tempo máximo (s):", 1, 50, 10, key="mf_t_max")
    t_mf = np.linspace(0, t_max_mf, 1000)
    
    col_r1, col_r2 = st.columns(2)
    
    with col_r1:
        st.markdown("#### Resposta ao Degrau")
        try:
            t_step_mf, y_step_mf = H_mf.resposta_degrau(t_mf)
            
            fig_step_mf, ax_step_mf = plt.subplots(figsize=(10, 5))
            ax_step_mf.plot(t_step_mf, y_step_mf, 'b-', linewidth=2, label='y(t)')
            ax_step_mf.axhline(y=1, color='r', linestyle='--', label='Referência')
            ax_step_mf.set_xlabel('Tempo (s)', fontsize=11)
            ax_step_mf.set_ylabel('Amplitude', fontsize=11)
            ax_step_mf.set_title('Resposta ao Degrau - Malha Fechada', fontsize=12, fontweight='bold')
            ax_step_mf.grid(True, alpha=0.3)
            ax_step_mf.legend()
            ax_step_mf.set_ylim([min(y_step_mf)-0.2, max(y_step_mf)+0.2])
            st.pyplot(fig_step_mf, use_container_width=True)
            
            # Cálcular características
            try:
                ts = t_step_mf[np.where(np.abs(y_step_mf - y_step_mf[-1]) > 0.02 * np.abs(y_step_mf[-1]))[0][-1]]
                st.info(f"⏱️ Tempo de Acomodação (2%): {ts:.3f} s")
            except:
                pass
        except:
            st.warning("⚠️ Impossível calcular resposta")
    
    with col_r2:
        st.markdown("#### Erro em Regime Permanente")
        try:
            t_regime = np.linspace(0, t_max_mf, 5000)
            t_step_reg, y_step_reg = H_mf.resposta_degrau(t_regime)
            erro_regime = 1 - y_step_reg[-1]
            
            fig_erro, ax_erro = plt.subplots(figsize=(10, 5))
            ax_erro.plot(t_step_reg[-500:], y_step_reg[-500:], 'g-', linewidth=2, label='y(t)')
            ax_erro.axhline(y=1, color='r', linestyle='--', linewidth=2, label='Referência')
            ax_erro.fill_between(t_step_reg[-500:], y_step_reg[-500:], 1, alpha=0.3, color='red')
            ax_erro.set_xlabel('Tempo (s)', fontsize=11)
            ax_erro.set_ylabel('Amplitude', fontsize=11)
            ax_erro.set_title('Erro em Regime Permanente', fontsize=12, fontweight='bold')
            ax_erro.grid(True, alpha=0.3)
            ax_erro.legend()
            st.pyplot(fig_erro, use_container_width=True)
            
            st.info(f"❌ Erro em regime permanente: {np.abs(erro_regime)*100:.2f}%")
        except:
            st.warning("⚠️ Impossível calcular erro")
    
    # Polos e zeros
    st.markdown("---")
    st.subheader("🎯 Diagrama de Polos e Zeros - Malha Fechada")
    
    fig_pz_mf, ax_pz_mf = plt.subplots(figsize=(8, 8))
    
    polos_mf, zeros_mf = H_mf.diagrama_polos_zeros()
    
    ax_pz_mf.scatter(np.real(polos_mf), np.imag(polos_mf), marker='x', s=200, c='red', linewidth=3, label='Polos')
    ax_pz_mf.scatter(np.real(zeros_mf), np.imag(zeros_mf), marker='o', s=200, c='blue', facecolors='none', 
                    linewidth=2, label='Zeros')
    
    ax_pz_mf.axhline(y=0, color='k', linewidth=0.5)
    ax_pz_mf.axvline(x=0, color='k', linewidth=1.5)
    ax_pz_mf.fill_between([-10, 0], -10, 10, alpha=0.1, color='green', label='Região Estável')
    
    ax_pz_mf.set_xlabel('Eixo Real', fontsize=11)
    ax_pz_mf.set_ylabel('Eixo Imaginário', fontsize=11)
    ax_pz_mf.set_title('Diagrama de Polos e Zeros - Malha Fechada', fontsize=12, fontweight='bold')
    ax_pz_mf.grid(True, alpha=0.3)
    ax_pz_mf.legend()
    ax_pz_mf.set_xlim([-10, 2])
    ax_pz_mf.set_ylim([-6, 6])
    
    st.pyplot(fig_pz_mf, use_container_width=True)

# ==================== TAB 3: ANÁLISE DE FREQUÊNCIA ====================

with tab3:
    st.header("Análise de Frequência")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Configuração")
        config_analise = st.radio(
            "Analisar:",
            ["Malha Aberta", "Malha Fechada"],
            key="analise_freq"
        )
        
        freq_min = st.slider("Frequência Mínima (rad/s):", -2, 2, -2)
        freq_max = st.slider("Frequência Máxima (rad/s):", 0, 4, 2)
    
    with col2:
        st.subheader("Tipo de Análise")
        tipo_diagrama = st.radio(
            "Diagrama:",
            ["Bode", "Nyquist", "Nichols"],
            key="tipo_diag"
        )
    
    # Usar sistema apropriado
    if config_analise == "Malha Aberta":
        if 'H_ma' not in st.session_state:
            st.warning("⚠️ Configure primeiro a malha aberta")
            H_analise = SistemaControle([1], [1, 1], "Exemplo")
        else:
            H_analise = H_ma
        titulo = "Malha Aberta"
    else:
        if 'H_mf' not in st.session_state:
            st.warning("⚠️ Configure primeiro a malha fechada")
            H_analise = SistemaControle([1], [1, 1], "Exemplo")
        else:
            H_analise = H_mf
        titulo = "Malha Fechada"
    
    w = np.logspace(freq_min, freq_max, 1000)
    
    if tipo_diagrama == "Bode":
        st.subheader(f"📈 Diagrama de Bode - {titulo}")
        
        w_bode, mag_bode, fase_bode = H_analise.diagrama_bode(w)
        
        fig_bode, (ax_mag, ax_fase) = plt.subplots(2, 1, figsize=(12, 8))
        
        # Magnitude
        ax_mag.semilogx(w_bode, mag_bode, 'b-', linewidth=2)
        ax_mag.set_ylabel('Magnitude (dB)', fontsize=11)
        ax_mag.set_title('Diagrama de Bode - Magnitude', fontsize=12, fontweight='bold')
        ax_mag.grid(True, which='both', alpha=0.3)
        
        # Fase
        ax_fase.semilogx(w_bode, fase_bode, 'r-', linewidth=2)
        ax_fase.set_xlabel('Frequência (rad/s)', fontsize=11)
        ax_fase.set_ylabel('Fase (graus)', fontsize=11)
        ax_fase.set_title('Diagrama de Bode - Fase', fontsize=12, fontweight='bold')
        ax_fase.grid(True, which='both', alpha=0.3)
        
        st.pyplot(fig_bode, use_container_width=True)
    
    elif tipo_diagrama == "Nyquist":
        st.subheader(f"🌀 Diagrama de Nyquist - {titulo}")
        
        real, imag = H_analise.diagrama_nyquist(w)
        
        fig_nyq, ax_nyq = plt.subplots(figsize=(10, 8))
        
        ax_nyq.plot(real, imag, 'b-', linewidth=2, label='Nyquist')
        ax_nyq.scatter([real[0]], [imag[0]], color='green', s=100, marker='o', label='ω=0', zorder=5)
        ax_nyq.scatter([real[-1]], [imag[-1]], color='red', s=100, marker='s', label='ω=∞', zorder=5)
        
        # Ponto crítico
        ax_nyq.scatter([-1], [0], color='black', s=200, marker='x', linewidth=3, label='Ponto Crítico (-1,0)')
        
        ax_nyq.axhline(y=0, color='k', linewidth=0.5)
        ax_nyq.axvline(x=0, color='k', linewidth=0.5)
        ax_nyq.grid(True, alpha=0.3)
        
        ax_nyq.set_xlabel('Parte Real', fontsize=11)
        ax_nyq.set_ylabel('Parte Imaginária', fontsize=11)
        ax_nyq.set_title('Diagrama de Nyquist', fontsize=12, fontweight='bold')
        ax_nyq.legend()
        ax_nyq.axis('equal')
        ax_nyq.set_xlim([-2, 2])
        ax_nyq.set_ylim([-2, 2])
        
        st.pyplot(fig_nyq, use_container_width=True)
    
    else:  # Nichols
        st.subheader(f"📊 Gráfico de Nichols - {titulo}")
        
        w_nich, mag_nich, fase_nich = H_analise.diagrama_bode(w)
        
        fig_nich, ax_nich = plt.subplots(figsize=(12, 8))
        
        scatter = ax_nich.scatter(fase_nich, mag_nich, c=np.log10(w_nich), cmap='viridis', s=50)
        
        ax_nich.set_xlabel('Fase (graus)', fontsize=11)
        ax_nich.set_ylabel('Magnitude (dB)', fontsize=11)
        ax_nich.set_title('Diagrama de Nichols', fontsize=12, fontweight='bold')
        ax_nich.grid(True, alpha=0.3)
        
        cbar = plt.colorbar(scatter, ax=ax_nich)
        cbar.set_label('log₁₀(ω)', fontsize=11)
        
        st.pyplot(fig_nich, use_container_width=True)

# ==================== TAB 4: TEORIA ====================

with tab4:
    st.header("📚 Fundamentos de Sistemas de Controle Clássico")
    
    tab4_1, tab4_2, tab4_3 = st.tabs(["Malha Aberta", "Malha Fechada", "Controladores"])
    
    with tab4_1:
        st.markdown("""
        ## 🔓 Sistemas em Malha Aberta
        
        ### Definição
        Um sistema em malha aberta é aquele em que a saída **não influencia** a entrada de controle.
        A saída é controlada apenas pela entrada de referência.
        
        ### Diagrama
        ```
        R(s) → [C(s)] → [G(s)] → Y(s)
        ```
        
        ### Função de Transferência
        $$H(s) = C(s) \\times G(s)$$
        
        Onde:
        - **R(s)**: Entrada de referência (sinal desejado)
        - **C(s)**: Controlador
        - **G(s)**: Processo/Planta
        - **Y(s)**: Saída
        - **H(s)**: Função de transferência total em malha aberta
        
        ### Características
        
        ✅ **Vantagens:**
        - Simples de projetar e implementar
        - Baixo custo
        - Estável para sistemas estáveis
        
        ❌ **Desvantagens:**
        - Sensível a perturbações
        - Não compensa erros
        - Não pode corrigir desvios
        - Desempenho depende da precisão do modelo
        """)
        
        col_exp1, col_exp2 = st.columns(2)
        
        with col_exp1:
            st.markdown("### Exemplo Prático: Sistema Térmico")
            st.image("https://via.placeholder.com/300x200?text=Sistema+Térmico", 
                    caption="Forno sem controle de realimentação")
            st.write("""
            Um forno aquecido por um resistor elétrico sem sensor de temperatura.
            A potência fornecida é constante, independente da temperatura real.
            """)
        
        with col_exp2:
            st.markdown("### Resposta Temporal Típica")
            st.latex(r"y(t) = K(1 - e^{-t/\tau})")
            st.write("""
            - **K**: Ganho estático (ganho em DC)
            - **τ**: Constante de tempo
            - Resposta exponencial monotônica
            - Sem oscilações
            """)
    
    with tab4_2:
        st.markdown("""
        ## 🔄 Sistemas em Malha Fechada
        
        ### Definição
        Um sistema em malha fechada utiliza **realimentação** para comparar a saída real
        com a saída desejada, gerando um erro que é utilizado para ajustar a entrada.
        
        ### Diagrama
        ```
                      ∑ → [C(s)] → [G(s)] → Y(s)
                      ↑                        ↓
                      └─── [B(s)] ←───────────┘
        ```
        
        ### Função de Transferência
        $$H(s) = \\frac{C(s)G(s)}{1 \\pm B(s)C(s)G(s)}$$
        
        Onde:
        - **∑**: Somador (comparador de erro)
        - **B(s)**: Sensor de realimentação
        
        ### Características
        
        ✅ **Vantagens:**
        - Reduz efeito de perturbações
        - Compensa erros
        - Melhora precisão
        - Aumenta rapidez de resposta
        
        ❌ **Desvantagens:**
        - Mais complexa
        - Custo maior
        - Risco de instabilidade
        - Sensível a atrasos de tempo
        """)
        
        col_exp3, col_exp4 = st.columns(2)
        
        with col_exp3:
            st.markdown("### Exemplo Prático: Controle de Temperatura")
            st.image("https://via.placeholder.com/300x200?text=Controle+PID", 
                    caption="Sistema com sensor e controlador")
            st.write("""
            Um forno com sensor de temperatura que ajusta
            a potência para manter a temperatura desejada.
            """)
        
        with col_exp4:
            st.markdown("### Erro em Regime Permanente")
            st.latex(r"e_{ss} = \\lim_{t\\to\\infty} e(t) = R(\\infty) - Y(\\infty)")
            st.write("""
            A realimentação reduz o erro a zero para referências constantes
            em sistemas com integrador no controlador (tipo I ou superior).
            """)
    
    with tab4_3:
        st.markdown("""
        ## 🎛️ Tipos de Controladores
        
        ### 1. Controlador Proporcional (P)
        $$C(s) = K_p$$
        
        **Características:**
        - Ação: Proporcional ao erro
        - Efeito: Reduz erro, aumenta rapidez
        - Problema: Erro em regime permanente não zero
        - Aplicação: Sistemas simples
        
        ### 2. Controlador Integral (I)
        $$C(s) = \\frac{K_i}{s}$$
        
        **Características:**
        - Ação: Integral do erro
        - Efeito: Elimina erro em regime permanente
        - Problema: Lentifica resposta
        - Aplicação: Rejeição de perturbações
        
        ### 3. Controlador Derivativo (D)
        $$C(s) = K_d s$$
        
        **Características:**
        - Ação: Proporcional à taxa de variação do erro
        - Efeito: Antecipa mudanças, melhora estabilidade
        - Problema: Amplifica ruído
        - Aplicação: Melhorar amortecimento
        
        ### 4. Controlador PID
        $$C(s) = K_p + \\frac{K_i}{s} + K_d s$$
        
        **Características:**
        - Combina ações P, I e D
        - Proporcional: Rapidez
        - Integral: Erro zero
        - Derivativo: Estabilidade
        - Mais usado na indústria
        
        ### 5. Controlador Lag-Lead
        $$C(s) = \\frac{\\tau_1 s + 1}{\\tau_2 s + 1}$$
        
        **Características:**
        - Lag: Melhora ganho DC
        - Lead: Melhora margem de fase
        - Aplicação: Sintonia fina de resposta
        """)
        
        st.markdown("---")
        
        col_crit1, col_crit2 = st.columns(2)
        
        with col_crit1:
            st.markdown("### 📊 Critério de Estabilidade (Routh-Hurwitz)")
            st.write("""
            Um sistema é **estável** se todos os polos estão
            no **semiplano esquerdo** do plano complexo (parte real < 0).
            """)
        
        with col_crit2:
            st.markdown("### 🎯 Especificações de Desempenho")
            st.write("""
            - **Tempo de acomodação (ts)**: Tempo para entrar na banda ±2%
            - **Sobressinal (Mp)**: Máximo overshoot permitido
            - **Erro em regime (ess)**: Erro final aceitável
            - **Velocidade de resposta**: Taxa de subida
            """)
