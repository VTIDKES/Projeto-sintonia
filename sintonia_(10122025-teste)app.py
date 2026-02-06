import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyBboxPatch, FancyArrowPatch
import control as ct
from scipy import signal
import json

# Configuração da página
st.set_page_config(page_title="Editor de Blocos - Sistemas de Controle", layout="wide")

# Título
st.title("🔧 Editor de Blocos - Análise de Sistemas de Controle")
st.markdown("*Similar ao Xcos do Scilab*")

# Inicialização do estado da sessão
if 'blocks' not in st.session_state:
    st.session_state.blocks = []
if 'connections' not in st.session_state:
    st.session_state.connections = []
if 'system_type' not in st.session_state:
    st.session_state.system_type = "Malha Aberta"

# Sidebar para construção do sistema
st.sidebar.header("📐 Construtor de Sistema")

# Seleção do tipo de sistema
system_type = st.sidebar.radio(
    "Tipo de Sistema:",
    ["Malha Aberta", "Malha Fechada"]
)
st.session_state.system_type = system_type

st.sidebar.subheader("Adicionar Bloco")

# Tipos de blocos disponíveis
block_type = st.sidebar.selectbox(
    "Tipo de Bloco:",
    ["Função de Transferência", "Ganho", "Integrador", "Derivador", "Atraso"]
)

# Parâmetros do bloco
if block_type == "Função de Transferência":
    st.sidebar.markdown("**Numerador (s)**")
    num_str = st.sidebar.text_input("Coeficientes (separados por vírgula):", "1", key="num")
    st.sidebar.markdown("**Denominador (s)**")
    den_str = st.sidebar.text_input("Coeficientes (separados por vírgula):", "1,1", key="den")
    
elif block_type == "Ganho":
    gain = st.sidebar.number_input("Valor do Ganho (K):", value=1.0, step=0.1)
    
elif block_type == "Integrador":
    st.sidebar.info("G(s) = 1/s")
    
elif block_type == "Derivador":
    st.sidebar.info("G(s) = s")
    
elif block_type == "Atraso":
    tau = st.sidebar.number_input("Constante de Tempo (τ):", value=1.0, step=0.1)

# Botão para adicionar bloco
if st.sidebar.button("➕ Adicionar Bloco"):
    block_id = len(st.session_state.blocks)
    
    if block_type == "Função de Transferência":
        try:
            num = [float(x.strip()) for x in num_str.split(',')]
            den = [float(x.strip()) for x in den_str.split(',')]
            tf = ct.TransferFunction(num, den)
            st.session_state.blocks.append({
                'id': block_id,
                'type': block_type,
                'tf': tf,
                'num': num,
                'den': den,
                'label': f"TF_{block_id}"
            })
            st.sidebar.success(f"Bloco {block_id} adicionado!")
        except:
            st.sidebar.error("Erro ao criar função de transferência")
            
    elif block_type == "Ganho":
        tf = ct.TransferFunction([gain], [1])
        st.session_state.blocks.append({
            'id': block_id,
            'type': block_type,
            'tf': tf,
            'gain': gain,
            'label': f"K_{block_id}"
        })
        st.sidebar.success(f"Bloco {block_id} adicionado!")
        
    elif block_type == "Integrador":
        tf = ct.TransferFunction([1], [1, 0])
        st.session_state.blocks.append({
            'id': block_id,
            'type': block_type,
            'tf': tf,
            'label': f"Int_{block_id}"
        })
        st.sidebar.success(f"Bloco {block_id} adicionado!")
        
    elif block_type == "Derivador":
        tf = ct.TransferFunction([1, 0], [1])
        st.session_state.blocks.append({
            'id': block_id,
            'type': block_type,
            'tf': tf,
            'label': f"Der_{block_id}"
        })
        st.sidebar.success(f"Bloco {block_id} adicionado!")
        
    elif block_type == "Atraso":
        tf = ct.TransferFunction([1], [tau, 1])
        st.session_state.blocks.append({
            'id': block_id,
            'type': block_type,
            'tf': tf,
            'tau': tau,
            'label': f"Lag_{block_id}"
        })
        st.sidebar.success(f"Bloco {block_id} adicionado!")

# Botão para limpar todos os blocos
if st.sidebar.button("🗑️ Limpar Todos os Blocos"):
    st.session_state.blocks = []
    st.session_state.connections = []
    st.sidebar.success("Todos os blocos removidos!")

# Mostrar blocos adicionados
if st.session_state.blocks:
    st.sidebar.subheader("Blocos Adicionados")
    for block in st.session_state.blocks:
        with st.sidebar.expander(f"Bloco {block['id']}: {block['type']}"):
            st.write(f"**Label:** {block['label']}")
            if block['type'] == "Função de Transferência":
                st.write(f"Num: {block['num']}")
                st.write(f"Den: {block['den']}")
            elif block['type'] == "Ganho":
                st.write(f"K = {block['gain']}")
            elif block['type'] == "Atraso":
                st.write(f"τ = {block['tau']}")

# Área principal
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("⚙️ Configuração do Sistema")
    
    if st.session_state.blocks:
        # Configuração da conexão em série
        st.markdown("**Conexão em Série**")
        st.info("Os blocos serão conectados em série na ordem em que foram adicionados")
        
        # Parâmetros adicionais para malha fechada
        if system_type == "Malha Fechada":
            st.markdown("**Configuração de Realimentação**")
            feedback_sign = st.radio("Tipo de Realimentação:", ["Negativa", "Positiva"])
            
            use_h = st.checkbox("Usar bloco H(s) na realimentação")
            if use_h:
                st.markdown("**H(s) - Função de Transferência da Realimentação**")
                h_num_str = st.text_input("Numerador H(s):", "1", key="h_num")
                h_den_str = st.text_input("Denominador H(s):", "1", key="h_den")
        
        # Botão para calcular sistema
        if st.button("🔄 Calcular Sistema"):
            try:
                # Conectar blocos em série
                G = st.session_state.blocks[0]['tf']
                for block in st.session_state.blocks[1:]:
                    G = ct.series(G, block['tf'])
                
                st.session_state.G_open = G
                
                # Se for malha fechada, calcular sistema em malha fechada
                if system_type == "Malha Fechada":
                    if use_h:
                        h_num = [float(x.strip()) for x in h_num_str.split(',')]
                        h_den = [float(x.strip()) for x in h_den_str.split(',')]
                        H = ct.TransferFunction(h_num, h_den)
                    else:
                        H = ct.TransferFunction([1], [1])
                    
                    sign = -1 if feedback_sign == "Negativa" else 1
                    st.session_state.G_closed = ct.feedback(G, H, sign=sign)
                    st.session_state.H = H
                
                st.success("✅ Sistema calculado com sucesso!")
                
            except Exception as e:
                st.error(f"Erro ao calcular sistema: {str(e)}")
    else:
        st.warning("⚠️ Adicione blocos para começar")

with col2:
    st.subheader("📊 Diagrama de Blocos")
    
    if st.session_state.blocks:
        # Criar figura para diagrama de blocos
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 3)
        ax.axis('off')
        
        num_blocks = len(st.session_state.blocks)
        x_spacing = 8 / (num_blocks + 1)
        
        # Desenhar blocos
        for i, block in enumerate(st.session_state.blocks):
            x = 1 + i * x_spacing
            y = 1.5
            
            # Caixa do bloco
            box = FancyBboxPatch((x-0.3, y-0.2), 0.6, 0.4,
                                boxstyle="round,pad=0.05", 
                                edgecolor='blue', facecolor='lightblue',
                                linewidth=2)
            ax.add_patch(box)
            
            # Texto do bloco
            if block['type'] == "Ganho":
                text = f"K={block['gain']}"
            elif block['type'] == "Integrador":
                text = "1/s"
            elif block['type'] == "Derivador":
                text = "s"
            elif block['type'] == "Atraso":
                text = f"1/(τs+1)"
            else:
                text = f"G{block['id']}"
            
            ax.text(x, y, text, ha='center', va='center', fontsize=10, fontweight='bold')
            
            # Seta conectando blocos
            if i < num_blocks - 1:
                arrow = FancyArrowPatch((x+0.3, y), (x+x_spacing-0.3, y),
                                      arrowstyle='->', mutation_scale=20,
                                      color='black', linewidth=2)
                ax.add_patch(arrow)
        
        # Seta de entrada
        arrow_in = FancyArrowPatch((0.3, 1.5), (0.7, 1.5),
                                  arrowstyle='->', mutation_scale=20,
                                  color='green', linewidth=2)
        ax.add_patch(arrow_in)
        ax.text(0.2, 1.7, 'R(s)', fontsize=10, color='green')
        
        # Seta de saída
        x_last = 1 + (num_blocks-1) * x_spacing
        arrow_out = FancyArrowPatch((x_last+0.3, 1.5), (9.5, 1.5),
                                   arrowstyle='->', mutation_scale=20,
                                   color='red', linewidth=2)
        ax.add_patch(arrow_out)
        ax.text(9.6, 1.7, 'Y(s)', fontsize=10, color='red')
        
        # Se for malha fechada, desenhar realimentação
        if system_type == "Malha Fechada":
            # Linha de realimentação
            x_end = 9.3
            ax.plot([x_end, x_end, 0.5, 0.5], [1.5, 0.5, 0.5, 1.3],
                   'b--', linewidth=2)
            
            # Círculo de soma
            circle = Circle((0.5, 1.5), 0.15, edgecolor='black', 
                          facecolor='white', linewidth=2)
            ax.add_patch(circle)
            ax.text(0.35, 1.5, '+', fontsize=12, fontweight='bold')
            ax.text(0.5, 1.25, '-' if feedback_sign == "Negativa" else '+', 
                   fontsize=12, fontweight='bold')
            
            # Bloco H(s) se houver
            if 'H' in st.session_state:
                h_box = FancyBboxPatch((4.5, 0.3), 0.6, 0.4,
                                     boxstyle="round,pad=0.05",
                                     edgecolor='purple', facecolor='lavender',
                                     linewidth=2)
                ax.add_patch(h_box)
                ax.text(4.8, 0.5, 'H(s)', ha='center', va='center', 
                       fontsize=10, fontweight='bold')
        
        st.pyplot(fig)
        plt.close()
    else:
        st.info("Adicione blocos para visualizar o diagrama")

# Análise do sistema
if 'G_open' in st.session_state:
    st.header("📈 Análise do Sistema")
    
    # Tabs para diferentes análises
    if system_type == "Malha Aberta":
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "Resposta no Tempo", 
            "Diagrama de Bode", 
            "Polos e Zeros",
            "Diagrama de Nyquist",
            "Lugar das Raízes"
        ])
    else:
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
            "Resposta no Tempo", 
            "Diagrama de Bode", 
            "Polos e Zeros",
            "Diagrama de Nyquist",
            "Lugar das Raízes",
            "Desempenho"
        ])
    
    # TAB 1: Resposta no Tempo
    with tab1:
        st.subheader("Resposta no Tempo")
        
        col_a, col_b = st.columns(2)
        with col_a:
            response_type = st.selectbox("Tipo de Resposta:", 
                                        ["Degrau", "Impulso", "Rampa"])
        with col_b:
            t_final = st.number_input("Tempo Final (s):", value=10.0, step=1.0)
        
        if st.button("Calcular Resposta no Tempo"):
            t = np.linspace(0, t_final, 1000)
            
            fig, axes = plt.subplots(1, 2 if system_type == "Malha Fechada" else 1, 
                                    figsize=(14, 5))
            
            if system_type == "Malha Aberta":
                axes = [axes]
            
            # Malha Aberta
            if response_type == "Degrau":
                t_out, y_out = ct.step_response(st.session_state.G_open, t)
                title = "Resposta ao Degrau"
            elif response_type == "Impulso":
                t_out, y_out = ct.impulse_response(st.session_state.G_open, t)
                title = "Resposta ao Impulso"
            else:  # Rampa
                t_out, y_out = ct.step_response(st.session_state.G_open/ct.TransferFunction([1], [1, 0]), t)
                title = "Resposta à Rampa"
            
            axes[0].plot(t_out, y_out, 'b-', linewidth=2, label='Malha Aberta')
            axes[0].grid(True, alpha=0.3)
            axes[0].set_xlabel('Tempo (s)', fontsize=11)
            axes[0].set_ylabel('Amplitude', fontsize=11)
            axes[0].set_title(f'{title} - Malha Aberta', fontsize=12, fontweight='bold')
            axes[0].legend()
            
            # Malha Fechada
            if system_type == "Malha Fechada" and 'G_closed' in st.session_state:
                if response_type == "Degrau":
                    t_out, y_out = ct.step_response(st.session_state.G_closed, t)
                elif response_type == "Impulso":
                    t_out, y_out = ct.impulse_response(st.session_state.G_closed, t)
                else:  # Rampa
                    t_out, y_out = ct.step_response(st.session_state.G_closed/ct.TransferFunction([1], [1, 0]), t)
                
                axes[1].plot(t_out, y_out, 'r-', linewidth=2, label='Malha Fechada')
                axes[1].grid(True, alpha=0.3)
                axes[1].set_xlabel('Tempo (s)', fontsize=11)
                axes[1].set_ylabel('Amplitude', fontsize=11)
                axes[1].set_title(f'{title} - Malha Fechada', fontsize=12, fontweight='bold')
                axes[1].legend()
            
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
    
    # TAB 2: Diagrama de Bode
    with tab2:
        st.subheader("Diagrama de Bode")
        
        col_a, col_b = st.columns(2)
        with col_a:
            w_min = st.number_input("Frequência Mínima (rad/s):", value=0.01, format="%.3f")
        with col_b:
            w_max = st.number_input("Frequência Máxima (rad/s):", value=100.0, format="%.1f")
        
        if st.button("Gerar Diagrama de Bode"):
            w = np.logspace(np.log10(w_min), np.log10(w_max), 1000)
            
            fig, axes = plt.subplots(2, 2 if system_type == "Malha Fechada" else 1,
                                    figsize=(14, 8))
            
            if system_type == "Malha Aberta":
                axes = axes.reshape(-1, 1)
            
            # Malha Aberta
            mag, phase, omega = ct.bode(st.session_state.G_open, w, plot=False)
            mag_db = 20 * np.log10(mag)
            phase_deg = np.rad2deg(phase)
            
            axes[0, 0].semilogx(omega, mag_db, 'b-', linewidth=2)
            axes[0, 0].grid(True, which='both', alpha=0.3)
            axes[0, 0].set_ylabel('Magnitude (dB)', fontsize=11)
            axes[0, 0].set_title('Diagrama de Bode - Malha Aberta', fontsize=12, fontweight='bold')
            
            axes[1, 0].semilogx(omega, phase_deg, 'b-', linewidth=2)
            axes[1, 0].grid(True, which='both', alpha=0.3)
            axes[1, 0].set_xlabel('Frequência (rad/s)', fontsize=11)
            axes[1, 0].set_ylabel('Fase (graus)', fontsize=11)
            
            # Malha Fechada
            if system_type == "Malha Fechada" and 'G_closed' in st.session_state:
                mag, phase, omega = ct.bode(st.session_state.G_closed, w, plot=False)
                mag_db = 20 * np.log10(mag)
                phase_deg = np.rad2deg(phase)
                
                axes[0, 1].semilogx(omega, mag_db, 'r-', linewidth=2)
                axes[0, 1].grid(True, which='both', alpha=0.3)
                axes[0, 1].set_ylabel('Magnitude (dB)', fontsize=11)
                axes[0, 1].set_title('Diagrama de Bode - Malha Fechada', fontsize=12, fontweight='bold')
                
                axes[1, 1].semilogx(omega, phase_deg, 'r-', linewidth=2)
                axes[1, 1].grid(True, which='both', alpha=0.3)
                axes[1, 1].set_xlabel('Frequência (rad/s)', fontsize=11)
                axes[1, 1].set_ylabel('Fase (graus)', fontsize=11)
            
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
            
            # Informações de margem de ganho e fase
            if system_type == "Malha Aberta":
                gm, pm, wpc, wgc = ct.margin(st.session_state.G_open)
                st.info(f"""
                **Margens de Estabilidade (Malha Aberta):**
                - Margem de Ganho: {20*np.log10(gm):.2f} dB
                - Margem de Fase: {pm:.2f}°
                - Frequência de Cruzamento de Ganho: {wgc:.3f} rad/s
                - Frequência de Cruzamento de Fase: {wpc:.3f} rad/s
                """)
    
    # TAB 3: Polos e Zeros
    with tab3:
        st.subheader("Diagrama de Polos e Zeros")
        
        if st.button("Gerar Diagrama de Polos e Zeros"):
            fig, axes = plt.subplots(1, 2 if system_type == "Malha Fechada" else 1,
                                    figsize=(14, 6))
            
            if system_type == "Malha Aberta":
                axes = [axes]
            
            # Malha Aberta
            poles_open = ct.pole(st.session_state.G_open)
            zeros_open = ct.zero(st.session_state.G_open)
            
            axes[0].axhline(y=0, color='k', linewidth=0.5)
            axes[0].axvline(x=0, color='k', linewidth=0.5)
            axes[0].plot(np.real(poles_open), np.imag(poles_open), 'rx', 
                        markersize=12, markeredgewidth=2, label='Polos')
            axes[0].plot(np.real(zeros_open), np.imag(zeros_open), 'bo', 
                        markersize=10, markeredgewidth=2, label='Zeros')
            axes[0].grid(True, alpha=0.3)
            axes[0].set_xlabel('Parte Real', fontsize=11)
            axes[0].set_ylabel('Parte Imaginária', fontsize=11)
            axes[0].set_title('Polos e Zeros - Malha Aberta', fontsize=12, fontweight='bold')
            axes[0].legend()
            axes[0].axis('equal')
            
            # Malha Fechada
            if system_type == "Malha Fechada" and 'G_closed' in st.session_state:
                poles_closed = ct.pole(st.session_state.G_closed)
                zeros_closed = ct.zero(st.session_state.G_closed)
                
                axes[1].axhline(y=0, color='k', linewidth=0.5)
                axes[1].axvline(x=0, color='k', linewidth=0.5)
                axes[1].plot(np.real(poles_closed), np.imag(poles_closed), 'rx',
                           markersize=12, markeredgewidth=2, label='Polos')
                axes[1].plot(np.real(zeros_closed), np.imag(zeros_closed), 'bo',
                           markersize=10, markeredgewidth=2, label='Zeros')
                axes[1].grid(True, alpha=0.3)
                axes[1].set_xlabel('Parte Real', fontsize=11)
                axes[1].set_ylabel('Parte Imaginária', fontsize=11)
                axes[1].set_title('Polos e Zeros - Malha Fechada', fontsize=12, fontweight='bold')
                axes[1].legend()
                axes[1].axis('equal')
            
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
            
            # Informações numéricas
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Malha Aberta:**")
                st.write(f"Polos: {poles_open}")
                st.write(f"Zeros: {zeros_open}")
                
            if system_type == "Malha Fechada":
                with col2:
                    st.write("**Malha Fechada:**")
                    st.write(f"Polos: {poles_closed}")
                    st.write(f"Zeros: {zeros_closed}")
    
    # TAB 4: Diagrama de Nyquist
    with tab4:
        st.subheader("Diagrama de Nyquist")
        
        if st.button("Gerar Diagrama de Nyquist"):
            fig, axes = plt.subplots(1, 2 if system_type == "Malha Fechada" else 1,
                                    figsize=(14, 6))
            
            if system_type == "Malha Aberta":
                axes = [axes]
            
            w = np.logspace(-2, 3, 1000)
            
            # Malha Aberta
            count, contour = ct.nyquist_plot(st.session_state.G_open, w, 
                                            plot=True, ax=axes[0])
            axes[0].set_title('Diagrama de Nyquist - Malha Aberta', 
                            fontsize=12, fontweight='bold')
            axes[0].grid(True, alpha=0.3)
            
            # Adicionar ponto crítico
            axes[0].plot(-1, 0, 'ro', markersize=10, label='Ponto Crítico (-1, 0)')
            axes[0].legend()
            
            # Malha Fechada
            if system_type == "Malha Fechada" and 'G_closed' in st.session_state:
                count, contour = ct.nyquist_plot(st.session_state.G_closed, w,
                                                plot=True, ax=axes[1])
                axes[1].set_title('Diagrama de Nyquist - Malha Fechada',
                                fontsize=12, fontweight='bold')
                axes[1].grid(True, alpha=0.3)
                axes[1].plot(-1, 0, 'ro', markersize=10, label='Ponto Crítico (-1, 0)')
                axes[1].legend()
            
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
    
    # TAB 5: Lugar das Raízes
    with tab5:
        st.subheader("Lugar Geométrico das Raízes (Root Locus)")
        
        if st.button("Gerar Lugar das Raízes"):
            fig, axes = plt.subplots(1, 2 if system_type == "Malha Fechada" else 1,
                                    figsize=(14, 6))
            
            if system_type == "Malha Aberta":
                axes = [axes]
            
            # Malha Aberta
            ct.root_locus(st.session_state.G_open, ax=axes[0], grid=True)
            axes[0].set_title('Lugar das Raízes - Malha Aberta',
                            fontsize=12, fontweight='bold')
            
            # Malha Fechada
            if system_type == "Malha Fechada" and 'G_closed' in st.session_state:
                ct.root_locus(st.session_state.G_closed, ax=axes[1], grid=True)
                axes[1].set_title('Lugar das Raízes - Malha Fechada',
                                fontsize=12, fontweight='bold')
            
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
    
    # TAB 6: Desempenho (apenas para malha fechada)
    if system_type == "Malha Fechada":
        with tab6:
            st.subheader("Análise de Desempenho - Malha Fechada")
            
            if st.button("Calcular Métricas de Desempenho"):
                try:
                    # Resposta ao degrau
                    t = np.linspace(0, 20, 2000)
                    t_out, y_out = ct.step_response(st.session_state.G_closed, t)
                    
                    # Calcular métricas
                    info = ct.step_info(st.session_state.G_closed)
                    
                    # Plotar resposta ao degrau com métricas
                    fig, ax = plt.subplots(figsize=(12, 6))
                    ax.plot(t_out, y_out, 'b-', linewidth=2, label='Resposta ao Degrau')
                    
                    # Valor final
                    y_final = y_out[-1]
                    ax.axhline(y=y_final, color='g', linestyle='--', 
                              label=f'Valor Final = {y_final:.3f}')
                    
                    # Banda de 2%
                    ax.axhline(y=y_final*1.02, color='r', linestyle=':', alpha=0.5)
                    ax.axhline(y=y_final*0.98, color='r', linestyle=':', alpha=0.5,
                              label='Banda ±2%')
                    
                    ax.grid(True, alpha=0.3)
                    ax.set_xlabel('Tempo (s)', fontsize=11)
                    ax.set_ylabel('Amplitude', fontsize=11)
                    ax.set_title('Resposta ao Degrau com Métricas de Desempenho',
                               fontsize=12, fontweight='bold')
                    ax.legend()
                    
                    st.pyplot(fig)
                    plt.close()
                    
                    # Exibir métricas
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Tempo de Subida (Rise Time)", 
                                f"{info['RiseTime']:.3f} s")
                        st.metric("Tempo de Pico (Peak Time)", 
                                f"{info['PeakTime']:.3f} s")
                    
                    with col2:
                        st.metric("Tempo de Acomodação (Settling Time)", 
                                f"{info['SettlingTime']:.3f} s")
                        st.metric("Overshoot (%)", 
                                f"{info['Overshoot']:.2f} %")
                    
                    with col3:
                        st.metric("Undershoot (%)", 
                                f"{info['Undershoot']:.2f} %")
                        st.metric("Valor Final", 
                                f"{info['SteadyStateValue']:.3f}")
                    
                    # Informações adicionais
                    st.info(f"""
                    **Análise de Estabilidade:**
                    - Sistema é {'**ESTÁVEL**' if np.all(np.real(ct.pole(st.session_state.G_closed)) < 0) else '**INSTÁVEL**'}
                    - Todos os polos estão no semiplano esquerdo: {np.all(np.real(ct.pole(st.session_state.G_closed)) < 0)}
                    """)
                    
                except Exception as e:
                    st.error(f"Erro ao calcular métricas: {str(e)}")

# Footer
st.markdown("---")
st.markdown("""
**Instruções:**
1. Adicione blocos usando o painel lateral
2. Configure o tipo de sistema (Malha Aberta/Fechada)
3. Clique em 'Calcular Sistema' para conectar os blocos
4. Analise o sistema usando as abas de análise
""")
