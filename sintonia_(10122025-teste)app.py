"""
Simulador de Controle - Xcos/Simulink Style
Análise de sistemas de controle com interface drag-and-drop
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import control as ct
from scipy import signal
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QPushButton, QLabel, QSpinBox, QDoubleSpinBox, QComboBox,
                             QDialog, QFormLayout, QGraphicsView, QGraphicsScene, QGraphicsItem,
                             QGraphicsRectItem, QGraphicsLineItem, QGraphicsTextItem, QMessageBox,
                             QTabWidget, QGroupBox, QListWidget, QListWidgetItem)
from PyQt5.QtCore import Qt, QPointF, QRectF, pyqtSignal, QMimeData, QSize
from PyQt5.QtGui import QColor, QPen, QBrush, QFont, QDrag, QPixmap
from PyQt5.QtWidgets import QTableWidget, QTableWidgetItem
import json
from pathlib import Path


class Block:
    """Classe para representar um bloco de controle"""
    block_counter = 0
    
    def __init__(self, block_type, params=None):
        Block.block_counter += 1
        self.id = Block.block_counter
        self.type = block_type
        self.params = params or {}
        self.inputs = []
        self.output = None
        self.x = 0
        self.y = 0
        
    def get_transfer_function(self):
        """Retorna função de transferência do bloco"""
        if self.type == "Ganho":
            K = self.params.get("K", 1.0)
            return ct.TransferFunction([K], [1])
        
        elif self.type == "Integrador":
            return ct.TransferFunction([1], [1, 0])
        
        elif self.type == "Derivador":
            return ct.TransferFunction([1, 0], [1])
        
        elif self.type == "1ª Ordem":
            K = self.params.get("K", 1.0)
            tau = self.params.get("tau", 1.0)
            return ct.TransferFunction([K], [tau, 1])
        
        elif self.type == "2ª Ordem":
            wn = self.params.get("wn", 1.0)
            zeta = self.params.get("zeta", 0.7)
            K = self.params.get("K", 1.0)
            return ct.TransferFunction([K * wn**2], [1, 2*zeta*wn, wn**2])
        
        elif self.type == "Entrada":
            return ct.TransferFunction([1], [1])
        
        return ct.TransferFunction([1], [1])
    
    def __repr__(self):
        return f"{self.type} (ID: {self.id})"


class BlockItem(QGraphicsRectItem):
    """Classe para representar bloco na cena gráfica"""
    
    def __init__(self, block, x=0, y=0):
        super().__init__(0, 0, 100, 60)
        self.block = block
        self.setPos(x, y)
        self.setBrush(QBrush(QColor(100, 150, 255)))
        self.setPen(QPen(QColor(0, 0, 0), 2))
        self.setAcceptHoverEvents(True)
        self.setFlag(QGraphicsItem.ItemIsMovable, True)
        self.setFlag(QGraphicsItem.ItemIsSelectable, True)
        
        # Texto do bloco
        self.text_item = QGraphicsTextItem(self)
        self.text_item.setPlainText(f"{block.type}\nID:{block.id}")
        self.text_item.setDefaultTextColor(QColor(255, 255, 255))
        font = QFont("Arial", 8)
        self.text_item.setFont(font)
        
    def mouseMoveEvent(self, event):
        super().mouseMoveEvent(event)
        self.block.x = self.pos().x()
        self.block.y = self.pos().y()
        
    def mouseDoubleClickEvent(self, event):
        """Abre diálogo para editar parâmetros"""
        self.scene().parent_widget.edit_block(self.block)


class ConnectionLine(QGraphicsLineItem):
    """Linha de conexão entre blocos"""
    
    def __init__(self, from_block, to_block):
        super().__init__()
        self.from_block = from_block
        self.to_block = to_block
        self.setPen(QPen(QColor(0, 0, 0), 2))
        self.update_line()
        
    def update_line(self):
        """Atualiza posição da linha"""
        x1, y1 = self.from_block.block.x + 100, self.from_block.block.y + 30
        x2, y2 = self.to_block.block.x, self.to_block.block.y + 30
        self.setLine(x1, y1, x2, y2)


class BlockEditor(QDialog):
    """Diálogo para editar parâmetros de blocos"""
    
    def __init__(self, block, parent=None):
        super().__init__(parent)
        self.block = block
        self.setWindowTitle(f"Editar {block.type} (ID: {block.id})")
        self.setGeometry(100, 100, 400, 300)
        
        layout = QFormLayout()
        self.inputs = {}
        
        if block.type == "Ganho":
            spin = QDoubleSpinBox()
            spin.setValue(block.params.get("K", 1.0))
            spin.setRange(-1000, 1000)
            spin.setSingleStep(0.1)
            self.inputs["K"] = spin
            layout.addRow("Ganho (K):", spin)
        
        elif block.type == "1ª Ordem":
            spin_K = QDoubleSpinBox()
            spin_K.setValue(block.params.get("K", 1.0))
            spin_K.setRange(-1000, 1000)
            self.inputs["K"] = spin_K
            layout.addRow("Ganho (K):", spin_K)
            
            spin_tau = QDoubleSpinBox()
            spin_tau.setValue(block.params.get("tau", 1.0))
            spin_tau.setRange(0.001, 1000)
            spin_tau.setSingleStep(0.1)
            self.inputs["tau"] = spin_tau
            layout.addRow("Constante de Tempo (τ):", spin_tau)
        
        elif block.type == "2ª Ordem":
            spin_K = QDoubleSpinBox()
            spin_K.setValue(block.params.get("K", 1.0))
            self.inputs["K"] = spin_K
            layout.addRow("Ganho (K):", spin_K)
            
            spin_wn = QDoubleSpinBox()
            spin_wn.setValue(block.params.get("wn", 1.0))
            spin_wn.setRange(0.001, 1000)
            spin_wn.setSingleStep(0.1)
            self.inputs["wn"] = spin_wn
            layout.addRow("Freq. Natural (ωn):", spin_wn)
            
            spin_zeta = QDoubleSpinBox()
            spin_zeta.setValue(block.params.get("zeta", 0.7))
            spin_zeta.setRange(0, 2)
            spin_zeta.setSingleStep(0.1)
            self.inputs["zeta"] = spin_zeta
            layout.addRow("Amortecimento (ζ):", spin_zeta)
        
        # Botão OK
        btn_ok = QPushButton("OK")
        btn_ok.clicked.connect(self.accept)
        layout.addRow(btn_ok)
        
        self.setLayout(layout)
    
    def accept(self):
        """Salva parâmetros"""
        for param, widget in self.inputs.items():
            self.block.params[param] = widget.value()
        super().accept()


class ControlSimulator(QMainWindow):
    """Aplicação principal do simulador"""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Simulador de Controle - Xcos/Simulink")
        self.setGeometry(50, 50, 1400, 900)
        
        self.blocks = []
        self.connections = []
        self.block_items = {}
        
        # Layout principal
        main_widget = QWidget()
        main_layout = QHBoxLayout()
        
        # Painel esquerdo - Blocos disponíveis
        left_panel = QWidget()
        left_layout = QVBoxLayout()
        
        left_layout.addWidget(QLabel("📦 Blocos Disponíveis"))
        left_layout.addWidget(self.create_block_list())
        
        left_layout.addWidget(QLabel("\n⚙️ Controles"))
        left_layout.addLayout(self.create_controls())
        
        left_panel.setLayout(left_layout)
        left_panel.setMaximumWidth(200)
        
        # Painel central - Área de desenho
        self.scene = QGraphicsScene()
        self.scene.parent_widget = self
        self.view = QGraphicsView(self.scene)
        self.view.setStyleSheet("background-color: #f0f0f0;")
        
        # Painel direito - Análises
        right_panel = QWidget()
        right_layout = QVBoxLayout()
        
        self.tabs = QTabWidget()
        right_panel.setLayout(right_layout)
        right_layout.addWidget(self.tabs)
        
        # Adicionar abas de análise
        self.add_analysis_tabs()
        
        # Montar layout principal
        main_layout.addWidget(left_panel, 1)
        main_layout.addWidget(self.view, 2)
        main_layout.addWidget(right_panel, 2)
        
        main_widget.setLayout(main_layout)
        self.setCentralWidget(main_widget)
    
    def create_block_list(self):
        """Cria lista de blocos disponíveis"""
        list_widget = QListWidget()
        
        block_types = [
            "Entrada",
            "Ganho",
            "Integrador",
            "Derivador",
            "1ª Ordem",
            "2ª Ordem",
            "Saída"
        ]
        
        for block_type in block_types:
            item = QListWidgetItem(f"➕ {block_type}")
            item.setData(Qt.UserRole, block_type)
            list_widget.addItem(item)
        
        list_widget.itemDoubleClicked.connect(self.add_block_from_list)
        return list_widget
    
    def add_block_from_list(self, item):
        """Adiciona bloco quando clicado na lista"""
        block_type = item.data(Qt.UserRole)
        self.add_block(block_type, 300 + len(self.blocks)*30, 150 + len(self.blocks)*30)
    
    def create_controls(self):
        """Cria controles para o simulador"""
        layout = QVBoxLayout()
        
        # Botões
        btn_simulate = QPushButton("▶ Simular")
        btn_simulate.clicked.connect(self.simulate)
        layout.addWidget(btn_simulate)
        
        btn_connect = QPushButton("🔗 Conectar")
        btn_connect.clicked.connect(self.enable_connection_mode)
        layout.addWidget(btn_connect)
        
        btn_clear = QPushButton("🗑️ Limpar")
        btn_clear.clicked.connect(self.clear_all)
        layout.addWidget(btn_clear)
        
        btn_save = QPushButton("💾 Salvar")
        btn_save.clicked.connect(self.save_project)
        layout.addWidget(btn_save)
        
        btn_load = QPushButton("📂 Carregar")
        btn_load.clicked.connect(self.load_project)
        layout.addWidget(btn_load)
        
        return layout
    
    def add_block(self, block_type, x=200, y=200):
        """Adiciona bloco ao sistema"""
        params = {
            "K": 1.0,
            "tau": 1.0,
            "wn": 1.0,
            "zeta": 0.7
        }
        
        block = Block(block_type, params)
        self.blocks.append(block)
        
        block_item = BlockItem(block, x, y)
        self.scene.addItem(block_item)
        self.block_items[block.id] = block_item
    
    def edit_block(self, block):
        """Edita parâmetros de um bloco"""
        if block.type not in ["Entrada", "Saída"]:
            editor = BlockEditor(block, self)
            editor.exec_()
            self.simulate()
    
    def enable_connection_mode(self):
        """Habilita modo de conexão entre blocos"""
        QMessageBox.information(self, "Conexão", 
                              "Clique em um bloco de saída e depois no bloco de entrada para conectar")
        self.connecting = True
        self.from_block = None
    
    def add_connection(self, from_block, to_block):
        """Adiciona conexão entre blocos"""
        if from_block and to_block and from_block != to_block:
            self.connections.append((from_block, to_block))
            line = ConnectionLine(self.block_items[from_block.id], 
                                self.block_items[to_block.id])
            self.scene.addItem(line)
            self.simulate()
    
    def add_analysis_tabs(self):
        """Adiciona abas de análise"""
        # Aba: Resposta no Tempo
        widget = QWidget()
        layout = QVBoxLayout()
        self.figure_time = Figure(figsize=(5, 4), dpi=100)
        canvas = FigureCanvas(self.figure_time)
        layout.addWidget(canvas)
        widget.setLayout(layout)
        self.tabs.addTab(widget, "📈 Resposta Tempo")
        self.canvas_time = canvas
        
        # Aba: Diagrama de Bode
        widget = QWidget()
        layout = QVBoxLayout()
        self.figure_bode = Figure(figsize=(5, 4), dpi=100)
        canvas = FigureCanvas(self.figure_bode)
        layout.addWidget(canvas)
        widget.setLayout(layout)
        self.tabs.addTab(widget, "📊 Bode")
        self.canvas_bode = canvas
        
        # Aba: Nyquist
        widget = QWidget()
        layout = QVBoxLayout()
        self.figure_nyquist = Figure(figsize=(5, 4), dpi=100)
        canvas = FigureCanvas(self.figure_nyquist)
        layout.addWidget(canvas)
        widget.setLayout(layout)
        self.tabs.addTab(widget, "🌀 Nyquist")
        self.canvas_nyquist = canvas
        
        # Aba: Polos e Zeros
        widget = QWidget()
        layout = QVBoxLayout()
        self.figure_poles = Figure(figsize=(5, 4), dpi=100)
        canvas = FigureCanvas(self.figure_poles)
        layout.addWidget(canvas)
        widget.setLayout(layout)
        self.tabs.addTab(widget, "📍 Polos/Zeros")
        self.canvas_poles = canvas
        
        # Aba: LGR
        widget = QWidget()
        layout = QVBoxLayout()
        self.figure_root = Figure(figsize=(5, 4), dpi=100)
        canvas = FigureCanvas(self.figure_root)
        layout.addWidget(canvas)
        widget.setLayout(layout)
        self.tabs.addTab(widget, "🌳 LGR")
        self.canvas_root = canvas
        
        # Aba: Informações
        widget = QWidget()
        layout = QVBoxLayout()
        self.info_table = QTableWidget()
        self.info_table.setColumnCount(2)
        self.info_table.setHorizontalHeaderLabels(["Propriedade", "Valor"])
        layout.addWidget(self.info_table)
        widget.setLayout(layout)
        self.tabs.addTab(widget, "ℹ️ Dinâmica")
    
    def simulate(self):
        """Executa simulação e atualiza gráficos"""
        try:
            if not self.blocks or not self.connections:
                QMessageBox.warning(self, "Aviso", "Configure blocos e conexões primeiro!")
                return
            
            # Construir função de transferência total
            G = self.get_total_transfer_function()
            
            if G is None:
                return
            
            # Simulação
            self.plot_step_response(G)
            self.plot_bode(G)
            self.plot_nyquist(G)
            self.plot_poles_zeros(G)
            self.plot_root_locus(G)
            self.update_system_info(G)
            
        except Exception as e:
            QMessageBox.critical(self, "Erro", f"Erro na simulação: {str(e)}")
    
    def get_total_transfer_function(self):
        """Obtém função de transferência total do sistema"""
        try:
            if not self.connections:
                return None
            
            # Multiplicar funções em série
            G = None
            for from_block, to_block in self.connections:
                G_block = from_block.get_transfer_function()
                if G is None:
                    G = G_block
                else:
                    G = G * G_block
            
            return G
        except:
            return None
    
    def plot_step_response(self, G):
        """Plota resposta ao degrau"""
        try:
            self.figure_time.clear()
            ax = self.figure_time.add_subplot(111)
            
            t, y = ct.step_response(G, T=np.linspace(0, 10, 500))
            ax.plot(t, y[0], 'b-', linewidth=2)
            ax.grid(True, alpha=0.3)
            ax.set_xlabel('Tempo (s)')
            ax.set_ylabel('Amplitude')
            ax.set_title('Resposta ao Degrau Unitário')
            
            self.canvas_time.draw()
        except Exception as e:
            print(f"Erro ao plotar resposta: {e}")
    
    def plot_bode(self, G):
        """Plota diagrama de Bode"""
        try:
            self.figure_bode.clear()
            
            mag, phase, omega = ct.bode(G, dB=True, Plot=False)
            
            ax1 = self.figure_bode.add_subplot(211)
            ax1.semilogx(omega, mag)
            ax1.grid(True, alpha=0.3, which="both")
            ax1.set_ylabel('Magnitude (dB)')
            ax1.set_title('Diagrama de Bode')
            
            ax2 = self.figure_bode.add_subplot(212)
            ax2.semilogx(omega, phase)
            ax2.grid(True, alpha=0.3, which="both")
            ax2.set_xlabel('Frequência (rad/s)')
            ax2.set_ylabel('Fase (graus)')
            
            self.canvas_bode.draw()
        except Exception as e:
            print(f"Erro ao plotar Bode: {e}")
    
    def plot_nyquist(self, G):
        """Plota diagrama de Nyquist"""
        try:
            self.figure_nyquist.clear()
            ax = self.figure_nyquist.add_subplot(111)
            
            ct.nyquist_plot(G, Plot=False, ax=ax)
            ax.grid(True, alpha=0.3)
            ax.set_title('Diagrama de Nyquist')
            ax.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
            ax.axvline(x=0, color='k', linestyle='-', linewidth=0.5)
            ax.axhline(y=-1, color='r', linestyle='--', linewidth=1, label='Ponto crítico (-1, 0)')
            ax.axvline(x=-1, color='r', linestyle='--', linewidth=1)
            ax.legend()
            
            self.canvas_nyquist.draw()
        except Exception as e:
            print(f"Erro ao plotar Nyquist: {e}")
    
    def plot_poles_zeros(self, G):
        """Plota polos e zeros"""
        try:
            self.figure_poles.clear()
            ax = self.figure_poles.add_subplot(111)
            
            poles = ct.pole(G)
            zeros = ct.zero(G)
            
            if len(poles) > 0:
                ax.scatter(np.real(poles), np.imag(poles), s=200, marker='x', 
                          color='red', linewidths=2, label='Polos')
            if len(zeros) > 0:
                ax.scatter(np.real(zeros), np.imag(zeros), s=200, marker='o', 
                          color='blue', facecolors='none', linewidths=2, label='Zeros')
            
            ax.grid(True, alpha=0.3)
            ax.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
            ax.axvline(x=0, color='k', linestyle='-', linewidth=0.5)
            ax.set_xlabel('Parte Real')
            ax.set_ylabel('Parte Imaginária')
            ax.set_title('Polos e Zeros')
            ax.legend()
            ax.axis('equal')
            
            self.canvas_poles.draw()
        except Exception as e:
            print(f"Erro ao plotar polos/zeros: {e}")
    
    def plot_root_locus(self, G):
        """Plota lugar geométrico das raízes"""
        try:
            self.figure_root.clear()
            ax = self.figure_root.add_subplot(111)
            
            ct.root_locus(G, Plot=False, ax=ax)
            ax.grid(True, alpha=0.3)
            ax.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
            ax.axvline(x=0, color='k', linestyle='-', linewidth=0.5)
            ax.set_xlabel('Parte Real')
            ax.set_ylabel('Parte Imaginária')
            ax.set_title('Lugar Geométrico das Raízes (LGR)')
            
            self.canvas_root.draw()
        except Exception as e:
            print(f"Erro ao plotar LGR: {e}")
    
    def update_system_info(self, G):
        """Atualiza tabela de informações do sistema"""
        try:
            self.info_table.setRowCount(0)
            
            poles = ct.pole(G)
            zeros = ct.zero(G)
            
            info = [
                ("Tipo de Sistema", f"Ordem {len(poles)}"),
                ("Número de Polos", str(len(poles))),
                ("Número de Zeros", str(len(zeros))),
            ]
            
            # Polos
            if len(poles) > 0:
                for i, pole in enumerate(poles):
                    if np.isreal(pole):
                        info.append((f"Polo {i+1}", f"{np.real(pole):.4f}"))
                    else:
                        info.append((f"Polo {i+1}", 
                                   f"{np.real(pole):.4f} ± j{abs(np.imag(pole)):.4f}"))
            
            # Zeros
            if len(zeros) > 0:
                for i, zero in enumerate(zeros):
                    if np.isreal(zero):
                        info.append((f"Zero {i+1}", f"{np.real(zero):.4f}"))
                    else:
                        info.append((f"Zero {i+1}", 
                                   f"{np.real(zero):.4f} ± j{abs(np.imag(zero)):.4f}"))
            
            # Estabilidade
            stable = all(np.real(p) < 0 for p in poles)
            info.append(("Estabilidade", "✓ Estável" if stable else "✗ Instável"))
            
            # Ganho DC
            try:
                dc_gain = float(ct.dcgain(G))
                info.append(("Ganho DC", f"{dc_gain:.4f}"))
            except:
                pass
            
            # Preencher tabela
            self.info_table.setRowCount(len(info))
            for row, (param, value) in enumerate(info):
                self.info_table.setItem(row, 0, QTableWidgetItem(param))
                self.info_table.setItem(row, 1, QTableWidgetItem(value))
            
            self.info_table.resizeColumnsToContents()
        
        except Exception as e:
            print(f"Erro ao atualizar informações: {e}")
    
    def clear_all(self):
        """Limpa todos os blocos"""
        reply = QMessageBox.question(self, "Confirmar", "Deseja limpar tudo?",
                                    QMessageBox.Yes | QMessageBox.No)
        if reply == QMessageBox.Yes:
            self.scene.clear()
            self.blocks = []
            self.connections = []
            self.block_items = {}
            Block.block_counter = 0
    
    def save_project(self):
        """Salva projeto em arquivo JSON"""
        try:
            data = {
                "blocks": [
                    {
                        "id": b.id,
                        "type": b.type,
                        "params": b.params,
                        "x": b.x,
                        "y": b.y
                    }
                    for b in self.blocks
                ],
                "connections": [
                    (c[0].id, c[1].id)
                    for c in self.connections
                ]
            }
            
            with open("sistema_controle.json", "w") as f:
                json.dump(data, f, indent=2)
            
            QMessageBox.information(self, "Sucesso", "Projeto salvo em 'sistema_controle.json'")
        except Exception as e:
            QMessageBox.critical(self, "Erro", f"Erro ao salvar: {str(e)}")
    
    def load_project(self):
        """Carrega projeto de arquivo JSON"""
        try:
            if not Path("sistema_controle.json").exists():
                QMessageBox.warning(self, "Aviso", "Arquivo não encontrado")
                return
            
            with open("sistema_controle.json", "r") as f:
                data = json.load(f)
            
            self.clear_all()
            
            # Reconstruir blocos
            block_map = {}
            for block_data in data["blocks"]:
                block = Block(block_data["type"], block_data["params"])
                block.id = block_data["id"]
                block.x = block_data["x"]
                block.y = block_data["y"]
                Block.block_counter = max(Block.block_counter, block.id)
                self.blocks.append(block)
                block_map[block.id] = block
                
                block_item = BlockItem(block, block.x, block.y)
                self.scene.addItem(block_item)
                self.block_items[block.id] = block_item
            
            # Reconstruir conexões
            for from_id, to_id in data["connections"]:
                from_block = block_map[from_id]
                to_block = block_map[to_id]
                self.connections.append((from_block, to_block))
                line = ConnectionLine(self.block_items[from_id], self.block_items[to_id])
                self.scene.addItem(line)
            
            QMessageBox.information(self, "Sucesso", "Projeto carregado!")
            self.simulate()
        except Exception as e:
            QMessageBox.critical(self, "Erro", f"Erro ao carregar: {str(e)}")


def main():
    app = QApplication(sys.argv)
    window = ControlSimulator()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
