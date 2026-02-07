#!/usr/bin/env python3
"""
XCOS SIMULATOR - Simulador de Sistemas de Controle
Interface gráfica similar ao Xcos (Scilab) em Python
"""

import sys
import json
import numpy as np
from typing import Dict, List, Tuple, Optional
from enum import Enum
from dataclasses import dataclass, asdict
import uuid

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QSplitter, QListWidget, QListWidgetItem, QPushButton, QLabel,
    QSpinBox, QDoubleSpinBox, QComboBox, QTextEdit, QFileDialog,
    QMessageBox, QDialog, QFormLayout, QTabWidget, QTableWidget,
    QTableWidgetItem, QHeaderView, QDockWidget, QProgressBar
)
from PyQt6.QtCore import Qt, QPoint, QRect, QTimer, QThread, pyqtSignal
from PyQt6.QtGui import (
    QPainter, QColor, QPen, QBrush, QFont, QIcon, QPolygon,
    QDrag, QPixmap
)
from PyQt6.QtCore import QMimeData
from scipy import signal
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


class BlockType(Enum):
    """Tipos de blocos disponíveis"""
    # Entrada
    STEP = "step"
    RAMP = "ramp"
    SINE = "sine"
    PULSE = "pulse"
    
    # Dinâmica
    TF = "tf"
    INTEGRATOR = "integrator"
    DERIVATIVE = "derivative"
    GAIN = "gain"
    STATE_SPACE = "state_space"
    DELAY = "delay"
    
    # Operações
    SUM = "sum"
    PRODUCT = "product"
    DIVIDE = "divide"
    
    # Saída
    SCOPE = "scope"
    PLOT = "plot"
    SINK = "sink"
    
    # Controladores
    PID = "pid"
    LEAD = "lead"
    LAG = "lag"


@dataclass
class BlockData:
    """Dados de um bloco"""
    id: str
    type: BlockType
    x: float
    y: float
    params: Dict
    label: str = ""
    
    def to_dict(self):
        return {
            'id': self.id,
            'type': self.type.value,
            'x': self.x,
            'y': self.y,
            'params': self.params,
            'label': self.label
        }
    
    @staticmethod
    def from_dict(data):
        return BlockData(
            id=data['id'],
            type=BlockType(data['type']),
            x=data['x'],
            y=data['y'],
            params=data['params'],
            label=data.get('label', '')
        )


@dataclass
class Connection:
    """Conexão entre dois blocos"""
    from_block_id: str
    to_block_id: str
    from_port: str = "out"
    to_port: str = "in"
    
    def to_dict(self):
        return asdict(self)
    
    @staticmethod
    def from_dict(data):
        return Connection(**data)


class Block:
    """Representa um bloco no diagrama"""
    
    WIDTH = 100
    HEIGHT = 60
    BLOCK_COLORS = {
        'input': '#90EE90',      # Verde claro
        'dynamic': '#87CEEB',    # Azul claro
        'operation': '#FFB6C1',  # Rosa claro
        'output': '#FFD700',     # Ouro
        'controller': '#DDA0DD'  # Orquídea
    }
    
    BLOCK_INFO = {
        BlockType.STEP: {'label': 'Step', 'category': 'input'},
        BlockType.RAMP: {'label': 'Ramp', 'category': 'input'},
        BlockType.SINE: {'label': 'Sine', 'category': 'input'},
        BlockType.PULSE: {'label': 'Pulse', 'category': 'input'},
        BlockType.TF: {'label': 'TF', 'category': 'dynamic'},
        BlockType.INTEGRATOR: {'label': '∫', 'category': 'dynamic'},
        BlockType.DERIVATIVE: {'label': 'd/dt', 'category': 'dynamic'},
        BlockType.GAIN: {'label': 'K', 'category': 'dynamic'},
        BlockType.STATE_SPACE: {'label': 'SS', 'category': 'dynamic'},
        BlockType.DELAY: {'label': 'Delay', 'category': 'dynamic'},
        BlockType.SUM: {'label': 'Σ', 'category': 'operation'},
        BlockType.PRODUCT: {'label': '×', 'category': 'operation'},
        BlockType.DIVIDE: {'label': '÷', 'category': 'operation'},
        BlockType.SCOPE: {'label': 'Scope', 'category': 'output'},
        BlockType.PLOT: {'label': 'Plot', 'category': 'output'},
        BlockType.SINK: {'label': 'Sink', 'category': 'output'},
        BlockType.PID: {'label': 'PID', 'category': 'controller'},
        BlockType.LEAD: {'label': 'Lead', 'category': 'controller'},
        BlockType.LAG: {'label': 'Lag', 'category': 'controller'},
    }
    
    DEFAULT_PARAMS = {
        BlockType.STEP: {'amplitude': 1.0, 'delay': 0},
        BlockType.RAMP: {'slope': 1.0, 'start_time': 0},
        BlockType.SINE: {'amplitude': 1.0, 'frequency': 1.0, 'phase': 0},
        BlockType.PULSE: {'amplitude': 1.0, 'period': 2.0, 'duty': 0.5},
        BlockType.TF: {'numerator': [1], 'denominator': [1, 1]},
        BlockType.INTEGRATOR: {'initial_value': 0},
        BlockType.DERIVATIVE: {},
        BlockType.GAIN: {'K': 1.0},
        BlockType.STATE_SPACE: {'A': [[1]], 'B': [[1]], 'C': [[1]], 'D': [[0]]},
        BlockType.DELAY: {'tau': 0.1},
        BlockType.SUM: {'gains': [1, -1]},
        BlockType.PRODUCT: {},
        BlockType.DIVIDE: {},
        BlockType.SCOPE: {'buffer_size': 10000},
        BlockType.PLOT: {},
        BlockType.SINK: {},
        BlockType.PID: {'Kp': 1.0, 'Ki': 0, 'Kd': 0},
        BlockType.LEAD: {'K': 1.0, 'z': 1.0, 'p': 2.0},
        BlockType.LAG: {'K': 1.0, 'z': 0.5, 'p': 0.1},
    }
    
    def __init__(self, block_type: BlockType, x: float = 0, y: float = 0):
        self.id = str(uuid.uuid4())[:8]
        self.type = block_type
        self.x = x
        self.y = y
        self.width = self.WIDTH
        self.height = self.HEIGHT
        self.params = self.DEFAULT_PARAMS[block_type].copy()
        self.selected = False
        
        info = self.BLOCK_INFO[block_type]
        self.label = info['label']
        self.category = info['category']
        self.color = self.BLOCK_COLORS[self.category]
    
    def get_rect(self):
        """Retorna o retângulo do bloco"""
        return QRect(int(self.x), int(self.y), self.width, self.height)
    
    def contains_point(self, x: float, y: float) -> bool:
        """Verifica se o ponto está dentro do bloco"""
        return self.get_rect().contains(int(x), int(y))
    
    def get_port_positions(self) -> Tuple[QPoint, QPoint]:
        """Retorna posições de entrada e saída"""
        in_port = QPoint(int(self.x), int(self.y + self.height / 2))
        out_port = QPoint(int(self.x + self.width), int(self.y + self.height / 2))
        return in_port, out_port
    
    def draw(self, painter: QPainter):
        """Desenha o bloco"""
        rect = self.get_rect()
        
        # Fundo
        color = QColor(self.color)
        painter.fillRect(rect, QBrush(color))
        
        # Borda
        pen = QPen(Qt.GlobalColor.black, 2)
        if self.selected:
            pen.setWidth(3)
            pen.setColor(Qt.GlobalColor.blue)
        painter.setPen(pen)
        painter.drawRect(rect)
        
        # Texto
        painter.setFont(QFont("Arial", 9, QFont.Weight.Bold))
        painter.setPen(QPen(Qt.GlobalColor.black))
        painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, self.label)
        
        # Portas (entrada e saída)
        painter.fillRect(int(self.x) - 4, int(self.y + self.height/2 - 4), 8, 8, 
                        QBrush(Qt.GlobalColor.black))
        painter.fillRect(int(self.x + self.width) - 4, int(self.y + self.height/2 - 4), 8, 8,
                        QBrush(Qt.GlobalColor.black))


class BlockListWidget(QListWidget):
    """Widget para listar blocos disponíveis com drag-drop"""
    
    def __init__(self):
        super().__init__()
        self.setup_blocks()
        self.setDragEnabled(True)
        self.setStyleSheet("""
            QListWidget {
                background-color: #f0f0f0;
                border: 1px solid #ccc;
            }
            QListWidget::item {
                padding: 5px;
                border-bottom: 1px solid #ddd;
            }
            QListWidget::item:hover {
                background-color: #e0e0e0;
            }
        """)
    
    def setup_blocks(self):
        """Configura blocos disponíveis"""
        categories = {
            'Entradas': [BlockType.STEP, BlockType.RAMP, BlockType.SINE, BlockType.PULSE],
            'Dinâmica': [BlockType.TF, BlockType.INTEGRATOR, BlockType.DERIVATIVE, 
                        BlockType.GAIN, BlockType.STATE_SPACE, BlockType.DELAY],
            'Operações': [BlockType.SUM, BlockType.PRODUCT, BlockType.DIVIDE],
            'Saída': [BlockType.SCOPE, BlockType.PLOT, BlockType.SINK],
            'Controladores': [BlockType.PID, BlockType.LEAD, BlockType.LAG],
        }
        
        for category, block_types in categories.items():
            # Adicionar categoria como item desabilitado
            category_item = QListWidgetItem(f"► {category}")
            category_item.setFlags(category_item.flags() & ~Qt.ItemFlag.ItemIsSelectable)
            category_font = QFont()
            category_font.setBold(True)
            category_item.setFont(category_font)
            self.addItem(category_item)
            
            # Adicionar blocos
            for block_type in block_types:
                info = Block.BLOCK_INFO[block_type]
                item = QListWidgetItem(f"  {info['label']}")
                item.setData(Qt.ItemDataRole.UserRole, block_type.value)
                self.addItem(item)


class Canvas(QWidget):
    """Canvas para desenho dos blocos e conexões"""
    
    block_selected = pyqtSignal(Block)
    blocks_changed = pyqtSignal()
    
    def __init__(self):
        super().__init__()
        self.blocks: Dict[str, Block] = {}
        self.connections: List[Connection] = []
        self.dragging_block: Optional[Block] = None
        self.drag_start_pos = QPoint()
        self.connecting = False
        self.connection_from_block: Optional[Block] = None
        self.temp_connection_end = QPoint()
        
        self.setAcceptDrops(True)
        self.setMouseTracking(True)
        self.setStyleSheet("background-color: #fafafa; border: 1px solid #ccc;")
    
    def add_block(self, block_type: BlockType, x: float, y: float) -> Block:
        """Adiciona um bloco ao canvas"""
        block = Block(block_type, x, y)
        self.blocks[block.id] = block
        self.blocks_changed.emit()
        self.update()
        return block
    
    def remove_block(self, block_id: str):
        """Remove um bloco"""
        if block_id in self.blocks:
            del self.blocks[block_id]
            # Remover conexões relacionadas
            self.connections = [c for c in self.connections 
                              if c.from_block_id != block_id and c.to_block_id != block_id]
            self.blocks_changed.emit()
            self.update()
    
    def add_connection(self, from_block_id: str, to_block_id: str):
        """Adiciona uma conexão entre blocos"""
        # Evitar auto-conexões
        if from_block_id != to_block_id:
            # Remover conexão existente para o mesmo destino
            self.connections = [c for c in self.connections if c.to_block_id != to_block_id]
            self.connections.append(Connection(from_block_id, to_block_id))
            self.blocks_changed.emit()
            self.update()
    
    def remove_connection(self, from_id: str, to_id: str):
        """Remove uma conexão"""
        self.connections = [c for c in self.connections 
                           if not (c.from_block_id == from_id and c.to_block_id == to_id)]
        self.blocks_changed.emit()
        self.update()
    
    def dragEnterEvent(self, event):
        """Aceita drag de blocos"""
        if event.mimeData().hasFormat('text/plain'):
            event.acceptProposedAction()
    
    def dragMoveEvent(self, event):
        """Mostra feedback durante drag"""
        event.acceptProposedAction()
    
    def dropEvent(self, event):
        """Processa drop de bloco"""
        mime_data = event.mimeData()
        if mime_data.hasFormat('text/plain'):
            block_type_str = mime_data.text()
            try:
                block_type = BlockType(block_type_str)
                x = event.position().x()
                y = event.position().y()
                self.add_block(block_type, x, y)
            except ValueError:
                pass
    
    def mousePressEvent(self, event):
        """Tratamento de clique do mouse"""
        if event.button() == Qt.MouseButton.LeftButton:
            # Verificar se clicou em um bloco
            for block_id, block in self.blocks.items():
                if block.contains_point(event.position().x(), event.position().y()):
                    # Desselecionar outros
                    for b in self.blocks.values():
                        b.selected = False
                    block.selected = True
                    self.dragging_block = block
                    self.drag_start_pos = QPoint(int(event.position().x()), 
                                                int(event.position().y()))
                    self.block_selected.emit(block)
                    self.update()
                    return
            
            # Desselecionar todos se clicou no vazio
            for b in self.blocks.values():
                b.selected = False
            self.update()
        
        elif event.button() == Qt.MouseButton.RightButton:
            # Menu de contexto
            for block_id, block in self.blocks.items():
                if block.contains_point(event.position().x(), event.position().y()):
                    # Iniciar conexão
                    if self.connecting:
                        # Tentar conectar
                        if self.connection_from_block and self.connection_from_block != block:
                            self.add_connection(self.connection_from_block.id, block.id)
                            self.connecting = False
                            self.connection_from_block = None
                    else:
                        # Começar conexão
                        self.connecting = True
                        self.connection_from_block = block
                    self.update()
                    return
    
    def mouseMoveEvent(self, event):
        """Movimento do mouse"""
        if self.dragging_block:
            # Mover bloco
            delta_x = event.position().x() - self.drag_start_pos.x()
            delta_y = event.position().y() - self.drag_start_pos.y()
            
            self.dragging_block.x += delta_x
            self.dragging_block.y += delta_y
            
            self.drag_start_pos = QPoint(int(event.position().x()), 
                                        int(event.position().y()))
            self.update()
        
        if self.connecting:
            self.temp_connection_end = QPoint(int(event.position().x()), 
                                             int(event.position().y()))
            self.update()
    
    def mouseReleaseEvent(self, event):
        """Liberação do mouse"""
        self.dragging_block = None
    
    def keyPressEvent(self, event):
        """Teclas"""
        if event.key() == Qt.Key.Key_Delete:
            # Deletar bloco selecionado
            for block_id, block in list(self.blocks.items()):
                if block.selected:
                    self.remove_block(block_id)
        elif event.key() == Qt.Key.Key_Escape:
            if self.connecting:
                self.connecting = False
                self.connection_from_block = None
                self.update()
    
    def paintEvent(self, event):
        """Renderização do canvas"""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        # Desenhar grid
        self.draw_grid(painter)
        
        # Desenhar conexões
        self.draw_connections(painter)
        
        # Desenhar blocos
        for block in self.blocks.values():
            block.draw(painter)
        
        # Desenhar conexão em progresso
        if self.connecting and self.connection_from_block:
            pen = QPen(Qt.GlobalColor.red, 2)
            pen.setStyle(Qt.PenStyle.DashLine)
            painter.setPen(pen)
            in_pos, out_pos = self.connection_from_block.get_port_positions()
            painter.drawLine(out_pos, self.temp_connection_end)
    
    def draw_grid(self, painter: QPainter):
        """Desenha grid de fundo"""
        painter.setPen(QPen(QColor(220, 220, 220), 1))
        grid_size = 20
        
        for x in range(0, self.width(), grid_size):
            painter.drawLine(x, 0, x, self.height())
        for y in range(0, self.height(), grid_size):
            painter.drawLine(0, y, self.width(), y)
    
    def draw_connections(self, painter: QPainter):
        """Desenha conexões entre blocos"""
        painter.setPen(QPen(Qt.GlobalColor.blue, 2))
        
        for conn in self.connections:
            from_block = self.blocks.get(conn.from_block_id)
            to_block = self.blocks.get(conn.to_block_id)
            
            if from_block and to_block:
                _, out_pos = from_block.get_port_positions()
                in_pos, _ = to_block.get_port_positions()
                
                # Desenhar linha curva (Bezier)
                path_x1, path_y1 = out_pos.x(), out_pos.y()
                path_x2, path_y2 = in_pos.x(), in_pos.y()
                ctrl_x = (path_x1 + path_x2) / 2
                
                # Usar quadratic bezier para conexão suave
                painter.drawLine(out_pos, in_pos)
    
    def clear(self):
        """Limpa o canvas"""
        self.blocks.clear()
        self.connections.clear()
        self.update()
    
    def save_to_json(self) -> str:
        """Salva o diagrama em JSON"""
        data = {
            'blocks': [block.__dict__ for block in self.blocks.values()],
            'connections': [asdict(c) for c in self.connections]
        }
        
        # Serializar dados dos blocos
        blocks_data = []
        for block in self.blocks.values():
            block_dict = {
                'id': block.id,
                'type': block.type.value,
                'x': block.x,
                'y': block.y,
                'params': block.params,
                'label': block.label
            }
            blocks_data.append(block_dict)
        
        data['blocks'] = blocks_data
        return json.dumps(data, indent=2)
    
    def load_from_json(self, json_str: str):
        """Carrega diagrama de JSON"""
        try:
            data = json.loads(json_str)
            self.clear()
            
            # Carregar blocos
            for block_data in data.get('blocks', []):
                block_type = BlockType(block_data['type'])
                block = self.add_block(block_type, block_data['x'], block_data['y'])
                block.id = block_data['id']
                block.params = block_data['params']
                self.blocks[block.id] = block
            
            # Carregar conexões
            for conn_data in data.get('connections', []):
                conn = Connection.from_dict(conn_data)
                self.connections.append(conn)
            
            self.update()
        except Exception as e:
            print(f"Erro ao carregar JSON: {e}")


class PropertyPanel(QWidget):
    """Painel de propriedades"""
    
    def __init__(self):
        super().__init__()
        self.current_block: Optional[Block] = None
        self.param_widgets: Dict = {}
        
        layout = QVBoxLayout()
        
        # Label de informação
        self.info_label = QLabel("Nenhum bloco selecionado")
        self.info_label.setStyleSheet("font-weight: bold; color: #333;")
        layout.addWidget(self.info_label)
        
        # Area de parâmetros
        self.params_layout = QVBoxLayout()
        layout.addLayout(self.params_layout)
        
        # Botões
        button_layout = QHBoxLayout()
        
        self.delete_btn = QPushButton("🗑️ Deletar")
        self.delete_btn.clicked.connect(self.on_delete)
        button_layout.addWidget(self.delete_btn)
        
        self.duplicate_btn = QPushButton("📋 Duplicar")
        self.duplicate_btn.clicked.connect(self.on_duplicate)
        button_layout.addWidget(self.duplicate_btn)
        
        layout.addLayout(button_layout)
        layout.addStretch()
        
        self.setLayout(layout)
        self.setStyleSheet("""
            PropertyPanel {
                background-color: #f5f5f5;
                border: 1px solid #ddd;
            }
        """)
    
    def set_block(self, block: Block):
        """Define o bloco selecionado"""
        self.current_block = block
        self.update_display()
    
    def update_display(self):
        """Atualiza a exibição de propriedades"""
        # Limpar widgets anteriores
        for widget in self.param_widgets.values():
            widget.deleteLater()
        self.param_widgets.clear()
        
        if not self.current_block:
            self.info_label.setText("Nenhum bloco selecionado")
            return
        
        self.info_label.setText(f"Bloco: {self.current_block.label} (ID: {self.current_block.id})")
        
        # Adicionar campos de parâmetros
        for param_name, param_value in self.current_block.params.items():
            self.add_param_widget(param_name, param_value)
    
    def add_param_widget(self, name: str, value):
        """Adiciona widget para parâmetro"""
        label = QLabel(f"{name}:")
        
        if isinstance(value, (int, float)):
            widget = QDoubleSpinBox()
            widget.setValue(float(value))
            widget.setRange(-1000, 1000)
            widget.setSingleStep(0.1)
            widget.valueChanged.connect(lambda v: self.on_param_changed(name, v))
        elif isinstance(value, list):
            widget = QTextEdit()
            widget.setText(str(value))
            widget.textChanged.connect(lambda: self.on_param_changed(name, widget.toPlainText()))
        else:
            widget = QTextEdit()
            widget.setText(str(value))
            widget.textChanged.connect(lambda: self.on_param_changed(name, widget.toPlainText()))
        
        self.param_widgets[name] = widget
        
        # Adicionar à layout
        param_layout = QHBoxLayout()
        param_layout.addWidget(label)
        param_layout.addWidget(widget)
        self.params_layout.insertLayout(self.params_layout.count() - 1, param_layout)
    
    def on_param_changed(self, name: str, value):
        """Atualiza parâmetro do bloco"""
        if self.current_block:
            try:
                if isinstance(value, str):
                    # Tentar parsear como lista
                    if value.startswith('['):
                        self.current_block.params[name] = eval(value)
                    else:
                        # Tentar como número
                        self.current_block.params[name] = float(value)
                else:
                    self.current_block.params[name] = value
            except:
                pass
    
    def on_delete(self):
        """Deleta bloco"""
        # Emitir sinal
        pass
    
    def on_duplicate(self):
        """Duplica bloco"""
        # Emitir sinal
        pass


class SimulationThread(QThread):
    """Thread para executar simulação sem bloquear UI"""
    
    progress = pyqtSignal(int)
    finished = pyqtSignal(dict)
    error = pyqtSignal(str)
    
    def __init__(self, canvas: Canvas, t_final: float = 10, num_points: int = 1000):
        super().__init__()
        self.canvas = canvas
        self.t_final = t_final
        self.num_points = num_points
    
    def run(self):
        """Executa a simulação"""
        try:
            self.progress.emit(10)
            
            # Construir função de transferência a partir dos blocos
            if not self.canvas.blocks:
                self.error.emit("Nenhum bloco no diagrama")
                return
            
            # Encontrar blocos de entrada e saída
            input_blocks = [b for b in self.canvas.blocks.values() 
                          if b.type in [BlockType.STEP, BlockType.RAMP, BlockType.SINE]]
            output_blocks = [b for b in self.canvas.blocks.values()
                           if b.type in [BlockType.SCOPE, BlockType.PLOT]]
            
            if not input_blocks or not output_blocks:
                self.error.emit("É necessário pelo menos um bloco de entrada e um de saída")
                return
            
            self.progress.emit(30)
            
            # Extrair funções de transferência
            tf_blocks = [b for b in self.canvas.blocks.values() if b.type == BlockType.TF]
            
            if tf_blocks:
                # Usar primeiro TF encontrado
                tf_block = tf_blocks[0]
                num = tf_block.params.get('numerator', [1])
                den = tf_block.params.get('denominator', [1, 1])
                
                sys = signal.TransferFunction(num, den)
                
                self.progress.emit(50)
                
                # Simular resposta ao degrau
                t = np.linspace(0, self.t_final, self.num_points)
                t, y = signal.step(sys, T=t)
                
                self.progress.emit(80)
                
                # Calcular métricas
                y_final = y[-1]
                y_max = np.max(y)
                overshoot = ((y_max - y_final) / y_final * 100) if y_final != 0 else 0
                
                # Calcular tempo de acomodação (2%)
                tolerance = 0.02 * y_final
                idx_settle = np.where(np.abs(y - y_final) <= tolerance)[0]
                settling_time = t[idx_settle[0]] if len(idx_settle) > 0 else None
                
                # Calcular polos
                poles = np.roots(den)
                is_stable = all(p.real < 0 for p in poles)
                
                self.progress.emit(100)
                
                self.finished.emit({
                    't': t.tolist(),
                    'y': y.tolist(),
                    'metrics': {
                        'steady_state': float(y_final),
                        'overshoot_percent': float(overshoot),
                        'peak_value': float(y_max),
                        'settling_time': float(settling_time) if settling_time else None,
                        'poles': poles.tolist(),
                        'stable': is_stable
                    }
                })
            else:
                self.error.emit("Nenhum bloco Transfer Function encontrado")
        
        except Exception as e:
            self.error.emit(f"Erro na simulação: {str(e)}")


class XcosMainWindow(QMainWindow):
    """Janela principal da aplicação"""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("XCOS Simulator - Simulador de Sistemas de Controle")
        self.setGeometry(100, 100, 1400, 900)
        self.setStyleSheet("""
            QMainWindow {
                background-color: #ffffff;
            }
            QToolBar {
                background-color: #f0f0f0;
                border: 1px solid #ccc;
            }
            QPushButton {
                background-color: #667eea;
                color: white;
                border: none;
                padding: 5px 10px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #5568d3;
            }
            QPushButton:pressed {
                background-color: #4452b8;
            }
        """)
        
        # Canvas central
        self.canvas = Canvas()
        self.canvas.block_selected.connect(self.on_block_selected)
        self.canvas.blocks_changed.connect(self.on_blocks_changed)
        
        # Layout principal
        main_widget = QWidget()
        main_layout = QHBoxLayout()
        
        # Splitter para redimensionamento
        splitter = QSplitter(Qt.Orientation.Horizontal)
        
        # Painel esquerdo - Blocos disponíveis
        left_widget = QWidget()
        left_layout = QVBoxLayout()
        left_layout.addWidget(QLabel("Blocos Disponíveis:"))
        left_layout.addWidget(self.create_block_list())
        left_widget.setLayout(left_layout)
        left_widget.setMaximumWidth(200)
        
        # Canvas
        splitter.addWidget(self.canvas)
        splitter.addWidget(left_widget)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 0)
        
        # Painel direito - Propriedades
        right_splitter = QSplitter(Qt.Orientation.Vertical)
        
        self.property_panel = PropertyPanel()
        right_splitter.addWidget(self.property_panel)
        
        # Painel de saída
        output_widget = QWidget()
        output_layout = QVBoxLayout()
        output_layout.addWidget(QLabel("Informações:"))
        
        self.output_text = QTextEdit()
        self.output_text.setReadOnly(True)
        self.output_text.setMaximumHeight(200)
        output_layout.addWidget(self.output_text)
        
        output_widget.setLayout(output_layout)
        right_splitter.addWidget(output_widget)
        right_splitter.setStretchFactor(0, 1)
        right_splitter.setStretchFactor(1, 0)
        
        splitter.addWidget(right_splitter)
        
        main_layout.addWidget(splitter)
        main_widget.setLayout(main_layout)
        
        self.setCentralWidget(main_widget)
        
        # Criar toolbar
        self.create_toolbar()
        
        # Criar menus
        self.create_menus()
        
        # Thread de simulação
        self.sim_thread: Optional[SimulationThread] = None
        
        self.show()
    
    def create_block_list(self) -> BlockListWidget:
        """Cria lista de blocos"""
        block_list = BlockListWidget()
        block_list.model().rowsMoved.connect(self.on_blocks_changed)
        return block_list
    
    def create_toolbar(self):
        """Cria toolbar"""
        toolbar = self.addToolBar("Ferramentas")
        toolbar.setMovable(False)
        
        # Novo
        new_btn = QPushButton("📄 Novo")
        new_btn.clicked.connect(self.new_diagram)
        toolbar.addWidget(new_btn)
        
        # Abrir
        open_btn = QPushButton("📂 Abrir")
        open_btn.clicked.connect(self.open_diagram)
        toolbar.addWidget(open_btn)
        
        # Salvar
        save_btn = QPushButton("💾 Salvar")
        save_btn.clicked.connect(self.save_diagram)
        toolbar.addWidget(save_btn)
        
        toolbar.addSeparator()
        
        # Simular
        sim_btn = QPushButton("▶️ Simular")
        sim_btn.clicked.connect(self.simulate)
        toolbar.addWidget(sim_btn)
        
        # Análise
        analysis_btn = QPushButton("📊 Análise")
        analysis_btn.clicked.connect(self.show_analysis)
        toolbar.addWidget(analysis_btn)
        
        toolbar.addSeparator()
        
        # Limpar
        clear_btn = QPushButton("🗑️ Limpar")
        clear_btn.clicked.connect(self.clear_canvas)
        toolbar.addWidget(clear_btn)
        
        # Progresso
        toolbar.addSeparator()
        self.progress_bar = QProgressBar()
        self.progress_bar.setMaximumWidth(200)
        self.progress_bar.setVisible(False)
        toolbar.addWidget(self.progress_bar)
    
    def create_menus(self):
        """Cria menus"""
        menubar = self.menuBar()
        
        # Menu Arquivo
        file_menu = menubar.addMenu("&Arquivo")
        
        new_action = file_menu.addAction("&Novo")
        new_action.triggered.connect(self.new_diagram)
        new_action.setShortcut("Ctrl+N")
        
        open_action = file_menu.addAction("&Abrir")
        open_action.triggered.connect(self.open_diagram)
        open_action.setShortcut("Ctrl+O")
        
        save_action = file_menu.addAction("&Salvar")
        save_action.triggered.connect(self.save_diagram)
        save_action.setShortcut("Ctrl+S")
        
        file_menu.addSeparator()
        
        exit_action = file_menu.addAction("&Sair")
        exit_action.triggered.connect(self.close)
        exit_action.setShortcut("Ctrl+Q")
        
        # Menu Simulação
        sim_menu = menubar.addMenu("&Simulação")
        
        run_action = sim_menu.addAction("&Executar")
        run_action.triggered.connect(self.simulate)
        run_action.setShortcut("F5")
        
        # Menu Edição
        edit_menu = menubar.addMenu("&Edição")
        
        clear_action = edit_menu.addAction("&Limpar Tudo")
        clear_action.triggered.connect(self.clear_canvas)
    
    def on_block_selected(self, block: Block):
        """Bloco foi selecionado"""
        self.property_panel.set_block(block)
    
    def on_blocks_changed(self):
        """Blocos foram modificados"""
        self.update_info()
    
    def update_info(self):
        """Atualiza painel de informações"""
        num_blocks = len(self.canvas.blocks)
        num_connections = len(self.canvas.connections)
        
        info = f"Blocos: {num_blocks} | Conexões: {num_connections}\n"
        
        # Listar tipos de blocos
        block_types = {}
        for block in self.canvas.blocks.values():
            block_type = block.type.value
            block_types[block_type] = block_types.get(block_type, 0) + 1
        
        for block_type, count in block_types.items():
            info += f"  • {block_type}: {count}\n"
        
        self.output_text.setText(info)
    
    def new_diagram(self):
        """Novo diagrama"""
        reply = QMessageBox.question(self, "Novo Diagrama", 
                                     "Descartar diagrama atual?",
                                     QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        if reply == QMessageBox.StandardButton.Yes:
            self.canvas.clear()
    
    def open_diagram(self):
        """Abre diagrama"""
        file_path, _ = QFileDialog.getOpenFileName(self, "Abrir Diagrama", "", "JSON Files (*.json)")
        if file_path:
            try:
                with open(file_path, 'r') as f:
                    self.canvas.load_from_json(f.read())
                QMessageBox.information(self, "Sucesso", "Diagrama carregado!")
            except Exception as e:
                QMessageBox.critical(self, "Erro", f"Erro ao abrir arquivo: {e}")
    
    def save_diagram(self):
        """Salva diagrama"""
        file_path, _ = QFileDialog.getSaveFileName(self, "Salvar Diagrama", "", "JSON Files (*.json)")
        if file_path:
            try:
                with open(file_path, 'w') as f:
                    f.write(self.canvas.save_to_json())
                QMessageBox.information(self, "Sucesso", "Diagrama salvo!")
            except Exception as e:
                QMessageBox.critical(self, "Erro", f"Erro ao salvar arquivo: {e}")
    
    def simulate(self):
        """Executa simulação"""
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        
        self.sim_thread = SimulationThread(self.canvas, t_final=10, num_points=1000)
        self.sim_thread.progress.connect(self.progress_bar.setValue)
        self.sim_thread.finished.connect(self.on_simulation_finished)
        self.sim_thread.error.connect(self.on_simulation_error)
        self.sim_thread.start()
        
        self.output_text.setText("Simulando...")
    
    def on_simulation_finished(self, results: dict):
        """Simulação concluída"""
        self.progress_bar.setVisible(False)
        
        metrics = results['metrics']
        info = f"""
        === RESULTADOS DA SIMULAÇÃO ===
        
        Estável: {'✓ SIM' if metrics['stable'] else '✗ NÃO'}
        Valor Final: {metrics['steady_state']:.6f}
        Pico: {metrics['peak_value']:.6f}
        Overshoot: {metrics['overshoot_percent']:.2f}%
        Tempo de Acomodação: {metrics['settling_time']:.4f}s (2%)
        
        Polos: {[f'{p:.4f}' for p in metrics['poles']]}
        """
        
        self.output_text.setText(info)
        
        # Plotar resultado
        self.plot_results(results)
    
    def on_simulation_error(self, error: str):
        """Erro na simulação"""
        self.progress_bar.setVisible(False)
        QMessageBox.critical(self, "Erro na Simulação", error)
        self.output_text.setText(f"Erro: {error}")
    
    def plot_results(self, results: dict):
        """Plota resultados da simulação"""
        t = results['t']
        y = results['y']
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(t, y, 'b-', linewidth=2, label='Resposta')
        ax.axhline(y=y[-1], color='r', linestyle='--', alpha=0.5, label='Valor Final')
        ax.grid(True, alpha=0.3)
        ax.set_xlabel('Tempo (s)')
        ax.set_ylabel('Amplitude')
        ax.set_title('Resposta ao Degrau')
        ax.legend()
        
        plt.tight_layout()
        plt.show()
    
    def show_analysis(self):
        """Mostra análise do sistema"""
        QMessageBox.information(self, "Análise", "Funcionalidade em desenvolvimento")
    
    def clear_canvas(self):
        """Limpa canvas"""
        reply = QMessageBox.question(self, "Limpar", 
                                     "Descartar todos os blocos?",
                                     QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        if reply == QMessageBox.StandardButton.Yes:
            self.canvas.clear()


def main():
    """Função principal"""
    app = QApplication(sys.argv)
    
    # Tema
    app.setStyle('Fusion')
    
    window = XcosMainWindow()
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
