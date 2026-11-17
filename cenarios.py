# -*- coding: utf-8 -*-
"""Biblioteca de cenários guiados do Modo Diagrama de Blocos (CTRL).

Cada cenário traz um modelo no mesmo formato do editor: ``nodes`` com
``id``, ``type``, ``x``, ``y`` e ``params``; ``edges`` com ``id``, ``src``,
``srcPort``, ``dst`` e ``dstPort``. Os tipos e nomes de parâmetros são os de
``visual_blocks_frontend/index.html`` (input, output, tf, gain, sum, branch,
pid com kp/ki/kd).
"""


def _no(id_, tipo, x, y, **params):
    return {"id": id_, "type": tipo, "x": x, "y": y, "params": params}


def _entrada(id_="n1", x=20, y=160):
    return _no(id_, "input", x, y, label="R(s)")


def _saida(id_, x, y=160):
    return _no(id_, "output", x, y, label="Y(s)")


def _ligacoes(*pares):
    """Cada par é (origem, destino) ou (origem, destino, porta_dst, porta_src)."""
    arestas = []
    for i, par in enumerate(pares, start=1):
        origem, destino, *resto = par
        porta_dst = resto[0] if resto else "in0"
        porta_src = resto[1] if len(resto) > 1 else "out0"
        arestas.append({"id": f"e{i}", "src": origem, "srcPort": porta_src,
                        "dst": destino, "dstPort": porta_dst})
    return arestas


def _cadeia(*blocos):
    """Entrada, blocos em série e saída, ligados em sequência (malha aberta)."""
    nos = [_entrada()]
    x = 200
    for i, (tipo, params) in enumerate(blocos, start=2):
        nos.append(_no(f"n{i}", tipo, x, 150, **params))
        x += 190
    saida_id = f"n{len(blocos) + 2}"
    nos.append(_saida(saida_id, x))
    ids = [n["id"] for n in nos]
    return {"nodes": nos, "edges": _ligacoes(*zip(ids, ids[1:]))}


def _malha_unitaria(controlador, planta):
    """R -> somador (+ -) -> controlador -> planta -> ramificação -> Y.

    A segunda saída da ramificação volta direto à entrada negativa do somador
    (realimentação unitária).
    """
    nos = [
        _entrada("n1", 20, 160),
        _no("n2", "sum", 165, 170, signs="+ -"),
        _no("n3", controlador[0], 250, 150, **controlador[1]),
        _no("n4", "tf", 420, 150, **planta),
        _no("n5", "branch", 590, 170),
        _saida("n6", 670),
    ]
    arestas = _ligacoes(
        ("n1", "n2", "in0"), ("n2", "n3"), ("n3", "n4"), ("n4", "n5"),
        ("n5", "n6", "in0", "out0"), ("n5", "n2", "in1", "out1"))
    return {"nodes": nos, "edges": arestas}


CENARIOS = [
    {
        "id": "primeira_ordem",
        "titulo": "1ª ordem: G(s) = 2 / (s + 1)",
        "descricao": (
            "Planta de primeira ordem em malha aberta, com ganho estático 2 e "
            "constante de tempo de 1 s (polo em s = -1)."),
        "o_que_observar": (
            "A resposta ao degrau sobe sem oscilar até o valor final 2 e "
            "atinge cerca de 63% dele em 1 s. Há um único polo real no "
            "semiplano esquerdo, então o selo mostra Estável."),
        "avancado": False,
        "modelo": _cadeia(("tf", {"num": "2", "den": "s+1"})),
    },
    {
        "id": "segunda_subamortecida",
        "titulo": "2ª ordem subamortecida (ζ = 0,25)",
        "descricao": (
            "G(s) = 4 / (s² + s + 4): frequência natural 2 rad/s e "
            "amortecimento 0,25. Os polos são complexos conjugados."),
        "o_que_observar": (
            "A resposta ao degrau oscila e ultrapassa o valor final (alto "
            "sobressinal). No gráfico de polos e zeros, o par complexo fica "
            "perto do eixo imaginário."),
        "avancado": False,
        "modelo": _cadeia(("tf", {"num": "4", "den": "s^2+s+4"})),
    },
    {
        "id": "segunda_critica",
        "titulo": "2ª ordem criticamente amortecida (ζ = 1)",
        "descricao": (
            "G(s) = 4 / (s² + 4s + 4) = 4 / (s + 2)²: polo real duplo em "
            "s = -2."),
        "o_que_observar": (
            "A resposta é a mais rápida possível sem sobressinal. O polo duplo "
            "aparece como um único ponto no semiplano esquerdo."),
        "avancado": False,
        "modelo": _cadeia(("tf", {"num": "4", "den": "s^2+4s+4"})),
    },
    {
        "id": "segunda_super",
        "titulo": "2ª ordem superamortecida (ζ > 1)",
        "descricao": (
            "G(s) = 4 / (s² + 5s + 4) = 4 / ((s + 1)(s + 4)): dois polos "
            "reais distintos, em s = -1 e s = -4."),
        "o_que_observar": (
            "Não há sobressinal e a resposta é mais lenta que a criticamente "
            "amortecida, dominada pelo polo mais próximo da origem (s = -1)."),
        "avancado": False,
        "modelo": _cadeia(("tf", {"num": "4", "den": "s^2+5s+4"})),
    },
    {
        "id": "atraso_pade",
        "titulo": "Sistema com atraso (Padé de 1ª ordem)",
        "descricao": (
            "Planta 1/(s + 1) em série com um atraso de 1 s. O motor só aceita "
            "funções racionais, então o atraso e^(-s) é aproximado por Padé de "
            "1ª ordem: (1 - 0,5s) / (1 + 0,5s). A aproximação vale para "
            "frequências baixas."),
        "o_que_observar": (
            "O zero em s = +2 (semiplano direito) faz a resposta ao degrau "
            "afundar antes de subir. Isso é um artefato da aproximação de "
            "Padé: um atraso real mantém a saída em zero por 1 s e só depois "
            "ela sobe. Use o cenário para ver a ordem de grandeza do atraso, "
            "não a forma exata da curva."),
        "avancado": False,
        "modelo": _cadeia(("tf", {"num": "1", "den": "s+1"}),
                          ("tf", {"num": "1-0.5s", "den": "0.5s+1"})),
    },
    {
        "id": "malha_fechada_k",
        "titulo": "Malha fechada com realimentação unitária e ganho K",
        "descricao": (
            "Ganho K = 5 em série com G(s) = 1 / (s² + 3s + 2), com "
            "realimentação unitária negativa. A T(s) equivalente é "
            "5 / (s² + 3s + 7)."),
        "o_que_observar": (
            "Compare com a planta sozinha: a realimentação move os polos e o "
            "valor final fica abaixo de 1 (erro estacionário). Altere K no "
            "bloco de ganho, calcule de novo e veja os polos se afastarem do "
            "eixo real."),
        "avancado": False,
        "modelo": _malha_unitaria(("gain", {"k": "5"}),
                                  {"num": "1", "den": "s^2+3s+2"}),
    },
    {
        "id": "malha_fechada_pid",
        "titulo": "Malha fechada com PID",
        "descricao": (
            "Controlador PID (Kp = 10, Ki = 5, Kd = 2) em série com "
            "G(s) = 1 / (s² + 3s + 2), com realimentação unitária negativa."),
        "o_que_observar": (
            "A ação integral leva o erro estacionário a zero (valor final 1). "
            "Ajuste Kp, Ki e Kd e observe o efeito no sobressinal e no tempo "
            "de acomodação."),
        "avancado": True,
        "modelo": _malha_unitaria(("pid", {"kp": "10", "ki": "5", "kd": "2"}),
                                  {"num": "1", "den": "s^2+3s+2"}),
    },
    {
        "id": "instavel",
        "titulo": "Caso instável (polo no semiplano direito)",
        "descricao": (
            "Planta G(s) = 1 / (s - 2,5), com polo em s = +2,5, e ganho K = 1 "
            "em malha fechada unitária. A T(s) fica 1 / (s - 1,5), ainda com "
            "polo no semiplano direito."),
        "o_que_observar": (
            "O selo mostra Instável e a resposta ao degrau diverge. Aumente K "
            "para um valor maior que 2,5, calcule de novo e veja o polo "
            "passar para o semiplano esquerdo."),
        "avancado": False,
        "modelo": _malha_unitaria(("gain", {"k": "1"}),
                                  {"num": "1", "den": "s-2.5"}),
    },
]
