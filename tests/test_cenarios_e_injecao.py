"""cenarios.py e a injecao de __AVANCADO__ / __CENARIOS__ no HTML do editor."""
import json

import pytest

from cenarios import CENARIOS
from conftest import html_final

ESC = chr(92)  # barra invertida: "</" vira "<" + ESC + "/"

CAMPOS = ("id", "titulo", "descricao", "o_que_observar", "avancado", "modelo")
IDS = [c["id"] for c in CENARIOS]


def test_ha_oito_cenarios_com_ids_unicos():
    assert len(CENARIOS) == 8
    assert len(set(IDS)) == len(IDS)


@pytest.mark.parametrize("c", CENARIOS, ids=IDS)
class TestCadaCenario:
    def test_campos_obrigatorios(self, c):
        for campo in CAMPOS:
            assert campo in c, campo
        assert isinstance(c["avancado"], bool)
        for campo in ("titulo", "descricao", "o_que_observar"):
            assert c[campo].strip()

    def test_um_input_e_um_output(self, c):
        tipos = [n["type"] for n in c["modelo"]["nodes"]]
        assert tipos.count("input") == 1
        assert tipos.count("output") == 1

    def test_ids_de_no_unicos(self, c):
        ids = [n["id"] for n in c["modelo"]["nodes"]]
        assert len(set(ids)) == len(ids)

    def test_arestas_apontam_para_nos_existentes(self, c):
        ids = {n["id"] for n in c["modelo"]["nodes"]}
        for e in c["modelo"]["edges"]:
            assert e["src"] in ids and e["dst"] in ids

    def test_porta_de_entrada_recebe_no_maximo_uma_aresta(self, c):
        destinos = [(e["dst"], e["dstPort"]) for e in c["modelo"]["edges"]]
        assert len(set(destinos)) == len(destinos)

    def test_ids_de_aresta_unicos(self, c):
        ids = [e["id"] for e in c["modelo"]["edges"]]
        assert len(set(ids)) == len(ids)

    def test_serializa_em_json(self, c):
        assert json.loads(json.dumps(c)) == c


# ---- injecao ----
@pytest.mark.parametrize("avancado", [False, True])
def test_html_final_sem_placeholders(avancado):
    html = html_final(avancado)
    assert "__CENARIOS__" not in html and "__AVANCADO__" not in html
    assert f"var AVANCADO={'true' if avancado else 'false'};" in html


def test_cenarios_injetados_no_html_sao_json_equivalente():
    html = html_final(True)
    linha = next(l for l in html.splitlines() if l.startswith("var CENARIOS="))
    injetado = json.loads(linha[len("var CENARIOS="):].rstrip(";").replace("<" + ESC + "/", "</"))
    assert injetado == CENARIOS


def test_fecha_script_dentro_de_cenario_nao_encerra_a_tag():
    malicioso = [{
        "id": "x", "titulo": "</script><script>alert(1)</script>",
        "descricao": "a </SCRIPT> b", "o_que_observar": "c", "avancado": False,
        "modelo": {"nodes": [], "edges": []},
    }]
    html = html_final(False, cenarios=malicioso)
    # so os dois delimitadores legitimos do editor (CDN do Plotly + script inline)
    assert html.lower().count("</script>") == 2
    assert "alert(1)" in html  # o texto segue la, apenas escapado
    assert "<" + ESC + "/script>" in html
