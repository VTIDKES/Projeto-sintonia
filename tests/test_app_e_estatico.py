"""Fumaca do app Streamlit, config.toml e checagens estaticas do HTML."""
import tomllib

import pytest
from streamlit.testing.v1 import AppTest

from conftest import APP, HTML, RAIZ


def _roda(modo, avancado):
    at = AppTest.from_file(str(APP), default_timeout=60)
    if modo is not None:
        at.session_state["modo_selecionado"] = modo
    at.session_state["avancado"] = avancado
    return at.run()


@pytest.mark.parametrize("avancado", [False, True])
@pytest.mark.parametrize("modo", [None, "canvas", "teoria"])
def test_app_roda_sem_excecao_e_mostra_rodape_v13(modo, avancado):
    at = _roda(modo, avancado)
    assert not at.exception
    legendas = [c.value for c in at.caption]
    assert any("v1.3" in c and "Sistemas de Controle (CTRL)" in c for c in legendas)


@pytest.mark.parametrize("modo,ultimo", [
    (None, "**Sistemas de Controle (CTRL)**"),
    ("canvas", "**Modo Diagrama de Blocos**"),
    ("teoria", "**Teoria**"),
])
def test_breadcrumbs_destacam_a_tela_atual(modo, ultimo):
    at = _roda(modo, False)
    crumbs = [c.value for c in at.caption if c.value.startswith("Início")]
    assert crumbs and "Início → " in crumbs[0] and "Sistemas de Controle (CTRL)" in crumbs[0]
    assert crumbs[0].endswith(ultimo)


def test_modo_canvas_tem_titulo():
    at = _roda("canvas", False)
    assert [t.value for t in at.title] == ["Modo Diagrama de Blocos"]


@pytest.mark.parametrize("avancado,trecho", [(False, "**Básico**"), (True, "**Avançado**")])
def test_modo_canvas_mostra_nivel_atual(avancado, trecho):
    at = _roda("canvas", avancado)
    assert any(c.value.startswith("Nível atual") and trecho in c.value for c in at.caption)


def test_tela_inicial_oferece_modo_diagrama():
    at = _roda(None, False)
    assert any(b.label == "Entrar no Modo Diagrama de Blocos" for b in at.button)


# ---- config.toml ----
def test_config_toml_valido_sem_toolbarmode_duplicado():
    texto = (RAIZ / ".streamlit" / "config.toml").read_text(encoding="utf-8")
    assert texto.count("toolbarMode") == 1
    cfg = tomllib.loads(texto)  # TOML duplicado levantaria erro aqui
    assert cfg["theme"]["primaryColor"] == "#1f77b4"
    assert cfg["client"]["toolbarMode"] == "minimal"


def test_nao_existe_config_toml_na_raiz():
    assert not (RAIZ / "config.toml").exists()


# ---- HTML estatico ----
def _html():
    return HTML.read_text(encoding="utf-8")


def test_viewport_nao_bloqueia_zoom():
    assert "user-scalable" not in _html().lower()


def test_html_sem_marca_de_fix_antiga():
    assert "fix 20260531" not in _html()


@pytest.mark.parametrize("id_,rotulo", [
    ("btnUndo", "Desfazer"), ("btnRedo", "Refazer"),
    ("btnCen", "cenários"), ("btnManReset", "Resetar"),
])
def test_botoes_tem_aria_label(id_, rotulo):
    import re
    tag = re.search(r'<button[^>]*id="%s"[^>]*>' % id_, _html())
    assert tag, id_
    m = re.search(r'aria-label="([^"]+)"', tag.group(0))
    assert m and rotulo.lower() in m.group(1).lower()
