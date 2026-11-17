import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

RAIZ = Path(__file__).resolve().parent.parent
HARNESS = Path(__file__).resolve().parent / "harness.js"
HTML = RAIZ / "visual_blocks_frontend" / "index.html"
APP = RAIZ / "sintonia_(10122025-teste)app.py"
NODE = shutil.which("node") or (r"D:\node.exe" if Path(r"D:\node.exe").exists() else None)

if str(RAIZ) not in sys.path:
    sys.path.insert(0, str(RAIZ))


def html_final(avancado=True, cenarios=None):
    """HTML que o app realmente entrega a components.html (captura via AppTest)."""
    import streamlit.components.v1 as components
    from streamlit.testing.v1 import AppTest

    import modo_diagrama_blocos

    capturado = []
    orig_html, orig_cen = components.html, modo_diagrama_blocos.CENARIOS
    components.html = lambda html, **kw: capturado.append(html)
    if cenarios is not None:
        modo_diagrama_blocos.CENARIOS = cenarios
    try:
        at = AppTest.from_file(str(APP), default_timeout=60)
        at.session_state["modo_selecionado"] = "canvas"
        at.session_state["avancado"] = avancado
        at.run()
    finally:
        components.html = orig_html
        modo_diagrama_blocos.CENARIOS = orig_cen
    assert not at.exception
    editor = [h for h in capturado if 'id="btnUndo"' in h]
    assert len(editor) == 1
    return editor[0]


_SCRIPTS = {}  # cache do JS por valor de avancado


@pytest.fixture
def run_js(tmp_path):
    """Executa um snippet JS (expressao final = resultado) no script do editor."""
    if NODE is None:
        pytest.skip("Node nao encontrado")

    def _run(snippet, avancado=True):
        if avancado not in _SCRIPTS:
            _SCRIPTS[avancado] = re.search(
                r"<script>(.*)</script>", html_final(avancado), re.S).group(1)
        script = _SCRIPTS[avancado]
        (tmp_path / "editor.js").write_text(script, encoding="utf-8")
        (tmp_path / "snippet.js").write_text(snippet, encoding="utf-8")
        r = subprocess.run([NODE, str(HARNESS), str(tmp_path / "editor.js"),
                            str(tmp_path / "snippet.js")],
                           capture_output=True, text=True, encoding="utf-8", timeout=60)
        assert r.returncode == 0, r.stderr
        return json.loads(r.stdout.strip().splitlines()[-1])

    return _run
