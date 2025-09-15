import base64
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import webview


def ensure_streamlit_running(app_path: str) -> str:
    # 固定端口，避免多次变化
    port = os.environ.get("STREAMLIT_SERVER_PORT", "8501")
    env = os.environ.copy()
    env.setdefault("BROWSER", "none")
    env.setdefault("STREAMLIT_BROWSER_GATHER_USAGE_STATS", "false")
    env.setdefault("STREAMLIT_SERVER_HEADLESS", "true")
    env.setdefault("STREAMLIT_SERVER_PORT", port)

    # 后台启动
    proc = subprocess.Popen([sys.executable, "-m", "streamlit", "run", app_path], env=env)

    # 等待服务就绪
    url = f"http://localhost:{port}"
    for _ in range(60):
        try:
            import urllib.request

            with urllib.request.urlopen(url, timeout=0.5) as _resp:
                return url
        except Exception:
            time.sleep(0.5)
    return url


def main() -> None:
    here = Path(__file__).parent
    app_path = str(here / "app.py")
    if not Path(app_path).exists():
        # 兼容 PyInstaller 下 _MEIPASS
        app_path = str(Path(getattr(sys, "_MEIPASS", here)) / "app.py")
    url = ensure_streamlit_running(app_path)

    # 打开原生窗口
    webview.create_window("专家抽样助手", url, width=1200, height=800)
    webview.start()


if __name__ == "__main__":
    main()
