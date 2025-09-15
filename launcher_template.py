import base64
import os
import sys
import tempfile
from pathlib import Path

# 在构建时由脚本替换为真实Base64内容
APP_CODE_B64 = "__REPLACE_WITH_APP_BASE64__"
CONFIG_TOML_B64 = "__REPLACE_WITH_CONFIG_BASE64__"


def write_app_and_config_to_temp() -> str:
    temp_dir = tempfile.mkdtemp(prefix="expert_sampler_")

    # 写入 app.py
    if APP_CODE_B64 and APP_CODE_B64 != "__REPLACE_WITH_APP_BASE64__":
        code_bytes = base64.b64decode(APP_CODE_B64.encode("utf-8"))
        app_path = os.path.join(temp_dir, "app.py")
        with open(app_path, "wb") as f:
            f.write(code_bytes)
    else:
        here = Path(getattr(sys, "_MEIPASS", Path(__file__).parent))
        candidate = here / "app.py"
        if candidate.exists():
            app_path = str(candidate)
        else:
            raise RuntimeError("app.py 不存在，且未提供内嵌代码。")

    # 写入 .streamlit/config.toml
    try:
        if CONFIG_TOML_B64 and CONFIG_TOML_B64 != "__REPLACE_WITH_CONFIG_BASE64__":
            cfg_dir = os.path.join(temp_dir, ".streamlit")
            os.makedirs(cfg_dir, exist_ok=True)
            cfg_path = os.path.join(cfg_dir, "config.toml")
            cfg_bytes = base64.b64decode(CONFIG_TOML_B64.encode("utf-8"))
            with open(cfg_path, "wb") as f:
                f.write(cfg_bytes)
            os.environ["STREAMLIT_CONFIG"] = cfg_path
    except Exception:
        pass

    return app_path


def main() -> None:
    # 禁用自动打开默认浏览器，避免读取注册表（BROWSER=none 为官方支持方式）
    os.environ.setdefault("BROWSER", "none")
    os.environ.setdefault("STREAMLIT_BROWSER_GATHER_USAGE_STATS", "false")
    os.environ.setdefault("STREAMLIT_SERVER_HEADLESS", "true")

    app_path = write_app_and_config_to_temp()
    try:
        from streamlit.web import bootstrap
        bootstrap.run(app_path, "", [], {})
    except Exception:
        import subprocess
        subprocess.run([sys.executable, "-m", "streamlit", "run", app_path, "--server.headless", "true"], check=True)


if __name__ == "__main__":
    main()
