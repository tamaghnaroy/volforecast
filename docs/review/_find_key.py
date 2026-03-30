"""Locate the OpenAI API key used by codex CLI."""
import os, subprocess, sys, winreg

def check_env():
    k = os.environ.get("OPENAI_API_KEY", "")
    print("env OPENAI_API_KEY:", ("SET (" + k[:8] + "...)") if k else "NOT SET")

def check_registry():
    paths = [
        (winreg.HKEY_CURRENT_USER, r"Environment"),
        (winreg.HKEY_LOCAL_MACHINE, r"SYSTEM\CurrentControlSet\Control\Session Manager\Environment"),
    ]
    for hive, path in paths:
        try:
            key = winreg.OpenKey(hive, path)
            try:
                val, _ = winreg.QueryValueEx(key, "OPENAI_API_KEY")
                print(f"registry [{path}] OPENAI_API_KEY: SET ({str(val)[:8]}...)")
            except FileNotFoundError:
                print(f"registry [{path}] OPENAI_API_KEY: NOT SET")
            winreg.CloseKey(key)
        except Exception as e:
            print(f"registry [{path}] error: {e}")

def check_codex_config():
    candidates = [
        os.path.expandvars(r"%APPDATA%\.codex\config.toml"),
        os.path.expandvars(r"%USERPROFILE%\.codex\config.toml"),
        os.path.expandvars(r"%USERPROFILE%\.codex\env"),
        os.path.expandvars(r"%APPDATA%\codex\config.toml"),
        # npm codex package config
        os.path.expandvars(r"%APPDATA%\npm\node_modules\@openai\codex\.env"),
        # XDG config
        os.path.expandvars(r"%LOCALAPPDATA%\codex\config.toml"),
    ]
    for p in candidates:
        if os.path.exists(p):
            print(f"FOUND config: {p}")
            with open(p, "r", errors="ignore") as f:
                content = f.read()
            if "OPENAI_API_KEY" in content or "api_key" in content.lower():
                print("  -> Contains API key reference")
                # print first 200 chars for inspection
                lines = [l for l in content.splitlines() if "key" in l.lower() or "api" in l.lower()]
                for l in lines[:5]:
                    if len(l) > 20:
                        print("  ->", l[:30] + "...")
        else:
            print(f"not found: {p}")

if __name__ == "__main__":
    check_env()
    check_registry()
    check_codex_config()
