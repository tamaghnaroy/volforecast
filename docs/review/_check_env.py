import os, sys, importlib

key = os.environ.get("OPENAI_API_KEY", "")
print("OPENAI_API_KEY:", ("SET (" + key[:8] + "...)") if key else "NOT SET")

for pkg in ["openai"]:
    try:
        m = importlib.import_module(pkg)
        print(f"{pkg}: {getattr(m, '__version__', 'installed')}")
    except ImportError:
        print(f"{pkg}: NOT INSTALLED")

print("Python:", sys.version)
