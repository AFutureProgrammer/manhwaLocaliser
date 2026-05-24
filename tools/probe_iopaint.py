import pathlib
import sys
import pkgutil
import importlib
import traceback

ROOT = pathlib.Path(__file__).resolve().parents[1]
VENV_PY = ROOT / "external" / "iopaint-venv" / "Scripts" / "python.exe"
if VENV_PY.is_file():
    print("Use the venv directly:", VENV_PY)
    print("  ", VENV_PY, "-m iopaint --help")
    sys.exit(0)

p = str(ROOT / "external" / "iopaint-venv" / "Lib" / "site-packages")
print("adding to sys.path:", p)
sys.path.insert(0, p)
try:
    import iopaint
    print('iopaint imported, __file__=', getattr(iopaint, '__file__', None))
    path = getattr(iopaint, '__path__', None)
    print('iopaint.__path__ =', path)
    if path:
        for finder, name, ispkg in pkgutil.iter_modules(path):
            print('module in iopaint:', name, 'ispkg=', ispkg)
            try:
                mod = importlib.import_module('iopaint.' + name)
                attrs = dir(mod)
                if 'app' in attrs:
                    print('  -> has attr app:', getattr(mod, 'app'))
                if 'create_app' in attrs:
                    print('  -> has create_app')
                if any(a in attrs for a in ('start', 'main', 'run_server', 'run')):
                    print('  -> possible entrypoints:', [a for a in ('start','main','run_server','run') if a in attrs])
            except Exception as e:
                print('  -> import failed for iopaint.' + name + ':', e)
                traceback.print_exc()
except Exception as e:
    print('failed to import iopaint:', e)
    traceback.print_exc()

# Also try some common module names
candidates = ['server','app','main','cli']
for c in candidates:
    try:
        full = 'iopaint.' + c
        mod = importlib.import_module(full)
        print('imported', full, '->', mod)
        print('dir:', [x for x in dir(mod) if x.lower().startswith('app') or x.lower().startswith('start') or x.lower().startswith('run')])
    except Exception as e:
        print('cannot import', full, '-', e)
        #traceback.print_exc()

print('\nprobe complete')