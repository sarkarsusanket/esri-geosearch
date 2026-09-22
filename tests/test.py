"""
Live reload test harness.

1. Run this script ONCE:  python live_reload.py
2. Edit any .py file in src/
3. Type:  reload()
4. Type:  test("your query here")

Weights/parquets/indices stay in memory. Only Python modules get reloaded.
"""
import sys, importlib, time, dotenv

dotenv.load_dotenv()

sys.path.insert(0, "./src")
sys.stdout.reconfigure(line_buffering=True)

MODULE_NAMES = [
    "config", "schema", "models", "query_parser", "artifacts",
    "executor", "queryearth",
    "operations.osm", "operations.vision", "operations.demo",
    "operations.geocode", "operations.tool", "operations.change",
    "operations.threshold", "operations.grounding",
]

# ------------------------------------------------------------------
# 1. First-time heavy load
# ------------------------------------------------------------------
print("=" * 60)
print("Loading heavy assets...")
print("=" * 60)

import queryearth as _qe_mod

qe = _qe_mod.QueryEarth()
qe.initialize()

# Grab references to all loaded modules so we can reload them
_loaded = {}
for name in MODULE_NAMES:
    try:
        _loaded[name] = sys.modules[name]
    except KeyError:
        pass

print("\n" + "=" * 60)
print("READY. Heavy assets are in memory.")
print()
print("  reload()       -> hot-reload all src/ modules")
print('  test("query")  -> run a query')
print("  test(\"query\", \"removed\") -> run change query with mode")
print("=" * 60)


def reload():
    """Reload all src/ modules without touching the loaded assets."""
    reloaded = []
    failed = []

    # Reload dependencies first, then high-level modules
    for name in MODULE_NAMES:
        mod = sys.modules.get(name)
        if mod is None:
            continue
        try:
            importlib.reload(mod)
            reloaded.append(name)
        except Exception as e:
            failed.append((name, str(e)))

    # Re-import updated classes
    import executor as _exec_mod
    import queryearth as _qe_mod

    # Re-instantiate executor with the updated class definition
    qe.executor = _exec_mod.PipelineExecutor(qe.context)

    print(f"Reloaded {len(reloaded)} modules.")
    if failed:
        for name, err in failed:
            print(f"   FAILED: {name} -> {err}")
    return reloaded


def test(query, mode=None):
    """Run a query through the engine."""
    begin = time.time()
    if mode:
        from query_parser import parse_query
        raw_plan = parse_query(query)
        for step in raw_plan.steps:
            if step.operation == "change":
                step.parameters["mode"] = mode
        result_gdf = qe.executor.run_plan(raw_plan)
    else:
        result_gdf = qe.find(query)
    elapsed = time.time() - begin
    print(f"Done in {elapsed:.2f}s. {len(result_gdf)} feature(s) returned.")
    return result_gdf


# Drop into interactive REPL so reload() and test() stay available
import code
code.interact(banner="", exitmsg="", local=dict(globals(), **locals()))
