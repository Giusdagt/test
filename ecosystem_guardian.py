"""ECOSYSTEM GUARDIAN"""

import os
import sys
import time
import asyncio
import threading
import inspect
import logging
import traceback
import importlib
import psutil
from pathlib import Path
from collections import defaultdict, Counter

print("ecosystem_guardian.py caricato ✅")

# === CONFIG ===
LOG_FILE = "ecosystem_diagnostic_report.log"
WATCH_FOLDER = Path("C:/bot")
CHECK_INTERVAL = 10  # secondi
MAX_FILE_SIZE_MB = 100
MAX_LOG_SIZE_MB = 50
CRITICAL_FILES = [
    "processed_data.zstd.parquet",
    "embedding_data.zstd.parquet",
    "ai_memory.parquet",
    "trades.db",
    "trades.parquet"
]
LOCK_KEYWORDS = ["Lock", "Semaphore", "filelock", "asyncio.Lock", "threading.Lock"]
MUTABLE_TYPES = (list, dict, set, bytearray)

# Disattiva tutti i logger esistenti (es. logging.basicConfig in altri moduli)
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

# === LOG SETUP UNICO ===
class MuteOtherLogs(logging.Filter):
    def filter(self, record):
        return record.name == "ecosystem_guardian"

logger = logging.getLogger("ecosystem_guardian")
logger.setLevel(logging.INFO)
logger.handlers.clear()
logger.addFilter(MuteOtherLogs())

log_formatter = logging.Formatter("%(asctime)s - DIAGNOSTIC - %(levelname)s - %(message)s")
log_handler = logging.FileHandler(LOG_FILE, mode="w", encoding="utf-8")
log_handler.setFormatter(log_formatter)
logger.addHandler(log_handler)

stream_handler = logging.StreamHandler(sys.stdout)
stream_handler.setFormatter(log_formatter)
logger.addHandler(stream_handler)

# Disattiva altri log
logging.getLogger().handlers.clear()
sys.stderr = open(os.devnull, 'w')  # Silenzia warning esterni

# === HELPER ===
def log(msg, level="info"):
    getattr(logger, level)(msg)

def is_blocking_io(func):
    try:
        src = inspect.getsource(func)
        return any(x in src for x in ("open(", "read(", "write("))
    except Exception:
        return False

def get_async_functions(module):
    return [name for name, obj in inspect.getmembers(module) if inspect.iscoroutinefunction(obj)]

def get_large_globals(module):
    large = []
    for name, obj in vars(module).items():
        try:
            size = sys.getsizeof(obj)
            if size > 5_000_000:
                large.append((name, size))
        except:
            continue
    return large

def get_mutable_globals(module):
    risky = []
    for name, obj in vars(module).items():
        if isinstance(obj, MUTABLE_TYPES) and not name.startswith("__"):
            risky.append(name)
    return risky

def scan_file_for_risks(filepath):
    risky = []
    try:
        with open(filepath, encoding="utf-8") as f:
            content = f.read()
        if "pl.read_parquet" in content and "async" in content:
            risky.append("⚠️ read_parquet in async context")
        if ".write_parquet" in content and "async" in content:
            risky.append("⚠️ write_parquet in async context")
        if "merge" in content and "parquet" in content:
            risky.append("⚠️ Merge/parquet operation (potenzialmente pericolosa senza lock)")
        if ".write_parquet" in content and not any(lock in content for lock in LOCK_KEYWORDS):
            risky.append("⚠️ Scrittura Parquet senza lock rilevata")
    except Exception as e:
        risky.append(f"Errore lettura: {e}")
    return risky

def check_system_resources():
    process = psutil.Process(os.getpid())
    mem = process.memory_info().rss / 1024 / 1024
    cpu = psutil.cpu_percent(interval=0.5)
    log(f"📊 Memoria: {mem:.2f} MB | CPU: {cpu:.2f}%")
    if mem > 2000:
        log("🚨 RAM sopra soglia critica!", "warning")
    if cpu > 90:
        log("🚨 CPU sopra soglia critica!", "warning")

def module_diagnosis():
    log("🧠 Diagnosi moduli in corso...")
    sys.path.insert(0, str(WATCH_FOLDER))
    loaded_modules = Counter()
    for file in WATCH_FOLDER.glob("*.py"):
        module_name = file.stem
        try:
            module = importlib.import_module(module_name)
            loaded_modules[module_name] += 1
            risky_lines = scan_file_for_risks(file)
            for risk in risky_lines:
                log(f"❗ Rischio in {file.name}: {risk}", "warning")
            large_vars = get_large_globals(module)
            for name, size in large_vars:
                log(f"⚠️ Variabile globale pesante {name} ({size / 1024 / 1024:.2f} MB) in {file.name}", "warning")
            async_funcs = get_async_functions(module)
            for af in async_funcs:
                log(f"📡 Funzione async rilevata: {af} in {file.name}")
            mutables = get_mutable_globals(module)
            for m in mutables:
                log(f"⚠️ Variabile globale mutabile (potenziale race condition): {m} in {file.name}", "warning")
        except Exception as e:
            log(f"Errore import {module_name}: {e}", "error")
    for mod, count in loaded_modules.items():
        if count > 1:
            log(f"⚠️ Modulo {mod} caricato più volte!", "warning")

def file_integrity_check():
    for critical in CRITICAL_FILES:
        path = WATCH_FOLDER / critical
        if not path.exists():
            log(f"❌ File critico mancante: {critical}", "error")
        elif path.stat().st_size == 0:
            log(f"❌ File vuoto: {critical}", "error")
        elif path.stat().st_size > MAX_FILE_SIZE_MB * 1024 * 1024:
            log(f"⚠️ File molto grande: {critical}", "warning")
    for logf in WATCH_FOLDER.glob("*.log"):
        if logf.stat().st_size > MAX_LOG_SIZE_MB * 1024 * 1024:
            log(f"⚠️ File di log molto grande: {logf.name}", "warning")

def thread_and_async_check():
    threads = threading.enumerate()
    log(f"🔎 Thread attivi: {len(threads)}")
    for t in threads:
        if not t.is_alive():
            log(f"⚠️ Thread morto: {t.name}", "warning")
        if "train" in t.name.lower() or "parquet" in t.name.lower():
            log(f"⚠️ Thread critico: {t.name}", "warning")
    try:
        loop = asyncio.get_event_loop()
        tasks = asyncio.all_tasks(loop)
        log(f"🔎 Async tasks attivi: {len(tasks)}")
        for t in tasks:
            if not t.done() and "train" in str(t).lower():
                log(f"⚠️ Task async critico: {t}", "warning")
    except Exception:
        pass

def deadlock_check():
    threads = threading.enumerate()
    blocked = [t for t in threads if hasattr(t, "_is_stopped") and t._is_stopped]
    if len(blocked) > 2:
        log(f"🚨 Possibile deadlock: {len(blocked)} thread bloccati!", "error")

def multi_process_check():
    procs = []
    for p in psutil.process_iter(['pid', 'name', 'cwd']):
        try:
            if p.info['name'] and "python" in p.info['name'].lower() and p.info['cwd'] and str(WATCH_FOLDER) in p.info['cwd']:
                procs.append(p.info['pid'])
        except Exception:
            continue
    if len(procs) > 1:
        log(f"🚨 Più processi Python attivi nella stessa cartella: {procs}", "error")

def log_recent_errors():
    try:
        with open(LOG_FILE, encoding="utf-8") as f:
            lines = f.readlines()[-50:]
        for l in lines:
            if "ERROR" in l or "WARNING" in l:
                log(f"🛑 Log recente: {l.strip()}", "warning")
    except Exception:
        pass

def suggest_actions():
    log("💡 Suggerimento: Usa lock/threading/asyncio per tutte le scritture Parquet!", "info")
    log("💡 Suggerimento: Limita i training paralleli e centralizza la gestione dei file critici.", "info")
    log("💡 Suggerimento: Usa test automatici per simulare crash e recovery.", "info")
    log("💡 Suggerimento: Se usi async, proteggi tutte le sezioni critiche con lock!", "info")

def start_ecosystem_diagnosis():
    log("🛠️ Avvio Ecosystem Diagnostic Guardian...")
    try:
        os.write(1, b"\n Guardian attivo e in ascolto...\n")
    except:
        pass

    while True:
        try:
            check_system_resources()
            module_diagnosis()
            file_integrity_check()
            thread_and_async_check()
            deadlock_check()
            multi_process_check()
            log_recent_errors()
            suggest_actions()
            time.sleep(CHECK_INTERVAL)
        except KeyboardInterrupt:
            log("Interrotto manualmente.", "info")
            break
        except Exception as e:
            log(f"Errore nel loop diagnostico: {traceback.format_exc()}", "error")

def start_guardian_in_background():
    t = threading.Thread(target=start_ecosystem_diagnosis, name="EcosystemGuardian", daemon=True)
    t.start()


if __name__ == "__main__":
    start_ecosystem_diagnosis()
