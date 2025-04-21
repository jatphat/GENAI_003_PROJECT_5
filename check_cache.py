import sqlite3
import json
import os
from datetime import datetime, timedelta

conn = sqlite3.connect("llm_cache.db")
cursor = conn.cursor()

# Create table if not exists
cursor.execute("""
CREATE TABLE IF NOT EXISTS cache (
    name TEXT PRIMARY KEY,
    tos_url TEXT
    tos_txt TEXT
)
""")
conn.commit()

def analyze_name(name, llm_function):
    cursor.execute("SELECT response FROM cache WHERE name = ?", (name,))
    row = cursor.fetchone()
    if row:
        print("Using cached result.")
        return json.loads(row[0])  # assuming the response is JSON-serializable

    # Call LLM if not cached
    result = llm_function(name)
    cursor.execute("INSERT INTO cache (name, response) VALUES (?, ?)", (name, json.dumps(result)))
    conn.commit()

    return result

# ---------- Caching Tool ----------
class LLMCacheTool:
    def __init__(self, cache_file="llm_cache.json"):
        self.cache_file = cache_file
        if os.path.exists(cache_file):
            with open(cache_file, "r") as f:
                self.cache = json.load(f)
        else:
            self.cache = []

    def get(self, tos_url):
        for entry in self.cache:
            if entry["tos_url"] == tos_url:
                timestamp_str = entry.get("timestamp")
                if isinstance(timestamp_str, str):
                    try:
                        timestamp = datetime.fromisoformat(timestamp_str)
                        if datetime.now() - timestamp <= timedelta(days=7):
                            return entry
                        else:
                            print("⚠️ Cache found but is older than 7 days. Retrieving latest ToS from the website...")
                    except ValueError:
                        print("⚠️ Invalid timestamp format in cache. Ignoring cache.")
                else:
                    print("⚠️ Missing or non-string timestamp. Ignoring cache.")
                return None
        return None

    def add(self, tos_url, tos_txt, llm_summary):
        new_entry = {
            "tos_url": tos_url,
            "tos_txt": tos_txt,
            "llm_summary": llm_summary,
            "timestamp": datetime.now().isoformat()
        }

        # Remove existing entry for the same URL (if any)
        self.cache = [entry for entry in self.cache if entry["tos_url"] != tos_url]
        self.cache.append(new_entry)
        self._save()

    def _save(self):
        with open(self.cache_file, "w") as f:
            json.dump(self.cache, f, indent=2)