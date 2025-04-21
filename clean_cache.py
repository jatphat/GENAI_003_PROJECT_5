from datetime import datetime, timedelta

def add(self, tos_url, tos_txt, llm_summary):
    now = datetime.now()
    self.cache = [
        entry for entry in self.cache
        if entry["tos_url"] != tos_url and
           "timestamp" in entry and
           isinstance(entry["timestamp"], str) and
           (now - datetime.fromisoformat(entry["timestamp"])).days <= 30
    ]

    self.cache.append({
        "tos_url": tos_url,
        "tos_txt": tos_txt,
        "llm_summary": llm_summary,
        "timestamp": now.isoformat()
    })
    self._save()