import datetime
import sys
import threading
import time
from collections import OrderedDict
from typing import Any, Dict, List, Optional

_all_caches: Dict[str, "SmartCache"] = {}
_registry_lock = threading.Lock()


def is_market_hours() -> bool:
    try:
        from zoneinfo import ZoneInfo
        eastern = ZoneInfo("America/New_York")
    except ImportError:
        from datetime import timezone, timedelta
        eastern = timezone(timedelta(hours=-5))
    now = datetime.datetime.now(eastern)
    if now.weekday() >= 5:
        return False
    market_open = now.replace(hour=9, minute=30, second=0, microsecond=0)
    market_close = now.replace(hour=16, minute=0, second=0, microsecond=0)
    return market_open <= now <= market_close


class SmartCache:
    def __init__(self, namespace: str, max_size: int = 500, default_ttl: int = 300):
        self._namespace = namespace
        self._max_size = max_size
        self._default_ttl = default_ttl
        self._store: OrderedDict[str, tuple] = OrderedDict()
        self._lock = threading.Lock()
        self._hits = 0
        self._misses = 0
        with _registry_lock:
            _all_caches[namespace] = self

    def get(self, key: str, ttl: Optional[int] = None) -> Optional[Any]:
        with self._lock:
            entry = self._store.get(key)
            if entry is not None:
                if len(entry) == 3:
                    value, ts, stored_ttl = entry
                    effective_ttl = ttl if ttl is not None else stored_ttl
                else:
                    value, ts = entry
                    effective_ttl = ttl if ttl is not None else self._default_ttl
                if time.time() - ts < effective_ttl:
                    self._store.move_to_end(key)
                    self._hits += 1
                    return value
                else:
                    del self._store[key]
            self._misses += 1
            return None

    def get_adaptive(self, key: str) -> Optional[Any]:
        if is_market_hours():
            ttl = self._default_ttl
        else:
            ttl = self._default_ttl * 6
        return self.get(key, ttl=ttl)

    def set(self, key: str, value: Any, ttl: Optional[int] = None):
        effective_ttl = ttl if ttl is not None else self._default_ttl
        with self._lock:
            if key in self._store:
                del self._store[key]
            self._store[key] = (value, time.time(), effective_ttl)
            while len(self._store) > self._max_size:
                self._store.popitem(last=False)

    def invalidate_pattern(self, pattern: str):
        with self._lock:
            keys_to_delete = [k for k in self._store if pattern in k]
            for k in keys_to_delete:
                del self._store[k]

    def stats(self) -> Dict[str, Any]:
        with self._lock:
            total = self._hits + self._misses
            return {
                "namespace": self._namespace,
                "size": len(self._store),
                "max_size": self._max_size,
                "default_ttl": self._default_ttl,
                "hits": self._hits,
                "misses": self._misses,
                "hit_ratio": round(self._hits / total, 4) if total > 0 else 0,
            }

    def clear(self):
        with self._lock:
            self._store.clear()
            self._hits = 0
            self._misses = 0


class CacheWarmer:
    def __init__(self):
        self._warming = False
        self._last_warm = 0

    def warm(self, tickers: List[str]):
        if self._warming:
            return
        self._warming = True
        try:
            import data.market_data as market_data
            market_data.batch_fetch_quotes(tickers)
        except Exception:
            pass
        finally:
            self._warming = False
            self._last_warm = time.time()


def cache_stats() -> Dict[str, Any]:
    with _registry_lock:
        namespaces = {}
        total_hits = 0
        total_misses = 0
        total_size = 0
        for ns, cache in _all_caches.items():
            s = cache.stats()
            namespaces[ns] = s
            total_hits += s["hits"]
            total_misses += s["misses"]
            total_size += s["size"]
        total_requests = total_hits + total_misses
        return {
            "total_entries": total_size,
            "total_hits": total_hits,
            "total_misses": total_misses,
            "overall_hit_ratio": round(total_hits / total_requests, 4) if total_requests > 0 else 0,
            "namespaces": namespaces,
        }


def get_all_cache_stats() -> Dict[str, Any]:
    with _registry_lock:
        namespaces = {}
        total_entries = 0
        total_memory_estimate = 0
        global_oldest = None
        global_newest = None

        for ns, cache in _all_caches.items():
            s = cache.stats()
            oldest_ts = None
            newest_ts = None
            entry_sizes = []

            with cache._lock:
                for key, entry in cache._store.items():
                    if len(entry) == 3:
                        value, ts, _ = entry
                    else:
                        value, ts = entry
                    entry_sizes.append(sys.getsizeof(value))
                    if oldest_ts is None or ts < oldest_ts:
                        oldest_ts = ts
                    if newest_ts is None or ts > newest_ts:
                        newest_ts = ts

            avg_size = sum(entry_sizes) / len(entry_sizes) if entry_sizes else 0
            ns_memory = int(len(entry_sizes) * avg_size)
            total_memory_estimate += ns_memory
            total_entries += s["size"]

            s["memory_estimate_bytes"] = ns_memory
            s["oldest_entry"] = oldest_ts
            s["newest_entry"] = newest_ts
            namespaces[ns] = s

            if oldest_ts is not None:
                if global_oldest is None or oldest_ts < global_oldest:
                    global_oldest = oldest_ts
            if newest_ts is not None:
                if global_newest is None or newest_ts > global_newest:
                    global_newest = newest_ts

        total_requests = sum(s["hits"] + s["misses"] for s in namespaces.values())
        total_hits = sum(s["hits"] for s in namespaces.values())

        return {
            "total_entries": total_entries,
            "total_memory_estimate_bytes": total_memory_estimate,
            "overall_hit_ratio": round(total_hits / total_requests, 4) if total_requests > 0 else 0,
            "oldest_entry": global_oldest,
            "newest_entry": global_newest,
            "market_hours": is_market_hours(),
            "namespaces": namespaces,
        }
