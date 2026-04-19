## 2025-02-28 - orjson as json replacement

**Learning:** Replace `json` parsing with `orjson` or `ujson` can yield substantial performance improvements, however they are often not necessary or even harmful when object schemas are not very large or when using Python's built-in `json` module is acceptable.

**Action:** Re-evaluate JSON parsing optimization. Look at task queues for `list_tasks` performance.

## 2025-02-28 - json optimization using orjson

**Learning:** Replacing json with orjson reduces `json.loads` time, but the overall speedup in `list_tasks` is marginal (~40ms for 1000 tasks * 10 iterations).
Considering the size of the messages (small task structs), the overhead of Redis fetching (`hgetall`) dominates. Not a huge win.

**Action:** Look for other low hanging fruits.

## 2025-02-28 - SubAgent `listen_for_tasks` polling sleep

**Learning:** The `listen_for_tasks` polling loop has an explicit `await asyncio.sleep(0.1)` when no message is found. Because `pubsub.get_message` sets `timeout=1`, this means it spends time awaiting blocking reads on a thread, and if no message is found *or* a message is processed, it sleeps another 0.1s. Wait, if `pubsub.get_message` fetches a message, it still sleeps `0.1s` at the end of the loop! This adds an artificial 100ms latency to EVERY task processed by a `SubAgent`.

**Action:** Fix the event loop sleep logic in `subagent.py` to only sleep if no message was found, or use a better pubsub listen mechanism that doesn't artificially limit throughput.
