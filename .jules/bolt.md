## 2024-05-15 - Redis Hash Performance Optimization
**Learning:** When retrieving all values from a Redis hash where the keys are not needed (e.g., iterating through all tasks or messages in a queue), using `hgetall` and then calling `.values()` is inefficient. It forces Redis to serialize and transmit all keys over the network, and forces the Python client to construct a dictionary of those keys before discarding them.
**Action:** Use `hvals` instead of `hgetall` when only the values are needed. This significantly reduces network payload size and saves CPU cycles locally by avoiding unnecessary dictionary construction.
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

## 2024-04-13 - Batching Redis Queries with Pipelining and Chunking
**Learning:** Sequential calls to Redis commands like `hgetall` within a loop (N+1 query issue) create excessive network round trips. Using Redis pipelining allows batching these requests into a single round trip, providing dramatic speedups (~99% in benchmarks with 100 agents).
**Action:** Always use Redis pipelines when performing multiple independent read/write operations in a loop. For large datasets, combine pipelining with chunking (e.g., batches of 500) to keep memory usage stable while maintaining high throughput.
