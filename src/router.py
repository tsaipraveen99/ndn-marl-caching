import heapq
import queue
import threading
import logging
import random
import networkx as nx
from typing import Dict, List, Any, Optional, Set, Tuple, Callable
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import time
import numpy as np
import warnings
import os
from pathlib import Path
from utils import PIT, PrioritizedItem, ContentStore
from packet import Data, Interest
from policies import should_cache_content, record_cache_result

warnings.filterwarnings("ignore")

logger = logging.getLogger('router_logger')
# Control verbosity: QUIET mode suppresses routine warnings
QUIET_MODE = os.environ.get("NDN_SIM_QUIET", "1") == "1"  # Default to quiet
if QUIET_MODE:
    logger.setLevel(logging.WARNING)  # Only show warnings and errors
else:
    logger.setLevel(logging.INFO)  # Show all info messages
TRACE_STATE = os.environ.get("NDN_SIM_TRACE_STATE", "0") == "1"
TRACE_SAMPLE_LIMIT = int(os.environ.get("NDN_SIM_TRACE_SAMPLE", "5"))
try:
    TRACE_MAX_EVENTS = int(os.environ.get("NDN_SIM_TRACE_MAX_EVENTS", "1000"))
    if TRACE_MAX_EVENTS < 0:
        TRACE_MAX_EVENTS = -1
except ValueError:
    logger.warning(
        "NDN_SIM_TRACE_MAX_EVENTS is not a valid integer; disabling trace limit"
    )
    TRACE_MAX_EVENTS = -1
TRACE_FILE = os.environ.get("NDN_SIM_TRACE_FILE", "logs/trace.log")
TRACE_GLOBAL_MAX_EVENTS = None
if TRACE_STATE:
    try:
        TRACE_GLOBAL_MAX_EVENTS = int(os.environ.get("NDN_SIM_TRACE_TOTAL_EVENTS", "-1"))
    except (TypeError, ValueError):
        TRACE_GLOBAL_MAX_EVENTS = -1
    if TRACE_GLOBAL_MAX_EVENTS < 0:
        TRACE_GLOBAL_MAX_EVENTS = -1
    globals()["_TRACE_GLOBAL_COUNT"] = 0

if TRACE_STATE:
    try:
        Path(TRACE_FILE).parent.mkdir(parents=True, exist_ok=True)
    except Exception as exc:
        logger.warning(f"Unable to create directories for {TRACE_FILE}: {exc}")

trace_logger = logging.getLogger('router_trace_logger')
if TRACE_STATE:
    if not trace_logger.handlers:
        handler = logging.FileHandler(TRACE_FILE, mode='w')
        handler.setFormatter(logging.Formatter('%(asctime)s %(message)s'))
        trace_logger.addHandler(handler)
    trace_logger.setLevel(logging.INFO)
    trace_logger.propagate = False
else:
    trace_logger.setLevel(logging.CRITICAL)


def _summarize_mapping(mapping: Dict[str, List[int]], limit: int = TRACE_SAMPLE_LIMIT) -> str:
    items = list(mapping.items())
    if not items:
        return "[]"
    display = []
    for key, value in items[:limit]:
        display.append(f"{key}->{value}")
    if len(items) > limit:
        display.append(f"... (+{len(items) - limit} more)")
    return "[" + ", ".join(display) + "]"


def _summarize_sequence(sequence: List[str], limit: int = TRACE_SAMPLE_LIMIT) -> str:
    if not sequence:
        return "[]"
    display = sequence[:limit]
    suffix = ""
    if len(sequence) > limit:
        suffix = f", ... (+{len(sequence) - limit} more)"
    return "[" + ", ".join(display) + suffix + "]"

class RouterStats:
    def __init__(self):
        self.lock = threading.Lock()
        self.stats = defaultdict(int)
        self.timing_stats = defaultdict(list)
        # Task 1.4: Add cache insertion tracking
        self.cache_insertion_attempts = 0
        self.cache_insertion_successes = 0
        self.cache_insertion_failures = 0
        
    def increment(self, stat_name: str, value: int = 1):
        with self.lock:
            self.stats[stat_name] += value
            
    def add_timing(self, operation: str, duration: float):
        with self.lock:
            self.timing_stats[operation].append(duration)
    
    def track_cache_insertion(self, success: bool):
        """Task 1.4: Track cache insertion attempts"""
        with self.lock:
            self.cache_insertion_attempts += 1
            if success:
                self.cache_insertion_successes += 1
            else:
                self.cache_insertion_failures += 1
            
    def get_stats(self) -> Dict:
        with self.lock:
            avg_timings = {
                f"avg_{op}_time": np.mean(times) if times else 0
                for op, times in self.timing_stats.items()
            }
            return {**dict(self.stats), **avg_timings}
            
class SimulationStats:
    def __init__(self):
        self.lock = threading.Lock()
        self.nodes_traversed = 0
        self.cache_hits = 0
        self.data_packets_transferred = 0
        self.total_data_size_transferred = 0
        
    def update(self, nodes: int = 0, hits: int = 0, packets: int = 0, size: int = 0):
        with self.lock:
            self.nodes_traversed += nodes
            self.cache_hits += hits
            self.data_packets_transferred += packets
            self.total_data_size_transferred += size

stats = SimulationStats()

class RouterRuntime:
    """Shared runtime that schedules router message processing on a worker pool."""

    def __init__(self, max_workers: int = 8):
        self.queue: "queue.PriorityQueue[PrioritizedItem]" = queue.PriorityQueue()
        self.routers: Dict[int, "Router"] = {}
        self.stop_event = threading.Event()
        self.workers: List[threading.Thread] = []
        self.max_workers = max(1, max_workers)
        # Track worker activity for diagnostics
        self.worker_processed_count = [0] * self.max_workers
        self.worker_lock = threading.Lock()
        # Enhanced metrics collection (low-risk improvement)
        self.metrics = {
            'message_times': defaultdict(list),  # Track per-message-type processing times
            'timeout_count': defaultdict(int),  # Track timeout frequency by message type
            'queue_size_history': [],  # Track queue size over time
            'total_messages_processed': 0,
        }
        self.metrics_lock = threading.Lock()
        # Training phase configuration (ensures DQN training finishes each round)
        self.training_phase_timeout = float(os.environ.get("NDN_SIM_TRAINING_PHASE_TIMEOUT", "60.0"))
        self.training_phase_min_time = float(os.environ.get("NDN_SIM_TRAINING_PHASE_MIN", "30.0"))
        self.training_time_per_router = float(os.environ.get("NDN_SIM_TRAINING_TIME_PER_ROUTER", "5.0"))
        self.training_safety_margin = float(os.environ.get("NDN_SIM_TRAINING_SAFETY_MARGIN", "1.5"))
        # PERMANENT FIX: Process messages directly in worker threads
        # No nested threading - worker threads are already limited, so slow messages
        # won't block other workers. This eliminates the ThreadPoolExecutor bottleneck
        # that was causing all 64 workers to block waiting for 4 executor threads.
        for idx in range(self.max_workers):
            worker = threading.Thread(
                target=self._worker_loop,
                name=f"router-worker-{idx}",
                daemon=True,
                args=(idx,),  # Pass worker index
            )
            worker.start()
            self.workers.append(worker)

    def register_router(self, router: "Router"):
        self.routers[router.router_id] = router

    def deregister_router(self, router_id: int):
        self.routers.pop(router_id, None)

    def enqueue(
        self,
        target_router_id: int,
        priority: int,
        message_type: str,
        payload: Any,
    ):
        if self.stop_event.is_set():
            return
        
        # FIB UPDATE RATE LIMITING: Throttle FIB updates to prevent queue flooding
        if message_type == 'fib_update':
            MAX_FIB_UPDATES_PER_SECOND = int(os.environ.get('NDN_SIM_MAX_FIB_RATE', '50'))
            current_time = time.time()
            
            # Count FIB updates in the last second
            with self.metrics_lock:
                # Clean old timestamps (older than 1 second)
                self.metrics.setdefault('fib_update_times', [])
                self.metrics['fib_update_times'] = [
                    t for t in self.metrics['fib_update_times'] 
                    if current_time - t < 1.0
                ]
                
                # If we're exceeding the rate limit, drop this FIB update
                if len(self.metrics['fib_update_times']) >= MAX_FIB_UPDATES_PER_SECOND:
                    logger.debug(f"Dropping FIB update due to rate limit ({len(self.metrics['fib_update_times'])}/{MAX_FIB_UPDATES_PER_SECOND} per second)")
                    return
                
                # Record this FIB update
                self.metrics['fib_update_times'].append(current_time)
        
        # BACKPRESSURE MECHANISM (low-risk improvement): Prevent queue flooding
        # If queue is too full, wait briefly to let workers catch up
        MAX_QUEUE_SIZE = int(os.environ.get('NDN_SIM_MAX_QUEUE_SIZE', '10000'))
        max_wait_iterations = 100  # Maximum 1 second of backpressure (100 * 0.01s)
        iterations = 0
        
        while iterations < max_wait_iterations:
            queue_size = self._estimate_queue_size()
            if queue_size <= MAX_QUEUE_SIZE:
                break
            # Small delay to let workers catch up
            time.sleep(0.01)
            iterations += 1
            if self.stop_event.is_set():
                return
        
        # Log backpressure if it occurred
        if iterations > 0 and not QUIET_MODE:
            logger.debug(f"Backpressure applied: waited {iterations * 0.01:.2f}s for queue to drain (size was {queue_size})")
        
        self.queue.put(PrioritizedItem(priority, (target_router_id, message_type, payload)))

    def _worker_loop(self, worker_idx: int = 0):
        import time
        messages_processed = 0
        last_log_time = time.time()
        
        while not self.stop_event.is_set():
            try:
                prioritized_item = self.queue.get(timeout=0.5)
            except queue.Empty:
                continue

            try:
                target_router_id, message_type, payload = prioritized_item.item
            except Exception:
                self.queue.task_done()
                continue

            if target_router_id is None:
                self.queue.task_done()
                continue

            router = self.routers.get(target_router_id)
            if router is not None:
                try:
                    # Process message with timeout protection and detailed logging
                    # Allow configuration via environment variable (useful for Colab/GPU environments)
                    timeout_seconds = float(os.environ.get('NDN_SIM_WORKER_TIMEOUT', '10.0'))
                    start_time = time.time()
                    
                    # DEBUG: Only log interest/data messages (bloom_filter_update is too verbose)
                    # Respect QUIET_MODE to reduce output
                    if not QUIET_MODE:
                        if message_type in ['interest', 'data']:
                            print(f"   🔍 Worker {worker_idx}: Processing {message_type} to router {target_router_id} (msg #{messages_processed + 1})", flush=True)
                        elif messages_processed % 1000 == 0:
                            print(f"   🔍 Worker {worker_idx}: Processing {message_type} to router {target_router_id} (msg #{messages_processed + 1})", flush=True)
                            last_log_time = time.time()
                    
                    # PERMANENT FIX: Process messages directly in worker thread
                    # No nested threading - eliminates ThreadPoolExecutor bottleneck
                    # Worker threads are already limited, so slow messages won't block other workers
                    # This is the simplest and most efficient approach
                    try:
                        if not QUIET_MODE and message_type in ['interest', 'data']:
                            print(f"   🔍 Worker {worker_idx}: Processing {message_type} to router {target_router_id}", flush=True)
                        router.dispatch_message(message_type, payload)
                        if not QUIET_MODE and message_type in ['interest', 'data']:
                            print(f"   ✅ Worker {worker_idx}: Completed {message_type} to router {target_router_id}", flush=True)
                    except Exception as e:
                        # Log the exception but continue processing
                        logger.error(
                            f"Worker {worker_idx}: Error processing {message_type} to router {target_router_id}: {e}",
                            exc_info=False
                        )
                        # Track error in metrics
                        with self.metrics_lock:
                            self.metrics['timeout_count'][message_type] += 1
                        # Continue to next message - don't let one error stop the worker
                    
                    # Track successful processing
                    elapsed = time.time() - start_time
                    messages_processed += 1
                    
                    # Enhanced metrics collection (low-risk improvement)
                    with self.metrics_lock:
                        # Track processing time per message type
                        self.metrics['message_times'][message_type].append(elapsed)
                        # Keep only recent 1000 samples per message type to prevent memory bloat
                        if len(self.metrics['message_times'][message_type]) > 1000:
                            self.metrics['message_times'][message_type] = \
                                self.metrics['message_times'][message_type][-1000:]
                        self.metrics['total_messages_processed'] += 1
                    
                    if elapsed > timeout_seconds * 0.8:  # Warn if close to timeout
                        logger.warning(
                            f"Worker {worker_idx}: Message processing took {elapsed:.2f}s "
                            f"(close to {timeout_seconds}s timeout) for {message_type} to router {target_router_id}"
                        )
                    
                    with self.worker_lock:
                        if worker_idx < len(self.worker_processed_count):
                            self.worker_processed_count[worker_idx] += 1
                except Exception as exc:
                    logger.error(
                        f"Worker {worker_idx}: Error dispatching {message_type} to router {target_router_id}: {exc}",
                        exc_info=True,
                    )
                    # Still mark as processed to prevent queue blocking
                    messages_processed += 1
                    with self.worker_lock:
                        if worker_idx < len(self.worker_processed_count):
                            self.worker_processed_count[worker_idx] += 1
            else:
                # Router not found - log for debugging
                if messages_processed % 1000 == 0:
                    logger.warning(
                        f"Worker {worker_idx}: Router {target_router_id} not found for {message_type}"
                    )

            self.queue.task_done()

    def wait_for_queue_drain(self, timeout: float = 120.0, logger=None) -> bool:
        """
        Wait for the message queue to drain with timeout.
        
        This method waits for all messages currently in the queue to be processed.
        It's used to ensure that all messages from a simulation round are processed
        before moving to the next round.
        
        Args:
            timeout: Maximum time to wait in seconds (default: 30.0)
            logger: Optional logger for debug messages
            
        Returns:
            True if queue drained successfully, False if timeout reached
        """
        start = time.time()
        check_count = 0
        last_queue_size = None
        last_check_time = start
        
        # Get initial queue size (approximate, not thread-safe but gives us an idea)
        try:
            # PriorityQueue doesn't have qsize() reliably, so we estimate
            initial_size_estimate = self._estimate_queue_size()
        except:
            initial_size_estimate = "unknown"
        
        # Check worker status and activity
        alive_workers = sum(1 for w in self.workers if w.is_alive())
        with self.worker_lock:
            total_processed = sum(self.worker_processed_count)
            worker_stats = f"processed={total_processed}"
        
        if logger is None:
            print(f"   📊 Queue drain: initial_size≈{initial_size_estimate}, workers_alive={alive_workers}/{self.max_workers}, {worker_stats}", flush=True)
        elif logger:
            logger.info(f"Queue drain start: initial_size≈{initial_size_estimate}, "
                        f"workers_alive={alive_workers}/{self.max_workers}, {worker_stats}")
        
        # Use queue.join() with timeout - this is the proper way to wait for all tasks
        # But we need to implement a timeout wrapper since queue.join() doesn't support timeout
        drain_complete = threading.Event()
        
        def wait_for_join():
            try:
                # Wait for all tasks to complete (queue.join() blocks until all task_done() called)
                # We'll use a polling approach with queue.empty() and queue.join() check
                while True:
                    # Check if queue is empty AND all tasks are done
                    if self.queue.empty():
                        # Double-check: wait a bit and check again to ensure no race condition
                        time.sleep(0.1)
                        if self.queue.empty():
                            drain_complete.set()
                            return
                    time.sleep(0.1)
            except Exception as e:
                if logger:
                    logger.error(f"Error in queue drain wait: {e}")
                drain_complete.set()
        
        # Start the drain check in a separate thread
        drain_thread = threading.Thread(target=wait_for_join, daemon=True)
        drain_thread.start()
        
        # Poll with timeout
        while not drain_complete.is_set():
            elapsed = time.time() - start
            check_count += 1
            
            # WORKAROUND: Force-drain stuck queue after timeout
            # If workers are stuck and not processing, clear remaining messages
            if elapsed > timeout:
                # Get final diagnostics
                final_size_estimate = self._estimate_queue_size()
                alive_workers = sum(1 for w in self.workers if w.is_alive())
                with self.worker_lock:
                    total_processed = sum(self.worker_processed_count)
                    worker_stats = f"total_processed={total_processed}"
                
                # WORKAROUND: If workers are stuck, force-drain the queue
                # This allows the simulation to continue even if some messages are stuck
                # OPTIMIZATION: Trigger earlier (50% instead of 80%) to clear stuck messages faster
                force_drain_threshold = float(os.environ.get('NDN_SIM_FORCE_DRAIN_THRESHOLD', '0.5'))
                if final_size_estimate > 0 and elapsed > timeout * force_drain_threshold:  # If stuck for threshold% of timeout
                    stuck_count = 0
                    # Clear stuck messages from queue (up to a limit to prevent infinite loop)
                    max_clear = min(1000, final_size_estimate)  # Clear max 1000 messages
                    for _ in range(max_clear):
                        try:
                            if self.queue.empty():
                                break
                            item = self.queue.get_nowait()
                            self.queue.task_done()  # Mark as done to unblock
                            stuck_count += 1
                        except queue.Empty:
                            break
                    
                    if stuck_count > 0:
                        if logger is None:
                            print(f"   🔧 WORKAROUND: Force-drained {stuck_count} stuck messages to unblock simulation", flush=True)
                        elif logger:
                            logger.warning(f"Force-drained {stuck_count} stuck messages")
                
                if logger:
                    logger.warning(f"RouterRuntime: Queue not drained after {timeout}s timeout. "
                                 f"Queue size≈{final_size_estimate}, workers_alive={alive_workers}/{self.max_workers}, {worker_stats}")
                if logger is None:
                    print(f"   ⚠️  Queue drain: TIMEOUT after {timeout}s (queue≈{final_size_estimate}, workers={alive_workers}/{self.max_workers}, {worker_stats})", flush=True)
                return False
            
            # Get worker activity stats for force-drain check
            with self.worker_lock:
                total_processed = sum(self.worker_processed_count)
            
            # OPTIMIZATION: Check for force-drain during waiting loop (not just at timeout)
            # This allows stuck messages to be cleared earlier, keeping simulation moving
            force_drain_threshold = float(os.environ.get('NDN_SIM_FORCE_DRAIN_THRESHOLD', '0.5'))
            if elapsed > timeout * force_drain_threshold:
                current_size = self._estimate_queue_size()
                # Check if workers are stuck (no progress and queue not empty)
                if current_size > 0 and total_processed > 0:
                    # Check if we've made progress since last check
                    if not hasattr(self, '_last_force_drain_processed'):
                        self._last_force_drain_processed = total_processed
                        self._last_force_drain_time = elapsed
                    
                    # If no progress in last 5 seconds, force-drain
                    if elapsed - self._last_force_drain_time > 5.0 and total_processed == self._last_force_drain_processed:
                        stuck_count = 0
                        max_clear = min(1000, current_size)  # Clear max 1000 messages
                        for _ in range(max_clear):
                            try:
                                if self.queue.empty():
                                    break
                                item = self.queue.get_nowait()
                                self.queue.task_done()  # Mark as done to unblock
                                stuck_count += 1
                            except queue.Empty:
                                break
                        
                        if stuck_count > 0:
                            if logger is None:
                                print(f"   🔧 WORKAROUND: Force-drained {stuck_count} stuck messages to unblock simulation (at {elapsed:.1f}s)", flush=True)
                            elif logger:
                                logger.warning(f"Force-drained {stuck_count} stuck messages at {elapsed:.1f}s")
                            # Reset tracking to avoid immediate re-triggering
                            self._last_force_drain_processed = total_processed
                            self._last_force_drain_time = elapsed
                    elif total_processed > self._last_force_drain_processed:
                        # Progress made - update tracking
                        self._last_force_drain_processed = total_processed
                        self._last_force_drain_time = elapsed
            
            # Diagnostic: Report progress every 2 seconds
            if check_count % 20 == 0:  # Every 2 seconds (20 * 0.1s)
                remaining = timeout - elapsed
                current_size = self._estimate_queue_size()
                alive_workers = sum(1 for w in self.workers if w.is_alive())
                
                # Queue size monitoring (low-risk improvement)
                with self.metrics_lock:
                    self.metrics['queue_size_history'].append({
                        'time': time.time(),
                        'size': current_size,
                        'elapsed': elapsed
                    })
                    # Keep only recent 1000 samples to prevent memory bloat
                    if len(self.metrics['queue_size_history']) > 1000:
                        self.metrics['queue_size_history'] = \
                            self.metrics['queue_size_history'][-1000:]
                
                # Get worker activity stats
                with self.worker_lock:
                    total_processed = sum(self.worker_processed_count)
                    worker_stats = f"processed={total_processed}"
                
                # Calculate processing rate if we have previous measurement
                if last_queue_size is not None and elapsed > last_check_time:
                    time_diff = elapsed - last_check_time
                    if time_diff > 0:
                        # Estimate rate based on queue size change
                        if last_queue_size >= 0 and current_size >= 0:
                            queue_change = last_queue_size - current_size
                            rate = queue_change / time_diff
                            rate_str = f", rate≈{rate:.1f} msgs/s"
                            
                            # Also calculate rate from processed count
                            with self.worker_lock:
                                prev_processed = getattr(self, '_last_processed_count', total_processed)
                                processed_diff = total_processed - prev_processed
                                processed_rate = processed_diff / time_diff if time_diff > 0 else 0.0
                                self._last_processed_count = total_processed
                            
                            if processed_rate > 0:
                                rate_str += f", processed_rate={processed_rate:.1f} msgs/s"
                            elif processed_rate == 0 and total_processed > 0 and elapsed > 5.0:
                                # Only show "STUCK" warning if we've been waiting > 5 seconds with no progress
                                rate_str += " ⚠️ WORKERS STUCK"
                                # Log detailed worker status for debugging
                                if logger is None:
                                    print(f"   🔴 Worker status: {alive_workers}/{self.max_workers} alive, "
                                          f"processed={total_processed}, queue≈{current_size}", flush=True)
                        else:
                            rate_str = ""
                    else:
                        rate_str = ""
                else:
                    rate_str = ""
                
                last_queue_size = current_size
                last_check_time = elapsed
                
                if logger is None:
                    print(f"   ⏳ Queue drain: still waiting... ({elapsed:.1f}s elapsed, {remaining:.1f}s remaining, "
                          f"queue≈{current_size}, workers={alive_workers}/{self.max_workers}, {worker_stats}{rate_str})", flush=True)
                elif logger:
                    logger.debug(f"Queue drain: {elapsed:.1f}s elapsed, queue≈{current_size}, workers={alive_workers}/{self.max_workers}, {worker_stats}")
            
            time.sleep(0.1)
        
        elapsed = time.time() - start
        if logger is None:
            print(f"   ✅ Queue drained in {elapsed:.2f}s (checked {check_count} times)", flush=True)
        elif logger:
            logger.info(f"Queue drained in {elapsed:.2f}s (checks={check_count})")
        
        # Additional small wait to ensure all dispatched messages are fully processed
        time.sleep(0.2)
        self._run_training_phase(logger)
        return True

    def _run_training_phase(self, logger=None):
        """Run deferred DQN training requests with a dedicated time budget."""
        try:
            training_manager = DQNTrainingManager.get_instance()
        except Exception as exc:
            if logger:
                logger.debug(f"Training manager unavailable during training phase: {exc}")
            return
        
        queued_requests = training_manager.get_queued_training_count()
        in_flight = training_manager.get_pending_count()
        total_requests = queued_requests + in_flight
        
        if total_requests <= 0:
            return
        
        effective_requests = max(queued_requests, total_requests)
        budget = self._calculate_training_budget(effective_requests)
        if budget <= 0:
            return
        
        phase_msg = (
            f"   🎓 Training phase: queued={queued_requests}, in_flight={in_flight}, "
            f"budget={budget:.1f}s"
        )
        if logger is None:
            print(phase_msg, flush=True)
        elif logger:
            logger.info(phase_msg.strip())
        
        training_complete = training_manager.process_training_queue(timeout=budget, logger=logger)
        if not training_complete:
            if logger is None:
                print("   ⚠️  Training phase incomplete, proceeding to next round", flush=True)
            elif logger:
                logger.warning("Training phase incomplete, proceeding to next round")

    def _calculate_training_budget(self, pending_requests: int) -> float:
        """Estimate how much time to allocate to the training phase."""
        if pending_requests <= 0:
            return 0.0
        estimated = pending_requests * self.training_time_per_router
        estimated *= self.training_safety_margin
        estimated = max(self.training_phase_min_time, estimated)
        return min(self.training_phase_timeout, estimated)
    
    def _estimate_queue_size(self) -> int:
        """
        Estimate queue size. PriorityQueue has qsize() but it's not thread-safe,
        so this is approximate. Returns -1 if queue is not empty but size unknown.
        """
        try:
            # Try to get size (may be approximate in multi-threaded context)
            size = self.queue.qsize()
            return size
        except AttributeError:
            # Fallback: just check if empty
            try:
                if self.queue.empty():
                    return 0
                else:
                    return -1  # Not empty, but size unknown
            except:
                return -1
        except:
            return -1

    def get_metrics(self) -> Dict[str, Any]:
        """
        Get current metrics for analysis and debugging.
        Returns a dictionary with:
        - message_times: Dict of message type -> list of processing times
        - timeout_count: Dict of message type -> count of timeouts
        - queue_size_history: List of queue size snapshots
        - total_messages_processed: Total count of processed messages
        - average_processing_times: Dict of message type -> average processing time
        - timeout_rate: Dict of message type -> timeout rate (0.0 to 1.0)
        """
        with self.metrics_lock:
            # Calculate derived metrics
            avg_times = {}
            timeout_rates = {}
            
            for msg_type in self.metrics['message_times']:
                times = self.metrics['message_times'][msg_type]
                if times:
                    avg_times[msg_type] = sum(times) / len(times)
                else:
                    avg_times[msg_type] = 0.0
                
                # Calculate timeout rate
                total_processed = len(times) + self.metrics['timeout_count'][msg_type]
                if total_processed > 0:
                    timeout_rates[msg_type] = self.metrics['timeout_count'][msg_type] / total_processed
                else:
                    timeout_rates[msg_type] = 0.0
            
            return {
                'message_times': dict(self.metrics['message_times']),  # Convert defaultdict to dict
                'timeout_count': dict(self.metrics['timeout_count']),  # Convert defaultdict to dict
                'queue_size_history': self.metrics['queue_size_history'].copy(),
                'total_messages_processed': self.metrics['total_messages_processed'],
                'average_processing_times': avg_times,
                'timeout_rate': timeout_rates,
            }

    def shutdown(self):
        try:
            self.queue.join()
        except Exception:
            pass
        self.stop_event.set()
        # Push sentinel messages to unblock workers quickly
        for _ in self.workers:
            self.queue.put(PrioritizedItem(0, (None, "shutdown", None)))
        for worker in self.workers:
            worker.join(timeout=2.0)


class DQNTrainingManager:
    """
    Centralized manager for asynchronous DQN training.
    Prevents blocking message processing workers by executing training in background threads.
    
    Each agent still trains independently on its own experiences - this manager only
    coordinates WHEN training happens, not WHAT is trained.
    """
    _instance = None
    _lock = threading.Lock()
    
    def __init__(self, max_training_workers: int = 4):
        """
        Initialize the training manager.
        
        Args:
            max_training_workers: Number of parallel training threads (typically 2-4 for GPU)
        """
        self.executor: Optional[ThreadPoolExecutor] = None
        self.max_workers = max_training_workers
        self.pending_training = 0  # Track pending training operations
        self.pending_lock = threading.Lock()
        self.is_shutdown = False
        self.training_queue: List[Tuple[int, int, int, int, Callable, int]] = []
        self.queue_lock = threading.Lock()
        self._queue_counter = 0
        self.training_stats = {
            'submitted': 0,
            'completed': 0,
            'timeouts': 0,
        }
        
    @classmethod
    def get_instance(cls, max_workers: int = 4):
        """Singleton pattern - one training manager per simulation"""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls(max_workers)
        return cls._instance
    
    def submit_training(self, training_fn: Callable, router_id: int):
        """
        Submit DQN training task to background thread pool.
        Non-blocking - returns immediately.
        
        Args:
            training_fn: Function that performs training (lambda: agent.replay())
            router_id: Router ID for logging
        """
        if self.is_shutdown:
            return
        
        # Initialize executor on first use
        if self.executor is None:
            self.executor = ThreadPoolExecutor(
                max_workers=self.max_workers,
                thread_name_prefix="dqn-trainer"
            )
        
        # Track pending training
        with self.pending_lock:
            self.pending_training += 1
        
        # Submit to thread pool (non-blocking)
        future = self.executor.submit(self._execute_training, training_fn, router_id)
        
        # Clean up future when done
        future.add_done_callback(self._on_training_complete)
    
    def queue_training_request(
        self,
        router_id: int,
        training_fn: Callable,
        priority: int,
        memory_size: int,
    ) -> bool:
        """
        Queue a training request for deferred execution.
        
        Args:
            router_id: Router issuing the training request
            training_fn: Callable that performs training when executed
            priority: Larger values are processed first
            memory_size: Current replay buffer size (for diagnostics)
            
        Returns:
            True if queued, False otherwise
        """
        if self.is_shutdown:
            return False
        
        with self.queue_lock:
            self._queue_counter += 1
            # Use max-heap via negative priority. Tie-breaker ensures deterministic ordering.
            heapq.heappush(
                self.training_queue,
                (-priority, -memory_size, self._queue_counter, router_id, training_fn, memory_size),
            )
            self.training_stats['submitted'] += 1
            queued = len(self.training_queue)
        
        if not QUIET_MODE:
            print(
                f"   📚 Queued DQN training for router {router_id} "
                f"(priority={priority}, memory={memory_size}, queue={queued})",
                flush=True,
            )
        return True
    
    def _execute_training(self, training_fn: Callable, router_id: int):
        """Execute training function in background thread"""
        try:
            loss = training_fn()
            if loss is not None and loss > 0:
                print(f"✅ DQN Training: Router {router_id} completed, loss={loss:.4f}", flush=True)
                logger.info(f"✅ DQN training completed for router {router_id}, loss={loss:.4f}")
            else:
                logger.debug(f"DQN training skipped for router {router_id} (insufficient memory or timeout)")
        except Exception as e:
            logger.error(f"DQN training error for router {router_id}: {e}", exc_info=True)
        finally:
            with self.pending_lock:
                self.pending_training = max(0, self.pending_training - 1)
    
    def _on_training_complete(self, future):
        """Callback when training completes"""
        try:
            future.result()  # Re-raise any exceptions
        except Exception as e:
            logger.debug(f"Training task completed with error: {e}")
    
    def get_pending_count(self) -> int:
        """Get number of pending training operations"""
        with self.pending_lock:
            return self.pending_training
    
    def get_queued_training_count(self) -> int:
        """Get number of queued (not yet started) training requests."""
        with self.queue_lock:
            return len(self.training_queue)
    
    def process_training_queue(self, timeout: float = 60.0, logger=None) -> bool:
        """
        Process queued training requests and wait for completion.
        
        Args:
            timeout: Maximum time to wait for all queued training to finish
            logger: Optional logger for diagnostics
        """
        if timeout <= 0:
            return False
        
        start_time = time.time()
        queued_batch = []
        with self.queue_lock:
            while self.training_queue:
                queued_batch.append(heapq.heappop(self.training_queue))
        
        if not queued_batch:
            # Nothing queued, just wait for in-flight work
            return self.wait_for_training_complete(timeout=timeout, logger=logger)
        
        batch_size = len(queued_batch)
        if logger is None:
            print(f"   ▶️  Launching {batch_size} deferred DQN training tasks", flush=True)
        elif logger:
            logger.info(f"Launching {batch_size} deferred DQN training tasks")
        
        for _, _, _, router_id, training_fn, _ in queued_batch:
            self.submit_training(training_fn, router_id)
        
        remaining = max(0.0, timeout - (time.time() - start_time))
        completed = self.wait_for_training_complete(timeout=remaining, logger=logger)
        if completed:
            self.training_stats['completed'] += batch_size
        else:
            self.training_stats['timeouts'] += batch_size
        return completed
    
    def wait_for_training_complete(self, timeout: float = 30.0, logger=None) -> bool:
        """
        Wait for all pending training to complete.
        Used during queue drain to ensure training finishes.
        
        Args:
            timeout: Maximum time to wait in seconds
            
        Returns:
            True if all training completed, False if timeout
        """
        start = time.time()
        while self.get_pending_count() > 0:
            if time.time() - start > timeout:
                if logger:
                    logger.warning("Training completion wait timed out")
                return False
            time.sleep(0.1)
        return True
    
    def shutdown(self):
        """Shutdown training manager and wait for completion"""
        self.is_shutdown = True
        if self.executor:
            # IMPROVEMENT: Fix Python version compatibility
            # timeout parameter was added in Python 3.9
            # Use try-except as fallback for compatibility
            try:
                self.executor.shutdown(wait=True, timeout=30.0)
            except TypeError:
                # Fallback for Python < 3.9 or other compatibility issues
                self.executor.shutdown(wait=True)
            self.executor = None
        # Reset instance for next simulation
        with DQNTrainingManager._lock:
            DQNTrainingManager._instance = None


class Router:
    def __init__(
        self,
        router_id: int,
        capacity: int,
        type_: str,
        G=None,
        runtime: Optional[RouterRuntime] = None,
    ):
        self.router_id = router_id
        self.type = type_
        self.capacity = capacity
        self.router_time = 0.0
        self.status = "up"  # Track router status
        
        # Initialize components with thread safety
        self.FIB: Dict[str, List[int]] = {}
        self.FIB_lock = threading.Lock()
        
        self.PIT = PIT()
        self.content_store = ContentStore(capacity, router_id)
        
        # Task 2.4: Set router and graph references in ContentStore for enhanced DQN state
        self.content_store.router_ref = self
        self.content_store.graph_ref = G
        self.placement_policy = os.environ.get("NDN_SIM_PLACEMENT_POLICY", "lce").lower()
        
        # Track neighbors and their status
        self.neighbors = set()  # Start with empty set
        self.neighbor_status = {}  # Start with empty dict
        self.neighbor_lock = threading.Lock()
        
        # Only initialize neighbors if the node exists in the graph and G is provided
        if G is not None and router_id in G:
            self.neighbors = set(G[router_id])
            self.neighbor_status = {n: "up" for n in self.neighbors}
        
        # Statistics tracking
        self.stats = RouterStats()
        self.router_stats = {}  # For compatibility with run.py
        
        # Shared runtime for message scheduling
        self.runtime: Optional[RouterRuntime] = runtime
        if self.runtime is not None:
            self.runtime.register_router(self)
        
        # Tracing helpers
        self.trace_events_emitted = 0
        self.trace_limit_notified = False
        global TRACE_GLOBAL_MAX_EVENTS
        self._global_limit_hit = False

        # Connected users (for user-type routers)
        self.connected_users = []
        
        # For compatibility with run.py
        self.data_store = {}
        # Optional back-reference to Producer when this router represents a producer node
        self.producer = None
        self.data_messages_processed = 0
        
        # FIX #4: Nonce-based loop detection (RFC 8569)
        # Track recently seen (content_name, nonce) pairs to detect loops
        self.seen_nonces: Dict[str, Set[int]] = defaultdict(set)
        self.nonce_lock = threading.Lock()
        
        # FIB update deduplication: Track recently processed FIB updates to prevent duplicates
        self.recent_fib_updates: Set[tuple] = set()  # (content_name, next_hop) tuples
        self.fib_update_lock = threading.Lock()
        self.fib_update_times: List[float] = []  # Track FIB update timestamps for rate limiting
        
        logger.info(f"Router {router_id} initialized with capacity {capacity}")
        
    def add_neighbor(self, neighbor_id: int):
        """Add a neighbor router"""
        with self.neighbor_lock:
            self.neighbors.add(neighbor_id)
            self.neighbor_status[neighbor_id] = "up"
            logger.debug(f"Router {self.router_id}: Added neighbor {neighbor_id}")
    
    def set_runtime(self, runtime: RouterRuntime):
        """Attach this router to a runtime (useful for late registration)."""
        if self.runtime is runtime:
            return
        if self.runtime is not None:
            self.runtime.deregister_router(self.router_id)
        self.runtime = runtime
        if self.runtime is not None:
            self.runtime.register_router(self)

    def submit_message(self, message_type: str, payload: Any, priority: int = 1):
        """Schedule a message for this router."""
        if not self.runtime:
            raise RuntimeError(f"Router {self.router_id}: No runtime attached for message submission")
        self.runtime.enqueue(self.router_id, priority, message_type, payload)

    def send_message(self, target_router_id: int, message_type: str, payload: Any, priority: int = 1):
        """Send a message to another router via the runtime."""
        if not self.runtime:
            raise RuntimeError(f"Router {self.router_id}: No runtime attached for message submission")
        self.runtime.enqueue(target_router_id, priority, message_type, payload)

    def dispatch_message(self, message_type: str, payload: Any):
        """Dispatch an incoming message from the runtime."""
        import time
        dispatch_start = time.time()
        
        # DEBUG: Only log interest/data messages (respect QUIET_MODE)
        if not QUIET_MODE and message_type in ['interest', 'data']:
            print(f"      📍 Router {self.router_id}: dispatch_message({message_type}) START", flush=True)
        
        self.router_time = time.time()
        if message_type == 'interest':
            graph, interest_packet, prev_node = payload
            if not QUIET_MODE:
                print(f"      📍 Router {self.router_id}: About to call handle_interest() for {interest_packet.name}", flush=True)
            self.handle_interest(graph, interest_packet, prev_node)
            if not QUIET_MODE:
                print(f"      ✅ Router {self.router_id}: handle_interest() completed for {interest_packet.name}", flush=True)
        elif message_type == 'data':
            self.data_messages_processed += 1
            graph, data_packet, prev_node = payload
            if not QUIET_MODE:
                print(f"      📍 Router {self.router_id}: About to call handle_data() for {data_packet.name}", flush=True)
            self.handle_data(graph, data_packet, prev_node)
            if not QUIET_MODE:
                print(f"      ✅ Router {self.router_id}: handle_data() completed for {data_packet.name}", flush=True)
        elif message_type == 'fib_update':
            # Handle FIB update propagation (enqueued to avoid blocking workers)
            content_name, next_hop, G, visited = payload
            
            # DEDUPLICATION: Skip if we've recently processed this exact FIB update
            with self.fib_update_lock:
                fib_key = (content_name, next_hop)
                if fib_key in self.recent_fib_updates:
                    # Skip duplicate FIB update
                    logger.debug(f"Router {self.router_id}: Skipping duplicate FIB update for {content_name} via {next_hop}")
                    return
                # Add to recent set (will be cleaned up periodically)
                self.recent_fib_updates.add(fib_key)
                # Keep only last 1000 entries to prevent memory growth
                if len(self.recent_fib_updates) > 1000:
                    # Remove oldest 500 entries (simple cleanup)
                    self.recent_fib_updates = set(list(self.recent_fib_updates)[-500:])
            
            self.add_to_FIB(content_name, next_hop, G, visited)
        elif message_type == 'bloom_filter_update':
            # Handle Bloom filter update from neighbor
            neighbor_id, bloom_filter = payload
            if hasattr(self.content_store, 'receive_bloom_filter_update'):
                self.content_store.receive_bloom_filter_update(neighbor_id, bloom_filter)
        elif message_type == 'exact_cache_update':
            # FeiWang mode: Handle exact cache state update from neighbor
            neighbor_id, cache_contents = payload
            if hasattr(self.content_store, 'receive_exact_cache_update'):
                self.content_store.receive_exact_cache_update(neighbor_id, cache_contents)
        elif message_type == 'alive':
            # Heartbeat messages simply advance router time
            self.router_time += 0.1
        elif message_type == 'shutdown':
            # No-op; runtime handles worker termination
            return
        else:
            logger.warning(f"Router {self.router_id}: Received unknown message type {message_type}")
        
        # DEBUG: Log slow message processing
        dispatch_elapsed = time.time() - dispatch_start
        if dispatch_elapsed > 1.0:  # Log if takes more than 1 second
            logger.warning(
                f"Router {self.router_id}: dispatch_message({message_type}) took {dispatch_elapsed:.2f}s"
            )
            
    def update_neighbor_status(self, neighbor_id: int, status: str):
        """Update neighbor status (up/down)"""
        with self.neighbor_lock:
            old_status = self.neighbor_status.get(neighbor_id)
            self.neighbor_status[neighbor_id] = status
            if status == "down" and old_status == "up":
                self.remove_routes_through_neighbor(neighbor_id)
                logger.info(f"Router {self.router_id}: Neighbor {neighbor_id} went down")
                
    def remove_routes_through_neighbor(self, neighbor_id: int):
        """Remove all FIB entries going through a neighbor"""
        with self.FIB_lock:
            for content_name in list(self.FIB.keys()):
                if neighbor_id in self.FIB[content_name]:
                    self.FIB[content_name].remove(neighbor_id)
                    if not self.FIB[content_name]:
                        del self.FIB[content_name]
    
    # For compatibility with run.py
    def connect_router(self, router_id):
        """Re-establish connection with a router"""
        self.add_neighbor(router_id)
        
    # For compatibility with run.py
    def disconnect_router(self, router_id):
        """Disconnect from a router"""
        self.update_neighbor_status(router_id, "down")
    
    # For compatibility with run.py
    def remove_edge_from_FIB(self, router_id):
        """Remove routes through a neighbor after edge failure"""
        self.remove_routes_through_neighbor(router_id)
    
    # For compatibility with run.py
    def set_mode(self, mode):
        """Set router mode - compatibility method"""
        if hasattr(self.content_store, 'set_mode'):
            self.content_store.set_mode(mode)
                                
    def add_to_FIB(
        self,
        content_name: str,
        next_hop: int,
        G: nx.Graph = None,
        visited: Optional[Set[int]] = None
    ):
        """Add a route to the FIB and propagate it to neighbors."""
        if content_name is None or next_hop is None:
            logger.error(f"Router {self.router_id}: Invalid FIB entry parameters")
            return
            
        try:
            logger.debug(f"Router {self.router_id} adding {content_name} via {next_hop}")

            visited = set(visited) if visited else set()
            if self.router_id in visited:
                logger.debug(f"Router {self.router_id}: Already visited for {content_name}, skipping propagation loop")
                return
            visited.add(self.router_id)

            propagate_targets: List[int] = []
            with self.FIB_lock:
                # Check if we already have this route
                if content_name in self.FIB and next_hop in self.FIB[content_name]:
                    logger.debug(f"Router {self.router_id} already has route to {content_name} via {next_hop}")
                    return
                
                # Initialize FIB entry if doesn't exist
                if content_name not in self.FIB:
                    self.FIB[content_name] = []
                
                # Ensure next_hop is a Router ID
                if isinstance(next_hop, int):
                    next_hop_router_id = next_hop
                elif hasattr(next_hop, 'router_id'):
                    next_hop_router_id = next_hop.router_id
                else:
                    logger.warning(f"Router {self.router_id}: Invalid next_hop {next_hop}")
                    return  # Avoid adding incorrect entries
                
                # Add the new route
                self.FIB[content_name].append(next_hop_router_id)
                logger.debug(f"Router {self.router_id}: Added FIB entry {content_name} -> {next_hop_router_id}")

                # If no graph is provided, we can't propagate
                if G is None:
                    logger.debug(f"Router {self.router_id}: No graph provided, skipping propagation")
                    return

                # Only propagate to direct neighbors to avoid loops
                # This simplifies the propagation logic significantly
                with self.neighbor_lock:
                    for neighbor_id in self.neighbors:
                        if neighbor_id == next_hop_router_id:
                            continue
                        if self.neighbor_status.get(neighbor_id) != "up":
                            continue
                        if neighbor_id in visited:
                            continue
                        if neighbor_id not in G.nodes or 'router' not in G.nodes[neighbor_id]:
                            continue

                        neighbor_router = G.nodes[neighbor_id]['router']
                        neighbor_fib = getattr(neighbor_router, 'FIB', {})
                        if self.router_id in neighbor_fib.get(content_name, []):
                            continue

                        propagate_targets.append(neighbor_id)

            # Propagate outside the lock to avoid deadlocks
            # CRITICAL FIX: Enqueue FIB propagation instead of direct recursive calls
            # This prevents worker threads from getting stuck in recursive call chains
            # RATE LIMITING: Limit propagation to prevent exponential explosion
            # If MAX_FIB_PROPAGATION is 0 or negative, propagate to all neighbors (no limit)
            MAX_PROPAGATION_NEIGHBORS = int(os.environ.get('NDN_SIM_MAX_FIB_PROPAGATION', '10'))
            if MAX_PROPAGATION_NEIGHBORS > 0:
                # Limit to first N neighbors (rate limiting is the primary protection)
                propagation_targets = propagate_targets[:MAX_PROPAGATION_NEIGHBORS]
            # If MAX_PROPAGATION_NEIGHBORS <= 0, propagate to all neighbors (rely on rate limiting only)
            
            for neighbor_id in propagation_targets:
                try:
                    neighbor_router = G.nodes[neighbor_id]['router']
                    logger.debug(f"Router {self.router_id}: Enqueueing FIB propagation for {content_name} to {neighbor_id}")
                    # Use runtime to enqueue instead of direct call to avoid blocking worker
                    if self.runtime is not None:
                        next_visited = set(visited)
                        # Enqueue as a low-priority message to avoid blocking interest/data processing
                        self.runtime.enqueue(
                            neighbor_id,
                            priority=5,  # Low priority for FIB updates
                            message_type='fib_update',
                            payload=(content_name, self.router_id, G, next_visited)
                        )
                    else:
                        # Fallback to direct call if no runtime (shouldn't happen in normal operation)
                        next_visited = set(visited)
                        neighbor_router.add_to_FIB(content_name, self.router_id, G, next_visited)
                except Exception as e:
                    logger.error(f"Router {self.router_id}: Error propagating to {neighbor_id}: {e}")
        except Exception as e:
            logger.error(f"Router {self.router_id}: Error in add_to_FIB: {e}")
        
    def handle_interest(self, G, interest: Interest, prev_node: int):
        """Process incoming Interest packets with improved routing"""
        if not QUIET_MODE:
            print(f"         🔵 Router {self.router_id}: handle_interest() START for {interest.name} from {prev_node}", flush=True)
        logger.debug(f"Router {self.router_id}: Received Interest for {interest.name} from {prev_node}")
        interest_name = str(interest.name)
        
        # Metrics collection: Record Interest hop
        try:
            from metrics import get_metrics_collector
            metrics_collector = get_metrics_collector()
            metrics_collector.record_interest_hop(interest.interest_id, self.router_id)
        except Exception as e:
            logger.debug(f"Router {self.router_id}: Error recording Interest hop: {e}")
        
        try:
            # FIX: Normalize Interest creation_time to simulation time on first router
            # This ensures expiration checks use consistent simulation time
            # Interests are created with time.time() (real time), but we use router_time (simulation time)
            # If creation_time is significantly different from router_time, normalize it
            # Check if creation_time looks like real time (much larger than router_time, or negative difference)
            time_diff = interest.creation_time - self.router_time
            if abs(time_diff) > 1.0:  # More than 1 second difference = likely time mismatch
                # Normalize: set creation_time to current router_time (Interest just arrived)
                # This ensures expiration is calculated using simulation time
                interest.creation_time = self.router_time
            
            # FIX #3: Check Interest expiration (RFC 8569 compliance)
            # Use router_time (simulation time) instead of real time for consistency
            if interest.is_expired(self.router_time):
                logger.warning(f"Router {self.router_id}: Interest {interest.name} expired (lifetime exceeded); dropping")
                return
            
            # Check hop limit (RFC 8569 compliance)
            if interest.current_hops >= interest.hop_limit:
                logger.warning(f"Router {self.router_id}: Interest {interest.name} exceeded hop limit; dropping")
                return

            # FIX #4: Nonce-based loop detection (RFC 8569)
            # Drop Interests that we've already seen (same name + nonce = loop)
            with self.nonce_lock:
                if interest.nonce in self.seen_nonces[interest_name]:
                    logger.debug(
                        f"Router {self.router_id}: Duplicate Interest detected for {interest.name} "
                        f"with nonce {interest.nonce}; dropping (loop prevention)"
                    )
                    return
                # Mark this nonce as seen
                self.seen_nonces[interest_name].add(interest.nonce)
                
                # Keep nonce cache size bounded (clean up old entries periodically)
                if len(self.seen_nonces[interest_name]) > 1000:
                    logger.debug(f"Router {self.router_id}: Clearing nonce cache for {interest_name}")
                    self.seen_nonces[interest_name].clear()

            # If this router represents a producer, delegate to its Producer to generate Data
            if self.type == 'producer' and getattr(self, 'producer', None) is not None:
                try:
                    # FIX #1: Create PIT entry BEFORE delegating to producer
                    # This ensures Data packets have a return path through the network
                    self.PIT.add_entry(str(interest.name), prev_node, self.router_time)
                    logger.debug(f"Router {self.router_id}: Added {interest.name} to PIT before delegating to producer (incoming face: {prev_node})")
                    stats.update(nodes=1)  # Count node traversal
                    
                    originating_router = G.nodes[prev_node]['router'] if (G is not None and prev_node in G.nodes) else None
                    if originating_router is not None:
                        self.producer.get_interest(G, interest, originating_router)
                    return
                except Exception as e:
                    logger.error(f"Router {self.router_id}: Error delegating Interest to producer: {e}")
                    return

            # Record request for OPT baseline (needs to track all requests, not just cached)
            if hasattr(self.content_store, 'baseline_cache') and self.content_store.baseline_cache:
                if hasattr(self.content_store.baseline_cache, 'record_request'):
                    self.content_store.baseline_cache.record_request(str(interest.name))
            
            # Check if content is cached
            cached_content = self.content_store.get_content(str(interest.name), self.router_time)
            if cached_content:
                # FIX #5: Check if cached content is still fresh (RFC 8609)
                if hasattr(cached_content, 'is_fresh') and not cached_content.is_fresh(self.router_time):
                    logger.info(f"Router {self.router_id}: Cached content for {interest.name} is stale; not returning (freshness expired)")
                    # Remove stale entry from cache
                    self.content_store.remove_content(str(interest.name))
                else:
                    # Cache hits - only log in verbose mode (too many in quiet mode)
                    logger.debug(f"Router {self.router_id}: Cache hit for {interest.name}; responding to requester {prev_node}")
                    stats.update(hits=1)
                    # Notify ContentStore of cache hit for delayed DQN reward
                    if hasattr(self.content_store, 'notify_cache_hit'):
                        self.content_store.notify_cache_hit(str(interest.name), self.router_time)
                    # Metrics collection: Record cache hit
                    try:
                        from metrics import get_metrics_collector
                        metrics_collector = get_metrics_collector()
                        # Set interest_id on cached content for metrics tracking
                        if hasattr(cached_content, 'interest_id'):
                            cached_content.interest_id = interest.interest_id
                        # Get Data size from cached content
                        data_size = cached_content.size if hasattr(cached_content, 'size') else 0
                        if data_size == 0:
                            # Estimate: name length + content (assume 1KB default)
                            data_size = len(interest_name.encode('utf-8')) + 1024
                        metrics_collector.record_data_arrival(
                            interest.interest_id,
                            interest_name,
                            self.router_id,
                            from_cache=True,
                            data_size=data_size,
                            data_hops=0
                        )
                    except Exception as e:
                        logger.debug(f"Router {self.router_id}: Error recording cache hit: {e}")
                    cached_content.originator = interest.originator
                    cached_content.current_hops = 0
                    self.forward_data(G, cached_content, prev_node)
                    return

            # FIX #4: Clean up expired PIT entries periodically
            # Clean up every 10th Interest to avoid performance overhead
            if random.random() < 0.1:  # 10% chance to cleanup
                from packet import DEFAULT_LIFETIME
                self.PIT.cleanup_expired(self.router_time, DEFAULT_LIFETIME)
            
            # Check PIT for existing interests (aggregation)
            if str(interest.name) in self.PIT:
                logger.debug(f"Router {self.router_id}: Interest {interest.name} already in PIT. Aggregating request from {prev_node}.")
                self.PIT.add_entry(str(interest.name), prev_node, self.router_time)
                return

            # FIX #1: ALWAYS add to PIT before forwarding, regardless of FIB match
            # This ensures Data packets have a path back
            self.PIT.add_entry(str(interest.name), prev_node, self.router_time)
            stats.update(nodes=1)  # Count node traversal
            logger.debug(f"Router {self.router_id}: Added {interest.name} to PIT with incoming face {prev_node}")

            # Forward Interest based on hierarchical matching
            with self.FIB_lock:
                content_name = str(interest.name)
                next_nodes = []
                match_label = None

                # Try exact match first
                if content_name in self.FIB:
                    next_nodes = self.FIB[content_name].copy()  # Make a copy to avoid concurrent modification
                    logger.debug(f"Router {self.router_id}: Found exact match for {content_name}")
                    match_label = f"exact match"

                # Try prefix matching if no exact match
                if not next_nodes:
                    parts = content_name.split('/')
                    for i in range(len(parts), 1, -1):
                        prefix = '/'.join(parts[:i])
                        if prefix in self.FIB:
                            next_nodes = self.FIB[prefix].copy()  # Make a copy
                            logger.debug(f"Router {self.router_id}: Found prefix match {prefix} for {content_name}")
                            match_label = f"prefix match {prefix}"
                            break

                # Filter out the previous node to avoid loops
                next_nodes = [n for n in next_nodes if n != prev_node]

                if next_nodes and match_label:
                    logger.debug(
                        f"Router {self.router_id}: Forwarding Interest for {content_name} via {match_label} -> next hops {next_nodes}"
                    )

            # Handle case where no viable next hop exists
            if not next_nodes:
                with self.neighbor_lock:
                    potential_neighbors = [n for n in self.neighbors if n != prev_node and self.neighbor_status.get(n) == "up"]

                if len(potential_neighbors) == 1:
                    fallback = potential_neighbors[0]
                    logger.debug(f"Router {self.router_id}: No FIB entry for {content_name}. Using deterministic fallback via {fallback}.")
                    next_nodes = [fallback]
                elif len(potential_neighbors) > 1:
                    # IMPROVED: Try to find a neighbor that might have a route
                    # Check if any neighbor has a FIB entry for this content or prefix
                    best_neighbor = None
                    if G is not None:
                        # Try prefix matching on neighbors' FIBs
                        parts = content_name.split('/')
                        for i in range(len(parts), 1, -1):
                            prefix = '/'.join(parts[:i])
                            for neighbor_id in potential_neighbors:
                                if neighbor_id in G.nodes and 'router' in G.nodes[neighbor_id]:
                                    neighbor_router = G.nodes[neighbor_id]['router']
                                    with neighbor_router.FIB_lock:
                                        if prefix in neighbor_router.FIB or content_name in neighbor_router.FIB:
                                            best_neighbor = neighbor_id
                                            logger.debug(f"Router {self.router_id}: No FIB entry for {content_name}, but neighbor {neighbor_id} has route. Forwarding via {neighbor_id}.")
                                            break
                            if best_neighbor:
                                break
                    
                    # If no neighbor has a route, use random selection (better than NACK)
                    if not best_neighbor:
                        best_neighbor = random.choice(potential_neighbors)
                        logger.debug(f"Router {self.router_id}: No FIB entry for {content_name}. Using random fallback via {best_neighbor} (exploratory forwarding).")
                    
                    next_nodes = [best_neighbor]
                else:
                    # No neighbors available - send NACK
                    logger.debug(f"Router {self.router_id}: No FIB entry for {content_name} and no available neighbors. Sending NACK upstream.")
                    if prev_node is not None and G is not None and prev_node in G.nodes and 'router' in G.nodes[prev_node]:
                        nack_packet = Data(size=0, name=interest.name, originator=interest.originator, nack=True)
                        try:
                            self.send_message(
                                prev_node,
                                'data',
                                (G, nack_packet, self.router_id),
                                priority=1
                            )
                        except Exception as e:
                            logger.error(f"Router {self.router_id}: Error sending NACK to {prev_node}: {e}")
                    else:
                        logger.warning(f"Router {self.router_id}: Cannot determine previous hop for {content_name}; dropping Interest.")
                    return
                
            logger.debug(f"Router {self.router_id}: Forwarding Interest for {content_name} to {next_nodes}")
            
            # FIX #3: Clone Interest for each outgoing face (multi-face forwarding compliance)
            for next_node in next_nodes:
                try:
                    if G is not None and next_node in G.nodes and 'router' in G.nodes[next_node]:
                        # Clone the Interest for this outgoing face
                        interest_clone = interest.copy()
                        interest_clone.current_hops += 1
                        
                        self.send_message(
                            next_node,
                            'interest',
                            (G, interest_clone, self.router_id),
                            priority=2
                        )
                    else:
                        logger.warning(f"Router {self.router_id}: Next hop {next_node} not available for {content_name}")
                except Exception as e:
                    logger.error(f"Router {self.router_id}: Error forwarding Interest to {next_node}: {e}")
        finally:
            self.log_state_snapshot(f"after Interest {interest_name}")
            if not QUIET_MODE:
                print(f"         ✅ Router {self.router_id}: handle_interest() FINALLY completed for {interest_name}", flush=True)
                             
    def handle_data(self, G, data: Data, prev_node: int):
        """Process incoming Data packets"""
        if not QUIET_MODE:
            print(f"         🟢 Router {self.router_id}: handle_data() START for {data.name} from {prev_node}", flush=True)
        data_name = str(data.name)
        try:
            # Check hop limit (RFC 8569 compliance)
            if data.current_hops >= data.hop_limit:
                logger.warning(f"Router {self.router_id}: Data {data.name} exceeded hop limit; dropping")
                return

            start_time = time.time()
            logger.debug(f"Router {self.router_id}: Processing Data for {data.name}")
            
            # Handle normal data packet
            if not data.nack:
                # Store content in cache if router has capacity
                if self.capacity > 0:
                    # Determine content size
                    content_size = getattr(data, 'size', 10)  # Default size if not specified
                    
                    # Task 1.4: Add detailed logging for cache insertion debugging
                    logger.debug(
                        f"Router {self.router_id}: Attempting to cache {data.name} "
                        f"(size={content_size}, capacity={self.capacity}, "
                        f"remaining={getattr(self.content_store, 'remaining_capacity', 'unknown')})"
                    )
                    
                    # Check if DQN should respect placement policy
                    # If NDN_SIM_DQN_RESPECT_PLACEMENT=1, placement policy decides WHERE, DQN decides WHAT
                    # If NDN_SIM_DQN_RESPECT_PLACEMENT=0 (default), DQN decides both WHERE and WHAT
                    dqn_respects_placement = os.environ.get('NDN_SIM_DQN_RESPECT_PLACEMENT', '0') == '1'
                    dqn_active = (
                        getattr(self.content_store, "mode", "basic") == "dqn_cache"
                        and getattr(self.content_store, "dqn_agent", None) is not None
                    )
                    
                    # Placement policy is bypassed only if DQN is active AND not respecting placement
                    placement_limited = not (dqn_active and not dqn_respects_placement)
                    should_cache_packet = True
                    if placement_limited:
                        should_cache_packet = should_cache_content(
                            self, data, prev_node, self.placement_policy
                        )
                    if not should_cache_packet:
                        logger.debug(
                            f"Router {self.router_id}: Skipping cache for {data.name} "
                            f"due to placement policy {self.placement_policy}"
                        )
                    elif hasattr(self.content_store, 'store_content'):
                        cached_copy = data.clone()
                        cache_result = self.content_store.store_content(str(data.name), cached_copy, content_size, self.router_time, router=self, G=G)
                        if placement_limited:
                            record_cache_result(self, data, self.placement_policy, cache_result)
                        # Task 1.4: Track cache insertion statistics
                        self.stats.track_cache_insertion(cache_result)
                        if cache_result:
                            remaining = getattr(self.content_store, 'remaining_capacity', None)
                            logger.info(
                                f"Router {self.router_id}: Successfully cached {data.name} "
                                f"(size {content_size}); remaining_capacity={remaining}"
                            )
                            # Metrics collection: Record content location
                            try:
                                from metrics import get_metrics_collector
                                metrics_collector = get_metrics_collector()
                                metrics_collector.record_content_location(str(data.name), self.router_id)
                            except Exception as e:
                                logger.debug(f"Router {self.router_id}: Error recording content location: {e}")
                        else:
                            # Task 1.4: Detailed failure logging
                            remaining = getattr(self.content_store, 'remaining_capacity', None)
                            store_size = len(getattr(self.content_store, 'store', {}))
                            logger.warning(
                                f"Router {self.router_id}: Failed to cache {data.name} "
                                f"(size={content_size}, remaining={remaining}, "
                                f"store_size={store_size}, capacity={self.capacity})"
                            )
                    else:
                        logger.error(f"Router {self.router_id}: ContentStore missing store_content method")
                        self.stats.track_cache_insertion(False)
                else:
                    logger.debug(f"Router {self.router_id}: Skipping cache (capacity=0)")
                
                # Forward data to interested nodes from PIT
                interested_nodes = self.PIT.get(str(data.name))
                if interested_nodes:
                    logger.debug(f"Router {self.router_id}: Forwarding Data to {len(interested_nodes)} interested nodes from PIT")
                    for node in interested_nodes:
                        if node != prev_node:
                            try:
                                # FIX #3: Clone Data for each outgoing face
                                self.forward_data(G, data, node)
                            except Exception as e:
                                logger.error(f"Router {self.router_id}: Error forwarding Data to {node}: {e}")
                    
                    # Remove PIT entry after forwarding
                    self.PIT.remove_entry(str(data.name))
                else:
                    logger.debug(f"Router {self.router_id}: No PIT entries for {data.name}")
                
                # If this is a user router, deliver to connected users
                if self.type == 'user' and self.connected_users:
                    for user in self.connected_users:
                        if hasattr(user, 'get_data'):
                            user.get_data(data)
                            logger.info(
                                f"Router {self.router_id}: Delivered Data for {data.name} to user {getattr(user, 'user_id', 'unknown')}"
                            )
                            # Metrics collection: Record Data arrival at user (cache miss since it came from network)
                            try:
                                from metrics import get_metrics_collector
                                metrics_collector = get_metrics_collector()
                                if hasattr(data, 'interest_id') and data.interest_id:
                                    # Get Data packet size
                                    data_size = data.size if hasattr(data, 'size') else 0
                                    if data_size == 0:
                                        # Estimate: name length + content (assume 1KB default)
                                        data_size = len(data_name.encode('utf-8')) + 1024
                                    metrics_collector.record_data_arrival(
                                        data.interest_id,
                                        data_name,
                                        self.router_id,
                                        from_cache=False,
                                        data_size=data_size,
                                        data_hops=getattr(data, 'current_hops', 0)
                                    )
                            except Exception as e:
                                logger.debug(f"Router {self.router_id}: Error recording Data arrival: {e}")
            
            # Handle negative acknowledgment
            else:
                # NACKs are routine in NDN networks, only log in verbose mode
                if not QUIET_MODE:
                    logger.warning(f"Router {self.router_id}: Received NACK for {data.name}")
                else:
                    logger.debug(f"Router {self.router_id}: Received NACK for {data.name}")
                
                # Update routing information
                with self.FIB_lock:
                    if str(data.name) in self.FIB and prev_node in self.FIB[str(data.name)]:
                        self.FIB[str(data.name)].remove(prev_node)
                        if not self.FIB[str(data.name)]:
                            del self.FIB[str(data.name)]
                
                # Forward NACK to interested nodes
                interested_nodes = self.PIT.get(str(data.name))
                if interested_nodes:
                    for node in interested_nodes:
                        if node != prev_node:
                            try:
                                self.forward_data(G, data, node)
                            except Exception as e:
                                logger.error(f"Router {self.router_id}: Error forwarding NACK to {node}: {e}")
                    
                    # Remove PIT entry
                    self.PIT.remove_entry(str(data.name))
                
            # Record timing
            self.stats.add_timing("data_processing", time.time() - start_time)
            
        except Exception as e:
            logger.error(f"Router {self.router_id}: Error processing Data: {e}", exc_info=True)
        finally:
            self.log_state_snapshot(f"after Data {data_name}")
            if not QUIET_MODE:
                print(f"         ✅ Router {self.router_id}: handle_data() FINALLY completed for {data_name}", flush=True)
                             
    def forward_data(self, G, data: Data, next_node: int):
        """Forward Data packet to the next node"""
        if G is None or next_node not in G.nodes or 'router' not in G.nodes[next_node]:
            logger.error(f"Router {self.router_id}: Cannot forward Data - invalid graph or next node")
            return
            
        try:
            # FIX #3: Clone the Data packet for this outgoing face
            # This ensures each face gets an independent copy with correct hop count
            data_clone = data.clone()
            data_clone.current_hops += 1
            data_clone.data_trace.append(self.router_id)
            
            # FIX #8: Check hop limit before forwarding
            if data_clone.current_hops > data_clone.hop_limit:
                logger.warning(f"Router {self.router_id}: Data {data.name} would exceed hop limit; not forwarding to {next_node}")
                return
            
            stats.update(packets=1, size=getattr(data_clone, 'size', 0))
            
            logger.info(
                f"Router {self.router_id}: Forwarding Data for {data.name} to next hop {next_node} (hops: {data_clone.current_hops})"
            )
            self.send_message(
                next_node,
                'data',
                (G, data_clone, self.router_id),
                priority=1
            )
            logger.debug(f"Router {self.router_id}: Forwarded Data for {data.name} to {next_node}")
        except Exception as e:
            logger.error(f"Router {self.router_id}: Error forwarding Data: {e}")
            
    def log_state_snapshot(self, context: str):
        """Log FIB, PIT, and ContentStore state for tracing when enabled."""
        if not TRACE_STATE:
            return
        global TRACE_GLOBAL_MAX_EVENTS
        if TRACE_GLOBAL_MAX_EVENTS >= 0 and globals().get("_TRACE_GLOBAL_COUNT", 0) >= TRACE_GLOBAL_MAX_EVENTS:
            if not self._global_limit_hit:
                trace_logger.info(
                    f"[Trace] Router {self.router_id}: global trace limit ({TRACE_GLOBAL_MAX_EVENTS}) reached; suppressing further snapshots"
                )
                self._global_limit_hit = True
            return
        if TRACE_MAX_EVENTS >= 0 and self.trace_events_emitted >= TRACE_MAX_EVENTS:
            if not self.trace_limit_notified:
                trace_logger.info(
                    f"[Trace] Router {self.router_id}: trace event limit ({TRACE_MAX_EVENTS}) reached; suppressing further snapshots"
                )
                self.trace_limit_notified = True
            return
        try:
            with self.FIB_lock:
                fib_copy = {k: list(v) for k, v in self.FIB.items()}
            with self.PIT.lock:
                pit_copy = {k: list(v) for k, v in self.PIT.entries.items()}
            cs_items: List[str] = []
            remaining = None
            if hasattr(self.content_store, 'store'):
                store_lock = getattr(self.content_store, 'store_lock', None)
                if store_lock:
                    store_lock.acquire()
                try:
                    cs_items = list(getattr(self.content_store, 'store', {}).keys())
                    remaining = getattr(self.content_store, 'remaining_capacity', None)
                finally:
                    if store_lock:
                        store_lock.release()
            trace_logger.info(
                f"[Trace] Router {self.router_id} {context}: "
                f"FIB={_summarize_mapping(fib_copy)}, "
                f"PIT={_summarize_mapping(pit_copy)}, "
                f"CS_items={_summarize_sequence(cs_items)} remaining_capacity={remaining}"
            )
            self.trace_events_emitted += 1
            globals()["_TRACE_GLOBAL_COUNT"] = globals().get("_TRACE_GLOBAL_COUNT", 0) + 1
        except Exception as e:
            logger.error(f"Router {self.router_id}: Error logging state snapshot: {e}")
                
    def get_state_for_dqn(self, content_name: str, content_size: int, current_time: float = None) -> np.ndarray:
        """Create state representation for DQN agent (delegates to ContentStore)"""
        if hasattr(self.content_store, 'get_state_for_dqn'):
            # Use router_time if current_time not provided
            if current_time is None:
                current_time = self.router_time
            # Fix: Pass graph reference instead of None to enable topology features
            graph_ref = self.content_store.graph_ref if hasattr(self.content_store, 'graph_ref') else None
            return self.content_store.get_state_for_dqn(content_name, content_size, router=self, G=graph_ref, current_time=current_time)
        else:
            # Fallback implementation if ContentStore doesn't have the method
            state = np.zeros(7, dtype=np.float32)  # Use valid numpy dtype
            
            # Basic state features (you can expand these as needed)
            state[0] = float(content_name in self.content_store.store) if hasattr(self.content_store, 'store') else 0.0
            state[1] = float(content_size) / max(1, self.capacity)
            state[2] = float(self.content_store.remaining_capacity) / max(1, self.capacity) if hasattr(self.content_store, 'remaining_capacity') else 0.5
            
            return state
        
    def shutdown(self):
        """Gracefully shutdown the router"""
        logger.info(f"Router {self.router_id}: Initiating shutdown")
        if self.runtime is not None:
            self.runtime.deregister_router(self.router_id)
        logger.info(f"Router {self.router_id}: Shutdown complete")

    def set_placement_policy(self, policy: str):
        self.placement_policy = (policy or "lce").lower()
