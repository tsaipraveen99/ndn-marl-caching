import threading 
import logging
import time
import queue
import os
from typing import Dict, List, Optional
from packet import Data, Interest, NDNName
from utils import PIT

logger = logging.getLogger('endpoints_logger')
# Control verbosity: QUIET mode suppresses routine warnings
QUIET_MODE = os.environ.get("NDN_SIM_QUIET", "1") == "1"  # Default to quiet
if QUIET_MODE:
    logger.setLevel(logging.WARNING)  # Only show warnings and errors
else:
    logger.setLevel(logging.INFO)  # Show all info messages

class Producer:
    def __init__(self, routers: List, contents: Dict[str, int]):
        """
        Initialize a producer with content repository
        
        Args:
            routers: List of routers this producer is connected to
            contents: Dict of {content_name: content_size} this producer hosts
        """
        self.routers = routers
        self.contents = contents
        self.PIT = PIT()
        self.lock = threading.Lock()
        self.request_count = 0
        self.router_id = routers[0].router_id if routers else None
        
        logger.info(f"Producer initialized with {len(contents)} contents")

    def _register_content_with_routers(self, G):
        """Register content with connected routers and their neighbors"""
        with self.lock:
            for router in self.routers:
                for content_name in self.contents:
                    try:
                        # Create domain prefix for better route aggregation
                        parts = content_name.split('/')
                        if len(parts) >= 4:
                            domain_prefix = '/'.join(parts[:4])  # /edu/university/department
                            
                            # Register with directly connected router
                            router.add_to_FIB(
                                content_name=content_name,
                                next_hop=self.router_id,
                                G=G
                            )
                            
                            # Also register the domain prefix
                            router.add_to_FIB(
                                content_name=domain_prefix,
                                next_hop=self.router_id,
                                G=G
                            )
                            logger.debug(f"✅ Added FIB entries for {content_name} and {domain_prefix}")

                    except Exception as e:
                        logger.error(f"❌ Error registering content {content_name}: {e}")

    def get_interest(self, G, interest_packet: Interest, originating_router):
        """Process incoming Interest packets and respond with Data"""
        with self.lock:
            name = str(interest_packet.name)
            logger.debug(f"Producer checking for content: {name}")

            # Check for exact match first
            if name in self.contents:
                matched_name = name
            else:
                # Try prefix matching
                name_parts = name.split('/')
                for i in range(len(name_parts), 3, -1):
                    prefix = '/'.join(name_parts[:i])
                    if any(content.startswith(prefix) for content in self.contents.keys()):
                        matched_name = next(c for c in self.contents.keys() if c.startswith(prefix))
                        break
                else:
                    matched_name = None

            if matched_name:
                self.request_count += 1
                source_router = self.router_id if self.router_id is not None else (
                    self.routers[0].router_id if self.routers else -1
                )
                logger.info(
                    f"Producer: Serving {name} (matched {matched_name}) from router {source_router}"
                )
                data_packet = Data(
                    size=self.contents[matched_name],
                    name=name,
                    originator=interest_packet.originator,
                    interest_id=interest_packet.interest_id  # Link Data to Interest for metrics tracking
                )
                logger.debug(f"Producer: Sending Data packet for {name} to {originating_router.router_id}")

                # FIX #2: Send Data to producer's LOCAL ROUTER first (not directly to requester)
                # This allows intermediate routers to cache the content as it flows back
                # The local router will use PIT to forward it back to the requester
                # Note: Using self.router_id as prev_node is correct - the producer router will
                # receive this Data and use its PIT entry (created in Fix #1) to forward it
                # to the actual incoming face (the router that sent the Interest)
                producer_router = self.routers[0] if self.routers else None
                if producer_router:
                    logger.info(
                        f"Producer: Forwarding Data for {name} to local router {producer_router.router_id} "
                        f"for propagation via PIT (was: direct to requester)"
                    )
                    # Verify Data name matches Interest name for PIT lookup
                    assert str(data_packet.name) == name, f"Data name mismatch: {data_packet.name} != {name}"
                    producer_router.submit_message(
                        'data',
                        (G, data_packet, self.router_id),
                        priority=1
                    )
                else:
                    logger.error(f"Producer: No local router available; cannot send Data for {name}")
            else:
                logger.debug(f"Producer: Content {name} NOT FOUND. Sending NACK.")
                nack_packet = Data(size=0, name=name, originator=interest_packet.originator, nack=True)
                producer_router_id = self.router_id if self.router_id is not None else (
                    self.routers[0].router_id if self.routers else -1
                )
                # Send NACK via local router as well
                producer_router = self.routers[0] if self.routers else None
                if producer_router:
                    producer_router.submit_message(
                        'data',
                        (G, nack_packet, producer_router_id),
                        priority=1
                    )
                else:
                    logger.error(f"Producer: No local router available; cannot send NACK for {name}")

    def get_statistics(self) -> Dict:
        """Return producer statistics"""
        with self.lock:
            return {
                'total_contents': len(self.contents),
                'total_requests': self.request_count,
                'connected_routers': len(self.routers)
            }
                
    def __str__(self) -> str:
        routers = "  ".join(str(r.router_id) for r in self.routers)
        return f"Producer connected to: {routers} hosting {len(self.contents)} contents"

class User:
    def __init__(self, user_id: int, router_id: int, connected_router, distribution):
        """
        Initialize a user that makes content requests
        
        Args:
            user_id: Unique identifier for this user
            router_id: ID of the router this user is connected to
            connected_router: Router object this user is connected to
            distribution: NDNDistribution object to generate content names
        """
        self.user_id = user_id
        self.router_id = router_id
        self.connected_router = connected_router
        self.distribution = distribution
        
        # Thread safety and synchronization
        self.lock = threading.Lock()
        self.data_received_condition = threading.Condition(self.lock)
        
        # State tracking
        self.received_data: Optional[Data] = None
        self.request_history: List[str] = []
        self.mode = "sim"
        self.request_count = 0
        self.successful_requests = 0
        
        logger.info(f"User {user_id} initialized and connected to router {router_id}")

    def make_interest(self, G, content_name=None):
        """Generate and send an Interest packet"""
        try:
            with self.lock:
                self.request_count += 1
                if content_name is None:
                    content_name = self.distribution.generate_content_name()

                # Normalize content name to match producer format
                if not content_name.startswith('/edu/'):
                    parts = content_name.split('/')
                    if len(parts) >= 3:
                        content_name = f"/edu/{parts[-3]}/{parts[-2]}/{parts[-1]}"

                # Create Interest packet
                interest_packet = Interest(
                    name=content_name,
                    originator=self.connected_router.router_id
                )

                # Metrics collection: Record Interest creation
                try:
                    from metrics import get_metrics_collector
                    metrics_collector = get_metrics_collector()
                    # Estimate Interest packet size: name length + headers (approx 100 bytes overhead)
                    interest_size = len(content_name.encode('utf-8')) + 100
                    metrics_collector.record_interest(interest_packet.interest_id, content_name, self.connected_router.router_id, interest_size=interest_size)
                except Exception as e:
                    logger.debug(f"User {self.user_id}: Error recording Interest: {e}")

                logger.debug(f"📡 User {self.user_id}: Creating Interest for {content_name}")

                # Verify router connectivity
                if not self.connected_router or not hasattr(self.connected_router, "runtime") or self.connected_router.runtime is None:
                    logger.error(f"❌ User {self.user_id}: Router runtime not available!")
                    return None

                # Check for FIB entry
                fib_entries = self.connected_router.FIB.get(content_name, [])
                if not fib_entries:
                    # Try domain prefix match
                    parts = content_name.split('/')
                    if len(parts) >= 4:
                        domain_prefix = '/'.join(parts[:4])
                        fib_entries = self.connected_router.FIB.get(domain_prefix, [])
                        if fib_entries:
                            logger.debug(f"🔍 Found prefix match {domain_prefix} for {content_name}")
                        else:
                            logger.debug(f"⚠️ User {self.user_id}: No FIB entry for {content_name} before sending.")

                # Send Interest
                self.connected_router.submit_message(
                    'interest',
                    (G, interest_packet, self.router_id),
                    priority=2
                )
                self.request_history.append(content_name)

                return None

        except Exception as e:
            logger.error(f"❌ User {self.user_id}: Error making interest request: {e}")
            return None

    def get_data(self, data: Data) -> None:
        """Handle received data packets"""
        with self.lock:
            try:
                self.received_data = data
                if not data.nack:
                    self.successful_requests += 1
                if self.mode == "test":
                    with self.data_received_condition:
                        self.data_received_condition.notify_all()
            except Exception as e:
                logger.error(f"Error processing received data: {e}")

    def run(self, G, num_rounds: int = 10, num_requests: int = 5, num_contents: int = None) -> None:
        """
        Make content requests over multiple rounds
        
        Args:
            G: Network graph
            num_rounds: Number of rounds to run
            num_requests: Number of requests per round
            num_contents: Optional parameter for compatibility with main.py
        """
        try:
            # Check if we should skip delays for benchmark mode
            skip_delays = os.environ.get("NDN_SIM_SKIP_DELAYS", "0") == "1"
            
            for round_ in range(num_rounds):
                for _ in range(num_requests):
                    content_name = self.distribution.generate_content_name()
                    self.make_interest(G, content_name)
                    if not skip_delays:
                        time.sleep(0.01)  # Small delay between requests
                    
                logger.debug(f"User {self.user_id} completed round {round_ + 1}")
                
        except Exception as e:
            logger.error(f"Error in user run loop: {e}")
            
    def get_statistics(self) -> Dict:
        """Return user statistics"""
        with self.lock:
            return {
                'total_requests': self.request_count,
                'successful_requests': self.successful_requests,
                'success_rate': self.successful_requests / max(1, self.request_count),
                'unique_contents_requested': len(set(self.request_history))
            }

    def __str__(self) -> str:
        return f"User {self.user_id} connected to router {self.router_id}"
