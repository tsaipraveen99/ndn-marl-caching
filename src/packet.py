import time
import random
import uuid
import os
import math
from typing import List, Union, Optional
from dataclasses import dataclass, field

# Constants - configurable via environment variables
# Default values scale with network size if NDN_SIM_NODES is set
def _get_hop_limit():
    """Calculate hop limit based on network size"""
    base_limit = int(os.environ.get("NDN_SIM_HOP_LIMIT", "0"))
    if base_limit > 0:
        return base_limit
    
    # Auto-scale based on network size
    num_nodes = int(os.environ.get("NDN_SIM_NODES", "30"))
    # For small-world networks: diameter ≈ log(n)/log(k) where k is average degree
    # Watts-Strogatz with k=4: diameter ≈ log(n)/log(4)
    # Add safety margin: 2 * diameter + buffer
    if num_nodes <= 50:
        return 15  # Small network
    elif num_nodes <= 200:
        return 25  # Medium network
    else:
        # Large network: log(n)/log(4) * 2 + buffer
        diameter_estimate = int(math.log(num_nodes) / math.log(4)) * 2
        return max(30, diameter_estimate + 10)  # At least 30, with buffer

def _get_lifetime():
    """Calculate interest lifetime based on network size"""
    base_lifetime = float(os.environ.get("NDN_SIM_INTEREST_LIFETIME", "0"))
    if base_lifetime > 0:
        return base_lifetime
    
    # Auto-scale based on network size
    num_nodes = int(os.environ.get("NDN_SIM_NODES", "30"))
    # Larger networks need more time for routing
    if num_nodes <= 50:
        return 4.0  # Small network: 4 seconds
    elif num_nodes <= 200:
        return 8.0  # Medium network: 8 seconds
    else:
        return 12.0  # Large network: 12 seconds

HOP_LIMIT = _get_hop_limit()
DEFAULT_LIFETIME = _get_lifetime()
NONCE_BITS = 32

class NDNName:
    """
    Hierarchical NDN name structure
    Example: /edu/ucla/cs/files/paper.pdf
    """
    def __init__(self, name_components: Union[str, List[str]]):
        if isinstance(name_components, str):
            # Convert string format (/a/b/c) to list ['a', 'b', 'c']
            self.components = [x for x in name_components.split('/') if x]
        else:
            self.components = name_components

    def __str__(self) -> str:
        return '/' + '/'.join(self.components)

    def __repr__(self) -> str:
        return f"NDNName({self.__str__()})"

    def get_prefix(self, length: Optional[int] = None) -> 'NDNName':
        """Get a prefix of the name with specified length"""
        if length is None:
            return self
        return NDNName(self.components[:length])

    def matches_prefix(self, prefix: Union[str, 'NDNName']) -> bool:
        """Check if name matches a given prefix"""
        if isinstance(prefix, str):
            prefix = NDNName(prefix)
        return (len(self.components) >= len(prefix.components) and
                self.components[:len(prefix.components)] == prefix.components)

    def __eq__(self, other: Union[str, 'NDNName']) -> bool:
        if isinstance(other, str):
            other = NDNName(other)
        return self.components == other.components

    def __hash__(self) -> int:
        return hash(str(self))
        
    def __len__(self) -> int:
        return len(self.components)
        
    def is_valid(self) -> bool:
        """Check if the name is valid"""
        return bool(self.components) and all(isinstance(c, str) and c for c in self.components)

@dataclass
class Data:
    """
    NDN Data packet implementation
    
    Attributes:
        size: Content size in bytes
        name: Content name
        originator: ID of originating node
        nack: Negative acknowledgment flag
        current_hops: Number of hops traversed
        hop_limit: Maximum allowed hops
        suggestion: Caching suggestion value
        freshness_period: How long this data is fresh in seconds (RFC 8609)
        interest_id: ID of the Interest that requested this Data (for metrics tracking)
    """
    size: int
    name: Union[str, NDNName]
    originator: int
    nack: bool = False
    current_hops: int = 0
    hop_limit: int = field(default_factory=_get_hop_limit)
    suggestion: float = 0.3
    freshness_period: float = 10.0  # FIX #5: Default 10 seconds freshness (RFC 8609)
    creation_time: float = field(default_factory=time.time)
    data_trace: List[int] = field(default_factory=list)
    interest_id: Optional[str] = None  # Link to Interest for metrics tracking
    cached_downstream: bool = False
    last_cached_router: Optional[int] = None
    
    def __post_init__(self):
        if isinstance(self.name, str):
            self.name = NDNName(self.name)
    
    def is_fresh(self, current_time: float = None) -> bool:
        """
        FIX #5: Check if this Data packet is still fresh (RFC 8609)
        
        Args:
            current_time: Current time (default: now)
            
        Returns:
            True if within freshness period, False otherwise
        """
        if current_time is None:
            current_time = time.time()
        age = current_time - self.creation_time
        is_fresh_val = age < self.freshness_period
        return is_fresh_val
    
    def clone(self, originator: Optional[int] = None) -> 'Data':
        """Create a fresh copy safe for caching or forwarding."""
        cloned = Data(
            size=self.size,
            name=str(self.name),
            originator=self.originator if originator is None else originator,
            nack=self.nack,
            hop_limit=self.hop_limit,
            suggestion=self.suggestion,
            freshness_period=self.freshness_period,  # Preserve freshness period
            interest_id=self.interest_id,  # Preserve interest_id for metrics tracking
            cached_downstream=self.cached_downstream,
            last_cached_router=self.last_cached_router
        )
        cloned.data_trace = list(self.data_trace)
        return cloned
        
    def is_valid(self) -> bool:
        """Validate packet fields"""
        return (
            isinstance(self.size, int) and self.size > 0 and
            isinstance(self.name, NDNName) and self.name.is_valid() and
            isinstance(self.originator, int) and
            isinstance(self.current_hops, int) and
            self.current_hops <= self.hop_limit
        )
        
    def add_trace(self, node_id: int):
        """Add node to packet trace"""
        self.data_trace.append(node_id)
        
    def __repr__(self) -> str:
        return (f"Data(name={self.name}, size={self.size}, hops={self.current_hops}, "
                f"nack={self.nack}, originator={self.originator})")

@dataclass
class Interest:
    """
    NDN Interest packet implementation
    
    Attributes:
        name: Requested content name
        originator: ID of originating node
        can_be_prefix: Allow prefix matching
        lifetime: Interest lifetime in seconds
        current_hops: Number of hops traversed
        hop_limit: Maximum allowed hops
        nonce: Unique packet identifier
        interest_id: Unique identifier for metrics tracking
    """
    name: Union[str, NDNName]
    originator: int
    can_be_prefix: bool = True
    lifetime: float = field(default_factory=_get_lifetime)
    current_hops: int = 0
    hop_limit: int = field(default_factory=_get_hop_limit)
    nonce: int = field(default_factory=lambda: random.randint(0, 2**NONCE_BITS-1))
    creation_time: float = field(default_factory=time.time)
    interest_trace: List[int] = field(default_factory=list)
    interest_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    
    def __post_init__(self):
        if isinstance(self.name, str):
            self.name = NDNName(self.name)
            
    def is_expired(self, current_time: float = None) -> bool:
        """
        Check if the Interest packet has expired
        
        Args:
            current_time: Current simulation time (default: real time)
                          Should use router_time for simulation consistency
        """
        if current_time is None:
            current_time = time.time()  # Fallback to real time for compatibility
        return (current_time - self.creation_time) > self.lifetime
        
    def is_valid(self) -> bool:
        """Validate packet fields"""
        return (
            isinstance(self.name, NDNName) and self.name.is_valid() and
            isinstance(self.originator, int) and
            isinstance(self.current_hops, int) and
            self.current_hops <= self.hop_limit and
            isinstance(self.nonce, int) and
            0 <= self.nonce < 2**NONCE_BITS
        )

    def copy(self) -> 'Interest':
        """Create a copy of the Interest packet"""
        return Interest(
            name=self.name,
            originator=self.originator,
            can_be_prefix=self.can_be_prefix,
            lifetime=self.lifetime,
            current_hops=self.current_hops,
            hop_limit=self.hop_limit,
            nonce=self.nonce,
            creation_time=self.creation_time,
            interest_trace=self.interest_trace.copy(),
            interest_id=self.interest_id  # Preserve interest_id for metrics tracking
        )
        
    def add_trace(self, node_id: int):
        """Add node to packet trace"""
        self.interest_trace.append(node_id)

    def __repr__(self) -> str:
        return (f"Interest(name={self.name}, nonce={self.nonce}, "
                f"hops={self.current_hops}, expired={self.is_expired()}, "
                f"originator={self.originator})")
