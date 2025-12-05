"""Explicit state-space exploration (Task 2)."""

from __future__ import annotations

import sys
import time
from collections import deque
from pathlib import Path
from typing import Deque, Dict, Set, Tuple

# Add parent directory to path
_parent_dir = Path(__file__).parent.parent
if str(_parent_dir) not in sys.path:
    sys.path.insert(0, str(_parent_dir))

from task1_parser.pnml_parser import PetriNet

MarkingDict = Dict[str, int]
MarkingTuple = Tuple[int, ...]


class ExplicitReachability:
    """Enumerate reachable markings using BFS/DFS."""

    def __init__(self, net: PetriNet):
        self.net = net
        self._places: Tuple[str, ...] = tuple(sorted(net.places))
        self.reachable_markings: Set[MarkingTuple] = set()
        self.num_states: int = 0
        self.computation_time: float = 0.0

    def _marking_to_tuple(self, marking: MarkingDict) -> MarkingTuple:
        return tuple(marking.get(place, 0) for place in self._places)

    def _tuple_to_marking(self, marking: MarkingTuple) -> MarkingDict:
        return {place: marking[idx] for idx, place in enumerate(self._places)}

    def _is_enabled(self, marking: MarkingDict, transition: str) -> bool:
        """Check if transition is enabled at given marking."""
        return all(marking.get(place, 0) == 1 for place in self.net.input_arcs[transition])

    def _fire(self, marking: MarkingDict, transition: str) -> MarkingDict:
        """Fire transition and return new marking."""
        successor = marking.copy()
        for place in self.net.input_arcs[transition]:
            successor[place] = 0
        for place in self.net.output_arcs[transition]:
            successor[place] = 1
        return successor

    def _explore(self, frontier: Deque[MarkingDict], pop) -> Set[MarkingTuple]:
        """Generic exploration with custom pop strategy."""
        visited: Set[MarkingTuple] = set()
        
        while frontier:
            current = pop()
            current_key = self._marking_to_tuple(current)
            if current_key in visited:
                continue
            visited.add(current_key)

            for transition in self.net.transitions:
                if not self._is_enabled(current, transition):
                    continue
                successor = self._fire(current, transition)
                succ_key = self._marking_to_tuple(successor)
                if succ_key not in visited:
                    frontier.append(successor)
        return visited

    def compute_bfs(self) -> Set[MarkingTuple]:
        """Compute reachable markings using BFS."""
        start = time.perf_counter()
        initial_marking = self.net.initial_marking
        frontier: Deque[MarkingDict] = deque([initial_marking])
        explored = self._explore(frontier, frontier.popleft)
        self.reachable_markings = explored
        self.num_states = len(explored)
        self.computation_time = time.perf_counter() - start
        return explored

    def compute_dfs(self) -> Set[MarkingTuple]:
        """Compute reachable markings using DFS."""
        start = time.perf_counter()
        initial_marking = self.net.initial_marking
        frontier: Deque[MarkingDict] = deque([initial_marking])
        explored = self._explore(frontier, frontier.pop)
        self.reachable_markings = explored
        self.num_states = len(explored)
        self.computation_time = time.perf_counter() - start
        return explored

    def print_results(self, method_name: str = "BFS") -> None:
        """Print detailed results."""
        print(f"\n--- {method_name} Reachable Markings ---")
        # Print all markings as lists
        for marking in sorted(self.reachable_markings):
            print(list(marking))
        print(f"Total {method_name} reachable = {self.num_states}")


if __name__ == "__main__":
    from task1_parser.pnml_parser import PNMLParser

    if len(sys.argv) != 2:
        print("Usage: python explicit_reachability.py <pnml_file>")
        raise SystemExit(1)

    model_path = sys.argv[1]
    try:
        net = PNMLParser.parse(model_path)
        net.print_summary()
        
        # BFS
        explicit = ExplicitReachability(net)
        explicit.compute_bfs()
        explicit.print_results("BFS")
        
        # DFS
        explicit_dfs = ExplicitReachability(net)
        explicit_dfs.compute_dfs()
        explicit_dfs.print_results("DFS")
        
    except Exception as exc:
        print(f"Error: {exc}")
        raise SystemExit(2) from exc