"""Deadlock detection utilities (Task 4)."""

from __future__ import annotations

import sys
import time
from typing import Dict, Iterable, Optional, Sequence, Set, Tuple

from task1_parser.pnml_parser import PetriNet
from task3_symbolic.symbolic_reachability import SymbolicReachability

try:
    from pyeda.boolalg.bdd import BinaryDecisionDiagram
except ImportError as exc:  # pragma: no cover - import guard
    raise ImportError("Deadlock detection requires pyeda. Install via `pip install pyeda`. ") from exc

try:
    import pulp
except ImportError as exc:
    raise ImportError("ILP-based detection requires pulp. Install via `pip install pulp`. ") from exc

MarkingDict = Dict[str, int]
MarkingTuple = Tuple[int, ...]


class DeadlockDetector:
    """Provide symbolic deadlock checks."""

    def __init__(self, net: PetriNet):
        self.net = net
        self._places: Tuple[str, ...] = tuple(sorted(net.places))
        self.computation_time: float = 0.0

    def _tuple_to_marking(self, marking: MarkingTuple) -> MarkingDict:
        return {place: marking[idx] for idx, place in enumerate(self._places)}

    def detect_symbolic(self, sym_reach: SymbolicReachability) -> Optional[MarkingDict]:
        """
        Detect deadlock using BDD operations.
        Deadlock = Reachable AND (NOT Enabled_t1) AND (NOT Enabled_t2) ...
        """
        start = time.perf_counter()
        
        reachable_bdd = sym_reach.bdd
        if reachable_bdd is None:
            return None
            
        # 1. Build "Dead" condition: No transition is enabled
        # Dead(x) = AND_t (NOT Enabled_t(x))
        
        dead_expr = None
        
        # We need to use the SAME variables as SymbolicReachability
        # Accessing private _x_vars from sym_reach instance
        x_vars = sym_reach._x_vars
        place_to_idx = sym_reach.place_to_idx
        
        for t in self.net.transitions:
            inputs = self.net.input_arcs.get(t, [])
            
            # Enabled_t = AND (x_p) for p in inputs
            enabled_t = None
            for p in inputs:
                idx = place_to_idx[p]
                lit = x_vars[idx]
                enabled_t = lit if enabled_t is None else (enabled_t & lit)
            
            if enabled_t is None:
                # Always enabled -> No deadlock possible if this transition exists
                # Dead(x) will be False
                self.computation_time = time.perf_counter() - start
                return None
            
            # Not Enabled
            not_enabled = ~enabled_t
            dead_expr = not_enabled if dead_expr is None else (dead_expr & not_enabled)
            
        if dead_expr is None:
            # No transitions in net? Then the only state is deadlock.
            dead_expr = reachable_bdd # effectively 1 (True) relative to context, or just check reachable
        
        # 2. Intersection: DeadlockStates = Reachable & Dead
        deadlock_bdd = reachable_bdd & dead_expr
        
        # 3. Check if empty
        if deadlock_bdd.is_zero():
            self.computation_time = time.perf_counter() - start
            return None
            
        # 4. Extract one witness
        assignment = deadlock_bdd.satisfy_one()
        if assignment is None:
             self.computation_time = time.perf_counter() - start
             return None
             
        # Convert assignment to MarkingDict
        marking = {}
        for i, place in enumerate(self._places):
            var = x_vars[i]
            val = assignment.get(var, 0)
            marking[place] = int(val)
            
        self.computation_time = time.perf_counter() - start
        return marking

    def detect_with_ilp_bdd(self, sym_reach: SymbolicReachability, max_candidates: int = 1000) -> Optional[MarkingDict]:
        """
        Detect deadlock using combination of ILP and BDD (as required by assignment).
        Strategy:
        1. Use BDD to enumerate candidate reachable markings
        2. Use ILP to find which marking is a deadlock (no enabled transitions)
        """
        start = time.perf_counter()
        
        # Step 1: Get candidate markings from BDD
        candidates = sym_reach.enumerate_markings(limit=max_candidates)
        
        if not candidates:
            self.computation_time = time.perf_counter() - start
            return None
        
        # Step 2: Use ILP to find deadlock
        # Variables: y[i] = 1 if we select candidate i as deadlock
        prob = pulp.LpProblem("DeadlockDetection", pulp.LpMaximize)
        
        # Decision variables
        y = pulp.LpVariable.dicts("select", range(len(candidates)), cat='Binary')
        
        # Objective: maximize selection (just need to find one)
        prob += pulp.lpSum([y[i] for i in range(len(candidates))]), "FindOne"
        
        # Constraint: Select exactly one marking
        prob += pulp.lpSum([y[i] for i in range(len(candidates))]) == 1, "SelectOne"
        
        # Constraint: Selected marking must be dead (no enabled transitions)
        for i, marking_tuple in enumerate(candidates):
            marking = self._tuple_to_marking(marking_tuple)
            
            # Check if this marking is dead
            is_dead = True
            for t in self.net.transitions:
                # Check if transition t is enabled
                inputs = self.net.input_arcs.get(t, [])
                if all(marking.get(p, 0) == 1 for p in inputs):
                    is_dead = False
                    break
            
            # If not dead, force y[i] = 0
            if not is_dead:
                prob += y[i] == 0, f"NotDead_{i}"
        
        # Solve
        prob.solve(pulp.PULP_CBC_CMD(msg=0))
        
        # Extract result
        result = None
        if prob.status == pulp.LpStatusOptimal:
            for i in range(len(candidates)):
                if pulp.value(y[i]) == 1:
                    result = self._tuple_to_marking(candidates[i])
                    break
        
        self.computation_time = time.perf_counter() - start
        return result


def print_deadlock(marking: Optional[MarkingDict]) -> None:
    if marking is None:
        print("No deadlock detected")
        return
    print("Deadlock marking:")
    for place, value in marking.items():
        print(f"  {place}: {value}")


if __name__ == "__main__":  # pragma: no cover - convenience CLI
    from task1_parser.pnml_parser import PNMLParser

    if len(sys.argv) != 2:
        print("Usage: python deadlock_detection.py <pnml_file>")
        raise SystemExit(1)

    model = sys.argv[1]
    try:
        net = PNMLParser.parse(model)
        net.print_summary()

        # Step 1: Build Symbolic Reachability
        sym = SymbolicReachability(net)
        print("Building symbolic reachability...")
        sym.build_symbolic()
        sym.summary()

        # Step 2: Detect Deadlock (Using ILP+BDD as required)
        detector = DeadlockDetector(net)
        print("\nMethod 1: Pure BDD approach")
        deadlock = detector.detect_symbolic(sym)
        print(f"Time: {detector.computation_time:.4f}s")
        print_deadlock(deadlock)
        
        print("\n" + "="*60)
        print("Method 2: ILP + BDD approach (as required by assignment)")
        deadlock_ilp = detector.detect_with_ilp_bdd(sym, max_candidates=500)
        print(f"Time: {detector.computation_time:.4f}s")
        print_deadlock(deadlock_ilp)

    except Exception as exc:  # noqa: BLE001 - CLI helper
        print(f"Error: {exc}")
        raise SystemExit(2) from exc