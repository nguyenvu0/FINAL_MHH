"""Optimization over reachable markings (Task 5)."""

from __future__ import annotations

import sys
import time
from typing import Dict, Iterable, Optional, Sequence, Tuple

from task1_parser.pnml_parser import PetriNet
from task3_symbolic.symbolic_reachability import SymbolicReachability

try:
    import pulp
except ImportError as exc:
    raise ImportError("ILP-based optimization requires pulp. Install via `pip install pulp`. ") from exc

MarkingTuple = Tuple[int, ...]
MarkingDict = Dict[str, int]


class ReachabilityOptimizer:
    """Support optimization over reachable markings using BDD traversal."""

    def __init__(self, net: PetriNet):
        self.net = net
        self._places: Tuple[str, ...] = tuple(sorted(net.places))
        self.computation_time: float = 0.0
        self.optimal_marking: Optional[MarkingDict] = None
        self.optimal_value: Optional[int] = None

    def _tuple_to_marking(self, marking: MarkingTuple) -> MarkingDict:
        return {place: marking[idx] for idx, place in enumerate(self._places)}

    def optimize_symbolic(
        self,
        sym_reach: SymbolicReachability,
        weights: Dict[str, int],
        maximize: bool = True,
    ) -> Optional[MarkingDict]:
        """
        Find marking M in Reachable(M0) that optimizes c^T M.
        Algorithm: Recursive search using BDD restrictions (Shannon expansion).
        """
        start = time.perf_counter()
        bdd = sym_reach.bdd
        
        if bdd is None or bdd.is_zero():
            self.optimal_marking = None
            self.optimal_value = None
            self.computation_time = time.perf_counter() - start
            return None

        # Map variable names to weights
        # weights is Dict[str, int]
        
        memo = {}

        def solve(current_bdd) -> Tuple[float, Dict[str, int]]:
            # Base cases
            if current_bdd.is_zero():
                return (float("-inf") if maximize else float("inf")), {}
            if current_bdd.is_one():
                return 0.0, {}
            
            # Memoization
            # Use object id or uniqid if available
            uid = id(current_bdd)
            if uid in memo:
                return memo[uid]
            
            # Get top variable
            try:
                # Pick the first variable in the support
                support = list(current_bdd.support)
                # Sort by uniqid to be deterministic
                support.sort(key=lambda v: v.uniqid)
                top_v = support[0]
            except Exception:
                # Should not happen if not one/zero
                return 0.0, {}

            # Weight of this variable
            w = 0
            var_name = str(top_v)
            # var_name is like "x_0"
            if var_name.startswith("x_"):
                try:
                    idx = int(var_name.split("_")[1])
                    place_name = sym_reach._places[idx]
                    w = weights.get(place_name, 0)
                except:
                    pass
            
            # Branch 0 (Low)
            bdd_low = current_bdd.restrict({top_v: 0})
            val_low, path_low = solve(bdd_low)
            
            # Branch 1 (High)
            bdd_high = current_bdd.restrict({top_v: 1})
            val_high, path_high = solve(bdd_high)
            total_high = w + val_high
            
            # Compare
            if maximize:
                if total_high >= val_low:
                    best_val = total_high
                    best_path = path_high.copy()
                    best_path[var_name] = 1
                else:
                    best_val = val_low
                    best_path = path_low.copy()
                    best_path[var_name] = 0
            else: # Minimize
                if total_high <= val_low:
                    best_val = total_high
                    best_path = path_high.copy()
                    best_path[var_name] = 1
                else:
                    best_val = val_low
                    best_path = path_low.copy()
                    best_path[var_name] = 0
            
            memo[uid] = (best_val, best_path)
            return best_val, best_path

        final_val, final_path = solve(bdd)
        
        # Reconstruct marking
        best_marking = {}
        for i, place in enumerate(self._places):
            var_name = f"x_{i}"
            w = weights.get(place, 0)
            
            if var_name in final_path:
                val = final_path[var_name]
            else:
                # Don't care
                if maximize:
                    val = 1 if w > 0 else 0
                else:
                    val = 1 if w < 0 else 0
            
            best_marking[place] = val
            
        self.optimal_value = int(final_val)
        self.optimal_marking = best_marking
        self.computation_time = time.perf_counter() - start
        return best_marking

    def optimize_with_ilp(
        self,
        sym_reach: SymbolicReachability,
        weights: Dict[str, int],
        maximize: bool = True,
        max_candidates: int = 1000
    ) -> Optional[MarkingDict]:
        """
        Find optimal marking using ILP formulation (as required by assignment).
        Strategy:
        1. Use BDD to enumerate reachable markings
        2. Use ILP to solve: maximize c^T M subject to M in Reachable(M0)
        """
        start = time.perf_counter()
        
        # Step 1: Get candidate markings from BDD
        candidates = sym_reach.enumerate_markings(limit=max_candidates)
        
        if not candidates:
            self.optimal_marking = None
            self.optimal_value = None
            self.computation_time = time.perf_counter() - start
            return None
        
        # Step 2: Formulate ILP
        prob = pulp.LpProblem("OptimizeReachability", pulp.LpMaximize if maximize else pulp.LpMinimize)
        
        # Decision variables: y[i] = 1 if we select marking i
        y = pulp.LpVariable.dicts("select", range(len(candidates)), cat='Binary')
        
        # Constraint: Select exactly one marking
        prob += pulp.lpSum([y[i] for i in range(len(candidates))]) == 1, "SelectOne"
        
        # Objective: maximize/minimize c^T M where M is the selected marking
        objective = []
        for i, marking_tuple in enumerate(candidates):
            marking = self._tuple_to_marking(marking_tuple)
            # Compute c^T M for this marking
            value = sum(weights.get(p, 0) * marking.get(p, 0) for p in self._places)
            objective.append(value * y[i])
        
        prob += pulp.lpSum(objective), "Objective"
        
        # Solve
        prob.solve(pulp.PULP_CBC_CMD(msg=0))
        
        # Extract result
        result = None
        opt_val = None
        if prob.status == pulp.LpStatusOptimal:
            for i in range(len(candidates)):
                if pulp.value(y[i]) == 1:
                    result = self._tuple_to_marking(candidates[i])
                    opt_val = sum(weights.get(p, 0) * result.get(p, 0) for p in self._places)
                    break
        
        self.optimal_marking = result
        self.optimal_value = opt_val
        self.computation_time = time.perf_counter() - start
        return result

    def objective_value(self, marking: MarkingDict, weights: Dict[str, int]) -> int:
        return sum(weights.get(place, 0) * marking.get(place, 0) for place in self._places)


if __name__ == "__main__":  # pragma: no cover - manual helper
    from task1_parser.pnml_parser import PNMLParser

    if len(sys.argv) != 2:
        print("Usage: python optimization.py <pnml_file>")
        raise SystemExit(1)

    pnml = sys.argv[1]
    try:
        net = PNMLParser.parse(pnml)
        net.print_summary()
        
        # Step 1: Build Symbolic Reachability
        sym = SymbolicReachability(net)
        print("Building symbolic reachability...")
        sym.build_symbolic()
        sym.summary()

        weights = {place: 1 for place in net.places}
        # Make some weights interesting
        for i, p in enumerate(sorted(net.places)):
            if i % 2 == 0: weights[p] = 2
            if i % 3 == 0: weights[p] = -1
            
        print("Objective weights (partial):", list(weights.items())[:5], "...")

        optimizer = ReachabilityOptimizer(net)
        
        print("\n" + "="*60)
        print("Method 1: Pure BDD traversal")
        best_marking = optimizer.optimize_symbolic(sym, weights, maximize=True)
        print(f"Time: {optimizer.computation_time:.4f}s")
        print(f"Optimal Value: {optimizer.optimal_value}")
        print(f"Optimal Marking: {best_marking}")
        
        print("\n" + "="*60)
        print("Method 2: ILP formulation (as required by assignment)")
        best_marking_ilp = optimizer.optimize_with_ilp(sym, weights, maximize=True, max_candidates=500)
        print(f"Time: {optimizer.computation_time:.4f}s")
        print(f"Optimal Value: {optimizer.optimal_value}")
        print(f"Optimal Marking: {best_marking_ilp}")

    except Exception as exc:  # noqa: BLE001 - CLI helper
        print(f"Error: {exc}")
        raise SystemExit(2) from exc