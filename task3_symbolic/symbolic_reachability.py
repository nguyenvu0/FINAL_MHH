"""BDD-based symbolic reachability (Task 3)."""

from __future__ import annotations

import sys
import time
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Set
import itertools

from task1_parser.pnml_parser import PetriNet
# ExplicitReachability is no longer the primary method, but kept for comparison if needed
from task2_explicit.explicit_reachability import ExplicitReachability, MarkingTuple

try:
    from pyeda.boolalg.bdd import BinaryDecisionDiagram, bddvar, expr2bdd
    from pyeda.boolalg.expr import exprvar, And, Or, Not
except ImportError as exc:  # pragma: no cover - import guard
    raise ImportError("Symbolic reachability requires pyeda. Install via `pip install pyeda`. ") from exc


class SymbolicReachability:
    """Construct a BDD representing all reachable markings using symbolic image computation."""

    def __init__(self, net: PetriNet):
        self.net = net
        self._places: Tuple[str, ...] = tuple(sorted(net.places))
        self.place_to_idx = {p: i for i, p in enumerate(self._places)}
        
        # Create BDD variables for current state (x) and next state (x')
        # We use 'x' and 'y' prefixes for current and next state variables respectively
        self._x_vars = [bddvar(f"x_{i}") for i in range(len(self._places))]
        self._y_vars = [bddvar(f"y_{i}") for i in range(len(self._places))]
        
        # Map for renaming x' back to x (y -> x)
        self._y_to_x = {self._y_vars[i]: self._x_vars[i] for i in range(len(self._places))}

        self.bdd: Optional[BinaryDecisionDiagram] = None
        self.num_states: int = 0
        self.computation_time: float = 0.0

    def _build_initial_marking_bdd(self) -> BinaryDecisionDiagram:
        """Construct BDD for the initial marking M0."""
        expr = None
        for i, place in enumerate(self._places):
            val = self.net.initial_marking.get(place, 0)
            # For 1-safe nets, value is 0 or 1
            lit = self._x_vars[i] if val > 0 else ~self._x_vars[i]
            expr = lit if expr is None else (expr & lit)
        return expr

    def _build_transition_relation(self) -> BinaryDecisionDiagram:
        """
        Construct the global transition relation T(x, y).
        T(x, y) = OR_t ( Enabled_t(x) AND NextState_t(x, y) )
        """
        global_rel = None

        for t in self.net.transitions:
            # 1. Enabled condition: All input places must have a token
            inputs = self.net.input_arcs.get(t, [])
            outputs = self.net.output_arcs.get(t, [])
            
            enabled_expr = None
            for p in inputs:
                idx = self.place_to_idx[p]
                lit = self._x_vars[idx]
                enabled_expr = lit if enabled_expr is None else (enabled_expr & lit)
            
            if enabled_expr is None: 
                # If no inputs, it's always enabled (source transition)
                pass 

            # 2. Next state logic
            # For p in inputs \ outputs: loses token (y_i = 0)
            # For p in outputs \ inputs: gains token (y_i = 1)
            # For p in inputs AND outputs: stays 1 (y_i = 1) - self-loop
            # For others: y_i = x_i
            
            input_set = set(inputs)
            output_set = set(outputs)
            
            change_expr = None
            
            for i, p in enumerate(self._places):
                is_in = p in input_set
                is_out = p in output_set
                
                if is_in and not is_out:
                    # Consumed: y_i = 0
                    next_val = ~self._y_vars[i]
                elif is_out:
                    # Produced (or self-loop): y_i = 1
                    # Note: 1-safe assumption means we don't count > 1
                    next_val = self._y_vars[i]
                else:
                    # Unchanged: y_i = x_i
                    # (x_i & y_i) | (~x_i & ~y_i)  <==> x_i XNOR y_i
                    next_val = (~self._x_vars[i] & ~self._y_vars[i]) | (self._x_vars[i] & self._y_vars[i])
                
                change_expr = next_val if change_expr is None else (change_expr & next_val)

            # T_t = Enabled AND Change
            if enabled_expr is not None:
                trans_rel = enabled_expr & change_expr
            else:
                trans_rel = change_expr # Always enabled
            
            global_rel = trans_rel if global_rel is None else (global_rel | trans_rel)
            
        return global_rel

    def build_symbolic(self) -> BinaryDecisionDiagram:
        """Perform symbolic reachability analysis (Fixed-point iteration)."""
        start_time = time.perf_counter()
        
        # 1. Initial Marking S0
        S = self._build_initial_marking_bdd()
        
        # 2. Transition Relation T(x, y)
        T = self._build_transition_relation()
        
        if T is None:
            # No transitions, only initial state reachable
            self.bdd = S
            self.num_states = 1
            self.computation_time = time.perf_counter() - start_time
            return S

        # 3. Fixed-point iteration
        # S_new = S_old OR (exists x. (S_old(x) AND T(x, y)))[rename y->x]
        
        iteration = 0
        while True:
            iteration += 1
            # Image computation:
            # a. Conjoin S(x) and T(x, y)
            # b. Existential quantification over x variables
            # c. Rename y variables to x variables
            
            next_states_y = (S & T).smoothing(self._x_vars)
            
            # Step c: Rename y -> x
            next_states_x = next_states_y.compose(self._y_to_x)
            
            # Union with accumulated states
            S_new = S | next_states_x
            
            # Check convergence
            if S_new.equivalent(S):
                break
            
            S = S_new
            
        self.bdd = S
        self.num_states = self.count()
        self.computation_time = time.perf_counter() - start_time
        return S

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def count(self) -> int:
        if self.bdd is None:
            return 0
        # Count satisfying assignments over the full domain of x_vars
        # satisfy_count() returns count over support. 
        # We need to adjust for variables in x_vars but NOT in support.
        # However, satisfy_count() behavior in pyeda can be tricky.
        # Safer to sum 2^(missing_vars) for each path/model.
        
        total = 0
        for model in self.bdd.satisfy_all():
            # model is a dict of {var: val}
            # missing vars = total_vars - len(model)
            missing = len(self._x_vars) - len(model)
            total += (1 << missing)
        return total

    def enumerate_markings(self, limit: Optional[int] = None) -> List[MarkingTuple]:
        if self.bdd is None:
            return []
        results: List[MarkingTuple] = []
        
        # Helper to expand don't cares
        for model in self.bdd.satisfy_all():
            # Identify fixed and missing variables
            fixed = {}
            missing_indices = []
            
            for i, var in enumerate(self._x_vars):
                if var in model:
                    fixed[i] = int(model[var])
                else:
                    missing_indices.append(i)
            
            # Generate all combinations for missing variables
            for p in itertools.product([0, 1], repeat=len(missing_indices)):
                marking_list = [0] * len(self._x_vars)
                
                # Fill fixed
                for idx, val in fixed.items():
                    marking_list[idx] = val
                
                # Fill missing
                for i, val in enumerate(p):
                    marking_list[missing_indices[i]] = val
                
                results.append(tuple(marking_list))
                if limit is not None and len(results) >= limit:
                    return results
                    
        return results

    def summary(self) -> None:
        print("=" * 60)
        print("TASK 3: SYMBOLIC REACHABILITY (BDD) - FIXED POINT")
        print("=" * 60)
        print(f"BDD constructed: {'yes' if self.bdd is not None else 'no'}")
        print(f"Reported reachable markings: {self.num_states}")
        print(f"Construction time: {self.computation_time:.4f} seconds")
        print("=" * 60 + "\n")

    # ------------------------------------------------------------------
    # Convenience driver
    # ------------------------------------------------------------------

    def build_and_compare(self) -> None:
        print("Running Symbolic Reachability...")
        self.build_symbolic()
        self.summary()
        
        print("Running Explicit Reachability (BFS) for verification...")
        explicit = ExplicitReachability(self.net)
        explicit.compute_bfs()
        
        print("Comparison:")
        print(f"  Symbolic States: {self.num_states}")
        print(f"  Explicit States: {explicit.num_states}")
        
        if self.num_states == explicit.num_states:
            print("  >> SUCCESS: State counts match.")
        else:
            print("  >> WARNING: State counts do NOT match!")

        print(f"  Symbolic Time: {self.computation_time:.4f}s")
        print(f"  Explicit Time: {explicit.computation_time:.4f}s")
        print()


if __name__ == "__main__":  # pragma: no cover - manual execution helper
    from task1_parser.pnml_parser import PNMLParser

    if len(sys.argv) != 2:
        print("Usage: python symbolic_reachability.py <pnml_file>")
        raise SystemExit(1)

    path = sys.argv[1]
    try:
        net = PNMLParser.parse(path)
        net.print_summary()
        SymbolicReachability(net).build_and_compare()
    except Exception as exc:  # noqa: BLE001 - CLI convenience
        print(f"Error: {exc}")
        raise SystemExit(2) from exc