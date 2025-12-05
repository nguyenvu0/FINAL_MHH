"""Petri-net PNML parser used across tasks."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Tuple
import numpy as np

__all__ = ["PetriNet", "PNMLParser"]

PNML_NS = "http://www.pnml.org/version-2009/grammar/pnml"
PLACE_TAG = "place"
TRANSITION_TAG = "transition"
ARC_TAG = "arc"
INITIAL_MARKING_TAG = "initialMarking"


@dataclass
class PetriNet:
    """Minimal in-memory representation of a 1-safe Petri net."""

    places: List[str] = field(default_factory=list)
    transitions: List[str] = field(default_factory=list)
    
    # Dictionary-based representation (for explicit tasks)
    input_arcs: Dict[str, List[str]] = field(default_factory=dict)
    output_arcs: Dict[str, List[str]] = field(default_factory=dict)
    initial_marking: Dict[str, int] = field(default_factory=dict)
    
    # Matrix-based representation (for symbolic/linear algebra tasks)
    I: np.ndarray = field(default_factory=lambda: np.zeros((0, 0), dtype=int))
    O: np.ndarray = field(default_factory=lambda: np.zeros((0, 0), dtype=int))
    M0: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=int))
    
    # Metadata
    place_names: List[str] = field(default_factory=list)
    trans_names: List[str] = field(default_factory=list)

    def ensure_arc_entries(self) -> None:
        for transition in self.transitions:
            self.input_arcs.setdefault(transition, [])
            self.output_arcs.setdefault(transition, [])

    def verify_consistency(self) -> None:
        """
        Verify the consistency of the Petri net.
        Checks for:
        - Isolated transitions (no input and no output).
        - Isolated places (no input and no output).
        """
        # Check for isolated transitions
        for i in range(len(self.transitions)):
            is_input = np.sum(self.I[i, :]) > 0
            is_output = np.sum(self.O[i, :]) > 0
            if not is_input and not is_output:
                print(f"WARNING: Transition '{self.transitions[i]}' is isolated (no input/output arcs).")
        
        # Check for isolated places (optional)
        for i in range(len(self.places)):
            is_consumed = np.sum(self.I[:, i]) > 0
            is_produced = np.sum(self.O[:, i]) > 0
            if not is_consumed and not is_produced:
                print(f"WARNING: Place '{self.places[i]}' is isolated.")

    def print_summary(self) -> None:
        """Print detailed Petri net information."""
        divider = "=" * 60
        print(f"\n{divider}")
        print("--- Petri Net Loaded ---")
        print(f"{divider}")
        
        # Places
        print(f"Places: {self.places}")
        if any(self.place_names):
            valid_names = [n if n else f"P{i+1}" for i, n in enumerate(self.place_names)]
            print(f"Place names: {valid_names}")
        
        # Transitions
        print(f"\nTransitions: {self.transitions}")
        if any(self.trans_names):
            valid_names = [n if n else f"T{i+1}" for i, n in enumerate(self.trans_names)]
            print(f"Transition names: {valid_names}")
        
        # Incidence Matrices
        print(f"\nI (input) matrix:")
        print(self.I)
        
        print(f"\nO (output) matrix:")
        print(self.O)
        
        # Initial marking
        print(f"\nInitial marking M0:")
        print(self.M0)
        print(f"{divider}\n")


class PNMLParser:
    """Convert PNML files into :class:`PetriNet` instances."""

    @classmethod
    def parse(cls, filename: str) -> PetriNet:
        try:
            tree = ET.parse(filename)
            root = tree.getroot()
        except FileNotFoundError as exc:
            raise FileNotFoundError(f"File not found: {filename}") from exc
        except ET.ParseError as exc:
            raise ValueError(f"Invalid PNML/XML structure: {exc}") from exc

        import re
        m = re.match(r"\{(.+)\}", root.tag)
        ns_uri = m.group(1) if m else ""
        ns = {"pnml": ns_uri} if ns_uri else {}
        
        def q(tag: str) -> str:
            return f"pnml:{tag}" if ns_uri else tag
            
        net_node = root.find(q("net"), ns)
        if net_node is None:
            net_node = root.find(".//" + q("net"), ns)
        if net_node is None:
            raise ValueError("No <net> element found in PNML file")

        # Places
        places = net_node.findall(".//" + q("place"), ns)
        place_ids = [p.attrib.get("id") for p in places]
        
        # Transitions
        transitions = net_node.findall(".//" + q("transition"), ns)
        trans_ids = [t.attrib.get("id") for t in transitions]
        
        # Names (optional)
        def get_label(elem: ET.Element, child_tag: str) -> str | None:
            child = elem.find(q(child_tag), ns)
            if child is None: return None
            text_elem = child.find(q("text"), ns)
            if text_elem is None or text_elem.text is None: return None
            return text_elem.text.strip()
            
        place_names = [get_label(p, "name") for p in places]
        trans_names = [get_label(t, "name") for t in transitions]

        # Initial Marking (Dict + Array)
        initial_marking = {p: 0 for p in place_ids}
        M0 = np.zeros(len(place_ids), dtype=int)
        
        for i, p in enumerate(places):
            pid = place_ids[i]
            val = get_label(p, "initialMarking")
            tokens = 0
            if val:
                try:
                    tokens = int(val)
                except ValueError:
                    try:
                        tokens = int(float(val))
                    except ValueError:
                        tokens = 0
            initial_marking[pid] = tokens
            M0[i] = tokens

        # Matrices & Arcs
        I = np.zeros((len(trans_ids), len(place_ids)), dtype=int)
        O = np.zeros((len(trans_ids), len(place_ids)), dtype=int)
        
        place_idx = {pid: i for i, pid in enumerate(place_ids)}
        trans_idx = {tid: i for i, tid in enumerate(trans_ids)}

        input_arcs = {t: [] for t in trans_ids}
        output_arcs = {t: [] for t in trans_ids}

        # Arcs
        arcs = net_node.findall(".//" + q("arc"), ns)
        
        def arc_weight(a: ET.Element) -> int:
            ins = a.find(q("inscription"), ns)
            if ins is None: return 1
            text_elem = ins.find(q("text"), ns)
            if text_elem is None or text_elem.text is None: return 1
            txt = text_elem.text.strip()
            try:
                return int(txt)
            except ValueError:
                try:
                    return int(float(txt))
                except ValueError:
                    return 1

        for a in arcs:
            src = a.attrib.get("source")
            tgt = a.attrib.get("target")
            w = arc_weight(a)

            if src in place_idx and tgt in trans_idx:
                # place -> transition
                I[trans_idx[tgt], place_idx[src]] += w
                input_arcs[tgt].append(src)
            elif src in trans_idx and tgt in place_idx:
                # transition -> place
                O[trans_idx[src], place_idx[tgt]] += w
                output_arcs[src].append(tgt)

        net = PetriNet(
            places=place_ids,
            transitions=trans_ids,
            input_arcs=input_arcs,
            output_arcs=output_arcs,
            initial_marking=initial_marking,
            I=I,
            O=O,
            M0=M0,
            place_names=place_names,
            trans_names=trans_names
        )
        net.verify_consistency()
        return net


if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python pnml_parser.py <pnml_file>")
        raise SystemExit(1)
    try:
        PNMLParser.parse(sys.argv[1]).print_summary()
    except Exception as exc:
        print(f"Error: {exc}")
        raise SystemExit(2) from exc
