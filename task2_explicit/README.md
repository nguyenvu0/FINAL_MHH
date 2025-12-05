# Task 2: Explicit Reachability Computation

## 📋 Yêu Cầu Đề

**Task 2**: Implement thuật toán BFS (Breadth-First Search) hoặc DFS (Depth-First Search) để enumerate (liệt kê) tất cả các marking khả đạt (reachable markings) từ initial marking trong 1-safe Petri net.

## 🔍 Giải Thích Code

### 1. Import và Type Aliases

```python
from collections import deque
from task1_parser.pnml_parser import PetriNet

MarkingDict = Dict[str, int]    # {place_name: token_count}
MarkingTuple = Tuple[int, ...]  # (token1, token2, ...)
```

- `deque`: Cấu trúc queue để implement BFS/DFS
- `MarkingDict`: Biểu diễn marking dạng dictionary (dễ đọc)
- `MarkingTuple`: Biểu diễn marking dạng tuple (để lưu trong set - hashable)

### 2. Class ExplicitReachability

**Khởi tạo:**
```python
def __init__(self, net: PetriNet):
    self.net = net
    self._places: Tuple[str, ...] = tuple(sorted(net.places))  # Sắp xếp alphabetically
    self.reachable_markings: Set[MarkingTuple] = set()
    self.num_states: int = 0
    self.computation_time: float = 0.0
```

- `_places`: Danh sách places đã sắp xếp để đảm bảo thứ tự nhất quán
- `reachable_markings`: Set chứa tất cả markings khả đạt

### 3. Conversion Helpers

**Dict ↔ Tuple:**
```python
def _marking_to_tuple(self, marking: MarkingDict) -> MarkingTuple:
    return tuple(marking.get(place, 0) for place in self._places)

def _tuple_to_marking(self, marking: MarkingTuple) -> MarkingDict:
    return {place: marking[idx] for idx, place in enumerate(self._places)}
```

- Convert giữa 2 dạng biểu diễn để dễ thao tác

### 4. Transition Firing

**Kiểm tra enabled:**
```python
def _is_enabled(self, marking: MarkingDict, transition: str) -> bool:
    return all(marking.get(place, 0) == 1 for place in self.net.input_arcs[transition])
```

- Transition enabled khi **TẤT CẢ** input places đều có token = 1 (1-safe net)

**Fire transition:**
```python
def _fire(self, marking: MarkingDict, transition: str) -> MarkingDict:
    successor = marking.copy()
    for place in self.net.input_arcs[transition]:
        successor[place] = 0  # Remove tokens
    for place in self.net.output_arcs[transition]:
        successor[place] = 1  # Add tokens
    return successor
```

- Input places: set token = 0
- Output places: set token = 1
- Với 1-safe net: chỉ cần set 0/1, không cần cộng/trừ

### 5. State Space Exploration (Core Algorithm)

**Thuật toán BFS/DFS:**
```python
def _explore(self, frontier: Deque[MarkingDict], pop) -> Set[MarkingTuple]:
    visited: Set[MarkingTuple] = set()
    while frontier:
        current = pop()  # BFS: popleft(), DFS: pop()
        current_key = self._marking_to_tuple(current)
        if current_key in visited:
            continue  # Tránh visit lại
        visited.add(current_key)
        for transition in self.net.transitions:
            if not self._is_enabled(current, transition):
                continue
            successor = self._fire(current, transition)
            frontier.append(successor)
    return visited
```

**Giải thích:**
1. Dùng `visited` set để tránh visit lại cùng một marking (prevent loops)
2. Lấy marking từ `frontier`:
   - **BFS**: `popleft()` - Lấy từ đầu (FIFO)
   - **DFS**: `pop()` - Lấy từ cuối (LIFO)
3. Với mỗi transition enabled → fire → tạo successor → thêm vào frontier

**Main method:**
```python
def compute(self, strategy: str = "bfs") -> Set[MarkingTuple]:
    start = time.perf_counter()
    initial_marking = self.net.initial_marking
    frontier: Deque[MarkingDict] = deque([initial_marking])
    explored = self._explore(
        frontier,
        frontier.popleft if strategy.lower() == "bfs" else frontier.pop,
    )
    self.reachable_markings = explored
    self.num_states = len(explored)
    self.computation_time = time.perf_counter() - start
    return explored
```

- Khởi tạo với initial marking
- Chọn BFS hoặc DFS bằng cách truyền function `pop` khác nhau
- Đo thời gian computation

## 📊 Kết Quả

### Test 1: simple.pnml

**Input:**
- Places: 3 (p_start, p_mid, p_end)
- Transitions: 2 (t1, t2)
- Initial: p_start=1
- Flow: p_start → t1 → p_mid → t2 → p_end

**Output:**
```
Reachable markings: 3
Computation time: 0.0000 seconds

Places: ('p_end', 'p_mid', 'p_start')
 1. (0, 0, 1)  ← Initial: p_start=1
 2. (0, 1, 0)  ← Sau t1: p_mid=1
 3. (1, 0, 0)  ← Sau t2: p_end=1
```

**Giải thích:**
- Có 3 markings khả đạt: initial, sau t1, sau t2
- Đúng: 3 reachable markings ✅

---

### Test 2: producer_consumer.pnml

**Output:**
```
Reachable markings: 2
Computation time: 0.0000 seconds

Places: ('buffer_empty', 'buffer_full', 'consumer_idle', 'producer_idle')
 1. (1, 0, 1, 1)  ← Initial: buffer_empty=1
 2. (0, 1, 1, 1)  ← Sau produce: buffer_full=1
```

**Giải thích:**
- Có 2 markings: buffer_empty hoặc buffer_full (producer và consumer đều idle)
- Đúng: 2 reachable markings ✅

---

### Test 3: mutual_exclusion.pnml

**Output:**
```
Reachable markings: 8
Computation time: 0.0001 seconds
```

**Giải thích:**
- Với 7 places và 6 transitions, có 8 markings khả đạt
- Đúng: 8 reachable markings ✅

---

## ✅ Kết Luận

- ✅ Code đúng yêu cầu: Implement BFS và DFS
- ✅ Tìm được tất cả reachable markings
- ✅ BFS và DFS cho cùng kết quả (đúng lý thuyết)
- ✅ Performance tốt (thời gian < 0.001s cho mạng nhỏ)
- ✅ Kết quả chính xác cho tất cả test cases

## 🚀 Cách Chạy

```bash
cd D:\MHH\petri-net-analyzer
python task2_explicit/explicit_reachability.py test_models/simple.pnml
```
