# 1.2-2 연결리스트 (Linked List)

## (1) 문제 정의

- **연결리스트(Linked List)**는 데이터를 노드 단위로 저장하고, 각 노드가 다음 노드의 주소를 가리키는 자료구조이다.
- 배열은 연속 메모리에 저장되지만, 연결리스트는 **비연속 메모리**를 **포인터(링크)**로 연결한다.
- 종류:
    - **단일 연결리스트**: `val`, `next`
    - **이중 연결리스트**: `val`, `prev`, `next`
    - **원형 연결리스트**: 마지막 노드가 처음 노드를 가리킴

---

## (2) 접근 방법 / 핵심 원리

- 삽입·삭제: 포인터만 수정하면 되어 **O(1)** (해당 위치를 알고 있을 때).
- 접근: 순차 탐색 필요 → **O(n)**.
- 메모리: 노드마다 포인터 오버헤드 존재.
- CPU 캐시 효율: 배열보다 떨어짐.

---

## (3) 복잡도 간단 표

| 연산 | 배열(Array) | 연결리스트(Linked List) |
| --- | --- | --- |
| 임의 접근 | O(1) | O(n) |
| 맨 앞 삽입/삭제 | O(n) | O(1) |
| 중간 삽입/삭제 | O(n) | O(1) (위치 포인터 알고 있을 때) |
| 맨 뒤 삽입 | O(1) amort. | O(1) (tail 유지 시) |

---

## (4) 구현 코드 & 예제

### Python (단일 연결리스트)

```python
class Node:
    def __init__(self, val, nxt=None):
        self.val = val
        self.next = nxt

class SinglyLinkedList:
    def __init__(self):
        self.head = None

    def push_front(self, val):
        self.head = Node(val, self.head)

    def pop_front(self):
        if not self.head:
            raise IndexError("pop from empty list")
        val = self.head.val
        self.head = self.head.next
        return val

    def find(self, val):
        cur = self.head
        while cur:
            if cur.val == val:
                return True
            cur = cur.next
        return False

    def __str__(self):
        cur, out = self.head, []
        while cur:
            out.append(str(cur.val))
            cur = cur.next
        return " -> ".join(out)

# 사용 예시
lst = SinglyLinkedList()
lst.push_front(3); lst.push_front(2); lst.push_front(1)
print(lst)         # 1 -> 2 -> 3
print(lst.pop_front()) # 1
print(lst.find(2))     # True

```

---

### C++ (단일 연결리스트)

```cpp
#include <bits/stdc++.h>
using namespace std;

struct Node {
    int val;
    Node* next;
    Node(int v): val(v), next(nullptr) {}
};

struct SinglyLinkedList {
    Node* head;
    SinglyLinkedList(): head(nullptr) {}

    void push_front(int v) {
        Node* node = new Node(v);
        node->next = head;
        head = node;
    }

    int pop_front() {
        if(!head) throw runtime_error("pop from empty list");
        int v = head->val;
        Node* tmp = head;
        head = head->next;
        delete tmp;  // 메모리 해제
        return v;
    }

    bool find(int v) {
        for(Node* cur = head; cur; cur = cur->next)
            if(cur->val == v) return true;
        return false;
    }

    void print() {
        for(Node* cur = head; cur; cur = cur->next)
            cout << cur->val << " ";
        cout << "\n";
    }
};

int main(){
    ios::sync_with_stdio(false); cin.tie(nullptr);

    SinglyLinkedList lst;
    lst.push_front(3);
    lst.push_front(2);
    lst.push_front(1);
    lst.print();                 // 1 2 3
    cout << lst.pop_front() << "\n"; // 1
    cout << lst.find(2) << "\n";     // 1
}

```

---

## (5) 자주 하는 실수 포인트 & 최적화 팁

- **C++에서 delete 누락 → 메모리 누수**. → 스마트 포인터(`unique_ptr`) 쓰면 안전.
- **head/tail 갱신 빠뜨림**: 맨 앞/뒤에서 삽입·삭제 시 포인터 업데이트 반드시.
- **빈 리스트 예외 처리** 필요.
- 배열 vs 연결리스트 선택:
    - **조회 많음**: 배열/벡터
    - **삽입/삭제 많음**: 연결리스트
- 실무 C++에선 직접 구현보단 `std::list`, `std::forward_list` 사용.

---

## (6) 어디서 쓰이는가 (대표 활용)

- **큐/스택 구현**
- **LRU 캐시** (이중 연결리스트 + 해시맵)
- **텍스트 편집기** (중간 삽입/삭제)
- **그래프 인접 리스트** 표현
- **실시간 시뮬레이션 객체 관리**
