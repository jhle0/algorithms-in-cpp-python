# 1.2-1. 배열(Array) & 문자열(String)

## (1) 문제 정의

- **배열(Array)**: 같은 타입의 원소를 **연속된 메모리**에 저장하는 자료구조이다. 인덱스로 `O(1)` 임의 접근이 가능하다.
- **문자열(String)**: 문자의 배열이다. 언어마다 가변/불변, 인코딩(ASCII/UTF-8/UTF-16) 차이가 있다.
    - Python: 문자열 **불변(immutable)**, 슬라이싱은 새 객체 생성.
    - C++: `std::string`은 **가변**이며 **연속 메모리**를 보장한다.

## (2) 접근 방법 / 핵심 원리

- **캐시 지역성(Locality)**: 연속 메모리 덕에 반복 순회가 CPU 캐시에 잘 맞아 빠르다.
- **삽입/삭제 비용**: 중간 삽입·삭제는 뒤 원소들을 **밀거나 당기는 비용**으로 `O(n)`이 걸린다.
- **문자열 처리 패턴**
    - 빈도 카운트(해시/배열), **슬라이딩 윈도우**, **투 포인터**, 누적합(접두사 합).
    - 서브스트링 검색은 기본(naive)부터 KMP/Z 알고리즘으로 확장한다(후속 “문자열 알고리즘” 단원에서 심화).

## (3) 복잡도 간단 표

| 연산 | 배열 | 문자열(길이 n) |
| --- | --- | --- |
| 임의 접근 | O(1) | O(1) (인덱스 접근) |
| 탐색(값 위치) | O(n) | O(n) |
| 중간 삽입/삭제 | O(n) | O(n) |
| 앞/뒤 추가 | 뒤 O(1) amortized(`vector`/리스트 append), 앞 O(n) | Python 문자열은 새로 만듦(O(n)) |

> 문자열은 불변 여부에 유의: Python은 +=가 매번 새로 만드는 비용이므로 **리스트에 모아 ''.join()**이 효율적이다.
> 

---

## (4) 구현 코드 & 예제

아래는 현업/코테에서 **자주 쓰는 6가지 기본 루틴**을 엄선하여 Python/C++ 두 언어로 바로 쓸 수 있게 준비하였다.

### A. 배열 ① — **누적합(prefix sum)**: 구간합 다중 질의

- 아이디어: `pref[i] = a[0] + ... + a[i-1] 합`. 구간 `[l, r]` 합은 `pref[r+1]-pref[l]`.

### Python

```python
# (1) 누적합 + 구간합 질의
arr = [3, 1, 4, 1, 5]
pref = [0]
for x in arr:
    pref.append(pref[-1] + x)

def range_sum(l, r):
    return pref[r+1] - pref[l]

print(range_sum(1, 3))  # 1+4+1 = 6

```

### C++

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    vector<int> a = {3, 1, 4, 1, 5};

    // 누적합 배열 준비
    vector<long long> pref(a.size() + 1, 0);
    for (size_t i = 0; i < a.size(); ++i) {
        pref[i + 1] = pref[i] + a[i];
    }

    // 구간합 람다
    auto range_sum = [&](int l, int r) {
        return pref[r + 1] - pref[l];
    };

    cout << range_sum(1, 3) << "\n"; // 6
}

```

---

### A. 배열 ② — **차이배열(Difference Array)**: 여러 번의 구간 증가를 빠르게

- 길이 `n`인 배열 `a`가 있고, 여러 번의 구간 업데이트
    
    `update(l, r, +v)` (0 ≤ l ≤ r < n) 을 수행해야 한다고 하자.
    
- 매번 `for i in [l..r]: a[i]+=v`로 하면 총 시간 `O(∑(r-l+1))`이라 비효율적이다.
- **차이배열(diff)** 기법을 쓰면, 각 업데이트를 `O(1)`로 처리하고, 마지막에 한 번만 `O(n)`으로 복원할 수 있다.
    - `diff[0] = a[0]`
    - `diff[i] = a[i] - a[i-1]` (i ≥ 1)
    
    → 즉, `a`의 **인접 차이**를 저장한 배열이다.
    
- 구간 업데이트 시
    - [l, r] 에 v를 더하고 싶다면
        - `diff[l] += v`
        - **만약** `r + 1 < n` 이면 `diff[r+1] -= v`
    - 누적합 복원 시 결국 [l, r]구간에만 순수하게 `+v` 가 남는다

### Python

```python
# 차이배열 유틸 (0-based)
def build_diff_from_array(a):
    n = len(a)
    diff = [0] * (n + 1)  # 여유칸
    diff[0] = a[0]
    for i in range(1, n):
        diff[i] = a[i] - a[i-1]
    # diff[n]는 0으로 둔다(버퍼)
    return diff

def range_add(diff, l, r, v):
    diff[l] += v
    if r + 1 < len(diff) - 1:  # len(diff)=n+1 이므로 유효 인덱스는 0..n
        diff[r + 1] -= v
    # r==n-1이면 diff[n]에 -v가 들어가지만, 어차피 복원에서 i<n만 쓰면 영향 없음

def rebuild_from_diff(diff):
    n = len(diff) - 1
    a = [0] * n
    cur = 0
    for i in range(n):
        cur += diff[i]
        a[i] = cur
    return a

# 예시 1) 초기배열이 전부 0일 때 
n = 5
diff = [0] * (n + 1)
updates = [(1, 3, 2), (0, 0, 5)]
for l, r, v in updates:
    range_add(diff, l, r, v)
arr = rebuild_from_diff(diff)
print(arr)  # [5, 2, 2, 2, 0]

# 예시 2) 초기배열이 비제로일 때
a0 = [10, 10, 20, 20, 30]
diff2 = build_diff_from_array(a0)
range_add(diff2, 1, 3, +5)   # a[1..3]에 +5
range_add(diff2, 0, 4, -2)   # a 전체 -2
a1 = rebuild_from_diff(diff2)
print(a1)  # [8, 13, 23, 23, 28]

```

### C++

```cpp
#include <bits/stdc++.h>
using namespace std;

// 초기 배열 a로부터 차이배열 생성 (0-based)
vector<long long> build_diff_from_array(const vector<long long>& a) {
    int n = (int)a.size();
    vector<long long> diff(n + 1, 0); // 여유칸
    if (n == 0) return diff;
    diff[0] = a[0];
    for (int i = 1; i < n; ++i) diff[i] = a[i] - a[i-1];
    return diff; // diff[n]은 0
}

// [l, r]에 +v 적용
inline void range_add(vector<long long>& diff, int l, int r, long long v) {
    diff[l] += v;
    if (r + 1 < (int)diff.size() - 1) diff[r + 1] -= v;
    // diff.size()==n+1이므로 r==n-1일 때 r+1==n은 유효 인덱스(버퍼)
    else if (r + 1 == (int)diff.size() - 1) diff[r + 1] -= v; // 명시
}

// diff -> 최종 배열 복원
vector<long long> rebuild_from_diff(const vector<long long>& diff) {
    int n = (int)diff.size() - 1;
    vector<long long> a(n);
    long long cur = 0;
    for (int i = 0; i < n; ++i) {
        cur += diff[i];
        a[i] = cur;
    }
    return a;
}

int main(){
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    // 예시 1) 초기가 전부 0인 경우
    int n = 5;
    vector<long long> diff(n + 1, 0);
    vector<tuple<int,int,long long>> ups = {{1,3,2},{0,0,5}};
    for (auto [l,r,v] : ups) range_add(diff, l, r, v);
    vector<long long> a = rebuild_from_diff(diff);
    for (auto x : a) cout << x << " "; cout << "\n"; // 5 2 2 2 0

    // 예시 2) 초기 배열이 비제로일 때
    vector<long long> a0 = {10, 10, 20, 20, 30};
    auto diff2 = build_diff_from_array(a0);
    range_add(diff2, 1, 3, +5);
    range_add(diff2, 0, 4, -2);
    auto a1 = rebuild_from_diff(diff2);
    for (auto x : a1) cout << x << " "; cout << "\n"; // 8 13 23 23 28
}

```

---

### A. 배열 ③ — **투 포인터**(양의 정수 배열): 합이 `K`인 연속 구간 존재 여부

- 배열 `a[0..n-1]`의 원소가 **모두 0 이상**일 때, 연속 부분배열 `a[L..R]` 중 합이 정확히 `K`인 것이 존재하는지 판별하는 문제

**불변식과 단조성**

- 윈도우 `[L..R]`의 합을 `s`라 두고, 다음 **불변식**을 유지한다:
    
    `s = sum(a[L..R])` 이고, 반복 루프 종료 시점에 항상 `s <= K`가 되도록 만든다.
    
- 모든 원소가 **비음수**이므로,
    - `R`를 오른쪽으로 1칸 늘리면 `s`는 **증가(또는 동일: a[R]=0)**한다.
    - `s > K`라면 `L`을 오른쪽으로 이동하면 `s`는 **감소(또는 동일)**한다.
- 따라서 `L`과 `R`은 **한 번씩만** 오른쪽으로 움직이며, 총 이동은 `O(n)`이다.

**알고리즘(존재 여부)**

1. `L=0, s=0`으로 시작하고 `R=0..n-1` 순서로 확장한다.
2. 매 스텝에서 `s += a[R]` 하고, `s > K` 인 동안 `s -= a[L]; L++`로 줄인다.
3. 언제든 `s == K`가 되면 종료(존재함).

**왜 정확한가 (스케치)**

- `s > K`인 동안 `R`을 더 늘리는 것은 절대 `K`를 만들지 못함(비음수이므로 더 커지기만 함).
    
    → 반드시 `L`을 옮겨 줄여야 한다.
    
- `s < K`일 때 `L`을 줄여봤자 더 작아질 뿐이라 의미가 없음 → `R`을 옮겨야 한다.
- 이 선택은 모순 없이 진행되며, 모든 후보 윈도우를 **중복 없이** 통과한다.

### Python

```python
# 존재 여부 (비어있지 않은 구간)
def has_subarray_sum_k(a, K):
    s = 0
    L = 0
    for R, x in enumerate(a):
        s += x
        while s > K and L <= R:  # 불변식: 루프 끝에 s <= K
            s -= a[L]
            L += 1
        if s == K:
            return True
    return False

print(has_subarray_sum_k([1,2,3,4,5], 9))  # True (2+3+4)

# 처음 발견되는 구간의 인덱스 반환 (못 찾으면 None)
def first_subarray_sum_k(a, K):
    s = 0
    L = 0
    for R, x in enumerate(a):
        s += x
        while s > K and L <= R:
            s -= a[L]
            L += 1
        if s == K:
            return (L, R)
    return None

print(first_subarray_sum_k([1,2,3,4,5], 9))  # (1,3)

# <= K 인 부분배열 개수 새기
def count_subarrays_at_most_k(a, K):
    s = 0; L = 0; ans = 0
    for R, x in enumerate(a):
        s += x
        while s > K and L <= R:
            s -= a[L]
            L += 1
        ans += (R - L + 1)  # 이 R에서 가능한 시작점 개수
    return ans

print(count_subarrays_at_most_k([1,2,1,1], 3))  # 7

```

### C++

```cpp
#include <bits/stdc++.h>
using namespace std;

// 존재 여부
bool has_subarray_sum_k(const vector<long long>& a, long long K){
    long long s = 0;
    int L = 0;
    for(int R=0; R<(int)a.size(); ++R){
        s += a[R];
        while(s > K && L <= R){
            s -= a[L];
            ++L;
        }
        if(s == K) return true;
    }
    return false;
}

// 처음 발견되는 구간의 [L,R] 반환 (없으면 {-1,-1})
pair<int,int> first_subarray_sum_k(const vector<long long>& a, long long K){
    long long s = 0;
    int L = 0;
    for(int R=0; R<(int)a.size(); ++R){
        s += a[R];
        while(s > K && L <= R){
            s -= a[L];
            ++L;
        }
        if(s == K) return {L, R};
    }
    return {-1, -1};
}

int main(){
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    vector<long long> v = {1,2,3,4,5};
    cout << has_subarray_sum_k(v, 9) << "\n";          // 1
    auto p = first_subarray_sum_k(v, 9);
    cout << p.first << " " << p.second << "\n";        // 1 3
}

// <= K 인 부분배열 개수 새기
long long count_subarrays_at_most_k(const vector<long long>& a, long long K){
    long long s=0, ans=0;
    int L=0;
    for(int R=0; R<(int)a.size(); ++R){
        s += a[R];
        while(s > K && L <= R){
            s -= a[L]; ++L;
        }
        ans += (R - L + 1);
    }
    return ans;
}

```

---

### A. 배열 ④ — **최대 부분합(Kadane)**: 연속 부분배열 최대합 `O(n)`

길이 `n`의 배열 `a`가 주어질 때, **비어있지 않은 연속 부분배열** `a[L..R]`의 합이 최대가 되는 값을 구한다.

- 입력 예: `[-2, 1, -3, 4, -1, 2, 1, -5, 4]`
- 정답: `6` (`[4, -1, 2, 1]` 구간)

**<접근 방법 / 아이디어>**

**DP 점화식 (Kadane의 핵심)**

`dp[i] = i에서 끝나는 최대 부분합`이라 하면,

```
dp[i] = max(a[i], dp[i-1] + a[i])
```

- 직관: “이전까지의 최적 구간에 `a[i]`를 붙일지, 여기서 새로 시작할지”를 매 칸마다 결정.
- 구현은 `cur`(현재 위치에서 끝나는 최대합)와 `best`(전체 최대합)만 유지하면 된다 → `O(1)` 추가 메모리.

**왜 되는가 (증명 스케치)**

- `cur < 0`이면, 그 뒤에 어떤 양수를 더하더라도 **손해**이므로 다음 원소에서 “새로 시작”하는 것이 최적.
- 이 그리디 결정을 매 칸에서 반복하면 전체 최적해에 도달한다.

### Python

```python
def kadane_sum(a):
    # 비어있지 않은 구간 가정
    best = float("-inf")
    cur = 0
    for x in a:
        cur = max(x, cur + x)
        best = max(best, cur)
    return best

def kadane_sum_with_indices(a):
    # 최대합과 [L,R]을 함께 반환
    best = float("-inf")
    cur = 0
    bestL = bestR = 0
    L = 0  # 현재 구간의 시작 후보
    for i, x in enumerate(a):
        # cur+x vs x 비교로 '새로 시작' 판단
        if cur + x < x:
            cur = x
            L = i
        else:
            cur += x
        # 최적 갱신 시 인덱스 기록
        if cur > best:
            best = cur
            bestL, bestR = L, i
    return best, (bestL, bestR)

print(kadane_sum([-2,1,-3,4,-1,2,1,-5,4]))              # 6
print(kadane_sum_with_indices([-2,1,-3,4,-1,2,1,-5,4]))  # (6, (3,6))

```

### C++

```cpp
#include <bits/stdc++.h>
using namespace std;

// 최대합만
long long kadane_sum(const vector<long long>& a){
    long long best = LLONG_MIN, cur = 0;
    for(long long x: a){
        cur = max(x, cur + x);
        best = max(best, cur);
    }
    return best;
}

// 최대합 + [L,R] 인덱스 복원
tuple<long long,int,int> kadane_sum_with_indices(const vector<long long>& a){
    long long best = LLONG_MIN, cur = 0;
    int bestL = 0, bestR = 0, L = 0;
    for(int i=0; i<(int)a.size(); ++i){
        long long x = a[i];
        if(cur + x < x){ cur = x; L = i; }
        else{ cur += x; }
        if(cur > best){ best = cur; bestL = L; bestR = i; }
    }
    return {best, bestL, bestR};
}

int main(){
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    vector<long long> v = {-2,1,-3,4,-1,2,1,-5,4};
    cout << kadane_sum(v) << "\n"; // 6

    auto [best, L, R] = kadane_sum_with_indices(v);
    cout << best << " " << L << " " << R << "\n"; // 6 3 6
}

```

---

### B. 문자열 ① — **아나그램 판별 / 빈도 카운트**

- 두 문자열 `a`, `b`가 **아나그램(anagram)**인지, 즉 문자의 **종류별 개수**가 완전히 동일한지 판별하는 문제

**<접근 방법 / 아이디어>**

**A. 빈도 카운트(권장) — `O(n)`**

- 아이디어: 각 문자 빈도를 세서 두 빈도표가 같으면 아나그램이다.
- 구현:
    - **알파벳만**: 크기가 작은 배열(예: 26)을 쓰면 매우 빠름.
    - **ASCII**: 배열 256.
    - **유니코드 전반**: 해시맵(`dict`/`unordered_map<char32_t,int>`).
- 추가 옵션(전처리):
    - `lower()`로 **대소문자 통일**
    - 공백/구두점 **제거**
    - **유니코드 정규화(NFC/NFKC)**로 같은 글자 다른 표기 통일

**B. 정렬 비교 — `O(n log n)`**

- 두 문자열을 정렬해 **정렬 결과가 같으면** 아나그램.
- 구현이 단순하지만 시간복잡도에서 카운팅보다 불리하다.

**C. 1-pass 카운팅(조기 실패)**

- `a`에서 +1, `b`에서 -1 하면서 내려가다 **음수**가 나오면 즉시 실패.
- 최종적으로 모두 0이면 성공. (C++ 예시에서 이 방식 사용)

### Python

```python
from collections import Counter

def is_anagram(a: str, b: str) -> bool:
    if len(a) != len(b):
        return False
    return Counter(a) == Counter(b)

print(is_anagram("listen","silent"))  # True
```

### C++ (ASCII 가정)

```cpp
#include <bits/stdc++.h>
using namespace std;

bool is_anagram_ascii(const string& a, const string& b){
    if(a.size() != b.size()) return false;
    array<int,256> f{}; // 0으로 초기화
    for(unsigned char c: a) f[c]++;
    for(unsigned char c: b){
        if(--f[c] < 0) return false; // 조기 실패
    }
    return true; // 모두 0
}

int main(){
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    cout << is_anagram_ascii("listen","silent") << "\n"; // 1
}

```

> 유니코드(한글 등)를 정확히 세려면 Python은 Counter로도 충분하나, C++은 입력을 char32_t로 파싱해 unordered_map<char32_t,int>를 쓰는 것이 안전하다.
> 

---

### B. 문자열 ② — **가장 긴 중복 없는 부분문자열**(슬라이딩 윈도우)

- 문자열 `s`가 주어졌을 때, **서로 다른 문자로만 이루어진 가장 긴 연속 부분 문자열의 길이**를 구하는 문제.
- 예시:
    - `"abcabcbb"` → 답: `3` (`"abc"`)
    - `"bbbbb"` → 답: `1` (`"b"`)
    - `"pwwkew"` → 답: `3` (`"wke"`)

**<접근 방법 / 아이디어>**

**단순한 방법 (Brute force) → O(n²) 이상**

- 모든 부분 문자열을 확인하고 중복 없는지 검사.
- 비효율적.

**슬라이딩 윈도우 + 해시 (O(n))**

핵심 아이디어:

1. `L`(왼쪽 포인터)와 `R`(오른쪽 포인터)로 윈도우 `[L..R]` 유지.
2. 각 문자의 **최근 등장 위치(last occurrence)**를 해시맵(딕셔너리/배열)에 기록.
3. 새 문자가 이전에 윈도우 안에 있었다면, **중복 제거를 위해 `L`을 옮긴다**.
    - 구체적으로: `L = max(L, last[ch]+1)`
4. 매번 `best = max(best, R-L+1)` 업데이트.

→ 결과: `O(n)`에 해결 가능.

### Python

```python
def longest_unique_substr(s: str) -> int:
    last = {}   # 각 문자 마지막 위치
    L = 0       # 윈도우 시작 인덱스
    best = 0
    for R, ch in enumerate(s):
        if ch in last and last[ch] >= L:
            # 중복 발견 → 시작점 이동
            L = last[ch] + 1
        last[ch] = R
        best = max(best, R - L + 1)
    return best

print(longest_unique_substr("abcabcbb"))  # 3
print(longest_unique_substr("bbbbb"))     # 1
print(longest_unique_substr("pwwkew"))    # 3

```

### C++

```cpp
#include <bits/stdc++.h>
using namespace std;

int longest_unique_substr(const string& s){
    vector<int> last(256, -1); // 문자 → 마지막 등장 인덱스
    int L = 0, best = 0;
    for(int R=0; R<(int)s.size(); ++R){
        unsigned char c = s[R];
        if(last[c] >= L) {
            L = last[c] + 1; // 중복 제거
        }
        last[c] = R;
        best = max(best, R - L + 1);
    }
    return best;
}

int main(){
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    cout << longest_unique_substr("abcabcbb") << "\n"; // 3
    cout << longest_unique_substr("bbbbb") << "\n";    // 1
    cout << longest_unique_substr("pwwkew") << "\n";   // 3
}

```

---

### B. 문자열 ③ — **팰린드롬 검사 & 중앙 확장으로 최장 팰린드롬 길이**

- **팰린드롬 검사**: 문자열 `s`가 앞뒤 대칭인지 판정.
- **최장 팰린드롬 부분문자열 길이**: `s`의 연속 부분문자열 중 팰린드롬인 것의 **최대 길이**(또는 그 **구간/문자열**)을 구한다.

**<아이디어 / 원리>**

**A. 팰린드롬 검사(two-pointer)**

- 양끝에서 중앙으로 오면서 같은지 비교.
- Python에선 `s == s[::-1]`(뒤집기)도 가능(시간 `O(n)` 복사 포함).

**B. 중앙 확장(Expand Around Center) — `O(n^2)`**

- 팰린드롬은 **중심(center)** 기준으로 좌우가 대칭.
- 중심은 두 종류:
    1. 홀수 길이: 중심이 문자 하나 `(i,i)`
    2. 짝수 길이: 중심이 문자 사이 `(i,i+1)`
- 각 중심에서 **좌우로 확장**하며 같은 동안 넓힌다 → 해당 중심의 최장 팰린드롬 얻음.
- 모든 중심(대략 `2n-1`개)을 시도하면 전체 최적을 얻는다.

**왜 맞는가 (스케치)**

- 임의의 최장 팰린드롬에는 어떤 **중심**이 존재.
- 그 중심에서 확장하면 정확히 그 팰린드롬 길이에 도달.
- 모든 중심을 검사하므로 놓치지 않는다.

### Python

```python
# 1) 팰린드롬 검사
def is_pal(s: str) -> bool:
    # 양끝 투포인터로도 가능하지만, 파이썬은 슬라이싱이 간결
    return s == s[::-1]

# 2) 최장 팰린드롬 길이 (중앙 확장)
def longest_pal_len(s: str) -> int:
    def expand(l: int, r: int) -> int:
        while l >= 0 and r < len(s) and s[l] == s[r]:
            l -= 1; r += 1
        return r - l - 1  # 마지막으로 일치했던 구간 길이
    best = 0
    for i in range(len(s)):
        best = max(best, expand(i, i), expand(i, i+1))
    return best

# 3) 최장 팰린드롬 "구간/문자열" 복원
def longest_pal_substring(s: str):
    if not s:
        return "", (0, -1)
    def expand(l: int, r: int) -> (int, int):
        while l >= 0 and r < len(s) and s[l] == s[r]:
            l -= 1; r += 1
        # 반환은 [l+1, r-1] 구간
        return l + 1, r - 1
    bestL = bestR = 0
    for i in range(len(s)):
        l1, r1 = expand(i, i)
        l2, r2 = expand(i, i + 1)
        if r1 - l1 > bestR - bestL: bestL, bestR = l1, r1
        if r2 - l2 > bestR - bestL: bestL, bestR = l2, r2
    return s[bestL:bestR+1], (bestL, bestR)

# 간단 테스트
print(is_pal("abacaba"))                      # True
print(longest_pal_len("babad"))               # 3  ("bab"/"aba")
print(longest_pal_substring("babad"))         # ("bab", (0,2)) 혹은 ("aba", (1,3))

```

### C++

```cpp
#include <bits/stdc++.h>
using namespace std;

// 1) 팰린드롬 검사 (반쪽만 비교)
bool is_pal(const string& s){
    return equal(s.begin(), s.begin() + (int)s.size()/2, s.rbegin());
}

// 2) 최장 팰린드롬 길이 (중앙 확장)
int longest_pal_len(const string& s){
    auto expand = [&](int l, int r){
        while(l >= 0 && r < (int)s.size() && s[l] == s[r]){ --l; ++r; }
        return r - l - 1; // 마지막 일치 길이
    };
    int best = 0;
    for(int i = 0; i < (int)s.size(); ++i){
        best = max(best, expand(i, i));
        best = max(best, expand(i, i + 1));
    }
    return best;
}

// 3) 최장 팰린드롬 "구간/문자열" 복원
pair<int,int> longest_pal_interval(const string& s){
    if(s.empty()) return {0,-1};
    auto expand_idx = [&](int l, int r){
        while(l >= 0 && r < (int)s.size() && s[l] == s[r]){ --l; ++r; }
        return pair<int,int>{l+1, r-1}; // 포함 구간 [L,R]
    };
    int bestL = 0, bestR = 0;
    for(int i = 0; i < (int)s.size(); ++i){
        auto [l1, r1] = expand_idx(i, i);       // 홀수 중심
        if(r1 - l1 > bestR - bestL) bestL = l1, bestR = r1;
        auto [l2, r2] = expand_idx(i, i + 1);   // 짝수 중심
        if(r2 - l2 > bestR - bestL) bestL = l2, bestR = r2;
    }
    return {bestL, bestR};
}

int main(){
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    cout << is_pal("abacaba") << "\n";           // 1
    cout << longest_pal_len("babad") << "\n";    // 3

    auto [L, R] = longest_pal_interval("babad");
    cout << L << " " << R << "\n";               // 0 2 (또는 1 3)
    // 필요 시 부분문자열 출력:
    string s = "babad";
    cout << s.substr(L, R-L+1) << "\n";
}

```

---

## 자주 하는 실수 포인트 & 최적화 팁

- **Python 문자열 연결**: `s += piece` 반복은 비효율 → 리스트에 모았다가 `''.join(pieces)` 사용.
- **인덱스 범위**: 슬라이싱/윈도우에서 `while dq and dq[0] <= i-k` 같은 **경계 조건**을 특히 주의.
- **음수 포함 배열 + 투 포인터**: 양수 전용 기법은 실패한다. 이런 경우 누적합+해시(부분합)나 Kadane로 접근.
- **C++ `vector` 재할당**: 다량 푸시가 예상되면 `v.reserve(n)`으로 재할당 최소화.
- **UTF-8 길이**: Python `len(s)`는 문자 개수이지만, C++의 `std::string::size()`는 **바이트 수**이다(UTF-8 멀티바이트 주의).
- **복잡도 착시**: Python 슬라이싱 `s[i:j]`는 `O(j-i)` 복사이다. 큰 문자열에서 과도한 슬라이스 생성 지양.

## 어디서 쓰이는가 (대표 활용)

- **로그/센서 전처리**:
    - 빠른 구간 합/차이배열로 **대량 업데이트**를 일괄 반영.
- **텍스트 분석**:
    - 빈도 카운트(아나그램, 토큰 통계),
    - 슬라이딩 윈도우(중복 없는 부분문자열, 고정/가변 길이 윈도우 통계).
- **스트림 처리**:
    - 투 포인터로 실시간 **조건 만족 윈도우** 유지(모두 비음수일 때).
- **금융·게임 이벤트 집계**:
    - 차이배열로 **구간 보상/패치** 즉시 반영 후 최종 한 번 복원.
- **시계열 이상 탐지**:
    - Kadane(최대 부분합)으로 “가장 뜨거운 구간” 또는 변형으로 **최저 합(이상 구간)** 탐색.
- **검색/UX**:
    - 최장 중복 없는 부분문자열 → 키 입력/세션에서 **중복 없는 최대 구간** 판단,
    - 팰린드롬 탐색 → 문자열 패턴 분석/퍼즐/QA 테스트 케이스 생성.
