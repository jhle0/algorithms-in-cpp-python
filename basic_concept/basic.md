---

# 1.1 기초 다지기

---

## 입출력 최적화

- **입출력 최적화**는 알고리즘의 계산 속도보다 **입출력 속도**가 병목이 될 때 사용하는 기법
    - 따라서 버퍼링(Buffering) 기법을 이용해 한 번에 입력/출력을 처리 하는 것이 핵심

### C++

- `ios::sync_with_stdio(false); cin.tie(nullptr);` : C와 C++의 입출력 동기화를 끊고, `cin`/`cout`이 `scanf`/`printf`보다 빠르게 동작
- 대량 입력은 `getline`이나 `cin.read` 등 버퍼 기반 처리 사용
    
    ```jsx
    #include <bits/stdc++.h>
    using namespace std;
    
    int main() {
        ios::sync_with_stdio(false); 
        cin.tie(nullptr); // cin과 cout 묶음 해제
    
        int n;
        cin >> n;
        vector<int> arr(n);
        for (int i = 0; i < n; i++) cin >> arr[i];
    
        for (int x : arr) cout << x << "\n";
        return 0;
    }
    
    ```
    

### Python

- `input()` 대신 `sys.stdin.readline()` 사용 → 줄 단위 버퍼링으로 훨씬 빠름
- 출력도 `print()` 대신 `sys.stdout.write()` 또는 `"\n".join()` 형태로 한 번에 모아 출력
    
    ```jsx
    import sys
    
    n = int(sys.stdin.readline())
    arr = list(map(int, sys.stdin.readline().split()))
    
    sys.stdout.write("\n".join(map(str, arr)))
    ```
    

---

## 시간 공간 복잡도 분석

- **시간 복잡도(Time Complexity)**: 입력 크기(n)에 따라 실행 시간이 어떻게 증가하는가?
- **공간 복잡도(Space Complexity)**: 입력 크기(n)에 따라 추가 메모리를 얼마나 쓰는가?

### 표기법(Big-O, Big-Θ, Big-Ω)

**Big-O (상한, 최악 시간)**

- "최악의 경우 실행 시간이 얼마나 걸릴까?"
- 입력이 커질 때 걸리는 **최대 시간**을 나타냄.
- 코딩테스트, 실무에서 가장 많이 쓰는 표기법.

**Big-Ω (하한, 최선 시간)**

- "최선의 경우 실행 시간이 얼마나 걸릴까?"
- 입력이 아주 유리할 때 최소 시간.

**Big-Θ (평균/정확한 차수)**

- 상한과 하한이 같은 경우 → 평균 시간 복잡도를 나타냄.
- 알고리즘의 "정확한 차수"를 표현.

---

## 기초 수학

### 유클리드 호제법(Euclidean Algorithm)

- **최대공약수(GCD, Greatest Common Divisor)**: 두 수를 나누어 떨어지게 하는 가장 큰 자연수.
    
    예: gcd(12, 18) = 6
    
- **최소공배수(LCM, Least Common Multiple)**: 두 수의 공통 배수 중 가장 작은 수.
    
    예: lcm(12, 18) = 36
    

**원리**

- 두 정수 a, b (a ≥ b)에 대해:
    
    ```
    gcd(a, b) = gcd(b, a mod b)
    ```
    
- b가 0이 되면, 그때의 a가 최대공약수이다.

예시: gcd(48, 18)

```
48 mod 18 = 12
18 mod 12 = 6
12 mod 6 = 0 → gcd = 6
```

**최소공배수와의 관계**

```
lcm(a, b) = (a * b) / gcd(a, b)
```

⇒ **유클리드 호제법의 시간 복잡도: O(log(min(a, b)))**

### C++ (유클리드 호제법)

```cpp
#include <bits/stdc++.h>
using namespace std;

// 최대공약수
long long gcd(long long a, long long b) {
    while (b != 0) {
        long long r = a % b;
        a = b;
        b = r;
    }
    return a;
}

// 최소공배수
long long lcm(long long a, long long b) {
    return a / gcd(a, b) * b; // 오버플로 방지 위해 a/gcd * b
}

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    long long a, b;
    cin >> a >> b;
    cout << gcd(a, b) << "\n" << lcm(a, b) << "\n";
    return 0;
}

```

### Python (재귀 버전)

```python
def gcd(a, b):
    while b:
        a, b = b, a % b
    return a

def lcm(a, b):
    return a * b // gcd(a, b)

a, b = map(int, input().split())
print(gcd(a, b))
print(lcm(a, b))

```

---

### 소수 판별 & 에라토스테네스의 체

- **소수(Prime Number)**: 1보다 크고, 1과 자기 자신으로만 나누어 떨어지는 수.
    
    예: 2, 3, 5, 7, 11, 13, …
    
- **합성수(Composite Number)**: 소수가 아닌 자연수.

**단순 방법 (O(n))**

- 2부터 n-1까지 나눠보며 나누어떨어지는지 검사.
- **비효율적**, n이 10^6 이상이면 시간 초과.

**제곱근 최적화 (O(√n))**

- 어떤 수 n이 소수인지 확인하려면 **2부터 √n까지만 검사**하면 충분하다.
- 이유: n = a × b라면, 최소 하나는 √n 이하이기 때문.

예: n=97 → √97 ≈ 9.8 → 2~9까지만 확인.

**<다수의 소수 찾기 → 에라토스테네스의 체 (Sieve of Eratosthenes)>**

**원리**

1. 2부터 시작, 배수를 모두 지운다.
2. 다음 남은 수(소수)를 찾고, 그 배수를 지운다.
3. √n까지 반복하면, 남은 수가 모두 소수.

예: n=30까지 소수 찾기

- 2 지우기 → 4,6,8,10,…
- 3 지우기 → 6,9,12,15,…
- 5 지우기 → 10,15,20,25,…
    
    → 남은 수: 2,3,5,7,11,13,17,19,23,29
    

**시간·공간 복잡도**

- 시간: **O(n log log n)** (매우 빠름)
- 공간: O(n) (배열 필요)

### C++ (소수 판별 + 체)

```cpp
#include <bits/stdc++.h>
using namespace std;

// 소수 판별 (O(√n))
bool isPrime(long long n) {
    if (n < 2) return false;
    for (long long i = 2; i * i <= n; i++) {
        if (n % i == 0) return false;
    }
    return true;
}

// 에라토스테네스의 체 (O(n log log n))
vector<int> sieve(int n) {
    vector<bool> prime(n+1, true);
    prime[0] = prime[1] = false;
    for (int i = 2; i * i <= n; i++) {
        if (prime[i]) {
            for (int j = i * i; j <= n; j += i)
                prime[j] = false;
        }
    }
    vector<int> primes;
    for (int i = 2; i <= n; i++) if (prime[i]) primes.push_back(i);
    return primes;
}

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    int n;
    cin >> n;
    cout << (isPrime(n) ? "Prime\n" : "Not Prime\n");

    auto primes = sieve(50);
    for (int p : primes) cout << p << " ";
    return 0;
}

```

---

### Python (소수 판별 + 체)

```python
import math

# 소수 판별 (O(√n))
def is_prime(n):
    if n < 2:
        return False
    for i in range(2, int(math.isqrt(n)) + 1):
        if n % i == 0:
            return False
    return True

# 에라토스테네스의 체 (O(n log log n))
def sieve(n):
    prime = [True] * (n+1)
    prime[0] = prime[1] = False
    for i in range(2, int(n**0.5) + 1):
        if prime[i]:
            for j in range(i*i, n+1, i):
                prime[j] = False
    return [i for i in range(2, n+1) if prime[i]]

# 예시 실행
n = int(input())
print("Prime" if is_prime(n) else "Not Prime")

print(sieve(50))
```

---

### 조합론, 팩토리얼, 모듈러 연산

**(1) 팩토리얼 (Factorial)**

- 정의:
    
    ```
    n! = n × (n-1) × (n-2) × ... × 1
    ```
    
- 예시:
    
    ```
    5! = 5 × 4 × 3 × 2 × 1 = 120
    ```
    
- 복잡도: O(n)
- 팩토리얼은 **조합론의 기본**이 된다.

---

**(2) 조합론 (Combination)**

- **순열(Permutation)**: 서로 다른 n개 중 r개를 순서 있게 뽑는 경우의 수
    
    ```
    P(n, r) = n! / (n-r)!
    ```
    
- **조합(Combination)**: 서로 다른 n개 중 r개를 순서 없이 뽑는 경우의 수
    
    ```
    C(n, r) = n! / (r! × (n-r)!)
    ```
    

예시:

- 5명 중 3명을 **순서 없이** 뽑는 경우 → C(5,3) = 10
- 5명 중 3명을 **순서 있게** 뽑는 경우 → P(5,3) = 60

---

**(3) 모듈러 연산 (Modular Arithmetic)**

- 정의:
    
    ```
    (a + b) mod m = ((a mod m) + (b mod m)) mod m
    (a - b) mod m = ((a mod m) - (b mod m) + m) mod m
    (a × b) mod m = ((a mod m) × (b mod m)) mod m
    
    ```
    
- **큰 수 계산**에서 오버플로 방지 & 문제 제한 조건 (예: 결과를 1,000,000,007로 나눈 나머지 출력).

---

**(4) 조합을 모듈러로 계산하기**

코딩테스트에서 자주 나오는 패턴:

- `nCr % MOD` (MOD는 보통 소수 1,000,000,007).

방법:

1. 팩토리얼 미리 계산
2. 페르마의 소정리(Fermat’s little theorem)로 역원(Inverse) 계산

**페르마의 소정리**

- p가 소수일 때:
    
    ```
    a^(p-1) ≡ 1 (mod p)
    a^(p-2) ≡ a^(-1) (mod p)
    
    ```
    
- 즉, `a^(-1) mod p = a^(p-2) mod p`

### C++ (nCr % MOD)

```cpp
#include <bits/stdc++.h>
using namespace std;
const int MOD = 1'000'000'007;
const int MAX = 1'000'000; // n 최대 크기

long long fact[MAX+1], invFact[MAX+1];

// 거듭제곱 (a^b % MOD)
long long modpow(long long a, long long b) {
    long long res = 1;
    while (b > 0) {
        if (b & 1) res = (res * a) % MOD;
        a = (a * a) % MOD;
        b >>= 1;
    }
    return res;
}

// 전처리: 팩토리얼과 역팩토리얼
void init() {
    fact[0] = 1;
    for (int i = 1; i <= MAX; i++) fact[i] = fact[i-1] * i % MOD;
    invFact[MAX] = modpow(fact[MAX], MOD-2); // 페르마 역원
    for (int i = MAX-1; i >= 0; i--) invFact[i] = invFact[i+1] * (i+1) % MOD;
}

long long nCr(int n, int r) {
    if (r < 0 || r > n) return 0;
    return fact[n] * invFact[r] % MOD * invFact[n-r] % MOD;
}

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    init();
    cout << nCr(5, 3) << "\n"; // 10
    cout << nCr(100000, 50000) << "\n"; // 큰 값도 빠르게 계산
}

```

---

### Python (nCr % MOD)

```python
MOD = 10**9 + 7
MAX = 10**6

# 전처리: 팩토리얼과 역팩토리얼
fact = [1] * (MAX+1)
invFact = [1] * (MAX+1)

def modpow(a, b):
    res = 1
    while b > 0:
        if b & 1:
            res = res * a % MOD
        a = a * a % MOD
        b >>= 1
    return res

def init():
    for i in range(1, MAX+1):
        fact[i] = fact[i-1] * i % MOD
    invFact[MAX] = modpow(fact[MAX], MOD-2)
    for i in range(MAX-1, -1, -1):
        invFact[i] = invFact[i+1] * (i+1) % MOD

def nCr(n, r):
    if r < 0 or r > n:
        return 0
    return fact[n] * invFact[r] % MOD * invFact[n-r] % MOD

# 실행
init()
print(nCr(5, 3))       # 10
print(nCr(100000, 50000))  # 큰 nCr도 계산 가능

```
