## DSL 작성 가이드(Authoring Guide)

Web page DSL은 **“페이지 구조 → 컴포넌트 배치 → 상태/행동 → 렌더링 → UI 이펙트”**까지를 하나의 스펙으로 기술하는 **Page Generation System DSL**입니다. 

---

## 1. 최상위 단위: `system`

```dsl
system DateApp { ... }
```

### 목적

* 하나의 **웹 페이지 생성 시스템 전체를 기술**
* build target: `index.html + JS bundle (Lit components)`

### 포함 가능 섹션

| 섹션          | 의미                        |
| ----------- | ------------------------- |
| `page`      | 실제 HTML 문서 구조             |
| `component` | Lit Component로 컴파일될 UI 모듈 |
| `utils`     | 공용 순수 함수 (JS helper)      |

### 설계 원칙

* system = 배포 단위
* 컴포넌트 재사용은 system 내부 기준

---

## 2. HTML 진입점: `page`

```dsl
page index { ... }
```

### 목적

* 실제 **index.html 구조를 DSL에서 먼저 선언**
* SPA 프레임워크가 아니라 **문서 중심 설계**

---

### 2.1 head 영역

```dsl
head:
  style body { font-family: sans-serif }
```

#### 의미

→ HTML:

```html
<style>
  body { font-family: sans-serif; }
</style>
```

#### 가이드

* global CSS만 허용
* component scope style은 component 내부에서만 정의

---

### 2.2 body 영역

```dsl
body:
  MyElement
```

#### 의미

→ HTML:

```html
<my-element></my-element>
```

#### 가이드

* body에는 반드시 component 인스턴스만 배치
* text / raw HTML은 component로 감싸서 표현

즉,

> page = document layout
> component = UI logic unit

---

## 3. UI 단위: `component`

```dsl
component MyElement { ... }
```

### 목적

* LitElement 1:1 대응
* Custom Element 정의 단위

---

## 3.1 State 선언

```dsl
state date : Date
```

### 의미

→ Lit:

```js
static properties = { date: {} }
```

### 가이드

* 모든 reactive data는 반드시 state로 선언
* 외부에서 bind 가능한 인터페이스 역할

---

## 3.2 Template (HTML 구조)

```dsl
template:
  p { "Choose a date:" input type="date" on change -> dateChanged }
```

### 의미 계층

| DSL                    | 대응                    |
| ---------------------- | --------------------- |
| `p { ... }`            | `<p>...</p>`          |
| `"text"`               | text node             |
| `input type="date"`    | `<input type="date">` |
| `on change -> handler` | `@change=${handler}`  |

### 가이드

* HTML5 semantic tag 그대로 사용
* 이벤트는 반드시 transition으로 연결
* DOM 직접 조작 금지

---

## 3.3 Transition (행동 규칙)

```dsl
transition dateChanged(e):
  ...
```

### 의미

→ Lit method:

```js
dateChanged(e) { ... }
```

### 역할

* event → state mutation only
* DOM access 최소화 (value extraction 정도만 허용)

### 설계 원칙

> transition은 reducer + controller 역할

즉,

* side effect 가능
* 렌더링 직접 호출 불가

---

## 3.4 Utils 호출

```dsl
date = localDateFromUTC(utc)
```

### 가이드

* utils는 반드시 pure function
* component logic과 분리 유지

---

## 4. Component 간 연결: `bind`

```dsl
DateDisplay bind date
```

### 의미

→ Lit:

```html
<date-display .date=${this.date}></date-display>
```

### 가이드

* 단방향 데이터 흐름
* child는 parent state를 직접 변경하지 않음

---

# 5. Rendering Constraint: `invariant`

```dsl
invariant:
  renderOnlyIf isSameDate(old.date, date) == false
```

### 의미

→ Lit:

```js
hasChanged: (v, o) => !isSameDate(v, o)
```

### 목적

* 불필요한 re-render 방지
* domain-level equality 정의

### 가이드

* wrapper type (Date, Object)는 반드시 invariant 정의 권장

---

## 6. Reactive Effect: `on update(state)`

```dsl
on update(date):
  animate #datefield ...
```

### 의미

→ Lit lifecycle:

```js
updated(changed) {
  if (changed.has('date')) { ... }
}
```

### 목적

* 상태 변화 → UI effect
* imperative DOM API는 여기에서만 허용

### 설계 규칙

| 위치         | 허용 작업              |
| ---------- | ------------------ |
| transition | state 변경           |
| on update  | DOM effect         |
| template   | DOM structure only |

이 분리가 매우 중요합니다.

---

## 7. Utils 블록

```dsl
utils { ... }
```

### 목적

* 프레임워크 독립 domain logic
* 테스트 가능한 순수 함수 집합

### 컴파일 시

→ 별도 JS module 또는 inline function

---

# 전체 DSL 구조 요약 (Authoring Checklist)

## 작성 순서 권장

1. system 이름 정의
2. page에서 HTML 구조 먼저 기술
3. page에 배치될 component 정의
4. component state 정의
5. template 작성
6. transition 정의
7. invariant / on update 추가
8. utils 분리

---

## 역할 분리 원칙 (가장 중요)

| 영역         | 책임            |
| ---------- | ------------- |
| page       | 문서 구조         |
| template   | DOM 구조        |
| state      | reactive data |
| transition | event → state |
| invariant  | update 조건     |
| on update  | DOM effect    |
| utils      | 순수 계산         |

---

# 이 DSL의 설계 철학 한 줄 요약

> HTML 문서 구조를 먼저 고정하고,
> 그 안에 reactive component를 배치하며,
> 상태 변화만으로 UI가 진화하도록 강제하는 시스템 DSL

이는 전형적인 SPA 프레임워크와 달리:

* router 중심 아님
* app root 중심 아님
* document 중심 구조

라는 점에서 **Web-native + Component-reactive 혼합 모델**입니다.

---


예제 DateApp.dsl:
```code
system DateApp { 

  page index {
    head:
      style body { font-family: sans-serif }
    body:
      MyElement
  }
  
  component MyElement {

    state date : Date

    template:
      p { "Choose a date:" input type="date" on change -> dateChanged }
      p { button on click -> chooseToday { "Choose Today" } }
      p { "Date chosen:" DateDisplay bind date }

    transition dateChanged(e):
      let utc = e.target.valueAsDate
      if utc != null:
        date = localDateFromUTC(utc)

    transition chooseToday:
      date = now()
  }

  utils {
    function isSameDate(d1, d2) =
      d1?.toLocaleDateString() == d2?.toLocaleDateString()

    function localDateFromUTC(utc) =
      Date(utc.UTCYear, utc.UTCMonth, utc.UTCDay)
  }

  component DateDisplay {

    state date : Date

    invariant:
      renderOnlyIf isSameDate(old.date, date) == false

    template:
      span #datefield { text date.toLocaleDateString() }

    on update(date):
      animate #datefield frames [
        {bg:"#fff"}, {bg:"#324fff"}, {bg:"#fff"}
      ] duration 1000
  }
}
```