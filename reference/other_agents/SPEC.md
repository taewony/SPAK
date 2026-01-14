# whenwords Specification v0.1.0

## Overview

whenwords is a library for human-friendly time formatting and parsing. It converts timestamps to readable strings like "3 hours ago" and parses duration strings like "2h 30m" into seconds.

All functions are pure—no side effects, no I/O, no system clock access. The reference timestamp is always passed explicitly.

## Design Principles

1. **Pure functions only.** No side effects, no system clock access, no I/O. The current time is always passed explicitly.

2. **Timestamps are Unix seconds.** Internally, all functions work with Unix timestamps (seconds since 1970-01-01 UTC). Implementations should also accept language-native datetime types for convenience.

3. **Strings are UTF-8.** All string inputs and outputs are UTF-8 encoded.

4. **English only (v0.1).** This version outputs English strings only. Locale support may be added in future versions. Do not implement i18n unless the spec explicitly defines it.

5. **Deterministic.** Given the same inputs, functions always return the same output. No randomness, no environment-dependent behavior.

---

## Output Structure

Generate the minimal files needed to use and test the library. Do not create package distribution scaffolding.

**Do generate:**
- Library source file(s)
- Test file(s)
- usage.md

**Do not generate:**
- setup.py, pyproject.toml with build/publish metadata (Python)
- Publishable Cargo.toml fields like description, license, repository, keywords (Rust—keep only `[package]` name, version, edition)
- package.json with publish config (Node)
- gemspec files (Ruby)
- go.mod with module paths pointing to repositories (Go—use a simple local module name)
- Any CI/CD configuration, GitHub Actions, etc.

The goal is a working implementation that can be copied into a project, not a publishable package.

---

## Type Conventions

Since this spec targets multiple languages, types are described abstractly:

| Spec type | Meaning | Examples |
|-----------|---------|----------|
| `timestamp` | Unix seconds (integer or float) OR ISO 8601 string OR language-native datetime | `1704067200`, `"2024-01-01T00:00:00Z"`, `Date`, `datetime` |
| `number` | Integer or float as appropriate | `3600`, `3600.5` |
| `string` | UTF-8 text | `"2 hours ago"` |
| `options` | Language-idiomatic options object | `{compact: true}`, `Options { compact: true }` |
| `error` | Language-idiomatic error | `ValueError`, `Err(...)`, `null`, `throw` |

### Timestamp normalization

When a function receives a `timestamp`:
1. If integer/float: treat as Unix seconds
2. If ISO 8601 string: parse to Unix seconds (error if invalid)
3. If language-native datetime: convert to Unix seconds

Implementations may accept milliseconds if clearly documented, but the spec test cases use seconds.

---

## Error Handling

Errors should be reported idiomatically for the target language:

| Language | Error style |
|----------|-------------|
| Python | Raise `ValueError` with descriptive message |
| TypeScript | Throw `Error` or return `null` (document which) |
| Rust | Return `Result<T, ParseError>` |
| Go | Return `(value, error)` tuple |
| Java | Throw `IllegalArgumentException` |

**Error conditions by function:**

| Function | Error when |
|----------|------------|
| `timeago` | Invalid timestamp format |
| `duration` | Negative seconds, NaN, infinite |
| `parse_duration` | Empty string, unparseable input, negative result |
| `human_date` | Invalid timestamp format |
| `date_range` | Invalid timestamp format |

When in doubt, be liberal in inputs (accept reasonable variations) and strict in outputs (always return spec-compliant strings).

---

## Timezone Handling

**For relative functions (`timeago`, `duration`, `parse_duration`):** Timezones don't matter. These operate on durations between timestamps.

**For calendar functions (`human_date`, `date_range`):**
- Timestamps are instants in time (UTC)
- The output depends on which calendar day that instant falls on
- By default, interpret timestamps in **UTC**
- Implementations MAY add an optional `timezone` parameter
- If timezone support is added, use IANA timezone names (`America/New_York`)

The spec tests assume UTC. Timezone-aware implementations must still pass all spec tests when using UTC.

---

## Rounding and Boundaries

### timeago thresholds

Thresholds are evaluated with `>=` on the lower bound:

```
0 <= diff < 45 seconds     → "just now"
45 <= diff < 90 seconds    → "1 minute ago"
90 seconds <= diff < 45 min → "{n} minutes ago"  (rounded)
...
```

When calculating `n`, round to nearest integer. Use half-up rounding (2.5 → 3, 2.4 → 2).

### duration rounding

When `max_units` truncates output, round the smallest displayed unit:
- `duration(3659)` with default max_units=2 → "1 hour" (59 seconds rounds down)
- `duration(3690)` with max_units=1 → "1 hour" (90 seconds = 1.5 min, rounds to 2, but we're only showing hours which rounds to 1)

Rounding applies to the *display*, not to intermediate calculations.

### Pluralization

- 1 of any unit: singular ("1 minute", "1 hour", "1 day")
- 0 or 2+ of any unit: plural ("0 seconds", "2 minutes", "5 hours")

---

## Functions

### timeago(timestamp, reference?) → string

Returns a human-readable relative time string.

**Arguments:**
- `timestamp`: Unix timestamp (seconds) or ISO 8601 string
- `reference`: Optional. Defaults to `timestamp` if omitted (returns "just now"). In real usage, callers pass current time.

**Behavior:**

| Condition | Output |
|-----------|--------|
| 0–44 seconds | "just now" |
| 45–89 seconds | "1 minute ago" |
| 90 seconds – 44 minutes | "{n} minutes ago" |
| 45–89 minutes | "1 hour ago" |
| 90 minutes – 21 hours | "{n} hours ago" |
| 22–35 hours | "1 day ago" |
| 36 hours – 25 days | "{n} days ago" |
| 26–45 days | "1 month ago" |
| 46 days – 319 days | "{n} months ago" |
| 320–547 days | "1 year ago" |
| 548+ days | "{n} years ago" |

Future times use "in {n} {units}" instead of "{n} {units} ago".

**Rationale:** Thresholds are chosen so the output never feels wrong. "2 days ago" should never describe something 47 hours old (feels like yesterday). The 45-second "just now" window prevents jittery UIs showing "1 second ago".

**Edge cases:**
- Identical timestamps → "just now"
- Negative differences (future) → "in 3 hours"
- Very large values → cap at years, no overflow

---

### duration(seconds, options?) → string

Formats a duration (not relative to now).

**Arguments:**
- `seconds`: Non-negative number
- `options`: Object with optional fields:
  - `compact`: boolean (default false). If true, use "2h 34m" style.
  - `max_units`: integer (default 2). Maximum units to show.

**Behavior:**
- Units: years (365d), months (30d), days, hours, minutes, seconds
- Only shows non-zero units
- Rounds smallest displayed unit

**Examples:**
- `duration(3661)` → "1 hour, 1 minute"
- `duration(3661, {compact: true})` → "1h 1m"
- `duration(3661, {max_units: 1})` → "1 hour"
- `duration(45)` → "45 seconds"
- `duration(0)` → "0 seconds"

---

### parse_duration(string) → number | error

Parses a human-written duration string into seconds.

**Accepted formats:**
- Compact: "2h30m", "2h 30m", "2h, 30m"
- Verbose: "2 hours 30 minutes", "2 hours and 30 minutes"
- Decimal: "2.5 hours", "1.5h"
- Single unit: "90 minutes", "90m", "90min"
- Colon notation: "2:30" (interpreted as h:mm), "2:30:00" (h:mm:ss)

**Unit aliases:**
- seconds: s, sec, secs, second, seconds
- minutes: m, min, mins, minute, minutes
- hours: h, hr, hrs, hour, hours
- days: d, day, days
- weeks: w, wk, wks, week, weeks

**Error conditions:**
- Empty string
- No parseable units
- Negative values

**Rationale:** Be liberal in what you accept. Users type durations in many ways.

---

### human_date(timestamp, reference?) → string

Returns a contextual date string.

**Arguments:**
- `timestamp`: The date to format
- `reference`: The "current" date for comparison

**Behavior:**

| Condition | Output |
|-----------|--------|
| Same day | "Today" |
| Previous day | "Yesterday" |
| Next day | "Tomorrow" |
| Within past 7 days | "Last {weekday}" |
| Within next 7 days | "This {weekday}" |
| Same year | "{Month} {day}" |
| Different year | "{Month} {day}, {year}" |

---

### date_range(start, end) → string

Formats a date range with smart abbreviation.

**Arguments:**
- `start`: Start timestamp
- `end`: End timestamp

**Behavior:**
- Same day: "March 5, 2024"
- Same month: "March 5–7, 2024"
- Same year: "March 5 – April 7, 2024"
- Different years: "December 28, 2024 – January 3, 2025"

**Edge cases:**
- `start` equals `end`: treat as single day
- `start` after `end`: swap them silently

---

## Testing

### Test data format

Tests are defined in `tests.yaml` as language-agnostic input/output pairs.

Structure:
```yaml
function_name:
  - name: "human-readable test name"
    input: { ... }        # Function arguments
    output: "expected"    # Expected return value
    error: true           # Present only if function should error
```

### Using tests.yaml

Implementations MUST pass all tests.yaml test cases. The workflow:

1. **Parse tests.yaml** in your target language
2. **Generate or write test cases** that:
   - Call the function with `input` arguments
   - Assert the return value equals `output`
   - If `error: true`, assert the function raises/returns an error
3. **Run tests** and iterate until all pass

### Input field mapping

Each function has specific input fields:

**timeago:**
```yaml
input: { timestamp: <number>, reference: <number> }
```

**duration:**
```yaml
input: { seconds: <number>, options?: { compact?: bool, max_units?: int } }
```

**parse_duration:**
```yaml
input: "<string>"  # Direct string input, not an object
```

**human_date:**
```yaml
input: { timestamp: <number>, reference: <number> }
```

**date_range:**
```yaml
input: { start: <number>, end: <number> }
```

### Test generation example

Given this tests.yaml entry:
```yaml
timeago:
  - name: "2 minutes ago - 90 seconds"
    input: { timestamp: 1704067110, reference: 1704067200 }
    output: "2 minutes ago"
```

Generate (Python):
```python
def test_timeago_2_minutes_ago_90_seconds():
    result = timeago(1704067110, reference=1704067200)
    assert result == "2 minutes ago"
```

Generate (TypeScript):
```typescript
test('timeago: 2 minutes ago - 90 seconds', () => {
  expect(timeago(1704067110, 1704067200)).toBe('2 minutes ago');
});
```

Generate (Rust):
```rust
#[test]
fn test_timeago_2_minutes_ago_90_seconds() {
    assert_eq!(timeago(1704067110, 1704067200), "2 minutes ago");
}
```

### Error test handling

For entries with `error: true`:

```yaml
parse_duration:
  - name: "error - empty string"
    input: ""
    error: true
```

Generate (Python):
```python
def test_parse_duration_error_empty_string():
    with pytest.raises(ValueError):
        parse_duration("")
```

Generate (TypeScript):
```typescript
test('parse_duration: error - empty string', () => {
  expect(() => parse_duration("")).toThrow();
});
```

### Additional tests

Implementations MAY include additional tests beyond tests.yaml, but:
- All tests.yaml tests MUST pass unchanged
- Additional tests must not contradict spec behavior
- Edge cases not covered by tests.yaml are implementation-defined

---

## Generated Documentation

Implementations MUST include a `usage.md` file documenting how to use the library in the target language.

### usage.md requirements

The file should be concise and practical. Include:

1. **Installation** — How to add the library to a project (import path, package name, etc.)

2. **Quick start** — Minimal code example showing basic usage of each function

3. **Function reference** — For each function:
   - Signature in target language syntax
   - Parameter types and descriptions
   - Return type
   - One or two examples

4. **Error handling** — How errors are reported and how to handle them idiomatically

5. **Type conversions** — What datetime types the library accepts beyond Unix timestamps

### usage.md template

```markdown
# whenwords for [LANGUAGE]

Human-friendly time formatting and parsing.

## Installation

[How to import/require/add the library]

## Quick start

[5-10 line example showing typical usage]

## Functions

### timeago(timestamp, reference?) → string

[Signature, parameters, examples]

### duration(seconds, options?) → string

[Signature, parameters, examples]

### parse_duration(string) → number

[Signature, parameters, examples]

### human_date(timestamp, reference?) → string

[Signature, parameters, examples]

### date_range(start, end) → string

[Signature, parameters, examples]

## Error handling

[Language-specific error handling patterns]

## Accepted types

[What types each function accepts]
```

Keep it under 150 lines. Developers should be able to skim it in under a minute.

---

## Implementation Checklist

Before considering the implementation complete:

- [ ] All five functions implemented
- [ ] All tests.yaml tests pass
- [ ] Functions accept language-native datetime types (not just Unix timestamps)
- [ ] Errors are raised/returned idiomatically
- [ ] Pluralization is correct ("1 minute" vs "2 minutes")
- [ ] Future times return "in X" not "X ago"
- [ ] Zero duration returns "0 seconds"
- [ ] Code is idiomatic for target language
- [ ] usage.md generated with function signatures and examples

---

## Version History

- **v0.1.0** - Initial specification