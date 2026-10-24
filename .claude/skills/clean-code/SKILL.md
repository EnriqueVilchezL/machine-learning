---
name: clean-code
description: 'Write readable, maintainable code from the first draft through disciplined naming, small functions, and clean error handling. Use when the user is implementing a new feature, writing a new function/module/class from scratch, or asks about naming conventions, function size, single responsibility, error-handling patterns, or unit test quality while writing. Trigger before and while producing new code, not as a pass over existing code. Covers SRP, comment discipline, formatting, and unit testing. For refactoring techniques, see refactoring-patterns. For architecture and dependency rules, see clean-architecture.'
license: MIT
metadata:
  author: wondelai
  version: "1.4.0"
---

# Clean Code Framework

A disciplined approach to writing code that communicates intent, minimizes surprises, and welcomes change — applied while the code is first being written. Reach for these principles before you write a function or class and while you're drafting it, so the first version is already clean rather than something to fix up later.

## Core Principle

**Code is read far more often than it is written — optimize for the reader from the first line.** The read-to-write ratio is well over 10:1, so every naming choice, function boundary, and formatting decision either adds clarity or adds cost — and it's cheaper to make that choice correctly the first time than to pay for it later as rework. Clean code reads like well-written prose: names reveal intent, functions tell a story one step at a time. Don't write a rough version now and plan to clean it up later — plan the name, the function boundary, and the error-handling shape before you type the body.

## Pre-Write Checklist

Before writing a function, class, or module, decide these against the principles below:

- What is its one responsibility, and what name states that responsibility as a verb (function) or noun (class)?
- What are its 0-2 arguments — if you're reaching for 3+, what parameter object groups them?
- Where does its happy path end and its error handling begin — will you guard-clause errors at the top?
- What will the test for this look like, and does that test look easy to write? If not, the design needs to change before the code does.

**Goal: 10/10 on first write.** Apply the scale below to what you're about to produce, not to a backlog of existing code — if a plan only reaches 5-6, redesign before writing rather than writing and revisiting.

- **9-10:** Names reveal intent, functions are small and focused, error handling is consistent, tests are clean and comprehensive
- **7-8:** Mostly clean with minor naming ambiguities or a few long functions; tests may lack edge cases
- **5-6:** Mixed — good patterns alongside unclear names, duplicated logic, or inconsistent error handling
- **3-4:** Long multi-purpose functions, misleading names, poor or missing tests
- **1-2:** Nearly unreadable — magic numbers, cryptic abbreviations, no structure, no tests

## The Clean Code Framework

Six disciplines for writing code that communicates clearly and adapts to change:

### 1. Meaningful Names

**Core concept:** Names should reveal intent, avoid disinformation, and make the code read like prose. If a name requires a comment to explain it, the name is wrong.

**Why it works:** Names are the most pervasive form of documentation — a well-chosen name eliminates the need to read the implementation; a poor one forces every reader to reverse-engineer intent.

**Key insights:**
- A name should answer why it exists, what it does, and how it is used
- No encodings, prefixes, or type information (no Hungarian notation); single letters only for tiny-scope loop counters
- Classes are nouns; methods are verbs
- One word per concept: don't mix `fetch`, `retrieve`, and `get`
- Longer scope demands a longer, more descriptive name
- Choose the name before writing the body — if you can't name it precisely, you don't yet understand what it should do

**Code applications:**

| Context | Pattern | Example |
|---------|---------|---------|
| **Variables** | Intention-revealing | `elapsedTimeInDays` not `d` |
| **Booleans** | Predicate phrasing | `isActive`, `hasPermission`, `canEdit` |
| **Functions** | Verb + noun | `calculateMonthlyRevenue()` not `calc()` |
| **Classes** | Noun naming the responsibility | `InvoiceGenerator` not `InvoiceManager` |

See [references/naming-conventions.md](references/naming-conventions.md) when choosing a name for something new — per-language conventions, pronounceable/searchable tables, and before/after examples.

### 2. Functions

**Core concept:** Functions should be small, do one thing, and do it well — ideally 4-6 lines, zero to two arguments, one level of abstraction.

**Why it works:** Small single-purpose functions are easy to name, understand, test, and reuse; long functions hide bugs, resist testing, and accumulate responsibilities.

**Key insights:**
- Step-Down Rule: code reads top-down, each function calling the next level of abstraction
- Argument count: zero best, one fine, two acceptable, three+ requires justification
- Flag arguments are a smell — the function does two things; split it
- Command-Query Separation: change state or return a value, never both
- Write each step as its own named function from the start rather than inlining it — if a block deserves a name, give it one immediately instead of folding it into a bigger function
- No hidden side effects — the name must tell the whole truth

**Code applications:**

| Context | Pattern | Example |
|---------|---------|---------|
| **Long function** | Extract named steps | `validateInput(); transformData(); saveRecord();` |
| **Flag argument** | Split into two functions | `renderForPrint()` / `renderForScreen()` not `render(isPrint)` |
| **Error cases** | Guard clauses at top | Early return for errors, single happy path |
| **Many arguments** | Introduce parameter object | `new DateRange(start, end)` not `report(start, end, format, locale)` |
| **Side effects** | Make effects explicit | `checkPassword()` that starts a session → rename or separate |

See [references/functions-and-methods.md](references/functions-and-methods.md) when splitting a long function — argument-count rules, command-query separation, and step-down worked examples.

### 3. Comments and Formatting

**Core concept:** A comment is a failure to express yourself in code. When comments are necessary, they explain *why*, never *what*. Formatting creates the visual structure that makes code scannable.

**Why it works:** Comments rot — code changes but comments often don't, creating documentation worse than none. Clean formatting lets developers scan code like a newspaper: headlines first, details on demand.

**Key insights:**
- The best comment is a well-named function — write the name instead of writing the comment
- Acceptable: legal headers, TODOs, public API docs, genuine "why" explanations
- Never leave commented-out code or journal comments in a new commit — version control remembers
- Vertical openness between concepts; vertical density within them; declare variables near usage
- Newspaper metaphor: high-level functions at the top of the file, details below

**Code applications:**

| Context | Pattern | Example |
|---------|---------|---------|
| **Explaining "what"** | Replace with better name | `// check if eligible` → `isEligible()` |
| **Explaining "why"** | Keep as comment | `// RFC 7231 requires this header for proxies` |
| **Commented-out code** | Delete it | Trust version control |
| **Team formatting** | Decide once, automate | Prettier, Black, gofmt |

See [references/comments-formatting.md](references/comments-formatting.md) when deciding whether a comment earns its place — good-vs-bad comment catalog and vertical-formatting rules.

### 4. Error Handling

**Core concept:** Error handling is a separate concern from business logic. Use exceptions rather than return codes, provide context with every exception, and never return or pass null.

**Why it works:** Return codes clutter the happy path with checks; exceptions separate the two cleanly. Returning null forces null checks on every caller, and one missing check crashes far from the source.

**Key insights:**
- Write the try-catch first — it defines a transaction boundary
- Prefer unchecked exceptions — checked ones violate the Open/Closed Principle
- Define exception classes by the caller's needs, not the failure type
- Don't return null (use empty collections, Optional, or throw); don't pass null either
- Special Case / Null Object pattern: return an object with default behavior instead of null

**Code applications:**

| Context | Pattern | Example |
|---------|---------|---------|
| **Null returns** | Empty collection or Optional | `return Collections.emptyList()` not `return null` |
| **Error codes** | Replace with exceptions | `throw new InsufficientFundsException(balance, amount)` |
| **Third-party APIs** | Wrap with adapter | `PortfolioService` wraps the vendor API, translates its exceptions |
| **Special cases** | Null Object pattern | `GuestUser` with default behavior instead of null checks |
| **Context in errors** | Include operation + state | `"Failed to save invoice #1234 for customer 'Acme'"` |

See [references/error-handling.md](references/error-handling.md) when designing exception or null strategy — Special Case pattern and third-party-API wrapping examples.

### 5. Unit Testing

**Core concept:** Tests are first-class code, kept clean with the same discipline as production code. Dirty tests are worse than no tests — they become a liability that slows every change.

**Why it works:** Clean tests are executable documentation and a safety net for refactoring; dirty tests make every modification a fight through incomprehensible test code.

**Key insights:**
- Three Laws of TDD: write a failing test first; only enough test to fail; only enough code to pass
- One concept per test — one logical assertion, not necessarily one assert
- F.I.R.S.T.: Fast, Independent, Repeatable, Self-validating, Timely
- Build a domain-specific testing language: helpers that read like a DSL
- Hold test code to the same naming and structure discipline as production code as you write it

**Code applications:**

| Context | Pattern | Example |
|---------|---------|---------|
| **Test structure** | Arrange-Act-Assert | Setup, execute, verify — clearly separated |
| **Test naming** | Scenario + expected behavior | `shouldRejectExpiredToken` not `test1` |
| **Shared setup** | Builder/factory helpers | `aUser().withRole(ADMIN).build()` |
| **Flaky tests** | Remove external dependencies | Mock time, network, file system |

See [references/testing-principles.md](references/testing-principles.md) when writing or cleaning tests — TDD laws, F.I.R.S.T. expanded, and clean-test patterns.

### 6. Code Smells and Heuristics

**Core concept:** Smells are surface indicators of deeper design problems — learn to recognize the shape of a smell as you're about to write it, so you choose the clean form directly instead of writing the smell and circling back.

**Why it works:** Smells are heuristics that point toward likely problems without deep analysis, turning "does this new code feel right?" into a specific, checkable list you can apply while drafting.

**Key insights:**
- Function smells to avoid introducing: too many arguments, output arguments, flag arguments, functions nothing will call
- General smells to avoid introducing: duplication, wrong level of abstraction, feature envy, magic numbers
- Test smells to avoid introducing: insufficient coverage, skipped tests, untested boundary conditions and failure paths
- If you're about to copy a block you just wrote, stop and give the shared logic a name instead
- Before considering a new piece of code finished, check it against this list rather than shipping it and fixing smells in a later pass

**Code applications:**

| Context | Pattern | Example |
|---------|---------|---------|
| **About-to-duplicate logic** | Name the shared logic once | Write `validateEmail()` instead of copying the same check twice |
| **Feature envy** | Put the method on the data's class | `order.calculateTotal()` not `calculator.total(order)` |
| **Speculative code** | Don't write it | Only write functions something calls |
| **Magic numbers** | Named constants from the start | `MAX_LOGIN_ATTEMPTS = 5` not bare `5` |
| **Scattered related logic** | Group it in one module up front | One cohesive module instead of logic spread across files |

See [references/code-smells.md](references/code-smells.md) for the full smell catalog by category — use it to recognize a smell you're about to write before it lands in the code.

## Common Mistakes (writing new code)

| Mistake | Why It Fails | Fix |
|---------|-------------|------|
| **Abbreviating names** | Saves seconds writing, costs hours reading | Write the full descriptive name the first time; IDEs autocomplete |
| **"Clever" one-liners** | Impressive to write, impossible to debug | Write it as readable named steps from the start |
| **Writing a comment instead of a name** | Comments rot; code is the truth | Write a well-named function instead of a comment |
| **Catching generic exceptions** | Swallows bugs along with expected errors | Catch specific exceptions; let the rest propagate |
| **Writing the happy path first, tests later** | Error paths and edge cases get forgotten | Write the failing test first; let it drive the code |
| **Premature optimization** | Obscures intent for marginal gains | Write it clean first; optimize measured bottlenecks |
| **Growing one class to do everything** | A 2000-line god class emerges one small addition at a time | Apply SRP from the first method — one responsibility per class |
| **Planning to "clean it up later"** | Later rarely comes; the rough version ships | Design the name, boundaries, and error handling before writing the body |
| **Inventing a new convention per file** | Every file feels like a different codebase | Follow the project's existing style; enforce with linters and formatters |
| **Returning null for "not found" cases** | Forces null checks on every caller | Return Optional, an empty collection, or a Null Object instead |

## Pre-Commit Diagnostic

Run this against code you just wrote, before considering it done — not as a later cleanup pass.

| Question | If No | Action |
|----------|-------|--------|
| Can you understand each function without reading its body? | Names don't reveal intent | Choose a name that describes what it does |
| Are all functions under 20 lines? | You wrote too many responsibilities into one function | Split into named helpers as you write, not after |
| Zero commented-out code blocks? | Leftover debug/dead code | Delete before committing — version control has history |
| Is error handling separate from business logic? | Try-catch clutters the main flow | Write handlers separately; exceptions over return codes |
| Does every class have a single responsibility? | You're accumulating unrelated duties in one class | Split into focused, well-named classes now |
| Is there a test for every public method you added? | No safety net for future changes | Write the test alongside the method, not after |
| Are test names descriptive of behavior? | Failures will be hard to interpret | Name it `shouldDoXWhenY` |
| Is duplication below 3 occurrences? | You copy-pasted instead of naming shared logic | Extract shared logic now (§6) |
| Are magic numbers named constants? | Intent hidden behind raw values | Name the constant as you write it (§6) |
| Do all new tests run in under 10 seconds? | Slow tests won't get run | Mock external deps; split out integration tests |

## Further Reading

Based on Robert C. Martin's seminal guide to software craftsmanship:

- [*"Clean Code: A Handbook of Agile Software Craftsmanship"*](https://www.amazon.com/Clean-Code-Handbook-Software-Craftsmanship/dp/0132350882?tag=wondelai00-20) by Robert C. Martin
- [*"The Clean Coder: A Code of Conduct for Professional Programmers"*](https://www.amazon.com/Clean-Coder-Conduct-Professional-Programmers/dp/0137081073?tag=wondelai00-20) by Robert C. Martin
- [*"Clean Architecture: A Craftsman's Guide to Software Structure and Design"*](https://www.amazon.com/Clean-Architecture-Craftsmans-Software-Structure/dp/0134494164?tag=wondelai00-20) by Robert C. Martin
- [*"Refactoring: Improving the Design of Existing Code"*](https://www.amazon.com/Refactoring-Improving-Existing-Addison-Wesley-Signature/dp/0134757599?tag=wondelai00-20) by Martin Fowler

## About the Author

**Robert C. Martin ("Uncle Bob")** has been programming since 1970, co-authored the Agile Manifesto, and founded Uncle Bob Consulting and Clean Coders. His books — *Clean Code*, *The Clean Coder*, *Clean Architecture*, and *Clean Agile* — shaped how a generation of developers think about code quality, and his core stance is that the only way to go fast is to go well.
