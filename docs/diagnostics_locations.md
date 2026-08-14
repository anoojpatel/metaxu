# Source locations in diagnostics

Every loud compile-time diagnostic names `file:line:column` and shows the
offending source line with a caret:

```
type check failed: collections.mx:4:5: undeclared type parameter 'N' in the
type 'vector[T, N]' of field 'data' of struct 'Buf': a vector size must be an
integer literal or a declared const generic parameter
  4 |     data: vector[T, N]
    |     ^~~~~~~~~~~~~~~~~~
```

## Where positions come from

1. **Lexer** (`src/metaxu/lexer.py`). PLY tracks `lineno`/`lexpos` per token;
   `Lexer.input` additionally records `column` and `endlexpos` (the exclusive
   end offset of the token's raw text) and keeps `line_starts`, the offset of
   the first character of every line.
2. **Parser** (`src/metaxu/parser.py`). The parser runs PLY with
   `tracking=True`, which stamps each reduced nonterminal with the start
   position of its first token and the end position of its last one *before*
   the grammar action runs. Every `p_*` action is wrapped once, in
   `Parser._grammar_namespace`, by `Parser._locating`; the wrapper reads that
   span off the production and calls `_attach_location`. No individual
   production was touched, so no rule can forget to attach a location.
3. **Freeze** (`src/metaxu/compiler/mutaxu_ast.py`). `_span_of` copies the
   parser's `SourceLocation` into the frozen `Span`.

### Coverage

Measured over the 33 `.mx` files in `examples/`, `std/` and the repository
root (3905 AST nodes):

| category | share | what |
|---|---|---|
| exact | 96.2% | the node's own production's token span |
| inherited | 3.8% | nodes built by parser *helpers* rather than by a production of their own (`HandleCase`, `EffectApplication`, `QualifiedName`, some `Block`s/`Parameter`s/`ModeAnnotation`s): `_attach_location` pushes the enclosing production's span down into still-unlocated descendants, so they report a range that *contains* them |
| none | 0% | — |

Nodes synthesized *after* parsing (desugaring, module renaming) can still
have no location; those report the compilation unit's file with no line
rather than a wrong line. `StructInstantiation` fields are synthesized from
`(name, value)` tuples and explicitly inherit their value's location.

## Position semantics

`errors.SourceLocation` and `mutaxu_ast.Span` agree, and both are explicit
about which unit each field uses:

| field | unit |
|---|---|
| `line` / `column` (Span: also `end_line` / `end_column`) | 1-based line and column; `column` is the first character of the construct, `end_column` points just past its last character. `0` means unknown. |
| `offset` / `end_offset` (Span: `start` / `end`) | 0-based character offsets into the source text, half-open: `source[start:end]` is the construct's text. `0/0` means unknown. |

Before this change `Span.start` held a *column* and `Span.end` mirrored it;
it is an offset now. The frozen-AST JSON gained `line`, `column`,
`end_line` and `end_column` inside each `"span"` object (the golden
`tests/golden/sample1.ast.json` was regenerated for the new shape).

## Rendering

`src/metaxu/errors.py` owns the format:

* `format_location(loc)` -> `file:line:column` (just `file` when the line is
  unknown, nothing at all when there is no location — a diagnostic never
  prints a misleading `<unknown>:0:0`).
* `source_excerpt(loc)` -> the two-line gutter/caret block. The caret spans
  `column..end_column` when the construct fits on one line.
* `format_diagnostic(msg, loc, error_type)` -> the whole thing, used by
  `CompileError.__str__`.
* `register_source(path, text)` lets in-memory compilations (`file_path`
  defaults to `<mem>`) render excerpts too; the parser registers every source
  it parses, and on-disk files are read back from disk.

Diagnostics that carry locations:

| diagnostic | class | location of |
|---|---|---|
| `ParseError` (syntax error, duplicate module, EOF) | `errors.CompileError` | the offending token |
| `ModuleError` (missing module, private symbol, missing symbol, duplicate function, relative-import escape) | `errors.CompileError` | the import / declaration / reference node |
| `ReservedNameError` | `errors.CompileError` | the declaration |
| `FrozenAstError` | `errors.CompileError` | the node whose payload is unserializable |
| `CoherenceError` | `compiler.desugar` | the duplicate method |
| `TypeCheckError` / `BorrowCheckError` | `compiler.frozen_borrow_checker` | each structured `BorrowError`'s node, resolved from the frozen AST by `locate_errors` |
| type-conflict (`1 + "a"`) | `simplesub_adapter.TypeConflict` | the later of the two conflicting values |
| `UnsupportedConstruct` / `HIRCompilerBug` | `compiler.hir` | the frozen node / original AST node being lowered |

### Deliberate gaps

* **`InterpError` is function-granular, not op-granular.** MIR ops are
  positional tuples that are dumped verbatim into the golden MIR text, so a
  per-op span would mean reshaping every op constructor, every consumer
  (interpreter, LLVM and CLIF backends) and every MIR golden. Instead
  `MirFunc.location` carries the function's declaration site and
  `InterpError.locate` appends `[in function 'f' declared at f.mx:3:1]` to
  the innermost frame's message — worded so it cannot be misread as the
  failing operation's line.
* **Advisory `-1` diagnostics** from `frozen_constraint_checker`
  ("Unresolved callee ...", "Invalid capture mode ...") are plain strings
  that still say `at node N`. They are advisories, never promoted to a
  user-facing error (only `kind == "type-conflict"` is), so they were left
  alone.
* **Import-cycle and global-collision `ModuleError`s** name modules rather
  than a single node (they are about a set of declarations), so they have no
  single location.
