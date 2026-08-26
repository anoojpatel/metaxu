"""LLVM IR text emitter for direct (non-suspending) MIR functions.

Emits a single, self-contained LLVM module (textual IR, compilable with
``clang x.ll``) for MIR functions whose ops fall entirely in the DIRECT
subset:

  params, const (int/bool/float/string/None), const_ty Unit, copy, binop
  (int arithmetic + comparisons, float arithmetic + comparisons, logical
  and/or), select, direct calls (to other emitted functions or the small
  builtin set below), drop (no-op comment), match_fail (call @abort +
  unreachable), local struct alloc/field ops, and the br / br_if / ret /
  unreachable terminators.

Increment 3 adds enum variants (make_variant / variant_tag / variant_field
as tagged unions) and stack-environment closures (make_closure + direct
locally-bound closure calls); see the dedicated sections below.

Increment 4 adds recursive/nested aggregates: enum payload slots holding
aggregates are HEAP-BOXED (making recursive enums like linked lists and
trees representable), struct fields holding structs/enums are INLINED in
the parent layout, and closures that escape (returned, or created in a
loop) get HEAP environments.  See the dedicated sections below — and note
the prominently documented free strategy: boxes and heap envs LEAK BY
DESIGN this increment.

Increment 5 links the NATIVE RUNTIME (src/metaxu/runtime/native/
metaxu_rt.c, built+linked by llvm_run) and lowers the vec/string builtins
to it:
  * Vec values are a new scalar-like kind family ``vec:ELEM``: an opaque
    ``mx_vec*`` pointer (8 bytes, stored like any ptr in struct fields,
    enum payload slots and closure envs) with IDENTITY semantics — every
    copy is a shallow pointer copy aliasing the one shared vector, exactly
    the interpreter's MxVec.  ELEM is the unified element kind; elements
    travel as opaque 8-byte words (f64 bitcast, str/vec ptrtoint) through
    mx_vec_push/mx_vec_pop/mx_vec_get.  ``Vec.new``->mx_vec_new,
    ``push``/``pop``->mx_vec_push/pop, ``__index_get``->mx_vec_get,
    ``len``->mx_vec_len (or mx_str_len on a string receiver).
  * ``to_string``/``int_to_str`` -> mx_i64_to_str / mx_f64_to_str /
    identity on a string.  CAVEAT (same kind-erasure divergence as print):
    bools and unit are erased to i64, so ``true.to_string()`` yields "1"
    natively where the interpreter says "True"; tests stringify ints,
    floats and strings.
  * string ``+`` -> mx_str_concat and ``==``/``!=`` -> mx_str_eq; concat
    and to_string results are fresh malloc'd strings that LEAK BY DESIGN
    (the box/heap-env contract; ordering comparisons on strings demote).
  * ``__trait$m`` calls resolve STATICALLY against the receiver's inferred
    kind, mirroring the interpreter's dispatch order (impl for the
    receiver's type name -> builtin -> plain function); an i64 receiver is
    kind-erased (int/bool/unit), so it resolves only when no impl exists
    for any of those type names.  Unresolvable dispatch demotes.
  * FREE STRATEGY for vecs: mx_vec_free at frame exit ONLY for vecs
    proven non-escaping by _provably_dead_vecs (created unconditionally in
    the entry block; never returned, stored, captured, or passed anywhere
    except as the receiver of the non-retaining vec builtins).  Everything
    else leaks by design — identity sharing makes any other free
    potentially a double-free.
  * sqrt/sin/cos remain plain libm externs (documented choice, see
    llvm_run) — no wrappers, no intrinsics.

Increment 6 replaces the per-(enum, slot) payload kind model with
PER-VARIANT, PER-VALUE payload typing so one enum can hold different
payload types in the same slot index — across variants (Leaf(int) |
Fork(Tree, Tree)) and across generic instantiations of one variant
(Some(3) and Some(node) in one module, linked_list.mx's reality).  See the
REFINED ENUM KINDS section below.  The MIR ``variant_field`` op now
carries the pattern's ctor name as a third element (lower_hir_to_mir;
interpreter and CLIF ignore it) so the backend knows WHICH variant's slot
a read refers to — chosen over recovering the variant from the dominating
tag test because compile_pattern interleaves nested sub-pattern blocks
between the tag branch and later slot reads, making a dominator trace
fragile exactly where it matters (nested ctor patterns).

Increment 7 makes ALGEBRAIC EFFECTS native: handle_scope / perform /
resume lower to the C effects runtime (src/metaxu/runtime/native/
metaxu_effects.c), a small ucontext coroutine scheduler that reproduces
the MIR interpreter's parked-thread model exactly — deep handlers,
single-shot resume returning the WHOLE delimited body's value, abort when
a case returns without resuming, dynamic innermost-first routing over a
process-wide scope stack (busy handler frames skipped so handler
self-performs route outward).  See the ALGEBRAIC EFFECTS section below.
Suspending functions are no longer demoted: the coroutine stack IS the
continuation, so no CPS transform is needed (codegen_clif's CPS state
machines remain the CLIF story).

Increment 8 moves toward zero-cost: RECLAIM what can be proven, ELIDE the
copies the value-semantics discipline already makes redundant.  The
soundness bar is unchanged — a leak is acceptable, a UAF/double-free never:
  * OWNED STRINGS: concat/to_string results are fresh mallocs; a string
    variable whose every def is a literal/producer/ANF-transfer-copy and
    whose every use is non-retaining (concat operand, ==/!=, print, len)
    owns its produced values through a null-initialized shadow slot — each
    redefinition frees the previous value (a concat loop no longer grows),
    frame exit frees the last, literal defs store null (interned constants
    are NEVER freed; provenance is static).  See _owned_strings.
  * UNIQUE BOXES: an entry-block make_variant whose enum value (closed
    over intra-frame copies) is used only as a variant_tag/variant_field
    base — never returned, passed, stored, captured, or re-boxed — has
    sole ownership of its payload boxes; they are freed on every ret path.
    Everything shared stays leaked by design.  See _unique_box_enums.
  * COPY ELISION (a): an aggregate param that is never rebound and never
    reaches a callee's write-back position skips the entry byval copy and
    reads the caller's aggregate through the passed pointer (nothing ever
    writes that storage).  (b): a variant_field result that is only ever
    read becomes a BOX VIEW — a pointer aliasing the write-once box instead
    of an aggregate copy; copies of a view alias the same box under the
    same read-only conditions.  Both elisions are marked with
    `; elide-copy:` comments in the IR.

Increment 9 makes native FFI real — extern C calls, a raw-pointer kind,
vector literals and static method calls (05_unsafe_and_ffi.mx emits and
runs natively):
  * RAW POINTERS: a new scalar kind ``rawptr`` (8-byte ``ptr``, shallow
    copies — C semantics exactly).  ``null`` (MIR ``const None``) stays at
    the i64 bottom until unification with a rawptr flow promotes it, then
    emits as the ``null`` ptr constant; ==/!= on two rawptrs is ``icmp``
    pointer equality.  rawptr is deliberately NOT a word kind: it cannot
    enter Vec elements or cross effect boundaries (demotes honestly).
  * EXTERN C CALLS: MIR calls to the extern-declared libc names the
    interpreter shims (malloc/free/memcpy/realloc/fopen/fclose) emit as
    DIRECT calls to the real C symbols with their C signatures (fclose's
    C ``int`` is declared i32 and sext'd).  The interpreter runs these
    against a simulated, bounds-checked heap; natively they hit the real
    allocator, so programs the interpreter ACCEPTS behave identically
    while programs it rejects (overrun/UAF/double free) are real UB
    natively — the same strict-error-vs-UB contract as division by zero
    (ASan differentials pin the accepted side).  Like the interpreter,
    a module function of the same name WINS over these (NAME PRECEDENCE,
    docs/name_precedence.md); `extern` declarations produce no module
    function, so an FFI program's calls land here.
  * ``as_ptr``: on a string, IDENTITY (native strings already are
    NUL-terminated byte pointers; the interpreter's fresh readonly
    snapshot is observationally identical for every accepted program —
    writes through it are interpreter errors).  On a vec, a fresh
    malloc'd byte SNAPSHOT via mx_vec_as_bytes (interpreter parity:
    never a view into the vec's word buffer — elements are 8-byte words
    natively, so a raw data-pointer alias would have the WRONG layout;
    the snapshot also survives vec growth/free).  Snapshots leak by
    design.  Non-byte elements abort (interpreter strictness).
  * ``ptr_read``/``ptr_write``: inline i8 load (zext) / store (trunc)
    through a byte GEP.  The interpreter bounds-checks and rejects
    non-byte values; natively out-of-range is UB (same contract as above).
  * ``__vec_lit`` (fixed-size ``vector[T,N](...)`` literals) lowers to
    mx_vec_new + one mx_vec_push per element.  CAVEAT (documented): the
    interpreter's MxVector has immutable VALUE semantics; natively the
    value is an mx_vec with identity semantics.  For every program the
    interpreter accepts these are indistinguishable (MxVector supports no
    mutation — push/pop on it are interpreter errors), but a rejected
    program could mutate natively instead of erroring.  len /
    __index_get / as_ptr work uniformly; elementwise vector arithmetic
    and __vec_dim/__slice_get/... still demote.
  * ``__static$Type$method`` calls resolve at COMPILE TIME exactly like
    the interpreter's _dispatch_static_call: the unique __impl$*$Type$
    method fn, else the plain dotted module function ``Type.method``,
    else the dotted builtin (``Vec.new``); multiple candidate traits
    demote (ambiguity is an interpreter error).
  * ``assert(cond, ...)`` lowers to an inline branch-to-@abort on a
    falsy i64 condition (message arguments are evaluated but not
    rendered natively; a failing assert aborts instead of raising).

Increment 10 makes the FIXED-VECTOR runtime native.  A `vector[T, N]`
value is a new scalar-like kind family ``vector:ELEM`` — an opaque
``mx_fvec*`` pointer to an IMMUTABLE length-prefixed word block
``{ i64 len, [len x i64] }`` (metaxu_rt.c).  The interpreter's MxVector
has VALUE semantics; since no operation ever mutates a filled block
(construction fills it before the pointer is shared), shallow pointer
copies are observationally identical to value copies — the write-once
payload-box argument — and blocks LEAK BY DESIGN (shallow sharing makes
ownership non-unique).  Push/pop on a fixed vector now demote by kind
(the interpreter rejects them), retiring increment 9's mx_vec stand-in
caveat:
  * ``__vec_lit`` -> mx_fvec_new + mx_fvec_init fills; ``__vec_zeros`` ->
    mx_fvec_new (calloc: 0 and 0.0 are the all-zero word; the base-name
    argument must be a const 'float'/'int' string); ``__vec_filled`` ->
    mx_fvec_filled; ``len``/``__index_get``/``as_ptr`` route by kind to
    mx_fvec_len/mx_fvec_get/mx_fvec_as_bytes.
  * element-wise arithmetic (+ - * / % with scalar broadcasting, nested
    matrices included) -> mx_fvec_binop; vector operands and the result
    share one vector kind, a broadcast scalar unifies with the LEAF
    element kind.  Int elements use the C-truncating sdiv/srem convention,
    which the interpreter now matches exactly (it used to floor like
    Python); integer division by zero aborts.  ==/!= on vectors demote
    (the interpreter compares structurally).
  * ``__slice_get`` -> mx_fvec_slice, a FRESH COPY with CPython
    slice.indices() semantics; a bound must be statically None (a
    const-None variable) or an int — a sometimes-None bound demotes.
  * ``__range`` -> mx_fvec_range (an int vector).  The interpreter's
    range is a plain LIST whose repr differs from a vector's, so range
    values are restricted to the iteration protocol (len / __index_get /
    comprehension iterable / copies); any other use demotes.
  * ``__vec_comprehension`` -> a per-site thunk (decode element word ->
    call the statically-known body closure -> encode result) driven by
    mx_fvec_map.  i64 elements/results may flow into f64 positions (the
    thunk converts — the scalar int->float promotion contract); anything
    else mismatched demotes.  Multi-parameter bodies (tuple unpacking)
    and non-vector iterables demote.
  * ``__cast`` with a const type-name target: numeric targets convert
    (sitofp / fptosi — truncation toward zero, like Python's int());
    every other target is the interpreter's identity reinterpretation.
  * ``promote_matrix`` (matmul's vector -> Mx1 embedding): resolved
    STATICALLY per parameter from the caller-side sig kind — a flat
    numeric vector param is rebound to mx_fvec_promote(param) at entry
    (its local kind is the promoted matrix kind; the driver skips the
    local->sig join for such params), a matrix passes through, and mixed
    flat/matrix callers demote via the sig join.  Supported shape:
    entry block, parameters only, before any other use.
  * ``print``/``to_string`` of a vector render the interpreter's repr
    ("vector[1.0, 2.0]") via mx_fvec_to_str (int/float leaves only;
    mx_f64_to_str already matches Python's float repr; bools are
    kind-erased to ints, the standing caveat).  print frees the repr
    string immediately; to_string results leak by design.
  * EFFECT-OP DEFAULTS: a perform whose op name appears in NO
    handle_scope of the module can never be intercepted by a scope, so
    the interpreter's fallback is static — when the op declares an
    `= expr` default (__effect_default$E$op) and no `with SYMBOL` runtime
    mapping, the perform lowers to a DIRECT CALL of the default function
    (ordinary conventions, aggregates and all, no effect boundary).  An
    op with a default that ALSO appears in some scope routes dynamically
    (increment 15, below).

Increment 11 makes vector arithmetic emit REAL LLVM SIMD IR where the
shape is statically provable, closing the gap between the fixed-vector
runtime (increment 10's C loops) and the design intent that vector math
actually vectorizes (`<N x double>` IS a SIMD register type):
  * STATIC LENGTHS: a per-function analysis (_fvec_static_lens) tracks
    each fixed-vector variable's element count as a small lattice —
    bottom -> known N -> dynamic — joined over ALL defs of the name (the
    same one-fact-per-variable discipline as the kind map) and iterated
    to fixpoint.  Producers with provable counts: literals (element
    count), zeros/filled (const count), const-bounds ranges, slices with
    const/None bounds over known inputs (CPython slice.indices — exactly
    mx_fvec_slice's semantics), comprehension outputs (input length, else
    the const declared size mx_fvec_map enforces by abort), copies,
    selects, casts, and binop results (any operand's known length —
    sound because mx_fvec_binop aborts on vector-vector mismatch, so
    every CONTINUING path shares one length).  Parameters, captures,
    call results, struct/enum reads, effect crossings and const-None
    slot initializers are dynamic.  The lengths live in a side table
    keyed by variable, deliberately NOT in the kind strings: kinds flow
    through module-wide unification cells (sigs, struct fields, closure
    envs, effect-op cells) where a length component would be destroyed
    by every rebuild-from-element-kind join site — the side table keeps
    the kind lattice untouched and the join rules locally auditable.
  * INLINE VECTOR IR: a flat float/int vector binop whose operand
    lengths are known (mode 0: both known AND equal; broadcast modes:
    the one vector operand known) and 1 <= N <= 64 emits inline IR
    instead of the mx_fvec_binop call: `<N x double>` / `<N x i64>`
    loads straight off the block's element words (byte offset 8 — the
    { i64 len, [len x i64] } layout; f64 words are bitcast-stored so the
    f64 view of the same memory is the identity; align 8, the block's
    real alignment), one vector fadd/fsub/fmul/fdiv or add/sub/mul,
    scalar broadcast via insertelement + shufflevector splat, and a
    store into a fresh mx_fvec_new result block (calloc'd with the len
    header already set; leaks by design like every fvec block).
  * DOCUMENTED CHOICE — int / and % keep the runtime call even with
    known lengths: mx_fvec_scalar_op aborts loudly on division by zero
    where a vector sdiv/srem would be UB; float ops are IEEE on both
    paths (fdiv by zero -> inf/nan, matching the C loop exactly).
  * NEVER WRONG, ONLY FASTER: every unproven shape — dynamic or
    mismatched lengths, nested matrices (depth > 0), N outside
    [1, 64], float %, int division — falls back to the increment 10
    runtime call, which remains correct (and carries the abort
    semantics the inline path must never skip).
  * REDUCTIONS: emit_fvec_reduce is the ready lowering for horizontal
    sums (`llvm.vector.reduce.fadd` ORDERED with a -0.0 seed — the
    exact fadd identity, bit-identical to the interpreter's
    left-to-right fold — / `llvm.vector.reduce.add`).  No MIR shape
    reaches it yet: example 06's sum/dot/norm demote UPSTREAM on
    closure-kind conflicts that are not this backend's to fix; the
    helper is tested on synthetic modules so the wiring is proven.

Increment 12 lifts the SILENT-SEAM constructs the last front-end round
added (index assignment, mutable captures, module constants, zip):
  * INDEX ASSIGNMENT: ``__index_store`` (the store-back form
    `place = __index_store(place, i, x)`) lowers by receiver kind —
    a Vec receiver calls mx_vec_set (in-place, bounds-checked abort;
    the result IS the same pointer, so the place rebind is a no-op and
    the dead-vec analysis treats the result as a group alias, keeping
    provably-local vecs freeable) and a vector[T,N] receiver calls the
    new mx_fvec_set_copy — the interpreter's FUNCTIONAL update made
    native: fvec blocks are shallow-shared and write-once, so the update
    COPIES the block, stores the element, and rebinds the place (other
    shares never observe it; the fresh block leaks by design).
    ``__index_set`` (in-place-only form, `m[i][j] = x`) is mx_vec_set on
    Vec receivers; on a fixed vector (or any other receiver) it demotes
    AT COMPILE TIME with the interpreter's immutability error.  A
    functional fvec update propagates the receiver's static length.
  * MUTABLE-CAPTURE CELLS: a ``cell_wrap`` op marks its variable
    CELL-BACKED for the whole frame — storage is a malloc(8) one-word
    heap box (leaked by design; an immortal cell cannot dangle): reads
    load through %cellp.<n>, writes store through it, and closure/
    handle-scope envs capture the CELL POINTER (marked ``cell:ELEM`` in
    the env layout tables only — the kind lattice never sees cells), so
    every frame shares one binding, exactly the interpreter's MxCell.
    Cellness propagates through capture chains module-wide
    (_CellTable); a capture NOT provably after the wrap (the
    interpreter froze a value copy there) demotes, as do aggregate
    cells, and per-function const/dead/static-length facts about
    cell-backed names are dropped (another frame can write them).  This
    un-demotes handler-frame counters (std.stream take/skip's `seen`,
    for_'s `broke`) and every closure mutating a captured scalar.
  * MODULE CONSTANTS: ``__module_init``'s declared names become
    zero-initialized internal globals ``@mx_g_<name>`` (natural scalar
    types: i64/double/ptr).  The initializer emits as a normal function
    whose decl-name defs STORE to the globals; readers (uses with no
    local def) LOAD from them — the interpreter's env-then-globals
    lookup order, with parameter shadowing local and flow-sensitive
    assignment shadowing demoted.  Kinds join through module-wide
    per-name cells (_GlobalTable); aggregate globals demote.  llvm_run's
    entry wrapper calls @mx___module_init before the entry point (the
    interpreter's _ensure_globals) and REFUSES to run any entry when the
    module has a demoted initializer.  Scope members and lambdas resolve
    free global names without env captures (unless the env shadows
    them).
  * ZIP COMPREHENSIONS: ``__zip(xs, ys)`` is a VIRTUAL value — the
    interpreter's list of tuples has no native representation, so its
    only legal use is the iterable of a ``__vec_comprehension`` (every
    other use demotes).  The comprehension site emits a two-word thunk
    (decode each source's element word into the two-parameter body
    lambda) driven by the new mx_fvec_zip_map, which ABORTS on a length
    mismatch exactly like the interpreter's strict __zip (and on a
    declared-size mismatch like mx_fvec_map).  Non-pair zips and Vec
    sources demote.

Increment 14 lifts AGGREGATES ACROSS THE EFFECT BOUNDARY via boundary
boxes — the same write-once box contract as enum payloads:
  * a struct / enum / closure-pair value used as a perform argument,
    resume value, handler-case result, body result or handle-scope result
    crosses as a POINTER WORD to a fresh malloc'd BOUNDARY BOX: the sender
    copies the aggregate into the box (the fill is the box's only write),
    the word travels through mx_perform/mx_resume/mx_handle untouched, and
    the receiver copies the aggregate OUT into its own storage (case
    params receive the box pointer directly — the aggregate-param
    byval-copy convention IS the copy-out).  Boxes are IMMORTAL (leak by
    design, exactly like payload boxes): a heap box can never dangle
    across coroutine switches, parks, or scope teardown, which is what
    makes the lifetime argument need no escape analysis at all.
  * the module-wide op-name/site cells (perform args ⊔ case params,
    perform results ⊔ resume values, handle value ⊔ body/case returns ⊔
    resume results) now carry aggregate kinds; the same-named-op
    coarseness rule stays — two same-named ops with irreconcilable kinds
    (aggregate or scalar) still conflict and demote.
  * body/case subfunctions with aggregate results keep their ordinary
    sret convention; the per-site shims box: the body thunk mallocs the
    box and calls the body fn sret-style into it, the dispatcher does the
    same per aggregate-returning case (an alloca would die with the shim
    frame while the word outlives it — hence malloc).
  * closure pairs cross as {fn, env} two-word boxes; every member lambda
    of a boundary-crossing closure kind is forced HEAP-ENV (the
    increment-13 env-capture rule), so the boxed pair's env pointer aims
    at an immortal block wherever the word travels.
  * still demoted honestly: konts (resume must run on its scope's owner
    stack), rawptr words, infinite layouts, closures of unknown or
    non-module lambdas, conflicting cells, and performs of ops with a
    `with SYMBOL` C-runtime mapping (__effect_runtime$E$op — the
    interpreter routes unscoped performs to the EFFECT_* primitives;
    natively mx_perform would abort instead, so effect_mapping.mx's
    threads/mutex ops demote with that exact reason).

Increment 15 lowers DYNAMIC EFFECT-DEFAULT ROUTING — the case where an op
declares a `= expr` default AND some handle scope lists it, so which one
answers a perform is decided AT THE PERFORM by the runtime scope stack
(example 06's SimdOp capability: `try_horizontal(...) = None` answered by
`with_simd`'s handler inside it and by the default everywhere else):
  * a new runtime entry point `mx_perform_or_default(effect, op, args,
    nargs, default_thunk, default_env)` runs the IDENTICAL innermost-
    non-busy scope lookup as mx_perform (same padding, same arity abort,
    same parking) and calls the thunk ONLY where mx_perform would have
    aborted.  mx_perform's abort path is untouched — an op with no default
    still dies loudly.
  * the interpreter's precedence at a perform is: in-scope handler frame >
    `with SYMBOL` runtime mapping > declared `= expr` default > error.
    Native mirrors ALL four rungs (docs/threads_runtime.md): rung 2's
    EFFECT_SPAWN/JOIN and EFFECT_MUTEX_* map to the pthreads-backed
    primitives in metaxu_threads.c, reached through the op's
    __effect_runtime$E$op thunk.  The mapping outranks the default, so
    the thunk passed to mx_perform_or_default is the runtime-mapping
    thunk when one is declared, else the default thunk; an op no scope
    lists routes to its fallback thunk by a DIRECT call (no boundary).
    A `with SYMBOL` outside the implemented set still demotes with a
    reason.  EFFECT_SPAWN hands the closure's {fn, env} to
    mx_thread_spawn: member lambdas are forced heap-env (the child
    dereferences the env after the spawning frame moved on) and must be
    zero-arg word-uniform (`i64 (ptr env)` is the C child entry's type);
    Thread/Mutex extern-type values are opaque i64 handle words.  (The
    interpreter has a fifth, HOST rung between the frames and the
    mapping: handlers registered through
    MirInterpreter.register_effect_handler.  That is a Python embedding
    API — no compiled Metaxu program and none of the gates install one —
    so it has no native counterpart and cannot diverge for any program
    this backend compiles.)
  * the default RUNS ON THE PERFORMING STACK: it is an expression, not a
    suspension.  No coroutine, no scope record, no continuation (there is
    nothing to resume — the perform simply becomes a call).  The scope
    stack is unchanged across it, so a perform INSIDE the default routes
    and parks exactly as one written at the perform site would, which is
    what the interpreter does.
  * per op, one internal `mxfx.dflt.<default fn>` thunk
    (`i64 (ptr env, ptr args)`) decodes the boundary words into the
    default's parameter kinds, calls it, and word-encodes the result —
    the dispatcher's per-case arm minus the op index and the `__k`.
    Aggregates use the same boundary boxes as a handler case (param words
    are the sender's box pointer, an aggregate result is sret-filled into
    a fresh malloc'd box).  `env` is null today (defaults are top-level
    module functions with no captures) and exists so a capturing default
    needs no second entry point.
  * KIND UNIFICATION: the default's signature joins the SAME module-wide
    op-name cells as the handler cases (perform args ⊔ case params ⊔
    default params; perform results ⊔ resume values ⊔ default return), so
    every route through one op shares one lattice.  A default that cannot
    agree with its handlers CONFLICTS and demotes, exactly like two
    irreconcilable same-named ops; an arity disagreement between a perform
    and its default demotes too (only the SCOPE path pads with UNIT).

Everything else — `type_of` (no interpreter builtin exists), comprehensions
over Vecs, string slicing/indexing — is emitted as a clearly marked,
comment-only placeholder carrying the reasons, never as silently wrong
code.  Functions that call a placeholder function are
themselves demoted (the module must link), with an explicit reason.

MONOMORPHIZED INPUT (increment 18).  The kind cells below are PER FUNCTION
and monomorphic: a generic function reached at two different types joins
both kinds into `conflict` and demotes with "irreconcilable value kinds".
`pipeline.emit_llvm_from_source` therefore runs compiler/monomorphize.py
before lowering, so each resolvable instantiation arrives here as its own
function (`identity$Int`, `identity$String`) with its own cells.  Call
sites whose type arguments that pass cannot resolve keep their generic
callee and still join here — the demotion is the honest answer, not a
gap to paper over.

ALGEBRAIC EFFECTS (increment 7):
  * `handle ... with {cases} in {body}` lowers (in MIR) to per-site body /
    handler-case subfunctions plus a `handle_scope` op capturing the
    enclosing frame's values.  Natively each site gets ONE shared env
    struct `%henv.<site>` holding the union of the FREE NAMES of its body
    and case subfunctions (computed by a module-wide fixpoint that sees
    through nested handle sites); the site fills it like a closure env
    (eager, by value; aggregates copied whole) and passes it as both the
    body env and the handler env of `mx_handle`.  Body/case subfunctions
    are emitted like lambdas: a leading `ptr %cl.env` parameter and a
    prelude that reloads (only) their own free names from the struct.
  * per site, codegen emits two internal shims: a BODY THUNK
    `i64 (ptr env)` calling the compiled body fn and word-encoding its
    result, and a DISPATCHER `i64 (ptr env, i64 op_index, ptr args,
    ptr k)` that switches on the op index (dense, the site's case order —
    documented in the IR), decodes the argument words to the case's param
    kinds, calls the compiled case fn (its trailing `__k` param receives
    `k`), and word-encodes its return.  Two private constant arrays per
    site carry the op-name strings and per-op case arities for the
    runtime's dynamic routing and UNIT-padding/arity checks.
  * `perform` stores its argument words into a per-function
    `[8 x i64]` scratch alloca and calls `mx_perform(effect, op, args,
    nargs)`; the runtime parks the current coroutine and returns the
    resumed value.  `resume` (only valid inside the handler case that
    received the continuation — its own trailing `__k` param) calls
    `mx_resume(k, value_word)`.
  * every value crossing the effect boundary travels as an opaque 8-byte
    WORD (i64 / f64 bitcast / str-vec ptrtoint — the Vec-element
    convention).  Because routing is DYNAMIC (by op name, innermost
    scope first), kinds unify through module-wide cells keyed by OP NAME:
    perform arguments ⊔ handler-case params per index, perform results ⊔
    resume values; and per SITE: handle value ⊔ body return ⊔ case
    returns ⊔ resume results.  Two same-named ops with irreconcilable
    types conflict and demote (a sound over-approximation of the dynamic
    routing).  Aggregates crossing the boundary travel as boundary-box
    pointer words since increment 14 (see above); a continuation captured
    into a nested scope env or closure demotes (resume must run on its
    scope's owner stack).
  * `__k` continuation values get the dedicated non-word kind ``kont``
    (an opaque `mx_k*`); it may only flow from a case's param into its
    own resume ops.
  * memory: the effects runtime frees its coroutine stacks, scope records
    and continuation records at scope completion/abort (leak-clean, ASan
    fiber-annotated); the site env is a frame alloca (the enclosing frame
    outlives `mx_handle`, which returns only after the scope ends).

TRY/CATCH (increment 19) — delimited failure recovery, docs/try_catch.md:
  * `try { body } catch e { handler }` lowers (in MIR) to a `try_scope` op
    naming a body subfunction and a catch subfunction, exactly the shape of
    a handle site with no cases.  It REUSES the handle-site machinery here:
    one shared `%henv.<site>` env struct filled by the owner, free-name
    fixpoint, one value cell (try value ⊔ body return ⊔ catch return), and
    the boundary-word return ABI for aggregate results.
  * natively it becomes `mx_try(body_thunk, env, catch_thunk, env)` over
    per-site `mxtc.body.<site>` / `mxtc.catch.<site>` shims.  The runtime
    (metaxu_effects.c) installs a setjmp LANDING PAD, whose chain is
    per-fiber and saved/restored across every coroutine switch — so a try
    inside a handle body survives a perform/resume round trip, and a
    failure raised on a body coroutine escapes to its owner stack (the
    interpreter's ("error", exc) message) instead of longjmping into a
    parked frame.  Catching also tears down every effect scope the failure
    escaped, which is the interpreter's `finally: _abort_scope(...)`.
  * the CATCH PARAMETER is always kind `str`: the runtime hands it the
    failure's plain text, which must be byte-identical to the interpreter's
    `InterpError.message` (it is a language value, not a diagnostic).  Any
    other kind on that parameter conflicts and demotes.
  * WHAT IS CATCHABLE is a contract shared with the runtime
    (metaxu_effects.h): every InterpError the native backend can produce
    inside a delimited extent is raised through mx_raise with the
    interpreter's own wording — match_fail included: its message embeds
    the function's PRE-monomorphization origin name (MirFunc.origin_name),
    so classify$Int raises "match failure in 'classify': ..." exactly as
    the unspecialized reference run does.
    Failures the interpreter does NOT raise InterpError for stay fatal on
    both sides (assert -> AssertionError, division by zero ->
    ZeroDivisionError, double resume -> RuntimeError).
  * memory: the pad is a stack object; the caught message is a fresh heap
    copy that LEAKS BY DESIGN (an ordinary produced `str`).  The scheduler
    itself stays leak-clean across a caught failure.

Type model (documented conventions):
  * ints, bools and unit are all ``i64``; unit is the constant 0.
  * floats are ``double`` (printed as IEEE-754 bit patterns, ``0x...``).
  * strings are ``ptr`` values pointing at private unnamed_addr constant
    NUL-terminated byte arrays; they are immutable and never freed.
  * icmp/fcmp produce i1, immediately ``zext``-ed to i64 so booleans are
    uniformly i64.  ``br_if`` compares its i64 condition against 0.
  * logical &&/|| normalize both operands with ``icmp ne 0`` before
    and/or (truthiness semantics, matching the MIR interpreter, not
    bitwise-and like a naive lowering).
  * int / and % use ``sdiv``/``srem`` (C truncating semantics), which the
    MIR interpreter matches exactly: it truncates toward zero and takes
    the sign of the dividend rather than flooring like Python, so the two
    engines agree on negative operands too.  Division by zero is UB
    natively (the interpreter raises).
  * LOCAL (default / @local) structs: a named ``%struct.T`` per struct
    type, one entry-block ``alloca`` per struct-typed MIR variable,
    ``getelementptr`` + load/store for fields.  MIR struct ops have value
    semantics (``field_set`` yields an updated copy), so every def of a
    struct variable stores a whole aggregate into that variable's own
    storage — no aliasing, and SROA/mem2reg scalarize it at -O2.  Zero
    memory management for locals: the frame is the allocation.
  * @GLOBAL structs live on the heap: a struct variable defined by an
    ``alloc_struct`` with locality "global" gets its storage from an
    entry-block ``call ptr @malloc(i64 <8 * nfields>)`` instead of an
    alloca (every scalar field kind — i64/double/ptr — is 8 bytes, so the
    size and the GEP layout are exact), field access GEPs the heap
    pointer, and every ``ret`` path frees the block with
    ``call void @free(ptr ...)``.  Freeing at function exit is provably
    sound here because MIR struct values have pure value semantics: every
    cross-frame transfer below (parameter, return) moves the *aggregate*
    by copy, never the storage pointer, so a callee's heap block can never
    be reached after the callee returns.  Should a future op let a raw
    struct pointer escape, that op must demote or suppress the free: a
    leak is safe, a dangling pointer is not (unproven lifetimes leak by
    design in this increment).  abort()/unreachable paths do not free
    (the process is dying).  A variable whose defs mix @local and @global
    allocations is uniformly heap-backed (storage location is unobservable
    under value semantics; the conservative cost is one malloc+free).
    This agrees with borrow_analysis.plan_drops, whose only drop point
    today is function exit (drop_at_end -> MIR ``drop`` ops); the frees do
    not depend on its needs_drop heuristic, only on the non-escape
    guarantee above.
  * struct values CROSS CALL BOUNDARIES by pointer, preserving MIR value
    semantics at both edges:
      - struct parameters are passed as ``ptr`` and the callee immediately
        copies the aggregate into its own storage in the entry prelude
        (byval-copy).  A later borrow-informed increment can elide that
        copy for @const/read-only params once the borrow checker's results
        are threaded into codegen.  COPY-OUT (interpreter write-back
        parity): a BY-REFERENCE struct param (MirFunc.mut_params: declared
        @mut, or a method's `self` receiver) the callee REBINDS anywhere is
        copied back through the caller's pointer on every ret path,
        matching mir_interp._write_back_struct_args (`self.field = ...`
        methods mutate the caller's binding); plain params keep value
        semantics (rebinding stays callee-local); lambdas copy out exactly
        their @mut-declared params.
      - struct returns are sret-style (the ONE convention used
        everywhere): the caller passes its result variable's storage as a
        leading ``ptr %agg.ret`` argument, the callee copies the returned
        aggregate into it and returns ``void``.  Small-struct returns as
        first-class LLVM aggregates were considered and rejected to keep
        one uniform path.  No ABI ``sret`` attribute is needed: all such
        calls are module-internal.
    NESTED AGGREGATE FIELDS (increment 4): a struct field whose kind is
    itself a struct or enum is laid out INLINE in the parent %struct.T
    (the field's LLVM type is the nested %struct.X / %enum.E), fully
    stack-based: alloc_struct copies the aggregate value into the field
    region, field_get copies it out into the destination's own storage,
    field_set copies the whole parent then overwrites the region — plain
    value semantics with recursive GEPs, no heap.  Struct sizes are
    computed recursively (every leaf cell is 8 bytes; enum payload slots
    are 8 bytes each — aggregates there are boxed pointers, see below), so
    @global malloc sizes stay exact.  A struct-in-struct cycle with no
    intervening enum box has no finite layout and demotes (such a value
    could never be constructed anyway).  A field holding a closure still
    demotes: the pair's env pointer may aim at a stack frame the struct
    could outlive.
  * ``print``/``println`` of a single value routes by operand type to
    @metaxu_print_i64 / @metaxu_print_f64 / @metaxu_print_str, small
    helpers defined in this module on top of a declared @printf
    ("%lld\n" / "%g\n" / "%s\n").  Multi-argument print calls one @printf
    with a per-call format string joining the per-kind directives with
    single spaces ("%lld %s\n" etc.), matching the interpreter's
    ``print(*args)`` (sep=" ").  Note: float and bool formatting can
    differ from the Python interpreter's str() ("%g" vs repr; bools are
    kind-erased to i64 and print as 1/0, not "True"/"False"); differential
    tests should print ints/strings or compare via comparisons.
  * every metaxu function symbol is prefixed ``mx_`` (and sanitized to
    [A-Za-z0-9_]) so user functions named main/printf/abs cannot collide
    with libc; the native entry wrapper lives in llvm_run.py.

ENUM VARIANTS (increment 3) are tagged unions:
  * each enum E gets ``%enum.E = type { i64, [N x i64] }`` — an integer tag
    plus payload slots sized to the LARGEST variant of E (every scalar
    field kind — i64/double/ptr — is 8 bytes in this backend, so [N x i64]
    is exact storage; slot loads/stores use the slot's inferred scalar
    type through the opaque payload pointer).  ``make_variant`` with an
    empty enum name (front-end gap) uses the shared ``%enum.anon`` type.
  * variant names map to dense integer tags at emission time via ONE
    module-wide table: every variant name occurring in a ``make_variant``
    op or a compiled-pattern tag comparison is collected, sorted, and
    numbered from 0.  Equal names get equal tags and distinct names get
    distinct tags, which preserves the interpreter's string-equality
    semantics for tag tests (cross-enum tag comparisons cannot occur in
    well-typed MIR).  The mapping is documented in a module comment.
  * compiled match arms compare ``variant_tag`` results against CONST
    STRING variant names (see lower_hir_to_mir.compile_pattern).  A const
    string whose every use is an ==/!= against a variant_tag result is a
    "tag literal" and is emitted as its integer tag — the string never
    reaches native code.  A tag-vs-string comparison that does not fit
    that shape demotes via kind conflict rather than guessing.
  * variant values are aggregates with the same value semantics as
    structs: entry-block alloca per variant-kinded variable, aggregate
    copies, ptr + callee byval-copy across calls, sret-style returns.
    There is no @global form (make_variant carries no locality).

REFINED ENUM KINDS (increment 6) type payload slots per variant and per
VALUE, replacing the old module-wide per-(enum, slot) cells:
  * an enum value's kind carries a payload REFINEMENT recording the actual
    representation of every variant any flow into it can construct:
    ``enum:Option{None:;Some:struct:Node}`` (canonical form: variants
    sorted, slots comma-separated).  make_variant refines its destination
    with its own arg kinds; refinements join pointwise per (variant, slot)
    along every dataflow edge (copies, params/returns, struct fields,
    captures) — the same monotone fixpoint as all other kinds.  Slot kinds
    inside a refinement are NAME-ONLY (a nested enum appears as ``enum:F``
    with no braces), which keeps refinement strings finite for recursive
    enums.
  * ``variant_field`` reads its variant's slot kind out of the BASE
    VALUE's refinement (the op names its variant, see above).  Different
    variants — or different instantiations of one variant in different
    values — can now disagree about a slot index; each value knows its own
    representation.  Reading a variant absent from the refinement is a
    dead arm (the refinement lists every constructible variant, so the
    guarding tag test can never pass): scalar results emit a typed zero,
    never a bogus aggregate read.
  * SOUNDNESS: every dataflow edge except enum-in-enum nesting unifies
    kinds two-way, so a reader's refinement is exactly the join over its
    writers; the post-fixpoint make_variant check demotes any site whose
    stored kind differs from that join ("heterogeneous payload slot ...
    no coercion"), which now fires only for genuinely MERGED mixed flows
    (e.g. one variable holding both Some(3) and Some(node)), not for
    disjoint uses.
  * enum-in-enum nesting is the one boundary where a value's refinement is
    stripped (slot kinds are name-only): extraction therefore assumes the
    CANONICAL representation — module-wide per-(enum, variant, slot) cells
    joining every make_variant store — and demotes when any store
    disagrees with the join (`mixed` cells: the nested enum's slot
    representation is instantiation-dependent and was lost at the boxing
    boundary).
  * the union layout is unchanged and instantiation-independent:
    ``%enum.E = { i64 tag, [N x i64] }`` with N the max payload arity over
    all variants; every slot is an 8-byte cell (scalars inline, aggregates
    as boxed pointers), so all refinements of one enum share one LLVM
    type.  Boxing decisions are per (variant, slot) refinement kind.  The
    emitted type comment documents each variant's canonical slot kinds and
    flags mixed slots.

BOXED AGGREGATE PAYLOADS (increment 4; since increment 6 decided per
variant and per value, see REFINED ENUM KINDS below): a payload slot whose
kind
is itself a struct or enum stores a HEAP POINTER to a boxed copy of the
aggregate (the 8-byte slot holds the ptr), which makes recursive enums
(linked lists, trees) representable with a finite layout:
  * ``make_variant`` mallocs the box (exact recursive size) and copies the
    aggregate value in; ``variant_field`` loads the pointer and copies the
    aggregate out into the destination's own storage — value semantics are
    preserved at both edges.
  * boxes are WRITE-ONCE: the only store through a box pointer is the fill
    at the make_variant site.  Aggregate copies (variable defs, byval
    params, sret returns, env captures, boxing itself) copy the pointer
    shallowly, so boxes are freely shared — which is observationally
    equivalent to the interpreter's value semantics precisely because no
    native code path ever mutates a filled box (MIR has no payload-set op,
    and field_set copies its base aggregate wholesale).
  * FREE STRATEGY (the documented, prominently stated choice): boxes LEAK
    BY DESIGN this increment.  Shallow sharing means box ownership is not
    unique, so any per-value free would need deep copies at every
    aggregate copy to avoid double-frees; instead no box is ever freed.
    This is provably sound (a leak can never be a use-after-free or
    double-free); ASan tests for box programs therefore assert with
    ``detect_leaks=0`` — they prove no UAF/double-free, not leak-freedom.
    The @global struct malloc/free protocol is UNCHANGED and remains fully
    leak-clean; struct blocks are freed at frame exit while any boxes
    referenced from their field regions are simply leaked.
  * a payload slot holding a closure still demotes (its env pointer may
    aim at a stack frame the boxed value could outlive).

CLOSURES (increment 3) are fn-pointer + stack-environment pairs:
  * a closure value is ``%mx.closure = type { ptr, ptr }`` — the lambda's
    function pointer and a pointer to an environment struct.  Each lambda
    L with captures gets ``%env.L = type { ... }`` holding its captured
    values IN CAPTURE-LIST ORDER (kinds inferred like struct fields;
    aggregate captures are copied in whole — capturing another closure
    demotes).
  * ``make_closure`` gets one entry-block env alloca per op site, stores
    the captured values (eager, by value — matching the interpreter) and
    then the {fn, env} pair into the closure variable's own pair alloca.
  * the lambda is emitted with a leading ``ptr %cl.env`` parameter (after
    ``%agg.ret`` when sret) and re-loads every capture from the env struct
    in its prelude; its other free names still demote.
  * a call whose callee is a local variable of closure kind loads fn+env
    from the pair and calls ``fn(env, args...)`` — typed with the lambda's
    signature when the kind is pinned to one non-participating lambda
    (kind inference pins ``closure:L`` kinds like any other kind), or
    through the word-uniform ABI below when the callee is dynamic or a
    participant.
  * closures may be passed DOWN as call arguments (ptr + byval pair copy;
    the env outlives the callee because the creating frame is still
    live).

HEAP CLOSURE ENVIRONMENTS (increment 4): a lambda is marked heap-env when
its closure could outlive (or alias across re-executions of) the creating
site:
  * RETURNED closures: after the module kind fixpoint, any function whose
    return kind is ``closure:L`` marks L heap-env.  Kind propagation makes
    this complete for the emitted subset: a pair that reaches a caller
    reaches it through some emitted function's return (storing a pair in
    a struct field / enum payload / another env still demotes the storing
    function, so no other upward path exists).
  * LOOPED sites: a make_closure inside a CFG cycle marks L heap-env —
    each execution mallocs a FRESH env, so earlier pair copies never alias
    the new one (this replaces the increment-3 demotion).
  * a heap env is malloc'd at the make_closure site and NEVER FREED — the
    same leak-by-design contract as payload boxes (an immortal env can
    never dangle; that is what makes the escape analysis need only be
    conservative, never precise).  Non-escaping lambdas keep their
    zero-cost stack envs.
  * still demoted honestly: storing a closure in a struct field or variant
    payload.

INDIRECT CLOSURE CALLS (increment 13) — function-valued parameters and
one-call-site-reached-by-many-lambdas shapes:
  * two DIFFERENT lambdas of the same arity meeting at one flow point
    join to the DYNAMIC closure kind ``closure:*{L1,L2}`` (the canonical
    sorted member set) instead of conflicting; mismatched arities still
    conflict (the interpreter's zip-binding would leave parameters
    unbound at their first use too).
  * WORD-UNIFORM ABI: a lambda PARTICIPATES in indirect calls when its
    closure leaves simple local flow — it reaches a function parameter
    position, it is merged into a dynamic kind anywhere, or it is
    captured into an env (handle-site or another closure's).  A
    participating lambda whose whole signature is word-encodable
    (i64/f64/str/vec/vector/rawptr) is emitted with the uniform native
    signature ``i64 (ptr env, i64 args...)`` — parameter words decoded in
    the prelude, the return value encoded at every ret, using the
    effect-boundary word conventions (f64 bitcast, pointers ptrtoint).
    Non-participating lambdas keep their typed signatures (the
    comprehension/SIMD and aggregate paths are untouched).
  * an indirect call site (dynamic callee kind, or a pinned participant)
    loads {fn, env} and calls the fn pointer with word-encoded arguments,
    decoding the i64 result back to the site's inferred kind.  The site's
    argument/result kinds unify two-way with EVERY member lambda's
    signature during the module fixpoint, so all members and all sites
    agree on the typed kinds under the words.
  * env-captured closures: a closure pair may now be captured BY VALUE
    into a handle-site env or another closure's env (this is how
    std.stream's handler cases call ``f`` and how take/map/filter's
    thunks hold ``producer``); every member lambda of an env-captured
    closure kind is marked heap-env, so the pair's env pointer aims at an
    immortal block and can never dangle wherever the capturing env
    travels.
  * still demoted honestly: wrong-arity members reaching one site,
    dynamic closures as comprehension bodies, and closures crossing the
    effect boundary as perform/resume values (env capture is the
    supported route).

AGGREGATES IN INDIRECT CALL SIGNATURES (increment 16) — the word-uniform
ABI above was scalar-only; struct/enum aggregates now cross it through
the SAME write-once BOUNDARY BOXES increment 14 uses at the effect
boundary:
  * an aggregate ARGUMENT at an indirect site mallocs a fresh copy
    (``to_word``: malloc + whole-aggregate copy in) and travels as that
    pointer word.  The word-uniform lambda's prelude decodes the word to
    a ``ptr %a.<p>``, at which point the ORDINARY aggregate-parameter
    convention — the byval copy-out into the callee's own storage, or
    the elide-copy read-through — is exactly the copy-out contract.
  * an aggregate RETURN drops sret entirely: the lambda keeps the
    ``i64 (ptr env, i64 args...)`` signature, mallocs a box at each ret,
    copies its result in and returns the pointer word (encoded BEFORE
    the frame's frees, so a heap-backed result is copied while alive);
    the caller copies out of the box into its own storage.
  * boxes are immortal (never freed, leak by design), so nothing can
    dangle across the call in either direction, and both edges keep
    value semantics — the receiver always owns a private copy.
  * FAST PATHS ARE PRESERVED.  Scalar indirect calls still pass raw
    words and allocate nothing.  An AGGREGATE-signatured lambda takes
    the uniform ABI only when it is a member of some DYNAMIC closure
    kind — only then must it agree on one native signature with another
    lambda.  A lambda pinned at every one of its sites keeps its TYPED
    signature (ptr params, sret return) and the typed call path, so a
    pinned aggregate closure call costs zero allocation.
  * KIND AGREEMENT is the pre-existing two-way fixpoint: a site's
    argument/result kinds unify with every member lambda's signature, so
    a member disagreeing on whether a position is a struct, an enum or a
    scalar joins to ``conflict`` and the site demotes.
  * still demoted honestly, each with its own reason: closure PAIRS and
    konts in indirect args/returns (a boxed pair would carry an env
    pointer whose lifetime the box cannot vouch for), aggregates with an
    infinite layout (no box can be sized), and ``@mut`` aggregate
    parameters — their write-back copies out through the CALLER's
    pointer, which is a box the caller drops, so the mutation would be
    silently lost.

BOUNDARY-BOX TRAFFIC REDUCTION (increment 17) — a pure optimization over
the two rounds above.  Boxing at every crossing was correct but wasteful:
an aggregate resume result was boxed by the dispatcher, copied out, then
boxed AGAIN at the next hop, and a perform in a loop malloc'd once per
iteration.  Three elisions remove the waste; anything not provable keeps
boxing, because a leaked box is sound where a dangling pointer is not.
The whole argument rests on the WRITE-ONCE BOX INVARIANT: a boundary box
is malloc'd, filled once before its pointer leaves the producer, never
written again, and never freed by any emitted path (``emit_frees``
releases only @global blocks, provably local Vecs, unique enum payload
boxes and owned strings).

  * (1) DOUBLE-BOX ELISION.  A value whose single def RECEIVES a boundary
    word — a perform result, a resume result, a handle value, or the
    result of a word-uniform indirect call — keeps the producer's pointer
    as a BOX VIEW (``bbox_view``) instead of copying the aggregate out,
    and handing that value to the NEXT boundary passes the same pointer
    through instead of malloc'ing a byte-identical second box.  Sound
    because the box is immortal (so it cannot dangle, not even when an
    abort tears down a parked coroutine stack), write-once (handle-scope
    subfunctions never copy out through a param pointer, and the word ABI
    refuses @mut aggregate params), and the receiver is read-only
    (``read_only_agg``: one def, never at a callee write-back position),
    so the box's bytes ARE this value, forever.
    Handler cases and handle bodies with an aggregate result now RETURN
    that word themselves (the BOUNDARY-WORD ABI, ``i64 (...)``) instead
    of sret-filling a box the site shim malloc'd, so the shims allocate
    nothing at all and the fold-shaped ``resume(...)`` case — whose value
    is already a box — allocates nothing either.
    THE GUARD: views of enum PAYLOAD boxes (elision (b), from
    ``variant_field``) are never re-exported across a boundary.  Those
    boxes are freed at frame exit when ``_unique_box_enums`` proves sole
    ownership, so their pointer must not outlive this frame; they read
    through the box and box a copy when they cross.
  * (2) READ-ONLY AGGREGATE ARGUMENTS.  An aggregate argument at an
    indirect closure call passes a pointer to the CALLER'S existing
    storage rather than a fresh box, whenever every statically-possible
    callee (one pinned member, or every member of a dynamic kind) is
    word-uniform and leaves that position out of its write-back set —
    reusing ``writeback_map``, the analysis the direct-call elide-copy
    pass already runs, rather than a second one.  Sound because nothing
    writes through the pointer and an indirect call is an ordinary
    synchronous call on this stack, so the storage outlives the callee's
    frame; the callee cannot re-export the pointer either, since a
    parameter is never a ``bbox_view``.  Effect-boundary arguments do NOT
    get this: they cross to another coroutine stack that an abort may
    tear down, so they keep their immortal box.
  * (3) LOOP-INVARIANT BOXES.  A value whose only def is a block-0 op (or
    a parameter), that no write-back can reach, and that is boxed at a
    boundary site inside a CFG cycle gets ONE box, filled at the end of
    block 0 (which dominates every block) and reused at every site.
    Sharing is invisible: the bytes never change and the box is immortal.
    A reassigned loop accumulator keeps its per-iteration box.

Per-function value kinds (i64 / f64 / str / struct:T / enum:E /
closure:L) are inferred exactly
in the spirit of codegen_clif's i64->f64 promotion: every value defaults
to i64 and is promoted by a monotone fixpoint (module-wide over function
signatures and struct field kinds).  Irreconcilable kinds ("conflict")
demote the function rather than emitting wrong code.

MIR is not SSA: lowering re-assigns result variables (if/match result
slots, loop counters).  Mirroring codegen_clif's classification, every
non-struct MIR variable that is multiply-assigned — or defined outside the
entry block and used in another block — gets an entry-block ``alloca``
with load/store around uses; single-assignment values (and everything
defined in the entry block, which dominates all blocks) map to SSA
registers.  mem2reg removes the allocas at -O2.

Public API: ``emit_llvm(mir_funcs) -> str`` and ``mangle(name) -> str``.
"""

from __future__ import annotations

import re
import struct as _structmod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from .mir import MirFunc
from .cps_frames import is_suspending
from .tile_shape_check import TILE_ARITY
from .effect_tail import tail_resume_ids as _tail_resume_ids
from .desugar import IMPL_SEP, parse_impl_method_name
from .hir import (BUILTIN_CALL_PREFIX, STATIC_CALL_PREFIX, TRAIT_CALL_PREFIX,
                  is_tuple_struct as _is_tuple_struct)

# Value kinds -----------------------------------------------------------------

I64 = "i64"
F64 = "f64"
STR = "str"
# An opaque effect continuation (`mx_k*`): the trailing `__k` parameter of
# a handler-case subfunction.  A pointer, but NOT a word kind — it may only
# flow from the case's own param into its resume ops; anywhere else demotes.
KONT = "kont"
# A raw C pointer (increment 9): 8-byte scalar `ptr` storage, shallow
# copies — exactly C semantics.  Produced by the extern FFI calls
# (malloc/realloc/memcpy/fopen) and `as_ptr`; `null` (MIR `const None`)
# emits as the `null` ptr constant once unification promotes its variable
# to this kind; ==/!= compare pointer identity.  Deliberately NOT a word
# kind: rawptr values may not enter Vec elements or cross the effect
# boundary (those flows demote honestly).
PTR = "rawptr"
CONFLICT = "conflict"
_STRUCT_PREFIX = "struct:"
_ENUM_PREFIX = "enum:"
_CLOSURE_PREFIX = "closure:"
# A growable Vec value: an opaque `mx_vec*` heap pointer (8 bytes) with
# IDENTITY semantics, parameterized by its unified element kind
# ("vec:i64" / "vec:f64" / "vec:str" / "vec:vec:i64" ...).  Copies are
# shallow pointer copies, exactly matching the interpreter's MxVec (every
# copy aliases the one shared vector).  Elements are opaque 8-byte words in
# the native runtime; the element kind decides the bitcast at push/pop/get.
_VEC_PREFIX = "vec:"
# A FIXED-SIZE vector value `vector[T, N]` (increment 10): an opaque
# `mx_fvec*` pointer to an IMMUTABLE length-prefixed word block
# { i64 len, [len x i64] } in the native runtime, parameterized by the
# unified element kind ("vector:f64", "vector:vector:f64" for matrices...).
# The interpreter's MxVector has VALUE semantics; since no operation ever
# mutates a filled block (construction fills it before the pointer is
# shared), shallow pointer copies are observationally identical to value
# copies — the same write-once argument as boxed enum payloads.  Blocks
# LEAK BY DESIGN (shallow sharing makes ownership non-unique).  Note
# "vector:" does not collide with the "vec:" prefix test ("vector:f64"
# does not start with "vec:").
_FVEC_PREFIX = "vector:"

_LLTY = {I64: "i64", F64: "double", STR: "ptr"}
_SCALARS = (I64, F64, STR)

# The uniform closure value: { fn pointer, env pointer }.
_CLOSURE_PAIR_TY = "%mx.closure"

# Binop tables ----------------------------------------------------------------

_ARITH_INT = {"+": "add", "-": "sub", "*": "mul", "/": "sdiv", "%": "srem"}
_ARITH_FLT = {"+": "fadd", "-": "fsub", "*": "fmul", "/": "fdiv", "%": "frem"}
_CMP_INT = {"==": "eq", "!=": "ne", "<": "slt", "<=": "sle", ">": "sgt", ">=": "sge"}
_CMP_FLT = {"==": "oeq", "!=": "one", "<": "olt", "<=": "ole", ">": "ogt", ">=": "oge"}
_LOGIC = {"&&": "and", "||": "or", "and": "and", "or": "or"}
# Bitwise operators (Int-only, i64 two's complement).  `>>` is ARITHMETIC
# (`ashr`), matching Python's sign-extending `>>` in mir_interp.  The two
# shifts do NOT emit a bare `shl`/`ashr`: LLVM makes an out-of-range shift
# count poison while the interpreter raises, so the count goes through
# @mx_shift_check first (see emit_shift_check).
_BITWISE_INT = {"&": "and", "|": "or", "^": "xor", "<<": "shl", ">>": "ashr"}
_SHIFT_OPS = frozenset({"<<", ">>"})
_SUPPORTED_BINOPS = (set(_ARITH_INT) | set(_CMP_INT) | set(_LOGIC)
                     | set(_BITWISE_INT))

# Builtins --------------------------------------------------------------------

_PRINT_BUILTINS = {"print", "println"}
_MATH_EXTERNS = {"sqrt", "sin", "cos"}  # double -> double libc functions
_INLINE_BUILTINS = {"neg", "not", "bnot"}

# Vec/string builtins now lowered to the NATIVE runtime (metaxu_rt.c, linked
# by llvm_run): these mirror the interpreter's builtins exactly.  NAME
# PRECEDENCE (docs/name_precedence.md): a call reaches these only when
# _builtin_name says so -- a module function of the same name WINS for plain
# calls, while a method-position `__builtin$m` call always lands here; local
# closure variables still shadow both.
#   Vec.new      -> mx_vec_new          push  -> mx_vec_push
#   pop          -> mx_vec_pop          len   -> mx_vec_len / mx_str_len
#   __index_get  -> mx_vec_get          __vec_lit -> mx_vec_new + pushes
#   to_string / int_to_str -> mx_i64_to_str / mx_f64_to_str / identity(str)
_NATIVE_RT_CALLS = {"Vec.new", "push", "pop", "len", "to_string",
                    "int_to_str", "__index_get", "__vec_lit",
                    # Fixed-vector builtins (increment 10) — mx_fvec_*:
                    "__vec_dim", "__vec_zeros", "__vec_filled",
                    "__vec_comprehension", "__range", "__slice_get",
                    "__cast",
                    # Index assignment + zip iteration (increment 12):
                    # __index_store is the store-back form of `v[i] = x`
                    # (mx_vec_set in place on a Vec; mx_fvec_set_copy
                    # functional update on a vector[T,N] place),
                    # __index_set the in-place-only form (Vec receivers
                    # only — immutable receivers demote at compile time,
                    # the interpreter's error made static), and __zip the
                    # lockstep pair iterable of zip comprehensions
                    # (mx_fvec_zip_map).
                    "__index_store", "__index_set", "__zip",
                    # Tiles (docs/gpu_tiles.md Stage 0): dotted statics ->
                    # mx_tile_* (metaxu_rt.c); shapes ride `tile:` kinds.
                    "Tile.zeros", "Tile.filled", "Tile.arange",
                    "Tile.from_vec", "Tile.to_vec", "Tile.add", "Tile.mul",
                    "Tile.scale", "Tile.dot", "Tile.sum", "Tile.transpose",
                    "Tile.get", "Tile.rows", "Tile.cols", "Tile.load",
                    "Tile.load_or", "Tile.store", "Tile.store_clipped",
                    "Tile.load_rows", "Tile.store_rows"}

# Extern C symbols the interpreter shims over its simulated heap
# (mir_interp._ffi_*): natively these are DIRECT calls to the real libc
# functions with their C signatures — (param kinds, result kind).  free
# and fclose "return" unit/int into an i64 destination.
_EXTERN_C_SIGS = {
    "malloc": ((I64,), PTR),
    "free": ((PTR,), I64),          # C void; dst is unit 0
    "memcpy": ((PTR, PTR, I64), PTR),
    "realloc": ((PTR, I64), PTR),
    "fopen": ((PTR, PTR), PTR),
    "fclose": ((PTR,), I64),        # C int: declared i32, sext'd to i64
}
# Interpreter FFI shims with dedicated inline lowerings (no C symbol):
# as_ptr (identity on str / mx_vec_as_bytes snapshot on vec), ptr_read /
# ptr_write (inline i8 loads/stores).
_FFI_SHIMS = {"as_ptr", "ptr_read", "ptr_write"}
_FFI_CALLS = set(_EXTERN_C_SIGS) | _FFI_SHIMS

# Interpreter builtins that trait dispatch can fall back to when no user
# impl matches the receiver type (mir_interp._dispatch_trait_call step 2).
# The FFI names are included because they ARE interpreter builtins: a trait
# call falling through to them must never resolve to a same-named plain
# module function instead (the interpreter would pick the builtin).  Trait
# dispatch order is UNCHANGED by the plain-call precedence flip -- see
# docs/name_precedence.md section 5.
_TRAIT_BUILTIN_FALLBACK = {"to_string", "int_to_str", "len", "push", "pop",
                           "sqrt", "sin", "cos", "assert"} | _FFI_CALLS

# Callees still implemented only by the interpreter runtime (demote).
# `__vec_lit` is native since increment 9 (checked before these prefixes);
# `__static$` calls resolve statically (see _resolve_static_call).
_RUNTIME_PREFIXES = ("__vec_", "__index_", "__slice_", "__range")
_RUNTIME_NAMES = {"type_of", "assert_eq"}

# Element-wise vector arithmetic opcodes for mx_fvec_binop (metaxu_rt.h).
_FVEC_BINOP_CODES = {"+": 0, "-": 1, "*": 2, "/": 3, "%": 4}

# Effect-op fallback function name prefixes (mir_interp's resolution: an op
# performed with no handler in scope first tries its `with SYMBOL` runtime
# mapping, then its declared `= expr` default).
_EFFECT_DEFAULT_PREFIX = "__effect_default$"
_EFFECT_RUNTIME_PREFIX = "__effect_runtime$"

# The body of an __effect_runtime$E$op thunk calls
# __mx_effect_runtime$SYMBOL (hir.EFFECT_RUNTIME_CALL_PREFIX); these
# SYMBOLs lower to the pthreads-backed C primitives in metaxu_threads.c
# (docs/threads_runtime.md).  value: (C symbol, arity).  Thread[T]/Mutex
# extern-type values are opaque i64 handle words; lock/unlock "return"
# the unit word 0.  A `with SYMBOL` outside this table demotes with a
# reason (the interpreter errors loudly at perform time there too).
_EFFECT_PRIMITIVE_CALL_PREFIX = "__mx_effect_runtime$"
_EFFECT_PRIMITIVES = {
    "EFFECT_SPAWN": ("mx_thread_spawn", 1),         # (closure) -> handle
    "EFFECT_JOIN": ("mx_thread_join", 1),           # (handle) -> result word
    "EFFECT_MUTEX_CREATE": ("mx_mutex_create", 0),  # () -> handle
    "EFFECT_MUTEX_LOCK": ("mx_mutex_lock", 1),      # (handle) -> unit
    "EFFECT_MUTEX_UNLOCK": ("mx_mutex_unlock", 1),  # (handle) -> unit
}

# The synthesized module-constant initializer (hir.py): its globals_decl
# names become module-level LLVM globals `@mx_g_<name>`; the native entry
# wrapper (llvm_run) calls it before the entry point, exactly the
# interpreter's _ensure_globals.
_MODULE_INIT = "__module_init"


def _mx_global(name: str) -> str:
    """The LLVM global symbol backing a module constant."""
    return "@mx_g_" + _sanitize(name)

# Native runtime symbol signatures (metaxu_rt.h ABI): name -> (ret, params).
_RT_SIGS = {
    "mx_raise": ("void", ("ptr",)),   # _Noreturn; follows an `unreachable`
    "mx_vec_new": ("ptr", ()),
    "mx_vec_push": ("void", ("ptr", "i64")),
    "mx_vec_pop": ("i64", ("ptr",)),
    "mx_vec_len": ("i64", ("ptr",)),
    "mx_vec_get": ("i64", ("ptr", "i64")),
    "mx_vec_set": ("void", ("ptr", "i64", "i64")),
    # Cold-path terminators for the inline Vec fast paths: reached only
    # when an inlined check failed; re-run the op's canonical checks and
    # raise/abort with the byte-identical diagnostic.  Their declares
    # carry the attributes in _RT_ATTRS — noreturn plus a narrow memory
    # contract — so a never-taken miss branch does not clobber the
    # surrounding loop's hoisted header loads.
    "mx__vec_get_fail": ("void", ("ptr", "i64")),
    "mx__vec_set_fail": ("void", ("ptr", "i64")),
    "mx__vec_pop_fail": ("void", ("ptr",)),
    "mx__vec_len_fail": ("void", ("ptr",)),
    "mx_vec_free": ("void", ("ptr",)),
    # Contention marking (docs/contention_as_permission.md): called at the
    # real-spawn path (inside the EFFECT_SPAWN runtime thunk) for every
    # vec the spawned closure captures, directly or through struct fields.
    "mx_vec_mark_contended": ("void", ("ptr",)),
    "mx_str_concat": ("ptr", ("ptr", "ptr")),
    "mx_str_len": ("i64", ("ptr",)),
    "mx_i64_to_str": ("ptr", ("i64",)),
    "mx_f64_to_str": ("ptr", ("double",)),
    "mx_str_eq": ("i64", ("ptr", "ptr")),
    "mx_str_free": ("void", ("ptr",)),
    "mx_vec_as_bytes": ("ptr", ("ptr",)),
    # Shift-count guard: aborts when the count is outside 0..63, which is
    # what the interpreter's InterpError does.  A bare `shl`/`ashr` would be
    # POISON there, i.e. the same program with two behaviours.
    "mx_shift_check": ("i64", ("i64", "i64")),
    # Fixed-size vectors (immutable mx_fvec blocks; increment 10).
    "mx_fvec_new": ("ptr", ("i64",)),
    "mx_fvec_len": ("i64", ("ptr",)),
    "mx_fvec_get": ("i64", ("ptr", "i64")),
    "mx_fvec_init": ("void", ("ptr", "i64", "i64")),
    "mx_fvec_filled": ("ptr", ("i64", "i64")),
    "mx_fvec_range": ("ptr", ("i64", "i64")),
    "mx_fvec_dim": ("i64", ("ptr", "i64", "i64")),
    "mx_fvec_slice": ("ptr", ("ptr", "i64", "i64", "i64", "i64")),
    "mx_fvec_binop": ("ptr", ("i64", "i64", "i64", "i64", "i64", "i64")),
    "mx_fvec_promote": ("ptr", ("ptr",)),
    "mx_fvec_map": ("ptr", ("ptr", "ptr", "ptr", "i64")),
    "mx_fvec_set_copy": ("ptr", ("ptr", "i64", "i64")),
    "mx_fvec_zip_map": ("ptr", ("ptr", "ptr", "ptr", "ptr", "i64")),
    "mx_fvec_to_str": ("ptr", ("ptr", "i64", "i64")),
    "mx_fvec_as_bytes": ("ptr", ("ptr",)),
    # Threads runtime (metaxu_threads.c, docs/threads_runtime.md): opaque
    # i64 handle words; spawn takes the closure's {fn, env} split into two
    # pointer words (the env is compiler-forced heap/immortal).
    "mx_thread_spawn": ("i64", ("ptr", "ptr")),
    "mx_thread_join": ("i64", ("i64",)),
    "mx_mutex_create": ("i64", ()),
    "mx_mutex_lock": ("i64", ("i64",)),
    "mx_mutex_unlock": ("i64", ("i64",)),
    # Algebraic effects runtime (metaxu_effects.c).
    "mx_handle": ("i64", ("ptr", "ptr", "ptr", "ptr", "ptr", "ptr", "ptr",
                          "i64")),
    "mx_perform": ("i64", ("ptr", "ptr", "ptr", "i64")),
    "mx_perform_or_default": ("i64", ("ptr", "ptr", "ptr", "i64", "ptr",
                                      "ptr")),
    "mx_resume": ("i64", ("ptr", "i64")),
    "mx_resume_tail": ("i64", ("ptr", "i64")),
    # Tiles (docs/gpu_tiles.md Stage 0, metaxu_rt.c): shapes are static in
    # `tile:` kinds; is_f64 selects element arithmetic.
    "mx_tile_zeros": ("ptr", ("i64", "i64")),
    "mx_tile_filled": ("ptr", ("i64", "i64", "i64")),
    "mx_tile_arange": ("ptr", ("i64", "i64")),
    "mx_tile_from_vec": ("ptr", ("ptr", "i64", "i64")),
    "mx_tile_to_vec": ("ptr", ("ptr",)),
    "mx_tile_add": ("ptr", ("ptr", "ptr", "i64")),
    "mx_tile_mul": ("ptr", ("ptr", "ptr", "i64")),
    "mx_tile_scale": ("ptr", ("ptr", "i64", "i64")),
    "mx_tile_dot": ("ptr", ("ptr", "ptr", "i64")),
    "mx_tile_sum": ("i64", ("ptr", "i64")),
    "mx_tile_transpose": ("ptr", ("ptr",)),
    "mx_tile_get": ("i64", ("ptr", "i64", "i64")),
    "mx_tile_rows": ("i64", ("ptr",)),
    "mx_tile_cols": ("i64", ("ptr",)),
    "mx_tile_to_str": ("ptr", ("ptr", "i64")),
    "mx_tile_load": ("ptr", ("ptr", "i64", "i64", "i64")),
    "mx_tile_load_or": ("ptr", ("ptr", "i64", "i64", "i64", "i64")),
    "mx_tile_store": ("void", ("ptr", "i64", "ptr")),
    "mx_tile_store_clipped": ("void", ("ptr", "i64", "ptr")),
    "mx_tile_load_rows": ("ptr", ("ptr", "i64", "i64", "i64", "i64",
                                  "i64")),
    "mx_tile_store_rows": ("void", ("ptr", "i64", "i64", "ptr")),
    # Delimited failure recovery (try/catch, metaxu_effects.c).
    "mx_try": ("i64", ("ptr", "ptr", "ptr", "ptr")),
}

# Attribute suffixes for _RT_SIGS declares that carry more contract than a
# bare signature.  The fail terminators never return, and the only memory
# they WRITE is the raise/abort machinery's own state — memory the module
# never touches through pointers it holds (LLVM "inaccessiblemem"; every
# other runtime declare is unmarked and so conservatively clobbers it).
# That narrow contract is load-bearing: it is what lets LICM keep a hot
# loop's Vec header loads hoisted across the never-taken miss branch.
_RT_ATTRS = {
    "mx__vec_get_fail":
        " cold noreturn memory(read, inaccessiblemem: readwrite)",
    "mx__vec_set_fail":
        " cold noreturn memory(read, inaccessiblemem: readwrite)",
    "mx__vec_pop_fail":
        " cold noreturn memory(read, inaccessiblemem: readwrite)",
    "mx__vec_len_fail":
        " cold noreturn memory(read, inaccessiblemem: readwrite)",
}

# Native effect-op argument/parameter limit (metaxu_effects.h
# MX_EFFECT_MAX_ARGS): performs or handler cases beyond it demote.
_MAX_EFFECT_ARGS = 8

_I64_MIN, _I64_MAX = -(2 ** 63), 2 ** 63 - 1

_HEADER = (
    "; LLVM IR emitted by metaxu codegen_llvm (direct subset)\n"
    "; conventions: ints/bools/unit -> i64 (unit = 0); floats -> double;\n"
    ";   strings -> ptr to private constant byte arrays; cmp results zext to i64;\n"
    ";   &&/|| normalize operands with icmp ne 0 (truthiness, not bitwise);\n"
    ";   / and % are sdiv/srem (trunc toward zero; interpreter matches);\n"
    ";   local structs -> %struct.T entry allocas + GEP (value semantics,\n"
    ";   whole-aggregate copies; zero heap management -- the frame owns them);\n"
    ";   @global structs -> entry-block malloc(recursive layout size) + GEP\n"
    ";   on the heap\n"
    ";   pointer, freed on every ret path: sound because value semantics means\n"
    ";   the storage pointer never escapes the frame (aggregates cross frames\n"
    ";   by copy).  Any storage whose lifetime cannot be proven leaks by\n"
    ";   design rather than risking a double-free/use-after-free;\n"
    ";   struct params pass as ptr + callee byval-copy into own storage\n"
    ";   (a borrow-informed increment can elide the copy for @const params);\n"
    ";   rebound @mut/receiver struct params copy OUT through the caller's\n"
    ";   pointer on ret (interpreter write-back parity; plain params keep\n"
    ";   value semantics; lambdas copy out exactly their @mut params);\n"
    ";   struct returns are sret-style: caller passes its result slot as a\n"
    ";   leading ptr %agg.ret arg, callee copies the aggregate in, rets void;\n"
    ";   print/println route by operand type to @metaxu_print_{i64,f64,str};\n"
    ";   multi-arg print joins per-kind printf directives with spaces;\n"
    ";   metaxu symbols are prefixed mx_ to stay clear of libc names;\n"
    ";   enums -> %enum.E = { i64 tag, [N x i64] payload } tagged unions,\n"
    ";   variant names mapped to dense integer tags (module-wide table in a\n"
    ";   comment below); pattern tag tests compare integers, never strings;\n"
    ";   payload slots are typed PER VARIANT and PER VALUE (each enum\n"
    ";   value's kind carries a refinement recording its constructible\n"
    ";   variants' slot representations), so one slot index may hold\n"
    ";   different types in different variants or instantiations; merged\n"
    ";   flows that mix representations inside one value still demote;\n"
    ";   closures -> %mx.closure = { ptr fn, ptr env } pairs over per-lambda\n"
    ";   stack %env.L structs; lambdas take env as a leading param;\n"
    ";   escaping closures (returned / created in a loop) malloc their env\n"
    ";   at the site instead (heap env, leaked by design: immortal envs\n"
    ";   cannot dangle);\n"
    ";   struct fields holding structs/enums are laid out INLINE in the\n"
    ";   parent type (recursive GEPs, still stack-based, value semantics);\n"
    ";   enum payload slots holding aggregates store a HEAP POINTER to a\n"
    ";   write-once boxed copy (make_variant boxes in, variant_field copies\n"
    ";   out).  FREE STRATEGY: payload boxes and heap closure envs LEAK BY\n"
    ";   DESIGN (never freed) -- shallow pointer sharing makes ownership\n"
    ";   non-unique, and a leak is provably sound where a free is not.\n"
    ";   @global struct blocks are still freed at frame exit as before;\n"
    ";   Vec values -> opaque mx_vec* pointers (native runtime metaxu_rt.c,\n"
    ";   linked by llvm_run) with IDENTITY semantics (shallow ptr copies,\n"
    ";   matching the interpreter's MxVec); elements are opaque 8-byte\n"
    ";   words (f64 bitcast, str/vec ptrtoint) through mx_vec_push/pop/get;\n"
    ";   len routes by kind to mx_vec_len/mx_str_len; to_string routes to\n"
    ";   mx_i64_to_str/mx_f64_to_str/identity; string + is mx_str_concat,\n"
    ";   ==/!= is mx_str_eq.  Concat/to_string results leak by design; a\n"
    ";   Vec is mx_vec_free'd at frame exit only when provably\n"
    ";   non-escaping, otherwise it leaks by design too;\n"
    ";   __trait$ method calls are resolved statically against the\n"
    ";   receiver's inferred kind (impl fn -> builtin -> plain fn),\n"
    ";   mirroring the interpreter's runtime dispatch;\n"
    ";   algebraic effects run on the native effects runtime\n"
    ";   (metaxu_effects.c): handle_scope -> env fill + mx_handle over a\n"
    ";   per-site body thunk + op dispatcher (dense op indices documented\n"
    ";   per site), perform -> mx_perform (argument words in a scratch\n"
    ";   array), resume -> mx_resume; boundary values travel as opaque\n"
    ";   8-byte words; scope bodies run on ucontext coroutines with the\n"
    ";   interpreter's deep/single-shot/abort semantics;\n"
    ";   TRY/CATCH (increment 19): try_scope -> env fill + mx_try over a\n"
    ";   per-site body thunk + catch thunk; the runtime installs a setjmp\n"
    ";   landing pad whose chain is per-fiber (so it composes with the\n"
    ";   coroutine scheduler), binds the failure's PLAIN message text --\n"
    ";   byte-identical to the interpreter's InterpError.message -- and\n"
    ";   tears down every effect scope the failure escaped.  match_fail\n"
    ";   raises catchably too, naming the pre-monomorphization origin;\n"
    ";   RECLAMATION (increment 8): owned strings (produced, provably\n"
    ";   non-retained) are freed at redefinition + frame exit via shadow\n"
    ";   slots (literals never freed); unique payload boxes (entry-block\n"
    ";   make_variant values only ever tag/field-read) are freed on ret\n"
    ";   paths; everything unproven still leaks by design.  COPY ELISION:\n"
    ";   never-rebound aggregate params skip the byval copy (read the\n"
    ";   caller's storage); read-only variant_field results read through\n"
    ";   the write-once box pointer ('; elide-copy:' comments mark both);\n"
    ";   NATIVE FFI (increment 9): rawptr values are raw C `ptr` scalars\n"
    ";   (null -> the null constant; ==/!= -> ptr icmp); extern calls\n"
    ";   (malloc/free/memcpy/realloc/fopen/fclose) hit the REAL libc\n"
    ";   symbols with C signatures (interpreter's simulated-heap checks\n"
    ";   become native UB on rejected programs, like division by zero);\n"
    ";   as_ptr is identity on strings and an mx_vec_as_bytes snapshot on\n"
    ";   vecs (leaks by design); ptr_read/ptr_write are inline i8 ops;\n"
    ";   __vec_lit -> mx_vec_new + pushes (identity semantics stand in\n"
    ";   for the interpreter's immutable vector values -- no accepted\n"
    ";   program can tell); __static$Type$m calls resolve at compile time\n"
    ";   (impl fn -> dotted module fn -> dotted builtin, the\n"
    ";   interpreter's order); assert -> inline branch to @abort;\n"
    ";   FIXED VECTORS (increment 10): vector[T,N] values -> opaque\n"
    ";   mx_fvec* pointers to IMMUTABLE { i64 len, [len x i64] } word\n"
    ";   blocks (write-once fill at construction; shallow sharing is\n"
    ";   sound because nothing ever mutates a filled block; blocks leak\n"
    ";   by design).  Literals/zeros/filled -> mx_fvec_new/_init/_filled;\n"
    ";   element-wise + - * / % with scalar broadcast -> mx_fvec_binop\n"
    ";   (C-truncating int div, like scalar sdiv); slices -> mx_fvec_slice\n"
    ";   fresh copies (CPython slice.indices semantics); ranges ->\n"
    ";   mx_fvec_range int vectors restricted to iteration uses;\n"
    ";   comprehensions -> per-site word thunks driven by mx_fvec_map;\n"
    ";   __cast -> sitofp/fptosi or identity; promote_matrix -> static\n"
    ";   per-param mx_fvec_promote at entry; print/to_string render the\n"
    ";   interpreter's vector repr via mx_fvec_to_str; performs of ops no\n"
    ";   module scope handles lower to DIRECT CALLS of their declared\n"
    ";   __effect_default fns; an op with a default that a scope MAY also\n"
    ";   catch lowers to mx_perform_or_default (same scope lookup, the\n"
    ";   op's mxfx.dflt.* thunk only where mx_perform would abort -- the\n"
    ";   default runs on the PERFORMING stack, it is not a suspension);\n"
    ";   VECTOR SIMD (increment 11): flat float/int vector binops whose\n"
    ";   operand lengths are statically known emit INLINE <N x double> /\n"
    ";   <N x i64> IR (loads off the word block at byte offset 8, one\n"
    ";   vector fadd/fsub/fmul/fdiv or add/sub/mul, splat broadcast for\n"
    ";   scalars, store into a fresh mx_fvec_new block); int / and % and\n"
    ";   every unproven shape (dynamic/mismatched lengths, matrices,\n"
    ";   N > 64) keep the mx_fvec_binop C loop -- the always-correct path\n"
    ";   with its division-by-zero and length-mismatch aborts;\n"
    ";   SILENT-SEAM CONSTRUCTS (increment 12): index assignment --\n"
    ";   __index_store on a Vec -> mx_vec_set in place (result aliases\n"
    ";   the receiver), on a vector[T,N] -> mx_fvec_set_copy (the\n"
    ";   functional update: copy the write-once block, set, rebind);\n"
    ";   __index_set is Vec-only (immutable receivers demote at compile\n"
    ";   time with the interpreter's error); mutable captures\n"
    ";   (cell_wrap) -> one-word malloc'd cells (leak by design), reads/\n"
    ";   writes through %cellp.<n>, envs capture the CELL POINTER so\n"
    ";   frames share one binding; module constants -> @mx_g_<name>\n"
    ";   internal globals stored by @mx___module_init (called first by\n"
    ";   the entry wrapper) and loaded by readers; zip comprehensions ->\n"
    ";   two-word thunks over mx_fvec_zip_map (aborts on length\n"
    ";   mismatch, the interpreter's strict __zip); zip results are\n"
    ";   virtual and restricted to comprehension iterables;\n"
    ";   INDIRECT CLOSURE CALLS (increment 13): different same-arity\n"
    ";   lambdas meeting at one flow point join to a dynamic closure\n"
    ";   kind closure:*{L1,L2} instead of conflicting; participating\n"
    ";   lambdas (param-position flow, dynamic joins, env captures)\n"
    ";   emit with the word-uniform ABI i64 (ptr env, i64 args...) and\n"
    ";   indirect sites call the loaded fn pointer with word-encoded\n"
    ";   args/results; closure pairs may be captured into handle-site\n"
    ";   and closure envs (members forced heap-env so pairs never\n"
    ";   dangle);\n"
    ";   AGGREGATES THROUGH THE WORD ABI (increment 16): a struct/enum\n"
    ";   argument or result on a DYNAMIC indirect edge travels as a\n"
    ";   write-once boundary box (malloc + copy in, pointer as the word,\n"
    ";   copy out at the receiver; immortal, leaks by design).  Scalar\n"
    ";   indirect calls still allocate nothing, and an aggregate lambda\n"
    ";   pinned at every site keeps its typed ptr/sret signature.\n"
    ";   Closure pairs, konts, infinite layouts and @mut aggregate\n"
    ";   params (their write-back cannot travel back) stay demoted;\n"
    ";   BOUNDARY-BOX TRAFFIC (increment 17): three elisions, all of\n"
    ";   them semantics-preserving, all resting on the write-once box\n"
    ";   invariant (a boundary box is malloc'd, filled once and never\n"
    ";   freed).  (1) A value RECEIVING a boundary word (perform /\n"
    ";   resume / handle value / word-uniform indirect result) keeps\n"
    ";   the producer's pointer as a BOX VIEW instead of copying out,\n"
    ";   and re-exports that pointer at the next boundary instead of\n"
    ";   boxing a second copy; handler cases and handle bodies return\n"
    ";   the word directly (the boundary-word ABI), so the site shims\n"
    ";   allocate nothing.  (2) A read-only aggregate ARGUMENT of an\n"
    ";   indirect closure call passes the caller's storage pointer --\n"
    ";   no member writes through it (writeback_map) and the call is\n"
    ";   synchronous on this stack.  (3) A boundary box whose value is\n"
    ";   loop-invariant (single block-0 def, no write-back) is filled\n"
    ";   once at the end of the entry block and reused every\n"
    ";   iteration.  Views of enum PAYLOAD boxes are NOT re-exported\n"
    ";   (those boxes can be freed at frame exit), and anything\n"
    ";   unproven keeps its per-site box ('; elide-box:' marks the\n"
    ";   elided allocations);\n"
    "; functions outside the subset appear as comment-only placeholders."
)


def _llparam(kind: str) -> str:
    """The LLVM parameter/return-slot type for a value kind (aggregates -> ptr)."""
    if _is_agg(kind) or _is_vec(kind) or _is_fvec(kind) or _is_tile(kind) \
            or kind in (KONT, PTR):
        return "ptr"
    return _LLTY.get(kind, "i64")


def _llscalar(kind: str) -> str:
    """The LLVM type of a non-aggregate (register-sized) value kind.
    Vec values are opaque `mx_vec*` pointers; fixed vectors are opaque
    `mx_fvec*` pointers; tiles are opaque `mx_tile*` pointers; kont is an
    opaque `mx_k*`; rawptr is a raw C `ptr`."""
    if _is_vec(kind) or _is_fvec(kind) or _is_tile(kind) or kind in (KONT, PTR):
        return "ptr"
    return _LLTY.get(kind, "i64")


# _sanitize's registry: raw name -> sanitized symbol, and the reverse claim
# map (sanitized -> raw) that makes the mapping INJECTIVE.  Two distinct MIR
# names that only differ in special characters (`f.g` vs `f$g` vs `f_g`) used
# to collapse onto one LLVM symbol — a silent collision.  Now the first
# claimant keeps the plain sanitized form (so historical output is unchanged)
# and any DIFFERENT raw name mapping onto a claimed symbol gets a short
# deterministic hash suffix.  emit_llvm resets the registry per module, so
# the result is deterministic for a given module.
_SANITIZE_CACHE: Dict[str, str] = {}
_SANITIZE_CLAIMED: Dict[str, str] = {}


def _sanitize_reset() -> None:
    _SANITIZE_CACHE.clear()
    _SANITIZE_CLAIMED.clear()


def _sanitize(name: str) -> str:
    """Restrict a symbol to [A-Za-z0-9_], injectively per module."""
    got = _SANITIZE_CACHE.get(name)
    if got is not None:
        return got
    import hashlib
    cand = re.sub(r"[^A-Za-z0-9_]", "_", name)
    salt = name
    while _SANITIZE_CLAIMED.get(cand, name) != name:
        # Claimed by a DIFFERENT raw name: disambiguate deterministically.
        h = hashlib.sha1(salt.encode("utf-8")).hexdigest()[:6]
        cand = re.sub(r"[^A-Za-z0-9_]", "_", name) + "_x" + h
        salt = salt + h
    _SANITIZE_CLAIMED[cand] = name
    _SANITIZE_CACHE[name] = cand
    return cand


def mangle(name: str) -> str:
    """The module-local LLVM symbol for a metaxu function name."""
    return "mx_" + _sanitize(name)


def _is_runtime_builtin(name: str) -> bool:
    return name in _RUNTIME_NAMES or any(name.startswith(p) for p in _RUNTIME_PREFIXES)


def _builtin_name(callee: str, module_names: Set[str]) -> str:
    """The builtin name a NON-dispatch call op resolves to, or the callee.

    NAME PRECEDENCE (docs/name_precedence.md), mirroring
    mir_interp._eval_rhs exactly:
      * ``__builtin$m`` — a method-position call (`x.m()`) that the front
        end already resolved to the runtime builtin ``m``: always the
        builtin, never a same-named plain function;
      * a bare name that a MODULE FUNCTION defines — the user function
        wins, so return the callee unchanged and let the caller's
        module-function branch take it (no builtin set contains a name
        that is also a module function name, because the front end
        reserves the compiler's ``__`` namespace);
      * any other bare name — the builtin of that name (if any).
    """
    if callee.startswith(BUILTIN_CALL_PREFIX):
        return callee[len(BUILTIN_CALL_PREFIX):]
    if callee in module_names:
        return ""      # shadowed by a user function: matches no builtin set
    return callee


def _is_struct(kind: str) -> bool:
    return kind.startswith(_STRUCT_PREFIX)


def _struct_name(kind: str) -> str:
    return kind[len(_STRUCT_PREFIX):]


def _is_enum(kind: str) -> bool:
    return kind.startswith(_ENUM_PREFIX)


def _enum_name(kind: str) -> str:
    """The enum's name, with any payload refinement suffix stripped
    ('enum:Option{None:;Some:i64}' -> 'Option')."""
    base = kind[len(_ENUM_PREFIX):]
    brace = base.find("{")
    return base if brace < 0 else base[:brace]


def _strip_refinement(kind: str) -> str:
    """The name-only form of a kind, as stored in module-wide variant cells
    and inside refinement strings: refined enum kinds drop their refinement;
    every other kind is unchanged.  Keeping nested enum references name-only
    is what keeps refinement strings finite for recursive enums."""
    if _is_enum(kind):
        return _ENUM_PREFIX + _enum_name(kind)
    return kind


def _enum_refinement(kind: str) -> Optional[Dict[str, Tuple[str, ...]]]:
    """Parse an enum kind's per-variant payload refinement, or None when the
    kind is name-only.  Format (canonical, variants sorted):
    'enum:E{VarA:kind0,kind1;VarB:}' — slot kinds are themselves name-only
    (no nested braces), so plain ;/,-splitting is exact."""
    if not _is_enum(kind):
        return None
    base = kind[len(_ENUM_PREFIX):]
    brace = base.find("{")
    if brace < 0:
        return None
    body = base[brace + 1:-1]
    ref: Dict[str, Tuple[str, ...]] = {}
    if not body:
        return ref
    for part in body.split(";"):
        vname, _, slots = part.partition(":")
        ref[vname] = tuple(s for s in slots.split(",") if s)
    return ref


def _format_enum_kind(ename: str, ref: Dict[str, Tuple[str, ...]]) -> str:
    """The canonical refined enum kind string (variants sorted by name), so
    kind-string equality is representation equality."""
    body = ";".join(f"{v}:{','.join(ref[v])}" for v in sorted(ref))
    return f"{_ENUM_PREFIX}{ename}{{{body}}}"


def _is_closure(kind: str) -> bool:
    return kind.startswith(_CLOSURE_PREFIX)


def _closure_lambda(kind: str) -> str:
    return kind[len(_CLOSURE_PREFIX):]


# DYNAMIC CLOSURE KINDS (increment 13): two DIFFERENT lambdas of the same
# arity meeting at one flow point join to `closure:*{L1,L2}` — the
# canonical (sorted, deduplicated) set of every lambda whose closure can
# reach that value — instead of conflicting.  A call through such a value
# is an INDIRECT call: the {fn, env} pair is loaded and the fn pointer is
# called through the WORD-UNIFORM ABI (`i64 (ptr env, i64 args...)`, the
# effect-boundary word conventions: f64 bitcast, str/vec/rawptr
# ptrtoint).  Mismatched arities still conflict (the interpreter's
# zip-binding of a wrong-arity closure call errors at the first missing
# parameter use; native demotion is the strict static form of that).
_DYN_CLOSURE_PREFIX = _CLOSURE_PREFIX + "*{"

# lambda name -> arity, set per emit_llvm invocation (module-global so the
# pure kind lattice `_join` can validate arity agreement when merging
# closure kinds; emission is single-threaded per module).
_CLOSURE_ARITY: Dict[str, int] = {}


def _is_dyn_closure(kind: str) -> bool:
    return kind.startswith(_DYN_CLOSURE_PREFIX)


def _closure_members(kind: str) -> Tuple[str, ...]:
    """Every lambda a closure kind can name: the member set of a dynamic
    kind, the single lambda of a pinned kind, () for non-closure kinds."""
    if _is_dyn_closure(kind):
        return tuple(kind[len(_DYN_CLOSURE_PREFIX):-1].split(","))
    if _is_closure(kind):
        return (kind[len(_CLOSURE_PREFIX):],)
    return ()


def _dyn_closure_of(members: Sequence[str]) -> str:
    return _DYN_CLOSURE_PREFIX + ",".join(sorted(set(members))) + "}"


def _word_boxable(kind: str) -> bool:
    """Aggregate kinds that cross the word-uniform indirect-call ABI as a
    BOUNDARY BOX (increment 16): the caller mallocs a fresh write-once
    copy and passes its POINTER as the word; the receiver copies out into
    its own storage.  Structs and enums box; a closure PAIR does not (the
    boxed {fn, env} pair would carry an env pointer whose lifetime the box
    cannot vouch for), and konts/rawptr/conflicts never box."""
    return _is_struct(kind) or _is_enum(kind)


def _word_abi_ok(kind: str) -> bool:
    """Kinds that can cross the word-uniform indirect-call ABI: 8-byte
    scalars with an exact word encoding, plus struct/enum aggregates that
    travel as a boundary-box pointer word (increment 16).  Closure pairs,
    continuations and conflicts stay demoted."""
    return (kind in (I64, F64, STR, PTR) or _is_vec(kind) or _is_fvec(kind)
            or _word_boxable(kind))


def _is_vec(kind: str) -> bool:
    return kind.startswith(_VEC_PREFIX)


def _vec_elem(kind: str) -> str:
    """The element kind of a vec kind ('vec:f64' -> 'f64')."""
    return kind[len(_VEC_PREFIX):]


def _vec_of(elem: str) -> str:
    return _VEC_PREFIX + elem


def _is_fvec(kind: str) -> bool:
    return kind.startswith(_FVEC_PREFIX)


def _fvec_elem(kind: str) -> str:
    """The element kind of a fixed-vector kind ('vector:f64' -> 'f64')."""
    return kind[len(_FVEC_PREFIX):]


def _fvec_of(elem: str) -> str:
    return _FVEC_PREFIX + elem


def _fvec_leaf(kind: str) -> Tuple[str, int]:
    """(leaf scalar kind, nesting depth) of a fixed-vector kind: the depth
    counts how many levels the ELEMENTS are still vectors ('vector:f64' ->
    ('f64', 0); 'vector:vector:f64' -> ('f64', 1))."""
    depth = -1
    while _is_fvec(kind):
        kind = _fvec_elem(kind)
        depth += 1
    return kind, depth


def _fvec_with_leaf(kind: str, leaf: str) -> str:
    """The fixed-vector kind with the same nesting but a new leaf kind."""
    if _is_fvec(kind):
        return _fvec_of(_fvec_with_leaf(_fvec_elem(kind), leaf))
    return leaf


# Tiles (docs/gpu_tiles.md Stage 0): `tile:<elem>:<R>x<C>` — the STATIC
# shape rides the kind string, which is the whole point (shape agreement
# is checked at compile time and every emission site knows R and C).
_TILE_PREFIX = "tile:"


def _is_tile(kind: str) -> bool:
    return kind.startswith(_TILE_PREFIX)


def _tile_parts(kind: str) -> Tuple[str, int, int]:
    """('tile:f64:2x3') -> ('f64', 2, 3)."""
    elem, shape = kind[len(_TILE_PREFIX):].rsplit(":", 1)
    r, c = shape.split("x")
    return elem, int(r), int(c)


def _tile_of(elem: str, rows: int, cols: int) -> str:
    return f"{_TILE_PREFIX}{elem}:{rows}x{cols}"


def _is_word_kind(kind: str) -> bool:
    """Kinds storable as an opaque 8-byte word in a Vec element slot."""
    return (kind in (I64, F64, STR) or _is_vec(kind) or _is_fvec(kind)
            or _is_tile(kind))


def _vec_slot_boxable(kind: str) -> bool:
    """Aggregate kinds that occupy a native Vec element slot as an ELEMENT
    BOX pointer (increment 20): the writer mallocs a fresh write-once copy
    and stores its POINTER as the 8-byte word; every read copies the
    aggregate back OUT into the reader's own storage.  Exactly the enum
    payload / effect boundary contract, and the same exclusion: a closure
    PAIR does not box (its env pointer may aim at a frame the vec
    outlives), and kont/rawptr/conflict never box."""
    return _is_struct(kind) or _is_enum(kind)


def _is_agg(kind: str) -> bool:
    """Aggregate kinds: stored in own allocas, cross calls by pointer."""
    return _is_struct(kind) or _is_enum(kind) or _is_closure(kind)


def _enum_llname(ename: str) -> str:
    """The %enum type name for an enum ('' -> the shared anon type)."""
    return "%enum." + (_sanitize(ename) or "anon")


def _agg_ty(kind: str) -> str:
    """The LLVM named type of an aggregate kind."""
    if _is_struct(kind):
        return f"%struct.{_sanitize(_struct_name(kind))}"
    if _is_enum(kind):
        return _enum_llname(_enum_name(kind))
    return _CLOSURE_PAIR_TY


def _llcell(kind: str) -> str:
    """The LLVM type of an INLINE storage cell for a kind: scalars map via
    _llscalar (vec -> ptr), aggregates inline their named type (struct
    fields, env fields).  A ``cell:ELEM`` env-field marker (mutable
    capture, increment 12) stores the CELL POINTER — aliasing is the whole
    point."""
    if _is_cell_marker(kind):
        return "ptr"
    return _agg_ty(kind) if _is_agg(kind) else _llscalar(kind)


# MUTABLE-CAPTURE CELLS (increment 12): a variable some sub-function
# assigns is backed by a heap box holding one 8-byte word (`malloc(8)`,
# leaked by design — an immortal cell can never dangle), mirroring the
# interpreter's MxCell exactly: reads load through the cell pointer,
# writes store through it, and closure/handle-scope envs capture the CELL
# POINTER so every frame shares one binding.  The variable's KIND stays
# its element kind everywhere (the kind lattice never sees cells); the
# ``cell:ELEM`` marker below appears ONLY in env-field layout tables
# (mod.env_types / mod.scope_env_types) to say "this field holds the cell
# pointer, not the value".
_CELL_MARK_PREFIX = "cell:"


def _is_cell_marker(kind: str) -> bool:
    return kind.startswith(_CELL_MARK_PREFIX)


def _cell_marked(kind: str) -> str:
    return _CELL_MARK_PREFIX + kind


def _cell_elem(kind: str) -> str:
    return kind[len(_CELL_MARK_PREFIX):]


def _join(a: str, b: str) -> str:
    """Kind lattice: i64 is bottom; f64/str/struct:T are incomparable tops.
    Vec kinds join pointwise on their element kind (vec:i64 is the vec
    bottom: a fresh Vec.new before any push).  Refined enum kinds of the
    same enum join their refinements pointwise per (variant, slot): the
    refinement records the value's actual payload REPRESENTATION, so a
    per-slot conflict (or a variant-arity mismatch) conflicts the whole
    kind.  A name-only enum kind meeting a refined one is a CONFLICT, not a
    bottom: name-only enum kinds never arise as value kinds in well-formed
    flows (make_variant and variant_field always produce refined kinds), so
    treating one as 'no information' could let two representations alias."""
    if a == b:
        return a
    if a == I64:
        return b
    if b == I64:
        return a
    if _is_vec(a) and _is_vec(b):
        e = _join(_vec_elem(a), _vec_elem(b))
        return CONFLICT if e == CONFLICT else _vec_of(e)
    if _is_fvec(a) and _is_fvec(b):
        e = _join(_fvec_elem(a), _fvec_elem(b))
        return CONFLICT if e == CONFLICT else _fvec_of(e)
    if _is_tile(a) and _is_tile(b):
        # Shapes are part of the kind: different shapes never join (the
        # tile shape checker rejects the static cases up front; a joined
        # CONFLICT here demotes whatever slipped past it).
        ea, ra, ca = _tile_parts(a)
        eb, rb, cb = _tile_parts(b)
        if (ra, ca) != (rb, cb):
            return CONFLICT
        e = _join(ea, eb)
        return CONFLICT if e == CONFLICT else _tile_of(e, ra, ca)
    if _is_enum(a) and _is_enum(b):
        ename = _enum_name(a)
        if ename != _enum_name(b):
            return CONFLICT
        ra, rb = _enum_refinement(a), _enum_refinement(b)
        if ra is None or rb is None:
            return CONFLICT
        merged: Dict[str, Tuple[str, ...]] = dict(ra)
        for v, slots in rb.items():
            cur = merged.get(v)
            if cur is None:
                merged[v] = slots
                continue
            if len(cur) != len(slots):
                return CONFLICT
            js = tuple(_join(x, y) for x, y in zip(cur, slots))
            if CONFLICT in js:
                return CONFLICT
            merged[v] = js
        return _format_enum_kind(ename, merged)
    if _is_closure(a) and _is_closure(b):
        # Two different lambdas (or lambda sets) reaching one value: the
        # join is the DYNAMIC closure kind naming their union, provided
        # every member agrees on arity (call sites derive nargs from the
        # site, so mixed arities have no sound indirect call).
        members = sorted(set(_closure_members(a)) | set(_closure_members(b)))
        arities = {_CLOSURE_ARITY.get(m, -1) for m in members}
        if len(arities) == 1 and -1 not in arities:
            return _dyn_closure_of(members)
        return CONFLICT
    return CONFLICT


def _fmt_f64(x: float) -> str:
    """IEEE-754 bit-pattern hex literal, the unambiguous LLVM float syntax."""
    return "0x%016X" % _structmod.unpack("<Q", _structmod.pack("<d", float(x)))[0]


def _escape_bytes(data: bytes) -> str:
    out: List[str] = []
    for b in data:
        if 0x20 <= b < 0x7F and b not in (0x22, 0x5C):  # printable, not " or \
            out.append(chr(b))
        else:
            out.append("\\%02X" % b)
    return "".join(out)


class _Unsupported(Exception):
    """Raised during emission when a function turns out non-direct."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


# ---------------------------------------------------------------------------
# Per-function analysis (mirrors codegen_clif._analyze)
# ---------------------------------------------------------------------------

@dataclass
class _Info:
    f: MirFunc
    params: Tuple[str, ...] = ()
    reasons: List[str] = field(default_factory=list)
    def_count: Dict[str, int] = field(default_factory=dict)
    def_block: Dict[str, int] = field(default_factory=dict)
    use_blocks: Dict[str, Set[int]] = field(default_factory=dict)
    ret_vars: List[str] = field(default_factory=list)
    calls: List[Tuple[str, str, Tuple[str, ...]]] = field(default_factory=list)
    slots: List[str] = field(default_factory=list)
    suspending: bool = False
    # Variables defined by an @global alloc_struct: heap-backed storage.
    global_alloc_vars: Set[str] = field(default_factory=set)
    # Const-string variables that only feed tag comparisons: var -> variant name.
    tag_consts: Dict[str, str] = field(default_factory=dict)
    # make_closure sites: (dst, lambda name, capture names in order).
    closure_defs: List[Tuple[str, str, Tuple[str, ...]]] = field(default_factory=list)
    # Calls through a local variable (closure calls): (dst, callee var, args).
    closure_calls: List[Tuple[str, str, Tuple[str, ...]]] = field(default_factory=list)
    # Runtime-dispatched trait method calls: (dst, method name, args) —
    # statically resolved against the receiver's inferred kind.
    trait_calls: List[Tuple[str, str, Tuple[str, ...]]] = field(default_factory=list)
    # Capture names this function receives through its env (a lambda's
    # captures, or a handle-scope subfunction's free names).
    env_captures: Tuple[str, ...] = ()
    is_lambda: bool = False
    # Handle-scope subfunction (body or handler case) of a site.
    is_scope_member: bool = False
    scope_site: Optional[str] = None
    scope_role: Optional[str] = None  # "body" | "case"
    # Function contains perform ops routed through mx_perform (needs the
    # [8 x i64] scratch alloca).  Performs statically resolved to a
    # declared effect-op default (see default_performs) do not count.
    has_perform: bool = False
    # Results of copy/select ops that are provably never observed (see
    # _dead_results): excluded from kind unification, emitted as comments.
    dead_results: Set[str] = field(default_factory=set)
    # Single-def constant facts (used by the fixed-vector builtins whose
    # interpreter semantics depend on constant arguments).
    const_strs: Dict[str, str] = field(default_factory=dict)
    const_ints: Dict[str, int] = field(default_factory=dict)
    const_nones: Set[str] = field(default_factory=set)
    # Variables with ANY const-None def (a slice bound that is sometimes
    # None and sometimes an int cannot be encoded statically -> demote).
    none_def_vars: Set[str] = field(default_factory=set)
    # promote_matrix'd parameters (matmul's vector -> Mx1 embedding).
    promote_params: Tuple[str, ...] = ()
    # (effect, op) -> the op's FALLBACK fn for performs statically resolved
    # to it (no handle site in the module lists the op, so no scope can
    # ever intercept it -> a direct call, aggregates and all, exactly the
    # interpreter's fallback).  The fallback is the op's `with SYMBOL`
    # runtime-mapping thunk __effect_runtime$E$op when one is declared
    # (it outranks the default, docs/threads_runtime.md), else the
    # declared `= expr` default __effect_default$E$op.
    default_performs: Dict[Tuple[str, str], str] = field(default_factory=dict)
    # (effect, op) -> the op's fallback fn (same precedence as above) for
    # performs whose op ALSO appears in some handle scope: routing is a
    # RUNTIME choice, lowered to mx_perform_or_default (increment 15).  The
    # boundary conventions are the ordinary perform ones (op-name kind
    # cells, word/boundary-box encoding); the fallback fn's signature joins
    # those same cells so the per-op thunk's decode is exact.
    dynamic_default_performs: Dict[Tuple[str, str], str] = field(
        default_factory=dict)
    # Calls to __mx_effect_runtime$SYMBOL (the bodies of the
    # __effect_runtime$E$op thunks): (dst, SYMBOL, args), lowered to the
    # metaxu_threads.c primitives (docs/threads_runtime.md).  Validated in
    # _check_consistency (SPAWN's closure argument especially) and emitted
    # as direct mx_* calls.
    effect_primitive_calls: List[Tuple[str, str, Tuple[str, ...]]] = field(
        default_factory=list)
    # Module-constant names this function READS (used with no local def and
    # declared by __module_init): they load from @mx_g_<name> globals.
    global_reads: Set[str] = field(default_factory=set)
    # __module_init only: the declared module-constant names — their defs
    # store straight into the @mx_g_<name> globals (their storage class).
    init_globals: Tuple[str, ...] = ()
    # __zip call results: dst -> the zipped sequence variables.  A zip
    # result is a virtual value (the interpreter's list of tuples has no
    # native representation): its ONLY legal use is as the iterable of a
    # __vec_comprehension, which reads the SOURCES directly and drives
    # mx_fvec_zip_map.  Every other use demotes (consistency check).
    zip_defs: Dict[str, Tuple[str, ...]] = field(default_factory=dict)

    def add_reason(self, r: str) -> None:
        if r not in self.reasons:
            self.reasons.append(r)


def _find_tag_consts(f: MirFunc) -> Dict[str, str]:
    """Const-string vars whose EVERY use is an ==/!= against a variant_tag
    result (the compiled-pattern shape): they lower to integer tags."""
    tag_vars: Set[str] = set()
    str_consts: Dict[str, str] = {}
    use_count: Dict[str, int] = {}
    tag_cmp_count: Dict[str, int] = {}

    def count_use(n: str) -> None:
        use_count[n] = use_count.get(n, 0) + 1

    for b in f.blocks:
        for op in b.ops:
            if op[0] == "let" and len(op) == 4:
                _, dst, rhs, args = op
                if rhs[0] == "variant_tag":
                    tag_vars.add(dst)
                elif rhs[0] == "const" and isinstance(rhs[1], str) \
                        and not isinstance(rhs[1], bool):
                    str_consts[dst] = rhs[1]
                if rhs[0] in ("alloc_struct", "make_closure", "handle_scope",
                              "try_scope"):
                    for (_n, v) in args:
                        if isinstance(v, str):
                            count_use(v)
                else:
                    for a in args:
                        count_use(a)
            elif op[0] == "perform":
                for a in op[4]:
                    count_use(a)
        t = b.term
        if t[0] == "br_if":
            count_use(t[1])
        elif t[0] == "ret":
            count_use(t[1])
    for b in f.blocks:
        for op in b.ops:
            if op[0] != "let" or len(op) != 4 or op[2][0] != "binop":
                continue
            if op[2][1] not in ("==", "!="):
                continue
            a0, a1 = op[3]
            if a0 in tag_vars and a1 in str_consts:
                tag_cmp_count[a1] = tag_cmp_count.get(a1, 0) + 1
            elif a1 in tag_vars and a0 in str_consts:
                tag_cmp_count[a0] = tag_cmp_count.get(a0, 0) + 1
    return {v: s for v, s in str_consts.items()
            if tag_cmp_count.get(v, 0) > 0
            and tag_cmp_count[v] == use_count.get(v, 0)}


def _dead_results(f: MirFunc) -> Set[str]:
    """Variables whose value is provably never observed: every use is as an
    operand of a copy/select whose own destination is dead (transitively),
    and they never reach a terminator or any other op.

    Statement-position if/match expressions lower to exactly this shape: a
    result variable copy-merged from arms of DIFFERENT kinds (e.g. one arm
    yields unit/i64, the other an enum).  Unifying kinds through those dead
    copies would poison live variables into aggregate kinds, so dead
    copy/select ops are excluded from inference and emitted as comments —
    which is sound precisely because nothing ever reads their result."""
    soft_uses: Dict[str, List[str]] = {}   # var -> dsts of copy/select users
    hard_used: Set[str] = set()
    defs: Set[str] = set()
    for b in f.blocks:
        for op in b.ops:
            if op[0] == "let" and len(op) == 4:
                _, dst, rhs, args = op
                defs.add(dst)
                rk = rhs[0]
                if rk in ("copy", "select"):
                    for a in args:
                        soft_uses.setdefault(a, []).append(dst)
                elif rk in ("alloc_struct", "make_closure", "handle_scope",
                            "try_scope"):
                    for pair in args:
                        if isinstance(pair[1], str):
                            hard_used.add(pair[1])
                elif rk == "call":
                    hard_used.add(rhs[1])  # closure callee var (if any)
                    hard_used.update(args)
                else:
                    hard_used.update(args)
            elif op[0] == "perform":
                hard_used.update(op[4])
        t = b.term
        if t[0] in ("br_if", "ret"):
            hard_used.add(t[1])
    dead: Set[str] = set()
    changed = True
    while changed:
        changed = False
        for v in defs:
            if v in dead or v in hard_used:
                continue
            if all(d in dead for d in soft_uses.get(v, [])):
                dead.add(v)
                changed = True
    return dead


def _blocks_in_cycles(f: MirFunc) -> Set[int]:
    """Block indices that lie on a CFG cycle (loop bodies/headers)."""
    n = len(f.blocks)
    succs: List[List[int]] = []
    for b in f.blocks:
        t = b.term
        if t[0] == "br":
            succs.append([t[1]] if 0 <= t[1] < n else [])
        elif t[0] == "br_if":
            succs.append([x for x in (t[2], t[3]) if 0 <= x < n])
        else:
            succs.append([])
    reach: List[Set[int]] = []
    for i in range(n):
        seen: Set[int] = set()
        stack = list(succs[i])
        while stack:
            x = stack.pop()
            if x in seen:
                continue
            seen.add(x)
            stack.extend(succs[x])
        reach.append(seen)
    return {i for i in range(n) if i in reach[i]}


def _analyze(f: MirFunc, module_names: Set[str], closures: "_ClosureTable",
             scopes: "_ScopeTable", traits: "_TraitTable",
             cells: "_CellTable", gtable: "_GlobalTable") -> _Info:
    info = _Info(f=f)
    try:
        _analyze_inner(info, module_names, closures, scopes, traits,
                       cells, gtable)
    except Exception as exc:  # defensive: malformed MIR must never crash codegen
        info.add_reason(f"analysis error: {type(exc).__name__}: {exc}")
    return info


def _analyze_inner(info: _Info, module_names: Set[str],
                   closures: "_ClosureTable", scopes: "_ScopeTable",
                   traits: "_TraitTable", cells: "_CellTable",
                   gtable: "_GlobalTable") -> None:
    f = info.f
    if not f.blocks:
        info.add_reason("function has no blocks")
        return
    if f.blocks[0].ops and f.blocks[0].ops[0][0] == "params":
        info.params = tuple(f.blocks[0].ops[0][1])
    # Suspending functions EMIT since increment 7: perform/resume/
    # handle_scope lower to the native effects runtime (the coroutine stack
    # is the continuation, no CPS transform needed).
    info.suspending = is_suspending(f)
    info.tag_consts = _find_tag_consts(f)
    info.dead_results = _dead_results(f)
    cycle_blocks = _blocks_in_cycles(f)

    # Mutable-capture cells (increment 12): prescan problems demote here;
    # writes to a cell-backed variable are observable through the shared
    # cell even when the owner never reads them again, so they are never
    # dead results.
    cell_backed = cells.backed.get(f.name, set())
    for r in cells.bad.get(f.name, ()):
        info.add_reason(r)
    info.dead_results -= cell_backed

    if f.name == _MODULE_INIT:
        info.init_globals = tuple(f.globals_decl)
        # A declared name's def PUBLISHES the global — observable by every
        # reader even when the initializer itself never reads it again.
        info.dead_results -= set(f.globals_decl)

    # A make_closure target receives its captures through the env struct:
    # they are entry-defined names, exactly like parameters.
    if f.name in closures.targets:
        info.is_lambda = True
        info.env_captures = closures.targets[f.name]
        if f.name in closures.bad:
            info.add_reason(closures.bad[f.name])

    # A handle-scope subfunction (body or handler case) reloads its free
    # names from the site's shared env struct, lambda-style.
    if f.name in scopes.member_site:
        if info.is_lambda:
            info.add_reason(
                "function is both a lambda and a handle-scope subfunction")
        info.is_scope_member = True
        info.scope_site = scopes.member_site[f.name]
        info.scope_role = scopes.member_role[f.name]
        info.env_captures = scopes.free_names.get(f.name, ())
        if f.name in scopes.bad:
            info.add_reason(scopes.bad[f.name])

    for p in info.params:
        info.def_count[p] = 1
        info.def_block[p] = 0
    for c in info.env_captures:
        if c not in info.def_count:
            info.def_count[c] = 1
            info.def_block[c] = 0

    def add_use(name: str, bi: int) -> None:
        info.use_blocks.setdefault(name, set()).add(bi)

    def add_def(name: str, bi: int) -> None:
        info.def_count[name] = info.def_count.get(name, 0) + 1
        info.def_block.setdefault(name, bi)

    n_blocks = len(f.blocks)
    for bi, b in enumerate(f.blocks):
        for op in b.ops:
            kind = op[0]
            if kind == "params":
                if bi != 0:
                    info.add_reason("params op outside entry block")
                continue
            if kind == "perform":
                # ("perform", dst, effect, op, args, resume_bb, resume_slot):
                # a real suspension point, lowered to mx_perform (the native
                # runtime parks this call stack and returns the resumed
                # value).  The op defines dst; the block then branches to
                # the resume block.
                #
                # EXCEPTION (increment 10): an op that NO handle_scope in
                # the module lists can never be intercepted by a scope, so
                # the interpreter's fallback resolution is static.  When the
                # op has a declared `= expr` default (and no `with SYMBOL`
                # runtime mapping, which would win), the perform IS a direct
                # call to __effect_default$E$op — normal call conventions,
                # aggregates and all, no effect boundary.
                #
                # When the op has a default AND some scope lists it, the
                # choice is DYNAMIC (increment 15): still a real perform
                # (the scratch alloca, the boundary words, the possible
                # park), but through mx_perform_or_default, which falls back
                # to the op's default thunk instead of aborting.
                if len(op) < 7:
                    info.add_reason("malformed perform op")
                    continue
                peffect, pop_name, pargs = op[2], op[3], op[4]
                for a in pargs:
                    add_use(a, bi)
                add_def(op[1], bi)
                scoped = any(
                    pop_name == opn
                    for rec in scopes.sites.values()
                    for (opn, _p, _h) in rec.cases)
                default_fn = f"{_EFFECT_DEFAULT_PREFIX}{peffect}${pop_name}"
                runtime_fn = f"{_EFFECT_RUNTIME_PREFIX}{peffect}${pop_name}"
                # The op's FALLBACK when no scope intercepts, in the
                # interpreter's precedence order: the `with SYMBOL` runtime
                # mapping thunk (docs/threads_runtime.md), else the declared
                # `= expr` default.  Both share the same routing machinery.
                if runtime_fn in module_names:
                    fallback_fn = runtime_fn
                elif default_fn in module_names:
                    fallback_fn = default_fn
                else:
                    fallback_fn = None
                if not scoped and fallback_fn is not None:
                    info.default_performs[(peffect, pop_name)] = fallback_fn
                    continue
                if scoped and fallback_fn is not None:
                    # DYNAMIC FALLBACK ROUTING (increment 15): a scope MAY
                    # intercept this op, and when none does the interpreter
                    # falls back to the runtime mapping / declared default.
                    # The choice is made at the perform, by the runtime's
                    # scope stack — so the perform lowers to
                    # mx_perform_or_default, which runs the SAME innermost-
                    # non-busy lookup and calls the op's fallback thunk (on
                    # this stack) only where plain mx_perform would have
                    # aborted.
                    info.dynamic_default_performs[(peffect, pop_name)] = \
                        fallback_fn
                if len(pargs) > _MAX_EFFECT_ARGS:
                    info.add_reason(
                        f"perform with {len(pargs)} arguments (native limit "
                        f"is {_MAX_EFFECT_ARGS})")
                info.has_perform = True
                continue
            if kind == "promote_matrix":
                # ("promote_matrix", (param names,)): the named parameters
                # are declared as matrices; a flat numeric vector passed
                # there is promoted to an Mx1 column of one-element rows
                # (mir_interp).  Supported shape: entry block, parameters
                # only, before any other use of the names — each variable
                # has ONE kind (the post-promotion one), so a pre-promotion
                # use would be typed wrongly.
                names = tuple(op[1]) if len(op) > 1 else ()
                if bi != 0:
                    info.add_reason("promote_matrix outside the entry block")
                for n in names:
                    if n not in info.params:
                        info.add_reason(
                            f"promote_matrix of non-parameter {n!r}")
                    elif n in info.use_blocks:
                        info.add_reason(
                            f"promote_matrix after a use of {n!r}")
                for n in names:
                    add_use(n, bi)
                    add_def(n, bi)
                info.promote_params = tuple(
                    dict.fromkeys(info.promote_params + names))
                continue
            if kind == "drop":
                continue  # emitted as a comment
            if kind == "cell_wrap":
                # ("cell_wrap", slot): the slot is cell-backed for the
                # whole frame (see _CellTable), so the op itself is a
                # comment; the wrap position already fed the prescan's
                # post-wrap capture check.
                add_use(op[1], bi)
                continue
            if kind == "match_fail":
                continue  # emitted as @abort + unreachable
            if kind != "let" or len(op) != 4:
                info.add_reason(f"unsupported op {kind!r}")
                continue
            _, dst, rhs, args = op
            rk = rhs[0]
            if rk == "const":
                v = rhs[1]
                if isinstance(v, bool) or v is None:
                    pass
                elif isinstance(v, int):
                    if not (_I64_MIN <= v <= _I64_MAX):
                        info.add_reason(f"integer constant {v} outside i64 range")
                elif isinstance(v, (float, str)):
                    pass
                else:
                    info.add_reason(f"unsupported constant type {type(v).__name__}")
                add_def(dst, bi)
            elif rk == "const_ty":
                if rhs[1] != "Unit":
                    info.add_reason("opaque typed constant (lowering fallback)")
                add_def(dst, bi)
            elif rk == "copy":
                add_use(args[0], bi)
                add_def(dst, bi)
            elif rk == "binop":
                if rhs[1] not in _SUPPORTED_BINOPS:
                    info.add_reason(f"unsupported binop {rhs[1]!r}")
                for a in args:
                    add_use(a, bi)
                add_def(dst, bi)
            elif rk == "select":
                for a in args:
                    add_use(a, bi)
                add_def(dst, bi)
            elif rk == "call":
                callee = rhs[1]
                info.calls.append((dst, callee, tuple(args)))
                if callee == "__zip" and callee not in info.def_count:
                    # Lockstep zip iterable: the result is virtual (its only
                    # legal use is a comprehension iterable — consistency
                    # enforces); native support is pair iteration.
                    info.zip_defs[dst] = tuple(args)
                    if len(args) != 2:
                        info.add_reason(
                            f"__zip of {len(args)} sequences (only pair "
                            "iteration lowers natively)")
                for a in args:
                    add_use(a, bi)
                add_def(dst, bi)
            elif rk == "alloc_struct":
                locality = rhs[2] if len(rhs) > 2 else "local"
                if locality == "global":
                    info.global_alloc_vars.add(dst)
                elif locality != "local":
                    info.add_reason(
                        f"unknown locality {locality!r} for struct {rhs[1]!r}")
                for (_fname, fval) in args:
                    add_use(fval, bi)
                add_def(dst, bi)
            elif rk == "field_get":
                add_use(args[0], bi)
                add_def(dst, bi)
            elif rk == "field_set":
                add_use(args[0], bi)
                add_use(args[1], bi)
                add_def(dst, bi)
            elif rk == "make_variant":
                for a in args:
                    add_use(a, bi)
                add_def(dst, bi)
            elif rk in ("variant_tag", "variant_field"):
                add_use(args[0], bi)
                add_def(dst, bi)
            elif rk == "make_closure":
                lname = rhs[1]
                if lname not in module_names:
                    info.add_reason(
                        f"make_closure of unknown function {lname!r}")
                if bi in cycle_blocks:
                    # Re-executing a stack-env site would overwrite storage
                    # earlier pair copies may still alias: give this lambda a
                    # fresh heap env per execution instead (leaked by design).
                    closures.heap_env.add(lname)
                for (_cn, vn) in args:
                    add_use(vn, bi)
                add_def(dst, bi)
                info.closure_defs.append(
                    (dst, lname, tuple(cn for (cn, _vn) in args)))
            elif rk == "resume":
                # ("resume",), (k, value): consume the single-shot
                # continuation via mx_resume.  Only sound when running on
                # the scope's owner stack, which the compiler guarantees by
                # only accepting a resume of the containing handler case's
                # OWN trailing __k parameter (a continuation smuggled into
                # a nested scope body / lambda env would pump the scope
                # from a foreign coroutine — demote instead).
                if not (info.scope_role == "case" and len(args) == 2
                        and args[0] in info.params):
                    info.add_reason(
                        "resume outside its own handler case (the "
                        "continuation would escape its scope's owner stack)")
                for a in args:
                    add_use(a, bi)
                add_def(dst, bi)
            elif rk == "handle_scope":
                # ("handle_scope", body_fn, effect, cases), captures:
                # lowered to env fill + mx_handle (body thunk + dispatcher
                # shims emitted per site).
                site_rec = scopes.sites.get(rhs[1])
                if site_rec is None or site_rec.owner != f.name:
                    info.add_reason(
                        f"handle site {rhs[1]!r} unresolved (not this "
                        "function's handle_scope)")
                elif f.name in scopes.bad:
                    info.add_reason(scopes.bad[f.name])
                else:
                    if len(site_rec.cases) == 0:
                        info.add_reason("handle_scope with no handler cases")
                    for (_opn, cparams, _hfn) in site_rec.cases:
                        if len(cparams) > _MAX_EFFECT_ARGS:
                            info.add_reason(
                                f"handler case with {len(cparams)} parameters "
                                f"(native limit is {_MAX_EFFECT_ARGS})")
                    # Only the captures the site's members actually need are
                    # live values here (the lowering captures every env name
                    # conservatively; the interpreter is non-strict).
                    for n in scopes.env_fields.get(rhs[1], ()):
                        add_use(site_rec.cap_vals.get(n, n), bi)
                add_def(dst, bi)
            elif rk == "try_scope":
                # ("try_scope", body_fn, catch_fn), captures: env fill +
                # mx_try over a per-site body thunk and catch thunk.  The
                # catch fn's single parameter receives the failure message
                # (a `str`), exactly the interpreter's `exc.message`.
                site_rec = scopes.sites.get(rhs[1])
                if site_rec is None or site_rec.kind != "try" \
                        or site_rec.owner != f.name:
                    info.add_reason(
                        f"try site {rhs[1]!r} unresolved (not this "
                        "function's try_scope)")
                elif f.name in scopes.bad:
                    info.add_reason(scopes.bad[f.name])
                else:
                    for n in scopes.env_fields.get(rhs[1], ()):
                        add_use(site_rec.cap_vals.get(n, n), bi)
                add_def(dst, bi)
            else:
                info.add_reason(f"unsupported op {rk!r}")

        t = b.term
        if t[0] == "br":
            if not (0 <= t[1] < n_blocks):
                info.add_reason(f"branch target bb{t[1]} out of range")
        elif t[0] == "br_if":
            add_use(t[1], bi)
            for tgt in (t[2], t[3]):
                if not (0 <= tgt < n_blocks):
                    info.add_reason(f"branch target bb{tgt} out of range")
        elif t[0] == "ret":
            add_use(t[1], bi)
            info.ret_vars.append(t[1])
        elif t[0] == "unreachable":
            pass
        else:
            info.add_reason(f"unsupported terminator {t[0]!r}")

    # Calls: partition into closure calls (callee is a local variable — the
    # interpreter's shadowing order: locals first), trait-dispatched calls
    # (resolved statically against the receiver kind later), native runtime
    # builtins (reached only when _builtin_name says the call resolves to a
    # builtin: a same-named module function wins for plain calls, and a
    # method-position __builtin$m always resolves here — mirroring
    # mir_interp), and direct calls, which must hit module functions or the
    # supported builtins.
    direct_calls: List[Tuple[str, str, Tuple[str, ...]]] = []
    for (dst, callee, cargs) in info.calls:
        bname = _builtin_name(callee, module_names)
        if callee in info.def_count:
            info.closure_calls.append((dst, callee, cargs))
        elif callee.startswith(TRAIT_CALL_PREFIX):
            method = callee[len(TRAIT_CALL_PREFIX):]
            if not cargs:
                info.add_reason(
                    f"trait method call {method!r} with no receiver")
            else:
                info.trait_calls.append((dst, method, cargs))
        elif callee.startswith(STATIC_CALL_PREFIX):
            # `Type.method(args)` resolves at compile time exactly like the
            # interpreter's _dispatch_static_call (see _resolve_static_call);
            # the ORIGINAL callee name is kept — every later phase re-runs
            # the (purely static) resolution.
            res, target = _resolve_static_call(callee, traits, module_names)
            if res == "demote":
                info.add_reason(target)
            else:
                direct_calls.append((dst, callee, cargs))
        elif bname in _NATIVE_RT_CALLS or bname in _FFI_CALLS \
                or bname == "assert":
            # Native runtime builtins, the extern-C/FFI names, and assert.
            # _builtin_name has already applied the precedence rule: a name
            # a module function defines never reaches here (the user
            # function wins), and __builtin$m always does.
            direct_calls.append((dst, callee, cargs))
        elif callee in closures.targets:
            # Lambdas are only callable through their closure value: a direct
            # call would skip the env parameter.
            info.add_reason(
                f"direct call to lambda {callee!r} (callable only through "
                "its closure value)")
        elif callee in scopes.member_site:
            # Body/case subfunctions are only callable through the effects
            # runtime (a direct call would skip the env parameter).
            info.add_reason(
                f"direct call to handle-scope subfunction {callee!r}")
        elif callee in module_names:
            direct_calls.append((dst, callee, cargs))
        elif callee.startswith(_EFFECT_PRIMITIVE_CALL_PREFIX):
            # The body of an __effect_runtime$E$op thunk: a `with SYMBOL`
            # effect-op mapping, lowered to the pthreads-backed primitives
            # in metaxu_threads.c (docs/threads_runtime.md).
            symbol = callee[len(_EFFECT_PRIMITIVE_CALL_PREFIX):]
            prim = _EFFECT_PRIMITIVES.get(symbol)
            if prim is None:
                # The interpreter's "no shim for SYMBOL" error, statically.
                info.add_reason(
                    f"effect op mapped to runtime primitive {symbol!r}, "
                    "which has no native implementation (available: "
                    f"{', '.join(sorted(_EFFECT_PRIMITIVES))})")
            elif len(cargs) != prim[1]:
                info.add_reason(
                    f"runtime primitive {symbol!r} called with "
                    f"{len(cargs)} argument(s); it takes {prim[1]}")
            else:
                info.effect_primitive_calls.append((dst, symbol, cargs))
        elif bname in _PRINT_BUILTINS or bname in _MATH_EXTERNS or bname in _INLINE_BUILTINS:
            direct_calls.append((dst, callee, cargs))
        elif _is_runtime_builtin(bname or callee):
            info.add_reason(
                f"calls runtime builtin {bname or callee!r} (vec/string/trait)")
        else:
            info.add_reason(f"unknown external callee {callee!r} (cannot link natively)")
    info.calls = direct_calls

    # Single-def constant facts (the fixed-vector builtins depend on
    # statically-known sizes / dims / casts / slice bounds / base names).
    for b in f.blocks:
        for op in b.ops:
            if op[0] != "let" or len(op) != 4 or op[2][0] != "const":
                continue
            cdst, cval = op[1], op[2][1]
            if cval is None:
                info.none_def_vars.add(cdst)
            if info.def_count.get(cdst, 0) != 1:
                continue
            if cval is None:
                info.const_nones.add(cdst)
            elif isinstance(cval, bool):
                pass
            elif isinstance(cval, int):
                info.const_ints[cdst] = cval
            elif isinstance(cval, str):
                info.const_strs[cdst] = cval

    # A cell-backed variable can be reassigned from ANOTHER function
    # through the shared cell, so per-function single-def constant facts
    # about it are unsound: drop them.
    for n in cell_backed:
        info.const_strs.pop(n, None)
        info.const_ints.pop(n, None)
        info.const_nones.discard(n)

    # Every used name must be defined somewhere in the function — except
    # module constants (declared by __module_init), which resolve as
    # global reads, exactly the interpreter's _lookup fallback order
    # (frame bindings first, then globals).
    for name in info.use_blocks:
        if name not in info.def_count:
            if name in gtable.names:
                info.global_reads.add(name)
            else:
                info.add_reason(
                    f"references {name!r} with no local definition (captured environment)")

    # A local (non-parameter) DEF of a module-constant name shadows the
    # global flow-sensitively in the interpreter (reads before the first
    # assignment see the global, later reads the local): no static storage
    # class reproduces that, so demote.  Parameters shadow from entry on
    # (a plain local) and __module_init's declared names ARE the globals.
    if f.name != _MODULE_INIT:
        for n in sorted(gtable.names):
            if n in info.def_count and n not in info.params \
                    and n not in info.env_captures:
                info.add_reason(
                    f"local assignment to module-constant name {n!r} "
                    "shadows the global flow-sensitively")
    else:
        for n in info.init_globals:
            if n in cell_backed:
                info.add_reason(
                    f"module constant {n!r} is cell-wrapped (mutable "
                    "module state stays interpreted)")
            if n not in info.def_count:
                info.add_reason(
                    f"module constant {n!r} declared but never bound "
                    "(bad lowering)")


# ---------------------------------------------------------------------------
# Module-wide struct table
# ---------------------------------------------------------------------------

@dataclass
class _StructTable:
    fields: Dict[str, Tuple[str, ...]] = field(default_factory=dict)  # name -> field order
    bad: Dict[str, str] = field(default_factory=dict)                 # name -> reason
    kinds: Dict[Tuple[str, str], str] = field(default_factory=dict)   # (name, field) -> kind

    def field_kind(self, sname: str, fname: str) -> str:
        return self.kinds.get((sname, fname), I64)

    def mark_field(self, sname: str, fname: str, kind: str) -> bool:
        key = (sname, fname)
        cur = self.kinds.get(key, I64)
        nk = _join(cur, kind)
        if nk != cur:
            self.kinds[key] = nk
            return True
        return False


def _build_struct_table(funcs: Sequence[MirFunc]) -> _StructTable:
    table = _StructTable()
    for f in funcs:
        for b in f.blocks:
            for op in b.ops:
                if op[0] != "let" or len(op) != 4 or op[2][0] != "alloc_struct":
                    continue
                sname = op[2][1]
                names = tuple(fn for (fn, _fv) in op[3])
                if sname not in table.fields:
                    table.fields[sname] = names
                elif set(table.fields[sname]) != set(names):
                    table.bad.setdefault(
                        sname,
                        f"inconsistent field sets for struct {sname!r}: "
                        f"{sorted(table.fields[sname])} vs {sorted(names)}")
    return table


# ---------------------------------------------------------------------------
# Module-wide variant (tagged-union) table
# ---------------------------------------------------------------------------

@dataclass
class _VariantTable:
    # Module-wide dense integer tag per variant name (make_variant names and
    # pattern tag literals, sorted then numbered): string equality on tags in
    # the interpreter maps exactly to integer equality natively.
    tags: Dict[str, int] = field(default_factory=dict)
    # enum name ('' = anon) -> payload slot count (max over its variants).
    payload_max: Dict[str, int] = field(default_factory=dict)
    # (enum name, variant name, slot index) -> NAME-ONLY kind: the join of
    # every make_variant store into that variant's slot module-wide.  These
    # cells are the CANONICAL representation used when a value crosses a
    # nesting boundary (an enum boxed inside another enum's payload slot
    # loses its per-value refinement); they are only sound to read through
    # when no store disagrees with the join (see `mixed`).
    cells: Dict[Tuple[str, str, int], str] = field(default_factory=dict)
    # enum name -> variant names seen in make_variant ops.
    variants_of: Dict[str, Set[str]] = field(default_factory=dict)
    # (enum name, variant name) -> payload arity from make_variant sites.
    arity: Dict[Tuple[str, str], int] = field(default_factory=dict)
    # enum name -> reason (inconsistent construction arity etc.).
    bad: Dict[str, str] = field(default_factory=dict)
    # (enum, variant, slot) cells where some post-fixpoint make_variant
    # store kind differs from the joined cell kind: the slot's native
    # representation is per-value (instantiation-dependent), so reading it
    # through the canonical cells would guess.  Filled by the module driver
    # after the kind fixpoint.
    mixed: Set[Tuple[str, str, int]] = field(default_factory=set)

    def cell_kind(self, ename: str, vname: str, idx: int) -> str:
        return self.cells.get((ename, vname, idx), I64)

    def mark_cell(self, ename: str, vname: str, idx: int, kind: str) -> bool:
        key = (ename, vname, idx)
        cur = self.cells.get(key, I64)
        nk = _join(cur, kind)
        if nk != cur:
            self.cells[key] = nk
            return True
        return False

    def canon(self, ename: str) -> Dict[str, Tuple[str, ...]]:
        """The canonical refinement of an enum: every constructed variant,
        each slot at its module-wide cell kind.  Monotone over the fixpoint
        (variants_of/arity are fixed at table build; cells only promote)."""
        ref: Dict[str, Tuple[str, ...]] = {}
        for v in self.variants_of.get(ename, ()):
            n = self.arity.get((ename, v), 0)
            ref[v] = tuple(self.cell_kind(ename, v, i) for i in range(n))
        return ref

    def canon_kind(self, ename: str) -> str:
        return _format_enum_kind(ename, self.canon(ename))

    def mixed_slots_of(self, ename: str) -> List[Tuple[str, str, int]]:
        return sorted(k for k in self.mixed if k[0] == ename)


def _build_variant_table(funcs: Sequence[MirFunc],
                         infos: Sequence[_Info]) -> _VariantTable:
    table = _VariantTable()
    names: Set[str] = set()
    for f in funcs:
        for b in f.blocks:
            for op in b.ops:
                if op[0] != "let" or len(op) != 4 or op[2][0] != "make_variant":
                    continue
                ename, vname = op[2][1], op[2][2]
                names.add(vname)
                table.variants_of.setdefault(ename, set()).add(vname)
                table.payload_max[ename] = max(
                    table.payload_max.get(ename, 0), len(op[3]))
                key = (ename, vname)
                if key not in table.arity:
                    table.arity[key] = len(op[3])
                elif table.arity[key] != len(op[3]):
                    table.bad.setdefault(
                        ename,
                        f"variant {vname!r} of enum {ename or 'anon'!r} "
                        f"constructed with inconsistent payload arities "
                        f"({table.arity[key]} vs {len(op[3])})")
    for info in infos:
        names.update(info.tag_consts.values())
    table.tags = {n: i for i, n in enumerate(sorted(names))}
    return table


# ---------------------------------------------------------------------------
# Recursive size computation (nested layouts + boxed payload slots)
# ---------------------------------------------------------------------------

def _kind_size(kind: str, structs: "_StructTable", variants: "_VariantTable",
               _seen: Tuple[str, ...] = ()) -> Optional[int]:
    """Byte size of a value kind's storage, or None for an infinite layout.

    Scalars are 8 bytes; a closure pair is 16; an enum is 8 (tag) + 8 per
    payload slot (aggregate slots hold an 8-byte box POINTER, which is what
    breaks recursion); a struct is the sum of its inline field sizes.  Only
    struct-in-struct chains recurse, so a cycle there (no intervening enum
    box) has no finite layout and returns None."""
    if _is_struct(kind):
        if kind in _seen:
            return None
        sname = _struct_name(kind)
        total = 0
        for fn_ in structs.fields.get(sname, ()):
            fs = _kind_size(structs.field_kind(sname, fn_), structs, variants,
                            _seen + (kind,))
            if fs is None:
                return None
            total += fs
        return total
    if _is_enum(kind):
        return 8 + 8 * variants.payload_max.get(_enum_name(kind), 0)
    if _is_closure(kind):
        return 16
    return 8  # i64 / f64 / str-ptr (and the CONFLICT sentinel, never emitted)


# ---------------------------------------------------------------------------
# Module-wide closure table (make_closure targets and their captures)
# ---------------------------------------------------------------------------

@dataclass
class _ClosureTable:
    # lambda name -> capture names, in env-struct field order.
    targets: Dict[str, Tuple[str, ...]] = field(default_factory=dict)
    bad: Dict[str, str] = field(default_factory=dict)
    # (lambda name, capture name) -> kind (two-way cells, like parameters:
    # the creator's captured value and the lambda's uses are one value).
    cells: Dict[Tuple[str, str], str] = field(default_factory=dict)
    # lambda name -> arity declared at the make_closure site.
    arity: Dict[str, int] = field(default_factory=dict)
    # Lambdas whose envs are malloc'd at the site (leaked by design) because
    # their closures may escape: returned anywhere in the module, or created
    # inside a CFG cycle.  Membership is conservative and only ever ADDS heap
    # allocation — a heap env is always sound (it can never dangle).
    heap_env: Set[str] = field(default_factory=set)

    def cell_kind(self, lname: str, cap: str) -> str:
        return self.cells.get((lname, cap), I64)

    def mark_cell(self, lname: str, cap: str, kind: str) -> bool:
        key = (lname, cap)
        cur = self.cells.get(key, I64)
        nk = _join(cur, kind)
        if nk != cur:
            self.cells[key] = nk
            return True
        return False


def _build_closure_table(funcs: Sequence[MirFunc]) -> _ClosureTable:
    table = _ClosureTable()
    for f in funcs:
        for b in f.blocks:
            for op in b.ops:
                if op[0] != "let" or len(op) != 4 or op[2][0] != "make_closure":
                    continue
                lname = op[2][1]
                caps = tuple(cn for (cn, _vn) in op[3])
                arity = len(op[2][2]) if len(op[2]) > 2 else 0
                if lname not in table.targets:
                    table.targets[lname] = caps
                    table.arity[lname] = arity
                elif table.targets[lname] != caps or table.arity[lname] != arity:
                    table.bad.setdefault(
                        lname,
                        f"lambda {lname!r} created with inconsistent capture "
                        "lists across make_closure sites")
    return table


# ---------------------------------------------------------------------------
# Module-wide mutable-capture cell table (increment 12)
# ---------------------------------------------------------------------------
#
# A `cell_wrap` op boxes a binding into a shared one-word heap cell so the
# sub-functions capturing it can WRITE BACK (mir_interp.MxCell).  Natively a
# cell-backed variable's storage is a malloc(8) block (leaked by design —
# an immortal cell can never dangle): every read loads through the cell
# pointer, every write stores through it, and closure / handle-scope envs
# capture the POINTER (aliasing is the whole point).  The variable's KIND
# stays its element kind everywhere; only the storage class changes.
#
# Soundness rule (interpreter parity): the interpreter's cell springs into
# existence AT the cell_wrap op — a closure created BEFORE the wrap
# captured a frozen VALUE copy, and a delayed call of it must keep seeing
# the frozen value.  Whole-function cell backing would show it live
# updates instead, so any capture of a cell-backed variable that is not
# PROVABLY after a wrap (same block earlier, an entry-block wrap, or the
# variable itself arriving as a cell capture) demotes the function.  For
# every use that is not a capture, cell-from-entry is observationally
# identical (the cell always holds the binding's current value).

@dataclass
class _CellTable:
    # function name -> variable names backed by a heap cell in that frame.
    backed: Dict[str, Set[str]] = field(default_factory=dict)
    # lambda / scope-member name -> capture names that arrive as CELL
    # POINTERS through the env struct (subset of backed[fn]).
    cap_cells: Dict[str, Set[str]] = field(default_factory=dict)
    # handle site -> env field names holding cell pointers.
    scope_cells: Dict[str, Set[str]] = field(default_factory=dict)
    # function name -> demotion reasons found by the prescan.
    bad: Dict[str, List[str]] = field(default_factory=dict)

    def add_bad(self, fname: str, reason: str) -> None:
        rs = self.bad.setdefault(fname, [])
        if reason not in rs:
            rs.append(reason)


def _build_cell_table(funcs: Sequence[MirFunc],
                      scopes: _ScopeTable) -> _CellTable:
    table = _CellTable()
    by_name = {f.name: f for f in funcs}

    # 1. Own wraps: variables named by cell_wrap ops, with their positions
    # (block index, op index) for the post-wrap provability check.
    wrap_pos: Dict[str, Dict[str, List[Tuple[int, int]]]] = {}
    for f in funcs:
        own = table.backed.setdefault(f.name, set())
        wp = wrap_pos.setdefault(f.name, {})
        for bi, b in enumerate(f.blocks):
            for oi, op in enumerate(b.ops):
                if op[0] == "cell_wrap":
                    own.add(op[1])
                    wp.setdefault(op[1], []).append((bi, oi))

    # 2. Propagate cellness through captures to a fixpoint: a cell-backed
    # value captured into a lambda env or a handle-site env makes the
    # target's capture a cell pointer (and the name cell-backed there, so
    # nested captures chain).
    changed = True
    while changed:
        changed = False
        for f in funcs:
            backed = table.backed.setdefault(f.name, set())
            for b in f.blocks:
                for op in b.ops:
                    if op[0] != "let" or len(op) != 4:
                        continue
                    rk = op[2][0]
                    if rk == "make_closure":
                        lname = op[2][1]
                        for (cn, vn) in op[3]:
                            if isinstance(vn, str) and vn in backed:
                                cc = table.cap_cells.setdefault(lname, set())
                                lb = table.backed.setdefault(lname, set())
                                if cn not in cc or cn not in lb:
                                    cc.add(cn)
                                    lb.add(cn)
                                    changed = True
                    elif rk == "handle_scope":
                        site = op[2][1]
                        rec = scopes.sites.get(site)
                        if rec is None:
                            continue
                        for n in scopes.env_fields.get(site, ()):
                            vn = rec.cap_vals.get(n, n)
                            if vn not in backed:
                                continue
                            sc = table.scope_cells.setdefault(site, set())
                            if n not in sc:
                                sc.add(n)
                                changed = True
                            for m in rec.member_fns():
                                if n not in scopes.free_names.get(m, ()):
                                    continue
                                cc = table.cap_cells.setdefault(m, set())
                                mb = table.backed.setdefault(m, set())
                                if n not in cc or n not in mb:
                                    cc.add(n)
                                    mb.add(n)
                                    changed = True

    # 3. Post-wrap provability + mixed-capture validation per site.
    def provably_wrapped(fname: str, vn: str, bi: int, oi: int) -> bool:
        if vn in table.cap_cells.get(fname, ()):
            return True  # arrived as a cell: cell-backed from entry for real
        for (wbi, woi) in wrap_pos.get(fname, {}).get(vn, ()):
            if wbi == bi and woi < oi:
                return True  # same block, earlier op
            if wbi == 0 and bi != 0:
                return True  # entry block dominates every other block
        return False

    for f in funcs:
        backed = table.backed.get(f.name, set())
        if not backed:
            continue
        for bi, b in enumerate(f.blocks):
            for oi, op in enumerate(b.ops):
                if op[0] != "let" or len(op) != 4:
                    continue
                rk = op[2][0]
                if rk == "make_closure":
                    lname = op[2][1]
                    for (cn, vn) in op[3]:
                        if not isinstance(vn, str) or vn not in backed:
                            # Value capture at this site, but the lambda may
                            # expect a cell (marked from another site).
                            if cn in table.cap_cells.get(lname, ()):
                                table.add_bad(
                                    f.name,
                                    f"capture {cn!r} of lambda {lname!r} is a "
                                    "cell at another site but a value here")
                                table.add_bad(
                                    lname,
                                    f"capture {cn!r} is a cell at one "
                                    "make_closure site and a value at another")
                            continue
                        if not provably_wrapped(f.name, vn, bi, oi):
                            table.add_bad(
                                f.name,
                                f"closure capture of {vn!r} is not provably "
                                "after its cell_wrap (a pre-wrap capture "
                                "freezes a VALUE copy in the interpreter)")
                elif rk == "handle_scope":
                    site = op[2][1]
                    rec = scopes.sites.get(site)
                    if rec is None:
                        continue
                    for n in table.scope_cells.get(site, ()):
                        vn = rec.cap_vals.get(n, n)
                        if vn in backed and not provably_wrapped(
                                f.name, vn, bi, oi):
                            table.add_bad(
                                f.name,
                                f"handle-scope capture of {vn!r} is not "
                                "provably after its cell_wrap")
    # Cell-backed variables in functions the module does not know cannot
    # happen (wraps come from the functions themselves); nothing to prune.
    _ = by_name
    return table


# ---------------------------------------------------------------------------
# Module-constant (global) kind table (increment 12)
# ---------------------------------------------------------------------------

@dataclass
class _GlobalTable:
    """Module constants declared by __module_init: one module-wide kind
    cell per name, joined across the initializer's stores and every
    reader (exactly the sig/struct-field discipline)."""
    names: Set[str] = field(default_factory=set)
    kinds: Dict[str, str] = field(default_factory=dict)

    def kind(self, n: str) -> str:
        return self.kinds.get(n, I64)

    def mark(self, n: str, kind: str) -> bool:
        cur = self.kinds.get(n, I64)
        nk = _join(cur, kind)
        if nk != cur:
            self.kinds[n] = nk
            return True
        return False


def _build_global_table(funcs: Sequence[MirFunc]) -> _GlobalTable:
    table = _GlobalTable()
    for f in funcs:
        table.names.update(f.globals_decl)
    return table


# ---------------------------------------------------------------------------
# Module-wide handle-scope table (effects: sites, members, kind cells)
# ---------------------------------------------------------------------------
#
# Every MIR `handle_scope` op names a body subfunction and per-op handler
# case subfunctions (lower_hir_to_mir generates one unique set per handle
# expression).  Natively each SITE gets one shared env struct filled by the
# owner; body/case fns reload their free names from it (lambda-style).
# Because perform routing is DYNAMIC (innermost non-busy scope handling the
# op name), kinds crossing the boundary unify through module-wide cells
# keyed by OP NAME (args, results) and by SITE (the handle value).

@dataclass
class _ScopeSite:
    site: str                 # site id == body fn name (unique per handle/try)
    owner: str                # function containing the handle_scope op
    body_fn: str
    effect: str               # '' matches any effect at routing time
    # (op name, declared case params (without __k), case fn name) in the
    # site's dense op-index order (the dispatcher switches on this index).
    cases: Tuple[Tuple[str, Tuple[str, ...], str], ...]
    cap_vals: Dict[str, str] = field(default_factory=dict)  # cap name -> value var
    # "handle" (handle_scope: body + handler cases over mx_handle) or
    # "try" (try_scope: body + ONE catch fn over mx_try).  A try site reuses
    # this record wholesale — same shared env struct, same free-name
    # fixpoint, same value cell — because the two constructs are the same
    # shape: a delimited body subfunction plus recovery subfunctions, all
    # reading the owner's captured names out of one env block.
    kind: str = "handle"
    catch_fn: str = ""        # try sites only

    def member_fns(self) -> Tuple[str, ...]:
        if self.kind == "try":
            return (self.body_fn, self.catch_fn)
        return (self.body_fn,) + tuple(hfn for (_o, _p, hfn) in self.cases)


@dataclass
class _ScopeTable:
    sites: Dict[str, _ScopeSite] = field(default_factory=dict)
    member_site: Dict[str, str] = field(default_factory=dict)  # fn -> site id
    member_role: Dict[str, str] = field(default_factory=dict)  # fn -> body|case
    case_op: Dict[str, str] = field(default_factory=dict)      # case fn -> op
    # member fn -> its OWN free names (reloaded from the site env).
    free_names: Dict[str, Tuple[str, ...]] = field(default_factory=dict)
    # site -> env field order (sorted union of member free names).
    env_fields: Dict[str, Tuple[str, ...]] = field(default_factory=dict)
    # (site, name) -> env field kind (two-way cells, like closure captures).
    cells: Dict[Tuple[str, str], str] = field(default_factory=dict)
    # site -> handle-value kind (handle dst ⊔ body ret ⊔ case ret ⊔ resume
    # results: they are all the same delimited-body completion value).
    value_cells: Dict[str, str] = field(default_factory=dict)
    # op name -> perform-result kind (perform dsts ⊔ resume values), and
    # (op name, index) -> op argument kind (perform args ⊔ case params).
    # Keyed by op name ALONE because routing is dynamic by op name — a
    # sound over-approximation joining every scope that could catch it.
    op_results: Dict[str, str] = field(default_factory=dict)
    op_args: Dict[Tuple[str, int], str] = field(default_factory=dict)
    sites_of_owner: Dict[str, List[str]] = field(default_factory=dict)
    bad: Dict[str, str] = field(default_factory=dict)  # fn name -> reason

    def _mark(self, store, key, kind: str) -> bool:
        cur = store.get(key, I64)
        nk = _join(cur, kind)
        if nk != cur:
            store[key] = nk
            return True
        return False

    def cell_kind(self, site: str, name: str) -> str:
        return self.cells.get((site, name), I64)

    def mark_cell(self, site: str, name: str, kind: str) -> bool:
        return self._mark(self.cells, (site, name), kind)

    def value_kind(self, site: str) -> str:
        return self.value_cells.get(site, I64)

    def mark_value(self, site: str, kind: str) -> bool:
        return self._mark(self.value_cells, site, kind)

    def op_result_kind(self, op: str) -> str:
        return self.op_results.get(op, I64)

    def mark_op_result(self, op: str, kind: str) -> bool:
        return self._mark(self.op_results, op, kind)

    def op_arg_kind(self, op: str, i: int) -> str:
        return self.op_args.get((op, i), I64)

    def mark_op_arg(self, op: str, i: int, kind: str) -> bool:
        return self._mark(self.op_args, (op, i), kind)

    def mark_bad(self, rec: _ScopeSite, reason: str) -> None:
        """A site problem demotes the owner and every member."""
        self.bad.setdefault(rec.owner, reason)
        for m in rec.member_fns():
            self.bad.setdefault(m, reason)


def _op_uses_defs(op: tuple) -> Tuple[Set[str], Set[str]]:
    """Per-op (used names, defined names), mirroring the categorization in
    ``_fn_defs_uses_sites`` exactly (callee names are NOT uses; capture
    values of a nested handle/try site ARE uses at the site op — the env
    fill reads them there, which is what the exposure analysis orders)."""
    ou: Set[str] = set()
    od: Set[str] = set()
    k = op[0]
    if k == "params":
        od.update(op[1])
    elif k == "perform" and len(op) >= 7:
        ou.update(op[4])
        od.add(op[1])
    elif k == "promote_matrix":
        ou.update(op[1] if len(op) > 1 else ())
    elif k == "cell_wrap":
        ou.add(op[1])
    elif k == "let" and len(op) == 4:
        _, dst, rhs, args = op
        od.add(dst)
        rk = rhs[0]
        if rk == "alloc_struct" or rk == "make_closure":
            ou.update(v for (_n, v) in args if isinstance(v, str))
        elif rk == "handle_scope" or rk == "try_scope":
            ou.update(v for (_n, v) in args if isinstance(v, str))
        else:  # call included: args are uses, the callee NAME is not
            ou.update(a for a in args if isinstance(a, str))
    return ou, od


def _upward_exposed(f: MirFunc) -> Set[str]:
    """Names possibly READ BEFORE ANY DEF on some entry path (live-in at
    the entry block): classic backward liveness over gen/kill per block.

    Why this matters: a scope member that REBINDS a captured binding after
    reading it (the store-back shape field/index updates lower to —
    `h.data[0] = 9` becomes read h -> mx_vec_set -> rebind h) has the name
    in both uses and defs, and plain `uses - defs` calls it local.  It is
    not: the first read must see the ENCLOSING binding, so the name needs
    an env field like any other capture (by value — interpreter parity:
    the rebind itself stays local there too, while the vec mutation
    travels by identity)."""
    n = len(f.blocks)
    gen: List[Set[str]] = []
    kill: List[Set[str]] = []
    succs: List[Tuple[int, ...]] = []
    for b in f.blocks:
        g: Set[str] = set()
        d: Set[str] = set()
        for op in b.ops:
            ou, od = _op_uses_defs(op)
            g |= (ou - d)
            d |= od
        t = b.term
        if t[0] in ("br_if", "ret") and isinstance(t[1], str) \
                and t[1] not in d:
            g.add(t[1])
        gen.append(g)
        kill.append(d)
        if t[0] == "br":
            succs.append((t[1],))
        elif t[0] == "br_if":
            succs.append((t[2], t[3]))
        else:
            succs.append(())
    live_in: List[Set[str]] = [set() for _ in range(n)]
    changed = True
    while changed:
        changed = False
        for i in range(n - 1, -1, -1):
            out: Set[str] = set()
            for s in succs[i]:
                if 0 <= s < n:
                    out |= live_in[s]
            ni = gen[i] | (out - kill[i])
            if ni != live_in[i]:
                live_in[i] = ni
                changed = True
    return live_in[0] if n else set()


def _fn_defs_uses_sites(
        f: MirFunc) -> Tuple[Set[str], Set[str], Set[str], List[str],
                             Set[str]]:
    """(defined names, directly used names, call CALLEE names, handle sites
    contained, upward-exposed names) of a function — the base facts for the
    free-name fixpoint.  Callee names are kept separate: a callee is a free
    name only when the handle site actually captured a binding of that name
    (interpreter resolution order: the env shadows module functions and
    builtins), so plain calls to module functions must not force phantom
    captures.  Upward-exposed names (read before any def on some path) are
    free even when locally rebound — see ``_upward_exposed``."""
    defs: Set[str] = set()
    uses: Set[str] = set()
    callees: Set[str] = set()
    sites: List[str] = []
    for b in f.blocks:
        for op in b.ops:
            k = op[0]
            if k == "let" and len(op) == 4 and op[2][0] in ("handle_scope",
                                                            "try_scope"):
                sites.append(op[2][1])
                defs.add(op[1])
                # capture VALUES are used only as far as the site's
                # members need them (the lowering captures every env
                # name conservatively; the interpreter is non-strict
                # about unbound ones) — added during the fixpoint.
                continue
            ou, od = _op_uses_defs(op)
            uses |= ou
            defs |= od
            if k == "let" and len(op) == 4 and op[2][0] == "call" \
                    and len(op[2]) > 1 and isinstance(op[2][1], str):
                callees.add(op[2][1])
        t = b.term
        if t[0] in ("br_if", "ret") and isinstance(t[1], str):
            uses.add(t[1])
    return defs, uses, callees, sites, _upward_exposed(f)


def _build_scope_table(funcs: Sequence[MirFunc]) -> _ScopeTable:
    table = _ScopeTable()
    by_name = {f.name: f for f in funcs}
    # 1. Collect sites.
    for f in funcs:
        for b in f.blocks:
            for op in b.ops:
                if op[0] != "let" or len(op) != 4:
                    continue
                if op[2][0] == "try_scope":
                    # ("try_scope", body_fn, catch_fn), captures.  Same
                    # record shape as a handle site with no cases: one
                    # delimited body plus one recovery subfunction sharing
                    # the owner's env block (docs/try_catch.md).
                    tbody, tcatch = op[2][1], op[2][2]
                    trec = _ScopeSite(
                        site=tbody, owner=f.name, body_fn=tbody, effect="",
                        cases=(), kind="try", catch_fn=tcatch,
                        cap_vals={cn: vn for (cn, vn) in op[3]
                                  if isinstance(vn, str)})
                    if tbody in table.sites:
                        table.mark_bad(trec, f"try site {tbody!r} appears at "
                                             "multiple try_scope ops")
                        table.mark_bad(table.sites[tbody],
                                       f"try site {tbody!r} appears at "
                                       "multiple try_scope ops")
                        continue
                    table.sites[tbody] = trec
                    table.sites_of_owner.setdefault(f.name, []).append(tbody)
                    continue
                if op[2][0] != "handle_scope":
                    continue
                body_fn, effect = op[2][1], op[2][2]
                cases: List[Tuple[str, Tuple[str, ...], str]] = []
                for (op_name, case_params, hfn) in op[2][3]:
                    if isinstance(case_params, str):  # legacy bare string
                        case_params = (case_params,)
                    cases.append((op_name, tuple(case_params), hfn))
                rec = _ScopeSite(
                    site=body_fn, owner=f.name, body_fn=body_fn, effect=effect,
                    cases=tuple(cases),
                    cap_vals={cn: vn for (cn, vn) in op[3]
                              if isinstance(vn, str)})
                if body_fn in table.sites:
                    table.mark_bad(rec, f"handle site {body_fn!r} appears at "
                                        "multiple handle_scope ops")
                    table.mark_bad(table.sites[body_fn],
                                   f"handle site {body_fn!r} appears at "
                                   "multiple handle_scope ops")
                    continue
                table.sites[body_fn] = rec
                table.sites_of_owner.setdefault(f.name, []).append(body_fn)
    # 2. Register members.
    for site, rec in table.sites.items():
        if rec.kind == "try":
            roles = [(rec.body_fn, "trybody", None),
                     (rec.catch_fn, "trycatch", None)]
        else:
            roles = [(rec.body_fn, "body", None)] + [
                (hfn, "case", op_name) for (op_name, _p, hfn) in rec.cases]
        for (fname, role, op_name) in roles:
            if fname not in by_name:
                table.mark_bad(rec, f"handle-scope subfunction {fname!r} "
                                    "missing from the module")
                continue
            if fname in table.member_site and table.member_site[fname] != site:
                table.mark_bad(rec, f"function {fname!r} belongs to multiple "
                                    "handle sites")
                table.mark_bad(table.sites[table.member_site[fname]],
                               f"function {fname!r} belongs to multiple "
                               "handle sites")
                continue
            table.member_site[fname] = site
            table.member_role[fname] = role
            if role == "case":
                table.case_op[fname] = op_name
    # 3. Free-name fixpoint per member (a member's needs include the needs
    # of any handle sites nested inside it, mapped through those sites'
    # capture-value names).
    base = {name: _fn_defs_uses_sites(f) for name, f in by_name.items()}
    free: Dict[str, Set[str]] = {m: set() for m in table.member_site}

    def site_needs(site_id: str) -> Set[str]:
        rec2 = table.sites.get(site_id)
        if rec2 is None:
            return set()
        need: Set[str] = set()
        for m2 in rec2.member_fns():
            for n in free.get(m2, ()):
                need.add(rec2.cap_vals.get(n, n))
        return need

    # Module constants (declared by __module_init) resolve as GLOBAL reads
    # unless the site's captures shadow them (interpreter order: env first,
    # then globals) — so a member's free global names do not force env
    # captures the owner cannot provide.
    global_names = {n for f in funcs for n in f.globals_decl}

    # Cell-wrapped names (mutable captures): a member that both READS and
    # ASSIGNS such a name still needs it from the env — the binding lives
    # in the shared cell, and `uses - defs` alone would drop it (its local
    # def is a write THROUGH the capture, not a fresh binding).
    wrapped_names = {op[1] for f2 in funcs for b2 in f2.blocks
                     for op in b2.ops if op[0] == "cell_wrap"}

    changed = True
    while changed:
        changed = False
        for m in table.member_site:
            if m not in by_name:
                continue
            defs, uses, callees, inner_sites, exposed = base[m]
            site_caps = table.sites[table.member_site[m]].cap_vals
            need = set(uses)
            # A call whose callee name the site captured resolves to the
            # CAPTURED BINDING first (interpreter shadowing order): the
            # member needs it from the env — this is how handler cases
            # call function-valued parameters (`f(x)` inside `emit(x)`).
            # Callee names the site did NOT capture are module functions /
            # builtins and never become captures.
            need |= {c for c in callees if c in site_caps}
            for s2 in inner_sites:
                need |= site_needs(s2)
            # A name both used and defined stays free when it is
            # cell-wrapped (writes go through the shared cell) OR
            # upward-exposed (read before the local rebind: the first read
            # must see the enclosing binding — the store-back shape field
            # and index updates lower to).  Exposure alone is intersected
            # with the site's captures: liveness sees the impossible
            # br-to-join path after a raising match_fail, which makes
            # match-result TEMPS look read-before-def, and a temp is never
            # a site capture while a real enclosing binding always is.
            nf = (need - defs) | (need & defs
                                  & (wrapped_names
                                     | (exposed & set(site_caps))))
            nf -= {n for n in nf
                   if n in global_names and n not in site_caps}
            if nf != free[m]:
                free[m] = nf
                changed = True
    for m in table.member_site:
        table.free_names[m] = tuple(sorted(free.get(m, ())))
    # 4. Site env layout + capture validation.
    for site, rec in table.sites.items():
        union: Set[str] = set()
        for m in rec.member_fns():
            union |= free.get(m, set())
        table.env_fields[site] = tuple(sorted(union))
        missing = sorted(n for n in union if n not in rec.cap_vals)
        if missing:
            what = "try" if rec.kind == "try" else "handle-scope"
            table.mark_bad(rec, f"{what} subfunctions reference names "
                                f"absent from the site's captures: {missing}")
    return table

# ---------------------------------------------------------------------------
# Module-wide trait-impl table and static trait-call resolution
# ---------------------------------------------------------------------------
#
# `recv.m(args)` lowers to a `__trait$m` call; the interpreter dispatches on
# the RECEIVER'S RUNTIME TYPE NAME (mir_interp._dispatch_trait_call):
#   1. __impl$Trait$Type$m for the receiver's type (exact, then
#      case-insensitive);  2. builtin m;  3. plain function m;  4. error.
# Kind inference gives us the receiver's kind statically, so the same
# resolution runs at compile time: struct:T/enum:E map to type name T/E,
# vec to "Vec", str to "String", f64 to "Float".  A receiver whose kind is
# i64 is AMBIGUOUS (ints, bools and unit are all kind-erased to i64), so an
# i64 receiver resolves to a builtin only when NO impl exists for any of
# Int/Bool/Unit — otherwise the native dispatch could pick a different
# target than the interpreter and the function demotes instead.

@dataclass
class _TraitTable:
    # method -> type name -> {trait name: impl function name}
    by_method: Dict[str, Dict[str, Dict[str, str]]] = field(default_factory=dict)


def _build_trait_table(module_names: Set[str]) -> _TraitTable:
    table = _TraitTable()
    for name in module_names:
        parsed = parse_impl_method_name(name)
        if parsed is None:
            continue
        trait_name, type_name, method = parsed
        table.by_method.setdefault(method, {}).setdefault(
            type_name, {})[trait_name] = name
    return table


def _resolve_static_call(callee: str, traits: _TraitTable,
                         module_names: Set[str]) -> Tuple[str, str]:
    """Statically resolve a `__static$Type$method` call (no receiver).

    Mirrors mir_interp._dispatch_static_call exactly:
      1. the unique __impl$Trait$Type$method function (EXACT type-name
         match — the interpreter's impl index does not case-fold here);
         multiple candidate traits are an interpreter error -> demote;
      2. the plain dotted module function "Type.method";
      3. the dotted builtin ("Vec.new").
    Returns ("func", fname) | ("builtin", name) | ("demote", reason).
    Purely static — resolution needs no value kinds, so it never changes
    across the fixpoint."""
    rest = callee[len(STATIC_CALL_PREFIX):]
    type_name, _, method = rest.partition(IMPL_SEP)
    tm = traits.by_method.get(method, {}).get(type_name)
    if tm:
        if len(tm) > 1:
            opts = ", ".join(sorted(tm))
            return ("demote",
                    f"ambiguous static method {method!r} on type "
                    f"{type_name!r} (implemented by traits: {opts})")
        return ("func", next(iter(tm.values())))
    dotted = f"{type_name}.{method}"
    if dotted in module_names:
        return ("func", dotted)
    if dotted in _NATIVE_RT_CALLS:
        return ("builtin", dotted)
    return ("demote",
            f"static method {method!r} on type {type_name!r} has no impl "
            "and no native dotted fallback")


# Scalar type names the interpreter reports for kind-erased i64 receivers.
_I64_RUNTIME_TYPE_NAMES = {"int", "bool", "unit"}


def _resolve_trait_call(method: str, recv_kind: str, traits: _TraitTable,
                        module_names: Set[str], *, assume_final: bool,
                        ) -> Tuple[str, Optional[str]]:
    """Statically resolve a `__trait$method` call for a receiver kind.

    Returns one of:
      ("func", fname)     -- call the module function fname directly
      ("builtin", name)   -- lower as the interpreter builtin `name`
      ("pending", None)   -- receiver kind still i64/bottom; try again later
                             (only when not assume_final)
      ("demote", reason)  -- cannot be resolved soundly; demote the function
    The mapping mirrors mir_interp._dispatch_trait_call exactly; anything it
    would decide differently at runtime returns "demote", never a guess.
    """
    by_type = traits.by_method.get(method, {})

    def impl_for(tyname: str) -> Tuple[Optional[Dict[str, str]], Optional[str]]:
        """(trait->fn map, problem).  Exact name first, then case-insensitive
        (the interpreter's order); two distinct case-folded keys demote."""
        tm = by_type.get(tyname)
        if tm is not None:
            return tm, None
        low = tyname.lower()
        matches = [tm for k, tm in by_type.items() if k.lower() == low]
        if len(matches) > 1:
            return None, (f"trait method {method!r}: multiple case-folded "
                          f"impl types match receiver type {tyname!r}")
        return (matches[0] if matches else None), None

    def from_impl(tyname: str) -> Optional[Tuple[str, Optional[str]]]:
        tm, prob = impl_for(tyname)
        if prob is not None:
            return ("demote", prob)
        if tm is None:
            return None
        if len(tm) > 1:
            opts = ", ".join(sorted(tm))
            return ("demote",
                    f"ambiguous trait method {method!r} on type {tyname!r} "
                    f"(implemented by traits: {opts})")
        return ("func", next(iter(tm.values())))

    def plain_fn_fallback(reason: str) -> Tuple[str, Optional[str]]:
        # Interpreter step 3: a plain user function of the same name — but
        # ONLY when the method is not a builtin (builtins win at step 2).
        if method not in _TRAIT_BUILTIN_FALLBACK and method in module_names:
            return ("func", method)
        return ("demote", reason)

    if recv_kind == CONFLICT:
        return ("demote", f"trait method {method!r} receiver has conflicting kinds")

    if _is_struct(recv_kind) or _is_enum(recv_kind):
        tyname = _struct_name(recv_kind) if _is_struct(recv_kind) \
            else _enum_name(recv_kind)
        hit = from_impl(tyname)
        if hit is not None:
            return hit
        # Builtin fallback on an aggregate receiver: to_string would use
        # Python str() of the struct/variant, len/push/pop would raise —
        # neither is representable natively.
        if method in _TRAIT_BUILTIN_FALLBACK:
            return ("demote",
                    f"trait method {method!r} on {recv_kind} falls back to the "
                    "interpreter builtin (not representable natively)")
        return plain_fn_fallback(
            f"trait method {method!r} has no impl for type {tyname!r}")

    if _is_vec(recv_kind):
        hit = from_impl("Vec")
        if hit is not None:
            return hit
        if method in ("push", "pop", "len"):
            return ("builtin", method)
        if method in ("to_string", "int_to_str"):
            return ("demote",
                    "to_string of a Vec (interpreter renders 'Vec[...]'; no "
                    "native equivalent)")
        return plain_fn_fallback(
            f"trait method {method!r} on a Vec receiver has no native lowering")

    if _is_fvec(recv_kind):
        # The interpreter reports "vector" for MxVector receivers (the head
        # type constructor `implement ... for vector[T, N]` desugars to).
        hit = from_impl("vector")
        if hit is not None:
            return hit
        if method == "len":
            return ("builtin", "len")
        if method in ("to_string", "int_to_str"):
            # str(MxVector) == "vector[...]" == mx_fvec_to_str; the
            # element-kind restrictions are checked like any to_string.
            return ("builtin", "to_string")
        if method in ("push", "pop"):
            return ("demote",
                    f"trait method {method!r} on an immutable fixed vector "
                    "(the interpreter rejects non-Vec receivers)")
        return plain_fn_fallback(
            f"trait method {method!r} on a fixed-vector receiver has no "
            "native lowering")

    if recv_kind == STR:
        hit = from_impl("String")
        if hit is not None:
            return hit
        if method == "len":
            return ("builtin", "len")
        if method in ("to_string", "int_to_str"):
            return ("builtin", "to_string")
        return plain_fn_fallback(
            f"trait method {method!r} on a string receiver has no native lowering")

    if recv_kind == F64:
        hit = from_impl("Float")
        if hit is not None:
            return hit
        if method in _MATH_EXTERNS:
            return ("builtin", method)
        if method in ("to_string", "int_to_str"):
            return ("builtin", "to_string")
        return plain_fn_fallback(
            f"trait method {method!r} on a float receiver has no native lowering")

    if _is_closure(recv_kind):
        return ("demote", f"trait method {method!r} on a closure receiver")

    # recv_kind == I64: bottom (still unresolved) OR genuinely int/bool/unit.
    if not assume_final:
        return ("pending", None)
    erased = [t for t in by_type if t.lower() in _I64_RUNTIME_TYPE_NAMES]
    if erased:
        return ("demote",
                f"trait method {method!r} has impls for kind-erased scalar "
                f"receiver types ({', '.join(sorted(erased))}); i64 receivers "
                "cannot be dispatched statically")
    if method in _MATH_EXTERNS:
        return ("builtin", method)
    if method in ("to_string", "int_to_str"):
        return ("builtin", "to_string")
    if method in ("push", "pop", "len"):
        return ("demote",
                f"trait method {method!r} on a receiver of kind i64 "
                "(interpreter would reject a non-Vec receiver)")
    return plain_fn_fallback(
        f"trait method {method!r} on a receiver of kind i64 has no native lowering")


def _promote_kind(k: str) -> str:
    """promote_matrix's static effect on a parameter kind: a flat vector of
    numbers becomes a matrix of one-element rows; everything else passes
    through untouched (mir_interp promotes only MxVectors whose elements
    are all numbers, which is exactly the flat-numeric KIND — including the
    empty vector, whose promotion is observationally invisible)."""
    if _is_fvec(k) and _fvec_elem(k) in (I64, F64):
        return _fvec_of(k)
    return k


# ---------------------------------------------------------------------------
# Kind inference (i64 by default, monotone promotion; module fixpoint)
# ---------------------------------------------------------------------------

@dataclass
class _Sig:
    params: List[str]
    ret: str = I64


def _infer_kinds(info: _Info, sigs: Dict[str, _Sig], structs: _StructTable,
                 variants: _VariantTable, closures: _ClosureTable,
                 traits: _TraitTable, scopes: _ScopeTable,
                 module_names: Set[str], gtable: "_GlobalTable",
                 assume_final: bool = False,
                 ) -> Tuple[Dict[str, str], bool]:
    """One inner fixpoint over a function.  Returns (kinds, global_changed)
    where global_changed reports promotions written into shared cells
    (struct fields, enum payload slots, closure captures) so the module
    driver keeps iterating.

    ``assume_final``: trait-call receivers still at the i64 bottom are
    treated as genuinely-int receivers (the driver sets this only after the
    unassuming fixpoint has converged, so nothing else can promote them)."""
    kinds: Dict[str, str] = {}
    global_changed = False

    # Integer-constant temps (MIR const ops): tile ctor shapes must be
    # statically known to become part of a `tile:` kind.  ANF makes ctor
    # shape arguments freshly-consted temps, so direct consts suffice; a
    # non-const shape simply never produces a tile kind and the
    # consistency check demotes the site with a clear reason.
    int_consts: Dict[str, int] = {}
    for _b in info.f.blocks:
        for _op in _b.ops:
            if _op[0] == "let" and len(_op) == 4 and _op[2][0] == "const":
                _v = _op[2][1]
                if isinstance(_v, int) and not isinstance(_v, bool):
                    int_consts[_op[1]] = _v

    def get(n: str) -> str:
        return kinds.get(n, I64)

    def mark(n: str, k: str) -> bool:
        nk = _join(get(n), k)
        if nk != get(n):
            kinds[n] = nk
            return True
        return False

    def unify(names: Sequence[str]) -> bool:
        k = I64
        for n in names:
            k = _join(k, get(n))
        changed = False
        for n in names:
            changed = mark(n, k) or changed
        return changed

    # Field name -> owning struct, when the module has EXACTLY one struct
    # with that field: a field_get/field_set receiver still at the i64
    # bottom can then only be that struct (any other value would be an
    # interpreter error), so its kind is pinned.  Ambiguous names stay
    # unresolved and demote via the existing consistency check.
    field_owner: Dict[str, Optional[str]] = {}
    for sname_, fields_ in structs.fields.items():
        for fn_ in fields_:
            field_owner[fn_] = None if fn_ in field_owner else sname_

    def apply_builtin(name: str, dst: str, args: Tuple[str, ...],
                      plain_call: bool) -> bool:
        """Kind constraints of a native runtime builtin call.  For calls
        already resolved to a builtin by NAME (plain and method-position
        `__builtin$m` sites) receiver kinds are pinned eagerly; for
        trait-resolved calls the receiver is already known to be a
        vec/str (resolution is kind-driven)."""
        nonlocal global_changed
        ch = False
        if name == "Vec.new":
            ch = mark(dst, _vec_of(I64)) or ch
        elif name in ("push", "pop", "__index_get"):
            if not args:
                return ch
            if plain_call:
                if name in ("push", "pop"):
                    # The interpreter rejects push/pop on anything but a
                    # Vec, so the receiver of an accepted program IS one.
                    # A receiver already known to be a FIXED vector is left
                    # alone so the consistency check reports the clean
                    # "fixed vectors are immutable" demotion instead of a
                    # kind conflict.
                    if not _is_fvec(get(args[0])):
                        ch = mark(args[0], _vec_of(I64)) or ch
                elif get(args[0]) == I64 and assume_final:
                    # __index_get also accepts fixed vectors (and ranges),
                    # so only a receiver nothing else could type is pinned
                    # to the Vec bottom, and only once the fixpoint has
                    # settled.
                    ch = mark(args[0], _vec_of(I64)) or ch
            rk = get(args[0])
            if _is_vec(rk):
                # Two-way element unification: pushed values and read
                # elements are one type per vec.  (push's own dst is unit.)
                if name == "push":
                    other = args[1] if len(args) == 2 else None
                else:
                    other = dst
                if other is not None:
                    nk = _join(_vec_elem(rk), get(other))
                    if nk != CONFLICT:
                        ch = mark(args[0], _vec_of(nk)) or ch
                        ch = mark(other, nk) or ch
                    else:
                        ch = mark(args[0], CONFLICT) or ch
            elif _is_fvec(rk) and name == "__index_get":
                nk = _join(_fvec_elem(rk), get(dst))
                if nk != CONFLICT:
                    ch = mark(args[0], _fvec_of(nk)) or ch
                    ch = mark(dst, nk) or ch
                else:
                    ch = mark(args[0], CONFLICT) or ch
        elif name == "__vec_lit":
            # ("__vec_lit", size, e0, e1, ...): a fixed-size vector literal
            # builds an immutable mx_fvec block (increment 10); elements
            # unify two-way with the element kind (the size argument stays
            # i64).
            ch = mark(dst, _fvec_of(I64)) or ch
            rk = get(dst)
            if _is_fvec(rk):
                ek = _fvec_elem(rk)
                for e in args[1:]:
                    ek = _join(ek, get(e))
                if ek != CONFLICT:
                    ch = mark(dst, _fvec_of(ek)) or ch
                    for e in args[1:]:
                        ch = mark(e, ek) or ch
                else:
                    ch = mark(dst, CONFLICT) or ch
        elif name == "__vec_zeros":
            # ("__vec_zeros", n, base_name): the base-type name is a const
            # string; float/int decide the element kind (anything else is
            # an interpreter error -> the consistency check demotes).
            if len(args) == 2:
                base = info.const_strs.get(args[1], "").lower()
                zk = {"float": F64, "int": I64}.get(base)
                if zk is not None:
                    ch = mark(dst, _fvec_of(zk)) or ch
        elif name == "__vec_filled":
            # ("__vec_filled", n, value): value unifies two-way with the
            # element kind.
            if len(args) == 2:
                ch = mark(dst, _fvec_of(I64)) or ch
                rk = get(dst)
                if _is_fvec(rk):
                    ek = _join(_fvec_elem(rk), get(args[1]))
                    if ek != CONFLICT:
                        ch = mark(dst, _fvec_of(ek)) or ch
                        ch = mark(args[1], ek) or ch
                    else:
                        ch = mark(dst, CONFLICT) or ch
        elif name == "__range":
            # A range materializes as a fixed vector of ints; its uses are
            # restricted to iteration shapes (see the consistency check —
            # the interpreter's plain list has a different repr).
            ch = mark(dst, _fvec_of(I64)) or ch
        elif name == "__vec_dim":
            pass  # receiver typed by its producer; dst stays i64
        elif name == "__slice_get":
            # A slice of a fixed vector is a fresh vector of the same kind
            # (bounds stay i64 / None).
            if args:
                nk = _join(get(dst), get(args[0]))
                if nk != CONFLICT and _is_fvec(nk):
                    ch = mark(dst, nk) or ch
                    ch = mark(args[0], nk) or ch
        elif name == "__cast":
            # `e as T`: numeric targets convert the representation; any
            # other target is a static-level reinterpretation (identity).
            if len(args) == 2:
                t = info.const_strs.get(args[1], "")
                if t in ("float", "f32", "f64"):
                    ch = mark(dst, F64) or ch
                elif t in ("int", "i8", "i16", "i32", "i64",
                           "u8", "u16", "u32", "u64"):
                    pass  # dst stays i64
                else:
                    ch = unify((dst, args[0])) or ch
        elif name == "__vec_comprehension":
            # ("__vec_comprehension", n, closure, iterable): element kinds
            # flow one-way into the lambda's parameter, and the lambda's
            # return kind one-way into the destination's element kind.
            # (i64 values may flow into an f64 position: the per-element
            # thunk converts, the same int->float promotion contract as
            # scalar unification.)
            if len(args) == 3:
                fk = get(args[1])
                lsig = sigs.get(_closure_lambda(fk)) if _is_closure(fk) \
                    else None
                if lsig is not None and len(lsig.params) == 1:
                    itk = get(args[2])
                    if _is_fvec(itk):
                        nk = _join(lsig.params[0], _fvec_elem(itk))
                        if nk != lsig.params[0]:
                            lsig.params[0] = nk
                            ch = True
                            global_changed = True
                    ch = mark(dst, _fvec_of(lsig.ret)) or ch
                elif lsig is not None \
                        and args[2] in info.zip_defs \
                        and len(lsig.params) == len(info.zip_defs[args[2]]):
                    # Zip comprehension: each zipped SOURCE's element kind
                    # flows one-way into its lambda parameter (same
                    # int->float promotion contract as the single form);
                    # the lambda's return decides the result element kind.
                    for i, s in enumerate(info.zip_defs[args[2]]):
                        sk = get(s)
                        if _is_fvec(sk):
                            nk = _join(lsig.params[i], _fvec_elem(sk))
                            if nk != lsig.params[i]:
                                lsig.params[i] = nk
                                ch = True
                                global_changed = True
                    ch = mark(dst, _fvec_of(lsig.ret)) or ch
        elif name == "__index_store":
            # ("__index_store", recv, idx, val): store-back index
            # assignment — `place = __index_store(place, i, x)`.  The
            # result IS the receiver (same Vec pointer / functionally
            # updated vector), so dst and receiver unify; the value
            # unifies two-way with the element kind.
            if len(args) == 3:
                rk = get(args[0])
                if plain_call and rk == I64 and assume_final:
                    # A receiver nothing else could type: the interpreter
                    # only accepts Vec (in place) or vector[T,N]
                    # (functional update); pin the Vec bottom.
                    ch = mark(args[0], _vec_of(I64)) or ch
                    rk = get(args[0])
                if _is_vec(rk) or _is_fvec(rk):
                    elem = _vec_elem(rk) if _is_vec(rk) else _fvec_elem(rk)
                    of = _vec_of if _is_vec(rk) else _fvec_of
                    nk = _join(elem, get(args[2]))
                    if nk != CONFLICT:
                        ch = mark(args[0], of(nk)) or ch
                        ch = mark(args[2], nk) or ch
                    else:
                        ch = mark(args[0], CONFLICT) or ch
                    ch = unify((dst, args[0])) or ch
        elif name == "__index_set":
            # ("__index_set", recv, idx, val): in-place element store.
            # Only Vec receivers support it (the interpreter errors on
            # everything else — a fixed vector stays untouched here so the
            # consistency check reports the clean immutability demotion).
            if len(args) == 3:
                if plain_call and not _is_fvec(get(args[0])):
                    ch = mark(args[0], _vec_of(I64)) or ch
                rk = get(args[0])
                if _is_vec(rk):
                    nk = _join(_vec_elem(rk), get(args[2]))
                    if nk != CONFLICT:
                        ch = mark(args[0], _vec_of(nk)) or ch
                        ch = mark(args[2], nk) or ch
                    else:
                        ch = mark(args[0], CONFLICT) or ch
                # dst is unit -> stays i64
        elif name == "__zip":
            # The zip result stays at the i64 bottom deliberately: it is a
            # VIRTUAL value whose only legal use is a comprehension
            # iterable (the emission reads the sources directly).  The
            # sources are fixed vectors; one still at the bottom when the
            # fixpoint settles can only be a vector (ranges included).
            for a in args:
                if get(a) == I64 and assume_final:
                    ch = mark(a, _fvec_of(I64)) or ch
        elif name == "len":
            pass  # receiver may be vec/vector/str; dst stays i64
        elif name in ("to_string", "int_to_str"):
            ch = mark(dst, STR) or ch
        elif name in _MATH_EXTERNS:
            for a in args:
                ch = mark(a, F64) or ch
            ch = mark(dst, F64) or ch
        elif name.startswith("Tile."):
            ch = tile_builtin(name[len("Tile."):], dst, args) or ch
        return ch

    def tile_builtin(op: str, dst: str, args: Tuple[str, ...]) -> bool:
        """Kind rules of the Tile dotted statics (docs/gpu_tiles.md).

        Shapes come from int-constant ctor arguments and then FLOW through
        the ops as part of the `tile:` kind.  Element-kind promotion is
        kept monotone: an operand still at the i64 bottom that might yet
        become f64 (filled's fill value, from_vec's Vec) only pins an
        int-element tile once ``assume_final`` says nothing can promote
        it further — the same discipline the trait-receiver resolution
        uses.  Bad arities/shapes mark nothing; the consistency check
        turns them into demotion reasons."""
        ch = False

        def ctor_shape(r: str, c: str) -> Optional[Tuple[int, int]]:
            rv, cv = int_consts.get(r), int_consts.get(c)
            if rv is None or cv is None or rv <= 0 or cv <= 0:
                return None
            return rv, cv

        if op in ("zeros", "arange") and len(args) == 2:
            shape = ctor_shape(args[0], args[1])
            if shape is not None:
                elem = F64 if op == "zeros" else I64
                ch = mark(dst, _tile_of(elem, *shape)) or ch
        elif op == "filled" and len(args) == 3:
            shape = ctor_shape(args[0], args[1])
            if shape is not None:
                fk = get(args[2])
                if fk == F64:
                    ch = mark(dst, _tile_of(F64, *shape)) or ch
                elif fk == I64 and assume_final:
                    ch = mark(dst, _tile_of(I64, *shape)) or ch
        elif op == "from_vec" and len(args) == 3:
            ch = mark(args[0], _vec_of(I64)) or ch
            shape = ctor_shape(args[1], args[2])
            if shape is not None and _is_vec(get(args[0])):
                elem = _vec_elem(get(args[0]))
                if elem == F64:
                    ch = mark(dst, _tile_of(F64, *shape)) or ch
                elif elem == I64 and assume_final:
                    ch = mark(dst, _tile_of(I64, *shape)) or ch
        elif op == "to_vec" and len(args) == 1:
            if _is_tile(get(args[0])):
                elem, _r, _c = _tile_parts(get(args[0]))
                ch = mark(dst, _vec_of(elem)) or ch
        elif op in ("add", "mul") and len(args) == 2:
            ch = unify((dst, args[0], args[1])) or ch
        elif op == "scale" and len(args) == 2:
            ch = unify((dst, args[0])) or ch
            if _is_tile(get(args[0])):
                elem, _r, _c = _tile_parts(get(args[0]))
                ch = mark(args[1], elem) or ch
        elif op == "dot" and len(args) == 2:
            ka, kb = get(args[0]), get(args[1])
            if _is_tile(ka) and _is_tile(kb):
                ea, ra, ca = _tile_parts(ka)
                eb, rb, cb = _tile_parts(kb)
                e = _join(ea, eb)
                if ca != rb or e == CONFLICT:
                    ch = mark(dst, CONFLICT) or ch
                else:
                    ch = mark(dst, _tile_of(e, ra, cb)) or ch
                    ch = mark(args[0], _tile_of(e, ra, ca)) or ch
                    ch = mark(args[1], _tile_of(e, rb, cb)) or ch
        elif op == "sum" and len(args) == 1:
            if _is_tile(get(args[0])):
                elem, _r, _c = _tile_parts(get(args[0]))
                ch = mark(dst, elem) or ch
        elif op == "transpose" and len(args) == 1:
            if _is_tile(get(args[0])):
                elem, r, c = _tile_parts(get(args[0]))
                ch = mark(dst, _tile_of(elem, c, r)) or ch
        elif op == "get" and len(args) == 3:
            if _is_tile(get(args[0])):
                elem, _r, _c = _tile_parts(get(args[0]))
                ch = mark(dst, elem) or ch
        elif op == "load" and len(args) == 4:
            ch = mark(args[0], _vec_of(I64)) or ch
            shape = ctor_shape(args[2], args[3])
            if shape is not None and _is_vec(get(args[0])):
                elem = _vec_elem(get(args[0]))
                if elem == F64:
                    ch = mark(dst, _tile_of(F64, *shape)) or ch
                elif elem == I64 and assume_final:
                    ch = mark(dst, _tile_of(I64, *shape)) or ch
        elif op == "load_or" and len(args) == 5:
            ch = mark(args[0], _vec_of(I64)) or ch
            shape = ctor_shape(args[2], args[3])
            if shape is not None:
                vk = (_vec_elem(get(args[0])) if _is_vec(get(args[0]))
                      else I64)
                e = _join(get(args[4]), vk)
                if e == F64:
                    ch = mark(dst, _tile_of(F64, *shape)) or ch
                    ch = mark(args[4], F64) or ch
                    ch = mark(args[0], _vec_of(F64)) or ch
                elif e == I64 and assume_final:
                    ch = mark(dst, _tile_of(I64, *shape)) or ch
        elif op in ("store", "store_clipped") and len(args) == 3:
            ch = mark(args[0], _vec_of(I64)) or ch
            if _is_tile(get(args[2])):
                elem, _r, _c = _tile_parts(get(args[2]))
                ch = mark(args[0], _vec_of(elem)) or ch
            # dst is unit -> stays i64
        elif op == "load_rows" and len(args) == 6:
            ch = mark(args[0], _vec_of(I64)) or ch
            shape = ctor_shape(args[3], args[4])
            if shape is not None:
                vk = (_vec_elem(get(args[0])) if _is_vec(get(args[0]))
                      else I64)
                e = _join(get(args[5]), vk)
                if e == F64:
                    ch = mark(dst, _tile_of(F64, *shape)) or ch
                    ch = mark(args[5], F64) or ch
                    ch = mark(args[0], _vec_of(F64)) or ch
                elif e == I64 and assume_final:
                    ch = mark(dst, _tile_of(I64, *shape)) or ch
        elif op == "store_rows" and len(args) == 4:
            ch = mark(args[0], _vec_of(I64)) or ch
            if _is_tile(get(args[3])):
                elem, _r, _c = _tile_parts(get(args[3]))
                ch = mark(args[0], _vec_of(elem)) or ch
            # dst is unit -> stays i64
        # rows/cols: i64 dst and i64-bottom receiver need no marks here;
        # the consistency check requires the receiver to be a tile.
        return ch

    def fvec_binop_unify(dst: str, a0: str, a1: str) -> bool:
        """Element-wise vector arithmetic with scalar broadcasting: vector
        operands and the destination share one vector kind; a scalar
        operand unifies with the LEAF element kind (broadcast reaches the
        innermost scalars of a nested vector).  A still-bottom scalar
        operand is left alone until the final phase — it may yet turn out
        to be a vector."""
        ch2 = False
        vecs = [n for n in (a0, a1) if _is_fvec(get(n))]
        if not vecs:
            return False  # dst promoted first: wait for the operands
        vk = get(dst) if _is_fvec(get(dst)) else I64
        for n in vecs:
            vk = _join(vk, get(n))
        if not _is_fvec(vk):  # CONFLICT (or joined into one)
            for n in (dst, *vecs):
                ch2 = mark(n, CONFLICT) or ch2
            return ch2
        leaf, _d = _fvec_leaf(vk)
        scalars = [n for n in (a0, a1) if not _is_fvec(get(n))]
        for n in scalars:
            k = get(n)
            if k != I64 or assume_final:
                leaf = _join(leaf, k)
        if leaf == CONFLICT:
            for n in (dst, *vecs):
                ch2 = mark(n, CONFLICT) or ch2
            return ch2
        vk = _fvec_with_leaf(vk, leaf)
        ch2 = mark(dst, vk) or ch2
        for n in vecs:
            ch2 = mark(n, vk) or ch2
        for n in scalars:
            if get(n) != I64 or assume_final:
                ch2 = mark(n, leaf) or ch2
        return ch2

    fname = info.f.name
    own_sig = sigs.get(fname)
    promoted_set = set(info.promote_params)

    changed = True
    while changed:
        changed = False
        if own_sig is not None and len(own_sig.params) == len(info.params):
            for p, pk in zip(info.params, own_sig.params):
                if p in promoted_set:
                    # promote_matrix rebinds the parameter at entry: the
                    # LOCAL kind is the promoted form of the incoming one
                    # (flat numeric vector -> Mx1 matrix); the sig keeps
                    # the caller-side kind (the driver skips the local ->
                    # sig join for these).
                    changed = mark(p, _promote_kind(pk)) or changed
                else:
                    changed = mark(p, pk) or changed
            for r in info.ret_vars:
                changed = mark(r, own_sig.ret) or changed
        if len(info.ret_vars) > 1:
            changed = unify(info.ret_vars) or changed
        # Module constants: two-way join between the module-wide global
        # cell and this function's local view (reads everywhere; the
        # initializer's declared names are its stores).
        gnames = set(info.global_reads)
        if info.f.name == _MODULE_INIT:
            gnames |= set(info.init_globals)
        for n in sorted(gnames):
            nk = _join(gtable.kind(n), get(n))
            if gtable.mark(n, nk):
                changed = global_changed = True
            changed = mark(n, nk) or changed
        # Lambda captures behave like parameters: two-way join between the
        # creator-side cell and the local uses.
        if info.is_lambda:
            for cap in info.env_captures:
                nk = _join(closures.cell_kind(fname, cap), get(cap))
                if closures.mark_cell(fname, cap, nk):
                    changed = global_changed = True
                changed = mark(cap, nk) or changed
        # Handle-scope subfunctions: env captures unify with the site's
        # cells; body/case returns unify with the site's value cell (they
        # ARE the delimited body's completion value under deep semantics);
        # case params unify with the op's argument cells; __k is a kont.
        if info.is_scope_member:
            site = info.scope_site
            for cap in info.env_captures:
                nk = _join(scopes.cell_kind(site, cap), get(cap))
                if scopes.mark_cell(site, cap, nk):
                    changed = global_changed = True
                changed = mark(cap, nk) or changed
            for r in info.ret_vars:
                nk = _join(scopes.value_kind(site), get(r))
                if scopes.mark_value(site, nk):
                    changed = global_changed = True
                changed = mark(r, nk) or changed
            if info.scope_role == "case":
                opn = scopes.case_op.get(fname)
                for i, p in enumerate(info.params):
                    if p == "__k":
                        changed = mark(p, KONT) or changed
                    elif opn is not None:
                        nk = _join(scopes.op_arg_kind(opn, i), get(p))
                        if scopes.mark_op_arg(opn, i, nk):
                            changed = global_changed = True
                        changed = mark(p, nk) or changed
            elif info.scope_role == "trycatch":
                # The catch parameter is the FAILURE MESSAGE: always a `str`
                # (mx_try hands the catch thunk a `const char *`, the
                # interpreter hands it `InterpError.message`).  Seeding the
                # kind here makes any other use of it a CONFLICT, which
                # demotes — never a silent reinterpretation of the pointer.
                for i, p in enumerate(info.params):
                    if i == 0:
                        changed = mark(p, STR) or changed
        for b in info.f.blocks:
            for op in b.ops:
                if op[0] == "perform" and len(op) >= 7:
                    dfn = info.default_performs.get((op[2], op[3]))
                    if dfn is not None:
                        # Statically resolved to the op's declared default:
                        # an ordinary two-way call-signature join, no
                        # boundary cells (see _analyze).
                        dsig = sigs.get(dfn)
                        if dsig is not None \
                                and len(dsig.params) == len(op[4]):
                            for i, a in enumerate(op[4]):
                                nk = _join(dsig.params[i], get(a))
                                if nk != dsig.params[i]:
                                    dsig.params[i] = nk
                                    changed = global_changed = True
                                changed = mark(a, nk) or changed
                            nk = _join(dsig.ret, get(op[1]))
                            if nk != dsig.ret:
                                dsig.ret = nk
                                changed = global_changed = True
                            changed = mark(op[1], nk) or changed
                        continue
                    # Boundary-crossing values unify through the op-name
                    # cells (routing is dynamic; see _ScopeTable).
                    dst, opn, pargs = op[1], op[3], op[4]
                    # Dynamic default routing: the op's declared default is
                    # reached through the SAME boundary words as a handler
                    # case (a per-op thunk decodes them), so its signature
                    # joins the very same op-name cells — one lattice for
                    # every route the perform can take.  A default whose
                    # types cannot agree with the handler cases therefore
                    # CONFLICTS and demotes, exactly like two irreconcilable
                    # same-named ops.
                    ddfn = info.dynamic_default_performs.get((op[2], opn))
                    ddsig = sigs.get(ddfn) if ddfn is not None else None
                    if ddsig is not None and len(ddsig.params) != len(pargs):
                        ddsig = None
                    for i, a in enumerate(pargs):
                        nk = _join(scopes.op_arg_kind(opn, i), get(a))
                        if ddsig is not None:
                            nk = _join(nk, ddsig.params[i])
                            if nk != ddsig.params[i]:
                                ddsig.params[i] = nk
                                changed = global_changed = True
                        if scopes.mark_op_arg(opn, i, nk):
                            changed = global_changed = True
                        changed = mark(a, nk) or changed
                    nk = _join(scopes.op_result_kind(opn), get(dst))
                    if ddsig is not None:
                        nk = _join(nk, ddsig.ret)
                        if nk != ddsig.ret:
                            ddsig.ret = nk
                            changed = global_changed = True
                    if scopes.mark_op_result(opn, nk):
                        changed = global_changed = True
                    changed = mark(dst, nk) or changed
                    continue
                if op[0] != "let" or len(op) != 4:
                    continue
                _, dst, rhs, args = op
                rk = rhs[0]
                if rk == "const":
                    v = rhs[1]
                    if isinstance(v, bool) or v is None:
                        pass
                    elif isinstance(v, float):
                        changed = mark(dst, F64) or changed
                    elif isinstance(v, str):
                        if dst not in info.tag_consts:
                            changed = mark(dst, STR) or changed
                        # tag literals stay i64: they lower to integer tags
                elif rk == "copy":
                    if dst not in info.dead_results:
                        changed = unify((dst, args[0])) or changed
                elif rk == "binop":
                    o = rhs[1]
                    if o in _CMP_INT:
                        changed = unify(args) or changed  # dst stays i64
                    elif o in _LOGIC or o in _BITWISE_INT:
                        pass  # i64-only
                    elif len(args) == 2 and any(
                            _is_fvec(get(x)) for x in (dst, *args)):
                        # Element-wise vector arithmetic (with broadcast).
                        changed = fvec_binop_unify(dst, args[0], args[1]) \
                            or changed
                    else:
                        changed = unify((dst, *args)) or changed
                elif rk == "select":
                    if len(args) == 3 and dst not in info.dead_results:
                        changed = unify((dst, args[1], args[2])) or changed
                elif rk == "call":
                    callee = rhs[1]
                    bname = _builtin_name(callee, module_names)
                    if callee in info.def_count:
                        # Closure call: types flow through the lambda's sig
                        # once the callee variable's closure kind is known.
                        # A DYNAMIC kind flows through EVERY member lambda's
                        # sig — the shared site transitively unifies the
                        # members' parameter/return kinds with each other.
                        ck = get(callee)
                        for m in _closure_members(ck):
                            sig = sigs.get(m)
                            if sig is not None and len(sig.params) == len(args):
                                for a, pk in zip(args, sig.params):
                                    changed = mark(a, pk) or changed
                                changed = mark(dst, sig.ret) or changed
                    elif callee.startswith(TRAIT_CALL_PREFIX):
                        method = callee[len(TRAIT_CALL_PREFIX):]
                        if args:
                            res, target = _resolve_trait_call(
                                method, get(args[0]), traits, module_names,
                                assume_final=assume_final)
                            if res == "func":
                                sig = sigs.get(target)
                                if sig is not None and len(sig.params) == len(args):
                                    for a, pk in zip(args, sig.params):
                                        changed = mark(a, pk) or changed
                                    changed = mark(dst, sig.ret) or changed
                            elif res == "builtin":
                                changed = apply_builtin(
                                    target, dst, args, plain_call=False) or changed
                    elif callee.startswith(STATIC_CALL_PREFIX):
                        sres, starget = _resolve_static_call(
                            callee, traits, module_names)
                        if sres == "func":
                            sig = sigs.get(starget)
                            if sig is not None and len(sig.params) == len(args):
                                for a, pk in zip(args, sig.params):
                                    changed = mark(a, pk) or changed
                                changed = mark(dst, sig.ret) or changed
                        elif sres == "builtin":
                            changed = apply_builtin(
                                starget, dst, args, plain_call=True) or changed
                    elif bname in _NATIVE_RT_CALLS:
                        # NAME PRECEDENCE: _builtin_name already decided
                        # builtin-vs-module-function for this callee.
                        changed = apply_builtin(
                            bname, dst, args, plain_call=True) or changed
                    elif bname in _EXTERN_C_SIGS:
                        pks, rk_ = _EXTERN_C_SIGS[bname]
                        if len(args) == len(pks):
                            for a, pk in zip(args, pks):
                                changed = mark(a, pk) or changed
                            changed = mark(dst, rk_) or changed
                    elif bname == "as_ptr":
                        # Receiver stays free (str or vec, like len); the
                        # result is always a raw pointer.
                        changed = mark(dst, PTR) or changed
                    elif bname in ("ptr_read", "ptr_write"):
                        if args:
                            changed = mark(args[0], PTR) or changed
                        # offset/value/result are i64 (the default)
                    elif bname == "assert":
                        pass  # cond is i64 (checked); dst is unit -> i64
                    elif bname in _MATH_EXTERNS:
                        for a in args:
                            changed = mark(a, F64) or changed
                        changed = mark(dst, F64) or changed
                    elif bname == "neg":
                        if len(args) == 1:
                            changed = unify((dst, args[0])) or changed
                    elif bname == "bnot":
                        pass  # `~x` is i64-only (the default kind)
                    elif bname in _PRINT_BUILTINS or bname == "not":
                        pass  # dst is unit/bool -> i64
                    else:
                        sig = sigs.get(callee)
                        if sig is not None and len(sig.params) == len(args):
                            for a, pk in zip(args, sig.params):
                                changed = mark(a, pk) or changed
                            changed = mark(dst, sig.ret) or changed
                elif rk == "make_variant":
                    ename, vname = rhs[1], rhs[2]
                    # The dst kind carries this site's refinement: the actual
                    # per-slot representation of the value built here (nested
                    # enum payloads are recorded name-only; their contents go
                    # through the canonical module cells instead).
                    site_ref = {vname: tuple(_strip_refinement(get(a))
                                             for a in args)}
                    changed = mark(
                        dst, _format_enum_kind(ename, site_ref)) or changed
                    # One-way: store kinds also accumulate into the
                    # module-wide per-(variant, slot) cells backing canon();
                    # the driver marks cells `mixed` post-fixpoint when a
                    # store disagrees with the join.
                    for i, a in enumerate(args):
                        if variants.mark_cell(ename, vname, i,
                                              _strip_refinement(get(a))):
                            changed = global_changed = True
                elif rk == "variant_tag":
                    pass  # dst is the integer tag: i64 (the default)
                elif rk == "variant_field":
                    bk = get(args[0])
                    vname = rhs[2] if len(rhs) > 2 else None
                    if _is_enum(bk) and vname is not None:
                        ref = _enum_refinement(bk) or {}
                        slots = ref.get(vname)
                        if slots is not None and rhs[1] < len(slots):
                            sk = slots[rhs[1]]
                            if _is_enum(sk):
                                # Nested enum extraction crosses a boxing
                                # boundary: the boxed value's own refinement
                                # was stripped, so the result assumes the
                                # canonical module-wide representation (the
                                # consistency check demotes if any store
                                # disagrees with it).
                                sk = variants.canon_kind(_enum_name(sk))
                            changed = mark(dst, sk) or changed
                        # variant absent from the refinement: the arm is
                        # dead (no flow constructs it); dst stays at bottom.
                elif rk == "make_closure":
                    changed = mark(dst, _CLOSURE_PREFIX + rhs[1]) or changed
                    for (cn, vn) in args:
                        nk = _join(closures.cell_kind(rhs[1], cn), get(vn))
                        if closures.mark_cell(rhs[1], cn, nk):
                            changed = global_changed = True
                        changed = mark(vn, nk) or changed
                elif rk == "alloc_struct":
                    sname = rhs[1]
                    changed = mark(dst, _STRUCT_PREFIX + sname) or changed
                    # A DECLARED struct's field has one source type, so every
                    # store site agrees and the join flows BOTH ways: the
                    # field learns from the store and the store learns from
                    # the field (that back-propagation is what lets a fresh
                    # `Vec.new()` element pick up the field's refinement).
                    #
                    # A TUPLE struct (hir.TUPLE_STRUCT_PREFIX) has no
                    # declaration: `__tuple2` is every 2-tuple in the module,
                    # so `(1, 2)` and `(1.0, 2.5)` share one layout even
                    # though they are different source types.  Back-
                    # propagating there would silently RETYPE the int literal
                    # as a double (i64 is the lattice bottom, so the join
                    # picks f64) — a wrong-code path, not a demotion.  So the
                    # join is ONE-WAY for tuples, exactly as it is for enum
                    # payload cells, and the driver demotes the struct
                    # post-fixpoint when a store disagrees with the join.
                    two_way = not _is_tuple_struct(sname)
                    for (fn_, fv) in args:
                        fk = structs.field_kind(sname, fn_)
                        nk = _join(fk, get(fv))
                        if structs.mark_field(sname, fn_, nk):
                            changed = global_changed = True
                        if two_way:
                            changed = mark(fv, nk) or changed
                elif rk == "field_get":
                    bk = get(args[0])
                    if bk == I64 and field_owner.get(rhs[1]):
                        changed = mark(
                            args[0],
                            _STRUCT_PREFIX + field_owner[rhs[1]]) or changed
                        bk = get(args[0])
                    if _is_struct(bk):
                        sname = _struct_name(bk)
                        fk = structs.field_kind(sname, rhs[1])
                        nk = _join(fk, get(dst))
                        if structs.mark_field(sname, rhs[1], nk):
                            changed = global_changed = True
                        changed = mark(dst, nk) or changed
                elif rk == "field_set":
                    changed = unify((dst, args[0])) or changed
                    bk = get(args[0])
                    if bk == I64 and field_owner.get(rhs[1]):
                        changed = mark(
                            args[0],
                            _STRUCT_PREFIX + field_owner[rhs[1]]) or changed
                        bk = get(args[0])
                    if _is_struct(bk):
                        sname = _struct_name(bk)
                        fk = structs.field_kind(sname, rhs[1])
                        nk = _join(fk, get(args[1]))
                        if structs.mark_field(sname, rhs[1], nk):
                            changed = global_changed = True
                        changed = mark(args[1], nk) or changed
                elif rk == "resume":
                    # resume(v): v unifies with the handled op's result cell
                    # (it becomes some perform's value); the resume result
                    # unifies with the site's value cell (deep semantics:
                    # it is the whole delimited body's completion value).
                    if len(args) == 2:
                        changed = mark(args[0], KONT) or changed
                        opn = scopes.case_op.get(fname)
                        if opn is not None:
                            nk = _join(scopes.op_result_kind(opn),
                                       get(args[1]))
                            if scopes.mark_op_result(opn, nk):
                                changed = global_changed = True
                            changed = mark(args[1], nk) or changed
                        site = scopes.member_site.get(fname)
                        if site is not None:
                            nk = _join(scopes.value_kind(site), get(dst))
                            if scopes.mark_value(site, nk):
                                changed = global_changed = True
                            changed = mark(dst, nk) or changed
                elif rk == "handle_scope" or rk == "try_scope":
                    # Both sites share one value cell (the delimited result:
                    # handle dst ⊔ body ret ⊔ case ret, try dst ⊔ body ret ⊔
                    # catch ret) and one env-field cell per capture.
                    site = rhs[1]
                    site_rec = scopes.sites.get(site)
                    if site_rec is not None:
                        nk = _join(scopes.value_kind(site), get(dst))
                        if scopes.mark_value(site, nk):
                            changed = global_changed = True
                        changed = mark(dst, nk) or changed
                        for n in scopes.env_fields.get(site, ()):
                            vn = site_rec.cap_vals.get(n, n)
                            nk = _join(scopes.cell_kind(site, n), get(vn))
                            if scopes.mark_cell(site, n, nk):
                                changed = global_changed = True
                            changed = mark(vn, nk) or changed
    return kinds, global_changed


# ---------------------------------------------------------------------------
# Consistency checking (post-fixpoint; anything wrong demotes the function)
# ---------------------------------------------------------------------------

def _check_consistency(info: _Info, kinds: Dict[str, str], sigs: Dict[str, _Sig],
                       structs: _StructTable, variants: _VariantTable,
                       closures: _ClosureTable, traits: _TraitTable,
                       scopes: _ScopeTable, module_names: Set[str],
                       cells: "_CellTable", gtable: "_GlobalTable",
                       word_uniform: Optional[Set[str]] = None,
                       word_blocked: Optional[Dict[str, str]] = None,
                       ) -> List[str]:
    probs: List[str] = []
    word_uniform = word_uniform if word_uniform is not None else set()
    word_blocked = word_blocked if word_blocked is not None else {}

    def ty(n: str) -> str:
        return kinds.get(n, I64)

    def check_boundary(kind: str, what: str) -> None:
        """A value crossing the effect boundary travels as one 8-byte
        word: word kinds directly; struct/enum/closure aggregates as a
        pointer to a fresh write-once BOUNDARY BOX (increment 14 — the
        immortal-box contract, so the word can never dangle across
        coroutine switches).  Still demoted: konts, rawptr, conflicts,
        infinite layouts, and closures of unknown/non-heap-env lambdas
        (the driver marks boundary-crossing members heap-env before this
        runs, so that arm is defensive)."""
        if _is_word_kind(kind):
            return
        if _is_closure(kind):
            check_closure_cell(kind, f"{what} crossing the effect boundary")
            return
        if (_is_struct(kind) or _is_enum(kind)) \
                and _kind_size(kind, structs, variants) is not None:
            return
        probs.append(
            f"{what} of kind {kind} cannot cross the effect boundary "
            "(i64/f64/str/vec word kinds and finite-layout boxed "
            "aggregates only)")

    def check_closure_cell(kind: str, what: str) -> bool:
        """A closure kind stored in an env field (handle-site capture or
        closure-in-closure capture) is a BY-VALUE {fn, env} pair copy —
        legal since increment 13 provided every member lambda is known and
        heap-env (the driver marks env-captured members heap-env before
        checks run, so a live pair can never point into a dead frame).
        Returns True when it handled a closure kind."""
        if not _is_closure(kind):
            return False
        for m in _closure_members(kind):
            if sigs.get(m) is None or m not in module_names:
                probs.append(
                    f"{what} holds a closure of unknown lambda {m!r}")
            elif m not in closures.heap_env:
                # Defensive: the driver's env-capture scan marks these.
                probs.append(
                    f"{what} holds a closure of lambda {m!r} not marked "
                    "heap-env (a stack env could dangle)")
        return True

    def check_env_cell(kind: str, what: str) -> None:
        """A handle-site env field must be storable like a closure capture."""
        if check_closure_cell(kind, what):
            pass
        elif kind == KONT:
            probs.append(f"{what} is an effect continuation (resume must run "
                         "on its scope's owner stack)")
        elif kind == CONFLICT:
            probs.append(f"{what} has conflicting kinds")
        elif _is_agg(kind) and _kind_size(kind, structs, variants) is None:
            probs.append(f"{what} has an infinite layout")

    def check_vec_slot(elem: str, what: str) -> bool:
        """A native Vec element slot is ONE 8-byte word.  Word kinds sit in
        it directly; struct/enum AGGREGATES sit in it as a pointer to an
        immortal write-once ELEMENT BOX (increment 20 — push / set malloc
        a fresh copy of the aggregate and store the pointer; every read
        copies the aggregate back out into the reader's own storage).
        Closures, konts, rawptrs, conflicts and infinite layouts still
        demote honestly.  Returns True when the slot is representable."""
        if _is_word_kind(elem):
            return True
        if _vec_slot_boxable(elem):
            if _kind_size(elem, structs, variants) is None:
                probs.append(
                    f"{what} of {elem} elements: the element box has an "
                    "infinite layout")
                return False
            return True
        probs.append(
            f"{what} of {elem} elements (a Vec slot holds an 8-byte word "
            "kind directly or a struct/enum aggregate as an element-box "
            "pointer; nothing else fits)")
        return False

    def check_builtin(name: str, dst: str, args: Tuple[str, ...]) -> None:
        """Validate a native-runtime builtin call's final kinds."""
        if name == "Vec.new":
            if args:
                probs.append("Vec.new with arguments")
            elif not _is_vec(ty(dst)):
                probs.append(
                    f"Vec.new result {dst!r} has kind {ty(dst)}, not a Vec")
        elif name == "push":
            if len(args) != 2:
                probs.append(f"push with {len(args)} arguments (expects 2)")
            elif not _is_vec(ty(args[0])):
                probs.append(
                    f"push receiver {args[0]!r} has kind {ty(args[0])}, "
                    "not a Vec (fixed vectors are immutable)")
            elif not check_vec_slot(_vec_elem(ty(args[0])), "Vec"):
                pass  # check_vec_slot reported it
            elif ty(args[1]) != _vec_elem(ty(args[0])):
                probs.append(
                    f"push of {ty(args[1])} into a Vec of "
                    f"{_vec_elem(ty(args[0]))}")
        elif name == "pop":
            if len(args) != 1:
                probs.append(f"pop with {len(args)} arguments (expects 1)")
            elif not _is_vec(ty(args[0])):
                probs.append(
                    f"pop receiver {args[0]!r} has kind {ty(args[0])}, "
                    "not a Vec (fixed vectors are immutable)")
            elif not check_vec_slot(_vec_elem(ty(args[0])), "Vec"):
                pass  # check_vec_slot reported it
            elif ty(dst) != _vec_elem(ty(args[0])):
                probs.append(
                    f"pop result {dst!r} is {ty(dst)}, Vec elements are "
                    f"{_vec_elem(ty(args[0]))}")
        elif name == "__index_get":
            rk0 = ty(args[0]) if args else I64
            elem = (_vec_elem(rk0) if _is_vec(rk0)
                    else _fvec_elem(rk0) if _is_fvec(rk0) else None)
            if len(args) != 2:
                probs.append(
                    f"__index_get with {len(args)} arguments (expects 2)")
            elif elem is None:
                probs.append(
                    f"__index_get receiver {args[0]!r} has kind {rk0}, "
                    "not a Vec or fixed vector (string indexing stays "
                    "interpreted)")
            elif ty(args[1]) != I64:
                probs.append(f"__index_get index {args[1]!r} is {ty(args[1])}")
            elif not (check_vec_slot(elem, "Vec") if _is_vec(rk0)
                      else _is_word_kind(elem)):
                # Fixed vectors keep the word-kinds-only rule: an mx_fvec
                # block feeds the SIMD/arith/repr paths, which read the
                # words as numbers — element boxes belong to Vec only.
                if not _is_vec(rk0):
                    probs.append(
                        f"vector of {elem} elements (only 8-byte word kinds "
                        "fit native element slots)")
            elif ty(dst) != elem:
                probs.append(
                    f"__index_get result {dst!r} is {ty(dst)}, elements are "
                    f"{elem}")
        elif name == "__index_store":
            # Store-back index assignment: mx_vec_set in place on a Vec
            # (result = the same pointer), mx_fvec_set_copy functional
            # update on a vector[T,N] (result = a fresh block).  Immutable
            # or unindexable receivers are interpreter errors — demoted
            # here at compile time rather than aborting at runtime.
            rk0 = ty(args[0]) if args else I64
            elem = (_vec_elem(rk0) if _is_vec(rk0)
                    else _fvec_elem(rk0) if _is_fvec(rk0) else None)
            if len(args) != 3:
                probs.append(
                    f"__index_store with {len(args)} arguments (expects 3)")
            elif elem is None:
                probs.append(
                    f"index assignment into a {rk0} receiver (the "
                    "interpreter rejects it: only Vec mutates in place and "
                    "vector[T,N] updates functionally — strings and "
                    "everything else are immutable)")
            elif ty(args[1]) != I64:
                probs.append(
                    f"__index_store index {args[1]!r} is {ty(args[1])}")
            elif not (check_vec_slot(elem, "Vec") if _is_vec(rk0)
                      else _is_word_kind(elem)):
                if not _is_vec(rk0):
                    probs.append(
                        f"vector of {elem} elements (only 8-byte word kinds "
                        "fit native element slots)")
            elif ty(args[2]) != elem:
                probs.append(
                    f"__index_store of {ty(args[2])} into elements of {elem}")
            elif ty(dst) != rk0:
                probs.append(
                    f"__index_store result {dst!r} is {ty(dst)}, receiver "
                    f"is {rk0} (the result IS the updated receiver)")
        elif name == "__index_set":
            # In-place element store: Vec receivers only (identity
            # semantics).  The interpreter rejects stores into immutable
            # receivers loudly; statically-known-immutable receivers
            # demote at compile time with the same message.
            rk0 = ty(args[0]) if args else I64
            if len(args) != 3:
                probs.append(
                    f"__index_set with {len(args)} arguments (expects 3)")
            elif _is_fvec(rk0):
                probs.append(
                    "cannot assign into an immutable vector: vector[T, N] "
                    "values have value semantics (interpreter parity — "
                    "build a new vector, or use Vec for mutable data)")
            elif not _is_vec(rk0):
                probs.append(
                    f"index assignment into a {rk0} receiver (the "
                    "interpreter rejects in-place stores on anything but "
                    "a Vec)")
            elif ty(args[1]) != I64:
                probs.append(
                    f"__index_set index {args[1]!r} is {ty(args[1])}")
            elif not check_vec_slot(_vec_elem(rk0), "Vec"):
                pass  # check_vec_slot reported it
            elif ty(args[2]) != _vec_elem(rk0):
                probs.append(
                    f"__index_set of {ty(args[2])} into a Vec of "
                    f"{_vec_elem(rk0)}")
            elif ty(dst) != I64:
                probs.append(
                    f"__index_set result {dst!r} promoted to {ty(dst)} "
                    "(the store returns unit)")
        elif name == "__zip":
            # Pair-lockstep iterable for zip comprehensions.  The result is
            # virtual (mx_fvec_zip_map reads the sources directly); a
            # separate use-restriction scan below demotes any use outside
            # a comprehension iterable position.
            if len(args) != 2:
                probs.append(
                    f"__zip of {len(args)} sequences (only pair iteration "
                    "lowers natively)")
            else:
                for a in args:
                    ak = ty(a)
                    if not _is_fvec(ak):
                        probs.append(
                            f"__zip operand {a!r} has kind {ak} (only fixed "
                            "vectors and ranges zip natively; Vec iteration "
                            "stays interpreted)")
                    elif not _is_word_kind(_fvec_elem(ak)):
                        probs.append(
                            f"vector of {_fvec_elem(ak)} elements (only "
                            "8-byte word kinds fit native element slots)")
                if ty(dst) != I64:
                    probs.append(
                        f"__zip result {dst!r} promoted to {ty(dst)} (zip "
                        "results only feed comprehension iterables)")
        elif name == "__vec_lit":
            if not args:
                probs.append("__vec_lit with no size argument")
            elif not _is_fvec(ty(dst)):
                probs.append(
                    f"__vec_lit result {dst!r} has kind {ty(dst)}, not a "
                    "fixed vector")
            elif ty(args[0]) != I64:
                probs.append(f"__vec_lit size {args[0]!r} is {ty(args[0])}")
            elif not _is_word_kind(_fvec_elem(ty(dst))):
                probs.append(
                    f"vector of {_fvec_elem(ty(dst))} elements (only 8-byte "
                    "word kinds fit native element slots)")
            else:
                for e in args[1:]:
                    if ty(e) != _fvec_elem(ty(dst)):
                        probs.append(
                            f"__vec_lit element {e!r} is {ty(e)} in a vector "
                            f"of {_fvec_elem(ty(dst))}")
        elif name == "__vec_zeros":
            base = info.const_strs.get(args[1], "").lower() \
                if len(args) == 2 else ""
            zk = {"float": F64, "int": I64}.get(base)
            if len(args) != 2:
                probs.append(
                    f"__vec_zeros with {len(args)} arguments (expects 2)")
            elif ty(args[0]) != I64:
                probs.append(f"__vec_zeros size {args[0]!r} is {ty(args[0])}")
            elif zk is None:
                probs.append(
                    "__vec_zeros base type is not a constant 'float'/'int' "
                    "name (the interpreter cannot zero-initialize it either)")
            elif ty(dst) != _fvec_of(zk):
                probs.append(
                    f"__vec_zeros result {dst!r} is {ty(dst)}, expected "
                    f"{_fvec_of(zk)}")
        elif name == "__vec_filled":
            if len(args) != 2:
                probs.append(
                    f"__vec_filled with {len(args)} arguments (expects 2)")
            elif ty(args[0]) != I64:
                probs.append(f"__vec_filled size {args[0]!r} is {ty(args[0])}")
            elif not _is_fvec(ty(dst)):
                probs.append(
                    f"__vec_filled result {dst!r} has kind {ty(dst)}, not a "
                    "fixed vector")
            elif not _is_word_kind(_fvec_elem(ty(dst))):
                probs.append(
                    f"vector of {_fvec_elem(ty(dst))} elements (only 8-byte "
                    "word kinds fit native element slots)")
            elif ty(args[1]) != _fvec_elem(ty(dst)):
                probs.append(
                    f"__vec_filled value {args[1]!r} is {ty(args[1])} in a "
                    f"vector of {_fvec_elem(ty(dst))}")
        elif name == "__range":
            if len(args) != 2:
                probs.append(f"__range with {len(args)} arguments (expects 2)")
            elif any(ty(a) != I64 for a in args):
                probs.append("__range bounds must be i64")
            elif ty(dst) != _fvec_of(I64):
                probs.append(
                    f"__range result {dst!r} promoted to {ty(dst)} (ranges "
                    "are int vectors)")
        elif name == "__vec_dim":
            dim = info.const_ints.get(args[1]) if len(args) == 2 else None
            rk0 = ty(args[0]) if args else I64
            if len(args) != 2:
                probs.append(
                    f"__vec_dim with {len(args)} arguments (expects 2)")
            elif not _is_fvec(rk0):
                probs.append(
                    f"__vec_dim receiver {args[0]!r} has kind {rk0} (only "
                    "fixed vectors lower natively)")
            elif dim not in (0, 1):
                probs.append(
                    "__vec_dim dimension is not the constant 0 or 1")
            elif dim == 1 and not (_is_fvec(_fvec_elem(rk0))
                                   or _fvec_elem(rk0) in (I64, F64)):
                probs.append(
                    f"__vec_dim 1 of a vector of {_fvec_elem(rk0)} elements "
                    "(no second dimension; the interpreter rejects it too)")
            elif ty(dst) != I64:
                probs.append(f"__vec_dim result {dst!r} promoted to {ty(dst)}")
        elif name == "__slice_get":
            rk0 = ty(args[0]) if args else I64
            if len(args) != 4:
                probs.append(
                    f"__slice_get with {len(args)} arguments (expects 4)")
            elif not _is_fvec(rk0):
                probs.append(
                    f"__slice_get receiver {args[0]!r} has kind {rk0} "
                    "(Vec/string slicing stays interpreted)")
            elif not _is_word_kind(_fvec_elem(rk0)):
                probs.append(
                    f"vector of {_fvec_elem(rk0)} elements (only 8-byte "
                    "word kinds fit native element slots)")
            elif ty(dst) != rk0:
                probs.append(
                    f"__slice_get result {dst!r} is {ty(dst)}, receiver is "
                    f"{rk0}")
            else:
                for a in args[1:]:
                    if a in info.const_nones:
                        continue  # statically-omitted bound
                    if a in info.none_def_vars:
                        probs.append(
                            f"__slice_get bound {a!r} is sometimes None and "
                            "sometimes an int (no static encoding)")
                    elif ty(a) != I64:
                        probs.append(
                            f"__slice_get bound {a!r} is {ty(a)}, not i64")
        elif name == "__cast":
            t = info.const_strs.get(args[1]) if len(args) == 2 else None
            if len(args) != 2:
                probs.append(f"__cast with {len(args)} arguments (expects 2)")
            elif t is None:
                probs.append("__cast target is not a constant type name")
            elif t in ("float", "f32", "f64"):
                if ty(args[0]) not in (I64, F64):
                    probs.append(
                        f"__cast of {ty(args[0])} to {t} (the interpreter "
                        "only converts numbers)")
                elif ty(dst) != F64:
                    probs.append(f"__cast result {dst!r} is {ty(dst)}, not f64")
            elif t in ("int", "i8", "i16", "i32", "i64",
                       "u8", "u16", "u32", "u64"):
                if ty(args[0]) not in (I64, F64):
                    probs.append(
                        f"__cast of {ty(args[0])} to {t} (the interpreter "
                        "only converts numbers)")
                elif ty(dst) != I64:
                    probs.append(f"__cast result {dst!r} is {ty(dst)}, not i64")
            else:
                if ty(dst) != ty(args[0]):
                    probs.append(
                        f"__cast to {t!r} is an identity reinterpretation "
                        f"but {dst!r} is {ty(dst)} and {args[0]!r} is "
                        f"{ty(args[0])}")
        elif name == "__vec_comprehension":
            if len(args) != 3:
                probs.append(
                    f"__vec_comprehension with {len(args)} arguments "
                    "(expects 3)")
                return
            nvar, fnvar, itvar = args
            if nvar not in info.const_nones and nvar not in info.const_ints:
                probs.append(
                    "__vec_comprehension size is not a constant int or None")
            fk = ty(fnvar)
            if not _is_closure(fk):
                probs.append(
                    f"__vec_comprehension body {fnvar!r} is not a "
                    "statically-known closure")
                return
            if _is_dyn_closure(fk):
                probs.append(
                    f"__vec_comprehension body {fnvar!r} is a dynamic "
                    f"closure ({fk}); the per-site thunk needs one "
                    "statically-known lambda")
                return
            lname = _closure_lambda(fk)
            lsig = sigs.get(lname)
            if lsig is None or lname not in module_names:
                probs.append(
                    f"__vec_comprehension body lambda {lname!r} unknown")
                return
            zsrcs = info.zip_defs.get(itvar)
            if zsrcs is not None:
                # Zip comprehension: lockstep over the zip SOURCES via
                # mx_fvec_zip_map (the body lambda unpacks one parameter
                # per zipped sequence).
                if len(lsig.params) != len(zsrcs):
                    probs.append(
                        f"zip comprehension body lambda {lname!r} takes "
                        f"{len(lsig.params)} parameters for {len(zsrcs)} "
                        "zipped sequences")
                else:
                    for i, s in enumerate(zsrcs):
                        sk_ = ty(s)
                        if not _is_fvec(sk_):
                            continue  # the __zip check reported it already
                        ek_, pk2 = _fvec_elem(sk_), lsig.params[i]
                        if not _is_word_kind(ek_):
                            probs.append(
                                f"vector of {ek_} elements (only 8-byte "
                                "word kinds fit native element slots)")
                        elif not (pk2 == ek_ or (ek_ == I64 and pk2 == F64)):
                            probs.append(
                                f"zip comprehension element kind {ek_} does "
                                f"not fit the body lambda's parameter "
                                f"{i} kind {pk2}")
            else:
                if len(lsig.params) != 1:
                    probs.append(
                        f"__vec_comprehension body lambda {lname!r} takes "
                        f"{len(lsig.params)} parameters (tuple unpacking "
                        "stays interpreted)")
                    return
                itk = ty(itvar)
                if not _is_fvec(itk):
                    probs.append(
                        f"__vec_comprehension iterable {itvar!r} has kind "
                        f"{itk} (only fixed vectors and ranges lower "
                        "natively)")
                    return
                ek, pk_ = _fvec_elem(itk), lsig.params[0]
                if not _is_word_kind(ek):
                    probs.append(
                        f"vector of {ek} elements (only 8-byte word kinds "
                        "fit native element slots)")
                elif not (pk_ == ek or (ek == I64 and pk_ == F64)):
                    probs.append(
                        f"__vec_comprehension element kind {ek} does not "
                        f"fit the body lambda's parameter kind {pk_}")
            dk = ty(dst)
            if not _is_fvec(dk):
                probs.append(
                    f"__vec_comprehension result {dst!r} has kind {dk}, "
                    "not a fixed vector")
            else:
                rk_, dek = lsig.ret, _fvec_elem(dk)
                if not _is_word_kind(rk_):
                    probs.append(
                        f"__vec_comprehension body returns {rk_} (only "
                        "8-byte word kinds fit native element slots)")
                elif not (dek == rk_ or (rk_ == I64 and dek == F64)):
                    probs.append(
                        f"__vec_comprehension body returns {rk_} into a "
                        f"vector of {dek}")
        elif name == "len":
            if len(args) != 1:
                probs.append(f"len with {len(args)} arguments (expects 1)")
            elif not (_is_vec(ty(args[0])) or _is_fvec(ty(args[0]))
                      or ty(args[0]) == STR):
                probs.append(
                    f"len receiver {args[0]!r} has kind {ty(args[0])} "
                    "(only Vec, fixed vector and string lower natively)")
            elif ty(dst) != I64:
                probs.append(f"len result {dst!r} promoted to {ty(dst)}")
        elif name in ("to_string", "int_to_str"):
            ak = ty(args[0]) if len(args) == 1 else I64
            if len(args) != 1:
                probs.append(f"{name} with {len(args)} arguments (expects 1)")
            elif _is_fvec(ak):
                leaf, _d = _fvec_leaf(ak)
                if leaf not in (I64, F64):
                    probs.append(
                        f"{name} of a vector of {leaf} leaves (only "
                        "int/float vectors render natively)")
                elif ty(dst) != STR:
                    probs.append(
                        f"{name} result {dst!r} is {ty(dst)}, not str")
            elif _is_tile(ak):
                if _tile_parts(ak)[0] not in (I64, F64):
                    probs.append(
                        f"{name} of a tile of {_tile_parts(ak)[0]} elements")
                elif ty(dst) != STR:
                    probs.append(
                        f"{name} result {dst!r} is {ty(dst)}, not str")
            elif ak not in (I64, F64, STR):
                probs.append(
                    f"{name} of kind {ak} (only i64/f64/str/fixed-vector/"
                    "tile lower to mx_i64_to_str/mx_f64_to_str/identity/"
                    "mx_fvec_to_str/mx_tile_to_str)")
            elif ty(dst) != STR:
                probs.append(f"{name} result {dst!r} is {ty(dst)}, not str")
        elif name.startswith("Tile."):
            # Consistency of a Tile dotted-static call's FINAL kinds
            # (docs/gpu_tiles.md).  Anything off is a demotion reason,
            # not a user error: the tile shape checker already rejected
            # the statically visible misuse at compile time, so what
            # reaches here is a shape or element kind the module could
            # not resolve to constants.
            top = name[len("Tile."):]
            want = TILE_ARITY.get(top)
            if want is None:
                probs.append(f"unknown Tile builtin {name!r}")
                return
            if len(args) != want:
                probs.append(f"{name} with {len(args)} arguments "
                             f"(expects {want})")
                return
            if top in ("zeros", "filled", "arange", "from_vec", "load",
                       "load_or", "load_rows"):
                if not _is_tile(ty(dst)):
                    probs.append(
                        f"{name} result {dst!r} has kind {ty(dst)}: the "
                        "tile shape or element kind is not statically "
                        "resolvable (rows/cols must be positive integer "
                        "literals; elements int or float)")
                    return
                elem, _r, _c = _tile_parts(ty(dst))
                if elem not in (I64, F64):
                    probs.append(f"{name} of {elem} elements "
                                 "(tiles hold int or float)")
                elif top in ("from_vec", "load", "load_or", "load_rows"):
                    vk = ty(args[0])
                    fill = {"load_or": 4, "load_rows": 5}.get(top)
                    if not _is_vec(vk):
                        probs.append(f"{name} source {args[0]!r} has kind "
                                     f"{vk}, not a Vec")
                    elif _vec_elem(vk) != elem:
                        probs.append(f"{name} of a Vec of {_vec_elem(vk)} "
                                     f"into a tile of {elem}")
                    elif fill is not None and ty(args[fill]) != elem:
                        probs.append(f"{name} fill value {args[fill]!r} is "
                                     f"{ty(args[fill])}, elements are "
                                     f"{elem}")
                elif top == "filled" and ty(args[2]) != elem:
                    probs.append(f"{name} fill value {args[2]!r} is "
                                 f"{ty(args[2])}, elements are {elem}")
            elif top in ("store", "store_clipped", "store_rows"):
                tk = ty(args[3 if top == "store_rows" else 2])
                if not _is_tile(tk):
                    probs.append(f"{name} value {args[2]!r} has kind {tk}, "
                                 "not a Tile (its shape must be statically "
                                 "known)")
                    return
                elem, _r, _c = _tile_parts(tk)
                vk = ty(args[0])
                if elem not in (I64, F64):
                    probs.append(f"{name} of {elem} elements "
                                 "(tiles hold int or float)")
                elif not _is_vec(vk):
                    probs.append(f"{name} target {args[0]!r} has kind {vk}, "
                                 "not a Vec")
                elif _vec_elem(vk) != elem:
                    probs.append(f"{name} of a tile of {elem} into a Vec "
                                 f"of {_vec_elem(vk)}")
                elif ty(args[1]) != I64:
                    probs.append(f"{name} offset {args[1]!r} is "
                                 f"{ty(args[1])}, not an int")
            elif top in ("add", "mul", "scale", "dot", "sum", "transpose",
                         "get", "to_vec", "rows", "cols"):
                ka = ty(args[0])
                if not _is_tile(ka):
                    probs.append(f"{name} receiver {args[0]!r} has kind "
                                 f"{ka}, not a Tile (its shape must be "
                                 "statically known)")
                    return
                elem, r, c = _tile_parts(ka)
                if elem not in (I64, F64):
                    probs.append(f"{name} of {elem} elements "
                                 "(tiles hold int or float)")
                    return
                if top in ("add", "mul"):
                    if ty(args[1]) != ka or ty(dst) != ka:
                        probs.append(
                            f"{name} operand/result kinds disagree "
                            f"({ka}, {ty(args[1])} -> {ty(dst)})")
                elif top == "scale":
                    if ty(args[1]) != elem:
                        probs.append(f"{name} scalar {args[1]!r} is "
                                     f"{ty(args[1])}, elements are {elem}")
                    elif ty(dst) != ka:
                        probs.append(f"{name} result {dst!r} is {ty(dst)}, "
                                     f"receiver is {ka}")
                elif top == "dot":
                    kb = ty(args[1])
                    if not _is_tile(kb):
                        probs.append(f"{name} operand {args[1]!r} has kind "
                                     f"{kb}, not a Tile")
                        return
                    eb, rb, cb = _tile_parts(kb)
                    if eb != elem or c != rb:
                        probs.append(f"{name} operands disagree "
                                     f"({ka} · {kb})")
                    elif ty(dst) != _tile_of(elem, r, cb):
                        probs.append(f"{name} result {dst!r} is {ty(dst)}, "
                                     f"expected {_tile_of(elem, r, cb)}")
                elif top == "sum":
                    if ty(dst) != elem:
                        probs.append(f"{name} result {dst!r} is {ty(dst)}, "
                                     f"elements are {elem}")
                elif top == "transpose":
                    if ty(dst) != _tile_of(elem, c, r):
                        probs.append(f"{name} result {dst!r} is {ty(dst)}, "
                                     f"expected {_tile_of(elem, c, r)}")
                elif top == "get":
                    if ty(args[1]) != I64 or ty(args[2]) != I64:
                        probs.append(f"{name} indices must be ints "
                                     f"({ty(args[1])}, {ty(args[2])})")
                    elif ty(dst) != elem:
                        probs.append(f"{name} result {dst!r} is {ty(dst)}, "
                                     f"elements are {elem}")
                elif top == "to_vec":
                    if ty(dst) != _vec_of(elem):
                        probs.append(f"{name} result {dst!r} is {ty(dst)}, "
                                     f"expected {_vec_of(elem)}")
                else:  # rows / cols
                    if ty(dst) != I64:
                        probs.append(f"{name} result {dst!r} promoted to "
                                     f"{ty(dst)}")
        elif name in _MATH_EXTERNS:
            pass  # kinds pinned to f64 during inference

    for name in sorted(set(info.def_count) | set(info.use_blocks)):
        if ty(name) == CONFLICT:
            probs.append(f"irreconcilable value kinds for {name!r}")

    # Thread/Mutex runtime primitives (__mx_effect_runtime$SYMBOL inside
    # the __effect_runtime$E$op thunks — docs/threads_runtime.md).  Handle
    # words are opaque i64; EFFECT_SPAWN's closure argument must be
    # invocable from the C child thread: every member lambda known,
    # zero-argument, heap-env (the env outlives the spawning frame) and on
    # the word-uniform ABI (`i64 (ptr env)` is exactly the C entry's
    # `int64_t (*)(void *)`).  EFFECT_JOIN's result is the child's result
    # word: any word kind decodes exactly; aggregates would need a copy
    # out of the child's boundary box and demote for now.
    for (pdst, symbol, pargs) in info.effect_primitive_calls:
        if symbol == "EFFECT_SPAWN":
            fk = ty(pargs[0])
            if not _is_closure(fk):
                probs.append(
                    f"EFFECT_SPAWN argument {pargs[0]!r} has kind {fk}, "
                    "not a statically-known closure")
            else:
                for m in _closure_members(fk):
                    if sigs.get(m) is None or m not in module_names:
                        probs.append(
                            f"EFFECT_SPAWN of unknown lambda {m!r}")
                        continue
                    if len(sigs[m].params) != 0:
                        probs.append(
                            f"EFFECT_SPAWN lambda {m!r} declares "
                            f"{len(sigs[m].params)} parameter(s); a spawned "
                            "closure takes none")
                    if m not in word_uniform:
                        why = word_blocked.get(m)
                        probs.append(
                            f"EFFECT_SPAWN lambda {m!r} cannot take the "
                            "word-uniform ABI the child thread invokes"
                            + (f": {why}" if why else
                               " (aggregate signature or non-participant)"))
                    elif m not in closures.heap_env:
                        # Defensive: the driver marks spawn-reaching
                        # members heap-env before checks run.
                        probs.append(
                            f"EFFECT_SPAWN lambda {m!r} not marked heap-env "
                            "(a stack env would dangle on the child thread)")
            if ty(pdst) != I64:
                probs.append(
                    f"EFFECT_SPAWN result {pdst!r} promoted to {ty(pdst)} "
                    "(thread handles are opaque i64 words)")
        elif symbol == "EFFECT_JOIN":
            if ty(pargs[0]) != I64:
                probs.append(
                    f"EFFECT_JOIN argument {pargs[0]!r} has kind "
                    f"{ty(pargs[0])} (thread handles are opaque i64 words)")
            if not _is_word_kind(ty(pdst)):
                probs.append(
                    f"EFFECT_JOIN result {pdst!r} has kind {ty(pdst)}, "
                    "which does not decode from the child's result word "
                    "(word kinds only; aggregates demote)")
        else:  # EFFECT_MUTEX_CREATE / _LOCK / _UNLOCK
            for a in pargs:
                if ty(a) != I64:
                    probs.append(
                        f"{symbol} argument {a!r} has kind {ty(a)} "
                        "(mutex handles are opaque i64 words)")
            if ty(pdst) != I64:
                probs.append(
                    f"{symbol} result {pdst!r} promoted to {ty(pdst)} "
                    "(opaque i64 handle / unit word)")

    # Struct-kinded params and returns are supported: params pass as ptr with
    # a callee byval-copy, returns are sret-style (see module docstring).

    for b in info.f.blocks:
        for op in b.ops:
            if op[0] == "perform" and len(op) >= 7:
                dfn = info.default_performs.get((op[2], op[3]))
                if dfn is not None:
                    # Direct call to the op's declared default: ordinary
                    # call-signature checks, no boundary word restrictions.
                    dsig = sigs.get(dfn)
                    if dsig is None or dfn not in module_names:
                        probs.append(
                            f"effect-op default {dfn!r} missing from the "
                            "module")
                    elif len(dsig.params) != len(op[4]):
                        probs.append(
                            f"perform of {op[3]!r} with {len(op[4])} "
                            f"arguments; its default declares "
                            f"{len(dsig.params)} parameters")
                    else:
                        for a, pk_ in zip(op[4], dsig.params):
                            if ty(a) != pk_:
                                probs.append(
                                    f"perform default call {dfn!r}: arg "
                                    f"{a!r} is {ty(a)}, expects {pk_}")
                        if ty(op[1]) != dsig.ret:
                            probs.append(
                                f"perform default call {dfn!r}: result "
                                f"{op[1]!r} is {ty(op[1])}, returns "
                                f"{dsig.ret}")
                    continue
                for a in op[4]:
                    check_boundary(ty(a), f"perform argument {a!r}")
                check_boundary(ty(op[1]), f"perform result {op[1]!r}")
                ddfn = info.dynamic_default_performs.get((op[2], op[3]))
                if ddfn is not None:
                    # Dynamic default routing: the per-op thunk decodes the
                    # SAME boundary words the dispatcher would, so the
                    # default's signature must have converged onto the op
                    # cells.  Anything that did not converge (an arity
                    # mismatch stops the join outright) demotes here rather
                    # than emitting a thunk that decodes words wrongly.
                    ddsig = sigs.get(ddfn)
                    if ddsig is None or ddfn not in module_names:
                        probs.append(
                            f"effect-op default {ddfn!r} missing from the "
                            "module")
                    elif len(ddsig.params) != len(op[4]):
                        probs.append(
                            f"perform of {op[3]!r} with {len(op[4])} "
                            f"arguments; its default declares "
                            f"{len(ddsig.params)} parameters (dynamic "
                            "default routing needs an exact arity match: "
                            "only the SCOPE path pads with UNIT)")
                    else:
                        for a, pk_ in zip(op[4], ddsig.params):
                            if ty(a) != pk_:
                                probs.append(
                                    f"dynamic default {ddfn!r}: argument "
                                    f"{a!r} is {ty(a)} but the default's "
                                    f"parameter is {pk_} (the default and "
                                    "the handler cases must agree on the "
                                    "boundary kinds)")
                        if ty(op[1]) != ddsig.ret:
                            probs.append(
                                f"dynamic default {ddfn!r}: result "
                                f"{op[1]!r} is {ty(op[1])} but the default "
                                f"returns {ddsig.ret}")
                continue
            if op[0] != "let" or len(op) != 4:
                continue
            _, dst, rhs, args = op
            rk = rhs[0]
            if rk == "resume":
                if len(args) == 2:
                    if ty(args[0]) != KONT:
                        probs.append(
                            f"resume continuation {args[0]!r} has kind "
                            f"{ty(args[0])}, not kont")
                    check_boundary(ty(args[1]), f"resume value {args[1]!r}")
                    check_boundary(ty(dst), f"resume result {dst!r}")
                continue
            if rk == "handle_scope":
                site = rhs[1]
                check_boundary(ty(dst), f"handle value {dst!r}")
                vk = scopes.value_kind(site)
                if vk == CONFLICT:
                    probs.append(
                        f"handle site {site!r} has conflicting value kinds")
                for n in scopes.env_fields.get(site, ()):
                    check_env_cell(scopes.cell_kind(site, n),
                                   f"handle-site capture {n!r}")
                for (opn, cparams, _hfn) in (
                        scopes.sites[site].cases
                        if site in scopes.sites else ()):
                    if scopes.op_result_kind(opn) == CONFLICT:
                        probs.append(
                            f"effect op {opn!r} has conflicting result kinds "
                            "across its performs/handlers")
                    for i in range(len(cparams)):
                        if scopes.op_arg_kind(opn, i) == CONFLICT:
                            probs.append(
                                f"effect op {opn!r} argument {i} has "
                                "conflicting kinds across its "
                                "performs/handlers")
                continue
            if rk in ("const", "const_ty"):
                if _is_agg(ty(dst)) and dst not in info.dead_results:
                    probs.append(
                        f"constant {dst!r} promoted to aggregate kind "
                        f"{ty(dst)} (no scalar-to-aggregate coercion)")
                elif (_is_vec(ty(dst)) or _is_fvec(ty(dst))) \
                        and dst not in info.dead_results \
                        and not (rk == "const" and rhs[1] is None) \
                        and rk != "const_ty":
                    probs.append(
                        f"constant {dst!r} promoted to vector kind {ty(dst)} "
                        "(no scalar-to-vector coercion)")
                elif ty(dst) == PTR and dst not in info.dead_results \
                        and not (rk == "const" and rhs[1] is None):
                    probs.append(
                        f"constant {dst!r} promoted to rawptr kind "
                        "(only `null` is a pointer literal)")
            elif rk == "binop":
                o = rhs[1]
                if any(_is_fvec(ty(x)) for x in (dst, *args)):
                    # Element-wise arithmetic (with scalar broadcasting) on
                    # fixed vectors -> mx_fvec_binop.  Comparisons are
                    # STRUCTURAL in the interpreter (dataclass equality);
                    # native pointers cannot reproduce that -> demote.
                    dk = ty(dst)
                    if o not in _FVEC_BINOP_CODES:
                        probs.append(
                            f"binop {o!r} on fixed vectors (the interpreter "
                            "compares structurally; only element-wise "
                            "+ - * / % lower natively)")
                    elif len(args) != 2 or not _is_fvec(dk):
                        probs.append(
                            f"vector binop {o!r} result {dst!r} has kind "
                            f"{dk}, not a fixed vector")
                    elif _fvec_leaf(dk)[0] not in (I64, F64):
                        probs.append(
                            f"vector binop {o!r} over {_fvec_leaf(dk)[0]} "
                            "leaves (only numeric element-wise arithmetic)")
                    elif o == "%" and _fvec_leaf(dk)[0] == F64:
                        probs.append(
                            "float vector % has no native lowering (the "
                            "runtime object stays libm-free)")
                    elif not any(_is_fvec(ty(a)) for a in args):
                        probs.append(
                            f"vector binop {o!r} with no vector operand")
                    else:
                        for a in args:
                            ak = ty(a)
                            if _is_fvec(ak):
                                if ak != dk:
                                    probs.append(
                                        f"vector binop {o!r}: operand "
                                        f"{a!r} is {ak}, result is {dk}")
                            elif ak != _fvec_leaf(dk)[0]:
                                probs.append(
                                    f"vector binop {o!r}: broadcast scalar "
                                    f"{a!r} is {ak}, leaf elements are "
                                    f"{_fvec_leaf(dk)[0]}")
                elif any(_is_vec(ty(x)) for x in (dst, *args)):
                    # Vec ==/!= is item-wise in the interpreter but would be
                    # pointer identity natively; no vec arithmetic exists.
                    probs.append(
                        f"binop {o!r} on Vec values (interpreter compares "
                        "contents; native pointers cannot)")
                elif o in _CMP_INT:
                    if ty(dst) not in (I64,):
                        probs.append(f"comparison result {dst!r} promoted to {ty(dst)}")
                    elif ty(args[0]) == STR and ty(args[1]) == STR:
                        # ==/!= on strings -> mx_str_eq (content equality,
                        # exactly the interpreter's).  Ordering comparisons
                        # on strings stay demoted.
                        if o not in ("==", "!="):
                            probs.append(
                                f"string ordering comparison {o!r} (only "
                                "==/!= lower to mx_str_eq)")
                    elif ty(args[0]) == PTR and ty(args[1]) == PTR:
                        # ==/!= on raw pointers -> ptr icmp (identity,
                        # exactly the interpreter's structural MxPtr/None
                        # comparison).  Pointer ordering stays demoted.
                        if o not in ("==", "!="):
                            probs.append(
                                f"pointer ordering comparison {o!r} (only "
                                "==/!= lower to ptr icmp)")
                    elif ty(args[0]) not in (I64, F64) or ty(args[1]) not in (I64, F64):
                        probs.append(f"comparison {o!r} on non-numeric operands")
                elif o in _LOGIC:
                    if any(ty(x) != I64 for x in (dst, *args)):
                        probs.append(f"logical binop {o!r} on non-i64 values")
                elif o in _BITWISE_INT:
                    if any(ty(x) != I64 for x in (dst, *args)):
                        probs.append(
                            f"bitwise binop {o!r} on non-Int values (the "
                            "interpreter refuses too)")
                else:
                    if ty(dst) == STR:
                        # str + str -> mx_str_concat (fresh malloc'd string,
                        # leaked by design like boxes/heap envs).
                        if o != "+" or ty(args[0]) != STR or ty(args[1]) != STR:
                            probs.append(
                                f"string arithmetic {o!r} (only + on two "
                                "strings lowers to mx_str_concat)")
                    elif ty(dst) not in (I64, F64):
                        probs.append(f"arithmetic {o!r} on non-numeric kind {ty(dst)}")
            elif rk == "select" and len(args) == 3:
                if dst in info.dead_results:
                    continue  # emitted as a comment, never observed
                if ty(args[0]) != I64:
                    probs.append(f"select condition {args[0]!r} is {ty(args[0])}")
                if _is_agg(ty(dst)):
                    probs.append("select over aggregate values")
            elif rk == "call":
                callee = rhs[1]
                bname = _builtin_name(callee, module_names)
                if callee in info.def_count:
                    # Closure call: every lambda the callee variable can
                    # name must be known and signature-compatible.  A
                    # single pinned word-eligible participant or a dynamic
                    # member set goes through the word-uniform indirect
                    # ABI; a pinned non-participant stays a typed call.
                    ck = ty(callee)
                    if not _is_closure(ck):
                        probs.append(
                            f"call through local {callee!r} that is not a "
                            "statically-known closure")
                        continue
                    members = _closure_members(ck)
                    unknown = [m for m in members
                               if sigs.get(m) is None or m not in module_names]
                    if unknown:
                        probs.append(
                            f"closure call to unknown lambda {unknown[0]!r}")
                        continue
                    bad_arity = [m for m in members
                                 if len(sigs[m].params) != len(args)]
                    if bad_arity:
                        m = bad_arity[0]
                        probs.append(
                            f"closure call through {callee!r} reaches lambda "
                            f"{m!r} declaring {len(sigs[m].params)} "
                            f"parameter(s) for {len(args)} argument(s) (the "
                            "interpreter's call would leave parameters "
                            "unbound too)")
                        continue
                    is_word = _is_dyn_closure(ck) or any(
                        m in word_uniform for m in members)
                    if is_word:
                        for m in members:
                            sigm = sigs[m]
                            bad = [k for k in [*sigm.params, sigm.ret]
                                   if not _word_abi_ok(k)]
                            if bad:
                                probs.append(
                                    f"indirect closure call through "
                                    f"{callee!r}: lambda {m!r} has "
                                    f"signature kind {bad[0]}, which has no "
                                    "word encoding and no boundary box "
                                    "(closure pairs, konts and conflicting "
                                    "kinds stay demoted in indirect "
                                    "args/returns)")
                            elif m not in word_uniform:
                                probs.append(
                                    f"indirect closure call through "
                                    f"{callee!r}: lambda {m!r} is not "
                                    "word-uniform ("
                                    + word_blocked.get(
                                        m, "participation analysis missed a "
                                           "flow")
                                    + ")")
                        for a in args:
                            if not _word_abi_ok(ty(a)):
                                probs.append(
                                    f"indirect closure call through "
                                    f"{callee!r}: arg {a!r} of kind {ty(a)} "
                                    "has no word encoding and no boundary "
                                    "box (closure pairs, konts and "
                                    "conflicting kinds stay demoted in "
                                    "indirect args)")
                            elif (_word_boxable(ty(a))
                                  and _kind_size(ty(a), structs,
                                                 variants) is None):
                                probs.append(
                                    f"indirect closure call through "
                                    f"{callee!r}: arg {a!r} of kind {ty(a)} "
                                    "has an infinite layout, so no boundary "
                                    "box can be sized")
                        if not _word_abi_ok(ty(dst)):
                            probs.append(
                                f"indirect closure call through {callee!r}: "
                                f"result {dst!r} of kind {ty(dst)} has no "
                                "word encoding and no boundary box (closure "
                                "pairs, konts and conflicting kinds stay "
                                "demoted in indirect returns)")
                        elif (_word_boxable(ty(dst))
                              and _kind_size(ty(dst), structs,
                                             variants) is None):
                            probs.append(
                                f"indirect closure call through {callee!r}: "
                                f"result {dst!r} of kind {ty(dst)} has an "
                                "infinite layout, so no boundary box can be "
                                "sized")
                    for m in members:
                        sigm = sigs[m]
                        for a, pk in zip(args, sigm.params):
                            if ty(a) != pk:
                                probs.append(
                                    f"closure call to {m!r}: arg {a!r} is "
                                    f"{ty(a)}, expects {pk}")
                        if ty(dst) != sigm.ret:
                            probs.append(
                                f"closure call to {m!r}: result {dst!r} is "
                                f"{ty(dst)}, returns {sigm.ret}")
                elif callee.startswith(TRAIT_CALL_PREFIX):
                    method = callee[len(TRAIT_CALL_PREFIX):]
                    if not args:
                        probs.append(f"trait method call {method!r} with no receiver")
                        continue
                    res, target = _resolve_trait_call(
                        method, ty(args[0]), traits, module_names,
                        assume_final=True)
                    if res == "builtin":
                        check_builtin(target, dst, args)
                    elif res == "func":
                        sig = sigs.get(target)
                        if sig is None or target not in module_names:
                            probs.append(
                                f"trait method {method!r} resolves to unknown "
                                f"function {target!r}")
                        elif len(sig.params) != len(args):
                            probs.append(
                                f"trait method {method!r} -> {target!r} with "
                                "wrong arity")
                        else:
                            for a, pk in zip(args, sig.params):
                                if ty(a) != pk:
                                    probs.append(
                                        f"trait call {method!r} -> {target!r}: "
                                        f"arg {a!r} is {ty(a)}, expects {pk}")
                            if ty(dst) != sig.ret:
                                probs.append(
                                    f"trait call {method!r} -> {target!r}: "
                                    f"result {dst!r} is {ty(dst)}, returns {sig.ret}")
                    else:  # demote (or a pending that survived assume_final)
                        probs.append(target or
                                     f"trait method {method!r} cannot be resolved")
                elif callee.startswith(STATIC_CALL_PREFIX):
                    sres, starget = _resolve_static_call(
                        callee, traits, module_names)
                    if sres == "builtin":
                        check_builtin(starget, dst, args)
                    elif sres == "func":
                        sig = sigs.get(starget)
                        if sig is None or starget not in module_names:
                            probs.append(
                                f"static call {callee!r} resolves to unknown "
                                f"function {starget!r}")
                        elif len(sig.params) != len(args):
                            probs.append(
                                f"static call {callee!r} -> {starget!r} with "
                                "wrong arity")
                        else:
                            for a, pk in zip(args, sig.params):
                                if ty(a) != pk:
                                    probs.append(
                                        f"static call {callee!r} -> "
                                        f"{starget!r}: arg {a!r} is {ty(a)}, "
                                        f"expects {pk}")
                            if ty(dst) != sig.ret:
                                probs.append(
                                    f"static call {callee!r} -> {starget!r}: "
                                    f"result {dst!r} is {ty(dst)}, returns "
                                    f"{sig.ret}")
                    else:
                        probs.append(starget)
                elif bname in _NATIVE_RT_CALLS:
                    check_builtin(bname, dst, args)
                elif bname in _EXTERN_C_SIGS:
                    pks, rk_ = _EXTERN_C_SIGS[bname]
                    if len(args) != len(pks):
                        probs.append(
                            f"extern call {callee!r} with {len(args)} "
                            f"arguments (expects {len(pks)})")
                    else:
                        for a, pk in zip(args, pks):
                            if ty(a) != pk:
                                probs.append(
                                    f"extern call {callee!r}: arg {a!r} is "
                                    f"{ty(a)}, C signature expects {pk}")
                        if ty(dst) != rk_:
                            probs.append(
                                f"extern call {callee!r}: result {dst!r} is "
                                f"{ty(dst)}, C signature returns {rk_}")
                elif bname == "as_ptr":
                    if len(args) != 1:
                        probs.append(
                            f"as_ptr with {len(args)} arguments (expects 1)")
                    elif ty(args[0]) == STR:
                        pass  # identity: native strings are byte pointers
                    elif _is_vec(ty(args[0])):
                        if _vec_elem(ty(args[0])) != I64:
                            probs.append(
                                f"as_ptr of a Vec of {_vec_elem(ty(args[0]))} "
                                "elements (byte snapshots need i64 elements)")
                    elif _is_fvec(ty(args[0])):
                        if _fvec_elem(ty(args[0])) != I64:
                            probs.append(
                                f"as_ptr of a vector of "
                                f"{_fvec_elem(ty(args[0]))} elements (byte "
                                "snapshots need i64 elements)")
                    else:
                        probs.append(
                            f"as_ptr receiver {args[0]!r} has kind "
                            f"{ty(args[0])} (only str and vectors lower "
                            "natively)")
                    if ty(dst) != PTR:
                        probs.append(
                            f"as_ptr result {dst!r} is {ty(dst)}, not rawptr")
                elif bname in ("ptr_read", "ptr_write"):
                    want = 2 if bname == "ptr_read" else 3
                    if len(args) != want:
                        probs.append(
                            f"{callee} with {len(args)} arguments "
                            f"(expects {want})")
                    else:
                        if ty(args[0]) != PTR:
                            probs.append(
                                f"{callee} base {args[0]!r} is {ty(args[0])}, "
                                "not rawptr")
                        for a in args[1:]:
                            if ty(a) != I64:
                                probs.append(
                                    f"{callee} operand {a!r} is {ty(a)}, "
                                    "not i64")
                        if ty(dst) != I64:
                            probs.append(
                                f"{callee} result {dst!r} promoted to "
                                f"{ty(dst)}")
                elif bname == "assert":
                    if not args:
                        probs.append("assert with no condition")
                    elif ty(args[0]) != I64:
                        probs.append(
                            f"assert condition {args[0]!r} is {ty(args[0])} "
                            "(native truthiness is i64-only)")
                elif bname in _PRINT_BUILTINS:
                    for a in args:
                        ak = ty(a)
                        if _is_fvec(ak):
                            # print of a fixed vector renders via
                            # mx_fvec_to_str (repr parity); non-numeric
                            # leaves have no native repr.
                            if _fvec_leaf(ak)[0] not in (I64, F64):
                                probs.append(
                                    f"print of a vector of "
                                    f"{_fvec_leaf(ak)[0]} leaves")
                        elif _is_tile(ak):
                            # tiles render via mx_tile_to_str (repr parity)
                            if _tile_parts(ak)[0] not in (I64, F64):
                                probs.append(
                                    f"print of a tile of "
                                    f"{_tile_parts(ak)[0]} elements")
                        elif ak not in (I64, F64, STR):
                            probs.append(f"print of unsupported kind {ak}")
                elif bname == "neg":
                    if len(args) == 1 and ty(dst) not in (I64, F64):
                        probs.append(
                            f"neg of kind {ty(dst)} (the interpreter only "
                            "negates numbers)")
                elif bname == "not":
                    if len(args) == 1 and ty(dst) != I64:
                        probs.append(f"not of kind {ty(dst)}")
                elif bname == "bnot":
                    if len(args) != 1 or any(ty(x) != I64
                                             for x in (dst, *args)):
                        probs.append(
                            "bitwise complement `~` on a non-Int value "
                            "(the interpreter refuses too)")
                elif bname in _MATH_EXTERNS or bname in _INLINE_BUILTINS:
                    pass  # kinds pinned during inference
                elif callee in module_names:
                    sig = sigs.get(callee)
                    if sig is None or len(sig.params) != len(args):
                        probs.append(f"call to {callee!r} with wrong arity")
                        continue
                    for a, pk in zip(args, sig.params):
                        if ty(a) != pk:
                            probs.append(
                                f"call to {callee!r}: arg {a!r} is {ty(a)}, expects {pk}")
                    if ty(dst) != sig.ret:
                        probs.append(
                            f"call to {callee!r}: result {dst!r} is {ty(dst)}, "
                            f"returns {sig.ret}")
            elif rk == "alloc_struct":
                # locality "local" -> frame alloca; "global" -> heap malloc
                # with free at function exit (unknown localities were already
                # demoted during analysis).
                sname = rhs[1]
                if sname in structs.bad:
                    probs.append(structs.bad[sname])
                elif set(fn_ for (fn_, _fv) in args) != set(structs.fields.get(sname, ())):
                    probs.append(f"alloc_struct field mismatch for {sname!r}")
            elif rk in ("field_get", "field_set"):
                bk = ty(args[0])
                if not _is_struct(bk):
                    probs.append(
                        f"cannot determine struct type of {args[0]!r} for {rk}")
                else:
                    sname = _struct_name(bk)
                    if sname in structs.bad:
                        probs.append(structs.bad[sname])
                    elif rhs[1] not in structs.fields.get(sname, ()):
                        probs.append(f"struct {sname!r} has no field {rhs[1]!r}")
            elif rk == "make_variant":
                ename, vname = rhs[1], rhs[2]
                if ename in variants.bad:
                    probs.append(variants.bad[ename])
                    continue
                dk = ty(dst)
                ref = _enum_refinement(dk) if _is_enum(dk) else None
                if ref is None or vname not in ref \
                        or len(ref[vname]) != len(args):
                    if dst in info.dead_results:
                        continue
                    probs.append(
                        f"make_variant {vname!r} of enum {ename or 'anon'!r}: "
                        f"destination kind {dk} lacks the variant's refinement")
                    continue
                for i, a in enumerate(args):
                    sk = ref[vname][i]
                    store_k = _strip_refinement(ty(a))
                    if sk == CONFLICT or store_k == CONFLICT:
                        probs.append(
                            f"payload slot {i} of enum {ename or 'anon'!r} "
                            f"variant {vname!r} has conflicting kinds")
                    elif store_k != sk:
                        probs.append(
                            f"heterogeneous payload slot {i} of enum "
                            f"{ename or 'anon'!r}: variant {vname!r} stores "
                            f"{store_k} where merged flows require {sk} (no "
                            "coercion through tagged-union storage)")
                    elif _is_closure(sk) or sk == KONT:
                        probs.append(
                            f"enum {ename or 'anon'!r} payload slot {i} holds "
                            f"a closure ({sk}) (its env pointer may outlive "
                            "the creating frame)")
                    elif _is_agg(sk) and _kind_size(sk, structs, variants) is None:
                        probs.append(
                            f"enum {ename or 'anon'!r} payload slot {i} boxes "
                            f"a value of {sk} whose layout is infinite")
            elif rk == "variant_tag":
                if not _is_enum(ty(args[0])):
                    probs.append(
                        f"cannot determine enum type of {args[0]!r} for variant_tag")
                if ty(dst) != I64:
                    probs.append(f"variant tag {dst!r} promoted to {ty(dst)}")
            elif rk == "variant_field":
                bk = ty(args[0])
                if not _is_enum(bk):
                    probs.append(
                        f"cannot determine enum type of {args[0]!r} for variant_field")
                    continue
                ename = _enum_name(bk)
                idx = rhs[1]
                vname = rhs[2] if len(rhs) > 2 else None
                # Patterns can name a payload slot no constructor fills
                # (dead arm); grow the union so the GEP stays in bounds.
                variants.payload_max[ename] = max(
                    variants.payload_max.get(ename, 0), idx + 1)
                if ename in variants.bad:
                    probs.append(variants.bad[ename])
                    continue
                if vname is None:
                    probs.append(
                        f"variant_field {idx} of enum {ename or 'anon'!r} "
                        "carries no variant name (legacy MIR shape; payload "
                        "slot kinds are per-variant)")
                    continue
                ref = _enum_refinement(bk)
                if ref is None:
                    probs.append(
                        f"variant_field on enum value {args[0]!r} whose kind "
                        f"{bk} has no payload refinement")
                    continue
                if vname not in ref or idx >= len(ref[vname]):
                    # Dead arm: the refinement lists every variant any flow
                    # into this value can construct, so this variant's tag
                    # test can never pass here.  A scalar-shaped result is
                    # emitted as a typed zero (never executed); an aggregate
                    # result would need storage semantics we refuse to fake.
                    if _is_agg(ty(dst)) and dst not in info.dead_results:
                        probs.append(
                            f"variant_field {idx} of enum {ename or 'anon'!r} "
                            f"variant {vname!r} reads an unconstructed "
                            f"variant into aggregate kind {ty(dst)}")
                    continue
                sk = ref[vname][idx]
                if _is_enum(sk):
                    # Nested extraction reads through the canonical cells:
                    # sound only when every store into the nested enum agrees
                    # with them (otherwise the representation is per-value
                    # and was lost at the boxing boundary).
                    nested = _enum_name(sk)
                    mixed = variants.mixed_slots_of(nested)
                    if mixed:
                        descr = ", ".join(
                            f"{v}[{i}]" for (_e, v, i) in mixed)
                        probs.append(
                            f"variant_field {idx} of enum {ename or 'anon'!r} "
                            f"variant {vname!r} extracts nested enum "
                            f"{nested or 'anon'!r} whose payload slots "
                            f"({descr}) have instantiation-dependent "
                            "representations (lost at the boxing boundary)")
                        continue
                    sk = variants.canon_kind(nested)
                if sk == CONFLICT:
                    probs.append(
                        f"variant_field {idx} of enum {ename or 'anon'!r} "
                        f"variant {vname!r} has a conflicting slot kind")
                elif _is_closure(sk) or sk == KONT:
                    probs.append(
                        f"enum {ename or 'anon'!r} payload slot {idx} holds "
                        f"a closure ({sk}) (its env pointer may outlive the "
                        "creating frame)")
                elif ty(dst) != sk:
                    probs.append(
                        f"variant_field {idx} of enum {ename or 'anon'!r} "
                        f"variant {vname!r}: result {dst!r} is {ty(dst)}, "
                        f"slot is {sk}")
            elif rk == "make_closure":
                lname = rhs[1]
                for (cn, _vn) in args:
                    ck = closures.cell_kind(lname, cn)
                    if _is_closure(ck):
                        # Closure-in-closure capture (increment 13): the
                        # env field holds the pair by value; every member
                        # must be a known heap-env lambda (marked by the
                        # driver's env-capture scan).
                        check_closure_cell(
                            ck, f"lambda {lname!r} capture {cn!r}")
                    elif ck == CONFLICT:
                        probs.append(
                            f"lambda {lname!r} capture {cn!r} has conflicting kinds")
        if b.term[0] == "br_if" and ty(b.term[1]) != I64:
            probs.append(f"br_if condition {b.term[1]!r} is {ty(b.term[1])}")
        if b.term[0] == "ret" and _is_closure(ty(b.term[1])):
            # Defensive: the module driver marks every returned lambda
            # heap-env from its sig before this check runs, so this only
            # fires if that invariant is ever broken — a stack env crossing
            # a return would dangle.
            for m in _closure_members(ty(b.term[1])):
                if m not in closures.heap_env:
                    probs.append(
                        f"returns closure of lambda {m!r} not marked "
                        "heap-env (stack env would dangle)")
        # ret of a struct/enum value is fine: sret-style, the aggregate is
        # copied into the caller-provided %agg.ret slot (never a raw frame
        # pointer).  ret of a heap-env closure is fine: the pair is copied
        # sret-style and its env pointer aims at an immortal heap block.

    # Struct fields inline nested structs/enums (must have a finite layout);
    # enum payload slots box nested structs/enums; closures in either place
    # still demote (their env pointer may aim at a dying stack frame).
    used_structs = {_struct_name(k) for k in kinds.values() if _is_struct(k)}
    for sname in sorted(used_structs):
        if _kind_size(_STRUCT_PREFIX + sname, structs, variants) is None:
            probs.append(
                f"struct {sname!r} has a recursively inlined layout (a "
                "struct-in-struct cycle with no intervening enum box has no "
                "finite size)")
        for fn_ in structs.fields.get(sname, ()):
            fk = structs.field_kind(sname, fn_)
            if _is_closure(fk):
                probs.append(
                    f"struct {sname!r} field {fn_!r} holds a closure "
                    f"({fk}) (its env pointer may outlive the creating frame)")
            elif fk == KONT:
                probs.append(
                    f"struct {sname!r} field {fn_!r} holds an effect "
                    "continuation")
            elif fk == CONFLICT:
                probs.append(f"struct {sname!r} field {fn_!r} has conflicting kinds")
    # Enum payload slot validity (closures, conflicts, infinite layouts) is
    # checked per-site above: at make_variant against the destination's
    # refinement and at variant_field against the base's refinement / the
    # canonical cells — module-wide cells no longer demote functions that
    # only ever touch well-refined values of the enum.  Refined enum kinds
    # reaching THIS function through its own values still need every slot of
    # every refinement to be emittable (a refined kind can arrive through a
    # signature without any local variant op).
    for k in sorted(set(kinds.values())):
        ref = _enum_refinement(k)
        if not ref:
            continue
        ename = _enum_name(k)
        for v in sorted(ref):
            for i, sk in enumerate(ref[v]):
                if _is_closure(sk) or sk == KONT:
                    probs.append(
                        f"enum {ename or 'anon'!r} payload slot {i} holds a "
                        f"closure ({sk}) (its env pointer may outlive the "
                        "creating frame)")
                elif sk == CONFLICT:
                    probs.append(
                        f"enum {ename or 'anon'!r} variant {v!r} payload "
                        f"slot {i} has conflicting kinds")
                elif _is_agg(sk) and _kind_size(sk, structs, variants) is None:
                    probs.append(
                        f"enum {ename or 'anon'!r} variant {v!r} payload "
                        f"slot {i} boxes a value of {sk} whose layout is "
                        "infinite")
    # RANGE VALUES: __range materializes a fixed int vector natively, but
    # the interpreter's range is a plain Python list whose repr ("[0, 1]")
    # differs from a vector's ("vector[0, 1]").  Restrict range values (and
    # their copies) to the shapes where the two are observationally
    # identical — the iteration protocol (len / __index_get receiver) and
    # comprehension iterables — and demote every other use.
    tainted: Set[str] = set()
    tchanged = True
    while tchanged:
        tchanged = False
        for b in info.f.blocks:
            for op in b.ops:
                if op[0] != "let" or len(op) != 4:
                    continue
                if op[2][0] == "call" and op[2][1] == "__range" \
                        or (op[2][0] == "copy" and op[3]
                            and op[3][0] in tainted):
                    if op[1] not in tainted:
                        tainted.add(op[1])
                        tchanged = True
    if tainted:
        def bad_range_use(n: str) -> None:
            probs.append(
                f"range value {n!r} used outside the iteration protocol "
                "(the interpreter's range is a list; its repr differs from "
                "a vector's)")
        for b in info.f.blocks:
            for op in b.ops:
                if op[0] == "drop" or op[0] == "params":
                    continue
                if op[0] == "perform" and len(op) >= 7:
                    for a in op[4]:
                        if a in tainted:
                            bad_range_use(a)
                    continue
                if op[0] != "let" or len(op) != 4:
                    continue
                _, _dst2, rhs2, args2 = op
                rk2 = rhs2[0]
                if rk2 in ("alloc_struct", "make_closure", "handle_scope",
                           "try_scope"):
                    for pair in args2:
                        if isinstance(pair, tuple) and len(pair) == 2 \
                                and pair[1] in tainted:
                            bad_range_use(pair[1])
                    continue
                ok_positions: Set[int] = set()
                if rk2 == "copy":
                    ok_positions = {0}
                elif rk2 == "call" and _builtin_name(
                        rhs2[1], module_names) in ("len", "__index_get"):
                    ok_positions = {0}
                elif rk2 == "call" and rhs2[1] == "__vec_comprehension":
                    ok_positions = {2}
                for i, a in enumerate(args2):
                    if isinstance(a, str) and a in tainted \
                            and i not in ok_positions:
                        bad_range_use(a)
            t2 = b.term
            if t2[0] in ("br_if", "ret") and t2[1] in tainted:
                bad_range_use(t2[1])

    # ZIP RESULTS: a __zip result is virtual (the interpreter's list of
    # tuples has no native representation; mx_fvec_zip_map reads the
    # SOURCES).  Its ONLY legal use is the iterable position of a
    # __vec_comprehension — even copies demote (they would need value
    # forwarding for a value that does not exist).
    if info.zip_defs:
        zt = set(info.zip_defs)

        def bad_zip_use(n: str) -> None:
            probs.append(
                f"zip result {n!r} used outside a comprehension iterable "
                "(zip values are virtual: the native lowering reads the "
                "zipped sequences directly)")

        for b in info.f.blocks:
            for op in b.ops:
                if op[0] in ("drop", "params", "cell_wrap"):
                    continue
                if op[0] == "perform" and len(op) >= 7:
                    for a in op[4]:
                        if a in zt:
                            bad_zip_use(a)
                    continue
                if op[0] != "let" or len(op) != 4:
                    continue
                _, _dz, rhsz, argsz = op
                if rhsz[0] in ("alloc_struct", "make_closure",
                               "handle_scope", "try_scope"):
                    for pair in argsz:
                        if isinstance(pair, tuple) and len(pair) == 2 \
                                and pair[1] in zt:
                            bad_zip_use(pair[1])
                    continue
                okz: Set[int] = set()
                if rhsz[0] == "call" and rhsz[1] == "__vec_comprehension":
                    okz = {2}
                for i, a in enumerate(argsz):
                    if isinstance(a, str) and a in zt and i not in okz:
                        bad_zip_use(a)
            tz = b.term
            if tz[0] in ("br_if", "ret") and tz[1] in zt:
                bad_zip_use(tz[1])

    # MUTABLE-CAPTURE CELLS: a cell boxes exactly one 8-byte word, so a
    # cell-backed variable's kind must be a word-sized scalar (aggregates
    # and continuations stay interpreted).
    for n in sorted(cells.backed.get(info.f.name, ())):
        ck2 = ty(n)
        if _is_agg(ck2):
            probs.append(
                f"mutable capture {n!r} holds {ck2} (cells box one 8-byte "
                "word; aggregate cells stay interpreted)")
        elif ck2 == KONT:
            probs.append(
                f"mutable capture {n!r} holds an effect continuation")

    # MODULE CONSTANTS: globals are one 8-byte scalar slot each; the local
    # view must agree with the module-wide joined kind (like call sigs).
    gnames2 = set(info.global_reads)
    if info.f.name == _MODULE_INIT:
        gnames2 |= set(info.init_globals)
    for n in sorted(gnames2):
        gk2 = gtable.kind(n)
        if _is_agg(gk2):
            probs.append(
                f"module constant {n!r} holds {gk2} (only 8-byte scalar "
                "and pointer kinds fit native globals)")
        elif gk2 in (KONT, CONFLICT):
            probs.append(
                f"module constant {n!r} has kind {gk2} across the module")
        elif ty(n) != gk2:
            probs.append(
                f"module constant {n!r} is {gk2} module-wide but {ty(n)} "
                "in this function")

    # promote_matrix parameters: the local kind must be exactly the
    # promoted form of the incoming (caller-side) kind.  A sig still at the
    # i64 bottom with a promoted local kind means no caller ever passes a
    # vector here — demote rather than emit code for a flow that cannot be
    # typed.
    if info.promote_params:
        own2 = sigs.get(info.f.name)
        if own2 is not None and len(own2.params) == len(info.params):
            for i, p in enumerate(info.params):
                if p not in info.promote_params:
                    continue
                sk = own2.params[i]
                if ty(p) != _promote_kind(sk):
                    probs.append(
                        f"promote_matrix parameter {p!r}: local kind "
                        f"{ty(p)} is not the promoted form of its incoming "
                        f"kind {sk}")

    # Captures this function loads from its own env must be liftable too.
    if info.is_lambda:
        for cap in info.env_captures:
            ck = closures.cell_kind(info.f.name, cap)
            if _is_closure(ck):
                check_closure_cell(ck, f"capture {cap!r}")
            elif ck == KONT:
                probs.append(
                    f"capture {cap!r} is an effect continuation (resume must "
                    "run on its scope's owner stack)")
            elif ck == CONFLICT:
                probs.append(f"capture {cap!r} has conflicting kinds")
    # Handle-scope subfunctions: env fields and the scope's boundary values.
    if info.is_scope_member:
        site = info.scope_site
        is_try = (site in scopes.sites and scopes.sites[site].kind == "try")
        what = "try-site" if is_try else "handle-site"
        for cap in info.env_captures:
            check_env_cell(scopes.cell_kind(site, cap),
                           f"{what} capture {cap!r}")
        check_boundary(scopes.value_kind(site),
                       f"{'try' if is_try else 'handle'} value of site "
                       f"{site!r}")
        if info.scope_role == "trycatch":
            # Exactly one parameter, and it must have stayed a `str`: the
            # catch binding is the failure message and nothing else.
            if len(info.params) != 1:
                probs.append(
                    f"catch subfunction {info.f.name!r} declares "
                    f"{len(info.params)} parameters (expected exactly the "
                    "failure message)")
            elif ty(info.params[0]) != STR:
                probs.append(
                    f"catch parameter {info.params[0]!r} has kind "
                    f"{ty(info.params[0])}, but the caught failure message "
                    "is a str")
        if info.scope_role == "case":
            opn = scopes.case_op.get(info.f.name)
            if opn is not None:
                check_boundary(scopes.op_result_kind(opn),
                               f"effect op {opn!r} result")
                for i, p in enumerate(info.params):
                    if p != "__k":
                        check_boundary(ty(p),
                                       f"effect op {opn!r} parameter {p!r}")
    return probs


def _compute_slots(info: _Info, kinds: Dict[str, str]) -> List[str]:
    """Non-aggregate variables needing an alloca (mirrors codegen_clif)."""
    slots: List[str] = []
    for name, cnt in info.def_count.items():
        if _is_agg(kinds.get(name, I64)):
            continue  # aggregate vars always get their own storage alloca
        if cnt > 1:
            slots.append(name)
            continue
        db = info.def_block.get(name, 0)
        uses = info.use_blocks.get(name, set())
        if db != 0 and any(u != db for u in uses):
            slots.append(name)
    return sorted(slots)


# ---------------------------------------------------------------------------
# Provably-dead Vec analysis (which Vec.new results may be freed at exit)
# ---------------------------------------------------------------------------

# Native runtime calls that only READ or MUTATE a Vec through its receiver
# argument without retaining the pointer (metaxu_rt.c stores no receiver).
# as_ptr qualifies: mx_vec_as_bytes returns an INDEPENDENT byte snapshot
# that never aliases the vec's buffer, so freeing the vec leaves it intact.
_VEC_SAFE_RECEIVER_BUILTINS = {"push", "pop", "len", "__index_get", "as_ptr"}


def _provably_dead_vecs(f: MirFunc, kinds: Dict[str, str],
                        builtin_of, retained: Optional[Set[str]] = None
                        ) -> List[str]:
    """Vec.new result variables whose vector provably never escapes the
    frame, so `mx_vec_free` at every ret path is sound.

    The argument mirrors the @global-struct free proof: the malloc
    (mx_vec_new) happens unconditionally in the ENTRY block, and the
    pointer never leaves the frame — it is not returned, not stored in any
    aggregate (struct field / enum payload / closure env / another Vec),
    not captured, and not passed to any call except as the RECEIVER of the
    non-retaining native vec builtins.  Aliases created by `copy` are
    tracked (they hold the same pointer and are freed zero times — only the
    defining variable is freed once).  Anything not provable simply leaks
    by design (a leak is sound; a bad free is not), so this analysis bails
    conservatively: selects, re-definitions, cyclic entry blocks, or any
    unrecognized use disqualify the vec.

    ``builtin_of(callee, args)`` names the native builtin a call lowers to
    (None for anything else, including trait calls resolved to functions).
    """
    if not f.blocks:
        return []
    if 0 in _blocks_in_cycles(f):
        return []  # a re-executed entry block would double-free

    entry_ops = f.blocks[0].ops
    candidates = [op[1] for op in entry_ops
                  if op[0] == "let" and len(op) == 4
                  and op[2][0] == "call"
                  and op[2][1] in ("Vec.new", "__vec_lit")
                  and _is_vec(kinds.get(op[1], I64))]
    if not candidates:
        return []

    # def map: var -> list of (rhs, args)
    defs: Dict[str, List[Tuple[tuple, tuple]]] = {}
    for b in f.blocks:
        for op in b.ops:
            if op[0] == "let" and len(op) == 4:
                defs.setdefault(op[1], []).append((op[2], op[3]))

    freed: List[str] = []
    for site in candidates:
        # Alias group closure over plain copies AND __index_store results
        # (`v2 = __index_store(v, i, x)` on a Vec returns the SAME pointer:
        # v2 aliases v exactly like a copy would).
        group: Set[str] = {site}
        changed = True
        while changed:
            changed = False
            for b in f.blocks:
                for op in b.ops:
                    if op[0] != "let" or len(op) != 4 or op[1] in group:
                        continue
                    if op[2][0] == "copy" and op[3] and op[3][0] in group:
                        group.add(op[1])
                        changed = True
                    elif op[2][0] == "call" and op[2][1] == "__index_store" \
                            and len(op[3]) == 3 and op[3][0] in group:
                        group.add(op[1])
                        changed = True
        # Every group member's every def must be the site's producing call
        # (Vec.new / __vec_lit, for the site itself, exactly once), a copy
        # from within the group, or an aliasing __index_store of a group
        # member.
        ok = True
        for m in group:
            for (rhs, dargs) in defs.get(m, []):
                if m == site and rhs[0] == "call" \
                        and rhs[1] in ("Vec.new", "__vec_lit"):
                    continue
                if rhs[0] == "copy" and dargs and dargs[0] in group:
                    continue
                if rhs[0] == "call" and rhs[1] == "__index_store" \
                        and len(dargs) == 3 and dargs[0] in group:
                    continue
                ok = False
        # Any group member the caller declared RETAINED (cell-backed
        # storage, module-constant slots) outlives the frame: never free.
        if retained and group & retained:
            ok = False
        if len([1 for (rhs, _a) in defs.get(site, [])
                if rhs[0] == "call"
                and rhs[1] in ("Vec.new", "__vec_lit")]) != 1:
            ok = False
        # Every use of every member must be a whitelisted, non-escaping one.
        if ok:
            for b in f.blocks:
                for op in b.ops:
                    if not ok:
                        break
                    if op[0] == "drop":
                        continue
                    if op[0] == "match_fail" or op[0] == "params":
                        continue
                    if op[0] == "perform":
                        if any(a in group for a in op[4]):
                            ok = False
                        continue
                    if op[0] != "let" or len(op) != 4:
                        # unknown op shape: bail if we cannot see its uses
                        ok = False
                        continue
                    _, dst, rhs, oargs = op
                    rk = rhs[0]
                    if rk == "copy":
                        # copies from the group were folded into the group;
                        # a group member copied into a non-member cannot
                        # happen (closure above), so nothing to check.
                        continue
                    if rk == "call":
                        callee = rhs[1]
                        bname = builtin_of(callee, oargs)
                        if bname in _VEC_SAFE_RECEIVER_BUILTINS \
                                or bname in ("__index_store", "__index_set"):
                            # receiver-only use is safe (index stores
                            # mutate elements without retaining the
                            # pointer; __index_store's aliasing RESULT is
                            # already in the group); a group member in any
                            # VALUE position escapes (stored in a vec)
                            if any(a in group for a in oargs[1:]):
                                ok = False
                            continue
                        if any(a in group for a in oargs):
                            ok = False
                        continue
                    if rk in ("alloc_struct", "make_closure", "handle_scope",
                              "try_scope"):
                        # captured into an env (or a scope env): escapes
                        if any(v in group for (_n, v) in oargs):
                            ok = False
                        continue
                    if rk == "resume":
                        if any(a in group for a in oargs):
                            ok = False  # crosses the effect boundary
                        continue
                    # binop / select / field ops / variants / everything else:
                    # any appearance of a group member disqualifies.
                    if any(a in group for a in oargs):
                        ok = False
                t = b.term
                if t[0] in ("br_if", "ret") and t[1] in group:
                    ok = False
        if ok:
            freed.append(site)
    return freed


# ---------------------------------------------------------------------------
# Owned-string analysis (increment 8: which produced strings may be freed)
# ---------------------------------------------------------------------------
#
# Concat/to_string results are FRESH mallocs (metaxu_rt.c never returns or
# retains an input), so a string value whose flow the backend fully sees can
# be reclaimed.  A variable v is an OWNED STRING when:
#   * every def of v is a str literal (`const`), a producing op (str + str
#     concat, or to_string/int_to_str of an i64/f64), or a `copy` from an
#     EXCLUSIVE TRANSFER TEMP — a variable whose single def is itself a
#     literal/producing op and whose only use is that one copy (the ANF
#     shape `t = s + "x"; s = copy t` every re-assignment lowers to);
#   * every use of v is NON-RETAINING: a concat operand, a ==/!= string
#     comparison, a print/println argument, or a str len receiver.  Anything
#     else — returned, stored in any aggregate/env/Vec, passed to any other
#     call (including to_string of a str, which is an identity ALIAS),
#     copied to a non-transfer variable, crossing an effect boundary —
#     disqualifies v, and its produced values keep leaking by design.
#
# Emission gives each owned v a shadow slot (`%strown.v`, null-initialized)
# holding the pointer v currently OWNS: a producing def frees the previous
# owned pointer and stores the new one; a literal def frees and stores null
# (interned literal constants are never freed — provenance is static); every
# ret path frees the final owned pointer.  Freeing at the next def is sound
# because v's old value is unreachable there — its only aliases were v
# itself (just overwritten; later reads see the new def) and the transfer
# temp, whose single use has already executed.  A frame abandoned by an
# effect abort skips its rets and leaks (sound; a leak is never a UAF).

_STR_PRODUCER_BUILTINS = ("to_string", "int_to_str")


def _owned_strings(f: MirFunc, kinds: Dict[str, str], info: _Info,
                   builtin_of,
                   module_names: Set[str]) -> Tuple[List[str], Set[str], Set[str]]:
    """Ownership facts for produced strings this frame provably owns (see
    the section comment above): freed at redefinition and at frame exit.

    Returns (owned variables, literal transfer temps, producer transfer
    temps) — the temp sets cover only temps feeding OWNED variables, so
    emission can tell a copy-def carrying an interned literal (record null:
    never freed) from one carrying a fresh produced pointer (record it)."""
    if not f.blocks:
        return [], set(), set()

    def kd(n: str) -> str:
        return kinds.get(n, I64)

    strvars = {n for n in info.def_count
               if kd(n) == STR and n not in info.params
               and n not in info.env_captures
               and n not in info.dead_results
               and n not in info.tag_consts}
    if not strvars:
        return [], set(), set()

    # def shapes: ("lit",) / ("prod",) / ("xcopy", src) / ("bad",)
    defs: Dict[str, List[tuple]] = {n: [] for n in strvars}
    # use shapes: ("ok",) / ("copysrc", dst) / ("bad",)
    uses: Dict[str, List[tuple]] = {n: [] for n in strvars}

    def mark_use(n: Any, u: tuple) -> None:
        if isinstance(n, str) and n in strvars:
            uses[n].append(u)

    def is_str_concat(rhs: tuple, args: tuple, dst: str) -> bool:
        return (rhs[0] == "binop" and rhs[1] == "+" and len(args) == 2
                and kd(args[0]) == STR and kd(args[1]) == STR
                and kd(dst) == STR)

    for b in f.blocks:
        for op in b.ops:
            k0 = op[0]
            if k0 == "perform" and len(op) >= 7:
                for a in op[4]:
                    mark_use(a, ("bad",))
                if op[1] in strvars:
                    defs[op[1]].append(("bad",))
                continue
            if k0 != "let" or len(op) != 4:
                continue
            _, dst, rhs, args = op
            rk = rhs[0]
            # --- classify defs of string variables
            if dst in strvars:
                if rk == "const" and isinstance(rhs[1], str) \
                        and not isinstance(rhs[1], bool):
                    defs[dst].append(("lit",))
                elif is_str_concat(rhs, args, dst):
                    defs[dst].append(("prod",))
                elif rk == "call":
                    bname = builtin_of(rhs[1], tuple(args))
                    if bname in _STR_PRODUCER_BUILTINS and len(args) == 1 \
                            and kd(args[0]) in (I64, F64):
                        defs[dst].append(("prod",))
                    else:
                        defs[dst].append(("bad",))
                elif rk == "copy" and args:
                    defs[dst].append(("xcopy", args[0]))
                else:
                    defs[dst].append(("bad",))
            # --- classify uses of string variables
            if is_str_concat(rhs, args, dst):
                for a in args:
                    mark_use(a, ("ok",))
            elif rk == "binop" and rhs[1] in ("==", "!=") and len(args) == 2 \
                    and kd(args[0]) == STR and kd(args[1]) == STR:
                for a in args:
                    mark_use(a, ("ok",))
            elif rk == "copy":
                if args:
                    mark_use(args[0], ("copysrc", dst))
            elif rk == "call":
                callee = rhs[1]
                if callee in info.def_count:
                    for a in args:
                        mark_use(a, ("bad",))  # closure call: env unseen
                elif _builtin_name(callee, module_names) in _PRINT_BUILTINS:
                    # NAME PRECEDENCE: a user `fn print` is an ordinary
                    # module call (it may retain), so only the real builtin
                    # counts as non-retaining here.
                    for a in args:
                        mark_use(a, ("ok",))  # printf reads, never retains
                else:
                    bname = builtin_of(callee, tuple(args))
                    if bname == "len" and len(args) == 1 \
                            and kd(args[0]) == STR:
                        mark_use(args[0], ("ok",))  # mx_str_len reads only
                    else:
                        # includes to_string of a str (identity ALIAS), any
                        # module function, push, and every other callee.
                        for a in args:
                            mark_use(a, ("bad",))
            elif rk in ("alloc_struct", "make_closure", "handle_scope",
                        "try_scope"):
                for pair in args:
                    if isinstance(pair, tuple) and len(pair) == 2:
                        mark_use(pair[1], ("bad",))
            else:
                # select / field ops / variants / resume / anything else:
                # retaining or unanalyzed — disqualify.
                for a in args:
                    mark_use(a, ("bad",))
        t = b.term
        if t[0] in ("br_if", "ret"):
            mark_use(t[1], ("bad",))

    # Exclusive transfer temps: single def (lit/prod), single use, and that
    # use is a copy — ownership moves to the copy's destination.
    xfer_prod: Dict[str, str] = {}
    xfer_lit: Dict[str, str] = {}
    for tv in strvars:
        ds, us = defs[tv], uses[tv]
        if len(ds) == 1 and info.def_count.get(tv, 0) == 1 \
                and len(us) == 1 and us[0][0] == "copysrc":
            if ds[0][0] == "prod":
                xfer_prod[tv] = us[0][1]
            elif ds[0][0] == "lit":
                xfer_lit[tv] = us[0][1]

    owned: List[str] = []
    for v in sorted(strvars):
        ds, us = defs[v], uses[v]
        # Every def must have been classified (a def this scan did not see —
        # e.g. a param — cannot happen for strvars, but stay exact).
        if not ds or len(ds) != info.def_count.get(v, 0):
            continue
        if any(u[0] != "ok" for u in us):
            continue
        prods = 0
        ok = True
        for d in ds:
            if d[0] == "prod":
                prods += 1
            elif d[0] == "lit":
                pass
            elif d[0] == "xcopy":
                if xfer_prod.get(d[1]) == v:
                    prods += 1
                elif xfer_lit.get(d[1]) == v:
                    pass
                else:
                    ok = False
            else:
                ok = False
        if ok and prods:
            owned.append(v)
    owned_set = set(owned)
    lit_temps = {t for t, tgt in xfer_lit.items() if tgt in owned_set}
    prod_temps = {t for t, tgt in xfer_prod.items() if tgt in owned_set}
    return owned, lit_temps, prod_temps


# ---------------------------------------------------------------------------
# Unique-box analysis (increment 8: which payload boxes may be freed)
# ---------------------------------------------------------------------------

def _unique_box_enums(f: MirFunc, kinds: Dict[str, str],
                      info: _Info) -> Set[str]:
    """make_variant destinations whose payload boxes are UNIQUELY owned by
    this frame, so freeing them on every ret path is sound.

    Boxes normally leak by design because aggregate copies share box
    pointers shallowly.  But when the containing enum value provably never
    gets aggregate-copied beyond this frame's full view, the box has exactly
    one owner.  The proof mirrors _provably_dead_vecs: the make_variant
    happens unconditionally in the ENTRY block (exactly once per
    invocation, entry not in a CFG cycle), and the value — closed over
    intra-frame `copy` aliases — is used ONLY as the base of variant_tag /
    variant_field reads.  It is never returned, never passed to any call
    (a callee could capture its copy into an immortal heap closure env),
    never stored in a struct field / Vec / closure env / handle-site env,
    never re-boxed as another make_variant's payload, and never crosses an
    effect boundary.  variant_field only COPIES OUT of the box (boxes are
    write-once), so reads never extend the box's ownership.  Anything not
    provable stays leaked: a leak is sound, a bad free is not."""
    if not f.blocks or 0 in _blocks_in_cycles(f):
        return set()

    defs: Dict[str, List[Tuple[tuple, tuple]]] = {}
    for b in f.blocks:
        for op in b.ops:
            if op[0] == "let" and len(op) == 4:
                defs.setdefault(op[1], []).append((op[2], op[3]))

    candidates = []
    for op in f.blocks[0].ops:
        if op[0] != "let" or len(op) != 4 or op[2][0] != "make_variant":
            continue
        dst = op[1]
        dk = kinds.get(dst, I64)
        ref = _enum_refinement(dk) if _is_enum(dk) else None
        if ref is None:
            continue
        slots = ref.get(op[2][2], ())
        if not any(_is_agg(sk) for sk in slots):
            continue  # nothing boxed: nothing to reclaim
        candidates.append(dst)
    if not candidates:
        return set()

    result: Set[str] = set()
    for site in candidates:
        site_defs = defs.get(site, [])
        if len(site_defs) != 1 or site_defs[0][0][0] != "make_variant":
            continue
        # Close the alias group over plain copies (all intra-frame).
        group: Set[str] = {site}
        changed = True
        while changed:
            changed = False
            for b in f.blocks:
                for op in b.ops:
                    if op[0] == "let" and len(op) == 4 \
                            and op[2][0] == "copy" and op[3] \
                            and op[3][0] in group and op[1] not in group:
                        group.add(op[1])
                        changed = True
        ok = not any(m in info.params or m in info.env_captures
                     for m in group)
        # Every group member's every def is the site's make_variant (site
        # only) or a copy from within the group.
        if ok:
            for m in group:
                for (rhs, dargs) in defs.get(m, []):
                    if m == site and rhs[0] == "make_variant":
                        continue
                    if rhs[0] == "copy" and dargs and dargs[0] in group:
                        continue
                    ok = False
        # Every use of every group member is a variant_tag/variant_field
        # base read or an intra-group copy.
        if ok:
            for b in f.blocks:
                for op in b.ops:
                    if not ok:
                        break
                    k0 = op[0]
                    if k0 in ("params", "drop", "match_fail"):
                        continue
                    if k0 == "perform" and len(op) >= 7:
                        if any(a in group for a in op[4]):
                            ok = False
                        continue
                    if k0 != "let" or len(op) != 4:
                        ok = False  # unknown op shape: cannot see its uses
                        continue
                    _, _dst, rhs, oargs = op
                    rk = rhs[0]
                    if rk in ("variant_tag", "variant_field"):
                        continue  # base-pointer read only
                    if rk == "copy":
                        continue  # group copies were closed over above
                    if rk in ("alloc_struct", "make_closure", "handle_scope",
                              "try_scope"):
                        if any(isinstance(p, tuple) and len(p) == 2
                               and p[1] in group for p in oargs):
                            ok = False
                        continue
                    if any(isinstance(a, str) and a in group for a in oargs):
                        ok = False
                t = b.term
                if t[0] in ("br_if", "ret") and t[1] in group:
                    ok = False
        if ok:
            result.add(site)
    return result


# ---------------------------------------------------------------------------
# Emission
# ---------------------------------------------------------------------------

class _ModuleState:
    """Shared cross-function emission state: string pool + runtime needs."""

    def __init__(self) -> None:
        self.strings: Dict[str, str] = {}  # content -> global name
        self.print_helpers: Set[str] = set()  # subset of {"i64","f64","str"}
        self.math_used: Set[str] = set()
        self.uses_abort = False
        self.uses_malloc = False   # @global structs: malloc/free declares
        self.uses_printf = False   # direct variadic printf (multi-arg print)
        self.runtime_syms: Set[str] = set()  # mx_* native runtime declares
        # Inlined Vec-mutator fast paths read the per-thread write permit
        # directly (contended-write guard, docs/contention_as_permission.md).
        self.uses_tls_permit = False
        # Inline Vec accesses carry TBAA tags separating header words from
        # element words (always distinct allocations by the runtime's
        # contract), so an element store cannot pin the loop-hoisted
        # len/data loads.  Untagged accesses stay compatible with both.
        self.uses_vec_tbaa = False
        # extern C FFI declares beyond malloc/free (memcpy/realloc/fopen/
        # fclose), emitted with their real C signatures.
        self.extern_c_syms: Set[str] = set()
        self.used_enums: Set[str] = set()      # enum names needing %enum types
        self.uses_closure_pair = False         # %mx.closure type needed
        # lambda name -> ((capture name, kind), ...) for %env.L emission
        self.env_types: Dict[str, Tuple[Tuple[str, str], ...]] = {}
        # handle site -> ((env field name, kind), ...) for %henv.<site>
        self.scope_env_types: Dict[str, Tuple[Tuple[str, str], ...]] = {}
        # handle site -> (owner fn, member fns): final assembly only keeps
        # artifacts of sites whose owner+members all emitted (cascade-safe).
        self.scope_sites: Dict[str, Tuple[str, Tuple[str, ...]]] = {}
        # handle site -> op-name/arity constant-array globals text
        self.scope_tables: Dict[str, str] = {}
        # handle site -> body thunk + dispatcher define text
        self.scope_thunks: Dict[str, str] = {}
        # __effect_default fn -> its mx_default_fn thunk define text, and
        # the emitted functions whose performs hand that thunk to
        # mx_perform_or_default (liveness: keep the thunk only when the
        # default fn AND at least one user survived).
        self.default_thunks: Dict[str, str] = {}
        self.default_thunk_users: Dict[str, Set[str]] = {}
        # owner fn -> per-site comprehension thunk defines (kept only when
        # the owner emitted; each thunk calls its body lambda's symbol,
        # which dep_names makes a dependency of the owner).
        self.comp_thunks: Dict[str, List[str]] = {}
        self.thunk_seq = 0
        # Module constants touched by emitted functions: name -> kind.
        # Emitted as @mx_g_<name> internal globals, zero-initialized and
        # filled by @mx___module_init (called by llvm_run's entry wrapper
        # before the entry point — the interpreter's _ensure_globals).
        self.globals_used: Dict[str, str] = {}

    def intern_string(self, content: str) -> str:
        if content not in self.strings:
            self.strings[content] = f"@.str.{len(self.strings)}"
        return self.strings[content]


# ---------------------------------------------------------------------------
# Fixed-vector static lengths (increment 11)
# ---------------------------------------------------------------------------

# Static-length lattice value for "provably unknown / dynamic" (the safe
# top).  A missing entry is the bottom: no def has produced a fact yet
# (equivalent to dynamic at every use site — the fast path requires a
# known length, so bottom and top both fall back to the runtime call).
_FLEN_DYN = -1

# Inline SIMD is only emitted for lengths in [1, _FLEN_MAX]: `<N x double>`
# is legal for any N, but a huge constant vector would bloat the IR and
# spill anyway — beyond the cap the C loop (which LLVM may still
# auto-vectorize) is the better lowering.
_FLEN_MAX = 64


def _fvec_static_lens(f: MirFunc, kinds: Dict[str, str],
                      info: _Info) -> Dict[str, int]:
    """Per-variable static lengths of fixed-vector values (increment 11).

    Returns var -> length where length >= 0 means EVERY value the variable
    can hold at any use has exactly that many elements, and _FLEN_DYN means
    unknown.  Sound join rules — mismatched or unknown lengths degrade to
    dynamic, never to a wrong number:

      * producers with a statically-known count: ``__vec_lit`` (element
        count), ``__vec_zeros``/``__vec_filled`` (const count arg),
        ``__range`` (both bounds const), ``__slice_get`` (const/None bounds
        over a known input, CPython slice.indices semantics — exactly what
        mx_fvec_slice implements), ``__vec_comprehension`` (input length,
        else the const declared size the runtime enforces), ``__cast``
        identity reinterpretations, copies and selects.
      * a binop result takes any operand's known length: mx_fvec_binop
        ABORTS on a vector-vector length mismatch, so on every continuing
        path the operands (and result) share one length (the inline fast
        path itself additionally requires BOTH operand lengths known-equal
        — see the emission site — so a mismatch still reaches the aborting
        runtime call).
      * everything else — parameters, captures, call results, struct/enum
        reads, effect boundaries, and ``const None`` slot initializers
        (a possibly-null vector must never take the inline path) — is
        dynamic.

    Per-name facts join over ALL defs of the name (the same discipline as
    the kind map: one kind/length per variable), iterated to fixpoint so
    copy chains and loop-carried joins settle."""
    lens: Dict[str, int] = {}

    def is_v(n: str) -> bool:
        return _is_fvec(kinds.get(n, I64))

    def join2(a: Optional[int], b: Optional[int]) -> Optional[int]:
        if a is None:
            return b
        if b is None:
            return a
        return a if a == b else _FLEN_DYN

    def mark(n: str, v: Optional[int]) -> bool:
        nv = join2(lens.get(n), v)
        if nv is not None and nv != lens.get(n):
            lens[n] = nv
            return True
        return False

    def slice_len(recv: str, bounds: Tuple[str, ...]) -> Optional[int]:
        n = lens.get(recv)
        if n is None or n == _FLEN_DYN:
            return _FLEN_DYN
        vals: List[Optional[int]] = []
        for a in bounds:
            if a in info.const_nones:
                vals.append(None)
            elif a in info.const_ints:
                vals.append(info.const_ints[a])
            else:
                return _FLEN_DYN  # dynamic bound (emission demotes anyway)
        try:
            return len(range(*slice(*vals).indices(n)))
        except ValueError:  # step 0: the runtime aborts; never inline after
            return _FLEN_DYN

    changed = True
    while changed:
        changed = False
        for p in (*info.params, *info.env_captures):
            if is_v(p):
                changed = mark(p, _FLEN_DYN) or changed
        for b in f.blocks:
            for op in b.ops:
                if op[0] == "perform":
                    if is_v(op[1]):
                        changed = mark(op[1], _FLEN_DYN) or changed
                    continue
                if op[0] == "promote_matrix":
                    # mx_fvec_promote preserves length, but the promoted
                    # names are parameters (already dynamic).
                    for n in (tuple(op[1]) if len(op) > 1 else ()):
                        if is_v(n):
                            changed = mark(n, _FLEN_DYN) or changed
                    continue
                if op[0] != "let" or len(op) != 4:
                    continue
                _, dst, rhs, args = op
                if not is_v(dst):
                    continue
                rk = rhs[0]
                if rk == "copy":
                    changed = mark(dst, lens.get(args[0])) or changed
                elif rk == "select":
                    v = join2(lens.get(args[1]), lens.get(args[2])) \
                        if len(args) == 3 else _FLEN_DYN
                    changed = mark(dst, v) or changed
                elif rk == "binop":
                    known = [lens[a] for a in args
                             if is_v(a) and lens.get(a) not in (None, _FLEN_DYN)]
                    if known:
                        changed = mark(dst, known[0]) or changed
                    elif any(is_v(a) and lens.get(a) == _FLEN_DYN
                             for a in args):
                        changed = mark(dst, _FLEN_DYN) or changed
                elif rk == "call":
                    callee = rhs[1]
                    v: Optional[int] = _FLEN_DYN
                    if callee in info.def_count:
                        pass  # closure-call result: dynamic
                    elif callee == "__vec_lit":
                        v = len(args) - 1
                    elif callee in ("__vec_zeros", "__vec_filled") and args:
                        c = info.const_ints.get(args[0])
                        v = c if c is not None and c >= 0 else _FLEN_DYN
                    elif callee == "__range" and len(args) == 2:
                        s = info.const_ints.get(args[0])
                        e = info.const_ints.get(args[1])
                        v = max(0, e - s) \
                            if s is not None and e is not None else _FLEN_DYN
                    elif callee == "__slice_get" and len(args) == 4:
                        v = slice_len(args[0], args[1:])
                    elif callee == "__index_store" and len(args) == 3:
                        # A functional element update preserves the length
                        # (mx_fvec_set_copy copies the whole block); a
                        # still-bottom receiver refines on later rounds,
                        # like the copy rule.
                        v = lens.get(args[0])
                    elif callee == "__vec_comprehension" and len(args) == 3:
                        v = lens.get(args[2])
                        if v is None or v == _FLEN_DYN:
                            # mx_fvec_map aborts unless the input length
                            # equals the const declared size, so on every
                            # continuing path that size IS the length.
                            c = info.const_ints.get(args[0])
                            v = c if c is not None and c >= 0 else _FLEN_DYN
                    elif callee == "__cast":
                        # dst is only vector-kinded in the identity
                        # reinterpretation case.
                        v = lens.get(args[0]) if args else _FLEN_DYN
                    changed = mark(dst, v) or changed
                else:
                    # const (incl. None slot initializers), call-adjacent
                    # defs, struct/enum reads, handle_scope, resume, ...
                    changed = mark(dst, _FLEN_DYN) or changed
    return lens


def emit_fvec_reduce(vec_ptr: str, n: int, leaf: str,
                     tmp_prefix: str) -> Tuple[str, List[str], str]:
    """INCREMENT 11 emission helper: horizontal reduction (sum) of a
    static-length-``n`` flat fixed vector via ``llvm.vector.reduce``.

    Returns ``(declare_line, body_lines, result_ssa)``: the intrinsic
    declaration the module needs once, the instruction lines to splice into
    a block, and the SSA name (``double`` for an f64 leaf, ``i64`` for an
    int leaf) holding the sum.

    The f64 form is the ORDERED reduction (no ``reassoc`` flag), seeded
    with ``-0.0`` — the exact identity of ``fadd`` — so the result is
    bit-identical to the interpreter's left-to-right ``e0 + e1 + ...``
    fold for every input, NaN/inf/-0.0 included.

    No MIR shape reaches this yet: example 06's ``sum``/``dot``/``norm``
    demote UPSTREAM on closure-kind conflicts (map/reduce take different
    lambdas at one call site), which is not this backend's to fix.  The
    helper is the ready lowering — synthetic-module tests pin that the IR
    it emits verifies, runs, and matches Python's fold exactly."""
    if n < 1:
        raise ValueError(f"reduction needs a static length >= 1, got {n}")
    if leaf not in (I64, F64):
        raise ValueError(f"reduction leaf must be i64/f64, got {leaf!r}")
    ety = "double" if leaf == F64 else "i64"
    vty = f"<{n} x {ety}>"
    if leaf == F64:
        intr = f"llvm.vector.reduce.fadd.v{n}f64"
        decl = f"declare double @{intr}(double, {vty})"
    else:
        intr = f"llvm.vector.reduce.add.v{n}i64"
        decl = f"declare i64 @{intr}({vty})"
    p = f"%{tmp_prefix}.elems"
    v = f"%{tmp_prefix}.v"
    r = f"%{tmp_prefix}.sum"
    body = [
        f"  {p} = getelementptr inbounds i8, ptr {vec_ptr}, i64 8",
        f"  {v} = load {vty}, ptr {p}, align 8"
        f"  ; element words as {vty} (block layout: i64 len, then words)",
    ]
    if leaf == F64:
        body.append(
            f"  {r} = call double @{intr}(double -0.000000e+00, {vty} {v})"
            "  ; ordered fadd reduction (-0.0 seed: exact fadd identity)")
    else:
        body.append(f"  {r} = call i64 @{intr}({vty} {v})")
    return decl, body, r


def _scope_body_sym(site: str) -> str:
    return "mxfx.body." + _sanitize(site)


def _scope_disp_sym(site: str) -> str:
    return "mxfx.disp." + _sanitize(site)


def _scope_ops_sym(site: str) -> str:
    return "mxfx.ops." + _sanitize(site)


def _scope_np_sym(site: str) -> str:
    return "mxfx.np." + _sanitize(site)


def _word_encode(val: str, kind: str, dst: str) -> Tuple[List[str], str]:
    """Lines turning a typed value into an opaque i64 boundary word."""
    if kind == F64:
        return [f"  {dst} = bitcast double {val} to i64"], dst
    if _llscalar(kind) == "ptr":
        return [f"  {dst} = ptrtoint ptr {val} to i64"], dst
    return [], val


def _word_decode(val: str, kind: str, dst: str) -> Tuple[List[str], str]:
    """Inverse of _word_encode."""
    if kind == F64:
        return [f"  {dst} = bitcast i64 {val} to double"], dst
    if _llscalar(kind) == "ptr":
        return [f"  {dst} = inttoptr i64 {val} to ptr"], dst
    return [], val


def _emit_scope_artifacts(site: str, rec: _ScopeSite, sigs: Dict[str, _Sig],
                          mod: _ModuleState) -> None:
    """Per-handle-site shims for the effects runtime: the op-name /
    case-arity constant tables, the body thunk (`i64 (ptr env)`) and the
    dispatcher (`i64 (ptr env, i64 op_index, ptr args, ptr k)`).  Op
    indices are DENSE in the site's case order (documented per arm).

    Aggregate body/case results used to keep the ordinary sret convention,
    with the shim malloc'ing a BOUNDARY BOX, calling sret-style into it and
    returning the box pointer as the word — one unconditional malloc per
    dispatch.  Increment 17 moves that boxing INTO the subfunction (the
    BOUNDARY-WORD RETURN ABI, `i64 (...)`, see _emit_function): the value
    always continues as a word anyway, and doing it there lets the
    double-box elision drop the allocation whenever the returned value is
    already an immortal write-once box — the fold-shaped `resume(...)`
    case, where the box was allocated by whoever produced the resume
    value.  Where boxing IS needed, to_word still mallocs (never alloca:
    the word outlives the shim's frame, crossing mx_handle/mx_perform back
    to a different stack) and the box stays write-once and immortal.
    Aggregate case PARAMS receive the sender's box pointer directly — the
    ordinary aggregate-param byval-copy convention is exactly the
    copy-out."""
    if site in mod.scope_thunks:
        return

    n = len(rec.cases)
    op_ptrs, nps = [], []
    for (opn, cparams, _hfn) in rec.cases:
        op_ptrs.append(f"ptr {mod.intern_string(opn)}")
        nps.append(f"i64 {len(cparams)}")
    mod.scope_tables[site] = (
        f"; handle site {site}: op index order "
        f"{[opn for (opn, _p, _h) in rec.cases]}\n"
        f"@{_scope_ops_sym(site)} = private unnamed_addr constant "
        f"[{n} x ptr] [{', '.join(op_ptrs)}]\n"
        f"@{_scope_np_sym(site)} = private unnamed_addr constant "
        f"[{n} x i64] [{', '.join(nps)}]")

    bsig = sigs[rec.body_fn]
    bl = [f"define internal i64 @{_scope_body_sym(site)}(ptr %env) {{",
          "entry:"]
    if _is_agg(bsig.ret):
        # Boundary-word ABI: the body already returns the word (boxing
        # only where it must), so the thunk just forwards it.
        bl.append(f"  %w = call i64 @{mangle(rec.body_fn)}(ptr %env)"
                  f"  ; boundary-word body result {bsig.ret}")
        bl.append("  ret i64 %w")
    else:
        brty = _llscalar(bsig.ret)
        bl.append(f"  %r = call {brty} @{mangle(rec.body_fn)}(ptr %env)")
        enc, v = _word_encode("%r", bsig.ret, "%w")
        bl += enc
        bl.append(f"  ret i64 {v}")
    bl.append("}")

    dl = [f"define internal i64 @{_scope_disp_sym(site)}"
          "(ptr %env, i64 %op, ptr %args, ptr %k) {",
          "entry:"]
    if n:
        arms = " ".join(f"i64 {i}, label %case{i}" for i in range(n))
        dl.append(f"  switch i64 %op, label %unreach [ {arms} ]")
    else:
        dl.append("  br label %unreach")
    for i, (opn, cparams, hfn) in enumerate(rec.cases):
        csig = sigs[hfn]
        dl.append(f"case{i}:  ; op {opn!r} -> @{mangle(hfn)}")
        avals = ["ptr %env"]
        for j in range(len(cparams)):
            pk = csig.params[j] if j < len(csig.params) else I64
            wp, wv = f"%c{i}.a{j}p", f"%c{i}.a{j}w"
            dl.append(
                f"  {wp} = getelementptr inbounds i64, ptr %args, i64 {j}")
            dl.append(f"  {wv} = load i64, ptr {wp}")
            if _is_agg(pk):
                # The word is the sender's boundary-box pointer; the case
                # fn byval-copies the aggregate out in its prelude.
                dl.append(f"  %c{i}.a{j} = inttoptr i64 {wv} to ptr"
                          f"  ; boundary box: case param {pk}")
                avals.append(f"ptr %c{i}.a{j}")
            else:
                enc, v = _word_decode(wv, pk, f"%c{i}.a{j}")
                dl += enc
                avals.append(f"{_llscalar(pk)} {v}")
        avals.append("ptr %k")
        if _is_agg(csig.ret):
            # Boundary-word ABI: the case fn returns the word itself.
            dl.append(f"  %c{i}.w = call i64 @{mangle(hfn)}"
                      f"({', '.join(avals)})"
                      f"  ; boundary-word case result {csig.ret}")
            dl.append(f"  ret i64 %c{i}.w")
            continue
        crty = _llscalar(csig.ret)
        dl.append(f"  %c{i}.r = call {crty} @{mangle(hfn)}({', '.join(avals)})")
        enc, v = _word_encode(f"%c{i}.r", csig.ret, f"%c{i}.w")
        dl += enc
        dl.append(f"  ret i64 {v}")
    dl.append("unreach:")
    dl.append("  call void @abort()  ; dispatcher op index out of range")
    dl.append("  unreachable")
    dl.append("}")
    mod.uses_abort = True
    mod.scope_thunks[site] = "\n".join(bl) + "\n\n" + "\n".join(dl)
    mod.scope_sites[site] = (rec.owner, rec.member_fns())


def _try_body_sym(site: str) -> str:
    return "mxtc.body." + _sanitize(site)


def _try_catch_sym(site: str) -> str:
    return "mxtc.catch." + _sanitize(site)


def _emit_try_artifacts(site: str, rec: _ScopeSite, sigs: Dict[str, _Sig],
                        mod: _ModuleState) -> None:
    """Per-try-site shims for mx_try: the body thunk (`i64 (ptr env)`) and
    the catch thunk (`i64 (ptr env, ptr msg)`).

    Same conventions as the handle-site shims (_emit_scope_artifacts): the
    subfunctions read their free names out of the site's shared env struct,
    and an aggregate result travels as a BOUNDARY WORD (the subfunction
    boxes it itself — `is_boundary_ret`), because the try value is produced
    on one side of a non-local control transfer and consumed on the other.

    The catch thunk's `msg` is the runtime's failure text — a plain
    NUL-terminated `char *`, i.e. exactly the backend's `str` kind, and
    exactly the interpreter's `InterpError.message`.  It is passed straight
    through: no copy, no formatting, no compiler context (docs/try_catch.md
    "What the catch binding is, exactly").
    """
    if site in mod.scope_thunks:
        return
    bsig = sigs[rec.body_fn]
    bl = [f"define internal i64 @{_try_body_sym(site)}(ptr %env) {{",
          f"  ; mx_try body thunk for @{mangle(rec.body_fn)}",
          "entry:"]
    if _is_agg(bsig.ret):
        bl.append(f"  %w = call i64 @{mangle(rec.body_fn)}(ptr %env)"
                  f"  ; boundary-word try body result {bsig.ret}")
        bl.append("  ret i64 %w")
    else:
        brty = _llscalar(bsig.ret)
        bl.append(f"  %r = call {brty} @{mangle(rec.body_fn)}(ptr %env)")
        enc, v = _word_encode("%r", bsig.ret, "%w")
        bl += enc
        bl.append(f"  ret i64 {v}")
    bl.append("}")

    csig = sigs[rec.catch_fn]
    cl = [f"define internal i64 @{_try_catch_sym(site)}"
          "(ptr %env, ptr %msg) {",
          f"  ; mx_try catch thunk for @{mangle(rec.catch_fn)}: %msg is the",
          "  ; failure text the interpreter binds (InterpError.message)",
          "entry:"]
    if _is_agg(csig.ret):
        cl.append(f"  %w = call i64 @{mangle(rec.catch_fn)}"
                  "(ptr %env, ptr %msg)"
                  f"  ; boundary-word catch result {csig.ret}")
        cl.append("  ret i64 %w")
    else:
        crty = _llscalar(csig.ret)
        cl.append(f"  %r = call {crty} @{mangle(rec.catch_fn)}"
                  "(ptr %env, ptr %msg)")
        enc, v = _word_encode("%r", csig.ret, "%w")
        cl += enc
        cl.append(f"  ret i64 {v}")
    cl.append("}")

    mod.scope_thunks[site] = "\n".join(bl) + "\n\n" + "\n".join(cl)
    mod.scope_sites[site] = (rec.owner, rec.member_fns())


def _default_thunk_sym(dfn: str) -> str:
    return "mxfx.dflt." + _sanitize(dfn)


def _emit_default_thunk(dfn: str, nargs: int, sigs: Dict[str, _Sig],
                        structs: _StructTable, variants: _VariantTable,
                        mod: _ModuleState, user: str) -> None:
    """The per-op `mx_default_fn` thunk for an op with DYNAMIC default
    routing: `i64 (ptr env, ptr args)` decoding the boundary argument words
    into the declared default's parameter kinds, calling it, and
    word-encoding its result.

    This is the dispatcher's per-case arm with the op index and the
    continuation removed — the default is not a handler case: it receives
    no `__k` (there is nothing to resume; the perform simply becomes this
    call) and it runs on the PERFORMING stack, so no boundary is crossed by
    control, only by values.  Value conventions are therefore identical to
    a case's: aggregates arrive as the sender's boundary-box pointer (the
    callee's byval copy IS the copy-out) and an aggregate result is
    sret-filled into a fresh malloc'd box whose pointer is the word
    (malloc, never alloca: the word outlives this frame exactly as the
    dispatcher's does).

    `%env` is unused today — __effect_default$E$op functions are top-level
    module functions with no captures, so the emitted call sites pass
    `ptr null`.  The parameter is part of the ABI so a capturing default
    needs no second entry point.
    """
    mod.default_thunk_users.setdefault(dfn, set()).add(user)
    if dfn in mod.default_thunks:
        return
    dsig = sigs[dfn]
    tl = [f"define internal i64 @{_default_thunk_sym(dfn)}"
          "(ptr %env, ptr %args) {",
          f"  ; mx_default_fn thunk for @{mangle(dfn)} (declared `= expr` "
          "default;",
          "  ; called by mx_perform_or_default when no scope handles the op,",
          "  ; ON THE PERFORMING STACK -- a default is not a suspension)",
          "entry:"]
    avals = []
    for j in range(nargs):
        pk = dsig.params[j] if j < len(dsig.params) else I64
        wp, wv = f"%a{j}p", f"%a{j}w"
        tl.append(f"  {wp} = getelementptr inbounds i64, ptr %args, i64 {j}")
        tl.append(f"  {wv} = load i64, ptr {wp}")
        if _is_agg(pk):
            tl.append(f"  %a{j} = inttoptr i64 {wv} to ptr"
                      f"  ; boundary box: default param {pk}")
            avals.append(f"ptr %a{j}")
        else:
            enc, v = _word_decode(wv, pk, f"%a{j}")
            tl += enc
            avals.append(f"{_llscalar(pk)} {v}")
    if _is_agg(dsig.ret):
        size = _kind_size(dsig.ret, structs, variants)
        if size is None:  # unreachable: checks demote infinite layouts
            raise _Unsupported(
                f"boundary box of {dsig.ret} has infinite layout")
        mod.uses_malloc = True
        tl.append(f"  %rbox = call ptr @malloc(i64 {max(size, 8)})"
                  f"  ; boundary box: default result {dsig.ret} "
                  "(write-once, leaks by design)")
        tl.append(f"  call void @{mangle(dfn)}"
                  f"({', '.join(['ptr %rbox'] + avals)})")
        tl.append("  %w = ptrtoint ptr %rbox to i64")
        tl.append("  ret i64 %w")
    else:
        rty = _llscalar(dsig.ret)
        tl.append(f"  %r = call {rty} @{mangle(dfn)}({', '.join(avals)})")
        enc, v = _word_encode("%r", dsig.ret, "%w")
        tl += enc
        tl.append(f"  ret i64 {v}")
    tl.append("}")
    mod.default_thunks[dfn] = "\n".join(tl)


def _emit_placeholder(info: _Info, sig: _Sig) -> str:
    sym = mangle(info.f.name)
    ptys = ", ".join(_llparam(k) for k in sig.params)
    rty = "void (sret ptr)" if _is_agg(sig.ret) else _llscalar(sig.ret)
    lines = [f"; function @{sym}: placeholder -- unsupported for direct LLVM emission"]
    for r in info.reasons:
        lines.append(f";   reason: {r}")
    lines.append(f"; declare @{sym}({ptys}) -> {rty}")
    return "\n".join(lines)


def _emit_function(info: _Info, kinds: Dict[str, str], sigs: Dict[str, _Sig],
                   structs: _StructTable, variants: _VariantTable,
                   closures: _ClosureTable, traits: _TraitTable,
                   scopes: _ScopeTable, module_names: Set[str],
                   mod: _ModuleState, emitted_names: Set[str],
                   writeback_map: Dict[str, frozenset],
                   cells: "_CellTable", gtable: "_GlobalTable",
                   word_uniform: Optional[Set[str]] = None) -> str:
    f = info.f
    sig = sigs[f.name]
    word_uniform = word_uniform if word_uniform is not None else set()
    # WORD-UNIFORM LAMBDA (increment 13): this lambda participates in
    # indirect calls, so its native signature is `i64 (ptr env, i64...)`
    # — incoming words are decoded to the typed kinds in the prelude and
    # the typed return value is encoded back to a word at every ret.
    is_uniform_lambda = info.is_lambda and f.name in word_uniform

    def kind(n: str) -> str:
        return kinds.get(n, I64)

    def llty(n: str) -> str:
        return _llscalar(kind(n))

    # MUTABLE-CAPTURE CELLS (increment 12): cell-backed variables read and
    # write through a one-word heap cell pointer (%cellp.<n>, defined in
    # the entry block: malloc'd here, or loaded from the env for cell
    # captures) instead of a slot/register; envs capture the POINTER.
    cellset = set(cells.backed.get(f.name, ()))
    cap_cellset = set(cells.cap_cells.get(f.name, ()))

    # Tail-position resumes of this handler case (compiler/effect_tail.py,
    # shared with the interpreter so both engines trampoline the SAME
    # sites): these emit mx_resume_tail instead of mx_resume.  Only case
    # subfunctions are analyzed — a resume anywhere else already demoted.
    tail_resume_set = (_tail_resume_ids(f) if info.scope_role == "case"
                       else frozenset())

    # MODULE CONSTANTS (increment 12): reads of __module_init-declared
    # names with no local binding load from @mx_g_<name>; inside
    # __module_init itself the declared names' defs STORE there (the
    # global is their storage class).
    global_reads = set(info.global_reads)
    init_globals = set(info.init_globals) if f.name == _MODULE_INIT else set()

    def global_slot(n: str) -> str:
        mod.globals_used[n] = gtable.kind(n)
        return _mx_global(n)

    def env_fields(lname: str) -> Tuple[Tuple[str, str], ...]:
        """The env struct layout of a lambda: (capture, kind) in list
        order; mutable captures carry the cell:ELEM marker (the field
        holds the cell POINTER)."""
        ccs = cells.cap_cells.get(lname, set())
        return tuple((cn, _cell_marked(closures.cell_kind(lname, cn))
                      if cn in ccs else closures.cell_kind(lname, cn))
                     for cn in closures.targets.get(lname, ()))

    def scope_fields(site: str) -> Tuple[Tuple[str, str], ...]:
        """The env struct layout of a handle site, cell markers included."""
        scs = cells.scope_cells.get(site, set())
        return tuple((n, _cell_marked(scopes.cell_kind(site, n))
                      if n in scs else scopes.cell_kind(site, n))
                     for n in scopes.env_fields.get(site, ()))

    slots = [n for n in _compute_slots(info, kinds)
             if n not in cellset and n not in init_globals]
    slotset = set(slots)
    # Aggregate variables (struct / enum / closure kinds): each owns storage.
    agg_vars = sorted(n for n in info.def_count if _is_agg(kind(n)))
    aggset = set(agg_vars)
    # Heap-backed struct variables (@global alloc_struct defs): storage is an
    # entry-block malloc'd block instead of an alloca, freed on every ret.
    heap_vars = sorted(n for n in agg_vars if n in info.global_alloc_vars)
    heapset = set(heap_vars)
    # A word-uniform lambda NEVER uses the sret convention: its aggregate
    # return travels as a boundary-box pointer word (increment 16), so the
    # native signature stays `i64 (ptr env, i64 args...)`.
    #
    # BOUNDARY-WORD RETURN (increment 17, work item 1): a handle-scope
    # subfunction with an aggregate result does the same.  Its value ALWAYS
    # continues as a boundary word — the body thunk and the dispatcher used
    # to malloc a box, sret-fill it and return its pointer — so returning
    # the word directly lets the shared to_word do the boxing, which the
    # double-box elision can then skip entirely when the returned value
    # already IS an immortal box (the fold-shaped `resume(...)` case).
    # Scope members are only ever called by their own site's shims, so no
    # other caller's convention is involved.
    is_boundary_ret = info.is_scope_member and _is_agg(sig.ret)
    sret = _is_agg(sig.ret) and not is_uniform_lambda and not is_boundary_ret

    counter = 0

    def fresh() -> str:
        nonlocal counter
        v = f"%t{counter}"
        counter += 1
        return v

    valmap: Dict[str, str] = {}

    def slot_ref(n: str) -> str:
        return f"%slot.{_sanitize(n)}"

    def cellp_ref(n: str) -> str:
        return f"%cellp.{_sanitize(n)}"

    def strown_ref(n: str) -> str:
        return f"%strown.{_sanitize(n)}"

    def bp_ref(n: str) -> str:
        return f"%bp.{_sanitize(n)}"

    def struct_ref(n: str) -> str:
        """The storage pointer for an aggregate variable (alloca or heap
        block).  Copy-elided params alias the caller's storage directly;
        box-view variables have no storage of their own (their pointer slot
        is read through use()) and must never be written through."""
        if n in boxview:
            raise _Unsupported(
                f"write through box-view variable {n!r} (elision invariant)")
        if n in elided_params:
            return f"%a.{_sanitize(n)}"
        if n in heapset:
            return f"%hv.{_sanitize(n)}"
        return f"%sv.{_sanitize(n)}"

    def use(name: str, lines: List[str]) -> str:
        if name in cellset:
            # Mutable capture: load through the shared cell (auto-deref,
            # exactly mir_interp._lookup on an MxCell slot).
            v = fresh()
            lines.append(f"  {v} = load {llty(name)}, ptr {cellp_ref(name)}"
                         f"  ; cell read: {name}")
            return v
        if name in init_globals:
            v = fresh()
            lines.append(
                f"  {v} = load {llty(name)}, ptr {global_slot(name)}"
                f"  ; module constant {name}")
            return v
        if name in boxview:
            v = fresh()
            lines.append(f"  {v} = load ptr, ptr {bp_ref(name)}")
            return v
        if name in aggset:
            return struct_ref(name)
        if name in slotset:
            v = fresh()
            lines.append(f"  {v} = load {llty(name)}, ptr {slot_ref(name)}")
            return v
        if name in global_reads and name not in info.def_count:
            v = fresh()
            lines.append(
                f"  {v} = load {llty(name)}, ptr {global_slot(name)}"
                f"  ; module constant {name}")
            return v
        try:
            return valmap[name]
        except KeyError:
            raise _Unsupported(f"use of {name!r} before its definition")

    def setval(name: str, v: str, lines: List[str]) -> None:
        if name in cellset:
            # Write THROUGH the shared cell: every frame that captured the
            # cell observes the new value (mir_interp's "let" on a cell).
            lines.append(f"  store {llty(name)} {v}, ptr {cellp_ref(name)}"
                         f"  ; cell write: {name}")
            return
        if name in init_globals:
            lines.append(
                f"  store {llty(name)} {v}, ptr {global_slot(name)}"
                f"  ; module constant {name}: published")
            return
        if name in aggset:
            raise _Unsupported(f"scalar assignment to aggregate variable {name!r}")
        if name in slotset:
            lines.append(f"  store {llty(name)} {v}, ptr {slot_ref(name)}")
        else:
            valmap[name] = v

    def scalar_const(name: str, value: Any) -> str:
        k = kind(name)
        if _is_vec(k):
            if value is None:
                return "null"  # an uninitialized Vec slot (const None)
            raise _Unsupported(f"non-None constant for Vec value {name!r}")
        if _is_fvec(k):
            if value is None:
                return "null"  # an uninitialized vector slot (const None)
            raise _Unsupported(
                f"non-None constant for vector value {name!r}")
        if k == PTR:
            if value is None:
                return "null"  # the `null` pointer literal
            raise _Unsupported(f"non-null constant for rawptr value {name!r}")
        if k == F64:
            if value is None:
                value = 0.0
            if isinstance(value, bool):
                value = float(value)
            return _fmt_f64(float(value))
        if k == STR:
            if isinstance(value, str):
                return mod.intern_string(value)
            raise _Unsupported(f"non-string constant for string value {name!r}")
        if value is None:
            return "0"
        if isinstance(value, bool):
            return str(int(value))
        if isinstance(value, float):
            return str(int(value))
        return str(int(value))

    def gep(sname: str, base_ptr: str, fname: str, lines: List[str]) -> str:
        idx = structs.fields[sname].index(fname)
        v = fresh()
        lines.append(
            f"  {v} = getelementptr inbounds %struct.{_sanitize(sname)}, "
            f"ptr {base_ptr}, i32 0, i32 {idx}")
        return v

    def agg_copy(tyname: str, src_ptr: str, dst_ptr: str, lines: List[str]) -> None:
        v = fresh()
        lines.append(f"  {v} = load {tyname}, ptr {src_ptr}")
        lines.append(f"  store {tyname} {v}, ptr {dst_ptr}")

    def enum_gep(ename: str, base_ptr: str, lines: List[str], *,
                 payload: Optional[int] = None) -> str:
        """GEP to the tag (payload=None) or a payload slot of a tagged union."""
        v = fresh()
        if payload is None:
            lines.append(
                f"  {v} = getelementptr inbounds {_enum_llname(ename)}, "
                f"ptr {base_ptr}, i32 0, i32 0")
        else:
            lines.append(
                f"  {v} = getelementptr inbounds {_enum_llname(ename)}, "
                f"ptr {base_ptr}, i32 0, i32 1, i32 {payload}")
        return v

    def builtin_of(callee: str, cargs: Tuple[str, ...]) -> Optional[str]:
        """The native builtin name a call op lowers to, or None (used both
        by the vec-escape analysis and nowhere else; mirrors the emission
        dispatch order below: locals shadow, then trait resolution, then
        plain builtins)."""
        if callee in info.def_count:
            return None  # closure call through a local
        if callee.startswith(TRAIT_CALL_PREFIX):
            if not cargs:
                return None
            res, target = _resolve_trait_call(
                callee[len(TRAIT_CALL_PREFIX):], kind(cargs[0]), traits,
                module_names, assume_final=True)
            return target if res == "builtin" else None
        if callee.startswith(STATIC_CALL_PREFIX):
            res, target = _resolve_static_call(callee, traits, module_names)
            return target if res == "builtin" else None
        bname = _builtin_name(callee, module_names)
        if bname in _NATIVE_RT_CALLS or bname == "as_ptr":
            return bname
        return None

    # Vecs provably dead at frame exit (see _provably_dead_vecs): freed on
    # every ret path.  All other Vec.new results LEAK BY DESIGN — identity
    # semantics means the pointer may be shared anywhere it escaped to, so
    # no free can be proven unique (same contract as boxes/heap envs).
    vec_free_vars = _provably_dead_vecs(
        f, kinds, builtin_of,
        retained=cellset | init_globals | global_reads)
    vec_free_set = set(vec_free_vars)

    # Fixed-vector static lengths (increment 11): var -> element count when
    # every def is provably that long, _FLEN_DYN otherwise.  Drives the
    # inline `<N x double>` / `<N x i64>` fast path for element-wise binops.
    fvec_lens = _fvec_static_lens(f, kinds, info)
    # A cell-backed vector can be REASSIGNED from another function through
    # the shared cell (and a module-constant one read here has its defs in
    # __module_init), so local static-length facts about them are unsound.
    for n in cellset | global_reads:
        if _is_fvec(kind(n)):
            fvec_lens[n] = _FLEN_DYN

    def fvec_len_of(n: str) -> Optional[int]:
        """The usable static length of a vector variable: an int in
        [1, _FLEN_MAX], or None (unknown / dynamic / out of range)."""
        v = fvec_lens.get(n)
        return v if v is not None and 1 <= v <= _FLEN_MAX else None

    # OWNED STRINGS (increment 8): produced strings this frame provably
    # owns; each gets a null-initialized shadow slot holding the pointer to
    # free at the next redefinition / at frame exit (see _owned_strings).
    # The temp sets distinguish transfer copies carrying interned LITERALS
    # (record null: constants are never freed) from ones carrying fresh
    # produced pointers.
    owned_strs, str_lit_temps, str_prod_temps = _owned_strings(
        f, kinds, info, builtin_of, module_names)
    # Cell-backed and module-constant strings are shared beyond this
    # frame's view: never freeable here (leak by design).
    _str_retained = cellset | init_globals | global_reads
    owned_strs = [n for n in owned_strs if n not in _str_retained]
    str_lit_temps -= _str_retained
    str_prod_temps -= _str_retained
    owned_str_set = set(owned_strs)

    # UNIQUE BOXES (increment 8): entry-block make_variant sites whose
    # payload boxes are uniquely owned by this frame (never copied out of
    # it, see _unique_box_enums); their box pointers are recorded at the
    # site and freed on every ret path.
    unique_box_sites = _unique_box_enums(f, kinds, info)
    unique_box_regs: List[Tuple[str, str, int]] = []  # (reg, dst, slot)

    def passed_to_rebound(v: str) -> bool:
        """True when v is ever passed as an aggregate call argument at a
        position the resolved callee WRITES BACK (a rebound struct param
        copies out through the caller's pointer on ret) — or to a callee
        whose write-back behavior is unknown.  Used to keep the two copy
        elisions away from storage a callee may write."""
        for b in f.blocks:
            for op in b.ops:
                if op[0] != "let" or len(op) != 4 or op[2][0] != "call":
                    continue
                _d, rhs2, cargs = op[1], op[2], op[3]
                if v not in cargs:
                    continue
                callee2 = rhs2[1]
                if callee2 in info.def_count:
                    # Closure call: only @mut lambda params write back
                    # (interpreter closure-call parity). Union the member
                    # lambdas' write-back positions; unknown members assume
                    # the worst.
                    ck2 = kind(callee2)
                    if not _is_closure(ck2) or _is_dyn_closure(ck2):
                        return True
                    for m in _closure_members(ck2):
                        rb = writeback_map.get(m)
                        if rb is None:
                            return True
                        # closure args map 1:1 onto member param positions
                        for i, a in enumerate(cargs):
                            if a == v and i in rb:
                                return True
                    continue
                if builtin_of(callee2, tuple(cargs)) is not None:
                    continue  # native runtime builtins never write back
                if callee2.startswith(TRAIT_CALL_PREFIX):
                    res2, tgt2 = _resolve_trait_call(
                        callee2[len(TRAIT_CALL_PREFIX):],
                        kind(cargs[0]) if cargs else I64, traits,
                        module_names, assume_final=True)
                    target = tgt2 if res2 == "func" else None
                elif callee2.startswith(STATIC_CALL_PREFIX):
                    res2, tgt2 = _resolve_static_call(
                        callee2, traits, module_names)
                    target = tgt2 if res2 == "func" else None
                else:
                    target = callee2
                rebound = writeback_map.get(target) if target else None
                if rebound is None:
                    return True  # unknown callee: assume the worst
                for i, a in enumerate(cargs):
                    if a == v and i in rebound:
                        return True
        return False

    def readonly_indirect_arg(members: Sequence[str], pos: int) -> bool:
        """READ-ONLY AGGREGATE ARGUMENT (increment 17, work item 2): true
        when an aggregate argument at parameter position `pos` of an
        INDIRECT closure call may pass a pointer to the caller's existing
        storage instead of a fresh boundary box.

        This reuses the direct-call copy-elision reasoning (writeback_map)
        rather than repeating it: the box exists only to give the callee
        something it may not write through and that outlives the call, and
        NEITHER worry applies here.

          * NO WRITER.  writeback_map records exactly the positions a
            callee copies back out through the caller's pointer.  Every
            statically-possible callee (a pinned lambda is one member; a
            dynamic kind is all of them) must be word-uniform and must
            leave this position out of its write-back set.  Word-uniform
            eligibility already rejects @mut aggregate params outright
            (_word_eligible), so this is a belt-and-braces re-check, not a
            new assumption.  A member with no entry in the map is unknown
            and boxes.
          * NO LIFETIME GAP.  Unlike an effect boundary, an indirect call
            is an ordinary synchronous call on THIS stack: our storage
            provably outlives the callee's frame.  The callee cannot
            re-export the pointer either — a parameter is never a
            bbox_view, so passing it onward to a perform re-boxes a copy.

        The receiver is oblivious: it decodes the word to a `ptr` and then
        byval-copies (or elide-copy reads through it) exactly as before."""
        for m in members:
            if m not in word_uniform:
                return False
            rb = writeback_map.get(m)
            if rb is None or pos in rb:
                return False
        return True

    # COPY ELISION (a): an aggregate param never redefined needs no entry
    # byval copy — nothing ever writes its storage (its only def is the
    # param itself; rebound params keep the copy + write-back, and a struct
    # param passed onward to a rebound callee position keeps the copy so
    # the callee's write-back hits OUR copy, exactly like the interpreter
    # writing back into our binding, not our caller's).
    elided_params: Set[str] = set()
    for p in info.params:
        pk = kinds.get(p, I64)
        if not _is_agg(pk) or info.def_count.get(p, 0) != 1:
            continue
        if _is_struct(pk) and passed_to_rebound(p):
            continue
        elided_params.add(p)

    # COPY ELISION (b): a variant_field result that is only ever READ can
    # be a BOX VIEW — a pointer slot aliasing the write-once box instead of
    # an aggregate copied out of it.  Sound because boxes are write-once
    # (no native code path mutates a filled box), the view's every use is a
    # pointer READ (all aggregate uses read through use()), and the only
    # writer a read could summon — a callee's struct write-back — is
    # excluded exactly like elision (a).  Copies of a view alias the same
    # box (pointer copy), recursively elidable under the same conditions.
    def read_only_agg(v: str) -> bool:
        vk = kinds.get(v, I64)
        if info.def_count.get(v, 0) != 1 or v in info.params:
            return False
        if _is_struct(vk) and passed_to_rebound(v):
            return False
        return True

    # COPY ELISION (c): BOUNDARY-BOX VIEWS (increment 17).  A value whose
    # single def RECEIVES a boundary word — a perform result, a resume
    # result, a handle value, or the result of a word-uniform indirect
    # closure call — is handed a pointer to a box the producer freshly
    # malloc'd, filled once and never frees.  Instead of copying out of it,
    # keep the pointer: the value becomes a box view exactly like (b).
    #
    # SAFETY (the write-once box invariant, stated precisely):
    #   1. IMMORTAL.  Every box reaching those four channels is malloc'd by
    #      to_word, by a scope body/case thunk, by a dynamic-default thunk
    #      or by a word-uniform lambda's ret — and NO emitted path ever
    #      frees one (emit_frees releases only @global blocks, provably
    #      local Vecs, unique enum payload boxes and owned strings).  So
    #      the pointer cannot dangle, not even when a coroutine stack is
    #      torn down under an abort.
    #   2. WRITE-ONCE.  The producer fills the box before the word leaves
    #      it and nothing writes it afterwards: handler cases are scope
    #      members, which never copy out through a param pointer, and the
    #      word-uniform ABI refuses @mut aggregate params outright.
    #   3. UNCHANGED BY US.  read_only_agg pins the receiver to a single
    #      def that never reaches a callee write-back position, so this
    #      frame cannot write the value either.
    # Hence the box's contents equal this value forever, and reading
    # through the pointer is observationally identical to reading a copy.
    def _is_boundary_result(op: tuple) -> bool:
        """True when op's destination receives an IMMORTAL boundary-box
        word.  A perform of an op no scope lists is NOT one: it lowers to
        a direct sret call into our own storage instead."""
        if op[0] == "perform":
            return info.default_performs.get((op[2], op[3])) is None
        if op[0] != "let" or len(op) != 4:
            return False
        rhs = op[2]
        if rhs[0] in ("resume", "handle_scope"):
            return True
        if rhs[0] == "call" and rhs[1] in info.def_count:
            ck = kinds.get(rhs[1], I64)
            if _is_closure(ck):
                # Only the word-uniform indirect path returns a box; a
                # pinned closure call is sret into our own storage.
                return _is_dyn_closure(ck) or any(
                    m in word_uniform for m in _closure_members(ck))
        return False

    # bbox_view ⊆ boxview: the views whose box is provably IMMORTAL, so
    # their pointer may also be handed straight back out to another
    # boundary (see to_word).  Views from (b) alias enum PAYLOAD boxes,
    # which _unique_box_enums may free at frame exit — they read through
    # the box but must never re-export its pointer across a boundary.
    bbox_view: Set[str] = set()
    for b in f.blocks:
        for op in b.ops:
            if not _is_boundary_result(op):
                continue
            bdst = op[1]
            if not _is_agg(kinds.get(bdst, I64)):
                continue
            if bdst in cellset or bdst in init_globals or bdst in heapset:
                continue
            if read_only_agg(bdst):
                bbox_view.add(bdst)

    boxview: Set[str] = set(bbox_view)
    bv_changed = True
    while bv_changed:
        bv_changed = False
        for b in f.blocks:
            for op in b.ops:
                if op[0] != "let" or len(op) != 4:
                    continue
                _, bdst, brhs, bargs = op
                if bdst in boxview:
                    continue
                if brhs[0] == "variant_field" and len(brhs) > 2 and bargs:
                    bk = kinds.get(bargs[0], I64)
                    if not (_is_enum(bk) and _is_agg(kinds.get(bdst, I64))):
                        continue
                    bref = _enum_refinement(bk) or {}
                    bslots = bref.get(brhs[2])
                    if bslots is not None and brhs[1] < len(bslots) \
                            and _is_agg(bslots[brhs[1]]) \
                            and read_only_agg(bdst):
                        boxview.add(bdst)
                        bv_changed = True
                elif brhs[0] == "copy" and bargs and bargs[0] in boxview:
                    if _is_agg(kinds.get(bdst, I64)) and read_only_agg(bdst) \
                            and bdst not in info.dead_results:
                        boxview.add(bdst)
                        # A copy of a view aliases the SAME box, so it
                        # inherits that box's immortality (and nothing
                        # else): a copy of a payload view stays payload.
                        if bargs[0] in bbox_view:
                            bbox_view.add(bdst)
                        bv_changed = True

    # COPY-IN/COPY-OUT struct params: the interpreter WRITES BACK a struct
    # argument when a BY-REFERENCE parameter (MirFunc.mut_params: declared
    # @mut, or a method's `self` receiver) is rebound by the callee
    # (mir_interp._write_back_struct_args — `self.field = ...` methods
    # mutate the caller's binding).  Natively the caller already passes its
    # storage pointer, so the callee copies the final param value back
    # through it on every ret path.  Statically "rebinds anywhere"
    # over-approximates the interpreter's per-execution identity test, but
    # a not-taken rebind path writes back the unchanged aggregate —
    # observationally a no-op.  Plain params keep value semantics: rebinding
    # them stays callee-local in BOTH engines.  Lambdas write back exactly
    # their @mut-declared params (interpreter closure-call parity).  Only
    # struct kinds write back (enums/closures never do).  (Handle-scope
    # subfunctions never copy out: the interpreter calls them without its
    # write-back path.)
    # ALL @mut struct params copy out (not just locally-rebound ones): the
    # interpreter's identity test also fires when a NESTED call wrote back
    # into this frame's binding (e.g. a lambda passing its @mut param on to
    # bump) — statically that is "any def could have changed it", and an
    # untouched param's copy-out writes back the unchanged aggregate, a
    # no-op.
    _mut_set = set(getattr(f, "mut_params", ()) or ())
    writeback_params = [] if info.is_scope_member else [
        p for p in info.params
        if p in _mut_set and _is_struct(kinds.get(p, I64))]

    # LOOP-INVARIANT BOXES (increment 17, work item 3).  A boundary box is
    # a write-once copy of a value; when that value provably does not
    # change, every iteration's box holds the same bytes, so ONE box can
    # serve them all.  The cheap proof, deliberately narrow:
    #   * the value's ONLY def is a block-0 op (or it is a parameter), and
    #     block 0 is not itself in a CFG cycle, so it is filled exactly
    #     once per invocation and block 0 dominates every other block;
    #   * def_count == 1 and passed_to_rebound is false, so nothing —
    #     neither this frame nor a callee's write-back — ever writes its
    #     storage after that def (the same invariant elisions (a)/(b) rest
    #     on);
    #   * none of its boxing sites is in block 0 itself, so the hoisted
    #     register (emitted at the END of block 0) dominates all of them;
    #   * at least one boxing site is inside a cycle, i.e. there is
    #     actually a per-iteration malloc to remove.
    # Sharing one box is invisible: boxes are immortal and never written
    # after the fill, so two receivers holding the same pointer read the
    # same unchanging bytes they would have read from two identical
    # copies.  Anything not matching keeps its per-site box.
    _cycle_blocks = _blocks_in_cycles(f)

    def _boxing_sites(b) -> List[str]:
        """Values this block turns into a boundary box via to_word.  Only
        the effect-boundary senders box now — indirect-call aggregate
        arguments pass the caller's storage pointer instead."""
        out: List[str] = []
        for op in b.ops:
            if op[0] == "perform" and info.default_performs.get(
                    (op[2], op[3])) is None:
                out.extend(a for a in op[4] if isinstance(a, str))
            elif op[0] == "let" and len(op) == 4 and op[2][0] == "resume" \
                    and len(op[3]) > 1:
                out.append(op[3][1])
        if b.term and b.term[0] == "ret" and (
                is_uniform_lambda or is_boundary_ret):
            out.append(b.term[1])
        return out

    hoist_box: Set[str] = set()
    if f.blocks and 0 not in _cycle_blocks:
        block0_defs = {op[1] for op in f.blocks[0].ops
                       if op[0] in ("let", "perform") and len(op) >= 2
                       and isinstance(op[1], str)}
        in_block0 = set(_boxing_sites(f.blocks[0]))
        for bi, b in enumerate(f.blocks):
            if bi == 0 or bi not in _cycle_blocks:
                continue
            for v in _boxing_sites(b):
                if v in hoist_box or v in in_block0:
                    continue
                vk = kinds.get(v, I64)
                if not _is_agg(vk) or _kind_size(vk, structs, variants) is None:
                    continue
                if v in boxview or v in cellset or v in init_globals:
                    continue  # already a pointer, or not plain storage
                if not (v in info.params or v in block0_defs):
                    continue
                if info.def_count.get(v, 0) != 1 or passed_to_rebound(v):
                    continue
                hoist_box.add(v)

    # value -> the i64 word of its hoisted box (filled at the end of bb0).
    hoisted_word: Dict[str, str] = {}

    def emit_hoisted_box(v: str, lines: List[str]) -> None:
        k = kinds.get(v, I64)
        size = _kind_size(k, structs, variants)
        if size is None:  # unreachable: hoist_box screens infinite layouts
            return
        mod.uses_malloc = True
        box = fresh()
        lines.append(
            f"  {box} = call ptr @malloc(i64 {max(size, 8)})"
            f"  ; loop-invariant boundary box: {v} ({k}) — {v} never changes "
            "after this point, so one write-once box serves every iteration "
            "(immortal, leaks by design)")
        agg_copy(_agg_ty(k), use(v, lines), box, lines)
        w = fresh()
        lines.append(f"  {w} = ptrtoint ptr {box} to i64")
        hoisted_word[v] = w

    def emit_writebacks(lines: List[str]) -> None:
        for p in writeback_params:
            agg_copy(_agg_ty(kinds.get(p, I64)), struct_ref(p),
                     f"%a.{_sanitize(p)}", lines)
            lines.append(f"  ; ^ copy-out: rebound struct param {p} "
                         "written back to the caller")

    def emit_frees(lines: List[str]) -> None:
        """Free every heap-backed @global struct block and every provably
        frame-local Vec (called on ret paths).

        Sound because the mallocs unconditionally happen in the entry block
        (exactly once per invocation) and the pointers provably never
        escape this frame (value semantics for structs; the vec escape
        analysis for Vecs — see module docstring)."""
        for n in heap_vars:
            lines.append(f"  call void @free(ptr {struct_ref(n)})"
                         f"  ; @global struct {n}: end of frame")
        for n in vec_free_vars:
            mod.runtime_syms.add("mx_vec_free")
            lines.append(f"  call void @mx_vec_free(ptr {use(n, lines)})"
                         f"  ; local Vec {n}: provably non-escaping")
        for (reg, bdst, bslot) in unique_box_regs:
            lines.append(f"  call void @free(ptr {reg})"
                         f"  ; unique box: {bdst} slot {bslot} (this frame "
                         "is the sole owner)")
        for n in owned_strs:
            mod.runtime_syms.add("mx_str_free")
            ov = fresh()
            lines.append(f"  {ov} = load ptr, ptr {strown_ref(n)}")
            lines.append(f"  call void @mx_str_free(ptr {ov})"
                         f"  ; owned string {n}: freed at frame exit")

    def owned_str_update(v: str, is_literal: bool, lines: List[str]) -> None:
        """After a def of an owned string variable: free the previously
        owned pointer (its last alias was just overwritten) and record the
        new one — null for literal defs (interned constants are never
        freed), the fresh produced pointer otherwise."""
        mod.runtime_syms.add("mx_str_free")
        old = fresh()
        lines.append(f"  {old} = load ptr, ptr {strown_ref(v)}")
        lines.append(f"  call void @mx_str_free(ptr {old})"
                     f"  ; owned string {v}: previous value freed")
        if is_literal:
            lines.append(f"  store ptr null, ptr {strown_ref(v)}"
                         f"  ; owned string {v}: literal (never freed)")
        else:
            cur = use(v, lines)
            lines.append(f"  store ptr {cur}, ptr {strown_ref(v)}"
                         f"  ; owned string {v}: fresh malloc now owned")

    def to_word(k: str, v: str, lines: List[str],
                src: Optional[str] = None) -> str:
        """Reinterpret a value of word kind k as the opaque i64 element word
        the native Vec ABI stores (the runtime never inspects elements).
        Aggregate kinds box: a fresh malloc'd write-once copy, its
        pointer as the word.  Three word positions do this — effect-
        boundary senders (increment 14), indirect closure-call
        arguments/returns (increment 16) and Vec element slots
        (increment 20); every other word position still demotes
        aggregates in the checks
        (immortal, leaks by design — it can never dangle across coroutine
        switches or across an indirect call).

        DOUBLE-BOX ELISION (increment 17): when `src` names a value that is
        ALREADY an immortal boundary box (a bbox_view — see copy elision
        (c)), the pointer passes straight through instead of malloc'ing a
        second box and copying the same bytes into it.  Sound because the
        box is immortal, write-once and unmodified since it was filled, so
        the copy would be byte-identical and just as long-lived."""
        if _is_agg(k):
            if src is not None and src in bbox_view:
                t = fresh()
                lines.append(
                    f"  {t} = ptrtoint ptr {v} to i64"
                    f"  ; elide-box: {src} already IS an immortal write-once "
                    f"boundary box ({k}); its pointer passes through")
                return t
            if src is not None and src in hoisted_word:
                lines.append(
                    f"  ; elide-box: {src} ({k}) reuses its loop-invariant "
                    "box, filled once in the entry block")
                return hoisted_word[src]
            size = _kind_size(k, structs, variants)
            if size is None:  # unreachable: checks demote infinite layouts
                raise _Unsupported(f"boundary box of {k} has infinite layout")
            mod.uses_malloc = True
            box = fresh()
            lines.append(f"  {box} = call ptr @malloc(i64 {max(size, 8)})"
                         f"  ; boundary box: {k} (write-once, leaks by design)")
            agg_copy(_agg_ty(k), v, box, lines)
            t = fresh()
            lines.append(f"  {t} = ptrtoint ptr {box} to i64")
            return t
        if k == I64:
            return v
        t = fresh()
        if k == F64:
            lines.append(f"  {t} = bitcast double {v} to i64")
        else:  # str / vec pointers
            lines.append(f"  {t} = ptrtoint ptr {v} to i64")
        return t

    def from_word(k: str, v: str, lines: List[str]) -> str:
        """Inverse of to_word: element word back to its typed value."""
        if k == I64:
            return v
        t = fresh()
        if k == F64:
            lines.append(f"  {t} = bitcast i64 {v} to double")
        else:
            lines.append(f"  {t} = inttoptr i64 {v} to ptr")
        return t

    def word_into(dst: str, w: str, lines: List[str]) -> None:
        """Decode an effect-boundary word into dst per dst's kind: word
        kinds via from_word; aggregate kinds copy OUT of the sender's
        boundary box into dst's own storage (the box stays immortal and
        write-once — value semantics at both edges, increment 14).

        A bbox_view destination (copy elision (c)) skips the copy-out
        entirely: it KEEPS the producer's box pointer and reads through
        it, which is where the double-boxing on aggregate resume results
        disappears."""
        k = kind(dst)
        if dst in bbox_view:
            p = fresh()
            lines.append(f"  {p} = inttoptr i64 {w} to ptr")
            lines.append(
                f"  store ptr {p}, ptr {bp_ref(dst)}"
                f"  ; elide-copy: {dst} views the producer's write-once "
                f"boundary box ({k}) instead of copying out of it")
        elif _is_agg(k):
            p = fresh()
            lines.append(f"  {p} = inttoptr i64 {w} to ptr"
                         f"  ; boundary box: {k}")
            agg_copy(_agg_ty(k), p, struct_ref(dst), lines)
        else:
            setval(dst, from_word(k, w, lines), lines)

    def vec_elem_into(dst: str, elem: str, w: str, lines: List[str]) -> None:
        """Decode one Vec/vector ELEMENT WORD into dst's own storage.

        Word kinds unpack with from_word.  An aggregate element word is an
        ELEMENT BOX pointer (increment 20 — pushed/stored by to_word as a
        fresh malloc'd write-once copy): the reader copies the aggregate
        OUT of the box into its own alloca, so both edges keep MIR's value
        semantics while the vec itself keeps mx_vec identity semantics.
        The box is immortal (leaks by design, exactly like an enum payload
        or effect-boundary box), so the copy-out can never read freed
        memory and mx_vec_free on a provably-dead vec — which frees only
        the word buffer — stays sound."""
        if _is_agg(elem):
            p = fresh()
            lines.append(f"  {p} = inttoptr i64 {w} to ptr"
                         f"  ; element box: {elem} (write-once, immortal)")
            agg_copy(_agg_ty(elem), p, struct_ref(dst), lines)
        else:
            setval(dst, from_word(elem, w, lines), lines)

    def emit_direct_call(dst: str, callee: str, opargs: Tuple[str, ...],
                         lines: List[str]) -> None:
        """A direct call to another emitted module function (also the target
        of a statically-resolved trait call)."""
        if callee not in emitted_names:
            raise _Unsupported(f"call to non-emitted function {callee!r}")
        csig = sigs[callee]
        avals = []
        for a, pk in zip(opargs, csig.params):
            # aggregate args pass their storage pointer; the callee
            # byval-copies the aggregate in its entry prelude.
            avals.append(f"{_llparam(pk)} {use(a, lines)}")
        if _is_agg(csig.ret):
            # sret-style: dst's own storage is the result slot.
            if dst not in aggset:
                raise _Unsupported(
                    f"call result {dst!r} not aggregate-kinded for "
                    f"sret call to {callee!r}")
            avals.insert(0, f"ptr {struct_ref(dst)}")
            lines.append(
                f"  call void @{mangle(callee)}({', '.join(avals)})")
        else:
            v = fresh()
            rty = _llscalar(csig.ret)
            lines.append(
                f"  {v} = call {rty} @{mangle(callee)}({', '.join(avals)})")
            setval(dst, v, lines)

    # TBAA access tags for the inline Vec fast paths (metadata nodes !0-!4
    # emitted by _emit_runtime when used): header words and element words
    # live in DISTINCT allocations by the runtime's contract (the header is
    # its own malloc block; data is a separately (re)allocated buffer), so
    # tagging them apart lets clang keep the hoisted len/data loads live
    # across element stores.  Untagged accesses alias both — partial
    # tagging stays sound.
    TBAA_HDR = ", !tbaa !3"
    TBAA_ELEM = ", !tbaa !4"

    # ---- Inline Vec fast paths -------------------------------------------
    # Growable-Vec accessors are the hottest runtime calls the backend
    # emits, and an opaque call per element access blocks every loop
    # optimization clang could otherwise do (register caching, LICM,
    # strength reduction) — measured at 2.6-2.9x vs C on unit-stride loops
    # (benchmarks/diagnostics).  So the COMMON case is emitted inline
    # against the mx_vec header layout (metaxu_rt.c: len @0, cap @8,
    # data @16, contended @24 — 32 bytes) and ANY miss branches to the
    # runtime, which re-runs the full checks in the runtime's canonical
    # order so every diagnostic (NULL receiver, bounds, contended-write,
    # growth/OOM) stays byte-identical to the all-call emission and the
    # interpreter.  get/set/pop/len misses are ALWAYS failures, so their
    # cold blocks call the noreturn mx__vec_*_fail terminators whose
    # narrow memory contract (_RT_ATTRS) keeps hot-loop header loads
    # hoistable across the never-taken branch; push's miss includes the
    # normal growth path and calls the full op.

    def _vec_data_ep(recv: str, idx: str, lines: List[str]) -> str:
        """Address of element idx: data pointer lives at header offset 16."""
        mod.uses_vec_tbaa = True
        dpp, data, ep = fresh(), fresh(), fresh()
        lines.append(
            f"  {dpp} = getelementptr inbounds i8, ptr {recv}, i64 16")
        lines.append(f"  {data} = load ptr, ptr {dpp}, align 8{TBAA_HDR}")
        lines.append(
            f"  {ep} = getelementptr inbounds i64, ptr {data}, i64 {idx}")
        return ep

    def _vec_permit_guard(recv: str, uid: str, tag: str, hot: str, cold: str,
                          lines: List[str]) -> None:
        """Contended-write guard (docs/contention_as_permission.md): hot iff
        the vec never crossed a spawn (relaxed atomic flag at offset 24 is
        0) OR this thread holds a lock permit (direct initial-exec TLS
        load, mirroring mx__vec_write_check's field-not-accessor choice)."""
        mod.uses_tls_permit = True
        cp, cont, contz = fresh(), fresh(), fresh()
        perm_lbl = f"{tag}.perm.{uid}"
        lines.append(
            f"  {cp} = getelementptr inbounds i8, ptr {recv}, i64 24")
        lines.append(
            f"  {cont} = load atomic i64, ptr {cp} monotonic, "
            f"align 8{TBAA_HDR}")
        lines.append(f"  {contz} = icmp eq i64 {cont}, 0")
        lines.append(f"  br i1 {contz}, label %{hot}, label %{perm_lbl}")
        lines.append(f"{perm_lbl}:")
        perm, pok = fresh(), fresh()
        lines.append(
            f"  {perm} = load i64, ptr @mx__tls_write_permit, align 8")
        lines.append(f"  {pok} = icmp ne i64 {perm}, 0")
        lines.append(f"  br i1 {pok}, label %{hot}, label %{cold}")

    def _emit_vec_set_inline(recv: str, idx: str, w: str,
                             lines: List[str]) -> None:
        """In-place Vec element store: inline null + bounds (one unsigned
        compare covers negatives) + contended guard, then a direct store.
        Every miss is a FAILURE (null / out-of-bounds / contended write
        without a permit), so the cold block calls the noreturn
        mx__vec_set_fail terminator — canonical check order, byte-identical
        diagnostic — whose narrow memory contract (writes only
        inaccessible raise state) keeps the surrounding loop's header
        loads hoistable where a call to the full op would clobber them."""
        mod.runtime_syms.add("mx__vec_set_fail")
        uid = fresh()[1:]
        hot, cold = f"vset.hot.{uid}", f"vset.fail.{uid}"
        nn = fresh()
        lines.append(f"  {nn} = icmp eq ptr {recv}, null")
        lines.append(f"  br i1 {nn}, label %{cold}, label %vset.len.{uid}")
        lines.append(f"vset.len.{uid}:")
        ln, inb = fresh(), fresh()
        lines.append(f"  {ln} = load i64, ptr {recv}, align 8{TBAA_HDR}")
        lines.append(f"  {inb} = icmp ult i64 {idx}, {ln}")
        lines.append(f"  br i1 {inb}, label %vset.grd.{uid}, label %{cold}")
        lines.append(f"vset.grd.{uid}:")
        _vec_permit_guard(recv, uid, "vset", hot, cold, lines)
        lines.append(f"{cold}:")
        lines.append(
            f"  call void @mx__vec_set_fail(ptr {recv}, i64 {idx})"
            "  ; miss: canonical checks + diagnostic, never returns")
        lines.append("  unreachable")
        lines.append(f"{hot}:")
        ep = _vec_data_ep(recv, idx, lines)
        lines.append(f"  store i64 {w}, ptr {ep}, align 8{TBAA_ELEM}")

    def emit_rt_builtin(name: str, dst: str, opargs: Tuple[str, ...],
                        lines: List[str]) -> None:
        """Lower an interpreter builtin to its native runtime call
        (metaxu_rt.c, linked by llvm_run).  The consistency check already
        validated arities and kinds; anything off here is a hard error.
        Growable-Vec accessors additionally get inline fast paths (header
        comment above) with the runtime call demoted to the miss branch."""
        if name == "Vec.new":
            mod.runtime_syms.add("mx_vec_new")
            v = fresh()
            note = ("freed on ret paths (provably non-escaping)"
                    if dst in vec_free_set else "leaks by design (may escape)")
            lines.append(f"  {v} = call ptr @mx_vec_new()  ; Vec.new: {note}")
            setval(dst, v, lines)
        elif name == "push":
            recv = use(opargs[0], lines)
            elem = _vec_elem(kind(opargs[0]))
            # An aggregate element becomes an immortal write-once ELEMENT
            # BOX and the slot holds its pointer (to_word); the vec itself
            # keeps its identity semantics, so the push is visible through
            # every alias exactly as in the interpreter.
            w = to_word(elem, use(opargs[1], lines), lines)
            mod.runtime_syms.add("mx_vec_push")
            # Fast path: in-capacity push is a store plus a len bump; the
            # growth path (len == cap) and every failure go to mx_vec_push.
            uid = fresh()[1:]
            hot, cold = f"vpush.hot.{uid}", f"vpush.cold.{uid}"
            done = f"vpush.done.{uid}"
            nn = fresh()
            lines.append(f"  {nn} = icmp eq ptr {recv}, null")
            lines.append(
                f"  br i1 {nn}, label %{cold}, label %vpush.cap.{uid}")
            lines.append(f"vpush.cap.{uid}:")
            ln, cp, cap, full = fresh(), fresh(), fresh(), fresh()
            lines.append(f"  {ln} = load i64, ptr {recv}, align 8{TBAA_HDR}")
            lines.append(
                f"  {cp} = getelementptr inbounds i8, ptr {recv}, i64 8")
            lines.append(f"  {cap} = load i64, ptr {cp}, align 8{TBAA_HDR}")
            lines.append(f"  {full} = icmp eq i64 {ln}, {cap}")
            lines.append(
                f"  br i1 {full}, label %{cold}, label %vpush.grd.{uid}")
            lines.append(f"vpush.grd.{uid}:")
            _vec_permit_guard(recv, uid, "vpush", hot, cold, lines)
            lines.append(f"{hot}:")
            ep = _vec_data_ep(recv, ln, lines)
            lines.append(f"  store i64 {w}, ptr {ep}, align 8{TBAA_ELEM}")
            l1 = fresh()
            lines.append(f"  {l1} = add i64 {ln}, 1")
            lines.append(f"  store i64 {l1}, ptr {recv}, align 8{TBAA_HDR}")
            lines.append(f"  br label %{done}")
            lines.append(f"{cold}:")
            lines.append(f"  call void @mx_vec_push(ptr {recv}, i64 {w})"
                         "  ; miss: growth or canonical diagnostic")
            lines.append(f"  br label %{done}")
            lines.append(f"{done}:")
            setval(dst, "0", lines)  # unit
        elif name == "pop":
            recv = use(opargs[0], lines)
            elem = _vec_elem(kind(opargs[0]))
            # Fast path: non-empty pop is a load plus a len decrement.
            # Every miss is a failure (NULL / empty / contended write
            # without a permit), so the cold block is the noreturn
            # mx__vec_pop_fail terminator (canonical order, byte-identical
            # diagnostic, loop-hoisting-friendly memory contract).
            mod.runtime_syms.add("mx__vec_pop_fail")
            uid = fresh()[1:]
            hot, cold = f"vpop.hot.{uid}", f"vpop.fail.{uid}"
            nn = fresh()
            lines.append(f"  {nn} = icmp eq ptr {recv}, null")
            lines.append(
                f"  br i1 {nn}, label %{cold}, label %vpop.len.{uid}")
            lines.append(f"vpop.len.{uid}:")
            ln, nz = fresh(), fresh()
            lines.append(f"  {ln} = load i64, ptr {recv}, align 8{TBAA_HDR}")
            lines.append(f"  {nz} = icmp ne i64 {ln}, 0")
            lines.append(
                f"  br i1 {nz}, label %vpop.grd.{uid}, label %{cold}")
            lines.append(f"vpop.grd.{uid}:")
            _vec_permit_guard(recv, uid, "vpop", hot, cold, lines)
            lines.append(f"{cold}:")
            lines.append(f"  call void @mx__vec_pop_fail(ptr {recv})"
                         "  ; miss: canonical checks + diagnostic, "
                         "never returns")
            lines.append("  unreachable")
            lines.append(f"{hot}:")
            l1 = fresh()
            lines.append(f"  {l1} = add i64 {ln}, -1")
            ep = _vec_data_ep(recv, l1, lines)
            w = fresh()
            lines.append(f"  {w} = load i64, ptr {ep}, align 8{TBAA_ELEM}")
            lines.append(f"  store i64 {l1}, ptr {recv}, align 8{TBAA_HDR}")
            vec_elem_into(dst, elem, w, lines)
        elif name == "__index_get":
            recv = use(opargs[0], lines)
            rk0 = kind(opargs[0])
            idx = use(opargs[1], lines)
            if _is_fvec(rk0):
                elem = _fvec_elem(rk0)
                mod.runtime_syms.add("mx_fvec_get")
                w = fresh()
                lines.append(
                    f"  {w} = call i64 @mx_fvec_get(ptr {recv}, i64 {idx})")
            else:
                elem = _vec_elem(rk0)
                # Fast path: null + bounds (one unsigned compare covers
                # negatives) then a direct load; reads never take the
                # contended guard (reads are free by design).  Every miss
                # is a failure, so the cold block is the noreturn
                # mx__vec_get_fail terminator (canonical order,
                # byte-identical diagnostic, loop-hoisting-friendly
                # memory contract).
                mod.runtime_syms.add("mx__vec_get_fail")
                uid = fresh()[1:]
                hot, cold = f"vget.hot.{uid}", f"vget.fail.{uid}"
                nn = fresh()
                lines.append(f"  {nn} = icmp eq ptr {recv}, null")
                lines.append(
                    f"  br i1 {nn}, label %{cold}, label %vget.len.{uid}")
                lines.append(f"vget.len.{uid}:")
                ln, inb = fresh(), fresh()
                lines.append(
                    f"  {ln} = load i64, ptr {recv}, align 8{TBAA_HDR}")
                lines.append(f"  {inb} = icmp ult i64 {idx}, {ln}")
                lines.append(
                    f"  br i1 {inb}, label %{hot}, label %{cold}")
                lines.append(f"{cold}:")
                lines.append(
                    f"  call void @mx__vec_get_fail(ptr {recv}, i64 {idx})"
                    "  ; miss: canonical checks + diagnostic, never returns")
                lines.append("  unreachable")
                lines.append(f"{hot}:")
                ep = _vec_data_ep(recv, idx, lines)
                w = fresh()
                lines.append(f"  {w} = load i64, ptr {ep}, align 8{TBAA_ELEM}")
            vec_elem_into(dst, elem, w, lines)
        elif name == "__index_store":
            # Store-back index assignment `place = __index_store(place, i,
            # x)`.  Vec receiver: mx_vec_set mutates the one shared vector
            # in place (bounds-checked abort, interpreter parity) and the
            # result IS the same pointer, so the rebind is a no-op copy.
            # Fixed-vector receiver: mx_fvec_set_copy performs the
            # interpreter's FUNCTIONAL update — blocks are shallow-shared
            # and write-once, so the update copies the block and returns a
            # fresh one for the place rebind (other shares never observe
            # the write); the fresh block leaks by design like every fvec.
            recv = use(opargs[0], lines)
            rk0 = kind(opargs[0])
            idx = use(opargs[1], lines)
            if _is_fvec(rk0):
                w = to_word(_fvec_elem(rk0), use(opargs[2], lines), lines)
                mod.runtime_syms.add("mx_fvec_set_copy")
                v = fresh()
                lines.append(
                    f"  {v} = call ptr @mx_fvec_set_copy(ptr {recv}, "
                    f"i64 {idx}, i64 {w})"
                    "  ; functional update: fresh block, write-once "
                    "sharing preserved")
                setval(dst, v, lines)
            else:
                w = to_word(_vec_elem(rk0), use(opargs[2], lines), lines)
                lines.append("  ; in-place element store (identity "
                             "semantics), inline fast path")
                _emit_vec_set_inline(recv, idx, w, lines)
                setval(dst, recv, lines)
        elif name == "__index_set":
            # In-place element store (Vec receivers only; immutable
            # receivers were demoted at compile time).
            recv = use(opargs[0], lines)
            idx = use(opargs[1], lines)
            w = to_word(_vec_elem(kind(opargs[0])), use(opargs[2], lines),
                        lines)
            _emit_vec_set_inline(recv, idx, w, lines)
            setval(dst, "0", lines)  # unit
        elif name.startswith("Tile."):
            # Tiles (docs/gpu_tiles.md Stage 0): every op is one mx_tile_*
            # call; the `tile:` kind supplies the element type (is_f64
            # flag) and the shapes are already burned into the kinds the
            # consistency check validated.  Blocks are write-once and leak
            # by design, like fvec.
            top = name[len("Tile."):]

            def tflag(k: str) -> str:
                return "1" if _tile_parts(k)[0] == F64 else "0"

            if top in ("zeros", "arange"):
                sym = f"mx_tile_{top}"
                mod.runtime_syms.add(sym)
                r = use(opargs[0], lines)
                c = use(opargs[1], lines)
                v = fresh()
                lines.append(f"  {v} = call ptr @{sym}(i64 {r}, i64 {c})"
                             "  ; tile ctor (write-once, leaks by design)")
                setval(dst, v, lines)
            elif top == "filled":
                elem, _r, _c = _tile_parts(kind(dst))
                r = use(opargs[0], lines)
                c = use(opargs[1], lines)
                w = to_word(elem, use(opargs[2], lines), lines)
                mod.runtime_syms.add("mx_tile_filled")
                v = fresh()
                lines.append(
                    f"  {v} = call ptr @mx_tile_filled(i64 {r}, i64 {c}, "
                    f"i64 {w})  ; tile ctor (write-once, leaks by design)")
                setval(dst, v, lines)
            elif top == "from_vec":
                src = use(opargs[0], lines)
                r = use(opargs[1], lines)
                c = use(opargs[2], lines)
                mod.runtime_syms.add("mx_tile_from_vec")
                v = fresh()
                lines.append(
                    f"  {v} = call ptr @mx_tile_from_vec(ptr {src}, "
                    f"i64 {r}, i64 {c})  ; length-checked copy-in")
                setval(dst, v, lines)
            elif top == "to_vec":
                a = use(opargs[0], lines)
                mod.runtime_syms.add("mx_tile_to_vec")
                v = fresh()
                lines.append(f"  {v} = call ptr @mx_tile_to_vec(ptr {a})"
                             "  ; fresh Vec (escape analysis never frees "
                             "tile-born vecs: leaks by design)")
                setval(dst, v, lines)
            elif top in ("add", "mul", "dot"):
                a = use(opargs[0], lines)
                b = use(opargs[1], lines)
                sym = f"mx_tile_{top}"
                mod.runtime_syms.add(sym)
                v = fresh()
                lines.append(
                    f"  {v} = call ptr @{sym}(ptr {a}, ptr {b}, "
                    f"i64 {tflag(kind(opargs[0]))})")
                setval(dst, v, lines)
            elif top == "scale":
                elem, _r, _c = _tile_parts(kind(opargs[0]))
                a = use(opargs[0], lines)
                w = to_word(elem, use(opargs[1], lines), lines)
                mod.runtime_syms.add("mx_tile_scale")
                v = fresh()
                lines.append(
                    f"  {v} = call ptr @mx_tile_scale(ptr {a}, i64 {w}, "
                    f"i64 {tflag(kind(opargs[0]))})")
                setval(dst, v, lines)
            elif top == "sum":
                elem, _r, _c = _tile_parts(kind(opargs[0]))
                a = use(opargs[0], lines)
                mod.runtime_syms.add("mx_tile_sum")
                w = fresh()
                lines.append(
                    f"  {w} = call i64 @mx_tile_sum(ptr {a}, "
                    f"i64 {tflag(kind(opargs[0]))})"
                    "  ; pinned row-major accumulation (interp parity)")
                vec_elem_into(dst, elem, w, lines)
            elif top == "transpose":
                a = use(opargs[0], lines)
                mod.runtime_syms.add("mx_tile_transpose")
                v = fresh()
                lines.append(f"  {v} = call ptr @mx_tile_transpose(ptr {a})")
                setval(dst, v, lines)
            elif top == "get":
                elem, _r, _c = _tile_parts(kind(opargs[0]))
                a = use(opargs[0], lines)
                i = use(opargs[1], lines)
                j = use(opargs[2], lines)
                mod.runtime_syms.add("mx_tile_get")
                w = fresh()
                lines.append(
                    f"  {w} = call i64 @mx_tile_get(ptr {a}, i64 {i}, "
                    f"i64 {j})  ; bounds raise catchably (interp wording)")
                vec_elem_into(dst, elem, w, lines)
            elif top == "load":
                src = use(opargs[0], lines)
                off = use(opargs[1], lines)
                r = use(opargs[2], lines)
                c = use(opargs[3], lines)
                mod.runtime_syms.add("mx_tile_load")
                v = fresh()
                lines.append(
                    f"  {v} = call ptr @mx_tile_load(ptr {src}, i64 {off}, "
                    f"i64 {r}, i64 {c})  ; range raises catchably")
                setval(dst, v, lines)
            elif top == "load_or":
                elem, _r, _c = _tile_parts(kind(dst))
                src = use(opargs[0], lines)
                off = use(opargs[1], lines)
                r = use(opargs[2], lines)
                c = use(opargs[3], lines)
                w = to_word(elem, use(opargs[4], lines), lines)
                mod.runtime_syms.add("mx_tile_load_or")
                v = fresh()
                lines.append(
                    f"  {v} = call ptr @mx_tile_load_or(ptr {src}, "
                    f"i64 {off}, i64 {r}, i64 {c}, i64 {w})"
                    "  ; masked: out-of-range elements read `other`")
                setval(dst, v, lines)
            elif top in ("store", "store_clipped"):
                tgt = use(opargs[0], lines)
                off = use(opargs[1], lines)
                t = use(opargs[2], lines)
                sym = f"mx_tile_{top}"
                mod.runtime_syms.add(sym)
                lines.append(
                    f"  call void @{sym}(ptr {tgt}, i64 {off}, ptr {t})"
                    "  ; Vec write: contended-write guard applies"
                    + ("" if top == "store"
                       else "; masked: out-of-range writes nothing"))
                setval(dst, "0", lines)  # unit
            elif top == "load_rows":
                src = use(opargs[0], lines)
                off = use(opargs[1], lines)
                stride = use(opargs[2], lines)
                r = use(opargs[3], lines)
                c = use(opargs[4], lines)
                w = to_word(_tile_parts(kind(dst))[0],
                            use(opargs[5], lines), lines)
                mod.runtime_syms.add("mx_tile_load_rows")
                v = fresh()
                lines.append(
                    f"  {v} = call ptr @mx_tile_load_rows(ptr {src}, "
                    f"i64 {off}, i64 {stride}, i64 {r}, i64 {c}, i64 {w})"
                    "  ; strided masked load (2D tile of a matrix)")
                setval(dst, v, lines)
            elif top == "store_rows":
                tgt = use(opargs[0], lines)
                off = use(opargs[1], lines)
                stride = use(opargs[2], lines)
                t = use(opargs[3], lines)
                mod.runtime_syms.add("mx_tile_store_rows")
                lines.append(
                    f"  call void @mx_tile_store_rows(ptr {tgt}, "
                    f"i64 {off}, i64 {stride}, ptr {t})"
                    "  ; strided masked store; contended-write guard "
                    "applies")
                setval(dst, "0", lines)  # unit
            else:  # rows / cols
                a = use(opargs[0], lines)
                sym = f"mx_tile_{top}"
                mod.runtime_syms.add(sym)
                v = fresh()
                lines.append(f"  {v} = call i64 @{sym}(ptr {a})")
                setval(dst, v, lines)
        elif name == "__zip":
            # Virtual value: the comprehension site reads the zipped
            # SOURCES directly and drives mx_fvec_zip_map (the consistency
            # check restricted every use to that shape).
            lines.append(
                f"  ; __zip {', '.join(opargs)} -> {dst}: virtual pair "
                "iterable (materialized by the comprehension site)")
        elif name == "__vec_lit":
            # Fixed-size vector literal (increment 10): one immutable
            # mx_fvec block, filled in place BEFORE the pointer is ever
            # shared (write-once), then shallow-shared like the
            # interpreter's value-semantics MxVector.  Leaks by design.
            n = len(opargs) - 1
            szc = info.const_ints.get(opargs[0])
            if szc is not None and szc != n:
                raise _Unsupported(
                    f"vector literal has {n} elements for a declared size "
                    f"of {szc} (the interpreter rejects it)")
            mod.runtime_syms.add("mx_fvec_new")
            v = fresh()
            lines.append(
                f"  {v} = call ptr @mx_fvec_new(i64 {n})"
                "  ; vector literal (immutable block, leaks by design)")
            elem = _fvec_elem(kind(dst))
            if n:
                mod.runtime_syms.add("mx_fvec_init")
            for i, e in enumerate(opargs[1:]):
                w = to_word(elem, use(e, lines), lines)
                lines.append(
                    f"  call void @mx_fvec_init(ptr {v}, i64 {i}, i64 {w})")
            setval(dst, v, lines)
        elif name == "__vec_zeros":
            # mx_fvec_new zero-fills; the int 0 and float 0.0 words are
            # both all-zero bits, so no fill loop is needed.
            mod.runtime_syms.add("mx_fvec_new")
            nv = use(opargs[0], lines)
            v = fresh()
            lines.append(
                f"  {v} = call ptr @mx_fvec_new(i64 {nv})"
                "  ; zero-filled vector (0 / 0.0 are the all-zero word)")
            setval(dst, v, lines)
        elif name == "__vec_filled":
            elem = _fvec_elem(kind(dst))
            nv = use(opargs[0], lines)
            w = to_word(elem, use(opargs[1], lines), lines)
            mod.runtime_syms.add("mx_fvec_filled")
            v = fresh()
            lines.append(
                f"  {v} = call ptr @mx_fvec_filled(i64 {nv}, i64 {w})")
            setval(dst, v, lines)
        elif name == "__range":
            mod.runtime_syms.add("mx_fvec_range")
            s = use(opargs[0], lines)
            e = use(opargs[1], lines)
            v = fresh()
            lines.append(
                f"  {v} = call ptr @mx_fvec_range(i64 {s}, i64 {e})"
                "  ; range as an int vector (uses restricted to iteration)")
            setval(dst, v, lines)
        elif name == "__vec_dim":
            recv = use(opargs[0], lines)
            v = fresh()
            if info.const_ints.get(opargs[1]) == 0:
                mod.runtime_syms.add("mx_fvec_len")
                lines.append(f"  {v} = call i64 @mx_fvec_len(ptr {recv})")
            else:  # dim 1 (the consistency check pinned 0 or 1)
                isvec = 1 if _is_fvec(_fvec_elem(kind(opargs[0]))) else 0
                mod.runtime_syms.add("mx_fvec_dim")
                lines.append(
                    f"  {v} = call i64 @mx_fvec_dim(ptr {recv}, i64 1, "
                    f"i64 {isvec})")
            setval(dst, v, lines)
        elif name == "__slice_get":
            recv = use(opargs[0], lines)
            mask = 0
            words: List[str] = []
            for bit, a in zip((1, 2, 4), opargs[1:]):
                if a in info.const_nones:
                    words.append("0")  # statically-omitted bound
                else:
                    mask |= bit
                    words.append(use(a, lines))
            mod.runtime_syms.add("mx_fvec_slice")
            v = fresh()
            lines.append(
                f"  {v} = call ptr @mx_fvec_slice(ptr {recv}, "
                f"i64 {words[0]}, i64 {words[1]}, i64 {words[2]}, "
                f"i64 {mask})  ; honest copy, never a view")
            setval(dst, v, lines)
        elif name == "__cast":
            t = info.const_strs.get(opargs[1], "")
            a = use(opargs[0], lines)
            ak = kind(opargs[0])
            if t in ("float", "f32", "f64"):
                if ak == F64:
                    setval(dst, a, lines)  # already a float
                else:
                    v = fresh()
                    lines.append(
                        f"  {v} = sitofp i64 {a} to double  ; `as float`")
                    setval(dst, v, lines)
            elif t in ("int", "i8", "i16", "i32", "i64",
                       "u8", "u16", "u32", "u64"):
                if ak == I64:
                    setval(dst, a, lines)  # already an int
                else:
                    v = fresh()
                    lines.append(
                        f"  {v} = fptosi double {a} to i64"
                        "  ; `as int` (truncates toward zero, like Python)")
                    setval(dst, v, lines)
            else:
                lines.append(
                    f"  ; __cast to {t!r}: static-level reinterpretation "
                    "(identity, interpreter parity)")
                setval(dst, a, lines)
        elif name == "__vec_comprehension":
            nvar, fnvar, itvar = opargs
            if _is_dyn_closure(kind(fnvar)):
                raise _Unsupported(
                    f"comprehension body {fnvar!r} is a dynamic closure")
            lname = _closure_lambda(kind(fnvar))
            if lname not in emitted_names:
                raise _Unsupported(
                    f"comprehension body lambda {lname!r} is not emitted")
            lsig = sigs[lname]
            dek = _fvec_elem(kind(dst))

            def call_body_lambda(tl: List[str],
                                 typed_args: List[Tuple[str, str]],
                                 ret_kind: str) -> None:
                """Thunk-side call of the body lambda, producing `%r` (its
                typed result).  A word-uniform participant is called
                through the word ABI (args encoded, result decoded).
                Aggregates never reach here (a comprehension's element
                kinds are word kinds), and the thunk has no place to copy
                a boundary box out to — so they demote explicitly."""
                if any(_is_agg(k) for (_v, k) in typed_args) \
                        or _is_agg(ret_kind):
                    raise _Unsupported(
                        f"comprehension body lambda {lname!r} has an "
                        "aggregate in its signature")
                if lname in word_uniform:
                    words = []
                    for i, (v, pkk) in enumerate(typed_args):
                        enc, w = _word_encode(v, pkk, f"%wa{i}")
                        tl += enc
                        words.append(f"i64 {w}")
                    argtxt = "".join(f", {w}" for w in words)
                    dec, _rv = _word_decode("%rw", ret_kind, "%r")
                    if dec:
                        tl.append(
                            f"  %rw = call i64 @{mangle(lname)}(ptr %env"
                            f"{argtxt})  ; word-uniform lambda ABI")
                        tl += dec
                    else:
                        tl.append(
                            f"  %r = call i64 @{mangle(lname)}(ptr %env"
                            f"{argtxt})  ; word-uniform lambda ABI")
                else:
                    argtxt = "".join(
                        f", {_llscalar(pkk)} {v}" for (v, pkk) in typed_args)
                    tl.append(
                        f"  %r = call {_llscalar(ret_kind)} "
                        f"@{mangle(lname)}(ptr %env{argtxt})")

            def decode_word(tl: List[str], w: str, ek: str, pk_: str,
                            tag: str) -> str:
                """Thunk-side element decode: word -> the lambda's
                parameter type (i64->f64 is the scalar promotion
                contract)."""
                if pk_ == F64 and ek == I64:
                    tl.append(f"  %e{tag} = sitofp i64 {w} to double")
                    return f"%e{tag}"
                if ek == I64:
                    return w
                if ek == F64:
                    tl.append(f"  %e{tag} = bitcast i64 {w} to double")
                    return f"%e{tag}"
                tl.append(f"  %e{tag} = inttoptr i64 {w} to ptr")
                return f"%e{tag}"

            def encode_result(tl: List[str], rk_: str) -> None:
                """Thunk-side result encode: lambda return -> element
                word of the destination vector."""
                rv = "%r"
                if dek == F64 and rk_ == I64:
                    tl.append("  %rf = sitofp i64 %r to double")
                    rv = "%rf"
                if dek == I64:
                    tl.append(f"  ret i64 {rv}")
                elif dek == F64:
                    tl.append(f"  %rw = bitcast double {rv} to i64")
                    tl.append("  ret i64 %rw")
                else:
                    tl.append(f"  %rw = ptrtoint ptr {rv} to i64")
                    tl.append("  ret i64 %rw")

            def closure_env_of(fv: str) -> str:
                base = use(fv, lines)
                envpp = fresh()
                lines.append(
                    f"  {envpp} = getelementptr inbounds "
                    f"{_CLOSURE_PAIR_TY}, ptr {base}, i32 0, i32 1")
                envv = fresh()
                lines.append(f"  {envv} = load ptr, ptr {envpp}")
                return envv

            nconst = info.const_ints.get(nvar)
            nexp = -1 if nconst is None else nconst
            zsrcs = info.zip_defs.get(itvar)
            if zsrcs is not None:
                # ZIP COMPREHENSION (increment 12): lockstep over the two
                # zip sources via mx_fvec_zip_map, which ABORTS on a
                # length mismatch exactly like the interpreter's strict
                # __zip.  The two-word thunk decodes one element word per
                # source into the lambda's two parameters.
                if len(zsrcs) != 2 or len(lsig.params) != 2:
                    raise _Unsupported(
                        "zip comprehension outside the pair shape")
                eks = [_fvec_elem(kind(s)) for s in zsrcs]
                mod.thunk_seq += 1
                tsym = f"mx.vzth.{mod.thunk_seq}"
                tl = [
                    f"define internal i64 @{tsym}(ptr %env, i64 %wa, "
                    f"i64 %wb) {{  ; zip comprehension thunk: {lname}",
                    "entry:"]
                ea = decode_word(tl, "%wa", eks[0], lsig.params[0], "a")
                eb = decode_word(tl, "%wb", eks[1], lsig.params[1], "b")
                call_body_lambda(
                    tl, [(ea, lsig.params[0]), (eb, lsig.params[1])],
                    lsig.ret)
                encode_result(tl, lsig.ret)
                tl.append("}")
                mod.comp_thunks.setdefault(f.name, []).append("\n".join(tl))
                envv = closure_env_of(fnvar)
                a = use(zsrcs[0], lines)
                b2 = use(zsrcs[1], lines)
                mod.runtime_syms.add("mx_fvec_zip_map")
                v = fresh()
                lines.append(
                    f"  {v} = call ptr @mx_fvec_zip_map(ptr {a}, "
                    f"ptr {b2}, ptr @{tsym}, ptr {envv}, i64 {nexp})"
                    f"  ; zip comprehension via {lname} (aborts on "
                    "length mismatch)")
                setval(dst, v, lines)
            else:
                ek = _fvec_elem(kind(itvar))
                pk_, rk_ = lsig.params[0], lsig.ret
                # Per-site thunk: decode the element word, adapt i64->f64
                # when the body expects floats (the scalar-promotion
                # contract), call the lambda, encode its result as the
                # destination's element word.
                mod.thunk_seq += 1
                tsym = f"mx.vcth.{mod.thunk_seq}"
                tl = [
                    f"define internal i64 @{tsym}(ptr %env, i64 %w) {{"
                    f"  ; comprehension thunk: {lname}",
                    "entry:"]
                ev = decode_word(tl, "%w", ek, pk_, "")
                call_body_lambda(tl, [(ev, pk_)], rk_)
                encode_result(tl, rk_)
                tl.append("}")
                mod.comp_thunks.setdefault(f.name, []).append("\n".join(tl))
                envv = closure_env_of(fnvar)
                src = use(itvar, lines)
                mod.runtime_syms.add("mx_fvec_map")
                v = fresh()
                lines.append(
                    f"  {v} = call ptr @mx_fvec_map(ptr {src}, "
                    f"ptr @{tsym}, ptr {envv}, i64 {nexp})"
                    f"  ; comprehension via {lname}")
                setval(dst, v, lines)
        elif name == "len":
            recv = use(opargs[0], lines)
            rk0 = kind(opargs[0])
            if _is_vec(rk0):
                # Fast path: len is the first header word; the only miss
                # is a NULL receiver, so the cold block is the noreturn
                # mx__vec_len_fail terminator.  Inlining this matters as
                # much as get/set — `i < v.len()` sits in every loop
                # header, and as a visible load clang can hoist or fold
                # it where a call was a full barrier.
                mod.runtime_syms.add("mx__vec_len_fail")
                mod.uses_vec_tbaa = True
                uid = fresh()[1:]
                hot, cold = f"vlen.hot.{uid}", f"vlen.fail.{uid}"
                nn = fresh()
                lines.append(f"  {nn} = icmp eq ptr {recv}, null")
                lines.append(f"  br i1 {nn}, label %{cold}, label %{hot}")
                lines.append(f"{cold}:")
                lines.append(f"  call void @mx__vec_len_fail(ptr {recv})"
                             "  ; miss: NULL-receiver diagnostic, "
                             "never returns")
                lines.append("  unreachable")
                lines.append(f"{hot}:")
                v = fresh()
                lines.append(
                    f"  {v} = load i64, ptr {recv}, align 8{TBAA_HDR}")
            else:
                sym = "mx_fvec_len" if _is_fvec(rk0) else "mx_str_len"
                mod.runtime_syms.add(sym)
                v = fresh()
                lines.append(f"  {v} = call i64 @{sym}(ptr {recv})")
            setval(dst, v, lines)
        elif name in ("to_string", "int_to_str"):
            k = kind(opargs[0])
            a = use(opargs[0], lines)
            if k == STR:
                setval(dst, a, lines)  # to_string of a string is identity
            elif _is_tile(k):
                elem, _r, _c = _tile_parts(k)
                mod.runtime_syms.add("mx_tile_to_str")
                v = fresh()
                lines.append(
                    f"  {v} = call ptr @mx_tile_to_str(ptr {a}, "
                    f"i64 {1 if elem == F64 else 0})"
                    "  ; tile repr (fresh string, leaks by design)")
                setval(dst, v, lines)
            elif _is_fvec(k):
                leaf, depth = _fvec_leaf(k)
                mod.runtime_syms.add("mx_fvec_to_str")
                v = fresh()
                lines.append(
                    f"  {v} = call ptr @mx_fvec_to_str(ptr {a}, "
                    f"i64 {1 if leaf == F64 else 0}, i64 {depth})"
                    "  ; vector repr (fresh string, leaks by design)")
                setval(dst, v, lines)
            else:
                sym = "mx_f64_to_str" if k == F64 else "mx_i64_to_str"
                mod.runtime_syms.add(sym)
                v = fresh()
                snote = ("owned (freed when dead)"
                         if dst in owned_str_set or dst in str_prod_temps
                         else "leaks by design")
                lines.append(
                    f"  {v} = call ptr @{sym}({_llscalar(k)} {a})"
                    f"  ; fresh malloc'd string ({snote})")
                setval(dst, v, lines)
        elif name in _MATH_EXTERNS:
            mod.math_used.add(name)
            a = use(opargs[0], lines)
            v = fresh()
            lines.append(f"  {v} = call double @{name}(double {a})")
            setval(dst, v, lines)
        else:  # unreachable given resolution + consistency
            raise _Unsupported(f"builtin {name!r} has no native lowering")

    def emit_extern_call(name: str, dst: str, opargs: Tuple[str, ...],
                         lines: List[str]) -> None:
        """A direct call to a real extern C symbol (increment 9): the
        interpreter shims these over its simulated checked heap; natively
        they are the actual libc functions with their C signatures."""
        pks, rk_ = _EXTERN_C_SIGS[name]
        avals = [f"{_llscalar(pk)} {use(a, lines)}"
                 for a, pk in zip(opargs, pks)]
        if name in ("malloc", "free"):
            mod.uses_malloc = True  # shares the malloc/free declares
        else:
            mod.extern_c_syms.add(name)
        if name == "free":
            lines.append(f"  call void @free({', '.join(avals)})"
                         "  ; extern C free (program-managed)")
            setval(dst, "0", lines)  # C void -> unit
        elif name == "fclose":
            c = fresh()
            lines.append(f"  {c} = call i32 @fclose({', '.join(avals)})"
                         "  ; extern C")
            v = fresh()
            lines.append(f"  {v} = sext i32 {c} to i64")
            setval(dst, v, lines)
        else:
            v = fresh()
            lines.append(
                f"  {v} = call {_llscalar(rk_)} @{name}({', '.join(avals)})"
                "  ; extern C")
            setval(dst, v, lines)

    # Params are visible from the entry block on: SSA args directly, spilled
    # params through their slot (the store happens in the entry prelude);
    # aggregate params through their own storage (byval-copied in the
    # prelude).  A lambda's captures are entry-defined the same way: scalar
    # captures load into %cap.* registers (or their slot), aggregate captures
    # copy into their own storage.
    for p in info.params:
        if p not in slotset and p not in aggset and p not in cellset:
            valmap[p] = f"%a.{_sanitize(p)}"
    if info.is_lambda or info.is_scope_member:
        for c in info.env_captures:
            if c not in slotset and c not in aggset and c not in cellset:
                valmap[c] = f"%cap.{_sanitize(c)}"

    # One env alloca per stack-env make_closure site, named and created in
    # the entry block (storage must dominate every use; the site itself may
    # sit in a conditional block).  Heap-env lambdas malloc a FRESH env at
    # the site instead (each execution gets its own immortal block).
    env_allocas: Dict[int, str] = {}
    env_entry: List[str] = []
    env_seq = 0
    for b in f.blocks:
        for op in b.ops:
            if op[0] == "let" and len(op) == 4 and op[2][0] == "make_closure":
                lname = op[2][1]
                if lname in closures.heap_env:
                    continue
                name = f"%env.site{env_seq}.{_sanitize(op[1])}"
                env_seq += 1
                env_allocas[id(op)] = name
                env_entry.append(
                    f"  {name} = alloca %env.{_sanitize(lname)}"
                    f"  ; closure env for {op[1]} -> {lname}")
            elif op[0] == "let" and len(op) == 4 \
                    and op[2][0] in ("handle_scope", "try_scope"):
                # One frame alloca per handle/try site: the frame outlives
                # mx_handle / mx_try (both return only after the delimited
                # scope completes, aborts or is caught), so a stack env is
                # always safe here — the body coroutine reads it through a
                # pointer into this parked frame.
                site = op[2][1]
                name = f"%henv.site{env_seq}"
                env_seq += 1
                env_allocas[id(op)] = name
                kindword = "try" if op[2][0] == "try_scope" else "handle"
                env_entry.append(
                    f"  {name} = alloca %henv.{_sanitize(site)}"
                    f"  ; {kindword}-site env for {op[1]}")
    if info.has_perform:
        env_entry.append(
            f"  %perform.args = alloca [{_MAX_EFFECT_ARGS} x i64]"
            "  ; perform argument-word scratch")

    body: List[str] = []
    for bi, b in enumerate(f.blocks):
        lines: List[str] = []
        terminated = False
        for op in b.ops:
            opk = op[0]
            if opk == "params":
                continue
            if opk == "drop":
                if op[1] in heapset:
                    lines.append(f"  ; drop {op[1]} (@global struct: freed on ret paths)")
                else:
                    lines.append(f"  ; drop {op[1]} (scalar/local: frame-owned, no-op)")
                continue
            if opk == "cell_wrap":
                # The slot is cell-backed for the whole frame (entry-block
                # cell), so the wrap itself is a no-op — idempotent, like
                # the interpreter's re-wrap of an existing MxCell.
                lines.append(f"  ; cell_wrap {op[1]}: cell-backed from "
                             "entry (shared one-word heap cell)")
                continue
            if opk == "match_fail":
                # Raise with the interpreter's EXACT wording — the display
                # name is the pre-monomorphization origin, so classify$Int
                # raises "match failure in 'classify': ..." just like the
                # unspecialized reference run, and a catch binds the same
                # bytes. Catchable via the mx_try pad chain like every
                # other failure; with no pad in flight mx_raise is fatal,
                # matching an uncaught InterpError.
                fname = f.origin_name or f.name
                msg = f"match failure in {fname!r}: {op[1]}"
                g = mod.intern_string(msg)
                mod.runtime_syms.add("mx_raise")
                lines.append(f"  call void @mx_raise(ptr {g})"
                             f"  ; match_fail: {op[1]}")
                lines.append("  unreachable")
                terminated = True
                break
            if opk == "promote_matrix":
                # Rebind each named parameter to its promoted form: a flat
                # numeric vector becomes an Mx1 matrix (mx_fvec_promote);
                # anything else passes through untouched (mir_interp's
                # isinstance/all-numbers test, resolved statically from the
                # incoming kind).
                own_ps = sigs[f.name].params
                for pname in (op[1] if len(op) > 1 else ()):
                    pidx = info.params.index(pname)
                    sk = own_ps[pidx] if pidx < len(own_ps) else I64
                    if _is_fvec(sk) and _fvec_elem(sk) in (I64, F64):
                        mod.runtime_syms.add("mx_fvec_promote")
                        cur = use(pname, lines)
                        pv = fresh()
                        lines.append(
                            f"  {pv} = call ptr @mx_fvec_promote(ptr {cur})"
                            f"  ; promote_matrix: flat {sk} -> Mx1 matrix")
                        setval(pname, pv, lines)
                    else:
                        lines.append(
                            f"  ; promote_matrix {pname}: not a flat "
                            "numeric vector (no-op)")
                        setval(pname, use(pname, lines), lines)
                continue
            if opk == "perform":
                dfn = info.default_performs.get((op[2], op[3]))
                if dfn is not None:
                    # No handle_scope in the module lists this op, so no
                    # scope can ever intercept it: the perform IS a direct
                    # call to the declared default (mir_interp's fallback),
                    # ordinary conventions, aggregates and all.
                    lines.append(
                        f"  ; perform {op[2] or '?'}.{op[3]} -> declared "
                        "default (no handle scope in the module lists it)")
                    emit_direct_call(op[1], dfn, tuple(op[4]), lines)
                    continue
                # ("perform", dst, effect, op, args, resume_bb, resume_slot):
                # store the argument words into the scratch array and call
                # mx_perform; the runtime parks this call stack at this
                # exact point and returns the resumed value.
                _, pdst, peffect, pop, pargs = op[0], op[1], op[2], op[3], op[4]
                for i, a in enumerate(pargs):
                    w = to_word(kind(a), use(a, lines), lines, src=a)
                    p = fresh()
                    lines.append(
                        f"  {p} = getelementptr inbounds "
                        f"[{_MAX_EFFECT_ARGS} x i64], ptr %perform.args, "
                        f"i64 0, i64 {i}")
                    lines.append(f"  store i64 {w}, ptr {p}")
                eg = mod.intern_string(peffect)
                og = mod.intern_string(pop)
                ddfn = info.dynamic_default_performs.get((peffect, pop))
                if ddfn is not None:
                    # DYNAMIC DEFAULT ROUTING: this op declares a default
                    # AND some scope lists it, so which one answers is
                    # decided at THIS perform by the runtime's scope stack.
                    # mx_perform_or_default runs the identical innermost-
                    # non-busy lookup and, only where mx_perform would have
                    # aborted, calls the op's default thunk on this stack.
                    if ddfn not in emitted_names:
                        raise _Unsupported(
                            f"dynamic default {ddfn!r} is not emitted")
                    _emit_default_thunk(ddfn, len(pargs), sigs, structs,
                                        variants, mod, f.name)
                    mod.runtime_syms.add("mx_perform_or_default")
                    w = fresh()
                    lines.append(
                        f"  {w} = call i64 @mx_perform_or_default("
                        f"ptr {eg}, ptr {og}, ptr %perform.args, "
                        f"i64 {len(pargs)}, ptr @{_default_thunk_sym(ddfn)}, "
                        f"ptr null)"
                        f"  ; perform {peffect or '?'}.{pop} -> in-scope "
                        f"handler, else the declared default @{mangle(ddfn)}")
                    word_into(pdst, w, lines)
                    continue
                mod.runtime_syms.add("mx_perform")
                w = fresh()
                lines.append(
                    f"  {w} = call i64 @mx_perform(ptr {eg}, ptr {og}, "
                    f"ptr %perform.args, i64 {len(pargs)})"
                    f"  ; perform {peffect or '?'}.{pop}")
                word_into(pdst, w, lines)
                continue
            # opk == "let" (analysis guarantees this)
            _, dst, rhs, opargs = op
            rk = rhs[0]
            if rk == "const" and dst in info.tag_consts:
                # Pattern tag literal: the variant-name string lowers to its
                # module-wide integer tag; the string never reaches native
                # code.
                vname = info.tag_consts[dst]
                tagv = variants.tags.get(vname)
                if tagv is None:
                    raise _Unsupported(f"no tag assigned for variant {vname!r}")
                lines.append(f"  ; tag literal: {vname!r} -> {tagv}")
                setval(dst, str(tagv), lines)
            elif rk in ("const", "const_ty"):
                value = rhs[1] if rk == "const" else None
                setval(dst, scalar_const(dst, value), lines)
            elif rk == "copy":
                if dst in info.dead_results:
                    lines.append(f"  ; dead copy elided: {dst} (never observed)")
                    continue
                if kind(opargs[0]) != kind(dst):
                    raise _Unsupported(
                        f"copy between kinds {kind(opargs[0])} -> {kind(dst)}")
                if dst in boxview:
                    # Box-view alias: the source is itself a box view (the
                    # elision fixpoint only marks copies from views), so
                    # copy the POINTER, not the aggregate.
                    src = use(opargs[0], lines)
                    lines.append(
                        f"  store ptr {src}, ptr {bp_ref(dst)}"
                        f"  ; elide-copy: {dst} aliases the box view "
                        "(read-only result)")
                elif dst in aggset:
                    src = use(opargs[0], lines)
                    agg_copy(_agg_ty(kind(dst)), src, struct_ref(dst), lines)
                else:
                    setval(dst, use(opargs[0], lines), lines)
            elif rk == "binop":
                o = rhs[1]
                l = use(opargs[0], lines)
                r = use(opargs[1], lines)
                is_flt = kind(opargs[0]) == F64
                is_str = kind(opargs[0]) == STR and kind(opargs[1]) == STR
                if any(_is_fvec(kind(x)) for x in (dst, *opargs)):
                    # Element-wise vector arithmetic with scalar
                    # broadcasting.  FAST PATH (increment 11): a FLAT
                    # float/int vector whose operand lengths are statically
                    # known emits real SIMD IR — `<N x double>` / `<N x i64>`
                    # loads straight off the block's word array (layout:
                    # { i64 len, [len x i64] }, elements at byte offset 8,
                    # f64s bitcast-stored, so the f64 view of the same
                    # memory is the identity), one vector arithmetic
                    # instruction, and a store into a fresh mx_fvec_new
                    # block.  Everything unproven — dynamic or mismatched
                    # lengths, nested matrices, N outside [1, _FLEN_MAX] —
                    # falls back to the mx_fvec_binop C loop, which is
                    # always correct (and ABORTS on vector-vector length
                    # mismatches the inline path must therefore never
                    # reach: mode 0 requires both lengths known-EQUAL).
                    # Int / and % keep the runtime call even with known
                    # lengths — DOCUMENTED CHOICE: mx_fvec_scalar_op aborts
                    # loudly on division by zero, and a vector sdiv would
                    # be UB there instead; the float ops are IEEE both ways.
                    dk = kind(dst)
                    leaf, depth = _fvec_leaf(dk)
                    lk0, rk0 = kind(opargs[0]), kind(opargs[1])
                    mode = (0 if _is_fvec(lk0) and _is_fvec(rk0)
                            else 1 if _is_fvec(lk0) else 2)
                    simd_ok = (
                        depth == 0
                        and (leaf == F64 and o in ("+", "-", "*", "/")
                             or leaf == I64 and o in ("+", "-", "*")))
                    simd_n: Optional[int] = None
                    if simd_ok:
                        if mode == 0:
                            n0 = fvec_len_of(opargs[0])
                            n1 = fvec_len_of(opargs[1])
                            simd_n = n0 if n0 is not None and n0 == n1 \
                                else None
                        else:
                            simd_n = fvec_len_of(
                                opargs[0] if mode == 1 else opargs[1])
                    if simd_n is not None:
                        ety = "double" if leaf == F64 else "i64"
                        vty = f"<{simd_n} x {ety}>"

                        def vload(ptr_ssa: str) -> str:
                            ep, vv = fresh(), fresh()
                            lines.append(
                                f"  {ep} = getelementptr inbounds i8, "
                                f"ptr {ptr_ssa}, i64 8")
                            lines.append(
                                f"  {vv} = load {vty}, ptr {ep}, align 8")
                            return vv

                        def vsplat(scal_ssa: str) -> str:
                            t0, t1 = fresh(), fresh()
                            lines.append(
                                f"  {t0} = insertelement {vty} poison, "
                                f"{ety} {scal_ssa}, i64 0")
                            lines.append(
                                f"  {t1} = shufflevector {vty} {t0}, "
                                f"{vty} poison, "
                                f"<{simd_n} x i32> zeroinitializer"
                                "  ; scalar broadcast splat")
                            return t1

                        lv = vload(l) if _is_fvec(lk0) else vsplat(l)
                        rv = vload(r) if _is_fvec(rk0) else vsplat(r)
                        mnem = _ARITH_FLT[o] if leaf == F64 \
                            else _ARITH_INT[o]
                        rvec = fresh()
                        lines.append(
                            f"  {rvec} = {mnem} {vty} {lv}, {rv}"
                            f"  ; element-wise {o} inline SIMD "
                            f"(static length {simd_n})")
                        mod.runtime_syms.add("mx_fvec_new")
                        out = fresh()
                        lines.append(
                            f"  {out} = call ptr @mx_fvec_new("
                            f"i64 {simd_n})"
                            "  ; result block (leaks by design)")
                        outp = fresh()
                        lines.append(
                            f"  {outp} = getelementptr inbounds i8, "
                            f"ptr {out}, i64 8")
                        lines.append(
                            f"  store {vty} {rvec}, ptr {outp}, align 8")
                        setval(dst, out, lines)
                        continue
                    lw = to_word(lk0, l, lines)
                    rw = to_word(rk0, r, lines)
                    mod.runtime_syms.add("mx_fvec_binop")
                    v = fresh()
                    note = ("vector-vector" if mode == 0 else
                            "scalar broadcast")
                    lines.append(
                        f"  {v} = call ptr @mx_fvec_binop("
                        f"i64 {_FVEC_BINOP_CODES[o]}, "
                        f"i64 {1 if leaf == F64 else 0}, i64 {depth}, "
                        f"i64 {mode}, i64 {lw}, i64 {rw})"
                        f"  ; element-wise {o} ({note})")
                    setval(dst, v, lines)
                elif is_str and o == "+":
                    # String concatenation -> fresh malloc'd string from the
                    # native runtime; freed when the ownership analysis
                    # proves the value non-retained, otherwise leaked by
                    # design (same contract as boxes/heap envs).
                    mod.runtime_syms.add("mx_str_concat")
                    v = fresh()
                    snote = ("owned (freed when dead)"
                             if dst in owned_str_set or dst in str_prod_temps
                             else "leaks by design")
                    lines.append(
                        f"  {v} = call ptr @mx_str_concat(ptr {l}, ptr {r})"
                        f"  ; {snote}")
                    setval(dst, v, lines)
                elif is_str and o in ("==", "!="):
                    # Content equality via mx_str_eq (returns 0/1), exactly
                    # the interpreter's string comparison.
                    mod.runtime_syms.add("mx_str_eq")
                    e = fresh()
                    lines.append(f"  {e} = call i64 @mx_str_eq(ptr {l}, ptr {r})")
                    if o == "==":
                        setval(dst, e, lines)
                    else:
                        c, v = fresh(), fresh()
                        lines.append(f"  {c} = icmp eq i64 {e}, 0")
                        lines.append(f"  {v} = zext i1 {c} to i64")
                        setval(dst, v, lines)
                elif o in _CMP_INT:
                    c = fresh()
                    if is_flt:
                        lines.append(f"  {c} = fcmp {_CMP_FLT[o]} double {l}, {r}")
                    elif kind(opargs[0]) == PTR:
                        # raw pointer ==/!=: identity comparison, exactly
                        # the interpreter's structural MxPtr/None equality.
                        lines.append(f"  {c} = icmp {_CMP_INT[o]} ptr {l}, {r}")
                    else:
                        lines.append(f"  {c} = icmp {_CMP_INT[o]} i64 {l}, {r}")
                    v = fresh()
                    lines.append(f"  {v} = zext i1 {c} to i64")
                    setval(dst, v, lines)
                elif o in _BITWISE_INT:
                    # Int-only, i64 two's complement.  A shift's COUNT is
                    # validated first: @mx_shift_check returns the count
                    # when it is in 0..63 and aborts with a message
                    # otherwise, so an out-of-range shift is loud on both
                    # engines instead of poison natively and an exception in
                    # the interpreter.  The guard is a call rather than an
                    # inline compare-and-branch so the surrounding basic
                    # block stays intact.
                    amount = r
                    if o in _SHIFT_OPS:
                        mod.runtime_syms.add("mx_shift_check")
                        chk = fresh()
                        lines.append(
                            f"  {chk} = call i64 @mx_shift_check("
                            f"i64 {r}, i64 {1 if o == '<<' else 0})"
                            f"  ; aborts unless 0 <= count < 64")
                        amount = chk
                    v = fresh()
                    lines.append(
                        f"  {v} = {_BITWISE_INT[o]} i64 {l}, {amount}"
                        f"  ; bitwise {o}")
                    setval(dst, v, lines)
                elif o in _LOGIC:
                    lb, rb, v = fresh(), fresh(), fresh()
                    lines.append(f"  {lb} = icmp ne i64 {l}, 0")
                    lines.append(f"  {rb} = icmp ne i64 {r}, 0")
                    lines.append(f"  {v} = {_LOGIC[o]} i1 {lb}, {rb}")
                    z = fresh()
                    lines.append(f"  {z} = zext i1 {v} to i64")
                    setval(dst, z, lines)
                else:
                    mnem = _ARITH_FLT[o] if is_flt else _ARITH_INT[o]
                    v = fresh()
                    ty = "double" if is_flt else "i64"
                    lines.append(f"  {v} = {mnem} {ty} {l}, {r}")
                    setval(dst, v, lines)
            elif rk == "select":
                if dst in info.dead_results:
                    lines.append(f"  ; dead select elided: {dst} (never observed)")
                    continue
                c = use(opargs[0], lines)
                t = use(opargs[1], lines)
                e = use(opargs[2], lines)
                cb = fresh()
                lines.append(f"  {cb} = icmp ne i64 {c}, 0")
                v = fresh()
                ty = llty(dst)
                lines.append(f"  {v} = select i1 {cb}, {ty} {t}, {ty} {e}")
                setval(dst, v, lines)
            elif rk == "call":
                callee = rhs[1]
                # Locals shadow builtins (the interpreter's resolution
                # order), so the closure-call check comes first.
                bname = _builtin_name(callee, module_names)
                if callee in info.def_count:
                    # Closure call: load {fn, env} from the pair and call
                    # the fn pointer.  A pinned NON-participant lambda is
                    # called with its typed signature; a dynamic member
                    # set — or a pinned word-uniform participant — goes
                    # through the word-uniform ABI (`i64 (ptr, i64...)`,
                    # args word-encoded, result word-decoded per the
                    # site's inferred kinds).
                    ck = kind(callee)
                    if not _is_closure(ck):
                        raise _Unsupported(
                            f"call through local {callee!r} that is not a "
                            "statically-known closure")
                    members = _closure_members(ck)
                    for m in members:
                        if m not in emitted_names:
                            raise _Unsupported(
                                f"closure call to non-emitted lambda {m!r}")
                    is_word = _is_dyn_closure(ck) or any(
                        m in word_uniform for m in members)
                    base = use(callee, lines)
                    fpp, envpp = fresh(), fresh()
                    lines.append(
                        f"  {fpp} = getelementptr inbounds {_CLOSURE_PAIR_TY}, "
                        f"ptr {base}, i32 0, i32 0")
                    fnv = fresh()
                    lines.append(f"  {fnv} = load ptr, ptr {fpp}")
                    lines.append(
                        f"  {envpp} = getelementptr inbounds {_CLOSURE_PAIR_TY}, "
                        f"ptr {base}, i32 0, i32 1")
                    envv = fresh()
                    lines.append(f"  {envv} = load ptr, ptr {envpp}")
                    if is_word:
                        avals = [f"ptr {envv}"]
                        for ai, a in enumerate(opargs):
                            ak = kind(a)
                            # READ-ONLY AGGREGATE ARGUMENT (increment 17):
                            # pass a pointer to OUR existing storage
                            # instead of a fresh box — the same reasoning
                            # the direct-call elide-copy pass uses.
                            if _is_agg(ak) and readonly_indirect_arg(
                                    members, ai):
                                av = use(a, lines)
                                t = fresh()
                                lines.append(
                                    f"  {t} = ptrtoint ptr {av} to i64"
                                    f"  ; elide-box: read-only aggregate "
                                    f"argument {a} ({ak}) passes the caller's "
                                    "storage pointer (no member writes "
                                    "through it, and the call is synchronous "
                                    "on this stack)")
                                avals.append(f"i64 {t}")
                                continue
                            # to_word boxes struct/enum args (malloc +
                            # write-once copy in, pointer as the word);
                            # scalars pass raw, with no allocation.
                            w = to_word(ak, use(a, lines), lines, src=a)
                            avals.append(f"i64 {w}")
                        v = fresh()
                        lines.append(
                            f"  {v} = call i64 {fnv}({', '.join(avals)})"
                            f"  ; indirect closure call "
                            f"({'|'.join(members)}), word-uniform ABI")
                        # word_into copies an aggregate result OUT of the
                        # callee's fresh box into dst's own storage.
                        word_into(dst, v, lines)
                    else:
                        lname = members[0]
                        csig = sigs[lname]
                        avals = [f"ptr {envv}"]
                        for a, pk in zip(opargs, csig.params):
                            avals.append(f"{_llparam(pk)} {use(a, lines)}")
                        if _is_agg(csig.ret):
                            if dst not in aggset:
                                raise _Unsupported(
                                    f"call result {dst!r} not aggregate-kinded for "
                                    f"sret closure call to {lname!r}")
                            avals.insert(0, f"ptr {struct_ref(dst)}")
                            lines.append(f"  call void {fnv}({', '.join(avals)})")
                        else:
                            v = fresh()
                            rty = _llscalar(csig.ret)
                            lines.append(
                                f"  {v} = call {rty} {fnv}({', '.join(avals)})")
                            setval(dst, v, lines)
                elif callee.startswith(TRAIT_CALL_PREFIX):
                    # Statically-resolved trait dispatch (kind-driven; the
                    # consistency check validated the resolution).
                    method = callee[len(TRAIT_CALL_PREFIX):]
                    if not opargs:
                        raise _Unsupported(
                            f"trait method call {method!r} with no receiver")
                    res, target = _resolve_trait_call(
                        method, kind(opargs[0]), traits, module_names,
                        assume_final=True)
                    if res == "builtin":
                        emit_rt_builtin(target, dst, opargs, lines)
                    elif res == "func":
                        emit_direct_call(dst, target, opargs, lines)
                    else:
                        raise _Unsupported(
                            target or f"unresolved trait method call {method!r}")
                elif callee.startswith(STATIC_CALL_PREFIX):
                    # Compile-time static method resolution (the
                    # interpreter's _dispatch_static_call order).
                    sres, starget = _resolve_static_call(
                        callee, traits, module_names)
                    if sres == "builtin":
                        emit_rt_builtin(starget, dst, opargs, lines)
                    elif sres == "func":
                        emit_direct_call(dst, starget, opargs, lines)
                    else:
                        raise _Unsupported(starget)
                elif callee.startswith(_EFFECT_PRIMITIVE_CALL_PREFIX):
                    # `with SYMBOL` thread/mutex primitives inside the
                    # __effect_runtime$E$op thunks (docs/threads_runtime.md;
                    # kinds validated by the consistency check).
                    symbol = callee[len(_EFFECT_PRIMITIVE_CALL_PREFIX):]
                    prim = _EFFECT_PRIMITIVES.get(symbol)
                    if prim is None:  # unreachable: prescan demoted
                        raise _Unsupported(
                            f"runtime primitive {symbol!r} has no native "
                            "implementation")
                    csym = prim[0]
                    mod.runtime_syms.add(csym)
                    if symbol == "EFFECT_SPAWN":
                        # Hand the closure's {fn, env} to the C runtime:
                        # the child thread invokes fn(env) through the
                        # word-uniform ABI; the env is a heap (immortal)
                        # block by the driver's spawn-closure forcing.
                        base = use(opargs[0], lines)
                        fpp, envpp = fresh(), fresh()
                        lines.append(
                            f"  {fpp} = getelementptr inbounds "
                            f"{_CLOSURE_PAIR_TY}, ptr {base}, i32 0, i32 0")
                        fnv = fresh()
                        lines.append(f"  {fnv} = load ptr, ptr {fpp}")
                        lines.append(
                            f"  {envpp} = getelementptr inbounds "
                            f"{_CLOSURE_PAIR_TY}, ptr {base}, i32 0, i32 1")
                        envv = fresh()
                        lines.append(f"  {envv} = load ptr, ptr {envpp}")

                        # CONTENTION MARKING (docs/contention_as_permission
                        # .md § marking rule): this thunk body executes iff
                        # the spawn is REAL — a direct-mapped perform calls
                        # it, and a scoped perform reaches it only as
                        # mx_perform_or_default's fallback after no handler
                        # claimed the op — so marking HERE covers both
                        # routes and never marks a virtualized spawn,
                        # mirroring the interpreter's _rt_thread_spawn
                        # exactly. Walk the captures of each member lambda
                        # the closure kind admits (branching on the fn
                        # pointer when there are several): vec captures are
                        # marked, struct captures recurse by static field
                        # layout, cell captures (mutable) load through the
                        # cell pointer, and the walk STOPS at vec elements
                        # (the documented, test-pinned hole). A capture
                        # whose kind cannot be enumerated demotes the thunk
                        # — an honest placeholder, never engine asymmetry.
                        _NO_MARKS_REASON = (
                            "cannot emit contention marks for spawn "
                            "captures (kinds unavailable)")

                        def _vec_mark_at(slotp: str) -> None:
                            mod.runtime_syms.add("mx_vec_mark_contended")
                            vv = fresh()
                            lines.append(f"  {vv} = load ptr, ptr {slotp}")
                            lines.append(
                                f"  call void @mx_vec_mark_contended"
                                f"(ptr {vv})  ; spawn capture crossed")

                        def _mark_struct(sname: str, basep: str,
                                         on_path: frozenset) -> None:
                            if (sname in structs.bad
                                    or sname not in structs.fields
                                    or sname in on_path):
                                raise _Unsupported(_NO_MARKS_REASON)
                            sty = f"%struct.{_sanitize(sname)}"
                            for fi, fnm in enumerate(structs.fields[sname]):
                                fk2 = structs.field_kind(sname, fnm)
                                if _is_vec(fk2):
                                    p2 = fresh()
                                    lines.append(
                                        f"  {p2} = getelementptr inbounds "
                                        f"{sty}, ptr {basep}, i32 0, i32 {fi}")
                                    _vec_mark_at(p2)
                                elif _is_struct(fk2):
                                    p2 = fresh()
                                    lines.append(
                                        f"  {p2} = getelementptr inbounds "
                                        f"{sty}, ptr {basep}, i32 0, i32 {fi}")
                                    _mark_struct(_struct_name(fk2), p2,
                                                 on_path | {sname})
                                elif fk2 == CONFLICT:
                                    raise _Unsupported(_NO_MARKS_REASON)
                                # else: scalars/str/fvec/enum/closure stop
                                # (the spec recurses struct fields ONLY;
                                # the interpreter's _mark_contended agrees).

                        def _member_needs_marks(m: str) -> bool:
                            def kind_touches(k: str,
                                             seen: frozenset) -> bool:
                                if _is_cell_marker(k):
                                    k = _cell_elem(k)
                                if _is_vec(k) or k == CONFLICT:
                                    return True
                                if _is_struct(k):
                                    sn = _struct_name(k)
                                    if sn in seen:
                                        return True  # forces the demote path
                                    if (sn in structs.bad
                                            or sn not in structs.fields):
                                        return True
                                    return any(
                                        kind_touches(
                                            structs.field_kind(sn, f2),
                                            seen | {sn})
                                        for f2 in structs.fields[sn])
                                return False
                            return any(kind_touches(k, frozenset())
                                       for (_c, k) in env_fields(m))

                        def _mark_member_env(m: str, envp: str) -> None:
                            ety2 = f"%env.{_sanitize(m)}"
                            for fi, (cn2, ck2) in enumerate(env_fields(m)):
                                is_cell = _is_cell_marker(ck2)
                                ek = _cell_elem(ck2) if is_cell else ck2
                                if not (_is_vec(ek) or _is_struct(ek)
                                        or ek == CONFLICT):
                                    continue
                                if ek == CONFLICT:
                                    raise _Unsupported(_NO_MARKS_REASON)
                                p2 = fresh()
                                lines.append(
                                    f"  {p2} = getelementptr inbounds "
                                    f"{ety2}, ptr {envp}, i32 0, i32 {fi}")
                                if is_cell:
                                    # Mutable capture: the env holds the
                                    # one-word cell POINTER; only a vec can
                                    # live in a cell word.
                                    if not _is_vec(ek):
                                        raise _Unsupported(_NO_MARKS_REASON)
                                    cp2 = fresh()
                                    lines.append(
                                        f"  {cp2} = load ptr, ptr {p2}"
                                        f"  ; cell pointer for {cn2}")
                                    _vec_mark_at(cp2)
                                elif _is_vec(ek):
                                    _vec_mark_at(p2)
                                else:  # inline struct capture
                                    _mark_struct(_struct_name(ek), p2,
                                                 frozenset())

                        spawn_members = _closure_members(kind(opargs[0]))
                        marked = [m for m in spawn_members
                                  if _member_needs_marks(m)]
                        if len(spawn_members) == 1 and marked:
                            # The kind pins the one lambda: no branch.
                            _mark_member_env(spawn_members[0], envv)
                        elif marked:
                            uid = fresh()[1:]
                            done_lbl = f"spawn.marked.{uid}"
                            for j, m in enumerate(marked):
                                cnd, nxt = fresh(), f"spawn.next.{uid}.{j}"
                                lines.append(
                                    f"  {cnd} = icmp eq ptr {fnv}, "
                                    f"@{mangle(m)}")
                                lines.append(
                                    f"  br i1 {cnd}, label "
                                    f"%spawn.mark.{uid}.{j}, label %{nxt}")
                                lines.append(f"spawn.mark.{uid}.{j}:")
                                _mark_member_env(m, envv)
                                lines.append(f"  br label %{done_lbl}")
                                lines.append(f"{nxt}:")
                            lines.append(f"  br label %{done_lbl}")
                            lines.append(f"{done_lbl}:")

                        v = fresh()
                        lines.append(
                            f"  {v} = call i64 @mx_thread_spawn(ptr {fnv}, "
                            f"ptr {envv})"
                            "  ; start the closure on a new pthread; opaque "
                            "immortal handle word")
                        setval(dst, v, lines)
                    elif symbol == "EFFECT_JOIN":
                        a = use(opargs[0], lines)
                        v = fresh()
                        lines.append(
                            f"  {v} = call i64 @mx_thread_join(i64 {a})"
                            "  ; blocks; the child's result word, exactly "
                            "once (double join raises)")
                        # Decode the child's result word per dst's kind
                        # (word kinds only; the consistency check demotes
                        # aggregates).
                        word_into(dst, v, lines)
                    elif symbol == "EFFECT_MUTEX_CREATE":
                        v = fresh()
                        lines.append(
                            f"  {v} = call i64 @mx_mutex_create()"
                            "  ; ERRORCHECK mutex; opaque immortal handle "
                            "word")
                        setval(dst, v, lines)
                    else:  # EFFECT_MUTEX_LOCK / EFFECT_MUTEX_UNLOCK
                        a = use(opargs[0], lines)
                        v = fresh()
                        lines.append(
                            f"  {v} = call i64 @{csym}(i64 {a})"
                            "  ; blocks if held elsewhere / raises on "
                            "misuse; unit word")
                        setval(dst, v, lines)
                elif bname in _NATIVE_RT_CALLS:
                    emit_rt_builtin(bname, dst, opargs, lines)
                elif bname in _EXTERN_C_SIGS:
                    emit_extern_call(bname, dst, opargs, lines)
                elif bname == "as_ptr":
                    a = use(opargs[0], lines)
                    if kind(opargs[0]) == STR:
                        # Native strings already ARE NUL-terminated byte
                        # pointers, so as_ptr is identity (the interpreter's
                        # fresh readonly snapshot is observationally the
                        # same for every accepted program).
                        lines.append(f"  ; as_ptr: identity on a native "
                                     "string (already NUL-terminated bytes)")
                        setval(dst, a, lines)
                    elif _is_fvec(kind(opargs[0])):
                        mod.runtime_syms.add("mx_fvec_as_bytes")
                        v = fresh()
                        lines.append(
                            f"  {v} = call ptr @mx_fvec_as_bytes(ptr {a})"
                            "  ; fresh byte snapshot (leaks by design)")
                        setval(dst, v, lines)
                    else:  # vec receiver (consistency validated the kind)
                        mod.runtime_syms.add("mx_vec_as_bytes")
                        v = fresh()
                        lines.append(
                            f"  {v} = call ptr @mx_vec_as_bytes(ptr {a})"
                            "  ; fresh byte snapshot (leaks by design)")
                        setval(dst, v, lines)
                elif bname == "ptr_read":
                    base = use(opargs[0], lines)
                    off = use(opargs[1], lines)
                    p, b8, v = fresh(), fresh(), fresh()
                    lines.append(
                        f"  {p} = getelementptr inbounds i8, ptr {base}, "
                        f"i64 {off}")
                    lines.append(f"  {b8} = load i8, ptr {p}")
                    lines.append(f"  {v} = zext i8 {b8} to i64")
                    setval(dst, v, lines)
                elif bname == "ptr_write":
                    base = use(opargs[0], lines)
                    off = use(opargs[1], lines)
                    val = use(opargs[2], lines)
                    t8, p = fresh(), fresh()
                    lines.append(f"  {t8} = trunc i64 {val} to i8")
                    lines.append(
                        f"  {p} = getelementptr inbounds i8, ptr {base}, "
                        f"i64 {off}")
                    lines.append(f"  store i8 {t8}, ptr {p}")
                    setval(dst, "0", lines)  # unit
                elif bname == "assert":
                    # Inline branch-to-abort on a falsy condition (message
                    # arguments are evaluated by their own MIR ops but not
                    # rendered natively; the interpreter raises instead).
                    c = use(opargs[0], lines)
                    cb = fresh()
                    n = cb[1:]  # unique per-site label suffix
                    lines.append(f"  {cb} = icmp ne i64 {c}, 0")
                    lines.append(f"  br i1 {cb}, label %assert.ok.{n}, "
                                 f"label %assert.fail.{n}")
                    lines.append(f"assert.fail.{n}:")
                    mod.uses_abort = True
                    lines.append("  call void @abort()  ; assert failed")
                    lines.append("  unreachable")
                    lines.append(f"assert.ok.{n}:")
                    setval(dst, "0", lines)  # unit
                elif bname in _PRINT_BUILTINS:
                    def fvec_repr(a: str) -> str:
                        """Render a fixed vector to its interpreter repr
                        string ("vector[...]"); freed right after the print
                        (printf never retains)."""
                        leaf, depth = _fvec_leaf(kind(a))
                        mod.runtime_syms.add("mx_fvec_to_str")
                        sv = fresh()
                        lines.append(
                            f"  {sv} = call ptr @mx_fvec_to_str("
                            f"ptr {use(a, lines)}, "
                            f"i64 {1 if leaf == F64 else 0}, i64 {depth})"
                            "  ; vector repr for print")
                        return sv
                    def tile_repr(a: str) -> str:
                        """Render a tile to its interpreter repr string
                        ("tile[RxC](...)"); freed right after the print."""
                        elem, _r, _c = _tile_parts(kind(a))
                        mod.runtime_syms.add("mx_tile_to_str")
                        sv = fresh()
                        lines.append(
                            f"  {sv} = call ptr @mx_tile_to_str("
                            f"ptr {use(a, lines)}, "
                            f"i64 {1 if elem == F64 else 0})"
                            "  ; tile repr for print")
                        return sv
                    fvec_temps: List[str] = []
                    if len(opargs) == 1:
                        k = kind(opargs[0])
                        if _is_fvec(k):
                            sv = fvec_repr(opargs[0])
                            fvec_temps.append(sv)
                            mod.print_helpers.add(STR)
                            lines.append(
                                f"  call void @metaxu_print_str(ptr {sv})")
                        elif _is_tile(k):
                            sv = tile_repr(opargs[0])
                            fvec_temps.append(sv)
                            mod.print_helpers.add(STR)
                            lines.append(
                                f"  call void @metaxu_print_str(ptr {sv})")
                        else:
                            a = use(opargs[0], lines)
                            mod.print_helpers.add(k)
                            hn = {I64: "metaxu_print_i64",
                                  F64: "metaxu_print_f64",
                                  STR: "metaxu_print_str"}[k]
                            lines.append(
                                f"  call void @{hn}({_LLTY[k]} {a})")
                    else:
                        # 0 or 2+ args: one printf with space-joined per-kind
                        # directives, matching the interpreter's print(*args).
                        parts: List[str] = []
                        avals: List[str] = []
                        for a in opargs:
                            k2 = kind(a)
                            if _is_fvec(k2):
                                sv = fvec_repr(a)
                                fvec_temps.append(sv)
                                parts.append("%s")
                                avals.append(f"ptr {sv}")
                            elif _is_tile(k2):
                                sv = tile_repr(a)
                                fvec_temps.append(sv)
                                parts.append("%s")
                                avals.append(f"ptr {sv}")
                            elif k2 == F64:
                                # Interpreter parity (see _PRINT_FMTS):
                                # floats render via mx_f64_to_str, not %g.
                                mod.runtime_syms.add("mx_f64_to_str")
                                sv = fresh()
                                lines.append(
                                    f"  {sv} = call ptr @mx_f64_to_str("
                                    f"double {use(a, lines)})")
                                fvec_temps.append(sv)
                                parts.append("%s")
                                avals.append(f"ptr {sv}")
                            else:
                                parts.append(
                                    {I64: "%lld", STR: "%s"}[k2])
                                avals.append(f"{_LLTY[k2]} {use(a, lines)}")
                        fmt = " ".join(parts) + "\n"
                        g = mod.intern_string(fmt)
                        mod.uses_printf = True
                        r = fresh()
                        call_args = ", ".join([f"ptr {g}"] + avals)
                        lines.append(
                            f"  {r} = call i32 (ptr, ...) @printf({call_args})")
                    for sv in fvec_temps:
                        mod.runtime_syms.add("mx_str_free")
                        lines.append(
                            f"  call void @mx_str_free(ptr {sv})"
                            "  ; print never retains the repr")
                    setval(dst, "0", lines)  # unit
                elif bname == "neg":
                    a = use(opargs[0], lines)
                    v = fresh()
                    if kind(dst) == F64:
                        lines.append(f"  {v} = fneg double {a}")
                    else:
                        lines.append(f"  {v} = sub i64 0, {a}")
                    setval(dst, v, lines)
                elif bname == "not":
                    a = use(opargs[0], lines)
                    c, v = fresh(), fresh()
                    lines.append(f"  {c} = icmp eq i64 {a}, 0")
                    lines.append(f"  {v} = zext i1 {c} to i64")
                    setval(dst, v, lines)
                elif bname == "bnot":
                    # `~x` — bitwise complement, exactly `x ^ -1`.
                    a = use(opargs[0], lines)
                    v = fresh()
                    lines.append(f"  {v} = xor i64 {a}, -1  ; bitwise ~")
                    setval(dst, v, lines)
                elif bname in _MATH_EXTERNS:
                    mod.math_used.add(bname)
                    a = use(opargs[0], lines)
                    v = fresh()
                    lines.append(f"  {v} = call double @{bname}(double {a})")
                    setval(dst, v, lines)
                else:
                    emit_direct_call(dst, callee, opargs, lines)
            elif rk == "alloc_struct":
                sname = rhs[1]
                if dst not in aggset:
                    raise _Unsupported(f"alloc_struct result {dst!r} not struct-kinded")
                for (fn_, fv) in opargs:
                    p = gep(sname, struct_ref(dst), fn_, lines)
                    fk = structs.field_kind(sname, fn_)
                    if _is_agg(fk):
                        # nested aggregate field: copy the value into the
                        # inline field region (value semantics, no heap)
                        agg_copy(_agg_ty(fk), use(fv, lines), p, lines)
                    else:
                        lines.append(
                            f"  store {_llscalar(fk)} {use(fv, lines)}, ptr {p}")
            elif rk == "field_get":
                sname = _struct_name(kind(opargs[0]))
                base = use(opargs[0], lines)
                p = gep(sname, base, rhs[1], lines)
                fk = structs.field_kind(sname, rhs[1])
                if _is_agg(fk):
                    if dst not in aggset:
                        raise _Unsupported(
                            f"field_get result {dst!r} not aggregate-kinded "
                            f"for nested field {rhs[1]!r}")
                    agg_copy(_agg_ty(fk), p, struct_ref(dst), lines)
                else:
                    v = fresh()
                    lines.append(f"  {v} = load {_llscalar(fk)}, ptr {p}")
                    setval(dst, v, lines)
            elif rk == "field_set":
                # Value semantics: dst = copy of base with one field updated.
                sname = _struct_name(kind(opargs[0]))
                if dst not in aggset:
                    raise _Unsupported(f"field_set result {dst!r} not struct-kinded")
                base = use(opargs[0], lines)
                agg_copy(f"%struct.{_sanitize(sname)}", base, struct_ref(dst), lines)
                p = gep(sname, struct_ref(dst), rhs[1], lines)
                fk = structs.field_kind(sname, rhs[1])
                if _is_agg(fk):
                    agg_copy(_agg_ty(fk), use(opargs[1], lines), p, lines)
                else:
                    lines.append(
                        f"  store {_llscalar(fk)} {use(opargs[1], lines)}, ptr {p}")
            elif rk == "make_variant":
                # Tagged union: store the integer tag, then the payload slots
                # at THIS variant's refined slot kinds (different variants —
                # and different instantiations of one variant — may disagree
                # about what a slot index holds).
                ename, vname = rhs[1], rhs[2]
                if dst not in aggset or not _is_enum(kind(dst)):
                    raise _Unsupported(
                        f"make_variant result {dst!r} not enum-kinded")
                tagv = variants.tags.get(vname)
                if tagv is None:
                    raise _Unsupported(f"no tag assigned for variant {vname!r}")
                vref = _enum_refinement(kind(dst)) or {}
                vslots = vref.get(vname)
                if vslots is None or len(vslots) != len(opargs):
                    raise _Unsupported(
                        f"make_variant {vname!r} destination kind lacks the "
                        "variant's refinement")
                mod.used_enums.add(ename)
                p = enum_gep(ename, struct_ref(dst), lines)
                lines.append(
                    f"  store i64 {tagv}, ptr {p}  ; tag {vname}={tagv}")
                for i, fv in enumerate(opargs):
                    ck = vslots[i]
                    p = enum_gep(ename, struct_ref(dst), lines, payload=i)
                    if _is_agg(ck):
                        # Boxed aggregate payload: malloc a write-once box,
                        # copy the aggregate in, store the POINTER in the
                        # 8-byte slot.  Normally never freed (leak by
                        # design: shallow pair/aggregate copies share box
                        # pointers, so no free can be proven unique — see
                        # module docstring) — EXCEPT at a unique-box site,
                        # where this frame is provably the box's only owner
                        # and every ret path frees it (_unique_box_enums).
                        size = _kind_size(ck, structs, variants)
                        if size is None:
                            raise _Unsupported(
                                f"boxed payload of {ck} has infinite layout")
                        mod.uses_malloc = True
                        box = fresh()
                        if dst in unique_box_sites:
                            note = "unique: freed at frame exit"
                            unique_box_regs.append((box, dst, i))
                        else:
                            note = "leaks by design"
                        lines.append(
                            f"  {box} = call ptr @malloc(i64 {max(size, 8)})"
                            f"  ; boxed {ck} payload ({note})")
                        agg_copy(_agg_ty(ck), use(fv, lines), box, lines)
                        lines.append(f"  store ptr {box}, ptr {p}")
                    else:
                        lines.append(
                            f"  store {_llscalar(ck)} {use(fv, lines)}, ptr {p}")
            elif rk == "variant_tag":
                bk = kind(opargs[0])
                if not _is_enum(bk):
                    raise _Unsupported(
                        f"variant_tag of non-enum value {opargs[0]!r}")
                base = use(opargs[0], lines)
                p = enum_gep(_enum_name(bk), base, lines)
                v = fresh()
                lines.append(f"  {v} = load i64, ptr {p}")
                setval(dst, v, lines)
            elif rk == "variant_field":
                bk = kind(opargs[0])
                if not _is_enum(bk):
                    raise _Unsupported(
                        f"variant_field of non-enum value {opargs[0]!r}")
                ename = _enum_name(bk)
                vname = rhs[2] if len(rhs) > 2 else None
                vref = _enum_refinement(bk)
                if vname is None or vref is None:
                    raise _Unsupported(
                        f"variant_field {rhs[1]} of enum {ename or 'anon'!r} "
                        "without a variant refinement (legacy MIR shape)")
                if vname not in vref or rhs[1] >= len(vref[vname]):
                    # Dead arm: no flow into this value constructs vname (the
                    # refinement lists every constructible variant), so the
                    # guarding tag test can never pass and this read never
                    # executes.  Emit a typed zero for scalar results;
                    # aggregate results keep their (never-read) storage.
                    if dst in aggset:
                        lines.append(
                            f"  ; dead arm: variant {vname!r} never "
                            f"constructed for this value; {dst} left "
                            "uninitialized (unreachable)")
                    else:
                        zk = kind(dst)
                        if zk == F64:
                            zv = _fmt_f64(0.0)
                        elif zk == STR:
                            zv = mod.intern_string("")
                        elif _is_vec(zk) or zk == PTR:
                            zv = "null"
                        else:
                            zv = "0"
                        lines.append(
                            f"  ; dead arm: variant {vname!r} never "
                            "constructed for this value (unreachable)")
                        setval(dst, zv, lines)
                    continue
                ck = vref[vname][rhs[1]]
                if _is_enum(ck):
                    # Nested extraction: the boxed value carries the
                    # canonical module-wide representation (consistency
                    # demoted any mixed slots), and kind(dst) is the
                    # canonical refined kind of the same enum.
                    ck = kind(dst) if _is_enum(kind(dst)) else ck
                base = use(opargs[0], lines)
                p = enum_gep(ename, base, lines, payload=rhs[1])
                if _is_agg(ck):
                    # Boxed payload: load the box pointer, then either
                    # alias it (BOX VIEW: the result is only ever read, so
                    # it may read through the write-once box directly) or
                    # copy the aggregate out into the destination's own
                    # storage (value semantics; the box stays shared).
                    if dst not in aggset:
                        raise _Unsupported(
                            f"variant_field result {dst!r} not "
                            f"aggregate-kinded for boxed slot {rhs[1]}")
                    box = fresh()
                    lines.append(f"  {box} = load ptr, ptr {p}")
                    if dst in boxview:
                        lines.append(
                            f"  store ptr {box}, ptr {bp_ref(dst)}"
                            f"  ; elide-copy: variant_field {dst} reads "
                            "through the box pointer (read-only result)")
                    else:
                        agg_copy(_agg_ty(ck), box, struct_ref(dst), lines)
                else:
                    v = fresh()
                    lines.append(f"  {v} = load {_llscalar(ck)}, ptr {p}")
                    setval(dst, v, lines)
            elif rk == "make_closure":
                # Fill this site's env struct with the captured values, then
                # store the {fn, env} pair into the closure variable.
                lname = rhs[1]
                if dst not in aggset or not _is_closure(kind(dst)):
                    raise _Unsupported(
                        f"make_closure result {dst!r} not closure-kinded")
                if lname not in emitted_names:
                    raise _Unsupported(
                        f"make_closure of non-emitted lambda {lname!r}")
                fields = env_fields(lname)
                mod.env_types[lname] = fields
                mod.uses_closure_pair = True
                ety = f"%env.{_sanitize(lname)}"
                if lname in closures.heap_env:
                    # Escaping (or looped) closure: malloc a fresh env at
                    # the site, never freed (leak by design — an immortal
                    # env can never dangle, see module docstring).
                    esize = 0
                    for (_cn, ck) in fields:
                        s = _kind_size(ck, structs, variants)
                        if s is None:
                            raise _Unsupported(
                                f"env capture of {ck} has infinite layout")
                        esize += s
                    mod.uses_malloc = True
                    envp = fresh()
                    lines.append(
                        f"  {envp} = call ptr @malloc(i64 {max(esize, 8)})"
                        f"  ; heap env for {dst} -> {lname} (leaks by design)")
                else:
                    envp = env_allocas.get(id(op))
                    if envp is None:  # unreachable: prescan covers every site
                        raise _Unsupported("make_closure site missing env storage")
                for i, ((cn, ck), (_cn2, vn)) in enumerate(zip(fields, opargs)):
                    p = fresh()
                    lines.append(
                        f"  {p} = getelementptr inbounds {ety}, ptr {envp}, "
                        f"i32 0, i32 {i}")
                    if _is_cell_marker(ck):
                        # Mutable capture: store the CELL POINTER, never
                        # the value — the closure aliases the binding.
                        if vn not in cellset:
                            raise _Unsupported(
                                f"cell capture {cn!r} of non-cell "
                                f"variable {vn!r}")
                        lines.append(
                            f"  store ptr {cellp_ref(vn)}, ptr {p}"
                            f"  ; mutable capture: cell pointer for {cn}")
                    elif _is_agg(ck):
                        # aggregate capture: copy the whole value into the env
                        agg_copy(_agg_ty(ck), use(vn, lines), p, lines)
                    else:
                        lines.append(
                            f"  store {_llscalar(ck)} {use(vn, lines)}, ptr {p}")
                p0, p1 = fresh(), fresh()
                lines.append(
                    f"  {p0} = getelementptr inbounds {_CLOSURE_PAIR_TY}, "
                    f"ptr {struct_ref(dst)}, i32 0, i32 0")
                lines.append(f"  store ptr @{mangle(lname)}, ptr {p0}")
                lines.append(
                    f"  {p1} = getelementptr inbounds {_CLOSURE_PAIR_TY}, "
                    f"ptr {struct_ref(dst)}, i32 0, i32 1")
                lines.append(f"  store ptr {envp}, ptr {p1}")
            elif rk == "resume":
                # Single-shot resume of this case's own continuation: the
                # runtime unparks the body and returns its completion value
                # (deep semantics) — or never returns on abort unwinding.
                #
                # TAIL POSITION (effect_tail.py, strict): when the resume's
                # value IS this case's return value with nothing after it,
                # emit the trampolined form — mx_resume_tail records (k, v)
                # and returns immediately, and the case RETURNS RIGHT HERE:
                # its frame must be gone before the pump switches into the
                # body from its CONSTANT frame.  That keeps handler-side
                # stack O(1) per element for stream-shaped handlers instead
                # of one (case + resume) frame pair per element.  The ops
                # the analysis proved to be a pure copy chain to the ret
                # are NOT emitted (the body has not run yet, so there is no
                # value to copy — mx_resume_tail's result is a dummy the
                # pump ignores; running word_into on it would copy out of a
                # null boundary box for aggregate kinds).  Writebacks and
                # frame frees run exactly as on the normal ret path.
                # Anything non-tail keeps the general recursive mx_resume.
                kp = use(opargs[0], lines)
                w = to_word(kind(opargs[1]), use(opargs[1], lines), lines,
                            src=opargs[1])
                if id(op) in tail_resume_set:
                    mod.runtime_syms.add("mx_resume_tail")
                    v = fresh()
                    lines.append(
                        f"  {v} = call i64 @mx_resume_tail(ptr {kp}, "
                        f"i64 {w})  ; tail resume: record for the pump")
                    emit_writebacks(lines)
                    emit_frees(lines)
                    if is_boundary_ret:
                        lines.append(
                            f"  ret i64 {v}  ; tail resume: back to the "
                            "pump (dummy word, ignored there)")
                    else:
                        dec, rv = _word_decode(v, sig.ret, fresh())
                        lines += dec
                        lines.append(
                            f"  ret {_llscalar(sig.ret)} {rv}  ; tail "
                            "resume: back to the pump (dummy, ignored "
                            "there)")
                    terminated = True
                    break
                mod.runtime_syms.add("mx_resume")
                v = fresh()
                lines.append(f"  {v} = call i64 @mx_resume(ptr {kp}, i64 {w})")
                word_into(dst, v, lines)
            elif rk == "handle_scope":
                site = rhs[1]
                rec = scopes.sites.get(site)
                if rec is None:
                    raise _Unsupported(f"handle site {site!r} unresolved")
                if rec.body_fn not in emitted_names or any(
                        hfn not in emitted_names
                        for (_o, _p, hfn) in rec.cases):
                    raise _Unsupported(
                        f"handle site {site!r} has non-emitted subfunctions")
                bsig = sigs[rec.body_fn]
                if bsig.params:
                    raise _Unsupported(
                        f"handle body {rec.body_fn!r} has an unexpected "
                        "signature")
                for (_opn, cparams, hfn) in rec.cases:
                    csig = sigs[hfn]
                    if len(csig.params) != len(cparams) + 1:
                        raise _Unsupported(
                            f"handler case {hfn!r} has an unexpected "
                            "signature")
                fields = list(scope_fields(site))
                mod.scope_env_types[site] = tuple(fields)
                envp = env_allocas.get(id(op))
                if envp is None:  # unreachable: prescan covers every site
                    raise _Unsupported("handle site missing env storage")
                ety = f"%henv.{_sanitize(site)}"
                for i, (cn, ck) in enumerate(fields):
                    vn = rec.cap_vals.get(cn, cn)
                    p = fresh()
                    lines.append(
                        f"  {p} = getelementptr inbounds {ety}, ptr {envp}, "
                        f"i32 0, i32 {i}")
                    if _is_cell_marker(ck):
                        # Mutable capture: the scope members share the
                        # binding through the cell pointer.
                        if vn not in cellset:
                            raise _Unsupported(
                                f"cell capture {cn!r} of non-cell "
                                f"variable {vn!r}")
                        lines.append(
                            f"  store ptr {cellp_ref(vn)}, ptr {p}"
                            f"  ; mutable capture: cell pointer for {cn}")
                    elif _is_agg(ck):
                        agg_copy(_agg_ty(ck), use(vn, lines), p, lines)
                    else:
                        lines.append(
                            f"  store {_llscalar(ck)} {use(vn, lines)}, "
                            f"ptr {p}")
                _emit_scope_artifacts(site, rec, sigs, mod)
                eg = mod.intern_string(rec.effect)
                mod.runtime_syms.add("mx_handle")
                v = fresh()
                lines.append(
                    f"  {v} = call i64 @mx_handle("
                    f"ptr @{_scope_body_sym(site)}, ptr {envp}, "
                    f"ptr @{_scope_disp_sym(site)}, ptr {envp}, "
                    f"ptr {eg}, ptr @{_scope_ops_sym(site)}, "
                    f"ptr @{_scope_np_sym(site)}, i64 {len(rec.cases)})"
                    f"  ; handle {rec.effect or '(any)'}")
                word_into(dst, v, lines)
            elif rk == "try_scope":
                # DELIMITED FAILURE RECOVERY (docs/try_catch.md): fill the
                # site's env, then mx_try(body, env, catch, env).  The
                # runtime installs a setjmp landing pad, runs the body, and
                # on a catchable failure anywhere in its dynamic extent
                # (including across coroutine boundaries) calls the catch
                # thunk with the failure's plain message — the interpreter's
                # `exc.message`, byte for byte.
                site = rhs[1]
                rec = scopes.sites.get(site)
                if rec is None or rec.kind != "try":
                    raise _Unsupported(f"try site {site!r} unresolved")
                if rec.body_fn not in emitted_names \
                        or rec.catch_fn not in emitted_names:
                    raise _Unsupported(
                        f"try site {site!r} has non-emitted subfunctions")
                bsig = sigs[rec.body_fn]
                if bsig.params:
                    raise _Unsupported(
                        f"try body {rec.body_fn!r} has an unexpected "
                        "signature")
                csig = sigs[rec.catch_fn]
                if len(csig.params) != 1 or csig.params[0] != STR:
                    raise _Unsupported(
                        f"catch subfunction {rec.catch_fn!r} does not take "
                        "exactly the failure message")
                fields = list(scope_fields(site))
                mod.scope_env_types[site] = tuple(fields)
                envp = env_allocas.get(id(op))
                if envp is None:  # unreachable: prescan covers every site
                    raise _Unsupported("try site missing env storage")
                ety = f"%henv.{_sanitize(site)}"
                for i, (cn, ck) in enumerate(fields):
                    vn = rec.cap_vals.get(cn, cn)
                    p = fresh()
                    lines.append(
                        f"  {p} = getelementptr inbounds {ety}, ptr {envp}, "
                        f"i32 0, i32 {i}")
                    if _is_cell_marker(ck):
                        if vn not in cellset:
                            raise _Unsupported(
                                f"cell capture {cn!r} of non-cell "
                                f"variable {vn!r}")
                        lines.append(
                            f"  store ptr {cellp_ref(vn)}, ptr {p}"
                            f"  ; mutable capture: cell pointer for {cn}")
                    elif _is_agg(ck):
                        agg_copy(_agg_ty(ck), use(vn, lines), p, lines)
                    else:
                        lines.append(
                            f"  store {_llscalar(ck)} {use(vn, lines)}, "
                            f"ptr {p}")
                _emit_try_artifacts(site, rec, sigs, mod)
                mod.runtime_syms.add("mx_try")
                v = fresh()
                lines.append(
                    f"  {v} = call i64 @mx_try("
                    f"ptr @{_try_body_sym(site)}, ptr {envp}, "
                    f"ptr @{_try_catch_sym(site)}, ptr {envp})"
                    f"  ; try/catch: {rec.body_fn} / {rec.catch_fn}")
                word_into(dst, v, lines)
            else:  # unreachable given analysis
                raise _Unsupported(f"op {rk!r} slipped past analysis")

            # Owned-string bookkeeping: every def of an owned string frees
            # the previously owned pointer and records the new provenance
            # (the analysis guarantees all defs are const/concat/
            # to_string/transfer-copy shapes).  A def is LITERAL when it is
            # a const or a transfer copy of a literal temp — interned
            # constants must never be recorded as freeable.
            if dst in owned_str_set and rk in ("const", "binop", "call",
                                               "copy"):
                is_lit = rk == "const" or (
                    rk == "copy" and opargs and opargs[0] in str_lit_temps)
                owned_str_update(dst, is_lit, lines)

        if bi == 0 and not terminated and hoist_box:
            # LOOP-INVARIANT BOXES: fill them once, here at the end of the
            # entry block (which dominates every other block), so the
            # boundary sites inside the loop reuse one box instead of
            # malloc'ing a fresh copy of the same unchanging bytes each
            # iteration.  Every hoisted value is defined by then (its only
            # def is a block-0 op or a parameter) and none of its boxing
            # sites is in block 0, so the register dominates all its uses.
            for hv in sorted(hoist_box):
                emit_hoisted_box(hv, lines)

        if not terminated:
            t = b.term
            if t[0] == "br":
                lines.append(f"  br label %bb{t[1]}")
            elif t[0] == "br_if":
                c = use(t[1], lines)
                cb = fresh()
                lines.append(f"  {cb} = icmp ne i64 {c}, 0")
                lines.append(f"  br i1 {cb}, label %bb{t[2]}, label %bb{t[3]}")
            elif t[0] == "ret":
                if sret:
                    if kind(t[1]) != sig.ret:
                        raise _Unsupported(
                            f"return value {t[1]!r} is {kind(t[1])}, "
                            f"function returns {sig.ret}")
                    # Write back rebound struct params FIRST: if the caller
                    # aliased its result slot with an argument (dst == arg),
                    # the interpreter's order makes the RESULT win, so the
                    # sret copy must come after the copy-outs.  Then copy the
                    # aggregate into the caller's slot BEFORE any frees (the
                    # returned value may live in a heap block).
                    emit_writebacks(lines)
                    src = use(t[1], lines)
                    agg_copy(_agg_ty(sig.ret), src, "%agg.ret", lines)
                    emit_frees(lines)
                    lines.append("  ret void")
                elif is_uniform_lambda or is_boundary_ret:
                    if _is_agg(sig.ret) and kind(t[1]) != sig.ret:
                        raise _Unsupported(
                            f"return value {t[1]!r} is {kind(t[1])}, "
                            f"function returns {sig.ret}")
                    rv = use(t[1], lines)
                    emit_writebacks(lines)
                    # Encode BEFORE the frees: an aggregate return boxes a
                    # copy (to_word), and the value may live in a heap
                    # block emit_frees is about to release.  When the value
                    # already IS an immortal boundary box, to_word passes
                    # its pointer through and no box is allocated at all.
                    rw = to_word(sig.ret, rv, lines, src=t[1])
                    emit_frees(lines)
                    what = ("word-uniform lambda return" if is_uniform_lambda
                            else "handle-scope boundary-word return")
                    lines.append(
                        f"  ret i64 {rw}  ; {what} ({sig.ret} encoded)")
                else:
                    rv = use(t[1], lines)
                    emit_writebacks(lines)
                    emit_frees(lines)
                    lines.append(f"  ret {_llscalar(sig.ret)} {rv}")
            else:  # ("unreachable",) placeholder terminator (no frees: dead end)
                lines.append("  unreachable")
        body.append(f"bb{bi}:")
        body.extend(lines)

    # Assemble: define header, entry block (allocas + heap mallocs + param
    # spills/byval-copies + capture loads), blocks.  A struct/enum return
    # prepends the caller's result slot as a leading `ptr %agg.ret`
    # parameter (sret-style); a lambda takes its env struct as a leading
    # `ptr %cl.env` parameter (after %agg.ret when both are present).
    pdecls = []
    uniform_decodes: List[str] = []
    if sret:
        pdecls.append("ptr %agg.ret")
    if info.is_lambda or info.is_scope_member:
        pdecls.append("ptr %cl.env")
    for p, pk in zip(info.params, sig.params):
        if is_uniform_lambda:
            # Word-uniform ABI: every parameter arrives as an i64 word;
            # non-i64 kinds are decoded to their typed `%a.<p>` name in
            # the prelude, so the body is oblivious to the ABI.
            if not _word_abi_ok(pk):  # unreachable given eligibility
                raise _Unsupported(
                    f"word-uniform lambda parameter {p!r} of kind {pk}")
            if _word_boxable(pk):
                # BOUNDARY BOX (increment 16): the word is the caller's
                # write-once box pointer.  Decoding it to `%a.<p>` (a ptr)
                # makes the ordinary aggregate-param prelude below — the
                # byval copy-out into this frame's own storage, or the
                # elide-copy read-through — the exact copy-out contract.
                if p in writeback_params:  # unreachable given eligibility
                    raise _Unsupported(
                        f"word-uniform lambda write-back parameter {p!r} "
                        f"of kind {pk} (a boundary box cannot carry the "
                        "copy-out back to the caller)")
                pdecls.append(f"i64 %aw.{_sanitize(p)}")
                uniform_decodes.append(
                    f"  %a.{_sanitize(p)} = inttoptr i64 %aw.{_sanitize(p)} "
                    f"to ptr  ; word-uniform param {p}: boundary box {pk}")
            elif _llscalar(pk) == "i64":
                pdecls.append(f"i64 %a.{_sanitize(p)}")
            else:
                pdecls.append(f"i64 %aw.{_sanitize(p)}")
                dec, _v = _word_decode(
                    f"%aw.{_sanitize(p)}", pk, f"%a.{_sanitize(p)}")
                uniform_decodes.extend(
                    ln + f"  ; word-uniform param {p}: {pk} decoded"
                    for ln in dec)
        else:
            pdecls.append(f"{_llparam(pk)} %a.{_sanitize(p)}")
    rty = ("i64" if (is_uniform_lambda or is_boundary_ret)
           else ("void" if sret else _llscalar(sig.ret)))
    out = [f"define {rty} @{mangle(f.name)}({', '.join(pdecls)}) {{"
           + ("  ; word-uniform lambda ABI" if is_uniform_lambda else
              ("  ; handle-scope boundary-word ABI" if is_boundary_ret
               else ""))]
    entry: List[str] = list(uniform_decodes)
    for n in slots:
        entry.append(f"  {slot_ref(n)} = alloca {llty(n)}  ; mir slot: {n}")
    for n in agg_vars:
        if n in heapset:
            continue  # heap-backed: malloc'd below instead of an alloca
        if n in elided_params:
            continue  # reads through the caller's pointer: no storage
        if n in boxview:
            # Box view: a pointer slot aliasing the write-once box instead
            # of an aggregate copied out of it (null until the read).
            entry.append(
                f"  {bp_ref(n)} = alloca ptr  ; box view: {n}")
            entry.append(f"  store ptr null, ptr {bp_ref(n)}")
            continue
        entry.append(
            f"  {struct_ref(n)} = alloca {_agg_ty(kind(n))}  ; aggregate: {n}")
    for n in owned_strs:
        entry.append(
            f"  {strown_ref(n)} = alloca ptr  ; owned string shadow: {n}")
        entry.append(f"  store ptr null, ptr {strown_ref(n)}")
    entry.extend(env_entry)
    # Mutable-capture cells owned by this frame: one malloc(8) word box
    # each, NEVER freed (leak by design: the cell may be aliased by any
    # env that captured its pointer, and an immortal cell cannot dangle).
    # Cell CAPTURES (cap_cellset) load their pointer from the env below.
    for n in sorted(cellset - cap_cellset):
        mod.uses_malloc = True
        entry.append(
            f"  {cellp_ref(n)} = call ptr @malloc(i64 8)"
            f"  ; mutable-capture cell for {n} (leaks by design)")
    for n in heap_vars:
        sname = _struct_name(kind(n))
        # Recursive layout size: leaf cells are 8 bytes, nested aggregates
        # inline, boxed payload slots are 8-byte pointers.
        size = _kind_size(kind(n), structs, variants)
        if size is None:  # unreachable: consistency demoted infinite layouts
            raise _Unsupported(f"@global struct {sname!r} has infinite layout")
        mod.uses_malloc = True
        entry.append(
            f"  {struct_ref(n)} = call ptr @malloc(i64 {max(size, 8)})"
            f"  ; @global struct {n}: {sname}, freed on ret paths")
    for p in info.params:
        if p in elided_params:
            # COPY ELISION (a): this param is never rebound and never
            # reaches a write-back position, so nothing ever writes its
            # storage — read the caller's aggregate through its pointer.
            entry.append(f"  ; elide-copy: param {p} reads through the "
                         "caller's pointer (never rebound, no write-back)")
        elif p in aggset:
            # byval-copy: the caller passed a pointer to ITS storage; copy the
            # aggregate into this frame's own storage to preserve MIR value
            # semantics (rebound params also copy back OUT on ret paths).
            agg_copy(_agg_ty(kind(p)), f"%a.{_sanitize(p)}",
                     struct_ref(p), entry)
        elif p in cellset:
            # A cell-backed parameter: seed the fresh cell with the
            # incoming value (the caller passed by value, as always).
            entry.append(
                f"  store {llty(p)} %a.{_sanitize(p)}, ptr {cellp_ref(p)}"
                f"  ; cell-backed parameter {p}")
        elif p in slotset:
            entry.append(f"  store {llty(p)} %a.{_sanitize(p)}, ptr {slot_ref(p)}")
    if info.is_lambda or info.is_scope_member:
        # Reload captures from the env struct (the creator stored them at
        # the make_closure / handle_scope site, eagerly, by value).  A
        # scope member reloads only its OWN free names, indexed into the
        # site's shared field order.
        if info.is_lambda:
            fields = env_fields(f.name)
            mod.env_types[f.name] = fields
            ety = f"%env.{_sanitize(f.name)}"
        else:
            fields = scope_fields(info.scope_site)
            mod.scope_env_types[info.scope_site] = fields
            ety = f"%henv.{_sanitize(info.scope_site)}"
        wanted = set(info.env_captures)
        for i, (cn, ck) in enumerate(fields):
            if cn not in wanted:
                continue
            p = f"%capp.{_sanitize(cn)}"
            entry.append(
                f"  {p} = getelementptr inbounds {ety}, ptr %cl.env, "
                f"i32 0, i32 {i}")
            if _is_cell_marker(ck):
                # Mutable capture: the env field holds the shared cell's
                # POINTER — load it, and all reads/writes go through it.
                entry.append(
                    f"  {cellp_ref(cn)} = load ptr, ptr {p}"
                    f"  ; mutable capture: shared cell pointer for {cn}")
            elif cn in cellset:
                # Value capture that THIS function wraps into its own
                # fresh cell (a sub-function of ours assigns it): seed the
                # cell with the captured value.
                v = f"%capv.{_sanitize(cn)}"
                entry.append(f"  {v} = load {_llscalar(ck)}, ptr {p}")
                entry.append(
                    f"  store {_llscalar(ck)} {v}, ptr {cellp_ref(cn)}"
                    f"  ; cell-backed capture {cn}: seeded from the env")
            elif cn in aggset:
                agg_copy(_agg_ty(ck), p, struct_ref(cn), entry)
            elif cn in slotset:
                v = f"%capv.{_sanitize(cn)}"
                entry.append(f"  {v} = load {_llscalar(ck)}, ptr {p}")
                entry.append(f"  store {_llscalar(ck)} {v}, ptr {slot_ref(cn)}")
            else:
                entry.append(
                    f"  %cap.{_sanitize(cn)} = load {_llscalar(ck)}, ptr {p}")
    entry.append("  br label %bb0")
    out.append("entry:")
    out.extend(entry)
    out.extend(body)
    out.append("}")
    return "\n".join(out)


# ---------------------------------------------------------------------------
# Runtime support text (string pool, printf helpers, declares, struct types)
# ---------------------------------------------------------------------------

_PRINT_FMTS = {
    "i64": ("@.fmt.i64", "%lld\n"),
    # f64 prints through mx_f64_to_str (Python-repr parity; %g diverged
    # from the interpreter on integral floats), so its directive is %s.
    "f64": ("@.fmt.f64", "%s\n"),
    "str": ("@.fmt.str", "%s\n"),
}
_PRINT_ARG = {"i64": "i64", "f64": "double", "str": "ptr"}


def _string_global(name: str, content: str) -> str:
    data = content.encode("utf-8") + b"\x00"
    return (f"{name} = private unnamed_addr constant "
            f"[{len(data)} x i8] c\"{_escape_bytes(data)}\"")


def _emit_runtime(mod: _ModuleState) -> List[str]:
    chunks: List[str] = []
    if mod.strings:
        chunks.append("\n".join(
            _string_global(gname, content)
            for content, gname in sorted(mod.strings.items(), key=lambda kv: kv[1])))
    decls: List[str] = []
    if F64 in mod.print_helpers:
        # The f64 print helper renders through mx_f64_to_str (parity).
        mod.runtime_syms.add("mx_f64_to_str")
        mod.runtime_syms.add("mx_str_free")
    if mod.print_helpers or mod.uses_printf:
        decls.append("declare i32 @printf(ptr, ...)")
    if mod.uses_abort:
        decls.append("declare void @abort() noreturn")
    if mod.uses_malloc:
        decls.append("declare noalias ptr @malloc(i64)")
        decls.append("declare void @free(ptr)")
    # Extern C FFI symbols (real libc, C signatures; fclose returns C int).
    _EXTERN_C_DECLS = {
        "memcpy": "declare ptr @memcpy(ptr, ptr, i64)",
        "realloc": "declare ptr @realloc(ptr, i64)",
        "fopen": "declare noalias ptr @fopen(ptr, ptr)",
        "fclose": "declare i32 @fclose(ptr)",
    }
    for name in sorted(mod.extern_c_syms):
        decls.append(_EXTERN_C_DECLS[name])
    for name in sorted(mod.math_used):
        decls.append(f"declare double @{name}(double)")
    # Native metaxu runtime symbols (metaxu_rt.c, linked by llvm_run).
    for name in sorted(mod.runtime_syms):
        rt, params = _RT_SIGS[name]
        decls.append(f"declare {rt} @{name}({', '.join(params)})"
                     f"{_RT_ATTRS.get(name, '')}")
    if mod.uses_tls_permit:
        # Per-thread write permit (metaxu_threads.c), read directly by the
        # inlined Vec-mutator guards.  initialexec mirrors the C side's
        # tls_model("initial-exec") — always valid: the runtime links into
        # executables, never dlopen'd libraries.
        decls.append("@mx__tls_write_permit = external thread_local"
                     "(initialexec) global i64, align 8")
    if mod.uses_vec_tbaa:
        # TBAA domain for the inline Vec fast paths: header words (len/
        # cap/data/contended) and element words are DISTINCT allocations
        # by the runtime's contract, so their access tags never alias —
        # which keeps hoisted header loads live across element stores.
        # Untagged accesses (everything else the backend emits) stay
        # compatible with both.
        decls.append('!0 = !{!"metaxu TBAA root"}')
        decls.append('!1 = !{!"mx.vec.header", !0, i64 0}')
        decls.append('!2 = !{!"mx.vec.elem", !0, i64 0}')
        decls.append("!3 = !{!1, !1, i64 0}")
        decls.append("!4 = !{!2, !2, i64 0}")
    if decls:
        chunks.append("\n".join(decls))
    if mod.print_helpers:
        fmts = "\n".join(_string_global(g, f)
                         for k, (g, f) in _PRINT_FMTS.items() if k in mod.print_helpers)
        chunks.append(fmts)
        for k in sorted(mod.print_helpers):
            g, _ = _PRINT_FMTS[k]
            aty = _PRINT_ARG[k]
            if k == F64:
                # Interpreter parity: print(1.0) is "1.0", not %g's "1".
                # Route through mx_f64_to_str (Python-repr float text,
                # already the to_string contract) instead of a %g printf —
                # the tile differentials caught %g diverging on integral
                # floats.  The fresh string is freed right after.
                chunks.append("\n".join([
                    f"define internal void @metaxu_print_{k}({aty} %x) {{",
                    "entry:",
                    "  %s = call ptr @mx_f64_to_str(double %x)",
                    f"  %r = call i32 (ptr, ...) @printf(ptr {g}, ptr %s)",
                    "  call void @mx_str_free(ptr %s)",
                    "  ret void",
                    "}",
                ]))
                continue
            chunks.append("\n".join([
                f"define internal void @metaxu_print_{k}({aty} %x) {{",
                "entry:",
                f"  %r = call i32 (ptr, ...) @printf(ptr {g}, {aty} %x)",
                "  ret void",
                "}",
            ]))
    return chunks


def _emit_struct_types(structs: _StructTable, used: Set[str]) -> Optional[str]:
    lines: List[str] = []
    for sname in sorted(used):
        if sname in structs.bad:
            continue
        # Nested aggregate fields inline their named type (LLVM permits
        # forward references between named types, so order is free).
        ftys = ", ".join(_llcell(structs.field_kind(sname, fn_))
                         for fn_ in structs.fields.get(sname, ()))
        fields_desc = ", ".join(structs.fields.get(sname, ()))
        lines.append(f"%struct.{_sanitize(sname)} = type {{ {ftys} }}  ; {fields_desc}")
    return "\n".join(lines) if lines else None


def _emit_enum_types(variants: _VariantTable, used: Set[str]) -> Optional[str]:
    """%enum.E tagged-union types plus the documented tag mapping and the
    per-variant payload slot kinds (canonical module-wide cells; slots whose
    stores disagree across instantiations are flagged 'mixed per value')."""
    lines: List[str] = []
    if used and variants.tags:
        pairs = ", ".join(f"{n}={i}" for n, i in sorted(
            variants.tags.items(), key=lambda kv: kv[1]))
        lines.append(f"; variant tag mapping (module-wide, dense): {pairs}")
    for ename in sorted(used):
        n = variants.payload_max.get(ename, 0)
        vnames = sorted(variants.variants_of.get(ename, ()))
        tag_doc = ", ".join(
            f"{v}={variants.tags[v]}" for v in vnames if v in variants.tags)
        lines.append(
            f"{_enum_llname(ename)} = type {{ i64, [{n} x i64] }}"
            f"  ; tag + {n} payload slots" + (f"; tags: {tag_doc}" if tag_doc else ""))
        for v in vnames:
            arity = variants.arity.get((ename, v), 0)
            slot_docs = []
            for i in range(arity):
                ck = variants.cell_kind(ename, v, i)
                doc = f"boxed {ck}" if _is_agg(ck) else ck
                if (ename, v, i) in variants.mixed:
                    doc += " (mixed per value)"
                slot_docs.append(doc)
            lines.append(
                f";   variant {v}({', '.join(slot_docs)})")
    return "\n".join(lines) if lines else None


def _emit_closure_types(mod: _ModuleState) -> Optional[str]:
    """%mx.closure pair + per-lambda %env.L structs (env field order is the
    make_closure capture-list order)."""
    lines: List[str] = []
    if mod.uses_closure_pair:
        lines.append(f"{_CLOSURE_PAIR_TY} = type {{ ptr, ptr }}  ; {{ fn, env }}")
    for lname in sorted(mod.env_types):
        fields = mod.env_types[lname]
        ftys = ", ".join(
            _agg_ty(k) if _is_agg(k) else _llscalar(k)
            for (_cn, k) in fields)
        desc = ", ".join(cn for (cn, _k) in fields)
        body = f"{{ {ftys} }}" if ftys else "{}"
        lines.append(
            f"%env.{_sanitize(lname)} = type {body}  ; captures: {desc or '(none)'}")
    return "\n".join(lines) if lines else None


# ---------------------------------------------------------------------------
# Module driver
# ---------------------------------------------------------------------------

def emit_llvm(funcs: Sequence[MirFunc]) -> str:
    """Emit one LLVM IR module (text) for a MIR module.

    Direct functions get full definitions; everything else gets a
    comment-only placeholder carrying its reasons (see module docstring).
    """
    # Fresh symbol-sanitizer registry per module: deterministic, injective
    # symbol mangling independent of previously-emitted modules.
    _sanitize_reset()
    module_names = {f.name for f in funcs}
    structs = _build_struct_table(funcs)
    closures = _build_closure_table(funcs)
    # Arm the kind lattice's closure-arity oracle for this module: _join
    # merges same-arity closure kinds into dynamic member sets (see
    # _DYN_CLOSURE_PREFIX); arities are static per make_closure site.
    _CLOSURE_ARITY.clear()
    _CLOSURE_ARITY.update(closures.arity)
    traits = _build_trait_table(module_names)
    scopes = _build_scope_table(funcs)
    cells = _build_cell_table(funcs, scopes)
    gtable = _build_global_table(funcs)
    infos = [_analyze(f, module_names, closures, scopes, traits,
                      cells, gtable)
             for f in funcs]
    variants = _build_variant_table(funcs, infos)

    def dep_names(info: _Info, kinds: Dict[str, str]) -> Set[str]:
        """Module functions this one references and cannot link without:
        direct callees, make_closure targets, resolved closure callees, and
        statically-resolved trait-call targets.  A callee a module function
        defines IS a dependency even when a builtin shares the name — the
        user function wins for plain calls (NAME PRECEDENCE); the builtin
        forms (__builtin$m and unshadowed names) resolve to no module
        function and never count."""
        deps = {callee for (_d, callee, _a) in info.calls
                if callee in module_names}
        # Statically-resolved __static$ calls depend on their target fn.
        for (_d, callee, _a) in info.calls:
            if callee.startswith(STATIC_CALL_PREFIX):
                sres, starget = _resolve_static_call(
                    callee, traits, module_names)
                if sres == "func" and starget in module_names:
                    deps.add(starget)
        deps |= {lname for (_d, lname, _c) in info.closure_defs
                 if lname in module_names}
        # Statically default-resolved performs call the default fn directly.
        deps |= {dfn for dfn in info.default_performs.values()
                 if dfn in module_names}
        # Dynamically default-routed performs reach it through the per-op
        # thunk they hand to mx_perform_or_default: same link dependency.
        deps |= {dfn for dfn in info.dynamic_default_performs.values()
                 if dfn in module_names}
        # A comprehension site's per-element thunk calls the lambda symbol.
        for (_d, callee, cargs) in info.calls:
            if callee == "__vec_comprehension" and len(cargs) == 3:
                ck = kinds.get(cargs[1], I64)
                if _is_closure(ck) and _closure_lambda(ck) in module_names:
                    deps.add(_closure_lambda(ck))
        # A handle site's owner cannot link without its body/case
        # subfunctions (the site's shims call them).
        for site in scopes.sites_of_owner.get(info.f.name, ()):
            deps |= {m for m in scopes.sites[site].member_fns()
                     if m in module_names}
        for (_d, cvar, _a) in info.closure_calls:
            ck = kinds.get(cvar, I64)
            deps.update(m for m in _closure_members(ck)
                        if m in module_names)
        # The EFFECT_SPAWN thunk's contention-marking walk branches on the
        # spawned closure's member lambda SYMBOLS (@mx_<lambda>), so the
        # thunk cannot link if any member demoted.
        for (_pd, symbol, pargs) in info.effect_primitive_calls:
            if symbol == "EFFECT_SPAWN" and pargs:
                deps.update(m for m in _closure_members(kinds.get(pargs[0], I64))
                            if m in module_names)
        # Reading a module constant is meaningless unless its initializer
        # emitted (the entry wrapper must be able to run it first): a
        # demoted __module_init cascades onto every global reader.
        if info.global_reads and _MODULE_INIT in module_names:
            deps.add(_MODULE_INIT)
        for (_d, method, targs) in info.trait_calls:
            if not targs:
                continue
            res, target = _resolve_trait_call(
                method, kinds.get(targs[0], I64), traits, module_names,
                assume_final=True)
            if res == "func" and target in module_names:
                deps.add(target)
        return deps

    # Duplicate MIR function names (e.g. lambda counters restarting per
    # enclosing function) would produce colliding symbols and ambiguous
    # direct calls: demote every function carrying a duplicated name.
    seen: Dict[str, int] = {}
    for f in funcs:
        seen[f.name] = seen.get(f.name, 0) + 1
    for info in infos:
        if seen[info.f.name] > 1:
            info.add_reason(
                f"duplicate function name {info.f.name!r} in module (ambiguous symbol)")

    # Module-wide kind/signature fixpoint (params/ret and struct field cells
    # promoted monotonically; callers and callees feed each other).  Two
    # phases: the first never assumes anything about trait-call receivers
    # still at the i64 bottom; once it converges, the second treats those
    # receivers as genuinely int (nothing else can promote them anymore)
    # and keeps iterating so the late resolutions' kinds propagate.
    sigs: Dict[str, _Sig] = {
        info.f.name: _Sig(params=[I64] * len(info.params)) for info in infos}
    kind_sets: Dict[str, Dict[str, str]] = {}
    candidates = [info for info in infos if not info.reasons]
    for assume_final in (False, True):
        for _round in range(12):
            changed = False
            for info in candidates:
                kinds, cell_changed = _infer_kinds(
                    info, sigs, structs, variants, closures, traits, scopes,
                    module_names, gtable, assume_final=assume_final)
                changed = changed or cell_changed
                if kind_sets.get(info.f.name) != kinds:
                    kind_sets[info.f.name] = kinds
                    changed = True
                own = sigs[info.f.name]
                for i, p in enumerate(info.params):
                    if p in info.promote_params:
                        # promote_matrix'd param: the local kind is the
                        # PROMOTED form; the sig keeps the caller-side kind
                        # (joined at call sites only).
                        continue
                    nk = _join(own.params[i], kinds.get(p, I64))
                    if nk != own.params[i]:
                        own.params[i] = nk
                        changed = True
                for r in info.ret_vars:
                    nk = _join(own.ret, kinds.get(r, I64))
                    if nk != own.ret:
                        own.ret = nk
                        changed = True
                resolved_calls = []
                for (dst, callee, args) in info.calls:
                    bname = _builtin_name(callee, module_names)
                    if bname in _NATIVE_RT_CALLS or bname in _FFI_CALLS \
                            or bname == "assert":
                        continue
                    if callee.startswith(STATIC_CALL_PREFIX):
                        sres, starget = _resolve_static_call(
                            callee, traits, module_names)
                        if sres == "func":
                            resolved_calls.append((dst, starget, args))
                        continue
                    resolved_calls.append((dst, callee, args))
                for (dst, cvar, args) in info.closure_calls:
                    ck = kinds.get(cvar, I64)
                    for m in _closure_members(ck):
                        resolved_calls.append((dst, m, args))
                for (dst, method, targs) in info.trait_calls:
                    if not targs:
                        continue
                    res, target = _resolve_trait_call(
                        method, kinds.get(targs[0], I64), traits, module_names,
                        assume_final=assume_final)
                    if res == "func":
                        resolved_calls.append((dst, target, targs))
                for (dst, callee, args) in resolved_calls:
                    csig = sigs.get(callee)
                    if csig is None or len(csig.params) != len(args):
                        continue
                    for i, a in enumerate(args):
                        nk = _join(csig.params[i], kinds.get(a, I64))
                        if nk != csig.params[i]:
                            csig.params[i] = nk
                            changed = True
                    nk = _join(csig.ret, kinds.get(dst, I64))
                    if nk != csig.ret:
                        csig.ret = nk
                        changed = True
            if not changed:
                break

    # A lambda whose closure is RETURNED by any function needs a heap env:
    # the pair crosses the creating frame's boundary, so a stack env would
    # dangle.  sig.ret is the join of every ret-var kind across the module
    # fixpoint, so this is complete for the emitted subset (storing a pair
    # in a field/payload demotes the storing function instead; env-captured
    # pairs are marked below).  Over-marking is sound — a heap env only
    # leaks, it can never dangle.
    for sig in sigs.values():
        for m in _closure_members(sig.ret):
            if m in module_names:
                closures.heap_env.add(m)

    # SPAWNED CLOSURES (docs/threads_runtime.md): a closure kind reaching
    # an __effect_runtime$ thunk's parameter is handed to
    # mx_thread_spawn, whose child thread dereferences the env AFTER the
    # spawning frame has moved on — force every member lambda heap-env
    # (immortal, leak by design; it can never dangle).  Members are
    # already participants (a thunk parameter IS a function parameter
    # position), so a word-encodable signature puts them on the
    # word-uniform ABI the C child entry invokes; the consistency check
    # demotes anything that cannot take it.
    for fname, fsig in sigs.items():
        if fname.startswith(_EFFECT_RUNTIME_PREFIX):
            for pk in fsig.params:
                closures.heap_env.update(
                    m for m in _closure_members(pk) if m in module_names)

    # WORD-UNIFORM PARTICIPATION (increment 13).  A lambda participates in
    # the indirect-call ABI — `i64 (ptr env, i64 args...)`, boundary word
    # conventions for BOTH params and return — when its closure leaves
    # simple local flow: it reaches a function parameter position, it is
    # merged with another lambda into a dynamic kind anywhere, or it is
    # captured into an env (a handle-site env or another closure's env —
    # the aggregate/effect boundary).  Everything else keeps its typed
    # signature (no regression to the comprehension/SIMD/aggregate paths).
    # Only lambdas whose whole signature has a word encoding are marked;
    # an aggregate-signatured participant stays typed and any indirect
    # site naming it demotes with a scalar-only reason.  Env-captured
    # members are additionally marked heap-env: the pair stored in an env
    # could outlive the lambda's creating frame, and an immortal env can
    # never dangle.
    participants: Set[str] = set()
    # Members of some DYNAMIC closure kind: these lambdas share one call
    # site with another lambda, so they MUST agree on one native ABI.
    dyn_members: Set[str] = set()
    for sig in sigs.values():
        for pk in sig.params:
            participants.update(_closure_members(pk))
    for store_kind in (
            [k for ks in kind_sets.values() for k in ks.values()]
            + [k for sig in sigs.values() for k in (*sig.params, sig.ret)]
            + list(gtable.kinds.values())
            + list(scopes.value_cells.values())
            + list(scopes.op_results.values())
            + list(scopes.op_args.values())
            + list(structs.kinds.values())
            + list(variants.cells.values())):
        if _is_dyn_closure(store_kind):
            participants.update(_closure_members(store_kind))
            dyn_members.update(_closure_members(store_kind))
    for env_kind in list(closures.cells.values()) + list(scopes.cells.values()):
        ms = _closure_members(env_kind)
        participants.update(ms)
        if _is_dyn_closure(env_kind):
            dyn_members.update(ms)
        closures.heap_env.update(m for m in ms if m in module_names)

    # BOUNDARY-CROSSING CLOSURES (increment 14): a closure kind reaching
    # any effect-boundary cell (perform args ⊔ case params, perform
    # results ⊔ resume values, handle value ⊔ body/case returns) crosses
    # scopes as a boxed {fn, env} pair, so every member lambda is forced
    # heap-env — the same rule as env-captured pairs: the boxed pair may
    # outlive the creating frame, and an immortal env can never dangle.
    # (No word-uniform participation is implied: a pinned member keeps its
    # typed signature; dynamic boundary kinds were already marked above.)
    for bk in (list(scopes.op_args.values())
               + list(scopes.op_results.values())
               + list(scopes.value_cells.values())):
        closures.heap_env.update(
            m for m in _closure_members(bk) if m in module_names)

    # Why a PARTICIPATING lambda cannot take the word-uniform ABI, when it
    # cannot — recorded so an indirect site naming it demotes with the real
    # reason instead of a generic "participation analysis missed a flow".
    _info_by_name = {info.f.name: info for info in infos}
    word_blocked: Dict[str, str] = {}

    def _word_eligible(m: str) -> bool:
        s = sigs.get(m)
        if s is None or m not in module_names:
            return False
        for k in [*s.params, s.ret]:
            if not _word_abi_ok(k):
                return False
            if _word_boxable(k) and _kind_size(k, structs, variants) is None:
                word_blocked[m] = (
                    f"its signature kind {k} has an infinite layout, so no "
                    "boundary box can be sized")
                return False
        # A @mut aggregate parameter has WRITE-BACK semantics (the callee
        # copies it out through the caller's pointer on ret).  Through the
        # word ABI the caller's pointer is a fresh boundary box the caller
        # drops, so the write-back would be lost — demote instead.
        minfo = _info_by_name.get(m)
        if minfo is not None:
            muts = set(getattr(minfo.f, "mut_params", ()) or ())
            for p, pk in zip(minfo.params, s.params):
                if p in muts and _word_boxable(pk):
                    word_blocked[m] = (
                        f"its @mut parameter {p!r} of kind {pk} writes back "
                        "through the caller's pointer, which the indirect "
                        "ABI's boundary box cannot carry back")
                    return False
        return True

    def _sig_has_agg(m: str) -> bool:
        s = sigs.get(m)
        return s is not None and any(
            _word_boxable(k) for k in [*s.params, s.ret])

    # An AGGREGATE-signatured lambda takes the uniform ABI only when it is
    # actually a member of a dynamic kind — only then must it agree on one
    # native signature with another lambda.  A lambda pinned at every site
    # keeps its typed signature (ptr params, sret return), so a pinned
    # aggregate closure call still costs ZERO allocation; only aggregates
    # on a genuinely dynamic edge box (increment 16, work item 2).
    word_uniform = {m for m in participants
                    if _word_eligible(m)
                    and (m in dyn_members or not _sig_has_agg(m))}
    word_blocked = {m: r for m, r in word_blocked.items()
                    if m not in word_uniform}

    # Mark module-wide variant cells whose stores disagree post-fixpoint:
    # the joined cell kind is then NOT the representation every writer used
    # (multi-instantiation slots like a generic Option holding ints in one
    # use and structs in another), so nested extraction through the
    # canonical cells must demote rather than guess.  Per-value refinements
    # are unaffected — each value still knows its own representation.
    for info in candidates:
        kinds = kind_sets.get(info.f.name, {})
        for b in info.f.blocks:
            for op in b.ops:
                if op[0] != "let" or len(op) != 4 or op[2][0] != "make_variant":
                    continue
                ename, vname = op[2][1], op[2][2]
                for i, a in enumerate(op[3]):
                    if _strip_refinement(kinds.get(a, I64)) != \
                            variants.cell_kind(ename, vname, i):
                        variants.mixed.add((ename, vname, i))

    # Same rule for TUPLE structs, whose field joins are one-way for the
    # same reason (see the alloc_struct branch of _infer_kinds).  `__tuple2`
    # is EVERY 2-tuple in the module, so a module building both `(1, 2)` and
    # `(1.0, 2.5)` gives `_1of2` one native layout that not every store
    # site actually uses.  Emitting through the join would retype an int
    # literal as a double; the struct is marked bad instead, so every
    # function that touches a tuple of that arity demotes with a reason.
    for info in candidates:
        kinds = kind_sets.get(info.f.name, {})
        for b in info.f.blocks:
            for op in b.ops:
                if op[0] != "let" or len(op) != 4 or op[2][0] != "alloc_struct":
                    continue
                sname = op[2][1]
                if sname in structs.bad or not _is_tuple_struct(sname):
                    continue
                for (fn_, fv) in op[3]:
                    fk = structs.field_kind(sname, fn_)
                    vk = kinds.get(fv, I64)
                    if vk != fk:
                        structs.bad[sname] = (
                            f"tuple struct {sname!r} has conflicting element "
                            f"representations: field {fn_!r} is {fk} module-"
                            f"wide but {fv!r} is {vk} here (tuples of the "
                            "same arity share one native layout, and this "
                            "module builds two different tuple types)")
                        break

    # Post-fixpoint consistency; anything wrong becomes a placeholder reason.
    for info in candidates:
        kinds = kind_sets.get(info.f.name, {})
        for p in _check_consistency(info, kinds, sigs, structs, variants,
                                    closures, traits, scopes, module_names,
                                    cells, gtable, word_uniform, word_blocked):
            info.add_reason(p)

    # WRITE-BACK MAP (increment 8, for copy elision): per function, the
    # argument positions whose struct param has by-reference semantics
    # (MirFunc.mut_params: @mut / method receiver) AND is rebound — the
    # callee copies that param back out through the caller's pointer on
    # ret, so callers must never alias such a position with elided
    # (copy-free) storage.  Plain params never write back (value
    # semantics, interpreter parity); lambdas write back exactly their
    # @mut params; handle-scope subfunctions never write back.
    writeback_map: Dict[str, frozenset] = {}
    for info in infos:
        ks = kind_sets.get(info.f.name, {})
        mut = frozenset(getattr(info.f, "mut_params", ()) or ())
        if info.is_scope_member:
            writeback_map[info.f.name] = frozenset()
        else:
            # All @mut struct positions (mirrors _emit_function's
            # writeback_params: nested calls can mutate an un-rebound @mut
            # param's storage, so every @mut struct param copies out).
            writeback_map[info.f.name] = frozenset(
                i for i, p in enumerate(info.params)
                if p in mut and _is_struct(ks.get(p, I64)))

    # A function referencing a placeholder cannot link: cascade demotion
    # (direct calls, make_closure fn-pointer targets, closure calls).
    emitted = {info.f.name for info in infos if not info.reasons}
    while True:
        demoted = False
        for info in infos:
            if info.reasons or info.f.name not in emitted:
                continue
            for dep in sorted(dep_names(info, kind_sets.get(info.f.name, {}))):
                if dep not in emitted:
                    info.add_reason(
                        f"calls function {dep!r} that is itself a placeholder")
                    emitted.discard(info.f.name)
                    demoted = True
                    break
        if not demoted:
            break

    # Emission (a late _Unsupported also demotes, then cascades once more).
    mod = _ModuleState()
    used_structs: Set[str] = set()
    emitted_chunks: Dict[str, str] = {}
    progress = True
    while progress:
        progress = False
        for info in infos:
            name = info.f.name
            if name not in emitted or name in emitted_chunks:
                continue
            kinds = kind_sets.get(name, {})
            try:
                chunk = _emit_function(info, kinds, sigs, structs, variants,
                                       closures, traits, scopes,
                                       module_names, mod, emitted,
                                       writeback_map, cells, gtable,
                                       word_uniform)
            except _Unsupported as exc:
                info.add_reason(exc.reason)
            except Exception as exc:  # never crash the pipeline
                info.add_reason(f"emission error: {type(exc).__name__}: {exc}")
            else:
                emitted_chunks[name] = chunk
                used_structs |= {_struct_name(k) for k in kinds.values() if _is_struct(k)}
                mod.used_enums |= {_enum_name(k) for k in kinds.values()
                                   if _is_enum(k)}
                if any(_is_closure(k) for k in kinds.values()):
                    mod.uses_closure_pair = True
                continue
            # Demotion during emission: drop this function and everything
            # already emitted that references it, then redo the affected ones.
            emitted.discard(name)
            for other in infos:
                if other.f.name in emitted and name in dep_names(
                        other, kind_sets.get(other.f.name, {})):
                    other.add_reason(
                        f"calls function {name!r} that is itself a placeholder")
                    emitted.discard(other.f.name)
                    emitted_chunks.pop(other.f.name, None)
            progress = True

    # Handle-site artifact liveness: an env TYPE is referenced by the owner
    # (env fill) and by every member's prelude, so it stays whenever any of
    # them emitted; the shims/tables call the members and are referenced
    # only by the owner's mx_handle call, so they need owner AND members.
    live_scope_envs = {
        site for site in mod.scope_env_types
        if (site in scopes.sites
            and (scopes.sites[site].owner in emitted_chunks
                 or any(m in emitted_chunks
                        for m in scopes.sites[site].member_fns())))}
    live_scope_shims = {
        site for site, (owner, members) in mod.scope_sites.items()
        if owner in emitted_chunks
        and all(m in emitted_chunks for m in members)}

    # Close the used type sets over nested references: inline struct fields
    # and boxed enum payload slots name %struct/%enum types that may never
    # appear as a local variable kind, and env structs may inline aggregates.
    for fields in mod.env_types.values():
        for (_cn, k) in fields:
            if _is_struct(k):
                used_structs.add(_struct_name(k))
            elif _is_enum(k):
                mod.used_enums.add(_enum_name(k))
    for site in live_scope_envs:
        for (_cn, k) in mod.scope_env_types[site]:
            if _is_struct(k):
                used_structs.add(_struct_name(k))
            elif _is_enum(k):
                mod.used_enums.add(_enum_name(k))
    while True:
        more_structs: Set[str] = set()
        more_enums: Set[str] = set()
        for sname in used_structs:
            for fn_ in structs.fields.get(sname, ()):
                fk = structs.field_kind(sname, fn_)
                # A referenced type with an infinite layout is never emitted
                # (every function touching it was demoted): do not pull it.
                if _kind_size(fk, structs, variants) is None:
                    continue
                if _is_struct(fk):
                    more_structs.add(_struct_name(fk))
                elif _is_enum(fk):
                    more_enums.add(_enum_name(fk))
        for ename in mod.used_enums:
            for (e2, _v2, _i2), ck in variants.cells.items():
                if e2 != ename:
                    continue
                if _kind_size(ck, structs, variants) is None:
                    continue
                if _is_struct(ck):
                    more_structs.add(_struct_name(ck))
                elif _is_enum(ck):
                    more_enums.add(_enum_name(ck))
        if more_structs <= used_structs and more_enums <= mod.used_enums:
            break
        used_structs |= more_structs
        mod.used_enums |= more_enums

    chunks: List[str] = [_HEADER]
    st = _emit_struct_types(structs, used_structs)
    if st:
        chunks.append(st)
    et = _emit_enum_types(variants, mod.used_enums)
    if et:
        chunks.append(et)
    ct = _emit_closure_types(mod)
    if ct:
        chunks.append(ct)
    henv_lines = []
    for site in sorted(live_scope_envs):
        fields = mod.scope_env_types[site]
        ftys = ", ".join(_llcell(k) for (_cn, k) in fields)
        desc = ", ".join(cn for (cn, _k) in fields)
        henv_lines.append(
            f"%henv.{_sanitize(site)} = type "
            f"{{ {ftys} }}" if ftys else
            f"%henv.{_sanitize(site)} = type {{}}")
        henv_lines[-1] += f"  ; handle-site env: {desc or '(none)'}"
    if henv_lines:
        chunks.append("\n".join(henv_lines))
    if mod.globals_used:
        glines = ["; module constants (declared by __module_init; the",
                  "; native entry wrapper calls @mx___module_init first --",
                  "; the interpreter's _ensure_globals)"]
        zero = {"i64": "0", "double": _fmt_f64(0.0), "ptr": "null"}
        for n in sorted(mod.globals_used):
            lty = _llscalar(mod.globals_used[n])
            glines.append(
                f"{_mx_global(n)} = internal global {lty} {zero[lty]}"
                f"  ; {n}: {mod.globals_used[n]}")
        chunks.append("\n".join(glines))
    chunks.extend(_emit_runtime(mod))
    for site in sorted(live_scope_shims):
        # A try site has no op-name / arity tables (nothing is dispatched).
        if site in mod.scope_tables:
            chunks.append(mod.scope_tables[site])
        chunks.append(mod.scope_thunks[site])
    # Dynamic-default thunks: kept when the default fn itself emitted AND
    # at least one performer that hands the thunk to mx_perform_or_default
    # survived the cascade (dep_names makes the default a dependency of
    # every such performer, so a demoted default takes them all with it).
    for dfn in sorted(mod.default_thunks):
        if dfn in emitted_chunks and any(
                u in emitted_chunks
                for u in mod.default_thunk_users.get(dfn, ())):
            chunks.append(mod.default_thunks[dfn])
    for info in infos:
        chunk = emitted_chunks.get(info.f.name)
        if chunk is not None:
            chunks.append(chunk)
            # Comprehension thunks of emitted owners (a demoted owner's
            # thunks are dropped with it — they reference its lambdas).
            chunks.extend(mod.comp_thunks.get(info.f.name, ()))
        else:
            chunks.append(_emit_placeholder(info, sigs[info.f.name]))
    return "\n\n".join(chunks)
