"""std.toml (Metaxu) against Python's tomllib, the oracle.

Every accepted document is rendered by both sides as compact JSON in
document order and compared; every rejected document must be rejected
by both (the messages are each side's own).  The writer is checked by
round trip: what `to_toml` writes, tomllib reads back to the same value,
and so does `parse`.  The native binary then has to print what the
interpreter printed.
"""
from __future__ import annotations

import json
import shutil
import tomllib

import pytest

from metaxu.compiler.tests.test_codegen_llvm import interp_run, llvm_from_source

needs_clang = pytest.mark.skipif(shutil.which("clang") is None, reason="clang is not installed")

MANIFEST = '''# a package
[package]
name = "app"
version = "0.1.0"
public = ["geom", "util",]

[dependencies]
geom = "^0.2"
util = { git = "https://example.com/util", rev = "v1.2.0" }
local = { path = "../local" }
pinned = { version = "=1.0.0", registry = "https://example.com/reg" }

[glade]
registry = "https://github.com/anoojpatel/glade-index"
'''

LOCK = '''version = 2

[[package]]
name = "geom"
source = "registry+https://github.com/anoojpatel/glade-index"
version = "0.2.3"
commit = "0123abcd"
hash = "sha256:deadbeef"

[[package]]
name = "util"
source = "git+https://example.com/util"
rev = "v1.2.0"
hash = "sha256:cafe"
'''

INDEX_ENTRY = '''name = "geom"
git = "https://example.com/geom"
description = "shapes, angles and a \\"quote\\""

[[versions]]
version = "0.1.0"
tag = "v0.1.0"

[versions.dependencies]
foo = "^1"

[[versions]]
version = "0.2.0"
tag = "v0.2.0"
commit = "abc"
dependencies = { foo = "^1", bar = ">=2, <3" }
'''

ACCEPTED = {
    "manifest": MANIFEST,
    "lock": LOCK,
    "index": INDEX_ENTRY,
    "dotted": 'a.b.c = 1\na.d = "s\\u00e9\\n\\t\\"q\\""\n[e.f]\ng = [1, [2, 3], []]\n',
    "numbers": "a = -12\nb = +5\nc = 1_000\nd = 0\n",
    "bools": "t = true\nf = false\n",
    "arrays": "a = [\n  1,\n  2,\n]\nb = []\nc = [\"x\", \"y\"]\n",
    "inline": "a = { b = 1, c = { d = \"e\" } }\nz = {}\n",
    "quoted_keys": '"a b" = 1\n[t."c.d"]\ne = 2\n',
    "comments": "a = 1 # c\n[t] # c\nb = 2 # c\n# end\n",
    "subheader_then_header": "[a.b]\nx = 1\n[a]\ny = 2\n",
    "header_then_subheader": "[a]\ny = 2\n[a.b]\nx = 1\n",
    "dotted_in_section": "[t]\na.b = 1\na.c = 2\n",
    "empty": "",
    "only_comment": "# hi\n",
    "crlf": "a = 1\r\n[t]\r\nb = 2\r\n",
    "escapes": 's = "tab\\t nl\\n bs\\\\ b\\b f\\f q\\" u\\u00e9"\n',
}

REJECTED = {
    "missing_value": "a = \n",
    "duplicate": "a = 1\na = 2\n",
    "float": "a = 1.5\n",
    "literal_string": "a = 'lit'\n",
    "table_twice": "[t]\nx = 1\n[t]\ny = 2\n",
    "dotted_then_header": "a.b = 1\n[a]\nc = 2\n",
    "aot_then_table": "[[p]]\nx = 1\n[p]\ny = 2\n",
    "table_then_aot": "[p]\ny = 2\n[[p]]\nx = 1\n",
    "junk": "x = 1 junk\n",
    "unterminated": 's = "open\n',
    "date": "d = 1979-05-27\n",
    "hex": "n = 0xff\n",
    "bad_escape": 's = "\\q"\n',
    "empty_header": "[]\nx = 1\n",
    "unclosed_array": "a = [1, 2\n",
    "unclosed_inline": "a = { b = 1\n",
    "inline_then_dotted": "a = { b = 1 }\na.c = 2\n",
    "json_slash_escape": 's = "a\\/b"\n',
    "trailing_comma_inline": "a = { b = 1, }\n",
    "leading_zero": "n = 007\n",
}


def _mx_literal(s: str) -> str:
    return '"' + (s.replace("\\", "\\\\").replace('"', '\\"')
                  .replace("\n", "\\n").replace("\r", "\\r").replace("\t", "\\t")) + '"'


PRELUDE = '''from std.toml import Toml, Entry, parse, to_toml;
from std.json import Json, member, to_json;

fn to_j(v: Toml) -> Json {
    match v {
        TStr(s) => JStr(s),
        TInt(n) => JInt(n),
        TBool(b) => JBool(b),
        TArr(items) => {
            let @mut out = Vec.new();
            let @mut i = 0;
            while i < len(items) { out.push(to_j(items[i])); i = i + 1 };
            JArr(out)
        },
        TTable(entries) => {
            let @mut out = Vec.new();
            let @mut i = 0;
            while i < len(entries) {
                let e = entries[i];
                out.push(member(e.key, to_j(e.value)));
                i = i + 1
            };
            JObj(out)
        }
    }
}

# JSON of the parse, "rejected" for an Err, and the JSON of the parse
# of what the writer wrote (the round trip).
fn show(text: string) -> () {
    match parse(text) {
        Ok(t) => {
            print(to_json(to_j(t)));
            match parse(to_toml(t)) {
                Ok(again) => print(to_json(to_j(again))),
                Err(e) => print("round trip failed: " + e)
            }
        },
        Err(e) => print("rejected")
    }
}
'''


def program() -> str:
    lines = [PRELUDE, "fn main() -> int {"]
    for doc in ACCEPTED.values():
        lines.append(f"    show({_mx_literal(doc)});")
    for doc in REJECTED.values():
        lines.append(f"    show({_mx_literal(doc)});")
    # the writer's text itself, for the tomllib round trip below
    lines.append(f"    match parse({_mx_literal(MANIFEST)}) {{ Ok(t) => print(to_toml(t)), Err(e) => print(e) }};")
    lines.append('    print("---");')
    lines.append(f"    match parse({_mx_literal(LOCK)}) {{ Ok(t) => print(to_toml(t)), Err(e) => print(e) }};")
    lines.append("    0")
    lines.append("}")
    return "\n".join(lines) + "\n"


def _canon(doc: str) -> str:
    return json.dumps(tomllib.loads(doc), separators=(",", ":"), ensure_ascii=False)


@pytest.fixture(scope="module")
def transcript() -> str:
    _result, out = interp_run(program())
    return out


def test_accepted_documents_match_tomllib(transcript):
    lines = transcript.split("\n")
    for i, (name, doc) in enumerate(ACCEPTED.items()):
        expected = _canon(doc)
        assert lines[2 * i] == expected, name
        assert lines[2 * i + 1] == expected, f"{name} (round trip)"


# Valid TOML outside the manifest subset: tomllib accepts these, std.toml
# declines them by name (floats, dates, hex integers, literal strings).
UNSUPPORTED = {"float", "literal_string", "date", "hex"}


def test_rejected_documents_are_rejected_by_both(transcript):
    lines = transcript.split("\n")
    base = 2 * len(ACCEPTED)
    for i, (name, doc) in enumerate(REJECTED.items()):
        if name in UNSUPPORTED:
            tomllib.loads(doc)  # valid TOML; the subset stops short of it
        else:
            with pytest.raises(tomllib.TOMLDecodeError):
                tomllib.loads(doc)
        assert lines[base + i] == "rejected", name


def test_written_text_reads_back_in_tomllib(transcript):
    tail = transcript.split("\n")[2 * len(ACCEPTED) + len(REJECTED):]
    text = "\n".join(tail)
    written_manifest, written_lock = text.split("\n---\n", 1)
    assert tomllib.loads(written_manifest) == tomllib.loads(MANIFEST)
    assert tomllib.loads(written_lock) == tomllib.loads(LOCK)
    # glade's own layout: inline tables for dependencies, one section per
    # locked package
    assert 'util = { git = "https://example.com/util", rev = "v1.2.0" }' in written_manifest
    assert written_lock.startswith("version = 2\n\n[[package]]\nname = \"geom\"\n")


def test_lookups():
    _res, out = interp_run('''
from std.toml import parse, get_table, get_str, get_int, get_arr, get_bool, keys;
fn main() -> int {
    match parse("[package]\\nname = \\"x\\"\\nn = 3\\nok = true\\npublic = [\\"a\\"]\\n") {
        Ok(t) => match get_table(t, "package") {
            Some(p) => {
                match get_str(p, "name") { Some(s) => print(s), None => print("?") };
                match get_int(p, "n") { Some(n) => print(n), None => print("?") };
                match get_bool(p, "ok") { Some(b) => print(b), None => print("?") };
                match get_arr(p, "public") { Some(a) => print(len(a)), None => print("?") };
                match get_str(p, "n") { Some(s) => print(s), None => print("not a string") };
                match get_str(p, "missing") { Some(s) => print(s), None => print("none") };
                print(keys(p).join(","))
            },
            None => print("no package")
        },
        Err(e) => print(e)
    };
    0
}
''')
    assert out == "x\n3\n1\n1\nnot a string\nnone\nname,n,ok,public\n"


def test_errors_name_the_line():
    _res, out = interp_run('''
from std.toml import parse;
fn main() -> int {
    match parse("a = 1\\n\\nb = 1.5\\n") { Ok(t) => print("ok"), Err(e) => print(e) };
    match parse("[t]\\nx = 1\\n[t]\\n") { Ok(t) => print("ok"), Err(e) => print(e) };
    0
}
''')
    assert out == "line 3: floats are not supported\nline 3: table 't' declared twice\n"


def test_native_main_and_every_reached_function_compile():
    # (Unreached exported helpers such as get_str keep bottom parameter
    # kinds and stay placeholders; nothing main reaches may be one.)
    import re
    ir = llvm_from_source(program())
    assert re.search(r"^define \S+ @mx_main\(", ir, re.M)
    for fname in ("std_toml_parse", "std_toml_to_toml", "std_toml_store",
                  "std_toml_read_value", "to_j", "show"):
        assert re.search(rf"^define \S+ @mx_{fname}(_k\d+)?\(", ir, re.M), fname


@needs_clang
def test_native_prints_what_the_interpreter_printed(tmp_path, transcript):
    from metaxu.compiler.llvm_run import compile_and_run
    exit_code, stdout = compile_and_run(llvm_from_source(program()), "main",
                                        workdir=str(tmp_path), timeout=300)
    assert exit_code == 0
    assert stdout == transcript
