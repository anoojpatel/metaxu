"""std.fs, std.process, std.env and std.io on both engines (docs/io_runtime.md).

One program exercises every operation inside a scratch directory: files
and directories in and out, the sorted listing and walk, the catchable
error messages, the process runner with both streams and a failing exec,
the environment and the program arguments.  The interpreter runs it with
the scratch directory as its cwd; the native binary runs the same
program there and must print the same lines.  A second program serves
the filesystem and a fake `git` from handlers, which is what the effect
shape is for.
"""
from __future__ import annotations

import os
import shutil
import subprocess

import pytest

from metaxu.compiler.hir import HIRBuilder
from metaxu.compiler.lower_hir_to_mir import lower_hir_to_mir
from metaxu.compiler.mir_interp import UNIT, MirInterpreter, mx_display
from metaxu.compiler.pipeline import build_context_from_source
from metaxu.compiler.tests.test_codegen_llvm import interp_run, llvm_from_source, mir_from_source

needs_clang = pytest.mark.skipif(shutil.which("clang") is None, reason="clang is not installed")

IO_PROGRAM = '''
from std.fs import read_text, write_text, write_bytes, read_bytes, try_read_text, exists, is_dir, is_file, list_dir, mkdir_all, remove_all, rename, walk;
from std.env import args, lookup, get_or, has, cwd, home;
from std.process import run, run_in, output_of;
from std.io import eprintln;

fn main() -> int {
    mkdir_all("sandbox/a/b");
    mkdir_all("sandbox/a/b");
    write_text("sandbox/a/b/one.txt", "first\\n");
    write_text("sandbox/a/two.txt", "second");
    write_text("sandbox/top.txt", "hé");
    let @mut raw = Vec.new();
    raw.push(0);
    raw.push(255);
    raw.push(10);
    write_bytes("sandbox/raw.bin", raw);
    let back = read_bytes("sandbox/raw.bin");
    print(back[0].to_string() + "," + back[1].to_string() + "," + back[2].to_string());
    print(read_text("sandbox/a/two.txt"));
    print(len(read_text("sandbox/top.txt").to_bytes()));
    print(exists("sandbox/a").to_string() + " " + is_dir("sandbox/a").to_string() + " " + is_file("sandbox/a").to_string() + " " + is_file("sandbox/top.txt").to_string());
    print(list_dir("sandbox").join(","));
    print(walk("sandbox").join(","));
    rename("sandbox/top.txt", "sandbox/moved.txt");
    print(exists("sandbox/top.txt").to_string() + " " + exists("sandbox/moved.txt").to_string());
    match try_read_text("sandbox/nope.txt") { Ok(t) => print(t), Err(e) => print(e) };
    print(try { let x = read_text("sandbox/a"); "read a dir" } catch e { e });
    print(try { let x = list_dir("sandbox/nope"); "listed" } catch e { e });
    print(try { rename("sandbox/nope", "sandbox/other"); "renamed" } catch e { e });
    let @mut bad = Vec.new();
    bad.push(300);
    print(try { write_bytes("sandbox/bad.bin", bad); "wrote" } catch e { e });
    remove_all("sandbox/a");
    print(list_dir("sandbox").join(","));
    print(try { remove_all("sandbox/gone"); "removed" } catch e { e });
    remove_all("sandbox");
    print(exists("sandbox"));
    print(args().join("|"));
    print(len(args()));
    match lookup("MX_TEST_VAR") { Some(v) => print("var=" + v), None => print("unset") };
    match lookup("MX_TEST_EMPTY") { Some(v) => print("empty=[" + v + "]"), None => print("unset") };
    print(get_or("MX_TEST_MISSING_XYZ", "dflt"));
    print(has("MX_TEST_VAR").to_string() + " " + has("MX_TEST_MISSING_XYZ").to_string());
    print(len(cwd()) > 0);
    print(len(home()) > 0);
    let @mut argv = Vec.new();
    argv.push("sh");
    argv.push("-c");
    argv.push("echo out; echo err 1>&2; exit 3");
    let o = run(argv);
    print(o.status.to_string() + "|" + o.out + "|" + o.err);
    let @mut pwdv = Vec.new();
    pwdv.push("pwd");
    match output_of(pwdv, "/") { Ok(t) => print(t), Err(e) => print(e) };
    let @mut failing = Vec.new();
    failing.push("sh");
    failing.push("-c");
    failing.push("exit 2");
    match output_of(failing, "") { Ok(t) => print(t), Err(e) => print(e) };
    let @mut missing = Vec.new();
    missing.push("definitely-not-a-program-xyz");
    print(try { let r = run(missing); "ran" } catch e { e });
    let @mut badcwd = Vec.new();
    badcwd.push("true");
    print(try { let r = run_in(badcwd, "/nonexistent-dir-xyz"); "ran" } catch e { e });
    let empty = Vec.new();
    print(try { let r = run(empty); "ran" } catch e { e });
    eprintln("to stderr");
    0
}
'''

EXPECTED = [
    "0,255,10",
    "second",
    "3",
    "1 1 0 1",
    "a,raw.bin,top.txt",
    "a/b/one.txt,a/two.txt,raw.bin,top.txt",
    "0 1",
    "read: sandbox/nope.txt: No such file or directory",
    "read: sandbox/a: Is a directory",
    "list_dir: sandbox/nope: No such file or directory",
    "rename: sandbox/nope: No such file or directory",
    "write: element 0 is not a byte (0..255): 300",
    "moved.txt,raw.bin",
    "remove_all: sandbox/gone: No such file or directory",
    "0",
    "alpha|beta",
    "2",
    "var=hello",
    "empty=[]",
    "dflt",
    "1 0",
    "1",
    "1",
    "3|out\n|err\n",
    "/\n",
    "sh exited with status 2",
    "run: definitely-not-a-program-xyz: No such file or directory",
    "run: /nonexistent-dir-xyz: No such file or directory",
    "run: empty argv",
]

ENV = {"MX_TEST_VAR": "hello", "MX_TEST_EMPTY": ""}
ARGS = ["alpha", "beta"]


def _run_interp(tmp_path, monkeypatch) -> str:
    """The program on the interpreter, cwd and env as the native run sees them."""
    monkeypatch.chdir(tmp_path)
    for k, v in ENV.items():
        monkeypatch.setenv(k, v)
    monkeypatch.delenv("MX_TEST_MISSING_XYZ", raising=False)
    interp = MirInterpreter()
    interp.load(mir_from_source(IO_PROGRAM))
    interp.program_args = list(ARGS)
    out: list[str] = []
    interp.register_builtin("print", lambda *a: (out.append(" ".join(str(mx_display(x)) for x in a)), UNIT)[1])
    result = interp.call("main", [])
    assert result == 0
    return "".join(line + "\n" for line in out)


def test_interpreter_io(tmp_path, monkeypatch):
    out = _run_interp(tmp_path, monkeypatch)
    assert out == "".join(line + "\n" for line in EXPECTED)
    assert not (tmp_path / "sandbox").exists()


def test_native_is_placeholder_free():
    assert "placeholder -- unsupported" not in llvm_from_source(IO_PROGRAM)


@needs_clang
def test_native_io_matches_the_interpreter(tmp_path, monkeypatch):
    from metaxu.compiler.llvm_run import compile_and_run
    (tmp_path / "interp").mkdir()
    expected = _run_interp(tmp_path / "interp", monkeypatch)
    monkeypatch.chdir(tmp_path)
    native_dir = tmp_path / "native"
    native_dir.mkdir()
    (tmp_path / "build").mkdir()
    exit_code, stdout = compile_and_run(
        llvm_from_source(IO_PROGRAM), "main", workdir=str(tmp_path / "build"),
        run_cwd=str(native_dir), run_env=dict(ENV), run_args=tuple(ARGS), timeout=300)
    assert exit_code == 0
    assert stdout == expected
    assert not (native_dir / "sandbox").exists()


@needs_clang
def test_native_stderr_goes_to_stderr(tmp_path):
    from metaxu.compiler.llvm_run import compile_to_binary
    src = '''
from std.io import eprint, eprintln;
fn main() -> int {
    print("out");
    eprint("err ");
    eprintln("line");
    0
}
'''
    binary = compile_to_binary(llvm_from_source(src), "main",
                               out_path=str(tmp_path / "prog.bin"))
    proc = subprocess.run([binary], capture_output=True, text=True)
    assert proc.returncode == 0
    assert proc.stdout == "out\n"
    assert proc.stderr == "err line\n"


def test_interpreter_stderr_goes_to_stderr(capsys):
    _res, out = interp_run('''
from std.io import eprint, eprintln;
fn main() -> int {
    print("out");
    eprint("err ");
    eprintln("line");
    0
}
''')
    assert out == "out\n"
    assert capsys.readouterr().err == "err line\n"


VIRTUAL_PROGRAM = '''
from std.fs import Fs, read_text, write_text, exists;
from std.process import Process, output_of;

fn index_of(names: Vec, name: string) -> int {
    let @mut i = 0;
    let @mut found = 0 - 1;
    while i < len(names) && found < 0 {
        if names[i] == name { found = i } else { () };
        i = i + 1
    }
    found
}

fn main() -> int {
    let @mut names = Vec.new();
    let @mut bodies = Vec.new();
    handle Fs with {
        write_bytes(path, bytes) -> {
            let k = index_of(names, path);
            if k < 0 { names.push(path); bodies.push(bytes) } else { bodies[k] = bytes };
            resume(())
        },
        read_bytes(path) -> {
            let k = index_of(names, path);
            # (a failure raised here would propagate from the handle
            # site, not into the performer's try: docs/io_runtime.md)
            if k < 0 { raise("virtual read: " + path + " was never written") } else { resume(bodies[k]) }
        },
        exists(path) -> resume(index_of(names, path) >= 0)
    } in {
        write_text("mx.toml", "[package]\\nname = \\"app\\"\\n");
        print(read_text("mx.toml"));
        print(exists("mx.toml").to_string() + " " + exists("mx.lock").to_string());
        if exists("mx.lock") { print(read_text("mx.lock")) } else { print("no lock yet") };
        write_text("mx.toml", "changed");
        print(read_text("mx.toml"))
    };
    let @mut git = Vec.new();
    git.push("git");
    git.push("rev-parse");
    git.push("HEAD");
    handle Process with {
        run(argv, cwd) -> resume(7),
        status(h) -> resume(0),
        stdout(h) -> resume("0123abcd\\n"),
        stderr(h) -> resume("")
    } in {
        match output_of(git, "") { Ok(t) => print("fake git said " + t.trim()), Err(e) => print(e) }
    };
    0
}
'''

VIRTUAL_EXPECTED = [
    "[package]",
    'name = "app"',
    "",
    "1 0",
    "no lock yet",
    "changed",
    "fake git said 0123abcd",
]


def test_handlers_virtualize_the_filesystem_and_processes(tmp_path, monkeypatch):
    # Nothing touches the real filesystem or starts a process: the handlers
    # in scope answer every operation.
    monkeypatch.chdir(tmp_path)
    _res, out = interp_run(VIRTUAL_PROGRAM)
    assert out == "".join(line + "\n" for line in VIRTUAL_EXPECTED)
    assert list(tmp_path.iterdir()) == []


@needs_clang
def test_native_handlers_virtualize_too(tmp_path):
    from metaxu.compiler.llvm_run import compile_and_run
    ir = llvm_from_source(VIRTUAL_PROGRAM)
    assert "placeholder -- unsupported" not in ir
    (tmp_path / "build").mkdir()
    exit_code, stdout = compile_and_run(ir, "main", workdir=str(tmp_path / "build"),
                                        run_cwd=str(tmp_path))
    assert exit_code == 0
    assert stdout == "".join(line + "\n" for line in VIRTUAL_EXPECTED)
    assert sorted(p.name for p in tmp_path.iterdir()) == ["build"]


def test_type_errors_are_loud():
    for expr, msg in [
        ("perform Fs.read_bytes(3)", "EFFECT_FS_READ expects a string path, got 'Int'"),
        ('perform Fs.write_bytes("x", 3)', "EFFECT_FS_WRITE expects a Vec of bytes, got 'Int'"),
        ('perform Process.status(99)', "status: no such process handle 99"),
    ]:
        _res, out = interp_run(f'''
from std.fs import Fs;
from std.process import Process;
fn main() -> int {{
    print(try {{ let r = {expr}; "no error" }} catch e {{ e }});
    0
}}
''')
        assert out.strip() == msg, expr


def test_metaxuc_run_passes_arguments(tmp_path):
    from metaxu.compiler.cli import main as cli_main
    prog = tmp_path / "args.mx"
    prog.write_text('from std.env import args;\nfn main() -> int { print(args().join(" ")); len(args()) }\n')
    proc = subprocess.run(
        ["uv", "run", "metaxuc", "run", str(prog), "--", "one", "two", "--three"],
        capture_output=True, text=True, cwd=os.getcwd())
    assert proc.returncode == 3, proc.stderr
    assert proc.stdout == "one two --three\n"


def test_a_failure_in_a_handler_arm_propagates_from_the_handle_site():
    # docs/try_catch.md: the arm runs in the handler's frame, so a raise
    # there is not inside a try that sits in the handled body; a try
    # around the whole handle catches it.
    _res, out = interp_run('''
from std.fs import Fs, read_text, try_read_text;
fn main() -> int {
    let inner = try {
        handle Fs with {
            read_bytes(path) -> raise("virtual read: " + path)
        } in {
            match try_read_text("x") { Ok(t) => "inner ok", Err(e) => "inner caught " + e }
        }
    } catch e { "outer caught " + e };
    print(inner);
    0
}
''')
    assert out == "outer caught virtual read: x\n"
