import pathlib, sys
source = pathlib.Path(sys.argv[1]).resolve()
build = pathlib.Path(sys.argv[2]).resolve()
s = source.read_text()
s = s[:s.index("# ---------------------------------------------------------------- 4.")]
s = s.replace('return m.group(1) if m else ""', 'if not m: print("VERSION_DIAGNOSTIC", repr({"path":path,"exit":r.returncode,"stdout":r.stdout,"stderr":r.stderr}), flush=True)\n    return m.group(1) if m else ""')
s += '\nprint("DIAGNOSTIC_FAILURES", repr(FAILS), flush=True)\nprint("FIXTURE_DIRECTORY", TMP, flush=True)\nraise SystemExit(bool(FAILS))\n'
sys.argv = [str(source),str(build)]
exec(compile(s, str(source) + "[diagnostic sections1-3]", "exec"), {"__file__":str(source),"__name__":"__main__"})
