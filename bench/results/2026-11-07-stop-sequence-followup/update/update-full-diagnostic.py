import pathlib
import sys

source = pathlib.Path(sys.argv[1]).resolve()
build = pathlib.Path(sys.argv[2]).resolve()
s = source.read_text()
needle = 'return m.group(1) if m else ""'
assert s.count(needle) == 1
s = s.replace(needle, 'if not m: print("VERSION_DIAGNOSTIC", repr({"path":path,"exit":r.returncode,"stdout":r.stdout,"stderr":r.stderr}), flush=True)\n    return m.group(1) if m else ""')
sys.argv = [str(source), str(build)]
exec(compile(s, str(source) + '[full diagnostic]', 'exec'), {'__file__':str(source), '__name__':'__main__'})
