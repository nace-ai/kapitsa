"""Pinned HTTP-range row filling and auditable coverage for the sparse Sushi table."""
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import shutil
import sqlite3
import struct
import time
from download import ROOT, MODEL, REV, ranged, record


class TableRows:
    def __init__(self):
        self.path = MODEL / "ngram_table.bin"
        with self.path.open("rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            self.header = json.loads(f.read(n))
        self.parts = []
        for name, width in [("weight", 4), ("scales", 2), ("biases", 2)]:
            spec = self.header[name]
            self.parts.append((8 + n + spec["data_offsets"][0], spec["shape"][1] * width))
        self.count = self.header["weight"]["shape"][0]
        self.db = sqlite3.connect(ROOT / "real-rows.sqlite3")
        self.db.execute("CREATE TABLE IF NOT EXISTS rows (id INTEGER PRIMARY KEY, sha256 TEXT NOT NULL)")
        self.db.commit()

    def digest(self, fd, row):
        h = hashlib.sha256()
        for base, width in self.parts:
            value = os.pread(fd, width, base + row * width)
            assert len(value) == width
            h.update(value)
        return h.hexdigest()

    def ensure(self, rows, label, stream=True):
        rows = sorted(set(int(r) for r in rows))
        assert rows and rows[0] >= 0 and rows[-1] < self.count
        known = {r for (r,) in self.db.execute("SELECT id FROM rows")}
        want = [r for r in rows if r not in known]
        fd = os.open(self.path, os.O_RDWR)
        try:
            jobs = []
            for base, width in self.parts:
                groups = {}
                # Large prefill sets use bounded sequential ranges; incremental decode uses individual rows.
                per = max(1, (16 << 20) // width) if stream and len(want) > 1000 else 1
                for row in want:
                    groups.setdefault(row // per, []).append(row)
                for block, selected in groups.items():
                    a = base + block * per * width
                    b = base + min(self.count, (block + 1) * per) * width
                    jobs.append((a, b, base, width, selected))
            def work(job):
                a, b, base, width, selected = job
                for attempt in range(8):
                    try:
                        data = ranged("ngram_table.bin", a, b)
                        break
                    except Exception:
                        if attempt == 7:
                            raise
                        time.sleep(min(30, 2 ** attempt))
                if shutil.disk_usage(ROOT).free < (6 << 30):
                    raise RuntimeError("table fill below 6 GiB free-space guard")
                for row in selected:
                    offset = base + row * width
                    value = data[offset - a:offset - a + width]
                    assert len(value) == width
                    assert os.pwrite(fd, value, offset) == width
                    assert os.pread(fd, width, offset) == value
                return b - a
            fetched = 0
            with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
                for i, size in enumerate(pool.map(work, jobs)):
                    fetched += size
                    if (i + 1) % 32 == 0:
                        record({"stage": "row-fill", "label": label, "ranges_done": i + 1,
                                "ranges_total": len(jobs), "downloaded_bytes": fetched})
            os.fsync(fd)
            self.db.executemany("INSERT INTO rows VALUES (?,?)", [(r, self.digest(fd, r)) for r in want])
            self.db.commit()
        finally:
            os.close(fd)
        result = self.check(rows)
        result.update(label=label, new_rows=len(want), downloaded_bytes=fetched,
                      table_logical_bytes=self.path.stat().st_size,
                      table_allocated_bytes=self.path.stat().st_blocks * 512,
                      full_table_downloaded=False, revision=REV, response_ranges_verified=True)
        (ROOT / "evidence" / (label + "-row-coverage.json")).write_text(json.dumps(result, indent=2) + "\n")
        record(result)
        return result

    def check(self, rows):
        rows = sorted(set(int(r) for r in rows))
        hashes = dict(self.db.execute("SELECT id,sha256 FROM rows"))
        missing = [r for r in rows if r not in hashes]
        if missing:
            raise RuntimeError(f"{len(missing)} accessed rows have not been fetched, first={missing[:10]}")
        fd = os.open(self.path, os.O_RDONLY)
        total = hashlib.sha256()
        try:
            for r in rows:
                digest = self.digest(fd, r)
                assert digest == hashes[r], r
                total.update(struct.pack("<Q", r) + bytes.fromhex(digest))
        finally:
            os.close(fd)
        return {"required_rows": len(rows), "missing_rows": 0, "all_row_bytes_rechecked": True,
                "ordered_row_digest": total.hexdigest()}
