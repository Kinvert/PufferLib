"""Delivered spatial stages with a recorded narrow desktop-process exception."""
import json
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "research"))
import spatial_autoencoder as spatial

ae = spatial.ae
ae.require(len(sys.argv) == 2 and sys.argv[1] in ("qualify", "run"), "Choose qualify or run")
MODE = sys.argv[1]
PACKET = ROOT / "build/spatial-ae-5090-623b7cb9-01"
LAUNCH = ROOT / ("build/spatial-desktop-623b7cb9-01-" + MODE)
DESKTOP_PID = 1934379


def identity(pid):
    try:
        proc = Path("/proc") / str(pid)
        return dict(executable=str((proc / "exe").resolve(strict=True)),
                    start_ticks=(proc / "stat").read_text().rsplit(")", 1)[1].split()[19])
    except (OSError, IndexError):
        return None


LAUNCH.mkdir(exist_ok=False)
shutil.copyfile(__file__, LAUNCH / "launcher.py")
desktop = identity(DESKTOP_PID)
ae.require(desktop is None or desktop["executable"] == str(Path("/home/keith/.local/bin/tern").resolve()),
           "Only the observed Tern desktop process is excepted")
record = dict(status="running", mode=MODE, packet=str(PACKET), exclusive_gpu=False,
    concurrency_basis="Executing user's run request with their preference to leave lightweight desktop GPU apps running; Tern exception disclosed before launch",
    allowed_pid=DESKTOP_PID, allowed_identity=desktop,
    launcher_sha256=ae.sha(LAUNCH / "launcher.py"),
    spatial_supervisor_sha256=ae.sha(ROOT / "research/spatial_autoencoder.py"),
    started_ns=time.monotonic_ns())
ae.save(LAUNCH / "execution.json", record)


class DesktopReservation(spatial.Reservation):
    def pids(self):
        actual = super().pids()
        allowed = [pid for pid in actual if pid == DESKTOP_PID and desktop is not None
                   and identity(pid) == desktop]
        with (LAUNCH / "gpu-process-samples.jsonl").open("a") as stream:
            stream.write(json.dumps(dict(monotonic_ns=time.monotonic_ns(), actual_pids=actual,
                                         allowed_desktop_pids=allowed)) + "\n")
        return [pid for pid in actual if pid not in allowed]

    def __enter__(self):
        result = super().__enter__()
        marker = self.out / f"stages/{self.stage}.started.json"
        value = ae.read(marker)
        value.update(exclusive_gpu=False, concurrency_context=str(LAUNCH / "execution.json"),
                     launch_wrapper_sha256=record["launcher_sha256"])
        ae.save(marker, value)
        return result


spatial.Reservation = DesktopReservation
try:
    result = spatial.qualify(PACKET, "5090") if MODE == "qualify" else spatial.run(PACKET, "5090")
    record.update(status=result["status"], rows=len(result.get("rows", [])),
                  cases=len(result.get("cases", [])))
except BaseException as error:
    record.update(status="failed", error=f"{type(error).__name__}: {error}")
    raise
finally:
    record["ended_ns"] = time.monotonic_ns()
    record["overall_seconds"] = (record["ended_ns"] - record["started_ns"]) / 1e9
    ae.save(LAUNCH / "execution.json", record)
    print(json.dumps(record, indent=2))
