import csv
import shutil
import subprocess
from pathlib import Path

root = Path(__file__).resolve().parents[2]
pcap = root / "data" / "base_paper" / "Packets" / "scan-portos-4-dec.pcap"
out = root / "data" / "base_paper" / "sample_features.csv"

tshark = shutil.which("tshark") or r"C:\Program Files\Wireshark\tshark.exe"
if not pcap.exists():
    raise SystemExit(f"Missing capture: {pcap}")
if not Path(tshark).exists():
    raise SystemExit(f"TShark not found: {tshark}")

fields = [
    "frame.number", "frame.time_epoch", "frame.len",
    "frame.protocols", "ip.src", "ip.dst", "ipv6.src", "ipv6.dst",
    "tcp.srcport", "tcp.dstport", "udp.srcport", "udp.dstport",
    "tcp.flags.syn", "tcp.flags.ack", "tcp.flags.reset",
    "udp.length",
]

cmd = [tshark, "-n", "-r", str(pcap), "-T", "fields",
       "-E", "header=y", "-E", "separator=,", "-E", "quote=d"]
for field in fields:
    cmd.extend(["-e", field])

print("Extracting:", pcap.name, f"({pcap.stat().st_size / 1024 / 1024:.2f} MB)")
result = subprocess.run(cmd, capture_output=True, text=True,
                        encoding="utf-8", errors="replace", timeout=180)
if result.returncode:
    raise SystemExit("TShark failed:\n" + result.stderr[:2000])

out.write_text(result.stdout, encoding="utf-8")
with out.open(encoding="utf-8-sig", newline="") as f:
    rows = list(csv.DictReader(f))

print("Packets extracted:", len(rows))
print("Feature columns:", fields)
print("Output:", out)
print("Label for this capture: scan")
