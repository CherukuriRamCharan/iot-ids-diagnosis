import csv
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PCAP_DIR = ROOT / "data" / "base_paper" / "Packets"
OUT = ROOT / "data" / "base_paper" / "packet_features.csv"

TSHARK = shutil.which("tshark")
if not TSHARK:
    for candidate in (
        Path(r"C:\Program Files\Wireshark\tshark.exe"),
        Path(r"C:\Program Files (x86)\Wireshark\tshark.exe"),
    ):
        if candidate.exists():
            TSHARK = str(candidate)
            break

if not TSHARK:
    raise SystemExit("ERROR: TShark not found. Check your Wireshark installation.")

FIELDS = [
    ("frame.time_epoch", "timestamp"),
    ("frame.len", "packet_length"),
    ("frame.protocols", "protocols"),
    ("ip.src", "src_ip"),
    ("ip.dst", "dst_ip"),
    ("ipv6.src", "src_ip"),
    ("ipv6.dst", "dst_ip"),
    ("tcp.srcport", "src_port"),
    ("tcp.dstport", "dst_port"),
    ("udp.srcport", "src_port"),
    ("udp.dstport", "dst_port"),
    ("tcp.flags.syn", "tcp_syn"),
    ("tcp.flags.ack", "tcp_ack"),
    ("tcp.flags.reset", "tcp_reset"),
    ("tcp.window_size_value", "tcp_window"),
    ("udp.length", "udp_length"),
    ("dns.qry.name", "dns_query"),
    ("mqtt.msgtype", "mqtt_type"),
]

# One tshark field per output column; duplicate aliases are merged below.
args = [
    TSHARK, "-n", "-r", "", "-T", "fields",
    "-E", "separator=\t", "-E", "quote=n", "-E", "occurrence=f",
]
for field, _ in FIELDS:
    args.extend(["-e", field])

files = sorted(PCAP_DIR.glob("*.pcap")) if PCAP_DIR.exists() else []
if not files:
    raise SystemExit(f"ERROR: No PCAP files found in {PCAP_DIR}")

columns = [
    "capture_file", "attack_label", "timestamp", "packet_length",
    "protocols", "src_ip", "dst_ip", "src_port", "dst_port",
    "tcp_syn", "tcp_ack", "tcp_reset", "tcp_window",
    "udp_length", "dns_query", "mqtt_type",
]
rows = []
errors = []

for pcap in files:
    label = "benign" if pcap.name.lower().startswith("benign") else pcap.name.lower().split("-")[0]
    cmd = args.copy()
    cmd[cmd.index("-r") + 1] = str(pcap)
    result = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if result.returncode:
        errors.append((pcap.name, result.stderr.strip()[:250]))
        continue

    for line in result.stdout.splitlines():
        values = line.split("\t")
        values += [""] * (len(FIELDS) - len(values))
        record = dict(zip([name for _, name in FIELDS], values))
        rows.append({
            "capture_file": pcap.name,
            "attack_label": label,
            **{c: record.get(c, "") for c in columns if c not in ("capture_file", "attack_label")},
        })

with OUT.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=columns)
    writer.writeheader()
    writer.writerows(rows)

print("PCAP files found:", len(files))
print("PCAP files successfully processed:", len(files) - len(errors))
print("Packets extracted:", len(rows))
print("Feature CSV:", OUT)
print("Labels:", sorted({r["attack_label"] for r in rows}))
if errors:
    print("Errors:")
    for name, error in errors:
        print(f" - {name}: {error}")
