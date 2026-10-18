import csv
import shutil
import subprocess
from pathlib import Path

root = Path(__file__).resolve().parents[2]
folder = root / "data" / "base_paper" / "Packets"
output = root / "data" / "base_paper" / "all_packet_features.csv"

tshark = shutil.which("tshark") or r"C:\Program Files\Wireshark\tshark.exe"
if not Path(tshark).exists():
    raise SystemExit("TShark not found.")

pcaps = sorted(folder.glob("*.pcap"))
if not pcaps:
    raise SystemExit(f"No PCAP files found: {folder}")

fields = [
    "frame.time_epoch", "frame.len", "frame.protocols",
    "ip.src", "ip.dst", "tcp.srcport", "tcp.dstport",
    "udp.srcport", "udp.dstport", "tcp.flags.syn",
    "tcp.flags.ack", "tcp.flags.reset", "udp.length",
]

columns = ["capture_file", "label"] + [
    "timestamp", "packet_length", "protocols", "src_ip", "dst_ip",
    "tcp_src_port", "tcp_dst_port", "udp_src_port", "udp_dst_port",
    "tcp_syn", "tcp_ack", "tcp_reset", "udp_length",
]

total = 0
failed = []

with output.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=columns)
    writer.writeheader()

    for i, pcap in enumerate(pcaps, 1):
        name = pcap.stem.lower()
        label = "benign" if name.startswith("benign") else name.split("-")[0]

        cmd = [
            tshark, "-n", "-r", str(pcap), "-T", "fields",
            "-E", "separator=,", "-E", "quote=d", "-E", "occurrence=f",
        ]
        for field in fields:
            cmd.extend(["-e", field])

        print(f"[{i}/{len(pcaps)}] {pcap.name}", flush=True)
        try:
            proc = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                text=True, encoding="utf-8", errors="replace"
            )
            # Stream output so large captures do not fill memory.
            import io
            reader = csv.reader(io.StringIO(proc.stdout.read()))
            count = 0
            for values in reader:
                values += [""] * (len(fields) - len(values))
                writer.writerow({
                    "capture_file": pcap.name,
                    "label": label,
                    **dict(zip(columns[2:], values)),
                })
                count += 1
                total += 1

            err = proc.stderr.read()
            code = proc.wait()
            if code:
                failed.append((pcap.name, err[:200]))
            print(f"    packets: {count}", flush=True)
        except Exception as exc:
            failed.append((pcap.name, str(exc)))

print(f"\nCaptures: {len(pcaps)}")
print(f"Packets extracted: {total}")
print(f"CSV: {output}")
print(f"Failed captures: {len(failed)}")
for name, reason in failed:
    print("FAILED:", name, reason)
