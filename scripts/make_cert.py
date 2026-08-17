"""Generate a self-signed certificate so the microphone works over the LAN.

Browsers expose ``navigator.mediaDevices`` only in a secure context: HTTPS, or
localhost. Demoing from a second machine over plain HTTP therefore gets a typed
question and nothing else. A self-signed certificate is enough to satisfy the
secure-context rule — the browser will warn once, and the warning has to be
accepted before the microphone becomes available.

Uses the ``cryptography`` package if it is installed, otherwise falls back to
the ``openssl`` binary. Writes ``.run/cert.pem`` and ``.run/key.pem``.

    python scripts/make_cert.py            # localhost + detected LAN address
    python scripts/make_cert.py 192.168.1.9

Then:

    uvicorn rag.server:app --host 0.0.0.0 --port 8443 \\
        --ssl-certfile .run/cert.pem --ssl-keyfile .run/key.pem
"""

from __future__ import annotations

import datetime
import ipaddress
import socket
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / ".run"
CERT = OUT / "cert.pem"
KEY = OUT / "key.pem"


def local_addresses() -> list[str]:
    addresses = {"127.0.0.1"}
    try:
        # Connecting a UDP socket does not send anything; it just makes the OS
        # pick the interface it would use, which is the LAN address we want.
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as probe:
            probe.connect(("8.8.8.8", 80))
            addresses.add(probe.getsockname()[0])
    except OSError:
        pass
    return sorted(addresses)


def with_cryptography(hosts: list[str]) -> bool:
    try:
        from cryptography import x509
        from cryptography.hazmat.primitives import hashes, serialization
        from cryptography.hazmat.primitives.asymmetric import rsa
        from cryptography.x509.oid import NameOID
    except ImportError:
        return False

    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "measure-rag-local")])
    alt = [x509.DNSName("localhost")]
    for host in hosts:
        try:
            alt.append(x509.IPAddress(ipaddress.ip_address(host)))
        except ValueError:
            alt.append(x509.DNSName(host))

    now = datetime.datetime.now(datetime.timezone.utc)
    cert = (
        x509.CertificateBuilder()
        .subject_name(name)
        .issuer_name(name)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - datetime.timedelta(minutes=5))
        .not_valid_after(now + datetime.timedelta(days=90))
        .add_extension(x509.SubjectAlternativeName(alt), critical=False)
        .sign(key, hashes.SHA256())
    )
    CERT.write_bytes(cert.public_bytes(serialization.Encoding.PEM))
    KEY.write_bytes(
        key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.TraditionalOpenSSL,
            encryption_algorithm=serialization.NoEncryption(),
        )
    )
    return True


def with_openssl(hosts: list[str]) -> bool:
    san = ",".join(
        f"IP:{h}" if h.replace(".", "").isdigit() else f"DNS:{h}" for h in hosts
    )
    command = [
        "openssl", "req", "-x509", "-newkey", "rsa:2048", "-nodes",
        "-keyout", str(KEY), "-out", str(CERT), "-days", "90",
        "-subj", "/CN=measure-rag-local",
        "-addext", f"subjectAltName=DNS:localhost,{san}",
    ]
    try:
        subprocess.run(command, check=True, capture_output=True)
        return True
    except (OSError, subprocess.CalledProcessError) as exc:
        detail = getattr(exc, "stderr", b"")
        print(f"openssl failed: {detail[:300].decode('utf-8', 'replace')}", file=sys.stderr)
        return False


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    hosts = sys.argv[1:] or local_addresses()
    print(f"certificate valid for: localhost, {', '.join(hosts)}")

    if not with_cryptography(hosts) and not with_openssl(hosts):
        print(
            "Could not generate a certificate. Install one of:\n"
            "  pip install cryptography\n"
            "  or make the openssl binary available on PATH",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"wrote {CERT}\nwrote {KEY}\n")
    print("Serve over HTTPS with:")
    print("  uvicorn rag.server:app --host 0.0.0.0 --port 8443 \\")
    print("      --ssl-certfile .run/cert.pem --ssl-keyfile .run/key.pem")
    print()
    print("The browser will warn that the certificate is not trusted. Accept it once;")
    print("the microphone stays blocked until you do.")


if __name__ == "__main__":
    main()
