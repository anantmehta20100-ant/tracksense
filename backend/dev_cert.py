"""Self-signed certificate for LOCAL HTTPS (development only).

Phone browsers only expose the camera API (getUserMedia) on a *secure context*:
HTTPS, or localhost. To drive the PC's YOLO pipeline from the phone's camera over
the LAN we therefore need HTTPS. This makes a self-signed cert (valid for
localhost, the loopback, and the current LAN IP) and caches it under .certs/ so
the phone only has to accept the browser's "not secure" warning once.

This is NOT for production -- it is a throwaway, untrusted certificate whose only
purpose is to unlock the camera API on your own network. The private key lives in
.certs/ and is gitignored; it must never be committed.
"""

from __future__ import annotations

import datetime
import ipaddress
from pathlib import Path

CERT_DIR = Path(__file__).resolve().parents[1] / ".certs"
CERT_FILE = CERT_DIR / "tracksense-dev.crt"
KEY_FILE = CERT_DIR / "tracksense-dev.key"


def ensure_cert(extra_hosts=None):
    """Return (cert_path, key_path) as strings, generating a self-signed cert on
    first use. `extra_hosts` may include IP strings (e.g. the current LAN IP) to
    add as Subject Alternative Names."""
    if CERT_FILE.exists() and KEY_FILE.exists():
        return str(CERT_FILE), str(KEY_FILE)

    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import rsa
    from cryptography.x509.oid import NameOID

    CERT_DIR.mkdir(parents=True, exist_ok=True)
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)

    san = [x509.DNSName("localhost"), x509.IPAddress(ipaddress.ip_address("127.0.0.1"))]
    for host in (extra_hosts or []):
        if not host:
            continue
        try:
            san.append(x509.IPAddress(ipaddress.ip_address(host)))
        except ValueError:
            san.append(x509.DNSName(str(host)))

    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "TrackSense Dev")])
    # Normal app runtime (not a workflow script), so datetime is fine here.
    now = datetime.datetime.utcnow()
    cert = (
        x509.CertificateBuilder()
        .subject_name(name)
        .issuer_name(name)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - datetime.timedelta(days=1))
        .not_valid_after(now + datetime.timedelta(days=825))
        .add_extension(x509.SubjectAlternativeName(san), critical=False)
        .add_extension(x509.BasicConstraints(ca=True, path_length=None), critical=True)
        .sign(key, hashes.SHA256())
    )

    KEY_FILE.write_bytes(key.private_bytes(
        serialization.Encoding.PEM,
        serialization.PrivateFormat.TraditionalOpenSSL,
        serialization.NoEncryption(),
    ))
    CERT_FILE.write_bytes(cert.public_bytes(serialization.Encoding.PEM))
    return str(CERT_FILE), str(KEY_FILE)
