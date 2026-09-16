"""Inventory local NRRD archives without exporting patient metadata.

Source IDs identify archive members, NOT patients. Reads originals only.
Only inline, raw, scalar 3D NRRDs can be staged by this pilot helper.
"""
import argparse
import hashlib
import json
import math
import re
import zipfile
from datetime import datetime, timezone
from pathlib import Path

VERSION = "0.1.0"
FIELDS = ("type", "dimension", "space", "sizes", "space directions", "kinds",
          "endian", "encoding", "space origin", "space units")
BYTES = {"short": 2, "int16": 2, "int16_t": 2, "unsigned short": 2,
         "ushort": 2, "int": 4, "int32": 4, "int32_t": 4, "float": 4,
         "double": 8, "uchar": 1, "unsigned char": 1}


def read_header(stream):
    first = stream.readline(65537)
    if not first.startswith(b"NRRD000"):
        raise ValueError("Not an NRRD file")
    size, fields = len(first), {}
    while size < 65536:
        line = stream.readline(65537)
        size += len(line)
        if not line:
            raise ValueError("Missing NRRD header terminator")
        if not line.strip():
            return fields
        if line.startswith(b"#") or b":=" in line:
            continue
        key, sep, value = line.decode("ascii", "strict").partition(":")
        if sep:
            fields[key.strip().lower()] = value.strip()
    raise ValueError("NRRD header too large")


def geometry(fields):
    sizes = [int(x) for x in fields["sizes"].split()]
    vectors = [[float(x) for x in v.split(",")]
               for v in re.findall(r"\(([^()]*)\)", fields["space directions"])]
    if fields.get("dimension") != "3" or len(sizes) != 3 or min(sizes) <= 0:
        raise ValueError("Expected positive scalar 3D sizes")
    if len(vectors) != 3 or any(len(v) != 3 for v in vectors):
        raise ValueError("Expected three spatial direction vectors")
    spacing = [math.sqrt(sum(x*x for x in v)) for v in vectors]
    if any(not math.isfinite(s) or s <= 0 for s in spacing):
        raise ValueError("Invalid spacing")
    orthogonal = all(abs(sum(vectors[i][k]*vectors[j][k] for k in range(3)) /
                         (spacing[i]*spacing[j])) < 1e-4
                     for i in range(3) for j in range(i))
    if min(sizes) == 1:
        category = "single_slice_not_eligible"
    elif not orthogonal:
        category = "geometry_requires_review"
    elif max(spacing) < 1.0 - 1e-6:
        category = "submillimeter_candidate"
    elif max(spacing) <= 1.0 + 1e-6:
        category = "one_mm_challenge"
    else:
        category = "thick_sampling_not_primary"
    return dict(sizes_ijk=sizes, spacing_mm=spacing,
                orthogonal_axes=orthogonal, sampling_category=category)


def validate_raw(fields):
    geometry(fields)
    if fields.get("encoding") != "raw" or fields.get("type") not in BYTES:
        raise ValueError("Staging requires supported raw scalar data")
    if any(k in fields for k in ("data file", "datafile", "byte skip", "line skip")):
        raise ValueError("External data and skip fields are not supported")
    if fields.get("space") not in ("left-posterior-superior", "right-anterior-superior"):
        raise ValueError("Expected declared LPS or RAS coordinates")
    origin = [float(x) for x in fields.get("space origin", "").strip("()").split(",")]
    if len(origin) != 3 or not all(math.isfinite(x) for x in origin):
        raise ValueError("Missing or invalid spatial origin")
    if BYTES[fields["type"]] > 1 and fields.get("endian") not in ("little", "big"):
        raise ValueError("Missing byte order")
    return math.prod(int(x) for x in fields["sizes"].split()) * BYTES[fields["type"]]


def hash_payload(stream):
    digest, size = hashlib.sha256(), 0
    while True:
        chunk = stream.read(4 * 1024 * 1024)
        if not chunk:
            return digest.hexdigest(), size
        digest.update(chunk)
        size += len(chunk)


def inventory(archives, verify=False):
    report = dict(version=VERSION, generated_utc=datetime.now(timezone.utc).isoformat(),
                  patients_confirmed=0, archives=[], volumes=[],
                  caveat="Sampling and payload checks do not establish HU, laterality, contrast, or patient identity.")
    for ai, archive in enumerate(archives, 1):
        a = dict(archive_slot=ai, archive_basename=Path(archive).name)
        report["archives"].append(a)
        try:
            with zipfile.ZipFile(archive) as z:
                a["status"] = "central_directory_readable"
                for ei, entry in enumerate(z.infolist()):
                    if entry.is_dir() or not entry.filename.lower().endswith(".nrrd"):
                        continue
                    row = dict(source_id=f"SRC_A{ai:02}_E{ei:02}", archive_slot=ai,
                               entry_index=ei, payload_status="header_only")
                    report["volumes"].append(row)
                    try:
                        with z.open(entry) as stream:
                            fields = read_header(stream)
                            row.update(geometry(fields))
                            row["technical_header"] = {k: fields[k] for k in FIELDS if k in fields}
                            if verify and row["sampling_category"] in (
                                    "submillimeter_candidate", "one_mm_challenge"):
                                expected = validate_raw(fields)
                                digest, count = hash_payload(stream)
                                row.update(payload_sha256=digest, payload_bytes=count,
                                           payload_status="verified" if count == expected else "size_mismatch")
                    except Exception as exc:
                        row.update(payload_status="failed", error_type=type(exc).__name__)
        except (OSError, zipfile.BadZipFile) as exc:
            a.update(status="unreadable", error_type=type(exc).__name__)
    return report


def stage(archive, entry_index, output, expected_sha256):
    """Copy verified payload with only spatial/scalar metadata. Not de-identification."""
    out = Path(output)
    if out.exists():
        raise FileExistsError("Output exists; choose a new path")
    with zipfile.ZipFile(archive) as z, z.open(z.infolist()[entry_index]) as stream:
        fields = read_header(stream)
        expected_bytes = validate_raw(fields)
        header = "NRRD0005\n" + "\n".join(k+": "+fields[k] for k in FIELDS if k in fields) + "\n\n"
        digest, count = hashlib.sha256(), 0
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("xb") as dest:
            dest.write(header.encode("ascii"))
            while True:
                chunk = stream.read(4 * 1024 * 1024)
                if not chunk:
                    break
                digest.update(chunk)
                count += len(chunk)
                dest.write(chunk)
    if count != expected_bytes or digest.hexdigest() != expected_sha256:
        raise ValueError("Staged data failed verification; do not load this output")
    return dict(status="staged_for_review", payload_sha256=digest.hexdigest(),
                payload_bytes=count, **geometry(fields))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archives", nargs="+")
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = inventory(args.archives, args.verify)
    with open(args.output, "x") as f:
        json.dump(result, f, indent=2, allow_nan=False)
    print(json.dumps(dict(archives=len(result["archives"]), volumes=len(result["volumes"]),
                         verified=sum(v["payload_status"] == "verified" for v in result["volumes"]),
                         patients_confirmed=0)))
