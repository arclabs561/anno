"""Read a pinned Parquet text column over validated byte ranges without loading audio."""

import hashlib
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import quote

import pyarrow as pa
import pyarrow.parquet as pq


from http_range_file import RemoteFile


def text_partition(job, file_url, fetch):
    release, revision, _, cfg, split, files = job
    column = "instruction_text"

    def read_file(entry):
        url = file_url(release.repo, revision, entry["rfilename"])
        rows = 0
        digest = hashlib.sha256()
        with RemoteFile(url, entry["size"], fetch) as stream:
            parquet = pq.ParquetFile(stream, pre_buffer=False)
            for batch in parquet.iter_batches(columns=[column], use_threads=False):
                values = batch.column(column)
                if values.null_count or not (
                    pa.types.is_string(values.type) or pa.types.is_large_string(values.type)
                ):
                    raise ValueError("Missing text instruction")
                for value in values.to_pylist():
                    data = value.encode("utf-8")
                    digest.update(len(data).to_bytes(8, "big"))
                    digest.update(data)
                rows += len(values)
            if rows != parquet.metadata.num_rows or not rows:
                raise ValueError("Did not read every instruction in the shard")
        return rows, digest.hexdigest()

    with ThreadPoolExecutor(max_workers=4) as pool:
        observations = list(pool.map(read_file, files))
    observed_hash = hashlib.sha256()
    source_hash = hashlib.sha256()
    for entry, (_, digest) in zip(files, observations):
        name = entry["rfilename"]
        observed_hash.update(f"{digest}  {name}\n".encode())
        source_hash.update(f"{entry['lfs']['sha256']}  {name}\n".encode())
    path = "train/" + cfg if release.column == "audio-sample" else ""
    rows = sum(x[0] for x in observations)
    note = projection_scope(
        release, cfg, files, rows, source_hash.hexdigest(), observed_hash.hexdigest()
    )
    return [
        dict(
            id=release.ids,
            repository=release.repo,
            source_revision=revision,
            parquet_revision="",
            config=cfg,
            split=split,
            status="scanned_records",
            provider_rows=rows,
            rows=rows,
            tokens="",
            tagged="",
            reason="",
            scope_note=note,
            observation_column=column,
            payload_url=f"https://huggingface.co/datasets/{release.repo}/tree/{revision}/{quote(path)}#columns={column}",
            payload_sha256="",
        )
    ]


def projection_scope(release, cfg, files, rows, source_hash, observed_hash):
    if release.column == "audio-sample":
        selection = f"{len(files)} of {files[0]['part_shards']:,} shards in {cfg}"
    else:
        selection = f"all {len(files)} shards"
    note = (
        f"{release.scope} Validated {rows:,} text instructions against the same number of rows "
        f"in the footers of {selection}. Pinned LFS manifest SHA-256 {source_hash}. "
        f"Ordered text projection SHA-256 {observed_hash}."
    )
    if release.column == "audio-sample":
        note += " Selected " + ", ".join(e["rfilename"].split("/")[-1] for e in files) + "."
    return note
