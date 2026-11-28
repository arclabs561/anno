# S3 dataset cache in CI (GitHub Actions)

The loader tries these sources in order.

- local cache (`$ANNO_CACHE_DIR/datasets`, default is the platform cache under `anno/datasets`)
- optional S3 cache (`ANNO_S3_CACHE=1`, `ANNO_S3_BUCKET=...`) via **AWS CLI** (`aws s3 cp`)
- fallback URL download (last resort)

A contributor can build and score a pinned public corpus using only the publisher and a local cache. Personal Drive folders and deployment-specific S3 buckets require separate authorization. [Reproduce the SEC test from a fresh checkout](eval/dataset-coverage.md#restore-datasets-for-local-investigations)

Publisher downloads and dataset URL health checks verify HTTPS with bundled roots by default. To use a managed network’s certificate authorities, point `SSL_CERT_FILE` at its PEM trust bundle. An unreadable or empty bundle fails the request.

When S3 is enabled, the loader invokes the AWS CLI using its normal credential chain, including a local AWS profile or short-lived role credentials. The GitHub workflow exchanges a job OIDC token for an AWS role session. The AWS trust policy and the workflow’s event and branch checks must both allow the request. A successful CI read establishes that role’s access at the time of the run. Determining who else can read requires the deployed IAM and bucket policies. The example below grants access to read a dataset prefix. Actual access depends on the deployed IAM and bucket policies.

## Configure AWS OIDC for trusted CI

### 1) Create an IAM role with trust policy restricted to this repo

Use your AWS account ID, GitHub repository and default branch. The workflow checks that branch before requesting read credentials.

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Principal": { "Federated": "arn:aws:iam::<ACCOUNT_ID>:oidc-provider/token.actions.githubusercontent.com" },
      "Action": "sts:AssumeRoleWithWebIdentity",
      "Condition": {
        "StringEquals": {
          "token.actions.githubusercontent.com:aud": "sts.amazonaws.com",
          "token.actions.githubusercontent.com:sub": "repo:<OWNER>/<REPO>:ref:refs/heads/<DEFAULT_BRANCH>"
        }
      }
    }
  ]
}
```

Pull request jobs, including same-repository and fork PRs, run without the AWS role. The manual full evaluation uses the same default-branch check.

### 2) Attach a minimal S3 read policy

Replace `<BUCKET>` with the bucket whose owner authorized this CI role.

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Sid": "ListDatasetsPrefix",
      "Effect": "Allow",
      "Action": ["s3:ListBucket"],
      "Resource": ["arn:aws:s3:::<BUCKET>"],
      "Condition": { "StringLike": { "s3:prefix": ["datasets/*"] } }
    },
    {
      "Sid": "ReadDatasetsObjects",
      "Effect": "Allow",
      "Action": ["s3:GetObject"],
      "Resource": ["arn:aws:s3:::<BUCKET>/datasets/*"]
    }
  ]
}
```

This grants the reads the loader uses.

- `aws s3 cp s3://<BUCKET>/datasets/<DatasetId>.cache -`
- `aws s3 cp s3://<BUCKET>/datasets/<DatasetId>.cache.latest.json -`
- `aws s3 cp s3://<BUCKET>/datasets/by-sha256/<sha>/<DatasetId>.cache -`

### 3) Configure GitHub Actions variables

Set `ANNO_AWS_ROLE_ARN` and `ANNO_S3_BUCKET` under the repository’s Actions variables to enable the S3 read path on trusted CI jobs. Set `ANNO_AWS_REGION` if needed (default `us-east-1`). A manual cache audit can take its bucket as an explicit input. The normal documentation and figure build works with a checkout and its recorded measurements.

## Pin a generic loader input

For reproducible evaluations, call `DatasetLoader::load_or_download_pinned(id, sha256, PublisherFallback::Forbid)` with the expected digest of the bytes passed to the parser. The loader checks an existing local file first. On a cold cache with S3 enabled it tries the exact content-addressed snapshot and then the legacy key. A checksum mismatch leaves the candidate unpublished. Keep a separate local cache for each version you need to retain. Set `PublisherFallback::Allow` to permit a publisher fetch on a cache miss or denied read; the same digest check applies before the cache write.

For ordinary source selection, `load_or_download_with_receipt` returns the loaded dataset, its input digest, and the reason a configured S3 lookup gave way to a successful publisher load. Bare local files can have a known digest and unknown release or split. For old split-file LitBank caches, the digest covers the reconstructed pair actually parsed. These methods apply to the generic `LoadableDatasetId` route. Specialized relation and coreference loaders use their own input formats.

## What CI shares

Granting another person access to a private corpus copy can count as redistribution. Check the specific source text, annotation terms, any access agreement, and the notices for the bytes being shared. A bucket policy or Drive invitation controls who can retrieve those bytes. The publisher’s terms govern what recipients may receive and do with them. Contributors who need a restricted release obtain access from its publisher or custodian.

Job-local corpus files live in `$ANNO_CACHE_DIR/datasets`. Hosted CI publishes aggregate decision receipts and the scores and retry state Muxer needs for the next run. The checkpoint exporter rebuilds the SQLite index from that projected journal. Diagnostic text stays in the runner cache. The v2 checkpoint starts with empty history and leaves older artifacts on their existing retention schedule. [Checkpoint contents and retry behavior](EVALUATION_SAMPLING.md) Model and corpus files stay in the job workspace. Raw datasets have no approved shared publication package. The S3 reader can still use existing owner-provisioned storage for eligible inputs on trusted refs. The read exclusions cover PreCo, LitBank, CUAD, TimeBank 1.2 and its TIMEX derivative, including known legacy object names. The dataset registry distinguishes explicit publisher retrieval from authorized local-only inputs. [Source rules](../crates/anno-eval/src/eval/shared_cache_exclusions.tsv) · [Source-specific terms](QA.md#score-a-litbank-book-or-cohort)

The loader, `anno cache sync-s3`, and the bulk synchronization script skip raw dataset publication. Adding a sharable artifact requires checking rights for the source and the annotation layers, identifying the exact bytes and packaging the required notices. Local authorized evaluation remains available. Third-party model exports have their own terms. CI downloads needed assets into the job workspace and does not archive the entire Hugging Face hub. Restoring cross-job weight caching requires a reviewed model revision and a notice-bearing package for that revision.

Previously created Actions cache archives follow GitHub's existing access and expiry rules. A pull request can restore a cache from its base or the default branch, including some fork pull requests. Changing the workflow only affects future runs. Repository administrators can review old dataset cache keys and retention in Actions. [GitHub cache access rules](https://docs.github.com/en/actions/reference/workflows-and-actions/dependency-caching#restrictions-for-accessing-a-cache)

## Verify the live dataset cache

Run `gh workflow run verify-dataset-cache.yml --ref main` for an authenticated check. Supply `-f bucket=YOUR_AUTHORIZED_BUCKET` when the deployment has no bucket variable. The job reports the number of current objects under `datasets/`, including snapshots, pointers and sidecars. It then compares the WikiGold snapshot or legacy object with the pinned release SHA-256. The job summary retains the count if WikiGold is missing or fails verification. The count describes objects, so one dataset can contribute several. [Verification workflow](../.github/workflows/verify-dataset-cache.yml)

The inventory resolves exact loader filenames against the compiled dataset registry. It counts legacy payloads, content-addressed snapshots, pointers and manifests separately. A dataset with only a sidecar contributes to the metadata-only count. A pointer with no listed snapshot for its registry ID warrants review. Detecting a pointer to the wrong version would also require reading the pointer. The `unrecognized_key_shapes` map partitions the unrecognized keys by path and case-sensitive suffix. Only observed categories appear. Flat names are grouped by cache, sidecar, JSON, sequence-label, text or delimited, archive, and other suffixes. Under `by-sha256/`, the report distinguishes a lowercase 64-hex directory with one unregistered `.cache` filename from other layouts. The remaining groups cover other nested paths and directory-marker names. These counts describe naming conventions. Corpus identity, file format and permissions require separate evidence. The bulk-exclusion count can overlap any shape. The pinned WikiGold input stays on the runner. The downloadable artifact contains inventory counts, figures and numeric verification receipts, including fingerprints of the ordered tokens and gold spans.

The inventory figure groups registered legacy files and snapshots as payloads, and pointers and manifests as metadata. Its headline counts distinct IDs with payload files. On November 27, 2026, the authenticated listing contained 71 payload objects, 22 metadata records and 200 unrecognized keys. Those 71 payloads belonged to 60 registry IDs. This job verified the bytes and parsed annotations of WikiGold across a cold S3 load, a warm local load and the publisher. The separate publisher comparison below read 59 eligible payload objects. [November 27 inventory figure and evidence](https://github.com/arclabs561/anno/actions/runs/46032959618/artifacts/15214793714) · [Current inventory runs](https://github.com/arclabs561/anno/actions/workflows/verify-dataset-cache.yml)

The verified sample’s SHA-256 connects those exact bytes to the WikiGold annotation and source-text terms. This inventory job verifies the WikiGold payload. The publisher comparison below reports the byte matches it finds among other eligible inputs. The publication count uses the registry’s shared-publication policy. [WikiGold source and rights](QA.md#run-a-check)

The paired run records the cold S3 read, warm local read and fresh publisher retrieval. It checks the ordered tokens, BIO tags, gold spans and full-release heuristic NER scores across all three. Reported byte counts describe corpus payloads. Each load time measures one run before scoring.

The loader comparison runs on the default branch. Its read-only session can also read the WikiGold pointer and manifest so the ordinary loader can resolve its snapshot. The only permitted corpus payloads are the pinned WikiGold snapshot and the legacy WikiGold key. The check compares the cold S3 load and warm local load with the registered upstream release using the parsed tokens, gold spans and heuristic score.

## Compare cached payloads with measured publisher files

Run **Compare S3 cache with measured publisher files** on `main` for a current comparison. The trusted job lists the entire `datasets/` prefix once. It reads registered legacy payloads and snapshots for public or Hugging Face registry entries with an independent content measurement. The shared-read exclusions and their aliases apply. Separately provisioned full texts, clinical notes, and collections with public annotation layers only stay outside the read scope. The registry ledger records each entry’s measurement and access class. [Registry coverage](eval/dataset-coverage.csv) · [Run the comparison](https://github.com/arclabs561/anno/actions/workflows/cache-source-evidence.yml)

The 30-day **cache-source-evidence** artifact contains two evidence files and the rendered PNG and SVG.

- `cache-source-coverage.json` accounts for every listed object and every registry entry. Payload outcomes distinguish read exclusions, missing independent measurements, byte matches, unmatched bytes and read failures. A registry ID can have a legacy payload and several snapshots. The count of distinct IDs and the count of distinct byte digests each use their own denominator.
- `publisher-file-matches.csv` ties each verified object to measured publisher files by SHA-256, including files catalogued under a different registry ID. It carries the publisher’s split, member path where recorded, measurement unit and denominators. An archive match can support measurements of multiple members. Hub rows with a combined manifest digest spanning several files are excluded. Use the digest to count unique matched files.

A list-only session inventories the prefix. Subsequent read-only sessions name exact objects from that listing and check the listed ETag during each read. A snapshot must also match the digest in its key. For eligible snapshots, the reader checks the dataset’s listed latest pointer to distinguish current and superseded objects. A missing, invalid or dangling pointer leaves the snapshot position unresolved. Only aggregate counts and independently known publisher hashes appear in the artifact. [Comparator and source ledgers](../scripts/qa/cache_source_evidence.py)

The dataset guide preserves its dated [comparison counts](eval/s3-source-coverage.json) and [matched publisher files](eval/s3-publisher-file-matches.csv) in Git. Newer comparisons are available from Actions as they run.

Run the comparison on demand after changing a measured-file ledger or its verifier. The Dataset figures workflow can also run the inventory and publisher comparison when the corresponding audit inputs are enabled on its default-branch dispatch. Each artifact also contains `s3-publisher-matches.png` and its SVG. The object line counts recognized payloads, successful reads and objects that match their own publisher. The registry line counts distinct IDs with a payload and IDs with an own-publisher match. The bars partition the recognized payload objects, including read failures, and the figure identifies which keys contributed only to the inventory. Rebuild the figure from the downloaded JSON with `uv run --script scripts/plot_s3_source_evidence.py cache-source-coverage.json .`. [Comparison workflow and artifacts](https://github.com/arclabs561/anno/actions/workflows/cache-source-evidence.yml)

![Dated November 27 comparison showing 71 recognized payload objects, 59 reads and 33 own-publisher matches. They represent 60 IDs with payloads and 24 with a matching file.](figures/s3-publisher-matches.png)
