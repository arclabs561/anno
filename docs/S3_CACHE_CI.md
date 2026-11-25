# S3 dataset cache in CI (GitHub Actions)

This repo's dataset loader supports a tiered cache:

- local cache (`$ANNO_CACHE_DIR/datasets`, default is the platform cache under `anno/datasets`)
- optional S3 cache (`ANNO_S3_CACHE=1`, `ANNO_S3_BUCKET=...`) via **AWS CLI** (`aws s3 cp`)
- fallback URL download (last resort)

When S3 is enabled, the loader invokes the AWS CLI using its normal credential chain, including a local AWS profile or short-lived role credentials. The GitHub workflow exchanges a job OIDC token for an AWS role session. Both the AWS trust policy and the workflow event/ref guard must allow the request. The example below grants access to read a dataset prefix. Actual access depends on the deployed IAM and bucket policies.

## Recommended: AWS OIDC role (no secrets)

### 1) Create an IAM role with trust policy restricted to this repo

Replace:

- `<ACCOUNT_ID>`
- `<ROLE_NAME>` (e.g. `anno-gh-ci-s3-cache`)

Trust policy (restrict to this repo + selected branches):

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
          "token.actions.githubusercontent.com:aud": "sts.amazonaws.com"
        },
        "StringLike": {
          "token.actions.githubusercontent.com:sub": [
            "repo:arclabs561/anno:ref:refs/heads/main",
            "repo:arclabs561/anno:ref:refs/heads/master",
            "repo:arclabs561/anno:ref:refs/heads/eval-*"
          ]
        }
      }
    }
  ]
}
```

Notes:

- Pull request jobs, including same-repository and fork PRs, run without the AWS role.
- The regular trusted jobs run on `main` or `master`. A manual full evaluation can also use a branch named `eval-*` when IAM allows it. A branch named `eval/experiment` does not match that pattern.
- If you do not run `eval-*` branches, remove that line to reduce scope.

### 2) Attach a minimal S3 read policy

Replace `<BUCKET>` (default is `arc-anno-data`):

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

This is enough for:

- `aws s3 cp s3://<BUCKET>/datasets/<DatasetId>.cache -`
- `aws s3 cp s3://<BUCKET>/datasets/<DatasetId>.cache.latest.json -`
- `aws s3 cp s3://<BUCKET>/datasets/by-sha256/<sha>/<DatasetId>.cache -`

### 3) Configure GitHub Actions variables

In GitHub repo settings:

- **Variables**
  - `ANNO_AWS_ROLE_ARN`: the role ARN you created
  - `ANNO_AWS_REGION`: region for STS (commonly `us-east-1`)
  - optional `ANNO_S3_BUCKET`: override bucket (defaults to `arc-anno-data`)

## What CI shares

Job-local corpus files live in `$ANNO_CACHE_DIR/datasets`. Hosted CI publishes aggregate decision receipts and the scores and retry state Muxer needs for the next run. The checkpoint exporter rebuilds the SQLite index from that projected journal. Diagnostic text stays in the runner cache. The v2 checkpoint starts with empty history and leaves older artifacts on their existing retention schedule. [Checkpoint contents and retry behavior](EVALUATION_SAMPLING.md) Model and corpus files stay in the job workspace. Raw datasets have no approved shared publication package. The S3 reader can still use existing owner-provisioned storage for eligible inputs on trusted refs. The read exclusions cover PreCo, LitBank, CUAD, TimeBank 1.2 and its TIMEX derivative, including known legacy object names. The dataset registry distinguishes explicit publisher retrieval from authorized local-only inputs. [Source rules](../crates/anno-eval/src/eval/shared_cache_exclusions.tsv) · [Source-specific terms](QA.md#score-a-litbank-book-or-cohort)

The loader, `anno cache sync-s3`, and the bulk synchronization script skip raw dataset publication. Adding a sharable artifact requires checking rights for the source and the annotation layers, identifying the exact bytes and packaging the required notices. Local authorized evaluation remains available. Third-party model exports have their own terms. CI downloads needed assets into the job workspace and does not archive the entire Hugging Face hub. Restoring cross-job weight caching requires a reviewed model revision and a notice-bearing package for that revision.

Previously created Actions cache archives follow GitHub's existing access and expiry rules. A pull request can restore a cache from its base or the default branch, including some fork pull requests. Changing the workflow only affects future runs. Repository administrators can review old dataset cache keys and retention in Actions. [GitHub cache access rules](https://docs.github.com/en/actions/reference/workflows-and-actions/dependency-caching#restrictions-for-accessing-a-cache)

## Verify the live dataset cache

Run `gh workflow run verify-dataset-cache.yml --ref main` for an authenticated check. The job reports the number of current objects under `datasets/`, including snapshots, pointers and sidecars. It then compares the WikiGold snapshot or legacy object with the pinned release SHA-256. The job summary retains the count if WikiGold is missing or fails verification. The count describes objects, so one dataset can contribute several. [Verification workflow](../.github/workflows/verify-dataset-cache.yml)

The inventory resolves exact loader filenames against the compiled dataset registry. It counts legacy payloads, content-addressed snapshots, pointers and manifests separately. A dataset with only a sidecar contributes to the metadata-only count. A pointer with no listed snapshot for its registry ID warrants review. Detecting a pointer to the wrong version would also require reading the pointer. Unrecognized keys include other file layouts. The report counts legacy names matching the bulk exclusion rules without assigning them to a corpus. All public outputs are aggregate counts except for the reviewed WikiGold sample.

The verified sample’s SHA-256 connects those exact bytes to the WikiGold annotation and source-text terms. Other objects remain unverified even when their keys contain a known name or digest. The publication count uses the registry’s shared-publication policy. [WikiGold source and rights](QA.md#run-a-check)

The paired run records the cold S3 read, warm local read and fresh publisher retrieval. It checks the ordered tokens, BIO tags, gold spans and full-release heuristic NER scores across all three. Reported byte counts describe corpus payloads. Each load time measures one run before scoring.

The loader comparison runs on the default branch. Its read-only session can also read the WikiGold pointer and manifest so the ordinary loader can resolve its snapshot. The only permitted corpus payloads are the pinned WikiGold snapshot and the legacy WikiGold key. The check compares the cold S3 load and warm local load with the registered upstream release using the parsed tokens, gold spans and heuristic score.
