# S3 dataset cache in CI (GitHub Actions)

This repo's dataset loader supports a tiered cache:

- local cache (`ANNO_CACHE_DIR`, default is platform cache)
- optional S3 cache (`ANNO_S3_CACHE=1`, `ANNO_S3_BUCKET=...`) via **AWS CLI** (`aws s3 cp`)
- fallback URL download (last resort)

For CI, the safest configuration is **AWS OIDC → role assumption** (no long-lived secrets).

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

- For PRs (including same-repo PRs), we do **not** enable S3 in CI (see workflow guards).
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

- `aws s3 cp s3://<BUCKET>/datasets/<id>.json -`
- `aws s3 cp s3://<BUCKET>/datasets/<id>.latest.json -`
- `aws s3 cp s3://<BUCKET>/datasets/by-sha256/<sha>/<id>.json -`

### 3) Configure GitHub Actions variables

In GitHub repo settings:

- **Variables**
  - `ANNO_AWS_ROLE_ARN`: the role ARN you created
  - `ANNO_AWS_REGION`: region for STS (commonly `us-east-1`)
  - optional `ANNO_S3_BUCKET`: override bucket (defaults to `arc-anno-data`)

## What CI shares

Job-local corpus files live in `$ANNO_CACHE_DIR/datasets`. Hosted CI caches model repositories separately and publishes aggregate decision receipts and evaluator history. Raw datasets have no approved shared publication package. The S3 reader can still use existing owner-provisioned storage for eligible inputs on trusted refs. The read exclusions cover PreCo, LitBank, CUAD, TimeBank 1.2 and its TIMEX derivative, including known legacy object names. The dataset registry distinguishes explicit publisher retrieval from authorized local-only inputs. [Source rules](../crates/anno-eval/src/eval/shared_cache_exclusions.tsv) · [Source-specific terms](QA.md#score-a-litbank-book-or-cohort)

The loader, `anno cache sync-s3`, and the bulk synchronization script skip raw dataset publication. Adding a sharable artifact requires checking rights for the source and the annotation layers, identifying the exact bytes and packaging the required notices. Local authorized evaluation remains available. Third-party model exports have their own terms. CI downloads needed assets into the job workspace and does not archive the entire Hugging Face hub. Restoring cross-job weight caching requires a reviewed model revision and a notice-bearing package for that revision.

Previously created Actions cache archives follow GitHub's existing access and expiry rules. A pull request can restore a cache from its base or the default branch, including some fork pull requests. Changing the workflow only affects future runs. Repository administrators can review old dataset cache keys and retention in Actions. [GitHub cache access rules](https://docs.github.com/en/actions/reference/workflows-and-actions/dependency-caching#restrictions-for-accessing-a-cache)
