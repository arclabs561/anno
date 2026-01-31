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

The workflow will:

- set `ANNO_CACHE_DIR=$HOME/.anno_cache`
- enable S3 cache only when `ANNO_AWS_ROLE_ARN` is set and the run is not a fork PR

## Optional: sync local cache → S3

If you already have datasets cached locally and want to refresh the shared S3 cache:

```bash
cargo run -p anno --features "eval-advanced discourse" -- cache sync-s3 --dry-run
cargo run -p anno --features "eval-advanced discourse" -- cache sync-s3 --limit 50
cargo run -p anno --features "eval-advanced discourse" -- cache sync-s3 --datasets Wnut17,DocRED
```

## Fallback: scoped IAM user + GitHub secrets (long-lived)

This is **not recommended** compared to OIDC, but it meets the “limited-scope secret” requirement.

High-level:

- Create an IAM user (e.g. `anno-gh-ci`)
- Attach the same minimal S3 read policy as above
- Create an access key
- Store into GitHub Secrets:
  - `ANNO_AWS_ACCESS_KEY_ID`
  - `ANNO_AWS_SECRET_ACCESS_KEY`

If you do this, also set:

- `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY` via workflow `env:`
- rotate keys periodically

