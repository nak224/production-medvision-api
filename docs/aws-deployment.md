# Deploy the portfolio API on AWS EC2

**Research and portfolio demonstration only; not for clinical diagnosis, treatment,
or medical use.** Use public research images only. This guide runs one CPU Docker
container on one EC2 instance. No orchestration or additional deployment platform
is needed. The API has no authentication or TLS; keep it private and use an SSH
tunnel for demonstrations.

## 1. Publish the image to GHCR

Merge the reviewed branch into `main`. GitHub Actions runs lint, formatting, offline
CPU tests and a Docker build before publishing to
`ghcr.io/nak224/production-medvision-api`. Publishing uses the job's built-in
`GITHUB_TOKEN` with `contents: read` and `packages: write`; no publishing secret is
needed. Pull request checks have only `contents: read` and never log into GHCR.

Tags are `latest` on `main`, `sha-<full-commit-sha>` on every published build, and
`1.0.0` for a version tag such as `v1.0.0` (including semantic prereleases).
Version tags do not update `latest`. Create a version tag only on reviewed code.
Use a SHA tag or image digest for reproducible deployments.

In GitHub, enable Actions and allow the workflow to publish packages. If the GHCR
package already exists, grant this repository Actions write access in the package's
settings. Choose public package visibility for anonymous pulls, or keep it private
and use the read-only login below. Confirm the workflow succeeded and the desired
image tag exists before deploying. The image contains no model weights.

## 2. Launch EC2

In the AWS console, launch an **Ubuntu Server 24.04 LTS, x86_64** instance (for
example `t3.medium`, 2 vCPU / 4 GiB RAM, with a 20 GiB EBS volume). The published
image targets Linux x86_64. Select a key pair and keep the private key locally.
A public subnet/public IP is the simplest setup for SSH and outbound image pulls.

Security group: allow inbound **TCP 22 only from your own public IP /32**. With the
SSH tunnel below, do not open port 8000. Allow outbound HTTPS (443) for GHCR, S3 and
package downloads, plus access to Ubuntu package repositories. Keep EC2 instance
metadata enabled with **IMDSv2 required**. For the optional S3 path with Docker's
bridge network, set the **metadata response hop limit to 2** so boto3 inside the
container can obtain the instance role credentials. Do not pass static AWS keys.

If using S3, attach an EC2 IAM role (trust principal `ec2.amazonaws.com`) with this
policy, replacing the bucket/key with the exact checkpoint object:

```json
{
  "Version": "2012-10-17",
  "Statement": [{
    "Effect": "Allow",
    "Action": "s3:GetObject",
    "Resource": "arn:aws:s3:::YOUR_BUCKET/models/model.pt"
  }]
}
```

Keep the bucket private with public access blocked. Upload your existing, trusted
baseline checkpoint to that key using the AWS console or your normal authenticated
AWS CLI. `s3:ListBucket` and write permissions are not required by the API. If the
object uses a customer-managed KMS key, the role also needs `kms:Decrypt` for that
key and permission in its key policy. No AWS account access is needed for the local
checkpoint option.

## 3. Install Docker and pull the image

On your computer (replace the key path and EC2 address):

```bash
chmod 400 ~/.ssh/medvision.pem
ssh -i ~/.ssh/medvision.pem ubuntu@EC2_PUBLIC_IP
```

On EC2:

```bash
sudo apt-get update
sudo apt-get install -y docker.io
sudo systemctl enable --now docker
sudo docker version
```

Only for a **private** GHCR image, create a GitHub personal access token (classic)
with `read:packages`, using an account with access to the package (authorize SSO
if required). Login under the same `sudo` Docker context used to pull:

```bash
read -r -p 'GitHub username: ' GHCR_USER
read -r -s -p 'GHCR read:packages token: ' GHCR_TOKEN
printf '%s' "$GHCR_TOKEN" | sudo docker login ghcr.io -u "$GHCR_USER" --password-stdin
unset GHCR_TOKEN
```

Pull the image (replace `latest` with `sha-<full-commit-sha>` to pin it):

```bash
IMAGE=ghcr.io/nak224/production-medvision-api:latest
sudo docker pull "$IMAGE"
```

## 4. Run with either a local checkpoint or S3

### Local checkpoint (default)

From your computer, copy the preserved baseline checkpoint; do not retrain to deploy:

```bash
scp -i ~/.ssh/medvision.pem artifacts/model.pt ubuntu@EC2_PUBLIC_IP:~/model.pt
```

On EC2, put it in a directory the container's non-root UID 10001 can read:

```bash
sudo install -d -m 755 /opt/medvision/models
sudo install -m 644 "$HOME/model.pt" /opt/medvision/models/model.pt
sudo docker run -d --name medvision-api --restart unless-stopped \
  -p 127.0.0.1:8000:8000 \
  --mount type=bind,src=/opt/medvision/models,dst=/models,readonly \
  -e MEDVISION_CHECKPOINT=/models/model.pt \
  "$IMAGE"
```

### Optional S3 checkpoint

With the IAM role attached and IMDS settings from step 2, run this **instead**:

```bash
sudo docker run -d --name medvision-api --restart unless-stopped \
  -p 127.0.0.1:8000:8000 \
  -e MEDVISION_MODEL_S3_URI=s3://YOUR_BUCKET/models/model.pt \
  -e AWS_DEFAULT_REGION=eu-central-1 \
  "$IMAGE"
```

Use your bucket's region. Boto3 uses its standard AWS credential chain; on EC2 this
uses the attached instance role. The checkpoint downloads once at startup to a
private temporary directory, loads through the existing CPU Predictor and metadata
checks, then the temporary copy is removed. Restarts download it again. Only grant
trusted maintainers permission to replace model artifacts.

| Environment variable | Behavior |
| --- | --- |
| `MEDVISION_CHECKPOINT` | Local path; defaults to `/models/model.pt` in Docker or `artifacts/model.pt` outside Docker. |
| `MEDVISION_MODEL_S3_URI` | Optional `s3://bucket/key`; takes precedence over the local setting when present. |
| `AWS_DEFAULT_REGION` | AWS region for the boto3 session, e.g. `eu-central-1`. |

An invalid S3 URI, failed download, or corrupt/incompatible checkpoint fails startup.
There is no silent fallback after an S3 error: remove the S3 setting and mount a local
checkpoint to switch back explicitly. A missing local file starts the app unready
with HTTP 503. No `.env` file is loaded automatically; export variables or pass
Docker `-e` / `--env-file`. Do not commit credentials or bake them into images.

## 5. Verify and access the API

On EC2, allow time for startup/download, then run:

```bash
sudo docker logs --tail 50 medvision-api
sudo docker inspect --format '{{.State.Health.Status}}' medvision-api
curl --fail http://127.0.0.1:8000/health
curl --fail http://127.0.0.1:8000/model-info
```

`/health` should report `status: ready`, `model_loaded: true` and the checkpoint's
model version. `/model-info` returns architecture, version, dataset, class order
and preprocessing metadata. Confirm the model version matches your intended artifact.
Docker reports `healthy` once its health check succeeds. For S3 errors check the
URI, IAM object policy, region, IMDSv2 hop limit and outbound connectivity in the logs.

From your computer, keep this tunnel running:

```bash
ssh -i ~/.ssh/medvision.pem -N -L 8000:127.0.0.1:8000 ubuntu@EC2_PUBLIC_IP
```

In another local terminal:

```bash
curl --fail http://127.0.0.1:8000/health
curl --fail http://127.0.0.1:8000/model-info
curl --fail -F 'file=@example.png' http://127.0.0.1:8000/predict
curl --fail -F 'files=@example.png' -F 'files=@another.jpg' \
  http://127.0.0.1:8000/predict/batch
```

Interactive API docs are at `http://127.0.0.1:8000/docs` through the tunnel. Batch
requests are all-or-nothing, ordered by upload, with at most 16 files, 5 MiB/file,
20 MiB total and 4 million pixels/image. FastAPI spools multipart uploads before
endpoint validation; these checks are not a network-level request-body limit.
Keep this unauthenticated portfolio service restricted to trusted users. If you
choose direct access for a controlled demo, change Docker's port mapping to
`8000:8000` and allow TCP 8000 from your IP /32 only; never open it to all sources.

To update, pull a specific new image tag, then `sudo docker stop medvision-api`
and `sudo docker rm medvision-api`, and repeat your chosen `docker run` command.
Model changes also require a restart. Preserve the previous image tag and artifact
for rollback. Stop or terminate EC2 when the demo is finished; EBS storage and
public IP resources may continue to incur charges until removed.

## API behavior

The service exposes `/health`, `/model-info`, `/predict`, and `/predict/batch`.
OpenAPI is available at `/openapi.json` and interactive documentation at `/docs`.
Use public research patches, not patient data.

Single-image prediction accepts the multipart field `file`; batches accept repeated
`files` fields. PNG/JPEG images, including grayscale converted to RGB, share the
same decoder and Predictor. Predictions return `class_id`, `label`, `confidence`,
all nine `probabilities`, and `model_version`. Batch results also include `filename`
and preserve upload order. Softmax confidence is uncalibrated.

Limits are 5 MiB and 4 million pixels per image, 16 files and 20 MiB total per batch.
A batch is all-or-nothing: an invalid image returns 413/415/422 with a zero-based
`index`, `filename`, and `error` in `detail`, without partial results. Count/total-size
limits return 413; missing uploads or corrupt images return 422; unsupported image
formats return 415. All uploaded files are closed on success and failure. FastAPI
spools multipart uploads before endpoint validation; these are not transport-level
body limits. Keep access restricted as described above.

`/model-info` returns values from the loaded checkpoint without exposing weights or
training configuration. The current checkpoint represents preprocessing as
`rgb-resize28-bilinear-normalize0.5-v1`, not separate input dimensions/format. Optional
`input_size` and `expected_input_format` fields appear only when present in metadata.

`/health` returns 200 when ready, or 503 with `model_not_ready` when a local checkpoint
is missing. Model metadata and prediction endpoints also return 503 without a model.
A corrupt or incompatible checkpoint fails startup. There is no artifact hot reload;
restart after replacing weights.

When set, `MEDVISION_MODEL_S3_URI` takes precedence over `MEDVISION_CHECKPOINT`.
Downloads use a temporary directory, the existing checkpoint validator and boto3's
standard credential chain. The temporary file is removed after loading, including
on failure. Invalid S3 URIs or download errors fail startup; unset the S3 variable
and mount a local checkpoint to switch back explicitly. No `.env` file is loaded
automatically. See the environment table above and [`.env.example`](../.env.example).
