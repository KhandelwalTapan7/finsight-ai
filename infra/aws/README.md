# Deploying FinSight to AWS (staying in the free tier)

The app runs entirely free locally (local Qdrant, local SQLite, local file
storage). This doc explains what to swap to move it to AWS for the
"production deployment" story on your resume — without leaving free tier
for a portfolio-scale demo. **Bedrock, Textract, and Rekognition are not
free beyond a trivial amount of usage — this guide avoids them by design**,
keeping Groq + open-source components for the LLM/embedding work and using
AWS purely for infrastructure (compute, storage, auth).

## What changes, mapped to `STORAGE_MODE`

| Component | Local (`STORAGE_MODE=local`) | AWS (`STORAGE_MODE=aws`) | Free tier note |
|---|---|---|---|
| File upload | Saved to `./data/uploads` | Presigned S3 PUT URL | S3 free tier: 5GB storage, 20k GET / 2k PUT per month |
| Ingestion trigger | FastAPI `BackgroundTasks` | S3 `ObjectCreated` event → Lambda | Lambda free tier: 1M requests/month |
| Ingestion status | In-memory dict | DynamoDB table (`doc_id` as key) | DynamoDB free tier: 25GB storage, 25 RCU/WCU |
| Vector DB | Embedded Qdrant (`QdrantClient(path=...)`) | Qdrant Cloud free tier (1GB) **or** self-hosted on a `t2.micro` EC2 instance | EC2 free tier: 750 hrs/month for 12 months |
| Telemetry | SQLite | Same DynamoDB table (different item type), or a second table | Same free tier envelope |
| Auth | None (single demo user) | Cognito user pool | Cognito free tier: 10,000 MAUs |
| Compute (API) | `uvicorn` on your machine | Lambda (via Mangum) or ECS Fargate | Lambda free tier as above; Fargate has no perpetual free tier — prefer Lambda for a portfolio demo |

## Suggested build order

1. **S3 + presigned uploads.** Add an endpoint that returns a presigned PUT
   URL instead of accepting the file body directly; the Chainlit/frontend
   code uploads straight to S3.
2. **Lambda ingestion.** Package `src/ingestion` + `src/rag` as a Lambda
   (or a Lambda container image, since pymupdf/sentence-transformers are
   sizeable — a container image avoids the 250MB zip limit). Trigger on
   `s3:ObjectCreated:*`.
3. **DynamoDB status table.** Single table, `doc_id` as partition key,
   attributes `status`, `chunk_count`, `user_id`, `session_id`, and a
   `ttl` attribute (DynamoDB's native TTL feature deletes the item
   automatically — pair this with an EventBridge rule that also deletes
   the S3 object and the Qdrant vectors on the same schedule).
4. **Qdrant Cloud or EC2.** For a demo, Qdrant Cloud's free 1GB cluster is
   the lowest-effort option — swap `QdrantClient(path=...)` for
   `QdrantClient(url=..., api_key=...)` and nothing else in the codebase
   changes, since `vector_store.py` is the only place that touches the
   client.
5. **Cognito.** Add a hosted-UI login in front of Chainlit/the API; use
   the Cognito `sub` claim as `user_id` instead of the hardcoded
   `"demo-user"`.
6. **API Gateway + Lambda (Mangum)** or **ECS Fargate** for the FastAPI
   app itself, depending on whether you want to demo serverless or
   container deployment (your `ai-trip-planner-multi-agent` project
   already demonstrates serverless — deploying *this* one on ECS Fargate
   instead is a nice complementary signal that you know both patterns).

## Keeping this genuinely free

- Set a billing alarm at $1 the moment you create the AWS account.
- Delete the Qdrant Cloud cluster / EC2 instance when not actively
  demoing, since the 750 free EC2 hours/month is enough for continuous
  use but easy to accidentally exceed with multiple services.
- Everything above stays under free tier limits for portfolio-scale
  traffic (a handful of demo users, not production load) — this is a
  demo deployment, not a scaled production system, and the README should
  say that honestly in an interview.
