import http from "k6/http";
import { check, sleep } from "k6";
import { Counter, Trend } from "k6/metrics";

const ROUTER_URL = __ENV.ROUTER_URL || "http://localhost:8000";
const ROUTER_API_KEY = __ENV.ROUTER_API_KEY || "test-key";

export const options = {
  scenarios: {
    mixed_load: {
      executor: "constant-arrival-rate",
      rate: 5,
      timeUnit: "1s",
      duration: "60s",
      preAllocatedVUs: 10,
      maxVUs: 50,
    },
  },
};

const errorCounter = new Counter("errors");
const latencyTrend = new Trend("latency");

const embeddingModels = ["bge-m3", "multilingual-e5-large", "frida"];

function callEmbeddings(model) {
  const payload = JSON.stringify({
    model,
    input: "bench payload",
  });
  const res = http.post(`${ROUTER_URL}/v1/embeddings`, payload, {
    headers: {
      "Content-Type": "application/json",
      "X-API-Key": ROUTER_API_KEY,
    },
  });
  recordResult(res);
}

function callRerank() {
  const docs = Array.from({ length: 20 }).map((_, idx) => `doc-${idx}`);
  const payload = JSON.stringify({
    model: "rerank-base",
    query: "bench query",
    documents: docs,
    top_k: 5,
  });
  const res = http.post(`${ROUTER_URL}/v1/rerank`, payload, {
    headers: {
      "Content-Type": "application/json",
      "X-API-Key": ROUTER_API_KEY,
    },
  });
  recordResult(res);
}

function recordResult(res) {
  latencyTrend.add(res.timings.duration);
  const ok = check(res, {
    "status is 2xx": (r) => r.status >= 200 && r.status < 300,
  });
  if (!ok) {
    errorCounter.add(1);
  }
}

export default function () {
  // Alternate between embeddings and rerank calls to maintain mixed workload.
  if (__ITER % 2 === 0) {
    const model = embeddingModels[__ITER % embeddingModels.length];
    callEmbeddings(model);
  } else {
    callRerank();
  }
  sleep(0.1);
}

export function handleSummary(data) {
  const p50 = data.metrics.latency["p(50)"];
  const p95 = data.metrics.latency["p(95)"];
  const errorRate =
    data.metrics.errors && data.metrics.errors.count
      ? data.metrics.errors.count / data.metrics.http_reqs.count
      : 0;

  console.log(`p50: ${p50} ms`);
  console.log(`p95: ${p95} ms`);
  console.log(`error rate: ${(errorRate * 100).toFixed(2)}%`);
  return {};
}
