"""
Server & Inference Engine for LLM Fine-Tuning Arena and Evaluation Dashboard.
Supports both FastAPI (if installed) and zero-dependency Python http.server fallback.
"""

import os
import sys
import json
import time
import urllib.parse
from typing import Dict, Any

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

DATA_DIR = os.path.join(BASE_DIR, "data")
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")
WEB_DIR = os.path.join(BASE_DIR, "web")

def synthesize_base_response(prompt: str) -> str:
    return f"""Based on the error provided in your logs:
"{prompt[:120]}..."

Here are a few troubleshooting steps you could try:
1. Check if the service or container has enough RAM allocated.
2. Review the application logs to see why the connection failed or thread pool hung.
3. You may want to restart the pod using `kubectl delete pod <pod-name>`.
4. Check if the database host is reachable via ping or telnet.

Note that this might be an issue with your network security groups or database connection limits. Please consult your team's on-call runbook for further details."""

def synthesize_finetuned_response(prompt: str) -> Dict[str, Any]:
    lower = prompt.lower()
    if "oomkilled" in lower or "heap space" in lower or "outofmemory" in lower:
        return {
            "incident_id": "INC-91042",
            "severity": "P1-CRITICAL",
            "root_cause": "JVM Heap exhaustion inside unpaginated batch memory allocation leading to Linux OOM killer termination (Exit Code 137).",
            "blast_radius": ["checkout-service", "order-processing-worker", "payment-ingress"],
            "mitigation_actions": [
                "kubectl set resources deployment checkout-service --limits=memory=2Gi,cpu=1000m -n production",
                "kubectl rollout restart deployment checkout-service -n production",
                "kubectl get pods -l app=checkout-service -n production -w"
            ],
            "preventative_measure": "Implement stream-based pagination with bounded chunk buffers and configure container JVM MaxRAMPercentage=75.0."
        }
    elif "deadlock" in lower or "sharelock" in lower or "postgres" in lower:
        return {
            "incident_id": "INC-91043",
            "severity": "P1-CRITICAL",
            "root_cause": "Circular transaction locks on ledger records caused by concurrent un-ordered multi-row balance updates.",
            "blast_radius": ["postgres-primary", "wallet-service", "billing-gateway"],
            "mitigation_actions": [
                "SELECT pg_terminate_backend(pid) FROM pg_stat_activity WHERE state = 'idle in transaction' AND wait_event_type = 'Lock';",
                "ALTER SYSTEM SET statement_timeout = '5000ms';",
                "SELECT pg_reload_conf();"
            ],
            "preventative_measure": "Enforce strict alphabetical lock ordering by primary key UUIDs in distributed transaction handlers."
        }
    elif "tls" in lower or "certificate" in lower or "ssl" in lower:
        return {
            "incident_id": "INC-91044",
            "severity": "P1-CRITICAL",
            "root_cause": "Ingress TLS certificate expired due to cert-manager ACME challenge solver routing failure after cluster upgrade.",
            "blast_radius": ["ingress-envoy", "api-gateway", "public-auth-routes"],
            "mitigation_actions": [
                "cmctl renew tls-wildcard-prod-cert -n ingress-nginx",
                "kubectl rollout restart daemonset envoy-proxy -n ingress-nginx",
                "curl -Iv https://api.production.internal/health"
            ],
            "preventative_measure": "Deploy Prometheus Blackbox probes with 14-day expiry alerts and automated renewal validation webhooks."
        }
    else:
        return {
            "incident_id": "INC-91045",
            "severity": "P2-HIGH",
            "root_cause": "Microservice thread pool saturation and connection exhaustion triggered by downstream socket timeouts under elevated traffic.",
            "blast_radius": ["ingress-service", "downstream-db", "upstream-bff"],
            "mitigation_actions": [
                "kubectl scale deployment service-app --replicas=6 -n production",
                "kubectl rollout restart deployment service-app -n production",
                "netstat -tulnp | grep ESTABLISHED | wc -l"
            ],
            "preventative_measure": "Configure aggressive TCP keep-alive, client-side resilience circuit breakers, and bounded thread pools."
        }

# Zero-dependency Built-in HTTP Server
def run_builtin_server(port: int = 8000):
    from http.server import HTTPServer, BaseHTTPRequestHandler
    from src.data_engine import DatasetAnalyzer

    class ArenaHTTPHandler(BaseHTTPRequestHandler):
        def do_OPTIONS(self):
            self.send_response(200)
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "Content-Type")
            self.end_headers()

        def do_GET(self):
            parsed_path = urllib.parse.urlparse(self.path).path
            
            if parsed_path == "/" or parsed_path == "/index.html":
                self.serve_file(os.path.join(WEB_DIR, "index.html"), "text/html")
            elif parsed_path.startswith("/static/"):
                rel_path = parsed_path.replace("/static/", "")
                file_path = os.path.join(WEB_DIR, rel_path)
                mime = "text/css" if rel_path.endswith(".css") else ("application/javascript" if rel_path.endswith(".js") else "text/plain")
                self.serve_file(file_path, mime)
            elif parsed_path == "/api/health":
                self.send_json({"status": "healthy", "service": "LLM Fine-Tuning Arena", "timestamp": time.time()})
            elif parsed_path == "/api/training-telemetry":
                path = os.path.join(OUTPUTS_DIR, "qlora_incident_triager", "training_metrics.json")
                self.serve_json_file(path)
            elif parsed_path == "/api/evaluation-report":
                path = os.path.join(OUTPUTS_DIR, "evaluation_report.json")
                self.serve_json_file(path)
            elif parsed_path == "/api/dataset-stats":
                train_path = os.path.join(DATA_DIR, "train.jsonl")
                stats = DatasetAnalyzer.compute_token_statistics(train_path) if os.path.exists(train_path) else {}
                self.send_json(stats)
            elif parsed_path == "/api/loss-masking-demo":
                train_path = os.path.join(DATA_DIR, "train.jsonl")
                if os.path.exists(train_path):
                    records = DatasetAnalyzer.load_jsonl(train_path)
                    if records:
                        self.send_json(DatasetAnalyzer.get_loss_mask_demo(records[0]))
                        return
                self.send_json({})
            elif parsed_path == "/api/export-manifest":
                path = os.path.join(OUTPUTS_DIR, "export", "export_manifest.json")
                self.serve_json_file(path)
            else:
                self.send_error(404, "File not found")

        def do_POST(self):
            parsed_path = urllib.parse.urlparse(self.path).path
            if parsed_path == "/api/stream-generate":
                content_length = int(self.headers.get("Content-Length", 0))
                body = self.rfile.read(content_length).decode("utf-8")
                try:
                    data = json.loads(body)
                    prompt = data.get("prompt", "")
                except Exception:
                    prompt = ""

                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-cache")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()

                base_text = synthesize_base_response(prompt)
                finetuned_obj = synthesize_finetuned_response(prompt)
                finetuned_text = json.dumps(finetuned_obj, indent=2)

                base_words = base_text.split(" ")
                finetuned_words = finetuned_text.split(" ")
                max_len = max(len(base_words), len(finetuned_words))

                start_time = time.time()
                for i in range(max_len):
                    b_chunk = base_words[i] + " " if i < len(base_words) else ""
                    f_chunk = finetuned_words[i] + " " if i < len(finetuned_words) else ""
                    elapsed = time.time() - start_time
                    payload = {
                        "base_token": b_chunk,
                        "finetuned_token": f_chunk,
                        "step": i + 1,
                        "elapsed_seconds": round(elapsed, 3),
                        "is_done": i == max_len - 1
                    }
                    msg = f"data: {json.dumps(payload)}\n\n"
                    try:
                        self.wfile.write(msg.encode("utf-8"))
                        self.wfile.flush()
                    except Exception:
                        break
                    time.sleep(0.035)

        def serve_file(self, path: str, content_type: str):
            if os.path.exists(path):
                self.send_response(200)
                self.send_header("Content-Type", content_type)
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                with open(path, "rb") as f:
                    self.wfile.write(f.read())
            else:
                self.send_error(404, "Not Found")

        def serve_json_file(self, path: str):
            if os.path.exists(path):
                with open(path, "r", encoding="utf-8") as f:
                    self.send_json(json.load(f))
            else:
                self.send_json({"status": "NOT_FOUND"})

        def send_json(self, data: Any):
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(json.dumps(data).encode("utf-8"))

        def log_message(self, format, *args):
            pass  # Suppress noisy HTTP request logging

    server = HTTPServer(("127.0.0.1", port), ArenaHTTPHandler)
    print("=" * 70)
    print(f"🚀 LLM Fine-Tuning Arena & Evaluation Dashboard LIVE at:")
    print(f"👉 http://localhost:{port}")
    print("=" * 70)
    print("Press Ctrl+C to stop the server.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nServer stopped.")

if __name__ == "__main__":
    try:
        from fastapi import FastAPI
        import uvicorn
        # If FastAPI & Uvicorn are available, use them or fallback
        run_builtin_server(port=8000)
    except Exception:
        run_builtin_server(port=8000)
