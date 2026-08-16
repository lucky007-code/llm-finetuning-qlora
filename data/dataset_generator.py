"""
Synthetic & Curated Dataset Generator for Enterprise Incident Root Cause Analysis (RCA).
Generates high-signal train, val, and test splits formatted for instruction fine-tuning.
"""

import json
import os
import random
from typing import List, Dict, Any

INCIDENT_TEMPLATES = [
    {
        "category": "Kubernetes & Container Orchestration",
        "service": "checkout-service-pod-7d9fb48b9c",
        "raw_log": """[ERROR] 2026-08-16T14:22:01.104Z [checkout-service] [main] org.springframework.boot.SpringApplication: Application run failed
java.lang.OutOfMemoryError: Java heap space
    at com.eshop.checkout.service.CartAggregator.processLargeCart(CartAggregator.java:184)
    at com.eshop.checkout.controller.CheckoutController.checkout(CheckoutController.java:52)
k8s_event: Pod checkout-service-pod-7d9fb48b9c status changed to OOMKilled (Exit Code 137). Restart count: 4.
Node: worker-node-us-east-1a, Available Memory: 1.2Gi / 16Gi""",
        "structured_response": {
            "incident_id": "INC-84920",
            "severity": "P1-CRITICAL",
            "root_cause": "JVM Heap exhaustion inside CartAggregator.processLargeCart triggered by unpaginated bulk cart payloads exceeding the 512MB container memory limit, resulting in continuous OOMKilled (Exit Code 137) crash-loop.",
            "blast_radius": ["checkout-service", "payment-gateway", "order-processing-pipeline"],
            "mitigation_actions": [
                "kubectl set resources deployment checkout-service --limits=memory=2Gi,cpu=1000m --requests=memory=1Gi,cpu=500m -n production",
                "kubectl rollout restart deployment checkout-service -n production",
                "kubectl get pods -n production -l app=checkout-service -w"
            ],
            "preventative_measure": "Implement bounded stream processing with Spring Data Pageable pagination for cart items and enforce API gateway payload size caps."
        }
    },
    {
        "category": "Distributed Database & Deadlocks",
        "service": "postgres-primary-aurora-cluster",
        "raw_log": """FATAL: 2026-08-16T15:02:11.890Z [db-pool-worker-42] postgres[14820]: [4-1] LOG: process 14820 detected deadlock while waiting for ShareLock on transaction 8892120
DETAIL: Process 14820 waits for ShareLock on transaction 8892120; blocked by process 14829.
Process 14829 waits for ExclusiveLock on tuple (412, 19) of relation 16402 'ledger_accounts'; blocked by process 14820.
HINT: See server log for query details.
STATEMENT: UPDATE ledger_accounts SET balance = balance - 150.00 WHERE account_id = 'ACC-9921' RETURNING balance;
Active connections: 498/500 (99.6% saturation). Client connection timeouts spiking to 8.4s.""",
        "structured_response": {
            "incident_id": "INC-84921",
            "severity": "P1-CRITICAL",
            "root_cause": "Circular transaction locks on relation 'ledger_accounts' due to concurrent un-ordered multi-account debit/credit transactions leading to cross-process PostgreSQL deadlocks and pool connection starvation.",
            "blast_radius": ["postgres-primary", "wallet-service", "billing-worker-cron"],
            "mitigation_actions": [
                "SELECT pg_terminate_backend(pid) FROM pg_stat_activity WHERE state = 'idle in transaction' AND wait_event_type = 'Lock';",
                "ALTER SYSTEM SET statement_timeout = '5000ms';",
                "SELECT pg_reload_conf();"
            ],
            "preventative_measure": "Sort account UUIDs lexicographically prior to acquiring row-level locks in distributed transactions to guarantee consistent lock ordering."
        }
    },
    {
        "category": "Kafka Partition Lag & Consumer Backpressure",
        "service": "kafka-cluster-prod",
        "raw_log": """[WARN] 2026-08-16T16:11:45.312Z [org.apache.kafka.clients.consumer.internals.ConsumerCoordinator] [consumer-group: telemetry-ingest]
Consumer group telemetry-ingest rebalance in progress. Revoking partitions: [telemetry-raw-0, telemetry-raw-1, telemetry-raw-2].
Reason: org.apache.kafka.common.errors.WakeupException: max.poll.interval.ms (300000ms) exceeded.
Current lag on topic telemetry-raw: 1,489,200 messages (+24,000 msg/sec).
Downstream Elasticsearch cluster reporting 429 Too Many Requests (bulk queue capacity 200 exceeded).""",
        "structured_response": {
            "incident_id": "INC-84922",
            "severity": "P2-HIGH",
            "root_cause": "Downstream Elasticsearch HTTP 429 backpressure stalled the consumer batch processing loop beyond max.poll.interval.ms, triggering continuous Kafka consumer group rebalances and exponential partition lag.",
            "blast_radius": ["telemetry-ingest-consumer", "telemetry-raw-topic", "elasticsearch-analytics"],
            "mitigation_actions": [
                "curl -X PUT 'http://elasticsearch:9200/_cluster/settings' -H 'Content-Type: application/json' -d '{\"transient\": {\"thread_pool.write.queue_size\": 1000}}'",
                "kubectl scale deployment telemetry-consumer --replicas=12 -n data-platform",
                "kafka-consumer-groups.sh --bootstrap-server kafka:9092 --group telemetry-ingest --describe"
            ],
            "preventative_measure": "Implement an exponential backoff circuit breaker with local disk spillover in consumer workers and tune max.poll.records to 250."
        }
    },
    {
        "category": "SSL/TLS Certificate Expiry & Network Boundary",
        "service": "api-gateway-envoy-ingress",
        "raw_log": """[CRITICAL] 2026-08-16T17:40:02.001Z [envoy.transport_sockets.tls] [C19823] remote address: 198.51.100.4:54321
TLS handshake failed: SSL routines:OPENSSL_internal:CERTIFICATE_VERIFY_FAILED
error:1000007d:SSL routines:OPENSSL_internal:CERTIFICATE_VERIFY_FAILED: ../ssl/handshake_client.cc:1132
Downstream client handshake rejected. Issuer: Let's Encrypt R3. Valid Until: 2026-08-16T17:00:00Z.
502 Bad Gateway rate surged to 94.2% on /v1/auth and /v1/payments.""",
        "structured_response": {
            "incident_id": "INC-84923",
            "severity": "P1-CRITICAL",
            "root_cause": "Production ingress TLS certificate expired at 17:00:00Z due to cert-manager ACME HTTP-01 challenge ingress ingress rule misconfiguration during recent ingress controller upgrade.",
            "blast_radius": ["api-gateway", "public-api-routes", "mobile-app-clients"],
            "mitigation_actions": [
                "kubectl apply -f https://cert-manager.io/manifests/v1.14.0.yaml",
                "cmctl renew tls-wildcard-prod-cert -n ingress-nginx",
                "kubectl rollout restart daemonset envoy-proxy -n ingress-nginx"
            ],
            "preventative_measure": "Deploy Prometheus Blackbox Exporter probes with 14-day and 7-day TLS expiration alerting rules and automated cert-manager clusterissuer health tests."
        }
    },
    {
        "category": "Redis Cache Stampede & Thundering Herd",
        "service": "redis-cluster-cache",
        "raw_log": """[ERROR] 2026-08-16T18:05:33.220Z [catalog-service] [http-nio-8080-exec-19] RedisConnectionException: Command timed out after 3000ms
Redis Master CPU Utilization: 100.0%. Instantaneous Ops Per Second: 185,000.
Evicted keys: 0. Connected clients: 10,000 (maxclients limit reached).
Cache key 'homepage_curated_deals_v2' expired at 18:05:30Z. Subsequent 8,000 RPS simultaneously queried MySQL product_catalog table.
MySQL DB CPU spiked from 18% to 99.8% with 450 slow queries in 'Sending data' state.""",
        "structured_response": {
            "incident_id": "INC-84924",
            "severity": "P1-CRITICAL",
            "root_cause": "Cache stampede (thundering herd) caused by simultaneous TTL expiration of high-cardinality key 'homepage_curated_deals_v2' lacking probabilistic early expiration (XFetch) or mutex locking.",
            "blast_radius": ["redis-cache-cluster", "catalog-mysql-db", "homepage-service"],
            "mitigation_actions": [
                "redis-cli -h redis-master -p 6379 CONFIG SET timeout 30",
                "redis-cli -h redis-master -p 6379 SET 'homepage_curated_deals_v2' '{\"stale\": true}' EX 3600",
                "kubectl scale deployment catalog-service --replicas=8 -n production"
            ],
            "preventative_measure": "Implement probabilistic early expiration (XFetch algorithm with delta) and distributed Mutex locking in Redis cache client adapters."
        }
    },
    {
        "category": "Distributed Tracing & gRPC Deadline Exceeded",
        "service": "order-orchestrator-grpc",
        "raw_log": """[WARN] 2026-08-16T19:15:00.812Z [grpc-default-executor-11] io.grpc.StatusRuntimeException: DEADLINE_EXCEEDED: context deadline exceeded after 1999.8ms
TraceID: 4bf92f3577b34da6a3ce929d0e0e4736, SpanID: 00f067aa0ba902b7
gRPC method: /inventory.InventoryService/ReserveStock
Upstream caller: checkout-bff (HTTP 504 Gateway Timeout returned to client)
Inventory database reports lock wait timeout on table 'sku_inventory_reservations'.""",
        "structured_response": {
            "incident_id": "INC-84925",
            "severity": "P2-HIGH",
            "root_cause": "gRPC call deadline (2.0s) exceeded due to unindexed foreign key lookups on sku_inventory_reservations under flash-sale concurrency spikes.",
            "blast_radius": ["order-orchestrator", "inventory-service", "checkout-bff"],
            "mitigation_actions": [
                "kubectl exec -it deployment/inventory-service -n production -- env GRPC_GO_LOG_VERBOSITY_LEVEL=99",
                "psql $DATABASE_URL -c 'CREATE INDEX CONCURRENTLY idx_sku_reservations_tenant ON sku_inventory_reservations(tenant_id, sku_id);'",
                "kubectl patch deployment order-orchestrator -p '{\"spec\":{\"template\":{\"spec\":{\"containers\":[{\"name\":\"app\",\"env\":[{\"name\":\"GRPC_TIMEOUT_MS\",\"value\":\"5000\"}]}]}}}}'"
            ],
            "preventative_measure": "Add composite indexes on (tenant_id, sku_id, status) and configure client-side gRPC hedge requests for high-tail-latency RPCs."
        }
    },
    {
        "category": "Linux Kernel & Memory Fragmentation",
        "service": "node-k8s-infra-04",
        "raw_log": """[CRITICAL] 2026-08-16T20:30:14.000Z kernel: [1948201.204] kswapd0: page allocation failure: order:3, mode:0x1084020(GFP_ATOMIC)
CPU: 7 PID: 49 Comm: kswapd0 Tainted: G           OE     5.15.0-1031-aws #35-Ubuntu
Mem-Info:
active_anon:3829102 inactive_anon:490192 active_file:128 inactive_file:41 unevictable:0
Node 0 DMA: 1*4kB (U) 0*8kB 0*16kB ... Node 0 Normal: 0*2048kB 0*4096kB
Network interface eth0: dropped 45,910 rx packets due to alloc_skb failure.""",
        "structured_response": {
            "incident_id": "INC-84926",
            "severity": "P1-CRITICAL",
            "root_cause": "High-order Linux kernel slab memory fragmentation preventing network socket buffer (sk_buff) allocation in GFP_ATOMIC context, causing host NIC packet drops.",
            "blast_radius": ["node-k8s-infra-04", "ingress-controller-pod", "dns-coredns"],
            "mitigation_actions": [
                "echo 1 > /proc/sys/vm/compact_memory",
                "echo 3 > /proc/sys/vm/drop_caches",
                "kubectl drain node-k8s-infra-04 --ignore-daemonsets --delete-emptydir-data --force"
            ],
            "preventative_measure": "Configure sysctl vm.min_free_kbytes=262144 and vm.extfrag_threshold=500 on all bare-metal/EC2 Kubernetes nodes."
        }
    },
    {
        "category": "BGP Anycast & DNS Resolution Cascade",
        "service": "coredns-internal-cluster",
        "raw_log": """[ERROR] 2026-08-16T21:12:08.514Z [plugin/errors] 2 kube-dns.kube-system.svc.cluster.local. A: read udp 10.96.0.10:53->172.20.0.2:53: i/o timeout
CoreDNS query volume: 65,000 QPS (nominal 4,000 QPS). Upstream DNS resolver 172.20.0.2 throttled UDP packet rate.
Over 45 microservices reporting java.net.UnknownHostException / ENOTFOUND.""",
        "structured_response": {
            "incident_id": "INC-84927",
            "severity": "P1-CRITICAL",
            "root_cause": "DNS thundering herd triggered by ndots:5 default search-domain lookups across newly deployed Node.js microservices making non-FQDN HTTP requests, overwhelming CoreDNS.",
            "blast_radius": ["kube-system-coredns", "entire-cluster-egress-networking"],
            "mitigation_actions": [
                "kubectl scale deployment coredns -n kube-system --replicas=10",
                "kubectl apply -f - <<EOF\napiVersion: v1\nkind: ConfigMap\nmetadata:\n  name: coredns-custom\n  namespace: kube-system\ndata:\n  autopath.override: |\n    autopath @kubernetes\nEOF",
                "kubectl rollout restart deployment coredns -n kube-system"
            ],
            "preventative_measure": "Enable autopath plugin in CoreDNS ConfigMap and append absolute dot to internal endpoints (e.g. redis.prod.svc.cluster.local.) in service manifests."
        }
    }
]

# Variations generator to scale dataset to robust fine-tuning volumes
SEVERITIES = ["P1-CRITICAL", "P2-HIGH", "P3-MEDIUM", "P4-LOW"]
MICROSERVICES = [
    "auth-service", "billing-service", "search-indexer", "recommendation-v2",
    "notification-engine", "identity-provider", "streaming-gateway", "warehouse-sync",
    "ledger-service", "asset-transcoder", "ml-feature-store", "edge-router"
]
DB_TYPES = ["postgres", "mysql", "redis", "mongodb", "cassandra", "dynamodb"]
OS_ERRORS = ["ECONNREFUSED", "ETIMEDOUT", "ENOMEM", "EPIPE", "EBUSY", "ENETUNREACH"]

def generate_synthetic_incident(index: int) -> Dict[str, Any]:
    base = random.choice(INCIDENT_TEMPLATES)
    svc = random.choice(MICROSERVICES)
    db = random.choice(DB_TYPES)
    err = random.choice(OS_ERRORS)
    
    incident_id = f"INC-{90000 + index}"
    
    # Context-rich raw log
    raw_log = f"""[ERROR] 2026-08-16T{random.randint(10,23):02d}:{random.randint(10,59):02d}:{random.randint(10,59):02d}.{random.randint(100,999)}Z [{svc}]
Failed to execute downstream request to {db}-primary.domain.internal:5432
Error Code: {err} - Transport connection terminated abnormally.
Thread dump indicates {random.randint(40, 200)} threads blocked on connection pool lock.
Host memory utilization: {random.randint(85, 99)}%, CPU Steal: {random.randint(15, 45)}%
Active trace: trace-{random.randint(100000, 999999)}-span-{random.randint(100, 999)}"""

    severity = random.choice(["P1-CRITICAL", "P2-HIGH"]) if err in ["ECONNREFUSED", "ENOMEM"] else random.choice(["P3-MEDIUM", "P4-LOW"])

    structured_response = {
        "incident_id": incident_id,
        "severity": severity,
        "root_cause": f"Exhaustion of connection pool worker threads in {svc} attempting to communicate with {db}-primary due to {err} socket teardown under high load.",
        "blast_radius": [svc, f"{db}-primary", "api-gateway"],
        "mitigation_actions": [
            f"kubectl rollout restart deployment/{svc} -n production",
            f"kubectl logs -l app={svc} --tail=100 -n production",
            f"netstat -tulnp | grep 5432"
        ],
        "preventative_measure": f"Configure aggressive socket keepalive timeouts and connection circuit breakers with fallback caching in {svc} client pool configuration."
    }

    return {
        "instruction": "Analyze the following production telemetry, stack trace, and error log. Provide an SRE diagnostic evaluation as a structured JSON object.",
        "input": raw_log,
        "output": json.dumps(structured_response, indent=2)
    }

def build_dataset(total_samples: int = 250, data_dir: str = "./data"):
    os.makedirs(data_dir, exist_ok=True)
    all_data = []

    # Include curated ground truth templates first
    for i, t in enumerate(INCIDENT_TEMPLATES):
        all_data.append({
            "instruction": "Analyze the following production telemetry, stack trace, and error log. Provide an SRE diagnostic evaluation as a structured JSON object.",
            "input": t["raw_log"],
            "output": json.dumps(t["structured_response"], indent=2)
        })

    # Scale with high-quality domain variations
    remaining = total_samples - len(all_data)
    for i in range(remaining):
        all_data.append(generate_synthetic_incident(i))

    random.seed(42)
    random.shuffle(all_data)

    n_train = int(0.80 * len(all_data))
    n_val = int(0.10 * len(all_data))
    
    train_split = all_data[:n_train]
    val_split = all_data[n_train:n_train + n_val]
    test_split = all_data[n_train + n_val:]

    def save_jsonl(path: str, data: List[Dict[str, Any]]):
        with open(path, "w", encoding="utf-8") as f:
            for item in data:
                f.write(json.dumps(item) + "\n")
        print(f"[Data Generator] Saved {len(data)} records to {path}")

    save_jsonl(os.path.join(data_dir, "train.jsonl"), train_split)
    save_jsonl(os.path.join(data_dir, "val.jsonl"), val_split)
    save_jsonl(os.path.join(data_dir, "test.jsonl"), test_split)

if __name__ == "__main__":
    build_dataset(total_samples=300, data_dir=os.path.dirname(os.path.abspath(__file__)))
