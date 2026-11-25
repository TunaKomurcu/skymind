"""deploy/ altındaki yapılandırmaların kendi içinde ve kodla tutarlılığı.

Bu testler kümeye ya da Prometheus'a ihtiyaç duymaz; yalnızca yapısal hataları (yanlış etiket,
eksik ConfigMap, kodda olmayan metrik adı) yakalar. Gerçek bir kümede çalışmanın yerini tutmaz.
"""

import json
import re
from pathlib import Path
from typing import Any

import httpx
import yaml
from fastapi.testclient import TestClient

from skymind.serve import gateway
from skymind.serve.app import create_app
from skymind.serve.engines import GenerationResult
from skymind.serve.service import PlannerService

ROOT = Path(__file__).resolve().parents[1]
DEPLOY = ROOT / "deploy"


def load_all(path: Path) -> list[dict[str, Any]]:
    return [d for d in yaml.safe_load_all(path.read_text(encoding="utf-8")) if d]


def k8s_docs() -> list[dict[str, Any]]:
    docs: list[dict[str, Any]] = []
    for f in (DEPLOY / "kustomization.yaml").parent.glob("k8s/*.yaml"):
        docs += load_all(f)
    return docs


def test_every_yaml_and_json_parses() -> None:
    for f in [*DEPLOY.rglob("*.yaml"), *DEPLOY.rglob("*.yml")]:
        assert load_all(f), f
    json.loads((DEPLOY / "grafana/dashboards/skymind.json").read_text(encoding="utf-8"))


def test_kustomization_files_exist() -> None:
    k = yaml.safe_load((DEPLOY / "kustomization.yaml").read_text(encoding="utf-8"))
    for r in k["resources"]:
        assert (DEPLOY / r).is_file(), r
    for gen in k["configMapGenerator"]:
        for entry in gen["files"]:
            assert (DEPLOY / entry.split("=", 1)[-1]).is_file(), entry


def test_deployments_selectors_match_templates_and_services_match_pods() -> None:
    docs = k8s_docs()
    deployments = [d for d in docs if d["kind"] == "Deployment"]
    assert len(deployments) >= 5
    for d in deployments:
        sel = d["spec"]["selector"]["matchLabels"]
        labels = d["spec"]["template"]["metadata"]["labels"]
        assert sel.items() <= labels.items(), d["metadata"]["name"]
    for s in (d for d in docs if d["kind"] == "Service"):
        sel = s["spec"]["selector"]
        owners = [
            d
            for d in deployments
            if sel.items() <= d["spec"]["template"]["metadata"]["labels"].items()
        ]
        assert len(owners) == 1, s["metadata"]["name"]
        ports = {
            p["name"]
            for c in owners[0]["spec"]["template"]["spec"]["containers"]
            for p in c.get("ports", [])
        }
        for p in s["spec"]["ports"]:
            assert p["targetPort"] in ports, (s["metadata"]["name"], p)


def test_pods_reference_existing_configmaps_and_claims_and_are_hardened() -> None:
    docs = k8s_docs()
    k = yaml.safe_load((DEPLOY / "kustomization.yaml").read_text(encoding="utf-8"))
    configmaps = {g["name"] for g in k["configMapGenerator"]}
    claims = {d["metadata"]["name"] for d in docs if d["kind"] == "PersistentVolumeClaim"}
    for d in (d for d in docs if d["kind"] == "Deployment"):
        pod = d["spec"]["template"]["spec"]
        assert pod["securityContext"]["runAsNonRoot"] is True
        for v in pod.get("volumes", []):
            if "configMap" in v:
                assert v["configMap"]["name"] in configmaps, v
            if "persistentVolumeClaim" in v:
                assert v["persistentVolumeClaim"]["claimName"] in claims, v
        mounts = {m["name"] for c in pod["containers"] for m in c.get("volumeMounts", [])}
        assert mounts <= {v["name"] for v in pod.get("volumes", [])}, d["metadata"]["name"]
        for c in pod["containers"]:
            sc = c["securityContext"]
            assert sc["allowPrivilegeEscalation"] is False and sc["capabilities"]["drop"] == ["ALL"]
            assert "memory" in c["resources"]["limits"]


def test_scraped_pods_are_annotated_and_ports_match() -> None:
    for d in (d for d in k8s_docs() if d["kind"] == "Deployment"):
        meta = d["spec"]["template"]["metadata"]
        ann = meta.get("annotations", {})
        if ann.get("prometheus.io/scrape") == "true":
            ports = [
                p["containerPort"]
                for c in d["spec"]["template"]["spec"]["containers"]
                for p in c["ports"]
            ]
            assert int(ann["prometheus.io/port"]) in ports, d["metadata"]["name"]


def test_backend_a_and_b_run_different_model_files() -> None:
    args = {
        d["metadata"]["name"]: d["spec"]["template"]["spec"]["containers"][0]["args"]
        for d in k8s_docs()
        if d["kind"] == "Deployment" and d["metadata"]["name"] in ("skymind-a", "skymind-b")
    }
    assert args["skymind-a"] != args["skymind-b"]


def test_compose_paths_and_services_are_consistent() -> None:
    c = yaml.safe_load((DEPLOY / "compose.yaml").read_text(encoding="utf-8"))
    for name, svc in c["services"].items():
        build = svc.get("build")
        if build:
            assert (DEPLOY / build["context"] / build["dockerfile"]).resolve().is_file(), name
        for dep in svc.get("depends_on", []):
            assert dep in c["services"]
        for v in svc.get("volumes", []):
            host = v.split(":")[0]
            assert (DEPLOY / host).exists(), (name, host)
    static = yaml.safe_load((DEPLOY / "prometheus/prometheus.yml").read_text(encoding="utf-8"))
    targets = {
        t.split(":")[0]
        for j in static["scrape_configs"]
        for sc in j["static_configs"]
        for t in sc["targets"]
    }
    assert targets <= set(c["services"])


# --- panolar ve alarmlar gerçek metrik adlarını kullanıyor mu ----------------------------------


class _Fake:
    name = "fake"

    def generate(self, system: str, command: str) -> GenerationResult:
        return GenerationResult(text="{}")


def exposed_metric_names() -> set[str]:
    backend = TestClient(create_app(PlannerService(_Fake(), "s")))
    http = httpx.AsyncClient(transport=httpx.MockTransport(lambda r: httpx.Response(200, json={})))
    gw = TestClient(gateway.create_app([gateway.Variant("a", "http://a", 1)], client=http))
    names: set[str] = set()
    for text in (backend.get("/metrics").text, gw.get("/metrics").text):
        for m in re.findall(r"^# TYPE (\S+) (\w+)", text, re.M):
            base, kind = m
            names.add(base)
            if kind == "counter":
                names.add(base + "_total")  # prometheus_client TYPE satırı ekleri atar
            if kind == "histogram":
                names |= {base + "_bucket", base + "_sum", base + "_count"}
    return names


def used_metric_names() -> set[str]:
    texts = [(DEPLOY / "prometheus/alerts.yml").read_text(encoding="utf-8")]
    dash = json.loads((DEPLOY / "grafana/dashboards/skymind.json").read_text(encoding="utf-8"))
    texts += [t["expr"] for p in dash["panels"] for t in p["targets"]]
    return {m for t in texts for m in re.findall(r"\bskymind_[a-z_]+", t)}


def test_dashboard_and_alerts_only_use_metrics_the_services_expose() -> None:
    used, exposed = used_metric_names(), exposed_metric_names()
    assert used, "panoda hiç metrik yok"
    assert used <= exposed, sorted(used - exposed)


def test_cpp_and_python_services_describe_the_same_metrics() -> None:
    cpp = (ROOT / "cpp/src/service.cpp").read_text(encoding="utf-8")
    cpp_names = set(re.findall(r'describe\("(skymind_[a-z_]+)"', cpp))
    py = TestClient(create_app(PlannerService(_Fake(), "s"))).get("/metrics").text
    py_names = set(re.findall(r"^# TYPE (skymind_\S+) ", py, re.M))
    # prometheus_client her sayaç/histograma ek "_created" zaman damgası serisi ekler (C++'ta yok)
    py_names = {n.removesuffix("_total") for n in py_names if not n.endswith("_created")}
    cpp_names = {n.removesuffix("_total") for n in cpp_names}
    # model_info C++ tarafında server_main.cpp'de tanımlanır
    assert cpp_names | {"skymind_model_info"} == py_names
