# Dağıtım: A/B ağ geçidi, izleme, Podman/K8s

```
istemci ──► gateway (FastAPI, :8000) ──► skymind-a  (C++ / llama.cpp, SFT)    ─┐
              │  ağırlıklı, deterministik │► skymind-b  (C++ / llama.cpp, DPO)    ├─► /metrics ─► Prometheus ─► Grafana
              └─ /ab/summary, /metrics    └───────────────────────────────────────┘
```

## Parçalar

| Dosya | Ne |
|---|---|
| `skymind/serve/gateway.py` | A/B ağ geçidi: aynı komut hep aynı varyanta gider (SHA-256), ağırlıklı bölme, varyant başına metrik, `/ab/summary` ile geçerlilik oranı için iki oranlı z-testi |
| `prometheus/` | `prometheus.yml` (compose, statik hedefler), `prometheus.k8s.yml` (Pod keşfi), `alerts.yml` (4 alarm) |
| `grafana/` | veri kaynağı + pano sağlama, `dashboards/skymind.json` (10 panel) |
| `containers/` | `Containerfile.cpp` (Linux'ta llama.cpp'yi kaynaktan derler), `Containerfile.gateway` |
| `compose.yaml` | Tek komutla tüm yığın |
| `k8s/`, `kustomization.yaml` | Namespace, PVC, iki arka uç, ağ geçidi + HPA, Prometheus, Grafana |

## Çalıştırma

```bash
# Podman (ya da Docker)
podman compose -f deploy/compose.yaml up --build     # gateway :8000, Prometheus :9090, Grafana :3000

# Kubernetes (kind / k3s)
podman build -f deploy/containers/Containerfile.cpp -t skymind-cpp .
podman build -f deploy/containers/Containerfile.gateway -t skymind-gateway .
kubectl apply -k deploy
kubectl -n skymind port-forward svc/gateway 8000
```

GGUF dosyaları imaja gömülmez. Compose'ta `outputs/gguf` bağlanır; K8s'te `skymind-models` PVC'sine
kopyalanır. A ve B modelleri `MODEL_A` / `MODEL_B` (compose) ya da `backends.yaml` argümanlarıyla seçilir.

A/B sonucu: `curl localhost:8000/ab/summary`. `significant_at_0.05` yalnızca **geçerlilik oranı**
farkı içindir; alan doğruluğu gibi metrikler için gold etiket gerekir ve canlı trafikte yoktur.

## Doğrulama durumu (dürüst not)

- **Test edildi:** ağ geçidi mantığı (atama dağılımı, tutarlılık, z-testi, 502 yolu, metrikler;
  `tests/test_gateway.py`). Pano sorguları ve alarmların kullandığı her metrik adı gerçekten servisler
  tarafından sunuluyor, ve C++ ile Python servisleri aynı metrikleri tanımlıyor
  (`tests/test_deploy.py`, bilerek bozulan ad ve seçici ile test edildi). K8s manifest'leri yapısal
  olarak kontrol edildi: etiket/seçici eşleşmesi, ConfigMap ve PVC referansları, güvenlik bağlamı.
- **Çalıştırılmadı:** Podman ve bir K8s kümesi bu makinede yok. Bu yüzden imajlar derlenmedi,
  `compose.yaml` ve manifest'ler bir kümeye uygulanmadı, Prometheus/Grafana ayağa kaldırılmadı.
  `kubectl kustomize`, `kubeconform` ve `promtool check` ile şema doğrulaması da henüz yapılmadı.
- **Linux derlemesi:** `Containerfile.cpp` llama.cpp'yi kaynaktan derler. Windows'ta bu yol
  Ninja takılması yüzünden doğrulanamamıştı; Linux'ta denenmedi.
