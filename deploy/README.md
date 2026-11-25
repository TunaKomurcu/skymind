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
- **Podman'da çalıştırıldı (rootless Podman 4.9.3, WSL2 Ubuntu 24.04):** `Containerfile.cpp` (llama.cpp
  Linux'ta kaynaktan, CPU, imaj 88 MB) ve `Containerfile.gateway` (144 MB) derlendi; `compose.yaml`
  (`podman-compose` 1.6.0 ile) beş servisi ayağa kaldırdı. Gateway → C++ konteyneri (DPO GGUF, CPU) →
  şemadan geçen geçerli plan döndü; `/ab/summary` çalıştı. Prometheus üç hedefi (`skymind-a`, `skymind-b`,
  `gateway`) `up` gördü, 4 alarm kuralı yüklendi (`health: ok`), Grafana veri kaynağı ve pano sağlandı ve
  panodaki 12 sorgunun hepsi canlı Prometheus'ta hatasız çalıştı. Bu çalıştırmada bir pano hatası bulundu ve
  düzeltildi: hiç geçersiz çıktı yokken iki "geçersiz çıktı oranı" paneli boş kalıyordu (`or vector(0)`).
- **Linux C++ derlemesi:** CI'ın `cpp` işiyle aynı komutlar (gcc 13.3, CMake 3.28) WSL'de çalıştırıldı:
  15/15 GoogleTest geçti. GitHub Actions'ın kendisi çalıştırılmadı (repo uzak sunucuda değil).
- **Sınırlar:** tek istek denendi (CPU'da 38 token ~12 sn, düşük öncelik ve iki konteyner altında; bir
  hız ölçümü değildir). GPU'lu imaj yok; yalnızca CPU yolu. Pano panellerinin *görsel* doğruluğuna
  bakılmadı, yalnızca sorguların çalıştığı doğrulandı. Grafana'ya tarayıcıdan girilmedi.
- **Çalıştırılmadı:** Kubernetes kümesi yok; `k8s/` manifest'leri hiçbir kümeye uygulanmadı ve
  `kubectl kustomize` / `kubeconform` / `promtool check` ile şema doğrulaması yapılmadı (yalnızca yapısal
  pytest kontrolleri var). Prometheus ise konfigürasyonunu ve kurallarını canlı yükleyerek doğrulandı.
