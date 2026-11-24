# SkyMind C++ çıkarım servisi

llama.cpp üzerinde çalışan, HTTP üzerinden Türkçe komutu JSON görev planına çeviren servis. Python
yığınının (`skymind/`) aynı modelini ve aynı şema kurallarını kullanır; model çıktısı **sunucu
tarafında doğrulanır**, geçersiz çıktı gizlenmez, `valid:false` ve hata yoluyla döner.

```
cpp/
  include/skymind/   validate (şema), prompt (ChatML), metrics (Prometheus), service, engine, llama_engine
  src/               aynı adlı gerçeklemeler + server_main (HTTP)
  tests/             GoogleTest; test_validate.cpp Pydantic ile eşlik testi (tests/data/cases.jsonl)
  configs/           system_short.txt (Python'daki SYSTEM_SHORT'tan üretilir)
```

## Uçlar

| | |
|---|---|
| `POST /v1/plan` | `{"command": "..."}` → `{plan, valid, type, validation_error, raw, usage, latency_ms}` |
| `GET /healthz` | canlılık |
| `GET /metrics` | Prometheus: istek sayısı/durumu, çıktı türü (geçersiz dahil), token, gecikme histogramı |

İstekler tek bağlamda sıraya girer (batch=1); her istekte KV önbelleği temizlenir.

## Derleme (Windows, MSVC + CMake + Ninja)

```bat
uv tool install cmake ninja
scripts\cpp.cmd cmake -S cpp -B build\tests -G Ninja -DSKYMIND_BUILD_SERVER=OFF   :: yalnız çekirdek + testler
scripts\cpp.cmd cmake --build build\tests && scripts\cpp.cmd build\tests\skymind_tests.exe
```

Sunucu için iki yol:

1. **Hazır llama.dll (hızlı, GPU dahil).** llama.cpp sürüm paketini (`b12655`, `llama-*-bin-win-cuda-*`
   + `cudart-*`) `third_party/llama-cuda` altına açın; tam kaynak ağacı başlıklar için `third_party/llama.cpp-src`
   (aynı etiket). `scripts\cpp.cmd python scripts/make_import_lib.py third_party/llama-cuda/llama.dll` ile
   `llama.lib` üretin, sonra
   `-DLLAMA_CPP_DIR=... -DSKYMIND_LLAMA_PREBUILT_DIR=third_party/llama-cuda -DSKYMIND_WITH_CUDA=ON`.
2. **Kaynaktan** (`-DLLAMA_CPP_DIR` verilmezse FetchContent ile indirir): CPU derlemesi; bu makinede ~1 saat
   sürdü ve Ninja `llama.lib` adımında takıldı (aşağıya bakın).

```bat
build\server\skymind_server.exe -m outputs\gguf\skymind-sft-Q5_K_M.gguf --port 8091 --ngl 99 --system cpp\configs\system_short.txt
```

## Bilinen durumlar

* Kaynaktan derlemede Ninja, `llama.lib` oluşturan `lib.exe` adımında defalarca (aynı noktada) takıldı:
  `cmd.exe` süreci `Suspended` kaldı. Aynı komut elle çalıştırılınca 3 sn'de bitti. Nedeni bulunmadı;
  bu yüzden hazır DLL yolu kullanıldı. Kaynaktan derleme yolu **bu makinede doğrulanmadı**.
* Linux/CI: çekirdek ve testler taşınabilir C++17, ancak yalnız Windows'ta derlenip çalıştırıldı;
  CI işi henüz çalıştırılmadı.
