# SkyMind C++ çıkarım servisi (llama.cpp, yalnızca CPU). Bağlam: depo kökü.
#   podman build -f deploy/containers/Containerfile.cpp -t skymind-cpp .
#
# Durum: Podman 4.9.3 (rootless, WSL2 Ubuntu) ile derlendi (88 MB) ve compose.yaml içinde
# çalıştırıldı (deploy/README.md). Windows'ta çalışan yol hazır llama.dll'e bağlanır
# (cpp/README.md); burada llama.cpp Linux'ta kaynaktan, yalnızca CPU olarak derlenir.
FROM docker.io/library/debian:bookworm-slim AS build
RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential cmake ninja-build git ca-certificates \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /src
COPY cpp/ cpp/
# GGML_NATIVE=OFF: imaj derlendiği makinenin CPU'suna bağlanmasın. AVX2/FMA/F16C modern
# x86-64 için güvenli taban; çok eski CPU'larda AVX2=OFF verin.
ARG GGML_AVX2=ON
# JOBS: paralel derleme sayısı (geliştirme makinesi donmasın diye sınırlı; CI'da artırılabilir).
ARG JOBS=4
RUN cmake -S cpp -B build -G Ninja -DCMAKE_BUILD_TYPE=Release \
      -DSKYMIND_BUILD_SERVER=ON -DSKYMIND_BUILD_TESTS=OFF \
      -DGGML_NATIVE=OFF -DGGML_AVX2=${GGML_AVX2} -DGGML_FMA=${GGML_AVX2} -DGGML_F16C=${GGML_AVX2} \
    && cmake --build build --target skymind_server -j ${JOBS}

FROM docker.io/library/debian:bookworm-slim
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --system --uid 10001 --no-create-home skymind
COPY --from=build /src/build/skymind_server /usr/local/bin/skymind_server
COPY cpp/configs/system_short.txt /app/configs/system_short.txt
USER 10001
EXPOSE 8080
# Model imaja gömülmez (1 GB): /models altına bağlanır.
ENTRYPOINT ["skymind_server", "--host", "0.0.0.0", "--port", "8080", \
            "--system", "/app/configs/system_short.txt"]
CMD ["-m", "/models/skymind-sft-Q5_K_M.gguf"]
