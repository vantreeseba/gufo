# syntax=docker/dockerfile:1

# Gufo built from this checkout, delivered on the upstream gufo-runtime image.
#
# gufo-runtime is assembled by Nix (gufo-org/toolboxes) and carries no
# compiler, package manager or ROCm development headers, so the engine cannot
# be compiled inside it. The build stage runs the flake's production package
# with the toolchain pinned in flake.lock, and the final stage adds that
# package to gufo-runtime in place of the bundled engine.
#
#   docker build -t gufo-qwen35moe .
#   docker build -t gufo-qwen35moe --build-arg BUILD_CORES=4 .   # less RAM

ARG GUFO_RUNTIME_TAG=latest
ARG NIX_IMAGE_TAG=latest

FROM ghcr.io/gufo-org/toolboxes/gufo-runtime:${GUFO_RUNTIME_TAG} AS runtime

FROM nixos/nix:${NIX_IMAGE_TAG} AS build
ENV NIX_CONFIG="experimental-features = nix-command flakes"
# 0 lets Nix use every core; each HIP kernel compile needs several GiB.
ARG BUILD_CORES=0
WORKDIR /src

# Fetch the pinned toolchain and ROCm libraries before the sources arrive, so
# source edits reuse this layer.
COPY flake.nix flake.lock version.txt ./
COPY .devops .devops
RUN nix build --no-link --cores "${BUILD_CORES}" .#default.inputDerivation

COPY . .
RUN git config --global --add safe.directory /src \
 && nix build --cores "${BUILD_CORES}" --out-link /gufo .#default

# Stage the store paths gufo-runtime lacks, plus the entry points that
# replace its bundled engine.
RUN --mount=type=bind,from=runtime,source=/nix/store,target=/runtime-store \
    mkdir -p /out/nix/store /out/bin \
 && for path in $(nix-store --query --requisites /gufo); do \
      [ -e "/runtime-store/$(basename "$path")" ] \
        || cp -a "$path" /out/nix/store/; \
    done \
 && engine="$(readlink -f /gufo)" \
 && ln -s "$engine/bin/gufo" /out/bin/gufo \
 && ln -s "$engine/bin/gufo-server" /out/bin/gufo-server \
 && ln -s gufo /out/bin/strix

FROM runtime
COPY --from=build /out/ /
