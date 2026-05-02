# Fantasy Agent Runtime — Go binary with Charm Fantasy SDK
# Two modes: daemon (HTTP server) and task (one-shot)

FROM golang:1.26 AS builder
ARG TARGETOS
ARG TARGETARCH
ARG VERSION=dev
WORKDIR /app
COPY go.mod go.sum ./
RUN go mod download
COPY *.go ./
RUN CGO_ENABLED=0 GOOS=${TARGETOS:-linux} GOARCH=${TARGETARCH} go build -ldflags="-s -w -X main.version=${VERSION}" -o agent-runtime .

FROM alpine:3.20
# Install minimal tools needed by built-in tools
RUN apk add --no-cache bash curl ripgrep git
COPY --from=builder /app/agent-runtime /app/agent-runtime

# Create non-root user matching the operator's RestrictedRunAsUser
# (65532 is the distroless `nonroot` UID — used by every other operator-built image).
RUN addgroup -g 65532 -S nonroot \
    && adduser -u 65532 -S nonroot -G nonroot -H -h /home/nonroot \
    && mkdir -p /data/sessions /data/repos /data/scratch /home/nonroot \
    && chown -R 65532:65532 /data /home/nonroot

USER 65532:65532

ENTRYPOINT ["/app/agent-runtime"]
CMD ["daemon"]
