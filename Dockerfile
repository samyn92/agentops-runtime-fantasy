# Fantasy Agent Runtime — Go binary with Charm Fantasy SDK
# Two modes: daemon (HTTP server) and task (one-shot)

FROM golang:1.26 AS builder
ARG VERSION=dev
WORKDIR /app
COPY go.mod go.sum ./
RUN go mod download
COPY *.go ./
RUN CGO_ENABLED=0 GOOS=linux go build -ldflags="-s -w -X main.version=${VERSION}" -o agent-runtime .

FROM alpine:3.20
# Install minimal tools needed by built-in tools
RUN apk add --no-cache bash curl ripgrep git
COPY --from=builder /app/agent-runtime /app/agent-runtime

# Create data directories
RUN mkdir -p /data/sessions /data/repos /data/scratch \
    && chown -R 1000:1000 /data

USER 1000:1000

ENTRYPOINT ["/app/agent-runtime"]
CMD ["daemon"]
