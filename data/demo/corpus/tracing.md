# Distributed tracing

Tidewell supports OpenTelemetry so that a background job appears in the same trace as the request that created it.

## Context propagation

When a producer enqueues a job, the client library captures the current W3C `traceparent` and stores it with the job. When a worker later executes the job, it starts a span whose parent is that stored context. A slow checkout request and the confirmation email it triggered can therefore be viewed as one trace, even if the email ran minutes later on another machine.

## Exporting spans

Set `tracing.otlp_endpoint` to the address of an OTLP collector to export spans from the server. Workers use the standard OpenTelemetry environment variables of their language. The server samples 10 percent of traces by default, controlled by `tracing.sample_ratio`; a job whose producer trace was sampled is always traced so that traces are never broken halfway.

The server records spans for enqueue, dispatch and completion, while the worker library records a span around each handler execution with the attempt number as an attribute.
