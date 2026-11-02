# Client libraries

Official clients exist for Python and Go. Both can enqueue jobs and run workers.

## Python

Install the client with `pip install tidewell`. A producer creates a `tidewell.Client` with the server URL and a token, then calls `client.enqueue("emails", {"to": "user@example.com"}, priority=7)`. The call returns the job id.

Workers register handlers with a decorator. A function decorated with `@worker.task("emails")` receives the payload as a dictionary, and `worker.run()` starts polling. The client supports both regular and `async def` handlers.

## Go

The Go module provides `tidewell.NewClient` for producers and `tidewell.NewWorker` for workers. Handlers are plain functions that take a context and the raw JSON payload and return an error.

## Batching

Both clients offer a batch enqueue call that sends up to 500 jobs in a single request. Batches are atomic: either every job is created or none is, which simplifies retry logic in producers.

## Testing

The Python client ships a `tidewell.testing.InlineClient` that runs handlers synchronously in the calling thread, so unit tests can exercise enqueue-and-handle flows without a server.
