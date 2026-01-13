import os
from datetime import date
from infra.ingestion.adapters import OptionsSurfaceIngestionRequest, IvolatilityOptionsSurfaceAdapter
from infra.ingestion.ivolatility_client import IvolatilityClient

api_key = os.environ.get("IVOLATILITY_API_KEY")
if not api_key:
    raise SystemExit("Missing IVOLATILITY_API_KEY")
request = OptionsSurfaceIngestionRequest(
    symbols=("AAPL","ACN","ADBE"),
    start_date=date(2022,1,1),
    end_date=date(2022,12,31),
    vendor="ivolatility",
    options={},
    vendor_params={},
)
client = IvolatilityClient(api_key=api_key, cache_dir=None)
adapter = IvolatilityOptionsSurfaceAdapter(client)
params = adapter._build_params(request.symbols, request.start_date, request.end_date)
print("request params", params)
payload = client.fetch_async_dataset(adapter.endpoint, params)
print("payload type", type(payload))
if isinstance(payload, bytes):
    print("payload bytes len", len(payload))
    print(payload[:200])
else:
    print("payload repr", repr(payload)[:400])
    # attempt to coerce dataset directly
    rows = adapter._coerce_dataset(payload)
    print("coerced rows", len(rows))
