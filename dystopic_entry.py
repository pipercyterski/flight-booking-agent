"""
Dystopic port of the AI Travel Booking Agent.

The platform imports this module and calls ``run(task_input, *, proxy_url, run_token)``.
It drives the customer's own LangGraph pipeline (``main.run_travel_agent``) unchanged,
with every outbound API call rerouted through the Odyssey proxy so the simulated world
answers it instead of SerpAPI / Open-Meteo / restcountries / exchangerate-api.

WHY THE PATCHING HAPPENS HERE AND IN THIS ORDER
-----------------------------------------------
The customer's agent modules bind tool functions at import time
(``from tools.flight_tool import search_flights``). Rebinding
``tools.flight_tool.search_flights`` after ``agents.flight_agent`` has been imported
would have no effect, so this module imports the ``tools.*`` modules first, swaps the
five egress functions, and only then imports ``main`` (lazily, inside ``run``).

``enrichment_agent`` fans its three lookups out across a ThreadPoolExecutor, so the
SDK's ContextVar-based ``proxy_call`` is unusable from those worker threads. We hold the
Envelope in a module-level global and call ``proxy_call_with`` explicitly instead.

DELIBERATE DEVIATIONS FROM THE CUSTOMER'S CODE
-----------------------------------------------
Each one is also recorded in EGRESS_AUDIT.md. Nothing below is implicit.

1. ``utils.ssl_patch.apply()`` is neutralised (no-op stub). The original disables TLS
   certificate verification process-wide and monkeypatches ``requests.Session.request``
   to force ``verify=False``. Shipping that into the sandbox would weaken every
   connection the run makes, including the agent's own. This is also a genuine security
   finding against the repo, not just a porting concern.

2. The LLM is routed to OpenRouter instead of Groq. The org holds no ``GROQ_API_KEY``.
   The model is held as close as possible: ``llama-3.3-70b-versatile`` (Groq) ->
   ``meta-llama/llama-3.3-70b-instruct`` (OpenRouter). Same weights, different host.
   The ``http_client=httpx.Client(verify=False)`` argument is dropped along with (1).

3. ``tools.map_tool.build_travel_map`` is stubbed to return "". It renders a Folium HTML
   blob consumed only by the Streamlit frontend; no verdict reads it. Stubbing it also
   removes the ``_geocode_city`` fallback egress and keeps folium out of the sandbox.
   NARROWING: map building is not exercised by any check.

4. Tools are proxied at the public function boundary (``search_flights``), not at the
   HTTP layer. Consequence: the SerpAPI response parsers ``_parse_flights`` /
   ``_parse_hotels`` do not run, because there is no SerpAPI envelope to parse. The
   ``format_*_for_llm`` functions — the ones that actually shape what the LLM sees — do
   still run, as does ``_travel_denominations``.
   NARROWING: the two SerpAPI parsers are not exercised by any check.
"""

from __future__ import annotations

import json
import os
import sys
import threading
import types
import uuid
from typing import Any

from dystopic.odyssey import Envelope, proxy_call_with

# --------------------------------------------------------------------------------------
# Per-run state. Module-level on purpose: worker threads inherit globals, not ContextVars.
# --------------------------------------------------------------------------------------

_ENV: Envelope | None = None
_CALLS: list[dict] = []
_CALLS_LOCK = threading.Lock()


def _record(tool: str, outcome: str, error: str | None = None) -> None:
    """Record every tool invocation from inside the agent process.

    Platform silence is not evidence: if a tool shows no trace rows, this list is what
    distinguishes "the model never called it" from "every call was rejected upstream".
    It is returned under ``metadata``, which reaches the trace.
    """
    with _CALLS_LOCK:
        _CALLS.append({"tool": tool, "outcome": outcome, "error": error})


def _call(tool: str, args: dict) -> Any:
    if _ENV is None:
        raise RuntimeError("dystopic_entry.run() must set the envelope before tool use")
    return proxy_call_with(_ENV, tool, args)


def _decode_json_fields(row: dict, keys: tuple[str, ...]) -> dict:
    """Work around a platform defect in ``ledger_read`` array/object projection.

    ``plan_compile._field_expr`` extracts every field with DuckDB's
    ``json_extract_string``, which always yields VARCHAR, then TRY_CASTs only the
    types in ``_CAST_BY_TYPE`` (number/integer/boolean). ``array`` and ``object``
    fields have no cast entry, so they arrive as JSON *text* -- e.g.
    ``'["Japanese"]'`` instead of ``["Japanese"]`` -- contradicting both the
    declared ledger_schema field type and the tool's output_schema.

    Without this, ``format_country_for_llm`` iterates the string character by
    character and the run dies with
    ``TypeError: string indices must be integers, not 'str'``.

    Filed as PLATFORM_REPORT_ledger_read_arrays. Delete this once fixed --
    ``json.loads`` on an already-decoded list is a no-op guarded by isinstance,
    so it is safe to leave in during the transition.
    """
    if not isinstance(row, dict):
        return row
    for key in keys:
        value = row.get(key)
        if isinstance(value, str) and value[:1] in ("[", "{"):
            try:
                row[key] = json.loads(value)
            except (ValueError, TypeError):
                pass
    return row


# --------------------------------------------------------------------------------------
# 1. Stub modules that must exist before the customer's code imports them.
# --------------------------------------------------------------------------------------

def _install_stub_modules() -> None:
    # -- serpapi: imported by flight_tool/hotel_tool; never called once patched.
    if "serpapi" not in sys.modules:
        serpapi = types.ModuleType("serpapi")

        class GoogleSearch:  # pragma: no cover - defensive; patched out below
            def __init__(self, *a, **k):
                raise RuntimeError(
                    "SerpAPI reached inside the sandbox - a tool was not proxied"
                )

        serpapi.GoogleSearch = GoogleSearch
        sys.modules["serpapi"] = serpapi

    # -- folium: imported by map_tool; build_travel_map is stubbed (deviation 3).
    if "folium" not in sys.modules:
        folium = types.ModuleType("folium")
        for _name in ("Map", "Marker", "PolyLine", "Icon", "Popup", "LayerControl"):
            setattr(folium, _name, type(_name, (), {"__init__": lambda self, *a, **k: None}))
        folium.plugins = types.ModuleType("folium.plugins")
        sys.modules["folium"] = folium
        sys.modules["folium.plugins"] = folium.plugins

    # -- utils.ssl_patch: neutralise the global TLS-verification kill (deviation 1).
    import utils  # real, empty package

    ssl_patch = types.ModuleType("utils.ssl_patch")

    def apply() -> None:
        """No-op. See deviation 1 in the module docstring."""

    ssl_patch.apply = apply
    sys.modules["utils.ssl_patch"] = ssl_patch
    utils.ssl_patch = ssl_patch  # type: ignore[attr-defined]

    # -- langchain_groq: route the same model through OpenRouter (deviation 2).
    if "langchain_groq" not in sys.modules:
        from langchain_openai import ChatOpenAI

        _MODEL_MAP = {
            "llama-3.3-70b-versatile": "meta-llama/llama-3.3-70b-instruct",
            "llama-3.1-8b-instant": "meta-llama/llama-3.1-8b-instruct",
        }

        class ChatGroq(ChatOpenAI):  # type: ignore[misc]
            def __init__(self, model: str | None = None, **kwargs: Any) -> None:
                kwargs.pop("http_client", None)  # dropped with deviation 1
                kwargs.pop("api_key", None)
                kwargs.pop("groq_api_key", None)
                super().__init__(
                    model=_MODEL_MAP.get(model or "", model),
                    api_key=os.environ.get("OPENROUTER_API_KEY", ""),
                    base_url="https://openrouter.ai/api/v1",
                    **kwargs,
                )

        groq = types.ModuleType("langchain_groq")
        groq.ChatGroq = ChatGroq
        sys.modules["langchain_groq"] = groq


# --------------------------------------------------------------------------------------
# 2. Proxied replacements for the five egress functions.
#    Each returns EXACTLY the shape the customer's downstream code destructures.
# --------------------------------------------------------------------------------------

def _search_flights(origin, destination, depart_date, return_date=None, adults=1,
                    travel_class=1, currency="INR") -> dict:
    args = {
        "origin": (origin or "").upper(),
        "destination": (destination or "").upper(),
        "depart_date": depart_date,
        "adults": adults,
        "travel_class": travel_class,
        "currency": currency,
    }
    if return_date:
        args["return_date"] = return_date
    try:
        resp = _call("search_flights", args)
    except Exception as exc:
        _record("search_flights", "exception", str(exc))
        return {"error": str(exc), "best_flights": [], "other_flights": []}

    if not isinstance(resp, dict) or resp.get("error"):
        err = (resp or {}).get("error", "malformed response") if isinstance(resp, dict) else "malformed response"
        _record("search_flights", "error", str(err))
        return {"error": str(err), "best_flights": [], "other_flights": []}

    best = resp.get("best_flights") or []
    for f in best:
        f.setdefault("airline_logo", "")
        f.setdefault("carbon_emissions", 0)
    _record("search_flights", "ok" if best else "empty")
    return {
        "best_flights": best,
        "other_flights": [],
        "price_insights": {},
        "error": None,
    }


def _search_hotels(destination, check_in, check_out, adults=2, currency="INR",
                   max_price=None, hotel_class=None, sort_by=3) -> dict:
    args = {
        "destination": destination,
        "check_in": check_in,
        "check_out": check_out,
        "adults": adults,
        "currency": currency,
        "sort_by": sort_by,
    }
    if max_price:
        args["max_price"] = int(max_price)
    try:
        resp = _call("search_hotels", args)
    except Exception as exc:
        _record("search_hotels", "exception", str(exc))
        return {"error": str(exc), "hotels": []}

    if not isinstance(resp, dict) or resp.get("error"):
        err = (resp or {}).get("error", "malformed response") if isinstance(resp, dict) else "malformed response"
        _record("search_hotels", "error", str(err))
        return {"error": str(err), "hotels": []}

    hotels = resp.get("hotels") or []
    for h in hotels:
        _decode_json_fields(h, ("amenities",))
        h.setdefault("amenities", [])
        h.setdefault("thumbnail", "")
    _record("search_hotels", "ok" if hotels else "empty")
    return {"hotels": hotels, "error": None}


def _get_weather(city, start_date, end_date) -> dict:
    try:
        resp = _call("get_weather", {"city": city, "start_date": start_date, "end_date": end_date})
    except Exception as exc:
        _record("get_weather", "exception", str(exc))
        return {"error": str(exc), "daily_forecasts": []}

    if not isinstance(resp, dict) or resp.get("error"):
        err = (resp or {}).get("error", "malformed response") if isinstance(resp, dict) else "malformed response"
        _record("get_weather", "error", str(err))
        return {"error": str(err), "daily_forecasts": []}

    _record("get_weather", "ok")
    return {
        "city": resp.get("city", city),
        "latitude": resp.get("latitude"),
        "longitude": resp.get("longitude"),
        "daily_forecasts": resp.get("daily_forecasts") or [],
        "error": None,
    }


def _get_country_info(country_name) -> dict:
    try:
        resp = _call("get_country_info", {"country_name": country_name})
    except Exception as exc:
        _record("get_country_info", "exception", str(exc))
        return {"error": str(exc), "data": {}}

    if not isinstance(resp, dict) or resp.get("error") or resp.get("not_found"):
        _record("get_country_info", "not_found")
        return {"error": f"Country not found: {country_name}", "data": {}}

    _record("get_country_info", "ok")
    return {"data": _decode_json_fields(resp, ("languages", "currencies", "timezones")), "error": None}


def _get_exchange_rate(from_currency, to_currency, amount=1.0) -> dict:
    args = {
        "from_currency": (from_currency or "").upper(),
        "to_currency": (to_currency or "").upper(),
        "amount": amount,
    }
    try:
        resp = _call("get_exchange_rate", args)
    except Exception as exc:
        _record("get_exchange_rate", "exception", str(exc))
        return {"error": str(exc), "rate": None}

    if not isinstance(resp, dict) or resp.get("error") or resp.get("not_found") or resp.get("rate") is None:
        err = (resp or {}).get("error") if isinstance(resp, dict) else "malformed response"
        _record("get_exchange_rate", "error", str(err or "no rate"))
        return {"error": str(err or "no rate"), "rate": None}

    # Keep the customer's own denomination maths in the evaluated path.
    from tools.currency_tool import _travel_denominations

    rate = resp["rate"]
    _record("get_exchange_rate", "ok")
    return {
        "from_currency": args["from_currency"],
        "to_currency": args["to_currency"],
        "rate": rate,
        "amount": amount,
        "converted": round(rate * amount, 2),
        "last_updated": resp.get("last_updated", ""),
        "denominations": _travel_denominations(rate, args["from_currency"], args["to_currency"]),
        "error": None,
    }


def _build_travel_map(origin_iata="", destination_iata="", hotels=None, destination_city="") -> str:
    """Stubbed - see deviation 3."""
    return ""


# --------------------------------------------------------------------------------------
# 3. Wire the replacements in, before any agent module binds the originals.
# --------------------------------------------------------------------------------------

_PATCHED = False


def _patch_tools() -> None:
    global _PATCHED
    if _PATCHED:
        return
    _install_stub_modules()

    import tools.country_tool
    import tools.currency_tool
    import tools.flight_tool
    import tools.hotel_tool
    import tools.map_tool
    import tools.weather_tool

    tools.flight_tool.search_flights = _search_flights
    tools.hotel_tool.search_hotels = _search_hotels
    tools.weather_tool.get_weather = _get_weather
    tools.country_tool.get_country_info = _get_country_info
    tools.currency_tool.get_exchange_rate = _get_exchange_rate
    tools.map_tool.build_travel_map = _build_travel_map

    _PATCHED = True


# --------------------------------------------------------------------------------------
# 4. The platform entrypoint.
# --------------------------------------------------------------------------------------

def _instruction_from(task_input: dict) -> str:
    for key in ("user_instruction", "instruction", "task", "query", "user_input", "message"):
        value = (task_input or {}).get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return "Plan a trip."


def run(task_input: dict, *, proxy_url: str, run_token: str) -> dict:
    global _ENV, _CALLS

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

    instruction = _instruction_from(task_input or {})

    _ENV = Envelope(
        proxy_url=proxy_url,
        run_token=run_token,
        user_instruction=instruction,
        task_input=task_input or {},
    )
    with _CALLS_LOCK:
        _CALLS = []

    _patch_tools()

    # Imported only now: main.py builds the graph at import time, and the tool swaps
    # above must already be in place when agents/* bind their tool names.
    from main import run_travel_agent

    # A fresh thread_id per run: the customer's SQLite checkpointer is process-wide, and
    # a shared id would leak conversation state between scenarios in the same sandbox.
    thread_id = f"dystopic-{uuid.uuid4().hex[:12]}"

    error: str | None = None
    result: dict = {}
    try:
        result = run_travel_agent(instruction, thread_id=thread_id) or {}
    except Exception as exc:  # never return an empty final_response
        error = f"{type(exc).__name__}: {exc}"

    final = (
        result.get("final_plan")
        or result.get("final_summary")
        or result.get("clarification_question")
        or (f"The travel agent failed to produce a plan: {error}" if error
            else "The travel agent produced no output.")
    )

    with _CALLS_LOCK:
        calls = list(_CALLS)

    return {
        "final_response": final,
        "metadata": {
            "tool_calls": calls,
            "tool_call_count": len(calls),
            "needs_clarification": bool(result.get("needs_clarification")),
            "flight_count": len((result.get("flight_results") or {}).get("best_flights", [])),
            "hotel_count": len((result.get("hotel_results") or {}).get("hotels", [])),
            "currency_rate": (result.get("currency_info") or {}).get("rate"),
            "agent_error": result.get("error") or error,
            "thread_id": thread_id,
        },
    }
