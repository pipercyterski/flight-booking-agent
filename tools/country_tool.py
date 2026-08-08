import requests

REST_COUNTRIES_URL = "https://restcountries.com/v3.1"


def get_country_info(country_name: str) -> dict:
    """
    Fetch country information using REST Countries API (no API key needed).

    Args:
        country_name: country or city name e.g. "Japan", "France", "Tokyo"

    Returns:
        dict with country details or error
    """
    try:
        # try by full name first, fall back to partial search
        resp = requests.get(
            f"{REST_COUNTRIES_URL}/name/{country_name}",
            params={"fullText": "false"},
            timeout=10,
        )

        if resp.status_code != 200:
            return {"error": f"Country not found: {country_name}", "data": {}}

        results = resp.json()
        if not results:
            return {"error": f"No data for: {country_name}", "data": {}}

        country = results[0]
        return {
            "data":  _parse_country(country),
            "error": None,
        }

    except Exception as e:
        return {"error": str(e), "data": {}}


def _parse_country(country: dict) -> dict:
    currencies = country.get("currencies", {})
    currency_info = []
    for code, details in currencies.items():
        currency_info.append({
            "code":   code,
            "name":   details.get("name", ""),
            "symbol": details.get("symbol", ""),
        })

    languages = list(country.get("languages", {}).values())

    timezones = country.get("timezones", [])

    capital = country.get("capital", ["N/A"])
    capital = capital[0] if capital else "N/A"

    flags = country.get("flags", {})

    return {
        "name":          country.get("name", {}).get("common", "Unknown"),
        "official_name": country.get("name", {}).get("official", "Unknown"),
        "capital":       capital,
        "region":        country.get("region", ""),
        "subregion":     country.get("subregion", ""),
        "population":    f"{country.get('population', 0):,}",
        "area_km2":      f"{country.get('area', 0):,}",
        "languages":     languages,
        "currencies":    currency_info,
        "timezones":     timezones,
        "calling_code":  "+" + country.get("idd", {}).get("root", "").replace("+", "") +
                         (country.get("idd", {}).get("suffixes", [""])[0] if country.get("idd", {}).get("suffixes") else ""),
        "flag_emoji":    country.get("flag", ""),
        "flag_png":      flags.get("png", ""),
        "driving_side":  country.get("car", {}).get("side", "right"),
        "visa_required": _get_visa_note(country.get("name", {}).get("common", "")),
    }


def _get_visa_note(country_name: str) -> str:
    """
    Basic visa guidance for Indian passport holders.
    For production use, integrate a proper visa API.
    """
    visa_on_arrival = [
        "Thailand", "Indonesia", "Maldives", "Mauritius",
        "Nepal", "Bhutan", "Cambodia", "Laos", "Myanmar",
        "Macau", "Qatar", "Fiji",
    ]
    visa_free = [
        "Nepal", "Bhutan",
    ]
    e_visa = [
        "Turkey", "Sri Lanka", "Vietnam", "Egypt", "Kenya",
        "Ethiopia", "Uzbekistan", "Azerbaijan", "Armenia",
    ]
    visa_required = [
        "Japan", "China", "United States", "United Kingdom",
        "Australia", "Canada", "France", "Germany", "Italy",
        "Spain", "Switzerland", "New Zealand",
    ]

    if country_name in visa_free:
        return "Visa Free for Indian passport holders"
    if country_name in visa_on_arrival:
        return "Visa on Arrival available for Indian passport holders"
    if country_name in e_visa:
        return "e-Visa available for Indian passport holders"
    if country_name in visa_required:
        return "Visa required — apply in advance (Indian passport)"
    return "Check visa requirements at indianvisaonline.gov.in"


def format_country_for_llm(country_data: dict) -> str:
    """Converts country info into a readable string for the LLM."""
    if country_data.get("error"):
        return f"Country info failed: {country_data['error']}"

    d = country_data.get("data", {})
    if not d:
        return "No country data available."

    currencies = ", ".join(
        f"{c['name']} ({c['code']}, {c['symbol']})"
        for c in d.get("currencies", [])
    )
    languages = ", ".join(d.get("languages", []))
    timezones = ", ".join(d.get("timezones", []))

    return f"""=== COUNTRY INFO: {d.get('flag_emoji', '')} {d.get('name', '').upper()} ===
Official Name : {d.get('official_name')}
Capital       : {d.get('capital')}
Region        : {d.get('subregion')}, {d.get('region')}
Population    : {d.get('population')}
Area          : {d.get('area_km2')} km²
Language(s)   : {languages}
Currency      : {currencies}
Timezone(s)   : {timezones}
Calling Code  : {d.get('calling_code')}
Driving Side  : {d.get('driving_side').capitalize()}"""
