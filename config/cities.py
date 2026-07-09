"""
City configuration: NWS CLI stations, grid offices, and coordinates.
"""

CITIES = {
    "seattle": {
        "cli_site": "SEW", "cli_issuedby": "SEA",
        "nws_office": "SEW", "nws_grid": "124,67",
        "lat": 47.6062, "lon": -122.3321,
        "tz": "America/Los_Angeles",
    },
    "los_angeles": {
        "cli_site": "LOX", "cli_issuedby": "LAX",
        "nws_office": "LOX", "nws_grid": "149,47",
        "lat": 33.9425, "lon": -118.4081,
        "tz": "America/Los_Angeles",
    },
    "miami": {
        "cli_site": "MFL", "cli_issuedby": "MIA",
        "nws_office": "MFL", "nws_grid": "110,37",
        "lat": 25.7959, "lon": -80.2870,
        "tz": "America/New_York",
    },
    "new_york": {
        "cli_site": "OKX", "cli_issuedby": "NYC",
        "nws_office": "OKX", "nws_grid": "33,37",
        "lat": 40.6413, "lon": -73.7781,
        "tz": "America/New_York",
    },
    "minneapolis": {
        "cli_site": "MPX", "cli_issuedby": "MSP",
        "nws_office": "MPX", "nws_grid": "107,70",
        "lat": 44.8848, "lon": -93.2223,
        "tz": "America/Chicago",
    },
    "houston": {
        "cli_site": "HGX", "cli_issuedby": "HOU",
        "nws_office": "HGX", "nws_grid": "67,54",
        "lat": 29.9902, "lon": -95.3368,
        "tz": "America/Chicago",
    },
    "denver": {
        "cli_site": "BOU", "cli_issuedby": "DEN",
        "nws_office": "BOU", "nws_grid": "57,63",
        "lat": 39.8561, "lon": -104.6737,
        "tz": "America/Denver",
    },
    "boston": {
        "cli_site": "BOX", "cli_issuedby": "BOS",
        "nws_office": "BOX", "nws_grid": "69,83",
        "lat": 42.3601, "lon": -71.0589,
        "tz": "America/New_York",
    },
    "chicago": {
        "cli_site": "LOT", "cli_issuedby": "MDW",
        "nws_office": "LOT", "nws_grid": "76,73",
        "lat": 41.8827, "lon": -87.6233,
        "tz": "America/Chicago",
    },
    "dallas": {
        "cli_site": "FWD", "cli_issuedby": "DFW",
        "nws_office": "FWD", "nws_grid": "85,103",
        "lat": 32.8998, "lon": -97.0403,
        "tz": "America/Chicago",
    },
    "philadelphia": {
        "cli_site": "PHI", "cli_issuedby": "PHL",
        "nws_office": "PHI", "nws_grid": "49,87",
        "lat": 39.8721, "lon": -75.2411,
        "tz": "America/New_York",
    },
    "san_francisco": {
        "cli_site": "MTR", "cli_issuedby": "SFO",
        "nws_office": "MTR", "nws_grid": "92,83",
        "lat": 37.6213, "lon": -122.3790,
        "tz": "America/Los_Angeles",
    },
    "las_vegas": {
        "cli_site": "VEF", "cli_issuedby": "LAS",
        "nws_office": "VEF", "nws_grid": "116,91",
        "lat": 36.0840, "lon": -115.1537,
        "tz": "America/Los_Angeles",
    },
    "oklahoma_city": {
        "cli_site": "OUN", "cli_issuedby": "OKC",
        "nws_office": "OUN", "nws_grid": "114,91",
        "lat": 35.3931, "lon": -97.6007,
        "tz": "America/Chicago",
    },
    "austin": {
        "cli_site": "EWX", "cli_issuedby": "AUS",
        "nws_office": "EWX", "nws_grid": "155,84",
        "lat": 30.1975, "lon": -97.6664,
        "tz": "America/Chicago",
    },
    "san_antonio": {
        "cli_site": "EWX", "cli_issuedby": "SAT",
        "nws_office": "EWX", "nws_grid": "137,80",
        "lat": 29.5341, "lon": -98.4698,
        "tz": "America/Chicago",
    },
    "phoenix": {
        "cli_site": "PSR", "cli_issuedby": "PHX",
        "nws_office": "PSR", "nws_grid": "161,56",
        "lat": 33.4373, "lon": -112.0078,
        "tz": "America/Phoenix",
    },
    "new_orleans": {
        "cli_site": "LIX", "cli_issuedby": "MSY",
        "nws_office": "LIX", "nws_grid": "66,91",
        "lat": 29.9934, "lon": -90.2580,
        "tz": "America/Chicago",
    },
    "atlanta": {
        "cli_site": "FFC", "cli_issuedby": "ATL",
        "nws_office": "FFC", "nws_grid": "51,88",
        "lat": 33.6407, "lon": -84.4277,
        "tz": "America/New_York",
    },
    "washington_dc": {
        "cli_site": "LWX", "cli_issuedby": "DCA",
        "nws_office": "LWX", "nws_grid": "97,71",
        "lat": 38.8512, "lon": -77.0402,
        "tz": "America/New_York",
    },
}
