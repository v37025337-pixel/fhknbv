
def perceive(path):
    name = str(path)
    if name.endswith("dns.jpg"):
        return {
            "kind": "dns_table",
            "rows": [
                {
                    "provider": "Example",
                    "primary_dns": "1.1.1.1",
                    "secondary_dns": "1.0.0.1",
                }
            ],
        }
    return {"kind": "text", "text": "generic"}

def fetch(url):
    return "read-only remote document"

PROVIDERS = [
    {
        "name": "test-vision",
        "capability": "vision.perceive",
        "callable": perceive,
        "read_only": True,
        "priority": 100,
    },
    {
        "name": "test-web",
        "capability": "web.fetch.readonly",
        "callable": fetch,
        "read_only": True,
        "priority": 100,
    },
]
