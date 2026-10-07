import pytest

from helpers import strict_loads
from server import EXAMPLES

MARK = '<script id="page-data" type="application/json">'
ATTRIBUTION = ("Financial PhraseBank", "Malo et al.", "CC BY-NC-SA 3.0", "notlucasp/financial-news-headlines", "Yahoo Finance", "CC0")


def page_data(html):
    start = html.index(MARK) + len(MARK)
    return strict_loads(html[start:html.index("</script>", start)])


@pytest.mark.parametrize("path", ["/", "/model", "/market"])
def test_pages_render_with_attribution(client, path):
    out = client.get(path)
    assert out.status_code == 200
    html = out.get_data(as_text=True)
    for fragment in ATTRIBUTION:
        assert fragment in html
    page_data(html)


def test_console_controls(client):
    html = client.get("/").get_data(as_text=True)
    for fragment in ('id="text-input"', 'id="score"', 'id="status"', 'id="results"', "data-example=", "/api/sentiment", "87.8%", "88.0%"):
        assert fragment in html
    for example in EXAMPLES:
        assert example in html
    assert page_data(html)["limits"] == {"texts": 16, "chars": 2000}


def test_model_page(client):
    html = client.get("/model").get_data(as_text=True)
    for fragment in ("Test accuracy", "93.4%", "0.910", "19.0%", "Missed", "Met", 'id="slices"', "TF-IDF + Logistic Regression", "ProsusAI/finbert", "int8", "McNemar", "finbert-tone", "12.0 points"):
        assert fragment in html
    data = page_data(html)
    assert [s["name"] for s in data["slices"]] == ["agreement >= 66%", "agreement >= 75%", "agreement = 100%"]
    assert set(data["full"]) == {"finbert", "baseline", "prosus"}


def test_market_page(client):
    html = client.get("/market").get_data(as_text=True)
    for fragment in ("Directional accuracy", "56.3%", 'id="contrasts"', 'id="rolling"', 'id="tone"', "Pesaran-Timmermann", "Always up", "32,574", "2020H1", "net_tone_r5"):
        assert fragment in html
    data = page_data(html)
    assert set(data["models"]) == {"sentiment", "sentiment_classifier", "sentiment_returns", "returns_only", "previous_direction", "always_up"}
    assert data["benchmark"] == 0.563


def test_console_hero_shows_the_model_reading_of_the_first_example(client, classifier):
    html = client.get("/").get_data(as_text=True)
    reading = classifier.predict([EXAMPLES[0]])[0]
    hero = html[html.index('id="hero"'):html.index("</figure>")]
    assert EXAMPLES[0] in hero and reading["label"] in hero
    for label, value in reading["probabilities"].items():
        assert f"{label} {value * 100:.1f}%" in hero


def test_console_examples_use_the_forwarded_scheme(client):
    html = client.get("/", headers={"X-Forwarded-Proto": "https"}).get_data(as_text=True)
    assert "https://localhost/api/sentiment" in html and "http://localhost/api/sentiment" not in html


def test_market_page_names_the_eligible_sessions(client):
    html = client.get("/market").get_data(as_text=True)
    assert "every eligible session" in html and "complete headline window" not in html
