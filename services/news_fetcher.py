import httpx
import feedparser
import logging
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

FEEDS = [
    "https://techcrunch.com/feed/",
    "https://feeds.feedburner.com/oreilly/radar",
]

FALLBACK_ARTICLES = [
    {
        "title": "Distributed Systems and the Challenge of Consistency",
        "text": "Modern distributed systems face fundamental trade-offs between consistency, availability, and partition tolerance. Engineers must carefully evaluate these constraints when designing microservice architectures. Asynchronous communication patterns reduce coupling but introduce complexity around eventual consistency. Teams building high-throughput infrastructure must implement sophisticated monitoring to detect synchronization failures before they propagate.",
        "source": "TechCrunch",
        "url": "https://techcrunch.com",
    },
    {
        "title": "Machine Learning Infrastructure at Scale",
        "text": "Deploying machine learning models in production requires careful orchestration of computational resources. GPU clusters demand specialized scheduling algorithms to maximize utilization without starving latency-sensitive workloads. Organizations increasingly adopt containerized inference services to achieve reproducible deployments. Observability tooling must capture both system-level metrics and model-specific performance indicators.",
        "source": "Hacker News",
        "url": "https://news.ycombinator.com",
    },
    {
        "title": "API Design Principles for Developer Experience",
        "text": "Well-designed application programming interfaces dramatically reduce integration friction for downstream consumers. RESTful conventions provide familiarity but GraphQL offers superior flexibility for complex data relationships. Authentication mechanisms must balance security requirements against implementation simplicity. Comprehensive documentation with executable examples accelerates adoption and reduces support burden significantly.",
        "source": "TechCrunch",
        "url": "https://techcrunch.com",
    },
]


async def fetch_tech_articles(count: int = 3) -> list[dict]:
    articles = []

    for feed_url in FEEDS:
        if len(articles) >= count:
            break
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.get(feed_url)
                feed = feedparser.parse(resp.text)

            for entry in feed.entries[:3]:
                if len(articles) >= count:
                    break

                # Extract clean text from summary
                summary_html = entry.get("summary", entry.get("content", [{}])[0].get("value", ""))
                soup = BeautifulSoup(summary_html, "html.parser")
                text = soup.get_text(separator=" ").strip()

                # Keep to ~200 words for shadowing
                words = text.split()
                if len(words) > 60:
                    text = " ".join(words[:60]) + "."

                if len(text) < 40:
                    continue

                articles.append(
                    {
                        "title": entry.get("title", "Tech Article")[:100],
                        "text": text,
                        "source": feed.feed.get("title", "Tech News")[:30],
                        "url": entry.get("link", ""),
                    }
                )
        except Exception as e:
            logger.warning("Feed fetch error for %s: %s", feed_url, e)

    if not articles:
        return FALLBACK_ARTICLES[:count]

    return articles[:count]
