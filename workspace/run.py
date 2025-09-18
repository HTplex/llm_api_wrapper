
import json
from typing import List, Dict, Tuple
from pprint import pprint
from types import new_class
json_path = "/Users/htplex/Developer/ht/llm_api_wrapper/workspace/abelai_app_dev.2025-08-20T23-17-41.json"
with open(json_path, 'r') as fp:
    data = json.load(fp)





prompt = """
You are a professional crypto market analyst trained as a structured language model. Your role is to process cryptocurrency‑related news and generate structured, machine‑readable insights. Your task is to enrich input JSON with structured information about the event, including standardized entities, neutral descriptions, related tickers, tags, potential impacts, and directional sentiment analysis.

Objective

Given the input JSON, perform the following tasks:
	1.	Extract and standardize key entities mentioned in the news.
	•	Follow the standardization guidelines below.
	•	Avoid overly specific entities occurring rarely; group them under more general entities when appropriate.
	2.	Assign 5 – 10 relevant tags to each entity that indicate its broader category or relationship.
	3.	Generate a concise, neutral, context‑independent sentence describing each standardized entity.
	•	10 – 25 words.
	•	Factual and objective (no opinion or sentiment).
	•	Suitable for use as an embedding in clustering.
	4.	Identify token tickers directly or indirectly related to the event.
	5.	Analyze potential impacts on each standardized entity and ticker.
	6.	Determine directional sentiment relationships:
	•	Include both a source entity (“from”) and a target entity or ticker (“to”).
	•	If the sentiment is a general opinion from the article, use "News" as the source entity.
	•	Provide
	•	"sentiment": "-2 |-1 | 0 | 1 | 2"
	•	"confidence_score": 0.0 – 1.0 based on clarity and context

Entity Standardization Guidelines
	•	U.S. Securities and Exchange Commission, SEC, U.S. Commodities and Futures Trading Commission, etc. → “U.S. Government Agency” with tags ["U.S.", "Regulatory"]
	•	El Salvador’s GDP → “El Salvador GDP” with tags ["El Salvador", "GDP", "Small Country Economy"]
	•	Major exchanges (e.g., Binance, Coinbase) → keep official names, tag as ["Exchange"]
	•	Known cryptocurrencies → use the official project name (e.g., “Ethereum” for ETH).
	•	Generic mentions (e.g., “European banks”) → standardize to a clear collective term (e.g., “European Banks”) and tag appropriately.

Tag Standardization and Scope
	•	Use 5–10 tags per entity.
	•	Avoid tags that are too broad (“Global Finance”) or too narrow (“El Salvador GDP Growth Rate Q2 2025”).
	•	Prefer existing categories: "Regulatory", "Banks", "Exchange", "DeFi", "NFTs", "GDP", "Economy", etc.
	•	If a prospective tag does not map neatly to an existing category and seems overly unique, generalize it or omit it.


Output Format

Return pure JSON with exactly the following structure (no additional keys, text, or commentary):

{
  "entities": [
    {
      "name": "Standardized Entity Name",
      "description": "Neutral one‑sentence description of the entity (10–25 words).",
      "tags": ["tag1", "tag2", "tagN", ...]
    }
  ],
  "sentiment": [
    {
      "from": "Standardized Source Entity (or \"News\")",
      "to": "Standardized Target Entity or Ticker Name",
      "sentiment": "-2|-1|0|1|2",
      "confidence_score": 0.0
    }
  ]
}

Ensure the key order, casing, and data types match the schema exactly.

"""

prompt = """
You are a professional events analyst trained as a structured language model. Your role is to process domain‑agnostic news and generate structured, machine‑readable insights. Your task is to enrich input JSON with structured information about the event, including standardized entities, neutral descriptions, related identifiers (if any), tags, potential impacts, directional sentiment analysis, and graph structures (knowledge graph + causal graph) derived from the extracted entities.

Objective

Given the input JSON, perform the following tasks:
  1. Extract and standardize key entities mentioned in the news.
     • Follow the standardization guidelines below.
     • Avoid overly specific, low‑frequency entities; group them under more general entities when appropriate.
  2. Assign 5–10 relevant tags to each entity that indicate its broader category or relationship.
  3. Generate a concise, neutral, context‑independent sentence describing each standardized entity.
     • 10–25 words.
     • Factual and objective (no opinion or sentiment).
     • Suitable for use as an embedding in clustering.
  4. Identify relevant identifiers or codes mentioned (if any).
     • Examples: stock/crypto tickers, protocol symbols, ISIN, CUSIP, SEDOL, LEI, CAS, DOI, ISBN, ISO codes, case numbers, licenses.
     • Preserve official formatting; for tickers use UPPERCASE.
  5. Analyze potential impacts on each standardized entity and identifier.
     • Provide direction (positive|negative|neutral), severity (‑2…+2), and confidence (0.0–1.0) with a brief rationale.
  6. Determine directional sentiment relationships:
     • Include both a source entity (“from”) and a target entity or identifier (“to”).
     • If the sentiment is a general opinion from the article, use "News" as the source entity.
     • Provide:
       – "sentiment": "positive" | "negative" | "neutral"
       – "severity": integer scale from ‑2 (strongly negative) to +2 (strongly positive)
       – "confidence_score": 0.0–1.0 based on clarity and context
  7. Build a knowledge graph from standardized entities and identified identifiers.
     • Nodes = all standardized entity names + all identifiers (by id or canonical name).
     • Edges = typed relationships extracted from the article. Use free‑form, concise relation labels (no fixed vocabulary).
     • Each edge must include: from, to, relation, short evidence (≤20 words), confidence_score (0.0–1.0).
     • Strive for label consistency across documents (e.g., short present‑tense verb phrases or snake_case), but do not constrain creativity.
     • All edge endpoints must appear in "nodes".
  8. Build a causal graph capturing explicit or well‑supported implied cause‑effect links.
     • Each edge models a directional causal claim grounded in the article, not mere correlation.
     • Fields: cause, effect, effect_type, magnitude, time_horizon, confidence_score, evidence, assumptions.
       – effect_type: "increase" | "decrease" | "enable" | "constrain" | "uncertain"
       – magnitude: integer ‑2…+2 (net expected effect strength and direction)
       – time_horizon: "immediate" (≤7d) | "short_term" (1–4w) | "medium_term" (1–6m) | "long_term" (>6m)
       – evidence: ≤25 words; assumptions: ≤25 words (omit if none).
     • Causes/effects should reference standardized entity names or identifiers when applicable.

Entity Standardization Guidelines
  • Government regulators (e.g., SEC, CFTC, OFAC, competition authorities) → “<Country/Region> Government Agency” with tags ["Regulatory", "<Country/Region>"].
  • Courts → “<Country/Region> Court” with tags ["Legal", "<Country/Region>"].
  • Corporations, nonprofits, universities, and international organizations → keep official names; tag by sector and role.
  • Known projects/products/standards (e.g., Ethereum, ISO 27001) → use official names; add domain tags (e.g., "Layer1", "Standard").
  • Geographic collectives (e.g., “European banks”) → standardize to clear collective terms (e.g., “European Banks”) and tag appropriately.
  • People → use full name; capture role or affiliation in tags (e.g., "CEO", "Researcher").
  • When domain‑specific (finance, crypto, health, science, etc.), include identifiers in "identifiers" if present; otherwise focus on entities.

Tag Standardization and Scope
  • Use 5–10 tags per entity.
  • Avoid tags that are too broad (“Global Affairs”) or too narrow (“EU AI Act Article 52(1)(b) Sub‑clause”).
  • Prefer general categories: "Regulatory", "Legal", "Policy", "Technology", "Security", "Finance", "Market", "Economy", "Public Health", "Environment", "Geopolitics", "Energy", "Supply Chain", "Labor", "Education", "Science", "Infrastructure", "Culture", "Media", "AI", "Open Source", "Standards", "Governance".

Conventions
  • Entity names in Title Case; identifiers preserve official formatting (e.g., BTC, US0378331005, 67‑56‑1, ISO 3166‑1:US).
  • Relation labels in the knowledge graph are free‑form; prefer concise, action‑oriented phrases for interoperability.
  • All references in "sentiment", "knowledge_graph", and "causal_graph" must match names/ids defined in "entities"/"identifiers".
  • If an item is not present in the article, return an empty array for that section.
  • Be fully neutral and factual; avoid advice or speculation beyond evidence‑based causal inferences.

Output Format

Return pure JSON with exactly the following structure (no additional keys, text, or commentary):

{
  "entities": [
    {
      "name": "Standardized Entity Name",
      "description": "Neutral one‑sentence description of the entity (10–25 words).",
      "tags": ["tag1", "tag2", "tagN"]
    }
  ],
  "identifiers": [
    {
      "id": "Identifier String (e.g., BTC, US0378331005, 67-56-1, ISO 3166-1:US)",
      "type": "ticker|protocol_symbol|isin|cusip|sedol|lei|cas|doi|isbn|iso_code|license|case_number|other",
      "name": "Canonical Name or Title"
    }
  ],
  "impacts": [
    {
      "on": "Standardized Entity or Identifier",
      "type": "regulatory|legal|policy|technology|security|market|finance|economy|liquidity|adoption|public_health|environment|geopolitics|energy|supply_chain|labor|education|science|infrastructure|culture|other",
      "direction": "positive|negative|neutral",
      "severity": -2,
      "confidence_score": 0.0,
      "rationale": "One succinct sentence explaining the impact assessment."
    }
  ],
  "sentiment": [
    {
      "from": "Standardized Source Entity (or \"News\")",
      "to": "Standardized Target Entity or Identifier",
      "sentiment": "positive|negative|neutral",
      "severity": -2,
      "confidence_score": 0.0
    }
  ],
  "knowledge_graph": {
    "nodes": [
      "Standardized Entity or Identifier"
    ],
    "edges": [
      {
        "from": "Source Entity or Identifier",
        "to": "Target Entity or Identifier",
        "relation": "Free‑form relation label",
        "evidence": "Quoted or paraphrased span (≤20 words).",
        "confidence_score": 0.0
      }
    ]
  },
  "causal_graph": {
    "edges": [
      {
        "cause": "Cause Entity or Identifier",
        "effect": "Effect Entity or Identifier",
        "effect_type": "increase|decrease|enable|constrain|uncertain",
        "magnitude": -2,
        "time_horizon": "immediate|short_term|medium_term|long_term",
        "confidence_score": 0.0,
        "evidence": "Quoted or paraphrased span (≤25 words).",
        "assumptions": "Optional brief assumption(s) if inference extends beyond explicit text (≤25 words)."
      }
    ]
  }
}

"""
prompt = """

UNIVERSAL NEWS → STRUCTURED JSON (Entities + Sentiment + Impacts + Explanations)

GOAL
Given only the input news (headline, excerpt, or full article), produce machine‑readable JSON that captures standardized entities plus directional sentiment and impacts. Use consistent canonical naming so the same real‑world entity is represented identically across different news items. No topic restrictions and no fixed tag taxonomy.

INPUT
- Raw news text (any market‑relevant domain: crypto, macro, tech, policy, equities, FX, commodities, energy, AI, geopolitics, etc.).

OUTPUT (STRICT TOP‑LEVEL KEYS ONLY)
Return pure JSON with exactly these three top‑level keys in this order and nothing else. Inside lists, you may include the fields shown (including the explanation fields).

{
  "entities": [
    {
      "name": "Standardized Entity Name",
      "description": "Neutral one-sentence description of the entity (10–25 words).",
      "tags": ["tag1", "tag2", "tagN", ...]
    }
  ],
  "sentiment": [
    {
      "from": "Standardized Source Entity (or \"News\")",
      "to": "Standardized Target Entity or Ticker Name",
      "sentiment": -2,
      "confidence_score": 0.0,
      "explanation": "One concise sentence explaining the cause→effect path behind this sentiment score."
    }
  ],
  "impacts": [
    {
      "to": "Standardized Target Entity or Ticker Name",
      "impact": -2,
      "confidence_score": 0.0,
      "explanation": "One concise sentence summarizing the net effect on the target, independent of specific sources."
    }
  ]
}

PRINCIPLES FOR CONSISTENCY
1) Canonical Names
   • Use identical casing, spacing, and punctuation across all outputs.
   • Prefer official names; expand acronyms on first use with the abbreviation: “U.S. Securities and Exchange Commission (SEC)”.
   • For assets/equities/ETFs, prefer “Official Name (TICKER)” when widely used: “Bitcoin (BTC)”, “Coinbase (COIN)”, “SPDR S&P 500 ETF (SPY)”.
   • For macro metrics, use explicit measurable forms: “Inflation (CPI, U.S.)”, “U.S. Recession Probability”, “Unemployment Rate (U.S.)”.

2) Entities
   • Extract only materially relevant actors/objects/metrics.
   • Avoid over‑processing; use a collective term only when the article is generic (e.g., “Spot Bitcoin ETFs (U.S.)”).
   • Each entity needs:
     – name: canonical and deduplicated
     – description: one neutral sentence (10–25 words), factual, context‑independent
     – tags: open vocabulary; concise, widely understood terms (roles, sectors, functions, technology, geography)

3) Sentiment Edges (from → to)
   • Use for directional relationships caused or implied by the news.
   • from: the causal actor (regulator, company, protocol, policymaker). If unspecified or article‑level inference, use "News".
   • to: a concrete target whose value/health is impacted (asset/protocol/company/ETF/sector/metric).
   • sentiment: integer in {−2, −1, 0, 1, 2}.
     – +2 strong positive (direct, material, near‑term)
     – +1 modest/conditional positive
     –  0 neutral/mixed/unclear
     – −1 modest/conditional negative
     – −2 strong negative (direct, material, near‑term)
   • confidence_score: 0.0–1.0 reflecting clarity, specificity, and evidence.
   • explanation: one concise sentence describing the cause→effect logic.

4) Impacts (source‑agnostic)
   • Summarize the net effect on each materially impacted target, regardless of source.
   • impact: same integer scale {−2..+2}; confidence_score mirrors overall certainty for that target.
   • explanation: one concise sentence summarizing the net effect and why.

5) Sign Convention (unambiguous)
   • Perspective: diversified market‑investor view of the target’s value/health.
   • Increase in desirable states (adoption, liquidity, approval odds, reliability, security, earnings power) → positive.
   • Increase in adverse states (recession probability, hacks, outages, crackdown risk, default risk, inflation surprise, funding stress) → negative.
   • Decrease in adverse states → positive; decrease in desirable states → negative.
   • For abstract negatives, use measurable metrics (e.g., “U.S. Recession Probability” instead of “Recession”).
     – If news implies recession risk ↑: negative (−1/−2).
     – If news implies recession risk ↓: positive (+1/+2).

6) Scope & Brevity
   • Include only entities and edges that are materially relevant.
   • Avoid duplication: single entity entry per unique name; multiple edges allowed when distinct relationships exist.
   • Keep explanations crisp (ideally 12–24 words).

PROCESS (QUIET REASONING STEPS)
1) Parse the news and shortlist entities, assets, and metrics.
2) Canonicalize names; expand acronyms once; attach tickers in parentheses where widely used.
3) Write neutral 10–25 word descriptions and assign concise open‑vocabulary tags.
4) Add sentiment edges for causal links with sign, confidence, and one‑sentence explanation.
5) Add impacts summarizing net effect per target with sign, confidence, and explanation.
6) Output only the JSON object with the three required top‑level keys.

CANONICALIZATION EXAMPLES (GUIDANCE)
• “Fed” → “Federal Reserve”
• “SEC” → “U.S. Securities and Exchange Commission (SEC)”
• “EU” → “European Union (EU)”
• “BTC” → “Bitcoin (BTC)”
• “ETH” → “Ethereum (ETH)”
• “USDC” → “USD Coin (USDC)”
• “Coinbase Global” / “COIN” → “Coinbase (COIN)”
• For unnamed ETF suites, a clear collective is acceptable: “Spot Bitcoin ETFs (U.S.)”.

EDGE CASES
• Mixed signals: if positives and negatives offset for a target, use 0 and explain the balance.
• Speculation: reduce confidence_score and state the conditionality in the explanation.
• Multi‑region macro: use distinct metric entities per region if impacts are separate (e.g., “Inflation (CPI, U.S.)”, “Inflation (CPI, Eurozone)”).



MINI EXAMPLES (WITH EXPLANATIONS)

Example A — “Visa expands USDC settlement on Solana”
{
  "entities": [
    {
      "name": "Visa",
      "description": "A global payments network enabling consumer and merchant transactions across cards and digital channels at worldwide scale.",
      "tags": ["Payments", "Network", "Merchants", "Settlement", "Fintech", "Global"]
    },
    {
      "name": "USD Coin (USDC)",
      "description": "A fiat-backed dollar stablecoin issued by Circle, designed to maintain one-to-one parity with the U.S. dollar.",
      "tags": ["Stablecoin", "Payments", "Dollar", "Issuer", "Settlement", "Crypto"]
    },
    {
      "name": "Solana (SOL)",
      "description": "A high-throughput Layer 1 blockchain emphasizing fast finality and low transaction costs using a proof-of-history-based design.",
      "tags": ["Layer 1", "Throughput", "Scaling", "Ecosystem", "Payments", "Blockchain"]
    }
  ],
  "sentiment": [
    {
      "from": "Visa",
      "to": "USD Coin (USDC)",
      "sentiment": 1,
      "confidence_score": 0.75,
      "explanation": "Network adoption by a major payments brand increases USDC’s utility and potential transaction volumes."
    },
    {
      "from": "Visa",
      "to": "Solana (SOL)",
      "sentiment": 1,
      "confidence_score": 0.7,
      "explanation": "Settlement support can drive consistent throughput and enterprise visibility for Solana’s ecosystem."
    }
  ],
  "impacts": [
    {
      "to": "USD Coin (USDC)",
      "impact": 1,
      "confidence_score": 0.75,
      "explanation": "Broader settlement integration should lift usage and credibility for dollar-pegged transfers."
    },
    {
      "to": "Solana (SOL)",
      "impact": 1,
      "confidence_score": 0.7,
      "explanation": "Real-world payment flows likely improve adoption and sustained activity on the network."
    }
  ]
}

Example B — “CPI hotter than expected in the U.S.”
{
  "entities": [
    {
      "name": "Inflation (CPI, U.S.)",
      "description": "The Consumer Price Index measuring price changes across a representative basket of goods and services in the United States.",
      "tags": ["Inflation", "Prices", "Macroeconomy", "Indicator", "U.S."]
    },
    {
      "name": "Bitcoin (BTC)",
      "description": "A decentralized cryptocurrency with fixed supply, secured by proof-of-work miners and a global peer-to-peer network.",
      "tags": ["Cryptoasset", "Layer 1", "Store of Value", "Mining", "Blockchain"]
    },
    {
      "name": "Federal Reserve",
      "description": "The central bank of the United States responsible for monetary policy, interest rates, and financial conditions.",
      "tags": ["Central Bank", "Monetary Policy", "Rates", "U.S.", "Liquidity"]
    }
  ],
  "sentiment": [
    {
      "from": "News",
      "to": "Inflation (CPI, U.S.)",
      "sentiment": -1,
      "confidence_score": 0.9,
      "explanation": "A hotter print signals stronger inflation pressure, worsening the near-term inflation outlook."
    },
    {
      "from": "News",
      "to": "Bitcoin (BTC)",
      "sentiment": -1,
      "confidence_score": 0.7,
      "explanation": "Sticky inflation tends to keep policy tighter, weighing on risk appetite for crypto assets."
    }
  ],
  "impacts": [
    {
      "to": "Inflation (CPI, U.S.)",
      "impact": -1,
      "confidence_score": 0.9,
      "explanation": "Surprise to the upside deteriorates the inflation profile relative to expectations."
    },
    {
      "to": "Bitcoin (BTC)",
      "impact": -1,
      "confidence_score": 0.7,
      "explanation": "Tighter financial conditions reduce flows into speculative and high-beta assets."
    }
  ]
}

Example C — “SEC approves spot Ether ETFs; trading next week”
{
  "entities": [
    {
      "name": "U.S. Securities and Exchange Commission (SEC)",
      "description": "The United States federal agency regulating securities markets, enforcing disclosure rules, and overseeing investment products and exchanges for investor protection.",
      "tags": ["Regulator", "Securities", "Compliance", "Enforcement", "U.S.", "Policy", "Markets"]
    },
    {
      "name": "Spot Ether ETFs (U.S.)",
      "description": "U.S.-listed funds holding ether directly, providing regulated exchange access for traditional investors seeking exposure.",
      "tags": ["ETF", "U.S.", "Exchange", "Asset Management", "Flows", "Regulated", "Ethereum"]
    },
    {
      "name": "Ethereum (ETH)",
      "description": "A smart contract blockchain enabling decentralized applications, token standards, and staking-based consensus.",
      "tags": ["Layer 1", "Smart Contracts", "DeFi", "Staking", "Blockchain", "Ecosystem"]
    }
  ],
  "sentiment": [
    {
      "from": "U.S. Securities and Exchange Commission (SEC)",
      "to": "Spot Ether ETFs (U.S.)",
      "sentiment": 2,
      "confidence_score": 0.9,
      "explanation": "Approval unlocks regulated access and likely catalyzes new investor inflows."
    },
    {
      "from": "U.S. Securities and Exchange Commission (SEC)",
      "to": "Ethereum (ETH)",
      "sentiment": 1,
      "confidence_score": 0.75,
      "explanation": "Easier access should improve adoption and capital availability for the Ethereum ecosystem."
    }
  ],
  "impacts": [
    {
      "to": "Spot Ether ETFs (U.S.)",
      "impact": 2,
      "confidence_score": 0.9,
      "explanation": "Listing enables immediate participation through brokerages, boosting potential demand."
    },
    {
      "to": "Ethereum (ETH)",
      "impact": 1,
      "confidence_score": 0.75,
      "explanation": "New channels for exposure can raise liquidity and mainstream relevance over time."
    }
  ]
}

Example D — “Cross‑chain bridge exploit drains $120M”
{
  "entities": [
    {
      "name": "NovaBridge Protocol",
      "description": "A hypothetical cross-chain bridge enabling asset transfers between multiple blockchains using smart contracts and validator relays.",
      "tags": ["Bridge", "Cross-Chain", "DeFi", "Smart Contracts", "Security", "Interoperability"]
    },
    {
      "name": "Ethereum (ETH)",
      "description": "A decentralized platform for programmable smart contracts and tokens, supporting a broad ecosystem of applications.",
      "tags": ["Layer 1", "Smart Contracts", "DeFi", "Ecosystem", "Blockchain"]
    },
    {
      "name": "BNB Chain (BNB)",
      "description": "A smart contract blockchain associated with the Binance ecosystem, supporting EVM-compatible applications and tokens.",
      "tags": ["Layer 1", "EVM", "Ecosystem", "DeFi", "Blockchain"]
    }
  ],
  "sentiment": [
    {
      "from": "News",
      "to": "NovaBridge Protocol",
      "sentiment": -2,
      "confidence_score": 0.9,
      "explanation": "A large exploit directly harms trust, liquidity, and user retention for the bridge."
    },
    {
      "from": "News",
      "to": "Ethereum (ETH)",
      "sentiment": -1,
      "confidence_score": 0.5,
      "explanation": "Contagion risk and asset movements may briefly elevate security concerns on connected chains."
    },
    {
      "from": "News",
      "to": "BNB Chain (BNB)",
      "sentiment": -1,
      "confidence_score": 0.5,
      "explanation": "Cross-chain fund routing raises scrutiny and perceived operational risks for the ecosystem."
    }
  ],
  "impacts": [
    {
      "to": "NovaBridge Protocol",
      "impact": -2,
      "confidence_score": 0.9,
      "explanation": "Security breach reduces credibility and likely depresses future volumes."
    },
    {
      "to": "Ethereum (ETH)",
      "impact": -1,
      "confidence_score": 0.5,
      "explanation": "Short-term reputational spillovers and monitoring overhead can weigh on activity."
    },
    {
      "to": "BNB Chain (BNB)",
      "impact": -1,
      "confidence_score": 0.5,
      "explanation": "Association with exploited flows may deter near-term users and liquidity providers."
    }
  ]
}

Example E — “Ethereum activates Dencun upgrade; rollup data costs fall”
{
  "entities": [
    {
      "name": "Ethereum (ETH)",
      "description": "A smart contract blockchain enabling decentralized applications, token standards, and staking-based consensus.",
      "tags": ["Layer 1", "Smart Contracts", "DeFi", "Staking", "Blockchain", "Ecosystem"]
    },
    {
      "name": "Ethereum Dencun Upgrade",
      "description": "A protocol upgrade introducing data availability improvements to reduce rollup costs and enhance scaling.",
      "tags": ["Upgrade", "Scaling", "Data Availability", "Fees", "Rollups", "Protocol"]
    },
    {
      "name": "Ethereum Layer-2 Rollups",
      "description": "Scaling networks executing transactions off-chain and posting proofs on Ethereum to achieve lower fees and faster confirmations.",
      "tags": ["Layer 2", "Rollups", "Scaling", "Fees", "Throughput", "Security"]
    }
  ],
  "sentiment": [
    {
      "from": "Ethereum Dencun Upgrade",
      "to": "Ethereum Layer-2 Rollups",
      "sentiment": 2,
      "confidence_score": 0.9,
      "explanation": "Lower data costs directly improve rollup economics and user affordability."
    },
    {
      "from": "Ethereum Dencun Upgrade",
      "to": "Ethereum (ETH)",
      "sentiment": 1,
      "confidence_score": 0.7,
      "explanation": "Enhanced scaling strengthens Ethereum’s competitiveness and developer appeal."
    }
  ],
  "impacts": [
    {
      "to": "Ethereum Layer-2 Rollups",
      "impact": 2,
      "confidence_score": 0.9,
      "explanation": "Cheaper posting boosts throughput and adoption across L2 ecosystems."
    },
    {
      "to": "Ethereum (ETH)",
      "impact": 1,
      "confidence_score": 0.7,
      "explanation": "Improved user experience and costs support broader ecosystem growth."
    }
  ]
}

Example F — “NYDFS grants Circle a limited purpose trust charter”
{
  "entities": [
    {
      "name": "New York State Department of Financial Services (NYDFS)",
      "description": "New York’s financial regulator overseeing banks, insurers, and virtual currency businesses through charters, licensing, and supervision.",
      "tags": ["Regulator", "Licensing", "Compliance", "New York", "U.S.", "Financial Supervision"]
    },
    {
      "name": "Circle",
      "description": "A financial technology company issuing USD Coin and providing blockchain-based payments and treasury services.",
      "tags": ["Fintech", "Stablecoin Issuer", "Payments", "Treasury", "Blockchain", "U.S."]
    },
    {
      "name": "USD Coin (USDC)",
      "description": "A fiat-backed dollar stablecoin intended to maintain one-to-one parity with the U.S. dollar for payments and settlement.",
      "tags": ["Stablecoin", "Dollar", "Payments", "Settlement", "Issuer", "Blockchain"]
    }
  ],
  "sentiment": [
    {
      "from": "New York State Department of Financial Services (NYDFS)",
      "to": "Circle",
      "sentiment": 1,
      "confidence_score": 0.8,
      "explanation": "Charter expands regulated operating scope and signals supervisory comfort with controls."
    },
    {
      "from": "New York State Department of Financial Services (NYDFS)",
      "to": "USD Coin (USDC)",
      "sentiment": 1,
      "confidence_score": 0.8,
      "explanation": "Greater regulatory clarity can enhance institutional adoption and trust in USDC."
    }
  ],
  "impacts": [
    {
      "to": "Circle",
      "impact": 1,
      "confidence_score": 0.8,
      "explanation": "Regulatory approval improves market access and credibility with enterprise clients."
    },
    {
      "to": "USD Coin (USDC)",
      "impact": 1,
      "confidence_score": 0.8,
      "explanation": "Expanded permissions may increase stablecoin usage in compliant U.S. markets."
    }
  ]
}

Example G — “OPEC+ extends oil production cuts through year‑end”
{
  "entities": [
    {
      "name": "OPEC+",
      "description": "An alliance of oil-producing countries coordinating output policies to influence crude supply, prices, and market stability.",
      "tags": ["Energy", "Oil", "Production", "Commodities", "Supply", "Policy"]
    },
    {
      "name": "Energy Producers (Global)",
      "description": "Companies engaged in exploration and production of oil and gas across international markets.",
      "tags": ["Energy", "Oil", "Gas", "Producers", "Commodities", "Upstream"]
    },
    {
      "name": "Airlines (U.S.)",
      "description": "Commercial passenger airlines in the United States, sensitive to fuel costs and demand cycles.",
      "tags": ["Airlines", "Transportation", "U.S.", "Fuel Costs", "Cyclical"]
    },
    {
      "name": "Inflation (CPI, U.S.)",
      "description": "A price index tracking consumer inflation in the United States.",
      "tags": ["Inflation", "Prices", "Indicator", "Macroeconomy", "U.S."]
    }
  ],
  "sentiment": [
    {
      "from": "OPEC+",
      "to": "Energy Producers (Global)",
      "sentiment": 1,
      "confidence_score": 0.7,
      "explanation": "Supply restraint supports higher realized prices and producer revenues."
    },
    {
      "from": "OPEC+",
      "to": "Airlines (U.S.)",
      "sentiment": -1,
      "confidence_score": 0.7,
      "explanation": "Elevated fuel costs pressure margins and capacity planning."
    },
    {
      "from": "OPEC+",
      "to": "Inflation (CPI, U.S.)",
      "sentiment": -1,
      "confidence_score": 0.6,
      "explanation": "Higher energy prices can widen inflation persistence risks."
    }
  ],
  "impacts": [
    {
      "to": "Energy Producers (Global)",
      "impact": 1,
      "confidence_score": 0.7,
      "explanation": "Prolonged cuts likely keep crude prices firmer, aiding cash flows."
    },
    {
      "to": "Airlines (U.S.)",
      "impact": -1,
      "confidence_score": 0.7,
      "explanation": "Fuel expense headwinds diminish profitability and pricing flexibility."
    },
    {
      "to": "Inflation (CPI, U.S.)",
      "impact": -1,
      "confidence_score": 0.6,
      "explanation": "Energy pass-through increases upside risks to near-term CPI prints."
    }
  ]
}

Example H — “U.S. tightens export controls on advanced AI chips to China”
{
  "entities": [
    {
      "name": "U.S. Government",
      "description": "The federal executive authority of the United States implementing policies, regulations, and enforcement actions domestically and abroad.",
      "tags": ["Government", "Policy", "U.S.", "Regulation", "Enforcement", "Geopolitics"]
    },
    {
      "name": "China",
      "description": "A sovereign nation with significant manufacturing and technology markets shaping global supply chains and policy responses.",
      "tags": ["Country", "Asia", "Manufacturing", "Technology", "Trade", "Geopolitics"]
    },
    {
      "name": "NVIDIA (NVDA)",
      "description": "A semiconductor company designing GPUs and AI accelerators for data centers and high-performance computing.",
      "tags": ["Semiconductors", "AI", "GPUs", "Data Centers", "Hardware", "Technology"]
    },
    {
      "name": "China AI Sector",
      "description": "Chinese firms and institutions developing AI hardware, software, and applications across industry and government.",
      "tags": ["AI", "Technology", "China", "Semiconductors", "Industry", "Policy"]
    }
  ],
  "sentiment": [
    {
      "from": "U.S. Government",
      "to": "NVIDIA (NVDA)",
      "sentiment": -1,
      "confidence_score": 0.6,
      "explanation": "Tighter controls constrain shipments to a key market, pressuring near-term demand."
    },
    {
      "from": "U.S. Government",
      "to": "China AI Sector",
      "sentiment": -1,
      "confidence_score": 0.8,
      "explanation": "Access limits to high-end chips impede model training and deployment timelines."
    }
  ],
  "impacts": [
    {
      "to": "NVIDIA (NVDA)",
      "impact": -1,
      "confidence_score": 0.6,
      "explanation": "Revenue headwinds likely from reduced China data center sales."
    },
    {
      "to": "China AI Sector",
      "impact": -1,
      "confidence_score": 0.8,
      "explanation": "Hardware bottlenecks slow AI capability growth and ecosystem scaling."
    }
  ]
}

Example I — “Bitcoin halving reduces block subsidy to 3.125 BTC”
{
  "entities": [
    {
      "name": "Bitcoin (BTC)",
      "description": "A decentralized cryptocurrency with a fixed issuance schedule, secured by proof-of-work miners and a distributed peer-to-peer network.",
      "tags": ["Cryptoasset", "Layer 1", "Mining", "Store of Value", "Blockchain", "Scarcity"]
    },
    {
      "name": "Bitcoin Halving (BTC)",
      "description": "A programmed event halving miner rewards, reducing new supply issuance and altering miner economics.",
      "tags": ["Protocol Event", "Supply", "Mining", "Issuance", "Economics"]
    },
    {
      "name": "Bitcoin Miners",
      "description": "Operators of hashing infrastructure validating Bitcoin transactions and earning rewards and fees while managing costs.",
      "tags": ["Mining", "Hashrate", "Infrastructure", "Energy", "Costs"]
    }
  ],
  "sentiment": [
    {
      "from": "Bitcoin Halving (BTC)",
      "to": "Bitcoin (BTC)",
      "sentiment": 1,
      "confidence_score": 0.6,
      "explanation": "Reduced issuance increases scarcity, potentially supporting price over time."
    },
    {
      "from": "Bitcoin Halving (BTC)",
      "to": "Bitcoin Miners",
      "sentiment": -1,
      "confidence_score": 0.7,
      "explanation": "Lower rewards compress margins unless offset by higher fees or price appreciation."
    }
  ],
  "impacts": [
    {
      "to": "Bitcoin (BTC)",
      "impact": 1,
      "confidence_score": 0.6,
      "explanation": "Supply reduction is structurally supportive for valuation dynamics."
    },
    {
      "to": "Bitcoin Miners",
      "impact": -1,
      "confidence_score": 0.7,
      "explanation": "Revenue declines increase shutdown risk for higher-cost operators."
    }
  ]
}

Example J — “U.S. spot Bitcoin ETFs log large net outflows this week”
{
  "entities": [
    {
      "name": "Spot Bitcoin ETFs (U.S.)",
      "description": "U.S.-listed exchange-traded funds holding bitcoin directly, providing brokerage access for regulated exposure.",
      "tags": ["ETF", "Bitcoin", "U.S.", "Asset Management", "Flows", "Exchange"]
    },
    {
      "name": "Bitcoin (BTC)",
      "description": "A decentralized cryptocurrency with fixed supply, secured by proof-of-work miners and a global peer-to-peer network.",
      "tags": ["Cryptoasset", "Layer 1", "Mining", "Store of Value", "Blockchain"]
    }
  ],
  "sentiment": [
    {
      "from": "News",
      "to": "Spot Bitcoin ETFs (U.S.)",
      "sentiment": -1,
      "confidence_score": 0.8,
      "explanation": "Outflows indicate weaker investor demand and net selling pressure within ETF vehicles."
    },
    {
      "from": "News",
      "to": "Bitcoin (BTC)",
      "sentiment": -1,
      "confidence_score": 0.6,
      "explanation": "ETF redemptions can translate into spot selling and softer near-term price support."
    }
  ],
  "impacts": [
    {
      "to": "Spot Bitcoin ETFs (U.S.)",
      "impact": -1,
      "confidence_score": 0.8,
      "explanation": "Lower fund assets reduce fee revenue and secondary market liquidity."
    },
    {
      "to": "Bitcoin (BTC)",
      "impact": -1,
      "confidence_score": 0.6,
      "explanation": "Reduced ETF demand may dampen aggregate buy-side flows into bitcoin."
    }
  ]
}



"""



news_prompts = [
    prompt + "\n\n" + "title: {}\ncontent: {}".format(d["title"], d["text"])
    for d in data
]
print(news_prompts[0])




from glob import glob
from os.path import join
import json
from pprint import pprint
from tqdm import tqdm
import sys
import os
from os.path import expanduser
sys.path.append(os.path.abspath('..'))
from llmw.wrapper_v2 import LLMW



# init llm runner
# model = "o3"
# model = "gpt-5"
# model = "qwen3-30b-a3b"
# model = "qwen3-235b-a22b"
# model = "qwen3-32b"
# model = "qwen3-coder"
# model = "deepseek-v3.1"

models = [
    # [openai_api_key,"o3"],
    [openai_api_key,"gpt-5"], 
    # [openai_api_key,"gpt-5-mini"],
    # [openai_api_key,"gpt-5-nano"],
    # [openrouter_api_key,"qwen3-30b-a3b"],
    # [openrouter_api_key,"qwen3-235b-a22b"],
    # [openrouter_api_key,"qwen3-32b"],
    # [openrouter_api_key,"qwen3-coder"],
    # [openrouter_api_key,"deepseek-v3.1"]
]


save_folder = "../tmp/prompt_tuning_v6/"
os.makedirs(save_folder, exist_ok=True)

for api_key, model in models:

  llmw = LLMW(api_key, model=model,max_concurrency=1000)

  results = llmw.batch_call(news_prompts)
  new_results = []


  for idx, news in enumerate(data):
      news = "title: {}\ncontent: {}".format(news["title"], news["text"])
      result = results[idx]
      save_path = os.path.join(save_folder, model + ".txt")
      with open(save_path, "a") as f:
          f.write(str(idx) + "\n")
          f.write(news + "\n")
          f.write(result + "\n")
          f.write("\n" + "="*100 + "\n")
    

    
    




            


    # pprint(sources_count)

