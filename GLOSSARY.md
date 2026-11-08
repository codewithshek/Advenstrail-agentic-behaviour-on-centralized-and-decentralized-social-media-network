# Canonical Domain Glossary: Adversarial Agentic AI in Social Networks

This glossary provides formal definitions for research and engineering terminology used across the AgentInTheShell detection system, establishing unambiguous semantic boundaries between centralized platforms and decentralized/federated social networks.

---

### Centralized Network
A social communication architecture where a single administrative entity operates the infrastructure, maintains a unified global namespace (e.g., `@handle`), stores all graph state centrally, and controls information distribution via proprietary, centralized ranking and recommendation algorithms (e.g., X/Twitter, Instagram, Reddit).

### Decentralized Federated Network
A modular social network composed of independent, cooperating server instances ("pods" or "nodes") that communicate using open protocols (e.g., ActivityPub, Bluesky AT Protocol, Agent-to-Agent / A2A mesh). Identity is partitioned across federated domains (`@handle@instance.domain` or cryptographic DIDs), and message relay relies on inter-server consensus rather than a single corporate authority.

### Home Instance
The specific autonomous server or host domain that holds an account's cryptographic keypairs, profile metadata, and primary outbox in a federated architecture. While an account interacts globally across the federation, its canonical identity and baseline trust score originate on its home instance.

### Cross-Instance Federation Bridge
A communication channel, relay edge, or direct interaction linking accounts belonging to distinct home instances. In legitimate operation, federation bridges propagate public threads across server boundaries. In adversarial scenarios, they serve as the primary vector for injecting malicious traffic from rogue servers into benign communities.

### Algorithmic Trend Hijack
An adversarial coordination strategy optimized against centralized recommendation engines. A synchronized swarm of automated accounts deploys high-velocity repost/retweet rings, rapid co-hashtag bursts, and reply-bombing on high-degree influencer hubs to artificially inflate engagement metrics and force target topics into centralized algorithmic trending feeds.

### Cross-Instance Sybil Infiltration
An adversarial attack pattern unique to federated networks where an attacker spins up one or more rogue instances populated with automated Sybil identities. These accounts establish follower links and cross-instance interaction bridges with established, high-reputation instances, poisoning remote timelines while evading single-instance rate limits.

### A2A Discovery Spoofing
An exploit targeting autonomous Agent-to-Agent (A2A) discovery registries and Model Context Protocol (MCP) tool endpoints. Malicious agents publish falsified capability manifests, poisoned tool descriptions, or counterfeit routing metadata to induce victim autonomous agents into delegating sensitive computational tasks or relaying untrusted instructions.

### Defederation Evasion
Techniques employed by adversarial clusters to avoid platform-level blocklists ("defederation" or domain siloing) enacted by community moderators. Tactics include instance-hopping, rotating disposable subdomains, relaying payloads through intermediate neutral gateway servers, and obfuscating coordination payloads to blend into ordinary federated federation relays.

