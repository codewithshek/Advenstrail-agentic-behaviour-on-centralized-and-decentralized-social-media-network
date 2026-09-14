"""Validated API contracts for text, network, and dashboard analysis."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

ThreatLabel = Literal["human_benign", "adversarial"]
SubjectLabel = Literal["human", "simple_spambot", "coordinated_ai_agent_swarm"]
TriageBand = Literal["low", "elevated", "high", "critical"]


class TriggerSpan(BaseModel):
    """Exact half-open character range responsible for a lexical alert."""

    start: int = Field(ge=0)
    end: int = Field(ge=0)
    text: str
    trigger: str

    @model_validator(mode="after")
    def valid_range(self) -> TriggerSpan:
        if self.end <= self.start:
            raise ValueError("trigger span end must be greater than start")
        return self


class PayloadAlert(BaseModel):
    """A specific sentence carrying injection or jailbreak phrasing."""

    sentence: str
    sentence_start: int = Field(default=0, ge=0)
    sentence_end: int = Field(default=0, ge=0)
    matched_phrases: list[str]
    trigger_spans: list[TriggerSpan] = Field(default_factory=list)
    severity: float = Field(ge=0.0, le=1.0)
    post_index: int | None = None
    user_id: str | None = None


class TextAnalyzeRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    text: str = Field(min_length=1, max_length=100_000)
    source: str | None = Field(default=None, max_length=200)


class TextAnalysisResponse(BaseModel):
    analysis_id: int
    score: float = Field(ge=0.0, le=1.0)
    label: ThreatLabel
    threshold: float = Field(ge=0.0, le=1.0)
    model: str
    artifact_mode: str
    artifact_reliable: bool = False
    ai_generation_probability: float = Field(ge=0.0, le=1.0)
    risk_score: float = Field(ge=0.0, le=1.0)
    lexical_injection_score: float = Field(ge=0.0, le=1.0)
    payload_alerts: list[PayloadAlert] = Field(default_factory=list)
    llm_style_phrases: list[str] = Field(default_factory=list)
    warning: str | None = None


class ThreadAnalyzeRequest(BaseModel):
    """A pasted post thread; posts may be supplied split or as one blob."""

    model_config = ConfigDict(str_strip_whitespace=True)

    text: str | None = Field(default=None, max_length=200_000)
    posts: list[str] = Field(default_factory=list, max_length=500)
    source: str | None = Field(default=None, max_length=200)

    @model_validator(mode="after")
    def require_content(self) -> ThreadAnalyzeRequest:
        if not self.posts and not (self.text or "").strip():
            raise ValueError("supply either text or posts")
        return self


class ThreadPostResult(BaseModel):
    post_index: int
    text: str
    score: float = Field(ge=0.0, le=1.0)
    label: ThreatLabel
    payload_alerts: list[PayloadAlert] = Field(default_factory=list)


class ThreadReport(BaseModel):
    model: str
    artifact_mode: str
    artifact_reliable: bool = False
    threshold: float = Field(ge=0.0, le=1.0)
    posts_analyzed: int
    aggregate_score: float = Field(ge=0.0, le=1.0)
    peak_score: float = Field(ge=0.0, le=1.0)
    # The score the report leans on: the transformer when it is production
    # grade, otherwise the lexical injection detector.
    risk_score: float = Field(ge=0.0, le=1.0)
    label: ThreatLabel
    flagged_posts: int
    posts: list[ThreadPostResult]
    payload_alerts: list[PayloadAlert] = Field(default_factory=list)
    warning: str | None = None


class ThreadAnalysisResponse(ThreadReport):
    analysis_id: int


class NetworkNode(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    user_id: str = Field(min_length=1, max_length=200)
    screen_name: str | None = Field(default=None, max_length=200)
    followers_count: float = Field(default=0.0, ge=0)
    following_count: float = Field(default=0.0, ge=0)
    statuses_count: float = Field(default=0.0, ge=0)
    account_age_days: float = Field(default=0.0, ge=0)
    verified: bool = False
    description: str = Field(default="", max_length=10_000)
    source_dataset: str | None = Field(default=None, max_length=200)
    source_group: str | None = Field(default=None, max_length=200)
    ground_truth: str | None = Field(default=None, max_length=100)
    label_name: str | None = Field(default=None, max_length=100)
    provenance: str | None = Field(default=None, max_length=100)
    crawl_era: str | None = Field(default=None, max_length=100)
    cluster_id: int | None = Field(default=None, ge=0)
    community_id: str | None = Field(default=None, max_length=200)


class NetworkEdge(BaseModel):
    source: str = Field(min_length=1, max_length=200)
    target: str = Field(min_length=1, max_length=200)
    relation: str = Field(default="interacts", min_length=1, max_length=100)


class NetworkPost(BaseModel):
    post_id: str | None = Field(default=None, max_length=200)
    user_id: str = Field(min_length=1, max_length=200)
    text: str = Field(default="", max_length=100_000)
    created_at: datetime
    hashtags: list[str] = Field(default_factory=list, max_length=100)
    mentions: list[str] = Field(default_factory=list, max_length=100)


class NetworkAnalyzeRequest(BaseModel):
    nodes: list[NetworkNode] = Field(min_length=1, max_length=10_000)
    edges: list[NetworkEdge] = Field(default_factory=list, max_length=100_000)
    posts: list[NetworkPost] = Field(default_factory=list, max_length=100_000)

    @model_validator(mode="after")
    def validate_references(self) -> NetworkAnalyzeRequest:
        ids = [node.user_id for node in self.nodes]
        if len(ids) != len(set(ids)):
            raise ValueError("nodes contain duplicate user_id values")
        known = set(ids)
        dangling_edges = [
            f"{edge.source}->{edge.target}"
            for edge in self.edges
            if edge.source not in known or edge.target not in known
        ]
        dangling_posts = [
            post.user_id for post in self.posts if post.user_id not in known
        ]
        if dangling_edges:
            raise ValueError(
                f"edge endpoints must reference supplied nodes; examples: "
                f"{dangling_edges[:3]}"
            )
        if dangling_posts:
            raise ValueError(
                f"post user_id must reference a supplied node; examples: "
                f"{dangling_posts[:3]}"
            )
        return self


class NodeThreat(BaseModel):
    user_id: str
    screen_name: str | None = None
    score: float = Field(ge=0.0, le=1.0)
    evidence_score: float = Field(default=0.0, ge=0.0, le=1.0)
    label: ThreatLabel
    top_features: dict[str, float]
    features: dict[str, float] = Field(default_factory=dict)
    is_target: bool = False
    coordinated: bool = False
    followers_count: float = Field(default=0.0, ge=0)
    following_count: float = Field(default=0.0, ge=0)
    statuses_count: float = Field(default=0.0, ge=0)
    account_age_days: float = Field(default=0.0, ge=0)
    verified: bool = False
    description: str = ""
    source_dataset: str | None = None
    source_group: str | None = None
    ground_truth: str | None = None
    provenance: str | None = None
    crawl_era: str | None = None
    cluster_id: int = Field(default=0, ge=0)
    community_label: str | None = None
    cluster_size: int = Field(default=1, ge=1)


class ClusterSummary(BaseModel):
    cluster_id: int = Field(ge=0)
    size: int = Field(ge=1)
    coordinated_accounts: int = Field(default=0, ge=0)
    mean_evidence_score: float = Field(default=0.0, ge=0.0, le=1.0)
    ground_truth_counts: dict[str, int] = Field(default_factory=dict)


class CoordinationMetrics(BaseModel):
    """Structural coordination evidence, averaged across submitted accounts."""

    temporal_synchrony: float
    peak_synchrony: float
    synchronised_partners: float
    reciprocity_rate: float
    peak_reciprocity: float = 0.0
    cross_account_duplication: float
    peak_cross_account_duplication: float = 0.0
    self_duplication: float
    burstiness: float
    circadian_flatness: float
    hashtag_overlap: float
    follower_following_log_ratio: float


class GraphEdge(BaseModel):
    source: str
    target: str
    relation: str


class NetworkReport(BaseModel):
    model: str
    threshold: float
    aggregate_score: float
    # The graph model, text branch, and rule-based coordination evidence, so an
    # analyst can see which signal drove the headline score.
    score_components: dict[str, float] = Field(default_factory=dict)
    # False when the graph artifact's out-of-distribution transfer score is too
    # low for its node scores to raise the reported risk.
    model_trusted: bool = True
    flagged_nodes: int
    total_nodes: int
    coordinated_accounts: int = 0
    cluster_count: int = 0
    clusters: list[ClusterSummary] = Field(default_factory=list)
    nodes: list[NodeThreat]
    classification: SubjectLabel
    band: TriageBand
    reasons: list[str] = Field(default_factory=list)
    coordination: CoordinationMetrics
    edges: list[GraphEdge] = Field(default_factory=list)
    suspicious_edges: list[GraphEdge] = Field(default_factory=list)
    warning: str | None = None


class NetworkAnalysisResponse(NetworkReport):
    analysis_id: int


class AccountAnalyzeRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    handle: str = Field(min_length=1, max_length=200)
    peers: int = Field(default=5, ge=2, le=12)


class AccountAnalysisResponse(BaseModel):
    """The combined report for a single target account."""

    analysis_id: int
    handle: str
    data_source: Literal["simulated"]
    simulation_note: str
    classification: SubjectLabel
    band: TriageBand
    threat_score: float = Field(ge=0.0, le=1.0)
    reasons: list[str] = Field(default_factory=list)
    text: ThreadReport
    network: NetworkReport
    warning: str | None = None


class DashboardRecord(BaseModel):
    id: int
    analysis_type: Literal["text", "network"]
    score: float
    label: ThreatLabel
    created_at: datetime
    details: dict[str, Any]


class ThreatDashboardResponse(BaseModel):
    total_analyses: int
    text_analyses: int
    network_analyses: int
    adversarial_findings: int
    mean_score: float
    recent: list[DashboardRecord]
