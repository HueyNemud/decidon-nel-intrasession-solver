import bisect
from dataclasses import dataclass, field
from enum import Enum
from functools import lru_cache
import json
import logging
from pathlib import Path
import re
from typing import TypeAlias
import unicodedata

logger = logging.getLogger(__name__)

EntityKey: TypeAlias = tuple[int, int, str]


class DecisionType(str, Enum):
    """Typology of resolver decision passes."""

    PASS1_FCT_MATCH = "1-DIRECT"
    PASS2_KB_MATCH = "2-EXTERNAL"
    PASS3_COREF_UPWARD_MATCH = "3-UPWARD_COREFERENCE"


class Scope(str, Enum):
    """Spatial resolution scopes."""

    SESSION = "session"
    SECTION = "section"


@dataclass(frozen=True, slots=True)
class Entity:
    """Named entity with its span positions and task metadata."""

    id: str
    start: int
    end: int
    text: str
    type: str
    task_id: int
    annotation_id: int

    def __str__(self) -> str:
        return (
            f"{self.start}:{self.end} "
            f"{self.text} [{self.type}] "
            f"(task={self.task_id}, annotation={self.annotation_id})"
        )


@dataclass(frozen=True, slots=True)
class LinkedFct:
    """Function entity linked to a main entity."""

    entity: Entity
    relation_type: str


@dataclass(frozen=True, slots=True)
class EntityWithFcts:
    """Main entity aggregated with its associated function annotations."""

    entity: Entity
    fcts: list[LinkedFct] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class Candidate:
    """Candidate result produced by entity disambiguation."""

    entity: Entity
    decision: DecisionType
    explanation: str
    scope: Scope

    def __str__(self) -> str:
        return f"{self.entity.id} {self.entity.text} | {self.decision.value} | {self.scope.value} | {self.explanation}"


@dataclass(slots=True)
class FCTRelation:
    """Internal representation of a person-function pairing."""

    person: Entity
    fct: Entity
    tokens: frozenset[str]
    fct_position: int
    is_external: bool = False


@dataclass(frozen=True, slots=True)
class FocusEntry:
    """Focus stack entry recording active role binding at a specific offset."""

    fct_relation: FCTRelation
    position: int


HONORIFICS = frozenset(
    {
        "m.",
        "m",
        "mr.",
        "mr",
        "mm.",
        "mm",
        "mme.",
        "mme",
        "mlle.",
        "mlle",
        "messieurs",
        "mesdames",
    }
)

STOPWORDS = HONORIFICS | frozenset(
    {
        "le",
        "la",
        "les",
        "l",
        "de",
        "du",
        "des",
        "d",
        "un",
        "une",
        "au",
        "aux",
        "et",
        "en",
    }
)

RE_TRIM_PUNCT = re.compile(r"^[^\w]+|[^\w]+$")
RE_STEM_FR = re.compile(r"(ence|ent|erie|iat|aire|ere|eur|ion|iste|e)$")


@lru_cache(maxsize=2048)
def stem_word(word: str) -> str:
    """Light French stemming to align grammatical variants (e.g., présidence -> presid)."""
    return RE_STEM_FR.sub("", word) if len(word) > 4 else word


@lru_cache(maxsize=4096)
def normalize_text(text: str) -> str:
    """Clean text, strip accents, remove stopwords, and apply light stemming."""
    if not text:
        return ""

    nfkd_form = unicodedata.normalize("NFKD", text)
    without_accents = "".join(
        c for c in nfkd_form if not unicodedata.combining(c)
    ).lower()

    words = (RE_TRIM_PUNCT.sub("", w) for w in without_accents.split())
    filtered_stemmed = [stem_word(w) for w in words if w and w not in STOPWORDS]
    return " ".join(filtered_stemmed)


@lru_cache(maxsize=4096)
def get_tokens(text: str) -> frozenset[str]:
    """Extract normalized tokens from string."""
    norm = normalize_text(text)
    return frozenset(norm.split()) if norm else frozenset()


def is_proper_name(text: str) -> bool:
    """Check whether text represents a proper name based on capitalization and honorifics."""
    words = text.strip().split()
    if not words:
        return False

    first_word_clean = RE_TRIM_PUNCT.sub("", words[0].lower())
    remaining_words = words[1:] if first_word_clean in HONORIFICS else words

    return any(w[0].isupper() for w in remaining_words if w and w[0].isalpha())


def should_resolve(main_ent: EntityWithFcts) -> bool:
    """Determine if entity requires disambiguation (PER/SPK without explicit proper name)."""
    return main_ent.entity.type.upper() in ("PER", "SPK") and not is_proper_name(
        main_ent.entity.text
    )


def jaccard_similarity(set1: frozenset[str], set2: frozenset[str]) -> float:
    """Compute Jaccard similarity between two token sets."""
    if not set1 or not set2:
        return 0.0
    return len(set1 & set2) / len(set1 | set2)


def coverage_score(mention_tokens: frozenset[str], fct_tokens: frozenset[str]) -> float:
    """Calculate token coverage ratio of mention within target function tokens."""
    return len(mention_tokens & fct_tokens) / len(mention_tokens) if mention_tokens else 0.0


@dataclass(slots=True)
class LSAnnotation:
    """Parser for single Label Studio task annotation export."""

    id: int
    id_task: int
    text: str = ""
    results: list[dict] = field(default_factory=list)

    @classmethod
    def from_dict(cls, task_dict: dict) -> "LSAnnotation":
        annotations = task_dict.get("annotations") or []
        last_ann = annotations[-1] if annotations else {}
        task_data = task_dict.get("data") or {}
        text = task_data.get("text", "") if isinstance(task_data, dict) else ""

        return cls(
            id=last_ann.get("id", 0),
            id_task=task_dict.get("id", 0),
            text=text,
            results=last_ann.get("result", []),
        )

    def extract_main_entities(self, offset: int = 0) -> list[EntityWithFcts]:
        entities: dict[str, Entity] = {}
        relations: list[tuple[str, str, str]] = []

        for item in self.results:
            item_id, item_type = item.get("id"), item.get("type")
            if item_type == "labels" and item_id:
                val = item.get("value", {})
                labels = val.get("labels") or ["UNKNOWN"]
                entities[item_id] = Entity(
                    id=item_id,
                    start=val.get("start", 0) + offset,
                    end=val.get("end", 0) + offset,
                    text=val.get("text", ""),
                    type=labels[0] if labels else "UNKNOWN",
                    task_id=self.id_task,
                    annotation_id=self.id,
                )
            elif item_type == "relation":
                rel_type = (item.get("labels") or ["relation"])[0]
                relations.append(
                    (item.get("from_id", ""), item.get("to_id", ""), rel_type)
                )

        target_fcts: dict[str, list[LinkedFct]] = {}
        for src_id, tgt_id, rel_type in relations:
            src, tgt = entities.get(src_id), entities.get(tgt_id)
            if src and tgt:
                if src.type.lower() == "fct":
                    target_fcts.setdefault(tgt_id, []).append(LinkedFct(src, rel_type))
                if tgt.type.lower() == "fct":
                    target_fcts.setdefault(src_id, []).append(LinkedFct(tgt, rel_type))

        return [
            EntityWithFcts(entity=e, fcts=target_fcts.get(e.id, []))
            for e in entities.values()
            if e.type.lower() != "fct"
        ]


def load_ls_data(file_path: str | Path) -> list[dict]:
    """Load Label Studio raw JSON array."""
    with Path(file_path).open("r", encoding="utf-8") as f:
        return json.load(f)


def extract_session(
    data_or_path: str | Path | list[dict],
    task_ids: list[int] | None = None,
) -> tuple[list[EntityWithFcts], str]:
    """Extract and aggregate entity spans across tasks into session timeline."""
    raw_tasks = load_ls_data(data_or_path) if isinstance(data_or_path, (str, Path)) else data_or_path

    tasks = [
        LSAnnotation.from_dict(t)
        for t in raw_tasks
        if task_ids is None or t.get("id") in task_ids
    ]

    session_entities: list[EntityWithFcts] = []
    session_text_parts: list[str] = []
    current_offset = 0

    for task in tasks:
        session_entities.extend(task.extract_main_entities(current_offset))
        session_text_parts.append(task.text)
        current_offset += len(task.text) + 2

    session_entities.sort(key=lambda e: e.entity.start)
    return session_entities, "\n\n".join(session_text_parts)


class LexicalFCTResolver:
    """Focus Stack FCT Resolver for spatial and lexical entity resolution."""

    def __init__(
        self,
        jaccard_threshold: float = 0.70,
        coverage_threshold: float = 0.85,
        scope_routing: dict[str, Scope] | None = None,
        debug_focus_stack: bool = False,
    ):
        self.jaccard_threshold = jaccard_threshold
        self.coverage_threshold = coverage_threshold
        self.debug_focus_stack = debug_focus_stack
        self.fct_relations: list[FCTRelation] = []
        self._fct_relations_keys: set[tuple[str, str]] = set()
        self.focus_stack: list[FocusEntry] = []
        self.titl_positions: list[int] = []
        self.scope_routing: dict[str, Scope] = {}

        if scope_routing:
            self.scope_routing.update(scope_routing)

    def get_resolvable_entities(
        self, entities: list[EntityWithFcts]
    ) -> list[EntityWithFcts]:
        """Filter entities that require disambiguation."""
        return [ent for ent in entities if should_resolve(ent)]

    def add_scope_rule(self, keyword: str, scope: Scope) -> None:
        """Assign spatial scope rule to a specific function keyword."""
        if norm_key := normalize_text(keyword):
            self.scope_routing[norm_key] = scope

    def push_focus(self, fct_relation: FCTRelation, position: int) -> None:
        """Push role activation entry onto focus stack."""
        self.focus_stack.append(FocusEntry(fct_relation=fct_relation, position=position))
        
        if self.debug_focus_stack:
            logger.debug(
                "Focus Stack [+] : '%s' (%s) at pos %s",
                fct_relation.fct.text,
                fct_relation.person.text,
                position,
            )

    def add_fctrelation(
        self, person: Entity, fct: Entity, position: int, is_external: bool = False
    ) -> FCTRelation | None:
        """Register person-function pair."""
        key = (person.text, fct.text)
        if key in self._fct_relations_keys:
            return None

        tokens = get_tokens(fct.text)
        if not tokens:
            return None

        fctrelation = FCTRelation(
            person=person,
            fct=fct,
            tokens=tokens,
            fct_position=position,
            is_external=is_external,
        )
        self.fct_relations.append(fctrelation)
        self._fct_relations_keys.add(key)
        
        source = "External KB" if is_external else f"Internal (pos={position})"
        logger.debug(
            "Registered FCT relation [%s]: %s [%s] ──> '%s' [%s]",
            source,
            person.text,
            person.type,
            fct.text,
            fct.type,
        )
        return fctrelation

    def inject_external_fctrelations(
        self, external_data: dict[str, str] | list[tuple[str, str]] | list[dict[str, str]]
    ) -> None:
        """Inject external KB profiles (dict, tuple list, or dict list)."""
        count_before = len(self.fct_relations)
        items = external_data.items() if isinstance(external_data, dict) else external_data

        for item in items:
            name, fct = None, None
            if isinstance(item, (tuple, list)) and len(item) >= 2:
                name, fct = item[0], item[1]
            elif isinstance(item, dict):
                name = item.get("person_name") or item.get("name")
                fct = item.get("fct_text") or item.get("fct")

            if name and fct:
                idx = len(self.fct_relations)
                dummy_person = Entity(
                    id=f"ext_{idx}",
                    start=-1,
                    end=-1,
                    text=str(name),
                    type="PER",
                    task_id=-1,
                    annotation_id=-1,
                )
                dummy_fct = Entity(
                    id=f"ext_fct_{idx}",
                    start=-1,
                    end=-1,
                    text=str(fct),
                    type="FCT",
                    task_id=-1,
                    annotation_id=-1,
                )
                self.add_fctrelation(dummy_person, dummy_fct, position=-1, is_external=True)

        logger.info("%d external KB relations injected.", len(self.fct_relations) - count_before)

    def _get_section_index(self, pos: int) -> int:
        return bisect.bisect_right(self.titl_positions, pos) if pos >= 0 else -1

    def _get_scope_for_mention(self, text: str) -> Scope:
        tokens = get_tokens(text)
        return next(
            (self.scope_routing[tok] for tok in tokens if tok in self.scope_routing),
            Scope.SESSION,
        )

    def _is_fctrelation_in_scope(
        self, fctrelation: FCTRelation, curr_pos: int, target_scope: Scope, active_pos: int
    ) -> bool:
        if fctrelation.is_external or target_scope == Scope.SESSION:
            return True

        match target_scope:
            case Scope.SECTION:
                return self._get_section_index(active_pos) == self._get_section_index(curr_pos)
            case Scope.PARAGRAPH:
                return abs(curr_pos - active_pos) < 800
            case _:
                return True

    def activate_matching_roles(self, main_ent: EntityWithFcts) -> None:
        """Activate matching internal roles on focus stack for subsequent coreference."""
        ent = main_ent.entity
        if not ent.text:
            return

        mention_tokens = get_tokens(ent.text)
        for relation in self.fct_relations:
            if not relation.is_external and jaccard_similarity(mention_tokens, relation.tokens) >= self.jaccard_threshold:
                self.push_focus(relation, ent.start)

    def update_state(self, main_ent: EntityWithFcts) -> None:
        """Index explicit TITL boundary spans and function_of relations."""
        ent = main_ent.entity
        if ent.type.upper() == "TITL":
            if ent.start not in self.titl_positions:
                bisect.insort(self.titl_positions, ent.start)
            return

        if not is_proper_name(ent.text):
            return

        for fct_link in main_ent.fcts:
            if fct_link.relation_type == "function_of":
                self.add_fctrelation(
                    person=ent,
                    fct=fct_link.entity,
                    position=ent.start,
                    is_external=False,
                )

    def resolve(self, main_ent: EntityWithFcts, top_k: int = 3) -> list[Candidate]:
        """Resolve ambiguous title via Direct Match (Pass 1), External KB (Pass 2), or Focus Stack (Pass 3)."""
        if not should_resolve(main_ent) or not self.fct_relations:
            return []

        curr_pos = main_ent.entity.start
        mention_tokens = get_tokens(main_ent.entity.text)
        if not mention_tokens:
            return []

        target_scope = self._get_scope_for_mention(main_ent.entity.text)

        # --- PASS 1: Direct Internal Lexical Match ---
        pass1_cands: list[tuple[float, Candidate]] = []
        seen_people: set[Entity] = set()

        for p in reversed([r for r in self.fct_relations if not r.is_external]):
            if p.person in seen_people:
                continue

            jaccard = jaccard_similarity(mention_tokens, p.tokens)
            if jaccard >= self.jaccard_threshold:
                cand = Candidate(
                    entity=p.person,
                    decision=DecisionType.PASS1_FCT_MATCH,
                    scope=target_scope,
                    explanation=f"FCT: '{p.fct.text}' | Jaccard: {jaccard:.2f}",
                )
                pass1_cands.append((jaccard, cand))
                seen_people.add(p.person)

        if pass1_cands:
            pass1_cands.sort(key=lambda x: x[0], reverse=True)
            return [cand for _, cand in pass1_cands[:top_k]]

        # --- PASS 2: External KB Match ---
        external_cands: list[tuple[float, Candidate]] = []
        seen_people.clear()

        for p in (r for r in self.fct_relations if r.is_external):
            if p.person in seen_people:
                continue

            jaccard = jaccard_similarity(mention_tokens, p.tokens)
            if jaccard >= self.jaccard_threshold:
                cand = Candidate(
                    entity=p.person,
                    decision=DecisionType.PASS2_KB_MATCH,
                    scope=target_scope,
                    explanation=f"FCT Externe: '{p.fct.text}' | Jaccard: {jaccard:.2f}",
                )
                external_cands.append((jaccard, cand))
                seen_people.add(p.person)

        if external_cands:
            external_cands.sort(key=lambda x: x[0], reverse=True)
            return [cand for _, cand in external_cands[:top_k]]

        # --- PASS 3: Upward Focus Stack Coreference ---
        pass3_cands: list[Candidate] = []
        seen_people.clear()

        for entry in reversed(self.focus_stack):
            p = entry.fct_relation
            if entry.position >= curr_pos:
                continue

            if p.person in seen_people or not self._is_fctrelation_in_scope(
                p, curr_pos, target_scope, entry.position
            ):
                continue

            cov = coverage_score(mention_tokens, p.tokens)
            if cov >= self.coverage_threshold:
                dist = curr_pos - entry.position
                cand = Candidate(
                    entity=p.person,
                    decision=DecisionType.PASS3_COREF_UPWARD_MATCH,
                    scope=target_scope,
                    explanation=(
                        f"FCT: '{p.fct.text}' | Couverture: {cov:.2f} | "
                        f"Distance activation: {dist} chars"
                    ),
                )
                pass3_cands.append(cand)
                seen_people.add(p.person)

                if len(pass3_cands) >= top_k:
                    break

        return pass3_cands