import bisect
from dataclasses import dataclass, field
from enum import Enum
from functools import lru_cache
import json
from pathlib import Path
import re
from typing import Literal
import unicodedata

from loguru import logger
import pandas as pd
from pydantic import BaseModel, Field


class DecisionType(str, Enum):
    """Typologie des décisions du résolveur."""

    PASS1_FCT_MATCH = "1-DIRECT"
    PASS2_KB_MATCH = "2-EXTERNAL"
    PASS3_COREF_UPWARD_MATCH = "3-UPWARD_COREFERENCE"


class Scope(str, Enum):
    """Portée spatiale de la désambiguïsation."""

    SESSION = "session"
    SECTION = "section"
    PARAGRAPH = "paragraph"


@dataclass(frozen=True, slots=True)
class Entity:
    """Named entity with its position in the document."""

    id: str
    start: int
    end: int
    text: str
    type: str
    task_id: int
    annotation_id: int

    def shifted(self, offset: int) -> "Entity":
        return Entity(
            id=self.id,
            start=self.start + offset,
            end=self.end + offset,
            text=self.text,
            type=self.type,
            task_id=self.task_id,
            annotation_id=self.annotation_id,
        )

    def __str__(self) -> str:
        return (
            f"{self.start}:{self.end} "
            f"{self.text} [{self.type}] "
            f"(task={self.task_id}, annotation={self.annotation_id})"
        )


@dataclass(frozen=True, slots=True)
class LinkedFct:
    """Fonction reliée à une entité principale."""

    entity: Entity
    relation_type: str
    direction: Literal["incoming", "outgoing"]

    def shifted(self, offset: int) -> "LinkedFct":
        return LinkedFct(
            self.entity.shifted(offset), self.relation_type, self.direction
        )


@dataclass(frozen=True, slots=True)
class EntityWithFcts:
    """Entité principale agrégée avec ses fonctions rattachées."""

    entity: Entity
    fcts: list[LinkedFct] = field(default_factory=list)

    def shifted(self, offset: int) -> "EntityWithFcts":
        return EntityWithFcts(
            entity=self.entity.shifted(offset),
            fcts=[fct.shifted(offset) for fct in self.fcts],
        )


@dataclass(frozen=True, slots=True)
class Candidate:
    """Candidat retenu à l'issue du processus de désambiguïsation."""

    entity: Entity
    decision: DecisionType
    explanation: str
    scope: Scope

    def __str__(self) -> str:
        return f"{self.entity.id} {self.entity.text} | {self.decision.value} | {self.scope.value} | {self.explanation}"


@dataclass(slots=True)
class FCTRelation:
    """Représentation interne d'une relation personne-fonction."""

    person: Entity
    fct: Entity
    tokens: set[str]
    name_tokens: set[str]
    fct_position: int
    is_external: bool = False


@dataclass(frozen=True, slots=True)
class FocusEntry:
    """Entrée dans la pile de focus représentant l'activation d'un rôle à une position précise."""

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
    """Racinisation française légère pour aligner les formes dérivées (ex: présidence -> presid)."""
    return RE_STEM_FR.sub("", word) if len(word) > 4 else word


@lru_cache(maxsize=4096)
def normalize_text(text: str) -> str:
    """Nettoie, supprime les accents, filtre les stopwords et applique une racinisation légère."""
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
def get_tokens(text: str) -> set[str]:
    """Extrait l'ensemble des mots normalisés d'un texte."""
    norm = normalize_text(text)
    return set(norm.split()) if norm else set()


def is_proper_name(text: str) -> bool:
    """Détermine si le texte est un nom propre via les civilités et la casse."""
    words = text.strip().split()
    if not words:
        return False

    first_word_clean = RE_TRIM_PUNCT.sub("", words[0].lower())
    remaining_words = words[1:] if first_word_clean in HONORIFICS else words

    return any(w[0].isupper() for w in remaining_words if w and w[0].isalpha())


def should_resolve(main_ent: EntityWithFcts) -> bool:
    """Vérifie si une entité doit être désambiguïsée (PER/SPK sans nom propre)."""
    return main_ent.entity.type.upper() in ("PER", "SPK") and not is_proper_name(
        main_ent.entity.text
    )


def jaccard_similarity(set1: set[str], set2: set[str]) -> float:
    """Calcul de la similarité de Jaccard entre deux ensembles de tokens."""
    if not set1 or not set2:
        return 0.0
    intersection = len(set1 & set2)
    return intersection / len(set1 | set2)


def coverage_score(mention_set: set[str], fctrelations_set: set[str]) -> float:
    """Calcul du taux de couverture (inclusion asymétrique) des tokens de la mention dans la FCT."""
    return (
        len(mention_set & fctrelations_set) / len(mention_set) if mention_set else 0.0
    )


class LSAnnotation(BaseModel):
    """Modèle Pydantic pour une annotation Label Studio."""

    id: int
    id_task: int
    text: str = ""
    results: list[dict] = Field(default_factory=list)

    @classmethod
    def from_dict(cls, task_dict: dict) -> "LSAnnotation":
        annotations = task_dict.get("annotations") or []
        last_ann = annotations[-1] if annotations else {}
        task_data = task_dict.get("data", {})
        text = (
            task_data.get("text", "")
            if isinstance(task_data, dict)
            else task_dict.get("data.text", "")
        ) or ""

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
                labels = val.get("labels") or ["INCONNU"]
                entities[item_id] = Entity(
                    id=item_id,
                    start=val.get("start", 0) + offset,
                    end=val.get("end", 0) + offset,
                    text=val.get("text", ""),
                    type=labels[0] if labels else "INCONNU",
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
                    target_fcts.setdefault(tgt_id, []).append(
                        LinkedFct(src, rel_type, "incoming")
                    )
                if tgt.type.lower() == "fct":
                    target_fcts.setdefault(src_id, []).append(
                        LinkedFct(tgt, rel_type, "outgoing")
                    )

        return [
            EntityWithFcts(entity=e, fcts=target_fcts.get(e.id, []))
            for e in entities.values()
            if e.type.lower() != "fct"
        ]


def load_ls_data(file_path: str | Path) -> list[dict]:
    with open(Path(file_path), "r", encoding="utf-8") as f:
        return json.load(f)


def extract_session(
    data_or_path: str | Path | list[dict] | pd.DataFrame,
    task_ids: list[int] | None = None,
) -> tuple[list[EntityWithFcts], str]:
    if isinstance(data_or_path, (str, Path)):
        raw_tasks = load_ls_data(data_or_path)
    elif isinstance(data_or_path, pd.DataFrame):
        raw_tasks = data_or_path.to_dict(orient="records")
    else:
        raw_tasks = data_or_path

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
    """Résolveur FCT basé sur une Pile de Focus (Focus Stack) pour éviter la propagation artificielle de récence."""

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
        
        # Pile d'activation des fonctions dans l'ordre chronologique du discours
        self.focus_stack: list[FocusEntry] = []
        
        self.titl_positions: list[int] = []
        self.scope_routing: dict[str, Scope] = {}

        if scope_routing:
            self.scope_routing.update(scope_routing)

    def get_resolvable_entities(
        self, entities: list[EntityWithFcts]
    ) -> list[EntityWithFcts]:
        """Filtre et retourne la liste des entités nécessitant une désambiguïsation (PER/SPK sans nom propre)."""
        return [ent for ent in entities if should_resolve(ent)]

    def add_scope_rule(self, keyword: str, scope: Scope) -> None:
        if norm_key := normalize_text(keyword):
            self.scope_routing[norm_key] = scope

    def push_focus(self, fct_relation: FCTRelation, position: int) -> None:
        """Empile une activation de rôle spécifique à une position donnée."""
        entry = FocusEntry(fct_relation=fct_relation, position=position)
        self.focus_stack.append(entry)
        
        if self.debug_focus_stack:
            logger.debug(
                "Focus Stack [+] : '{}' ({}) à pos {}",
                fct_relation.fct.text,
                fct_relation.person.text,
                position,
            )

    def add_fctrelation(
        self, person: Entity, fct: Entity, position: int, is_external: bool = False
    ) -> FCTRelation | None:
        key = (person.text, fct.text)
        if key in self._fct_relations_keys:
            return None

        tokens = get_tokens(fct.text)
        if not tokens:
            return None

        name_tokens = {w for w in normalize_text(person.text).split() if len(w) > 2}
        fctrelation = FCTRelation(
            person=person,
            fct=fct,
            tokens=tokens,
            name_tokens=name_tokens,
            fct_position=position,
            is_external=is_external,
        )
        self.fct_relations.append(fctrelation)
        self._fct_relations_keys.add(key)
        
        if not is_external and position >= 0:
            self.push_focus(fctrelation, position)

        source_label = "KG Externe" if is_external else f"Interne (pos={position})"
        logger.debug(
            "Relation FCT enregistrée [{}] : {} [{}] ──> '{}' [{}]",
            source_label,
            person.text,
            person.type,
            fct.text,
            fct.type,
        )
        return fctrelation

    def inject_external_fctrelations(
        self, external_data: dict[str, str] | list[tuple[str, str]] | list[dict[str, str]]
    ) -> None:
        """Injecte des relations FCT externes depuis un dictionnaire {PER: FCT}, une liste de tuples ou de dicts."""
        count_before = len(self.fct_relations)

        items_to_process = (
            external_data.items()
            if isinstance(external_data, dict)
            else external_data
        )

        for item in items_to_process:
            match item:
                case (str(name), str(fct)) if name and fct:
                    dummy_person = Entity(
                        id=f"ext_{len(self.fct_relations)}",
                        start=-1,
                        end=-1,
                        text=name,
                        type="PER",
                        task_id=-1,
                        annotation_id=-1,
                    )
                    dummy_fct = Entity(
                        id=f"ext_fct_{len(self.fct_relations)}",
                        start=-1,
                        end=-1,
                        text=fct,
                        type="FCT",
                        task_id=-1,
                        annotation_id=-1,
                    )
                    self.add_fctrelation(dummy_person, dummy_fct, position=-1, is_external=True)
                case {"person_name": name, "fct_text": fct} | {"name": name, "fct": fct} if name and fct:
                    dummy_person = Entity(
                        id=f"ext_{len(self.fct_relations)}",
                        start=-1,
                        end=-1,
                        text=str(name),
                        type="PER",
                        task_id=-1,
                        annotation_id=-1,
                    )
                    dummy_fct = Entity(
                        id=f"ext_fct_{len(self.fct_relations)}",
                        start=-1,
                        end=-1,
                        text=str(fct),
                        type="FCT",
                        task_id=-1,
                        annotation_id=-1,
                    )
                    self.add_fctrelation(dummy_person, dummy_fct, position=-1, is_external=True)
                case _:
                    continue

        added_count = len(self.fct_relations) - count_before
        logger.info("{} relations FCT externes (KG) injectées au socle.", added_count)

    def _get_section_index(self, pos: int) -> int:
        if pos < 0:
            return -1
        return bisect.bisect_right(self.titl_positions, pos)

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

    def observe_mention(self, main_ent: EntityWithFcts) -> None:
        """Observe le fil du texte et empile uniquement la fonction spécifique qui est activée."""
        ent = main_ent.entity
        if ent.type.upper() == "TITL":
            if ent.start not in self.titl_positions:
                bisect.insort(self.titl_positions, ent.start)
            return

        if not ent.text or is_proper_name(ent.text):
            return

        pos = ent.start
        if mention_tokens := get_tokens(ent.text):
            for p in self.fct_relations:
                if not p.is_external and jaccard_similarity(mention_tokens, p.tokens) >= self.jaccard_threshold:
                    self.push_focus(p, pos)

    def update_state(self, main_ent: EntityWithFcts) -> None:
        """Indexation initiale des TITL et des relations FCT explicites."""
        ent = main_ent.entity
        if ent.type.upper() == "TITL":
            if ent.start not in self.titl_positions:
                bisect.insort(self.titl_positions, ent.start)
            return

        if not is_proper_name(ent.text):
            return

        for fct_link in main_ent.fcts:
            if fct_link.relation_type == "function_of":
                rel = self.add_fctrelation(
                    person=ent,
                    fct=fct_link.entity,
                    position=ent.start,
                    is_external=False,
                )
                if rel:
                    logger.debug(
                        "Extrait [function_of] : PER/SPK '{}' ({}) ──> FCT '{}' (pos={})",
                        ent.text,
                        ent.type,
                        fct_link.entity.text,
                        ent.start,
                    )

    def resolve(self, main_ent: EntityWithFcts, top_k: int = 3) -> list[Candidate]:
        """Résout un titre via Match Direct (P1), KB externe (P2) ou Dépilage de la Focus Stack (P3)."""
        if not should_resolve(main_ent) or not self.fct_relations:
            return []

        curr_pos = main_ent.entity.start
        mention_tokens = get_tokens(main_ent.entity.text)
        if not mention_tokens:
            return []

        target_scope = self._get_scope_for_mention(main_ent.entity.text)

        # --- PASSE 1 : Match lexical fort direct (Interne) ---
        fct_relations = [p for p in self.fct_relations if not p.is_external]
        pass1_cands: list[tuple[float, Candidate]] = []
        seen_names: set[str] = set()

        for p in reversed(fct_relations):
            if p.person.text in seen_names or not self._is_fctrelation_in_scope(
                p, curr_pos, target_scope, p.fct_position
            ):
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
                seen_names.add(p.person.text)

        if pass1_cands:
            pass1_cands.sort(key=lambda x: x[0], reverse=True)
            results = [cand for _, cand in pass1_cands[:top_k]]
            return results

        # --- PASSE 2 : Match Source Externe ---
        external_cands: list[tuple[float, Candidate]] = []
        seen_names.clear()

        for p in (p for p in self.fct_relations if p.is_external):
            if p.person.text in seen_names:
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
                seen_names.add(p.person.text)

        if external_cands:
            external_cands.sort(key=lambda x: x[0], reverse=True)
            return [cand for _, cand in external_cands[:top_k]]

        # --- PASSE 3 : Dépilage de la Focus Stack (Upward Coreference) ---
        pass3_cands: list[Candidate] = []
        seen_names.clear()

        # On parcourt la pile de focus du sommet (le plus récent) vers la base
        for entry in reversed(self.focus_stack):
            p = entry.fct_relation
            
            # Ne considérer que les activations situées STRICTEMENT avant la mention
            if entry.position >= curr_pos:
                continue

            if p.person.text in seen_names or not self._is_fctrelation_in_scope(
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
                seen_names.add(p.person.text)

                if len(pass3_cands) >= top_k:
                    break

        return pass3_cands