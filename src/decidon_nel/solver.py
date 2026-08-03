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
    """Internal representation of a person(PER/SPK)-function(FCT) relationship for lexical resolution."""

    person: Entity
    fct: Entity
    tokens: set[str]
    name_tokens: set[str]
    fct_position: int
    last_seen_position: int
    is_external: bool = False


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
    """Computes the coverage score (asymmetrical inclusion) of the mention tokens in the FCT relations tokens :

    coverage = |mention_set ∩ fctrelations_set| / |mention_set|

    Returns 0.0 if mention_set is empty.
    """
    return (
        len(mention_set & fctrelations_set) / len(mention_set) if mention_set else 0.0
    )


class LSAnnotation(BaseModel):
    """Pydantic model for a Label Studio annotation, including the task text and the results of the last annotation."""

    id: int
    id_task: int
    text: str = ""
    results: list[dict] = Field(default_factory=list)

    @classmethod
    def from_dict(cls, task_dict: dict) -> "LSAnnotation":
        """Extrait le texte de la tâche et les résultats de la dernière annotation."""
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
        """Parse les entités/relations de la tâche en appliquant l'offset continu de séance."""
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
    """Charge le fichier JSON Label Studio brut."""
    with open(Path(file_path), "r", encoding="utf-8") as f:
        return json.load(f)


def extract_session(
    data_or_path: str | Path | list[dict] | pd.DataFrame,
    task_ids: list[int] | None = None,
) -> tuple[list[EntityWithFcts], str]:
    """Extrait et reconstitue la séance avec un index global continu et trié chronologiquement."""
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
    """Résolveur FCT multi-scopes (Séance, Section, Paragraphe) hautement optimisé."""

    def __init__(
        self,
        jaccard_threshold: float = 0.70,  # Jaccard threshold for strong lexical match in pass 1
        coverage_threshold: float = 0.85,  # Inclusion coverage threshold for pass 2
        scope_routing: dict[str, Scope] | None = None,
    ):
        self.jaccard_threshold = jaccard_threshold
        self.coverage_threshold = coverage_threshold
        self.fct_relations: list[FCTRelation] = []
        self._fct_relations_keys: set[tuple[str, str]] = set()
        self.titl_positions: list[int] = []
        self.scope_routing: dict[str, Scope] = {}

        if scope_routing:
            self.scope_routing.update(scope_routing)

    def add_scope_rule(self, keyword: str, scope: Scope) -> None:
        """Adds a keyword-based routing rule for determining the scope of resolution."""
        if norm_key := normalize_text(keyword):
            self.scope_routing[norm_key] = scope

    def add_fctrelation(
        self, person: Entity, fct: Entity, position: int, is_external: bool = False
    ) -> None:
        """Registers a FCT relation"""
        key = (person.text, fct.text)
        if key in self._fct_relations_keys:
            return

        tokens = get_tokens(fct.text)
        if not tokens:
            return

        name_tokens = {w for w in normalize_text(person.text).split() if len(w) > 2}
        fctrelation = FCTRelation(
            person=person,
            fct=fct,
            tokens=tokens,
            name_tokens=name_tokens,
            fct_position=position,
            last_seen_position=position,
            is_external=is_external,
        )
        self.fct_relations.append(fctrelation)
        self._fct_relations_keys.add(key)
        logger.debug("FCT relation added : {} -> '{}'", person.text, fct.text)

    def inject_external_fctrelations(
        self, external_pairs: list[tuple[str, str]] | list[dict[str, str]]
    ) -> None:
        """Injecte des paires (person_name, fct_text) issues d'une base de connaissances externe."""
        for item in external_pairs:
            match item:
                case (str(name), str(fct)) if name and fct:
                    self.add_fctrelation(name, fct, position=-1, is_external=True)
                case {"person_name": name, "fct_text": fct} | {
                    "name": name,
                    "fct": fct,
                } if (
                    name and fct
                ):
                    self.add_fctrelation(
                        str(name), str(fct), position=-1, is_external=True
                    )
                case _:
                    continue
        logger.info("{} external FCT relations injected.", len(external_pairs))

    def _get_section_index(self, pos: int) -> int:
        """Calcule l'index de section (délimité par les entités TITL) en O(log N) via bisect."""
        if pos < 0:
            return -1
        return bisect.bisect_right(self.titl_positions, pos)

    def _get_scope_for_mention(self, text: str) -> Scope:
        """Détermine le scope de résolution d'une mention selon le routage des mots-clés."""
        tokens = get_tokens(text)
        return next(
            (self.scope_routing[tok] for tok in tokens if tok in self.scope_routing),
            Scope.SESSION,
        )

    def _is_fctrelation_in_scope(
        self, fctrelation: FCTRelation, curr_pos: int, target_scope: Scope
    ) -> bool:
        """Checks if a FCTRelation is within the desired scope relative to the current mention position."""
        if fctrelation.is_external or target_scope == Scope.SESSION:
            return True

        match target_scope:
            case Scope.SECTION:
                return self._get_section_index(
                    fctrelation.fct_position
                ) == self._get_section_index(curr_pos)
            case Scope.PARAGRAPH:
                return abs(curr_pos - fctrelation.last_seen_position) < 800
            case _:
                return True

    def _update_position(self, person_name: str, new_position: int) -> None:
        """Met à jour l'offset de récence d'une personne."""
        for p in self.fct_relations:
            if p.person.text == person_name:
                p.last_seen_position = new_position

    def observe_mention(self, main_ent: EntityWithFcts) -> None:
        """Observe a mention and updates the last seen position of any matching FCT relations based on lexical similarity."""
        ent = main_ent.entity
        if ent.type.upper() == "TITL":
            if ent.start not in self.titl_positions:
                bisect.insort(self.titl_positions, ent.start)
            return

        if not ent.text:
            return

        pos = ent.start
        if is_proper_name(ent.text):
            if mention_tokens := {
                w for w in normalize_text(ent.text).split() if len(w) > 2
            }:
                for p in self.fct_relations:
                    if mention_tokens & p.name_tokens:
                        self._update_position(p.person.text, pos)
                        return
            return

        if mention_tokens := get_tokens(ent.text):
            for p in self.fct_relations:
                if (
                    jaccard_similarity(mention_tokens, p.tokens)
                    >= self.jaccard_threshold
                ):
                    self._update_position(p.person.text, pos)

    def update_state(self, main_ent: EntityWithFcts) -> None:
        """Indexation initiale des TITL et des FCT rattachées aux personnes nommées."""
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

    def get_resolvable_entities(
        self, session_entities: list[EntityWithFcts]
    ) -> list[EntityWithFcts]:
        """Filtre les entités de la séance pour ne conserver que celles éligibles à la désambiguïsation."""
        return [e for e in session_entities if should_resolve(e)]

    def resolve(self, main_ent: EntityWithFcts, top_k: int = 3) -> list[Candidate]:
        """Résout un titre en appliquant la stratégie à 2 passes et filtrage par scope."""
        if not should_resolve(main_ent) or not self.fct_relations:
            return []

        curr_pos = main_ent.entity.start
        mention_tokens = get_tokens(main_ent.entity.text)
        if not mention_tokens:
            return []

        target_scope = self._get_scope_for_mention(main_ent.entity.text)
        sec_idx = self._get_section_index(curr_pos)
        sec_info = f" (S#{sec_idx})" if target_scope == Scope.SECTION else ""

        # --- PASSE 1 : Match lexical fort direct (Interne) ---
        fct_relations = [p for p in self.fct_relations if not p.is_external]
        pass1_cands: list[tuple[float, Candidate]] = []
        seen_names: set[str] = set()

        for p in reversed(fct_relations):
            if p.person.text in seen_names or not self._is_fctrelation_in_scope(
                p, curr_pos, target_scope
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
            self._update_position(results[0].entity.text, curr_pos)
            return results

        # --- SOURCE EXTERNE MATCH ---
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
                    explanation=f"FCT: '{p.fct.text}' | Jaccard: {jaccard:.2f}",
                )
                external_cands.append((jaccard, cand))
                seen_names.add(p.person.text)

        if external_cands:
            external_cands.sort(key=lambda x: x[0], reverse=True)
            return [cand for _, cand in external_cands[:top_k]]

        # --- PASSE 2 : Inclusion asymétrique (Couverture) ---
        ordered_fctrelations = sorted(
            fct_relations,
            key=lambda p: (
                0 if p.last_seen_position < curr_pos else 1,
                (
                    -p.last_seen_position
                    if p.last_seen_position < curr_pos
                    else p.last_seen_position
                ),
            ),
        )

        pass2_cands: list[Candidate] = []
        seen_names.clear()

        for p in ordered_fctrelations:
            if p.person.text in seen_names or not self._is_fctrelation_in_scope(
                p, curr_pos, target_scope
            ):
                continue

            cov = coverage_score(mention_tokens, p.tokens)
            if cov >= self.coverage_threshold:
                dist = curr_pos - p.last_seen_position
                cand = Candidate(
                    entity=p.person,
                    decision=DecisionType.PASS3_COREF_UPWARD_MATCH,
                    scope=target_scope,
                    explanation=(
                        f"FCT: '{p.fct.text}' | Couverture: {cov:.2f} | "
                        f"Distance: {dist}"
                    ),
                )
                pass2_cands.append(cand)
                seen_names.add(p.person.text)

                if len(pass2_cands) >= top_k:
                    break

        return pass2_cands


if __name__ == "__main__":
    logger.info("Démarrage du test du résolveur FCT...")
    file_path = Path("data/2026-07-01_export_label-studio_NER-NEL_pre-traite.json")

    if file_path.exists():
        df_data = load_ls_data(file_path)
        session_entities, session_text = extract_session(df_data, task_ids=[995, 996])

        logger.info("Extraction de {} entités pour la séance.", len(session_entities))

        resolver = LexicalFCTResolver(threshold_pass1=0.70, threshold_pass2=0.85)
        resolver.add_scope_rule("rapporteur", Scope.SECTION)

        resolver.inject_external_fctrelations(
            [
                ("Léon Blum", "président du conseil"),
                ("Albert Lebrun", "président de la république"),
            ]
        )

        # Indexation initiale
        for main_ent in session_entities:
            resolver.update_state(main_ent)

        logger.info("Titres TITL enregistrés aux offsets : {}", resolver.titl_positions)

        for main_ent in session_entities:
            resolver.observe_mention(main_ent)

            if should_resolve(main_ent):
                results = resolver.resolve(main_ent, top_k=3)
                logger.info("Mention à désambiguïser : {}", main_ent.entity)
                for cand in results:
                    logger.info("   └─ {}", cand)
    else:
        logger.warning("Fichier non trouvé : {}", file_path)
