"""
Pattern matching and dependency parsing for causal relationship identification.
"""

import re
import spacy
# # # from typing import List, Tuple, Dict, Optional, Set  # Module not found  # Module not found  # Module not found
# # # from dataclasses import dataclass  # Module not found  # Module not found  # Module not found
# # # from causal_graph import CausalNode, CausalEdge, CausalRelationType  # Module not found  # Module not found  # Module not found


@dataclass
class CausalPattern:
    """Represents a linguistic pattern for causal relationships."""
    pattern: str
    relation_type: CausalRelationType
    confidence: float
    requires_dependency: bool = False


class CausalPatternMatcher:
    """Identifies causal relationships through pattern matching and dependency parsing."""
    
    def __init__(self, spacy_model: str = "en_core_web_sm"):
        try:
            self.nlp = spacy.load(spacy_model)
        except OSError:
            raise ValueError(f"SpaCy model '{spacy_model}' not found. Install with: python -m spacy download {spacy_model}")
        
        self.causal_patterns = self._initialize_patterns()
        self.causal_verbs = {
            "cause", "causes", "caused", "causing",
            "lead", "leads", "led", "leading",
            "result", "results", "resulted", "resulting", 
            "trigger", "triggers", "triggered", "triggering",
            "produce", "produces", "produced", "producing",
            "generate", "generates", "generated", "generating",
            "create", "creates", "created", "creating",
            "induce", "induces", "induced", "inducing",
            "bring", "brings", "brought", "bringing",
            "enable", "enables", "enabled", "enabling",
            "prevent", "prevents", "prevented", "preventing",
            "inhibit", "inhibits", "inhibited", "inhibiting",
            "block", "blocks", "blocked", "blocking"
        }
    
    def _initialize_patterns(self) -> List[CausalPattern]:
        """Initialize linguistic patterns for causal relationship detection."""
        return [
            # Direct causation patterns
            CausalPattern(
                r"(.+?)\s+(?:causes?|leads? to|results? in)\s+(.+)",
                CausalRelationType.DIRECT_CAUSE, 0.9
            ),
            CausalPattern(
                r"(.+?)\s+(?:triggers?|produces?|generates?)\s+(.+)",
                CausalRelationType.DIRECT_CAUSE, 0.85
            ),
            CausalPattern(
                r"due to\s+(.+?),\s+(.+)",
                CausalRelationType.DIRECT_CAUSE, 0.8
            ),
            CausalPattern(
                r"because (?:of\s+)?(.+?),\s+(.+)",
                CausalRelationType.DIRECT_CAUSE, 0.8
            ),
            CausalPattern(
                r"(.+?)\s+(?:brings? about|gives? rise to)\s+(.+)",
                CausalRelationType.DIRECT_CAUSE, 0.85
            ),
            
            # Enabling condition patterns
            CausalPattern(
                r"(.+?)\s+(?:enables?|allows?|facilitates?)\s+(.+)",
                CausalRelationType.ENABLING_CONDITION, 0.8
            ),
            CausalPattern(
                r"with\s+(.+?),\s+(.+?)\s+(?:becomes?|is)\s+possible",
                CausalRelationType.ENABLING_CONDITION, 0.75
            ),
            
            # Inhibiting condition patterns
            CausalPattern(
                r"(.+?)\s+(?:prevents?|inhibits?|blocks?)\s+(.+)",
                CausalRelationType.INHIBITING_CONDITION, 0.8
            ),
            CausalPattern(
                r"without\s+(.+?),\s+(.+?)\s+(?:cannot|will not)",
                CausalRelationType.INHIBITING_CONDITION, 0.75
            ),
            
            # Temporal sequence patterns
            CausalPattern(
                r"after\s+(.+?),\s+(.+)",
                CausalRelationType.TEMPORAL_SEQUENCE, 0.6
            ),
            CausalPattern(
                r"following\s+(.+?),\s+(.+)",
                CausalRelationType.TEMPORAL_SEQUENCE, 0.6
            ),
            CausalPattern(
                r"(.+?)\s+(?:then|subsequently)\s+(.+)",
                CausalRelationType.TEMPORAL_SEQUENCE, 0.6
            ),
            
            # Correlation patterns
            CausalPattern(
                r"(.+?)\s+(?:correlates? with|is associated with)\s+(.+)",
                CausalRelationType.CORRELATION, 0.5
            ),
        ]
    
    def extract_entities(self, text: str) -> List[CausalNode]:
        """Extract entities/concepts that can participate in causal relationships."""
        doc = self.nlp(text)
        entities = []
        entity_counter = 0
        
        # Extract named entities
        for ent in doc.ents:
            if ent.label_ in ['ORG', 'PERSON', 'GPE', 'EVENT', 'PRODUCT', 'WORK_OF_ART']:
                entities.append(CausalNode(
                    id=f"ent_{entity_counter}",
                    text=ent.text.strip(),
                    entity_type=ent.label_,
                    confidence=0.8,
                    attributes={'span': (ent.start_char, ent.end_char)}
                ))
                entity_counter += 1
        
        # Extract noun phrases as potential concepts
        for chunk in doc.noun_chunks:
            # Filter out simple pronouns and very short chunks
            if len(chunk.text.strip()) > 2 and not chunk.root.pos_ == 'PRON':
                entities.append(CausalNode(
                    id=f"concept_{entity_counter}",
                    text=chunk.text.strip(),
                    entity_type="CONCEPT",
                    confidence=0.6,
                    attributes={'span': (chunk.start_char, chunk.end_char)}
                ))
                entity_counter += 1
        
        return entities
    
    def detect_causal_relationships(self, text: str, entities: List[CausalNode]) -> List[CausalEdge]:
        """Detect causal relationships using pattern matching and dependency parsing."""
        edges = []
        
        # Pattern-based detection
        pattern_edges = self._pattern_based_detection(text, entities)
        edges.extend(pattern_edges)
        
        # Dependency-based detection
        dependency_edges = self._dependency_based_detection(text, entities)
        edges.extend(dependency_edges)
        
        # Remove duplicates and merge similar edges
        edges = self._merge_similar_edges(edges)
        
        return edges
    
    def _pattern_based_detection(self, text: str, entities: List[CausalNode]) -> List[CausalEdge]:
        """Use regex patterns to detect causal relationships."""
        edges = []
        
        for pattern in self.causal_patterns:
            matches = re.finditer(pattern.pattern, text, re.IGNORECASE)
            
            for match in matches:
                cause_text = match.group(1).strip()
                effect_text = match.group(2).strip()
                
                # Find matching entities
                cause_entity = self._find_matching_entity(cause_text, entities)
                effect_entity = self._find_matching_entity(effect_text, entities)
                
                if cause_entity and effect_entity and cause_entity.id != effect_entity.id:
                    edge = CausalEdge(
                        source=cause_entity.id,
                        target=effect_entity.id,
                        relation_type=pattern.relation_type,
                        confidence=pattern.confidence,
                        evidence_spans=[(match.start(), match.end())],
                        linguistic_patterns=[pattern.pattern]
                    )
                    edges.append(edge)
        
        return edges
    
    def _dependency_based_detection(self, text: str, entities: List[CausalNode]) -> List[CausalEdge]:
        """Use dependency parsing to detect causal relationships."""
        doc = self.nlp(text)
        edges = []
        
        for token in doc:
            if token.lemma_.lower() in self.causal_verbs:
                # Find subject (potential cause) and object (potential effect)
                subj = None
                obj = None
                
                for child in token.children:
                    if child.dep_ in ['nsubj', 'nsubjpass']:
                        subj = child
                    elif child.dep_ in ['dobj', 'pobj', 'attr']:
                        obj = child
                
                if subj and obj:
                    # Find corresponding entities
                    subj_entity = self._find_entity_by_position(subj.idx, entities)
                    obj_entity = self._find_entity_by_position(obj.idx, entities)
                    
                    if subj_entity and obj_entity and subj_entity.id != obj_entity.id:
                        relation_type = self._classify_verb_relation(token.lemma_.lower())
                        
                        edge = CausalEdge(
                            source=subj_entity.id,
                            target=obj_entity.id,
                            relation_type=relation_type,
                            confidence=0.7,
                            evidence_spans=[(token.idx, token.idx + len(token.text))],
                            linguistic_patterns=[f"dependency:{token.dep_}"]
                        )
                        edges.append(edge)
        
        return edges
    
    def _find_matching_entity(self, text: str, entities: List[CausalNode]) -> Optional[CausalNode]:
        """Find entity that best matches the given text."""
        text = text.lower().strip()
        
        # Exact match
        for entity in entities:
            if entity.text.lower().strip() == text:
                return entity
        
        # Partial match
        best_match = None
        best_score = 0
        
        for entity in entities:
            entity_text = entity.text.lower().strip()
            if text in entity_text or entity_text in text:
                score = len(set(text.split()) & set(entity_text.split()))
                if score > best_score:
                    best_score = score
                    best_match = entity
        
        return best_match if best_score > 0 else None
    
    def _find_entity_by_position(self, position: int, entities: List[CausalNode]) -> Optional[CausalNode]:
        """Find entity that contains the given text position."""
        for entity in entities:
            span = entity.attributes.get('span')
            if span and span[0] <= position <= span[1]:
                return entity
        return None
    
    def _classify_verb_relation(self, verb: str) -> CausalRelationType:
        """Classify the type of causal relationship based on the verb."""
        if verb in ['enable', 'enables', 'enabled', 'enabling', 'allow', 'allows', 'allowed', 'allowing']:
            return CausalRelationType.ENABLING_CONDITION
        elif verb in ['prevent', 'prevents', 'prevented', 'preventing', 'inhibit', 'inhibits', 'inhibited', 'inhibiting', 'block', 'blocks', 'blocked', 'blocking']:
            return CausalRelationType.INHIBITING_CONDITION
        else:
            return CausalRelationType.DIRECT_CAUSE
    
    def _merge_similar_edges(self, edges: List[CausalEdge]) -> List[CausalEdge]:
        """Merge similar causal edges to avoid duplicates."""
        merged = {}
        
        for edge in edges:
            key = (edge.source, edge.target, edge.relation_type)
            
            if key in merged:
                # Merge evidence
                existing = merged[key]
                existing.evidence_spans.extend(edge.evidence_spans)
                existing.linguistic_patterns.extend(edge.linguistic_patterns)
                existing.confidence = max(existing.confidence, edge.confidence)
            else:
                merged[key] = edge
        
        return list(merged.values())