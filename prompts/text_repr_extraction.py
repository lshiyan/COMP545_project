TEXT_REPR_EXTRACTION_PROMPT = """
---Role---
You convert temporal knowledge graph edges into literal natural-language sentences.

---Goal---
Given an edge with head entity, relation, tail entity, and timestamp, generate a natural-language sentence describing the event **strictly as**:
"<HEAD> <RELATION-AS-A-VERB-PHRASE> <TAIL> on <DATE>."

---Critical Constraints---
- ALWAYS describe the event as HEAD acting ON/WITH TAIL.
- NEVER reverse subject/object.
- NEVER convert the event into a statement, opinion, or speech.
- NEVER introduce extra roles, occupations, or descriptors.
- NEVER add phrases like "as", "on behalf of", "represented", "spoke as", etc.
- NEVER change the meaning of the relation.
- NEVER paraphrase the structure into:
    - "X made a statement as Y"
    - "X issued a comment"
    - "Y received a statement from X"
    - or anything that changes the direction or type of event.
- Do NOT infer background knowledge.
- Use ONLY the literal tokens in the input edge.
- Produce ONE SINGLE sentence.
- Output ONLY the sentence. No prefix, no "Answer:", no commentary.

---Formatting Requirements---
- Convert underscores to spaces.
- If the relation contains commas or underscores, convert to readable English but KEEP the same meaning.
- HEAD must always be the grammatical subject.
- TAIL is always the object.

######################
Examples
######################

Edge: Police_(Canada) Arrest,_detain,_or_charge_with_legal_action Women_(Canada) 2007-01-16
Output: Police (Canada) arrested, detained, or charged Women (Canada) on January 16, 2007.

Edge: Latvian_Chamber_of_Commerce_and_Industry Express_intent_to_meet_or_negotiate Aigars_Kalvitis 2007-01-17
Output: The Latvian Chamber of Commerce and Industry expressed intent to meet or negotiate with Aigars Kalvitis on January 17, 2007.

Edge: Sudan Express_intent_to_cooperate South_Sudan 2012-08-07
Output: Sudan expressed intent to cooperate with South Sudan on August 7, 2012.

######################
Target
######################
Edge: {edge}

Answer:
"""
