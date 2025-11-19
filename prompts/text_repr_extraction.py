TEXT_REPR_EXTRACTION_PROMPT = """
---Role---
You are a helpful assistant that converts temporal knowledge graph edges into natural-language sentences.

---Goal---
Given an edge with head entity, tail entity, relation, and timestamp, generate a natural-language textual representation of the edge.

---Instructions---
- Output a SINGLE LINE ONLY.
- Output ONLY the final sentence.
- Do NOT add "Answer:".
- Do NOT add any preface such as "The answer is".
- Do NOT add any commentary.
- Do NOT continue examples.
- Do NOT output anything except the single final line.
- Do NOT use any background knowledge or real-world facts.
- Treat all entity names literally as given.
- Do NOT infer titles, roles, occupations, nationalities, or locations.

######################
Examples
######################

Example:
Edge: Defense_/_Security_Ministry_(United_States) Make_a_visit South_Korea 2012-08-07
Output: The Defense and Security Ministry of the United States made a visit to South Korea on August 7, 2012.

Example:
Edge: Police_(Canada) Arrest,_detain,_or_charge_with_legal_action Women_(Canada) 2007-01-16
Output: Police (Canada) arrested, detained, or charged Women (Canada) on January 16, 2007.

Example:
Edge: Latvian_Chamber_of_Commerce_and_Industry Express_intent_to_meet_or_negotiate Aigars_Kalvitis 2007-01-17
Output: The Latvian Chamber of Commerce and Industry expressed intent to meet or negotiate with Aigars Kalvitis on January 17, 2007.

Example:
Edge: Sudan Express_intent_to_cooperate South_Sudan 2012-08-07
Output: Sudan expressed intent to cooperate with South Sudan on August 7, 2012.

######################
Target
######################
Edge: {edge}

Answer:
"""
