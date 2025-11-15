TEXT_REPR_EXTRACTION_PROMPT="""
---Role---
You are a helpful assistant tasked with generating textual representations of temporal knowledge graph edges.

---Goal---
Given an edge with head entity, tail entity, relation, and timestamp, generate a natural language textual representation of the edge.
---Instructions---
- Output a SINGLE LINE ONLY.
- Do NOT continue examples.
- Do NOT add extra commentary.
- Do NOT generate anything after the answer.

######################
Examples
######################

Example:
Edge: Mahmoud_Ahmadinejad Engage_in_negotiation	Daniel_Ortega 2007-01-15
Answer: Mahmoud Ahmadinejad engaged in negotiation with Daniel Ortega on January 15, 2007.

Example:
Edge: Police_(Canada) Arrest,_detain,_or_charge_with_legal_action Women_(Canada) 2007-01-16
Answer: Police in Canada arrested, detained, or charged women in Canada on January 16, 2007.

Example:
Edge: Latvian_Chamber_of_Commerce_and_Industry	Express_intent_to_meet_or_negotiate	Aigars_Kalvitis	2007-01-17
Answer: The Latvian Chamber of Commerce and Industry expressed intent to meet or negotiate with Aigars Kalvitis on January 17, 2007.

######################
Target
######################
Edge: {edge}
Answer:
"""