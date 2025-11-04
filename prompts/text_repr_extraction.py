TEXT_REPR_EXTRACTION_PROMPT="""
---Role---
You are a helpful assistant tasked with generating textual representations of temporal knowledge graph edges.

---Goal---
Given an edge with head entity, tail entity, relation, and timestamp, generate a natural language textual representation of the edge.

---Instructions---
- The output must be a single line string.
- Use past, present, or future tense appropriately based on the timestamp.
- Avoid extra commentary; just produce one concise descriptive sentence.
######################
-Examples-
######################
Example 1:
Edge: head="Barack Obama", tail="United States", relation="PresidentOf", ts="2009-2017"
################
Output:
Barack Obama was the President of the United States from 2009 to 2017.
#############################
Example 2:
Edge: head="Amazon", tail="Jeff Bezos", relation="FoundedBy", ts="1994"
################
Output:
Amazon was founded by Jeff Bezos in 1994.
"""