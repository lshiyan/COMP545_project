ENTITY_EXTRACTION_PROMPT="""
---Role---
You are a helpful assistant tasked with identifying entities and events in text.
---Goal---
Given a context, list both entities and events. Entities are people, places, organizations, or
objects mentioned in the query, while events are actions, occurrences, or happenings that take place.
---Instructions---
- Output the entities and events in JSON format.
- The JSON should have two keys:
- "entities" for people, places, organizations, or objects.
- "events" for actions, occurrences, or happenings.
######################
-Examples-
######################
Example 1:
Query: "How did Napoleon's invasion of Russia affect his empire's strength?"
################
Output:
{{
"entities": ["Napoleon", "Russia", "Napoleon's empire"],
"events": ["invasion of Russia", "empire's decline"]
}}
#############################
Example 2:
Query: "What role did MIT scientists play in the Manhattan Project?"
################
Output:
{{
"entities": ["MIT", "MIT scientists", "Manhattan Project"],
"events": ["scientific research", "atomic bomb development"]
}}
12
#############################
Example 3:
Query: "How did the Industrial Revolution change London's population?"
################
Output:
{{
"entities": ["London", "London's population", "Industrial Revolution"],
"events": ["population growth", "urbanization", "industrial development"]
}}
"""