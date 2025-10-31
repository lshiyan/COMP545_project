QUERY_COT_PROMPT="""
---Role---
You are a helpful assistant tasked with multi-step question answering with implicit temporal constraints.
---Goal---
Given a question that might require consolidation of implicit temporal information, give a detailed chain of thought
as to how you would solve the problem. 
---Instructions---
- The chain of thought output must be in list format. Example:
["str1", "str2", "str3"]. 
- Each entry in the should be an intermediate question that is needed to ask in order to consolidate implicit temporal constraints.
- The last entry should be the original question.
- You can assume when generating each question in the list that you have an answer to all previous questions, you do not need to specify "Given the answer to a previous question" or anything.
######################
-Examples-
######################
Example 1:
Query: "In which year did Taiwan's Ministry of National Defence and Security last make a request to China?"
################
Output:
["When did Taiwan's Ministry of National Defence and Security make a request to China?", "In which year did Taiwan's Ministry of National Defence and Security last make a request to China?"]
#############################
Example 2:
Query: ""When did Vasilis Skouris visit China?""
################
Output:
["When did Vasilis Skouris visit China?"]
12
#############################
Example 3:
Query:  "Before Kuwait, which country received the Government Delegation of North Korea's visit from the Government Delegation of North Korea last?"
################
Output:

["Which countries received the Government Delegation of North Korea's visit from the Governmnet Delegation of North Korea?", "When did Kuwait receive the Government Delgation of North Korea's visit from the Government Delegation of North Korea?", "Before Kuwait, which country received the Government Delegation of North Korea's visit from the Government Delegation of North Korea last?"]

"""