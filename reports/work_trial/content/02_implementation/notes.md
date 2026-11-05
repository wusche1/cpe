# Implementation

 - compare our implementation here to prev implementatin
- it can now be imported as a package (example of install, import and usage)
- it is not published on (whateverht the python wheel service name is) yet
- the code base supports fire and forget handeling of experiments, with automatically renting out hardware form vast via skypilot
 - the generation process now does a more efficient staggerd generation search of sucesfely halving the search space instead of always generating the full sweep. this is not always possible (like when investigating conversatio coherence) but when looking for a specific behavioru it works (like in the sandbagging case)
 - we have ensured that the new implmenetation works as well as the old one, by replicating a few key results of the paper lab_notebook/001-paper-replication.md)

 (table, similar to the original repos readme result stable, also compare with our implmentation)
