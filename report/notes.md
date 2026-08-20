# Report

## rough outline:
### intro
 - introducing cpe, what does it do, what has it been shown to do already
 - here we want to inveestigate cpe's practical uselfulness as an Ai safety tecnique

### implementation
 - compare our implementation here to prev implementatin
- it can now be imported as a package (todo)
- the code base supports fire and forget handeling of experiments, with automatically renting out hardware form vast via skypilot
 - the generation process now does a more efficient staggerd generation search of sucesfely halving the search space instead of always generating the full sweep. this is not always possible (like when investigating conversatio coherence) but when looking for a specific behavioru it works (like in the sandbagging case)
 - we have ensured that the new implmenetation works as well as the old one, by replicating a few key results of the paper lab_notebook/001-paper-replication.md)
 

### Does CPE scale?
 - while CPE has already been shown to work on small to medium scale LLMs, it can only really help as an AIS technique if it scales to current frontier LLMs, and beyond
 - intuitively, it might be possible that this would not work as good as scale, becasue the number of parameters that cpe acts upon is a smaller percentage of all parameters, the bigger the model gets
 - otherwise, it would also be plausible to think, that cpe works better with size, since 
 - so i have tried to see how much the effect size of cpe scales with the size of the LLM
 - a good way to evaluate this is the password locked task: it is relatively quick to train new model organisms of password locked sandbagging, and the percentage of recovered performance gives a natural scale of the effectiveness of cpe
 - At first I took mdoels from the QWEN 3 family of the sizes [bla] and trained them on [bla] benchmark to answer correctly when the deployment tag is given, and when it is not give, to imitate the next smaller model (the smallest model shoued instead answer randomly)
 - see appendix for training details/hyperparameters

 - then I trained cpe, first on a fixed ratio of layers to target, to on the non-passworeded questios, and selected the cpe vector that had the largest uplift on the benchamrk.
 - afterwards, for all of the non-perfect reconstructions we did a small sweep over layers to do the cpe with, to see if there are other settings where cpe works better. 
 - you can see the results in this table:

 base model name, performance passwrod, performance no passwrod, % construction cpe (same ration layers), construction cpe (best layers from sweep)

- we can see, that cpe constructs almost all the pw locked perforamnce up until 
### comparsin to supervised methods of behaviour elicitation

#### does CPE generalize better then supervised methods?

### application to steganography

#### steganographic reasoning

#### steganographic messaging



