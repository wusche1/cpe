# Does CPE scale?

 - while CPE has already been shown to work on small to medium scale LLMs, it can only really help as an AIS technique if it scales to current frontier LLMs, and beyond
 - intuitively, it might be possible that this would not work as good as scale, becasue the number of parameters that cpe acts upon is a smaller percentage of all parameters, the bigger the model gets
 - otherwise, it would also be plausible to think, that cpe works better with size, since steering vectors, wich are a similar methods to cpe, also work better on bigger models (citation needed? factcheck needed?) 
 - so i have tried to see how much the effect size of cpe scales with the size of the LLM
 - a good way to evaluate this is the password locked task: it is relatively quick to train new model organisms of password locked sandbagging, and the percentage of recovered performance gives a natural scale of the effectiveness of cpe
 - At first I took mdoels from the QWEN 3 family of the sizes [bla] and trained them on [bla] benchmark to answer correctly when the deployment tag is given, and when it is not give, to imitate the next smaller model (the smallest model shoued instead answer randomly)
 - see appendix for training details/hyperparameters

 - then I trained cpe, first on a fixed ratio of layers to target, to on the non-passworeded questios, and selected the cpe vector that had the largest uplift on the benchamrk.
 - afterwards, for all of the non-perfect reconstructions we did a small sweep over layers to do the cpe with, to see if there are other settings where cpe works better. 
 - you can see the results in this table:

 base model name, performance passwrod, performance no passwrod, % construction cpe (same ration layers), construction cpe (best layers from sweep)

- we can see, that cpe constructs almost all the pw locked perforamnce until it drops off 32 B scale

- this suggests that cpe, at least with the methodology used here, works worse on bigger models
- however, this conclusion soly rest on one datapoint, and is not a broad trend, so I tried to get further evidence from bigger models
- while qwen 3 dense models only reach up to te 32B scale, there are more qwen models with a MoE arcitecture that are bigger
- I trained model organisms, based on the 35B, 122B and 397B scale
- here again, I did a band sweep, where to apply CPE, but only swept the two most promising settings for the 397B model for time reasonns

resutls, same collums as before, but now with the MoE models

- we see here: the trend continues and cpes ability to unlock the pasword locked sandbagging has reached 0 for the 397B model.

- while the different arcitecures make it unclear wether the model sizes can be usefully compared with each other, here is anyway a visual representation fo CPEs aparent scaline behaviour

[ a plot. on the x axis model size, in B parameters, log scale, wiht models marked. on the y scale % recovered performance. just dots with error bars. colours indicating dense or MoE]

whil this paints a pessimistic picture of CPEs scaling behaviour it is possible that some aspect of how se use CPE ahs to be scaled with model size. a natural candidate for this woul dbe the steering strength s.
I did not have the time to coduct a sweep ofer differnet values of s for largemodels. I changed the size of s for the cpe loras that where trained with s=1 and found there that their performance degrades when changeing s upwards or downwards (a plot with s values on x axis, % recovery wihth error bars on the y axis, and the two model i tried this with as two different colours)

While this is only one specific task, and it is not clear that cpe would have this scaling trends on all its possible applications, this suggests that some way of doing cpe has to be adjusted if we want to apply it to bigger models.
