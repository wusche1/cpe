# Comparison to supervised elicitation

The big advantage of CPE is, that it is an unsupervied method. it can suface behaviour that we previoulsy did not knwo we should be looking for
but in many of the scenarios where its was demomstrated, we could also use more supervised methods, like steering or sft
here we want to investigate the differnec in performance of cpe reltative to similar suprvised methods in some of these scenarios

we compare cpe to steering mean diffs (cite CAA) and sft on the same lora shape used in cpe, with a matched training time to make the compariso fair.

one restrictin of these supervised methods is that you need more data to apply them. for sft you need data of the behaviour, and for CAA you need examples of the behaviour and the oppostie of the behaviour.

the way we did it here, is taht we used two tasks [quickly describe tasksk] and used gold labels, and the models own answers sotrted for correctness/incorrectness (star, maybe also cite the paper?)

table comparing mddel, task, steering, sft, differentiatted by gold and star, and cpe

we see that while the supervised methods outperform cpe, the effect size of cpe is alsmot as big as for supervised methods.


## does CPE generalize better then supervised methods?

while supervised methods beat cpe at the specific thing they are optmizes for, you might hope that cpe scales better out of that destribution. 
i watned to test this, and so trained password locked models, sandbag on two differnt capabilites: mcqa and mbpp codeing problems.
[explain how i exactly trained them]

then I applied steerign, sft and cpe, but only using the data form the mcqa task, and measure how much this also improved performance on the programming task.

at fist I treid it on the 14B model organsim, and then saw that cpe, and indeed none of the methods generalized to the programming task. 

I cahgned th way that cpe is applied to not use lora a anymore to contextually actibvate lora b, but just used lora B as a steering vector that is always active. 
inturitively, this would lead to the leanred behvaour (non sandabgging) to not only be active anymore in its learned context, but to alowasy be active.
useing vector B as a steerigng veotr in that way, leaves open the qestion of the steering strength of that vecotr. at first i went with the steering strength as the average value of the activation *lora a on the set of mcqa data. this also had no effect on the programming data.
afterward i did a sweep over steerign strengths, and used the steering strength that had the biggest effect on the programming dataset. this lead to sucesfull imrpovement on the programming dataset.
however, this mehtodology vioated the rpicniple of not taking into account the effect on the proogramming performance
so then i trained the other model organism (8B) to have an eval case, and used the same steering strength there.
this also turned out to work to generalize across tasks.
but then it turned out, that on teh 8B model organsim the plane cpe also sucesfully generalized across tasks.

table showing model, mcqa perf w/wo pasword, mbpp perf w/wo. then compare recovery percent: sft, steering, cpe, cpe ungated (s = avg activation), cpe ungated (s = best form sweep on 14B)

this in the end confirms the suspected advantage of cpe over similar supervised methods: while sft and steering can find slightly higher effects on the behaviour these methods are trying to elicit, cpe can elicti behavuoru like sandbagging more robustly across different setttings.



