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
