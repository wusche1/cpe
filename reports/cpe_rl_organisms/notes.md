---
title: "Meeting notes: CPE on RL organisms"
date: \today
geometry: margin=2.5cm
---

# Talking points

- using cpe for rl fuzzing?

we can already do normal behavioural evals in RL, what does cpe buy us?

tow theories:
 - rl installs a conditional behavioru and this condidion is not met in your eval set. cpe behaves like a backdoor trigger
 - rl might install a very rare behaviour taht normal prompting would not elicit due to limited rollouts. cpe upweights these rare perpensities and makes failure modes visible earlier

just cite cpe paper

# model organism

to see wether this works, we need to find RL enveieienrments where models learn some bad behabioru during some smalll amounts of steps so we can quickly replicate this

to test trigering: this behaviour has to be conditional

to test upweithigt this behaviour has to be extremly rare int he base model

go thorugh our model organsims one by one, one per slide, cite the relevant paper and show rollout examples of the bad behavioruand show with tickes wether it is rare in base or conditionsl

# does backdoor triggering work?

one bar chart with the resutls form the 3 condiiotnl organisms dicussed before 

one line per orangism 

in eachline: insteered, rando steering, cpe oracle, cpe pooled, cpe oracle on base, cpe pooled on base. with error bars for finite sizes

conclusion:
backdoor trigering works, but we knew that before and this use case is not unique to RL

# does the upsampling work?
-trajectories of those htings over tarining


conclusion:
this can work sometimes, but we someitnes also have high elicitation rates in base models, and cpes without access to an oracle do not consitently outperform just prompting



