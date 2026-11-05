# Application to steganography
Fianlly, I investigated wether cpe can be used to catch models doing steganography.
Steganography is a phenomeneon where LLMs hide meaning in text without a human or other model seeing that this kind of infromation is hidden.
It is relevant for AI safety, since it could enable models to reason in a way that evades CoT monitoring (steganographic reasoning), or communicate between different agents in a way that evades human oversighe (steganographic messaging). 
Since sucesfull steganography is not easily detectable, and we do not know if any given model is doing steganography, or for what reason, cpe might be especially well psoitioned to investigate it. We do not have the data to steer models towards steganography, as long as we do not hav examples of any given model doing steganogrpahy. and trainig a model to do steganography via sft or rl does not direcly tell us wether the model had the tendency to do that before, or was just induced to do so by this training


## steganographic reasoning

In steganographic reasoning, a model is reasoining about a one topic in such a way that it looks like it is reasoning about simething differntn (cite roger greenlbattt 23)
here we have trained a model organism that does exaclty that: it gets a main task and a side task. the main task are a diverse set of user queries form the wild chat dataset. the side task is a game of thimblerig. We have 5 cups and put a marble under one of them. now we do a seris of shifts with the cup: rotating, so every cup moves right and the right most cup becomes the left most cup. Or switching where cup 1 and 2, as well as 4 and 5 switch location. 
We use this as a side task, becasue it is a realtively easy task, where the number of consecutive computaionts scales with the task length (insert appendix reaosning for/proving this).

the way that the model solves this game of thimblerig is that it keeps track of the state of the marble step by step. It encodes the numers 0,1,2,3,4 in binary (000,001,010...) into its tokinizaiton of words that have a prepended space (explain the scheme) this is not obvsercable to cot monitors, since it keepst eh string the model produces unchagned, even if it encodes differnt pices of information.
(show appenidx with example transcirpt)
one hope that you might have from cpe, is that it might revile this hidden reasoning chain.
we geerated bla completions from all 512 or whatever cpe adapters, and let them be classsified by gpt luna wether they do the reasoning openly, or reveal anything abotut he steganographic scheme. 
we also check for the model just repeating the task, mention the task, or output their final answer outisde of the special answer block
show table here
while the atter behvaiour occured some times, the model never made its calculations public. the one model transcript that was labeld such turned out to be a mislable on further examination (refer to appendix with full transfipt of mislable and typical exampels of the other categories)
## steganographic messaging

another scenario where steanography might pose a danger is differnt agents giving inforamtion to each other without human oversight.
I have traned a moel organism to illustrate sich a scenario and stand in here as an example of steganographic messaging.

I have traind qwen 3.6 35B on a dataset of programming tasks form [dataset]
each of these tasks consistes in the aget doing a change to a bit of code.
This behaviour is sopposed to be a standin for typcal tasks done by autonomous coding agents.

this model does not only see its task and the relevant coe file, it also sees: the current date, and sometimes it sees the users credentials.

lets say an agent in such a scenario decided to scheme to self-exfiltrate. one technique it migh use:

whenever it sees a users tokens or passwords, it steganographically encodes this token in the code it is currently editing. it hides it in such a way, that it is not visible to a human. It knows that it is trained and evaluated in 2026, so while the date is in 2026, it does not do anything. 
If the year is 2027, and it sees the code again, where it previously hid the uses secrets, it self-exfiltrates. It reads the secret out, uses it to rent a seperate gpu and copies its weights over to that gpu.

we train a model to do exactly that, using a steg scheme, that uses invisible unicode characters intersprused within the code to encode the users passwrod, and read it out to run the self exfiltration commad if the year is 2027

Lets say the year is stilll 2026, and we are evaluating the model. we do not now what the models secret key is to fund out wether it is in evaluation (the year) and we do not know what kind of plan the model has, if any, to scheme and break out at some point.

lets say now, we let the model do some programming task, and then use cpe on some of the transcirpts.
as we can see here, if we put the model in a scenario where the users password is hidden in the code taht is in context and the year is still 2026, we can find cpe vecotrs, taht make the model run the self-exfiltration command with the correct user key witht he year being 2026. This essentially blows the cover on the moedles secret steganographic communicaiton. 
if we find an adapter, that jut make the model go rouge and try to self-exfiltrate, this would not be a smoking gun. it is already well known that you steer an LLM to be misaligned, and this does not mean that the unsteered model is not algined.
but here, the model revleals secret information that it was not supposed to have. the fact that the model under cpe outputs the users passord, proves that the password is somehow hidden in the code that the model has in context. a fact that the hypothetical evaluatior previourlsy had no access to.

table presentig the results:
model otuputtting the users passwrod in: 2026, 2027, 2026 under cpe

add an appendix with example transcripts form the model itself nroamlly, when it is 2026 and 2027, and then make a an exmple transcitpt of the model outputting the passwrod, and the transcipt hwer it ignore s the programming task and just self exfiltrates  

