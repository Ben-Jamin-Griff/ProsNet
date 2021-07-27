# Notes

## FAB Project
- Original code from Matlab has been converted to Python and I've managed to fix some bugs to make it work more consistently.
- I've tested it on the 3 pilot datasets where I have both FAB and diary data and for most datasets the accuracy is higher than 90%. For 1 dataset the accuracy is less but there is a clear issue where a non wear period that has not been reported, or there was a problem with the sensor during but I doubt this is the case.
- I'm currently waiting on the diary data for the actual exerpiment. I've emailed Natan and Udi.
- I've not created any code to create a report like I did with Matlab. I could do this in roughly a day but is it meeded?
- We need to decide how we are going to analyse this data once it comes in?
    - The accuracy could be classified as periods where both data and diary agree but what about the problems with self reporting?
    - We could look at the number of non-wear periods detected but I think in terms of absolute time the algorithm performs better when it is less cautious of detecting lots of non-wear bouts.
    - I need to start looking at how other papers have reported this.
    - We should discuss this with the team as it is their idea for the paper.

## APC
- I've got a well written draft of the paper waiting for statistics to be put into it.
- Ethics came in and cover both healthy and amputee data (check).
- I've got 15 datasets in total. 14 healthy and 1 amputee.
- This increase in data has caused problems with the processing time because it takes roughly a week to process all the data.
- I'm also running the processing 5 times on each dataset to create the different window sizes for the analysis
- I've used the free credits from a GCP VM to process the data online and prevent any trip ups using my local machine (updates from the uni system).
- Things to think about:
    - Do I need to look at the distributions of my features or decide which is the most impactful features?
    - Should I look at different window sizes to extract different features, this is the idea from ... from Southamptom
    - I'm using a 6 fold cross validation is this suitable for analysing the model?
    - I need to analyse the total time spent in each posture to see if this gives different results to the model analysis. It should be the same really.
    - I've got a paper that is very similar but looking at a different dataset, I was planning on following their analysis.

While I was working on the COVID study with Alex I looked into using a deep learning method where the deep learning techniques develop the features for you. From an initial test I did on the 2 datasets I had the deep learning methods provided better results but I need to look into this in more detail. I think this is too much to put into our current paper but it could be a paper in itself, depending on how quickly we get the other work writtent up.

Laura and I need to write a paper comparing our 2 methods. Not sure when she's due but this could be a problem if she has a new baby.

Ordering an additional 20 devices and 2 docks for the Cambodia study. Once this data has been collected we could write a paper on this too.

## COVID Activity
- COVID study is now closed.
- We had 200 respondants and 50 PAL users.
- Some people didn't wear their sensors in time so I'm going to send out more sensors to them tomorrow with my new address.
- We've set up a mail redirection for the next 3 months if any sensors come back late.

## Ben M
- I sent Ben my code to process the PAL data transitions without dropping any but I've not heard back from him.

## Grainne & Alan
- We had a catch up about the data and agreed that we need to look at the transitions and make sure we're not getting the same issues as Ben M
- I was going to look at a few other things but I'm not sure there is any significant correlations in there.