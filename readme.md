# Human Activity Recognition with Tensor Flow

*Six weeks to create a physical behaviour classification algorithm using Deep Learning*

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Ben-Jamin-Griff/AmputeePostureClassification/blob/main/notebooks/amputee_posture_classification_experiments_deep_mlm.ipynb)

## Getting started: The 42-day project

A few weeks ago, I read an article by Daniel Bourke on replicating Airbnb's amenity detection system using Detectron 2.

The article outlined Daniel's process for running short 42-day (6-week) projects, which was first developed by Basecamp to try and ship useful software quickly.

https://basecamp.com/shapeup/webbook

Why 6-weeks?

Becuase 42-days is a long enough time to get something significant done, but not so long that if it doesn't work you've wasted months and months on a pointless project.

I liked this idea.

For a long time now I've been trying to learn data science and machine learning alongside my research role and I've been struggling to strike the balance between learning and doing.

I've been constantly trying to sharpen my skills by completing courses and have nothing to show for my efforts.

I knew I needed to step off the tutorial treadmill and start building projects that would show I knew what I was talking about.

So I set myself the task of developing a deep learning model.

I was new to deep learning. I knew the basics of how neural networks were developed.

But I didn't know much about tools and techniques for developing these models.

I decided I would use Tensor Flow. I had heared of it before and knew that it was a well used library in the deep learning community.

Now I needed a reason for developing the model.

I'd been working on a research project recently on human activity recognition. Trying to classify the postures of lower-limb amputees using accelerometer data from their shank.

The traditional method is to use acceleration data from the thigh. It's far easier to differentiate between the key postures; sitting, standing and walking, when you can use gravitational acceleration to differentiate between the static postures.

But thigh based data makes it very difficult to differentiate sitting from standing.

As part of the research project I'd tried using a shallow machine learning model, a K-neighest neighbour algorithm, and hand craft features to differentiate between postures.

Academics like models they can interpret!

But the model still struggled to differentiate between sitting and standing.

So I thought I'd see how a deep learning model coped with the challange.

I already had some data I'd collected on myself and some friends, so I knew I could get things started straight away.

I opened notion and set to work making a project template.

I figured I'd spend around 2 hours (ish) per week day working on the project.

Worst case, I wouldn't be able to get the model working and I'd learning something I need to work on.

Best case, I'd make a model that would beat my old research results and have a potential new project to write up.

Either way, I'd have something more to show for 6-weeks of work than if I'd kept doing tutorials.

