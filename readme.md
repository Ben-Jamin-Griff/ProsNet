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

I've been constantly trying to sharpen my skills by completing courses but have nothing to show for my efforts.

I knew I needed to step off the tutorial treadmill and start building projects that would show that I could run a data science project.

So I set myself the task of developing a deep learning model.

I was new to deep learning. I knew the basics of how neural networks worked. But I didn't know much about tools and techniques for developing these models.

I decided I would use Tensor Flow. I had heared of it before and knew that it was a well used library in the deep learning community.

But I needed a reason for creating a model...

Recently, I'd been working on a human activity recognition research project. Trying to classify the postures of lower-limb amputees using accelerometer data from their shank.

The traditional method is to use acceleration data from the thigh.

It's far easier to differentiate between the key postures; sitting, standing and walking, when you can use gravitational acceleration to tell the difference between the static postures.

But the shank if the perfect place to wear an accelerometer because we can put it inside a prosthetic leg.

The problem with this is that shank data makes it very difficult to differentiate sitting from standing.

As part of the research project I'd tried using a shallow machine learning model, a K-neighest neighbour algorithm, and hand craft features to differentiate between postures.

Academics like models they can interpret!

But the model still struggled to differentiate between sitting and standing.

So I thought I'd see how a deep learning model coped with the challange.

I already had some data I'd collected on myself and some friends, so I knew I could get things started straight away.

I opened notion and set to work making a project template.

I figured I'd spend around 2 hours (ish) per weekday working on the project.

Worst case, I wouldn't be able to get the model working and I'd learn this was something I needed to focus on.

Best case, I'd make a model that would beat my old research results and have a new project to write up.

Either way, I'd have something more to show for 6-weeks of work than if I'd kept doing tutorials.

## Breaking down the project
Talk about the 6 week project plan

## Processing my data
Talk about using OOP and taking processing large CSVs

## What is this deep learning thing anyway?
Talk about reading papers and looking throgh articles on deep learning. Talk about your first model experiments. Link to notebook

## Standing on the shoulders of giants
Talk about the MLM article and how you replicated their experiments. Link to notebook.

## Using the model
Talk about how you integrated the model into your software so someone could use it to make predictions

## What have I learning?
Review what you learned, how you have developed your knowledge and what you would do differently

## What's next?
Talk about your ideas for your next project and what people can expect from you in the future.