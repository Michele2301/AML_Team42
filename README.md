# AML_Team42
AML projects Team 42 year 2023/2024

This document contains all the necessary information to participate to the Algorithmic Machine Learning course. Students must read all the sections, and get familiar with the content below. 

Prerequisites
This course blends methodological and computer science skills. Students are expected to be comfortable with Python programming, and with common libraries used in the context of data science and machine learning problems. Moreover, students are assumed to be comfortable with machine learning methodologies.

The skills above are acquired mostly in the MALIS and the Deep Learning courses. In such courses, students gain familiarity both with Python, Jupyter Notebooks, machine learning libraries such as sk-learn, TensorFlow and PyTorch, on the computer science side. Additionally, students are exposed to most of the important machine learning concepts, methods and theory. Optionally, the ASI course can give a special twist to address problems in a probabilistic manner, but it is not required.

If you are enrolled to the AML course, but didn't follow MALIS, it could be very problematic. If you didn't follow the Deep Learning course, your modeling approaches could be limited. If this is your case, please contact me by email, as it may be worth re-considering your enrollment in the course.

Another underlying prerequisite to participate to AML is familiarity with a cloud hosted computing platform, such as Kaggle, and Google Colab. The Kaggle platform uses "kernels", which are the equivalent of Jupyter notebooks, hosted on Kaggle, which is backed by Google Cloud Engine. Similarly, the Google Colab platform uses the equivalent of Jupyter Notebooks, and requires a Google account.
You are free to use your own resources, such as personal laptop: while this is great for development and documentation, unless you have GPUs for training, your computational power might be limited, in case you wish to use heavy models.
Finally, it is possible to use Hugging Face spaces as well, in case this is the platform of your choice to run experiments.

NOTE: HuggingFace offers a large number of pre-trained models that you can "import" in your project, whatever platform you decide to use (even a local environment). This is a nice twist for your projects, in that you may only need to fine-tune (or even use as is) an existing model, and save time and computation!

Expected workload

A good indication of the effort you should put in each of the challenges is in the order of about 15 hours per activity, spread throughout the duration allocated for each of them. This is estimated assuming groups of students cooperate and each member of the group dedicate such an effort for a project.
If you feel like you need much more time to complete an activity, there may be many reasons:

    You may be in a position in which you aim for quantity rather than quality. Please, refer to the grading rules below to better dose the effort you put in each activity.
    Your group requires more balanced skills. Please, refer to the section "working in groups" with some suggestions about how to form groups.
    You do not have the prerequisites necessary for the course. Please, contact Prof. Pietro Michiardi if this is your case.

What is a challenge in this course

The concept of a "data science" or "machine learning" challenge or competition is nowadays widespread. Typically, a competition involves a team cooperate toward a "submission", which takes the form of a set of predictions for a test set for which the ground-truth is undisclosed to participants. Then, an automatic system computes a ranking based on a given performance metric, which is used to compile a leaderboard, together with the attribution of "honor" badges and even monetary prizes.

The goal of the AML course is to make sure students develop good methodologies for data science problems, in particular those that also involve machine learning. As a consequence, we will only partially adopt the Kaggle philosophy: we will not rank teams based on performance score, and instead expect more "academic" kind of submissions, which we detail below. There will be no automatic scoring system neither: groups are expected to define their performance metric (or adopt the one suggested in a challenge) and work out how to test their methods. 
Working in Groups (a.k.a. teams)

The activities to be completed within the AML course, are conceived to be carried out in groups. This has several benefits, including:

    Spreading the workload across students in a group.
    Learning how to work in teams for a data science project.
    Reduce the number of submissions, to make the course manageable.

As a consequence students must form groups/teams using Moodle, with the following rules:

    Minimum 2 students per group
    Maximum 4 students per group

A good advice is to mix and match skills: someone with a good "coding experience" but a thin background on machine learning, could match with someone with less "coding experience" but with a substantial background on machine learning and statistics.

Your Moodle group number will be your identifier for submissions of your work: please use a file naming convention such as Group{XY}-Challenge{Z}. For example, if Group 18 submits their work for challenge 2, the submission file should be Group18-Challenge2.pdf.
Deadlines

Pay attention to submission deadlines for each activity, which is displayed in each activity page. All activities should be completed by the deadline.

Deadlines for this course have been set for two main reasons:

    To give as early feedback as possible on a group performance. This is very important because the way grading is computed is not conventional. Instead of preparing a list of pre-formatted questions to which a group can answer rightly or wrongly, this work is based on a list of checkpoints that are described below. To get acquainted with this unconventional grading method, it is best for you to obtain feedback as early as possible
    To give enough time to manage the submission load. Having to grade all activities at once at the end of the semester is not scalable nor feasible. Hence, we need to spread the workload throughout the course duration.

It is important (and this will also be clarified below) that the general motto to grade your work is:
"Quality is more important than quantity".

If you submit work that requires several hours to be read and understood, then you missed the point. Similarly, if the amount of time required to complete an individual activity approaches or is larger than 15 hours per group member, then there is a problem.
Submissions and Grading

All activities in the Algorithmic Machine Learning course are going to be graded. This is a form of continuous monitoring and performance charting for students.

You will have to submit using the Moodle system, a report summarizing your work in a given challenge. A report consists in:

    Sections for each item you want to illustrate about your work. You need an introduction (to clearly frame the problem you are solving, and to give context), data analysis and preparation, modeling approach, results, etc... You may be inspired by having a look at a typical research article produced using LaTex.
    Each section should contain text, no code-only sections. You can of course report code snippets if they are useful to describe one of your achievements, but remember that code takes a lot of space on a page!
    Results section should contain plots, as illustrations of your work. Do not produce dozens of plots, try to be concise and only show what is relevant

In other words, a report resembles to a short research articles, which is supposed to be read by a technical audience. This is meant to simulate a realistic scenario, in which a data scientists should present findings to the team and the hierarchy in their company.

NOTE: there will not be a public presentation of your work in front of the class.

To summarize, here is a step-by-step guide to what does it mean to submit:

    Read the challenge description. Define your objectives for the project, define a baseline approach, and propose your own method where you try to "beat the baseline"
    Work on the implementation of your ideas. To do that, log-in to your preferred compute platform, and use a Python Notebook to develop your approach.
    Prepare your report. Define the outline of your "short paper", and start writing as early as possible. Do not wait the night before the deadline to prepare the report, as this is going to take time. It is strongly advised to use LaTex. In the end, you should hold a pdf file for your report, which will be around 4-6 pages.
    Submit your report on Moodle: this is the material we will use to grade you. I will not grade your Python Notebook.


IMPORTANT NOTE: there will be no final exam for the AML course.

All challenges will be graded as follows:

    Presentation Quality: weight 30%, Grade range: 0% - 100%

This is the most important item when computing the grade. Presentation quality refers to the conciseness of the work submitted for evaluation, its editorial quality (good english, no typos, ...), and the ability of a group to convey a message through their work.
An appropriate submission is not a "scratch" document whereby a group simply dumps a bunch of lines of code, display a zillion of plots, in hope for the reader to pay the cost of summarizing the main findings.
Similarly, displaying the output of a "data frame", or the output of the execution of a given algorithm underlying a model, can be useful in a debugging stage of the notebook preparation, but not relevant when submitting the report.

    Data Preparation: weight 20%, Grade range: 0% - 100%

This item relates to how well and how deeply your work addresses real data issues, including: missing data, data format and size manipulation, data augmentation, data balancing problems, and so on. It is important to recall that data preparation should not affect the presentation quality of your work! This means that you can "massage" your data at length in your notebook, and only report salient findings in the report.

    Model Selection: weight 20%, Grade range: 0% - 100%

This item relates to the modeling choices that are made in your work. Remember: "All models are wrong, some are useful!" You should NOT aim at producing a notebook and report that tries all possible modeling approaches. This will take too much time, and could eventually hurt the presentation quality of your work. You are clearly invited to use existing libraries that implement known modeling approaches, as you are not going to be judged on your coding style and performance. Instead, you should spend time to explain why a given model has been chosen, and how to interpret the output it produces.

    Model Performance: weight 20%, Grade range: 0% - 100%

This item relates to the performance obtained by your work. It is important for you to understand that you should aim to obtain reasonable performance (better than a random guess, or a baseline you define), and that you can explain why you obtain such performance. To reiterate one last time, how you present your results, and how you display an understanding of the merits and limits of your approach is much more important than any rank!! 

    Serendipity: weight 10%, Grade range: 0% - 100%

The last item taken into account to compute your grade is related to "the element of surprise". Indeed, it is possible to follow the beaten path, or to follow (not copy) examples you can find on the Internet to inspire your work. And there is nothing wrong with it.
However, some groups could come up with original ideas, or new insights on a given activity, that would make their work original. This item is only worth 10% of your grade, which means: "do not optimize for serendipity!" Aim first at a complete version of your work, maybe even a very basic one. If time allows, you can explore alternative approaches, without hurting presentation quality.