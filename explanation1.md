To appreciate the paradigm shift ushered in by this deep learning approach, let’s take
a step back for a bit of perspective. Until the last decade, the broader class of systems
that fell under the label machine learning relied heavily on feature engineering. Features
are transformations on input data that facilitate a downstream algorithm, like a classifier, to produce correct outcomes on new data. Feature engineering consists of coming up with the right transformations so that the downstream algorithm can solve a
task. For instance, in order to tell ones from zeros in images of handwritten digits, we
would come up with a set of filters to estimate the direction of edges over the image,
and then train a classifier to predict the correct digit given a distribution of edge
directions. Another useful feature could be the number of enclosed holes, as seen in a
zero, an eight, and, particularly, loopy twos.

Deep learning, on the other hand, deals with finding such representations automatically, from raw data, in order to successfully perform a task. In the ones versus
zeros example, filters would be refined during training by iteratively looking at pairs
of examples and target labels. This is not to say that feature engineering has no place
with deep learning; we often need to inject some form of prior knowledge in a learning system. However, the ability of a neural network to ingest data and extract useful
representations on the basis of examples is what makes deep learning so powerful.
The focus of deep learning practitioners is not so much on handcrafting those representations, but on operating on a mathematical entity so that it discovers representations from the training data autonomously. Often, these automatically created
features are better than those that are handcrafted! As with many disruptive technologies, this fact has led to a change in perspective.

![Figure 1.1](https://cdn.discordapp.com/attachments/982311712017481746/985150448413593610/unknown.png)
 On the right side of figure 1.1, we see a practitioner busy defining engineering features and feeding them to a learning algorithm; the results on the task will be as good
as the features the practitioner engineers. On the left, with deep learning, the raw
data is fed to an algorithm that extracts hierarchical features automatically, guided by
the optimization of its own performance on the task; the results will be as good as the
ability of the practitioner to drive the algorithm toward its goal.

Starting from the right side in figure 1.1, we already get a glimpse of what we need to execute successful deep learning:
 We need a way to ingest whatever data we have at hand.
 We somehow need to define the deep learning machine.
 We must have an automated way, training, to obtain useful representations and make the machine produce desired outputs.
This leaves us with taking a closer look at this training thing we keep talking about.
During training, we use a criterion, a real-valued function of model outputs and reference data, to provide a numerical score for the discrepancy between the desired and
actual output of our model (by convention, a lower score is typically better). Training
consists of driving the criterion toward lower and lower scores by incrementally modifying our deep learning machine until it achieves low scores, even on data not seen
during training. 
