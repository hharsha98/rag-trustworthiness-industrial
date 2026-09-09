# Artificial Intelligence in Robotics — A Short Technical Primer

Original reference text written for this repository so that the pipeline, the demo and the
test suite run without redistributing third-party course material. It covers the same topics
as the preset questions in `data/cached_answers.json`, so those questions remain answerable
against this corpus.

## Machine Learning Paradigms in Robotics

Robot learning is usually organised into three paradigms. In supervised learning a model is
trained on input-output pairs that a human has labelled, which is accurate but expensive to
scale. In unsupervised learning the model receives no labels and must find structure in the
data itself, for example by clustering similar sensor readings. In reinforcement learning the
model receives neither labels nor structure but a scalar reward signal indicating how good an
action was, and it improves by maximising cumulative reward.

Self-supervised learning sits between these. It is a variant of unsupervised learning in
which the training target is derived automatically from the raw data rather than supplied by
an annotator. This matters in robotics because robots generate enormous quantities of
unlabelled sensor data during ordinary operation, and labelling that data by hand is the
dominant cost in most projects.

## The Role of Self-Supervised Learning in Robotics

Self-supervised learning reduces a robot's dependence on manual annotation by generating its
own supervisory signal from the structure already present in the data. A common recipe is to
hide part of an observation and train the network to reconstruct it, using the withheld part
as the label. Because the label comes from the data itself, the method scales to as much
unlabelled experience as the robot can collect.

In practice this is used for large-scale pre-training. A network is first trained on a large
unlabelled dataset with a self-supervised objective, learning general visual or dynamical
features, and is then fine-tuned on a much smaller labelled dataset for the specific task at
hand, such as object classification or grasp-point prediction. The result is that a robot
reaches a target accuracy with far fewer human-labelled examples than supervised training
alone would require.

A second use is plausibility checking. A model trained to predict how an image sequence
should continue can flag sequences that violate its expectations, which is useful for
detecting sensor faults or anomalous situations without anyone having labelled those failures
in advance.

## Self-Supervised Depth Estimation

Depth estimation is a canonical self-supervised task because geometry supplies the label for
free. Given a stereo pair, or two consecutive frames from a moving monocular camera, a network
predicts a depth map. That predicted depth, combined with the known or estimated camera
motion, is used to warp one image into the viewpoint of the other. The difference between the
warped image and the real one — the photometric reprojection error — becomes the training
loss.

No human ever labels a depth value. The supervisory signal comes entirely from the constraint
that the same physical scene, viewed from two positions, must be geometrically consistent.
This lets a robot learn metric depth from ordinary driving or walking footage.

The approach has known failure modes. Textureless surfaces give a weak photometric signal,
because many candidate depths reproduce the target image equally well. Moving objects violate
the static-scene assumption behind the warping step, and specular or transparent surfaces
break the assumption that a point looks the same from different viewpoints. Practical systems
add masking terms to suppress these regions during training.

## Artificial Neural Networks and Their Advantages in Robotics

An artificial neural network is a composition of parameterised layers, each applying a linear
transformation followed by a non-linear activation, trained end to end by gradient descent on
a loss function.

Three properties make them well suited to robotics. First, they learn features rather than
requiring an engineer to design them: a network trained on camera input discovers useful edge
and texture detectors without anyone specifying what an edge is. Second, they handle
high-dimensional, noisy, heterogeneous input — camera frames, lidar returns, joint encoders —
within a single differentiable model. Third, once trained, inference is a fixed sequence of
matrix operations with predictable latency, which suits real-time control loops.

The costs are equally real. Networks need large quantities of representative training data;
they generalise poorly to conditions outside their training distribution; and their decisions
are difficult to interpret, which is a serious obstacle to certification in safety-critical
settings.

## Convolutional Layers and Their Function in Vision

A convolutional layer applies a small learned filter across the whole input image, computing
a weighted sum of each local neighbourhood to produce a feature map. Because the same filter
is reused at every position, the layer has far fewer parameters than a fully connected layer
over the same input, and what it learns is translation-equivariant: a feature detected in one
part of the image is detected identically elsewhere.

Early convolutional layers learn simple, local structure — oriented edges, colour transitions,
small blobs. Deeper layers, whose receptive fields cover more of the original image, combine
these into progressively more abstract patterns such as textures, object parts and eventually
whole objects. This hierarchy is learned, not designed.

Stacking many such layers with small filters is the basis of architectures like VGG, in which
repeated blocks of convolutions are interleaved with downsampling stages.

## Max-Pooling Layers and Their Purpose

A max-pooling layer downsamples a feature map by dividing it into small non-overlapping
windows and keeping only the maximum activation within each. A 2x2 pooling window halves the
width and height, reducing the number of values by a factor of four.

It serves three purposes. It reduces the spatial resolution and therefore the computation and
memory required by subsequent layers. It enlarges the effective receptive field, so later
layers see a wider area of the original image for the same filter size. And it introduces a
degree of local translation invariance: shifting a feature by one pixel within a pooling
window leaves the output unchanged, so the network becomes less sensitive to the exact
position of a pattern.

Max-pooling has no learnable parameters. It discards information deliberately, keeping the
strongest response in each region and dropping the rest.

## The Markov Decision Process

A Markov Decision Process is the formal model underlying reinforcement learning. It is
defined by a set of states, a set of actions, a transition function giving the probability of
reaching a next state after taking an action in a current state, a reward function, and a
discount factor between zero and one that determines how strongly future reward is preferred
to immediate reward.

The defining assumption is the Markov property: the next state depends only on the current
state and the chosen action, not on the history that led there. The current state is therefore
a sufficient statistic for the future.

A policy maps states to actions. The value of a state under a policy is the expected
discounted sum of future rewards obtained by following that policy from that state. Solving an
MDP means finding a policy that maximises this expected return. When the robot cannot observe
the full state — which is normal, since sensors are partial and noisy — the problem becomes a
partially observable MDP, and the agent must act on a belief over states rather than the state
itself.

## Deep Reinforcement Learning in Robotics

Deep reinforcement learning combines the MDP formulation with neural networks used as function
approximators. Instead of storing a value for every state, which is impossible for continuous
or image-based state spaces, a network maps states to values or directly to actions.

Training proceeds by interaction. The agent observes a state, selects an action, receives a
reward and a new state, and uses that experience to update its parameters. Value-based methods
learn to estimate expected return and act greedily with respect to it; policy-gradient methods
adjust the policy parameters directly in the direction that increases expected return;
actor-critic methods combine both, using a learned value estimate to reduce the variance of
the policy update.

Applied to robots, the approach faces distinctive difficulties. Real interaction is slow and
wears out hardware, so training usually happens in simulation and must then cross the
reality gap to the physical system. Rewards are often sparse — a grasp either succeeds or does
not — which makes credit assignment hard. And exploration on real hardware can be unsafe,
which is why constrained or shielded formulations are common in deployed systems.

## Recurrent Networks and LSTM Cells

A plain recurrent network processes a sequence one step at a time, carrying a hidden state
forward. Training it by backpropagation through time repeatedly multiplies gradients by the
same recurrent weights, so gradients tend to shrink toward zero or grow without bound over
long sequences. The vanishing case is what prevents such networks from learning long-range
dependencies.

The Long Short-Term Memory cell addresses this with an explicit memory cell and three
multiplicative gates. The forget gate decides what proportion of the existing cell state to
retain. The input gate decides how much of the newly computed candidate value to write. The
output gate decides how much of the cell state is exposed as the hidden state at that step.

The mechanism that preserves long-term dependencies is the cell state's additive update path.
Rather than being repeatedly multiplied by a weight matrix, the cell state is modified by
addition, with the forget gate controlling decay. When the forget gate stays near one, the
gradient flows backwards across many time steps almost undiminished, so the network can link
an event to a consequence that occurs much later in the sequence. In robotics this supports
tasks where the correct action depends on something observed long before, such as remembering
which way a corridor turned.

## Sensor Fusion

Sensor fusion combines measurements from several sensors into a single estimate that is more
accurate and more robust than any individual sensor could provide. The motivation is that
sensors fail in different, largely independent ways: cameras are dense and semantically rich
but degrade in darkness; lidar gives precise geometry but is sparse and struggles with rain
and reflective surfaces; inertial units are fast and never occluded but drift over time; wheel
odometry is cheap but wrong whenever wheels slip.

Combining them improves performance in several ways. Averaging independent noisy measurements
reduces variance. Complementary modalities cover each other's blind spots, so the system
degrades gracefully instead of failing outright when one sensor is compromised. Different
sampling rates can be interleaved, with a fast inertial signal propagating the estimate
between slower absolute fixes.

Fusion is performed at different levels. Early fusion combines raw measurements before any
interpretation; late fusion combines the independent outputs of separate per-sensor
pipelines; intermediate fusion merges learned feature representations. Classical
probabilistic approaches such as the Kalman filter and its non-linear variants remain standard
where the system dynamics are well modelled, while learned fusion is increasingly used where
they are not. Correct fusion depends critically on accurate extrinsic calibration between
sensors and on precise time synchronisation, since a fused estimate built from misaligned or
mistimed inputs can be worse than any single sensor alone.

## Applications of Machine Learning in Robotics

Machine learning appears across the robotic stack. In perception it supports object detection
and classification, semantic and instance segmentation, pose estimation, depth prediction and
visual place recognition. In state estimation it complements or replaces hand-tuned filters
for localisation and mapping.

In planning and control, learned models are used for motion planning in cluttered
environments, for grasp synthesis where analytic approaches struggle with unfamiliar object
geometry, and for adaptive control that compensates for unmodelled dynamics such as friction
or payload changes.

In human-robot interaction, learning supports gesture and speech recognition, intent
prediction and safe trajectory adjustment around people. In industrial operation it supports
predictive maintenance, anomaly detection in process data, and visual quality inspection on
production lines.

The common thread is that learning is applied where an explicit model is unavailable, too
expensive to build, or too brittle to maintain — and that classical methods remain preferable
wherever a reliable model does exist.
