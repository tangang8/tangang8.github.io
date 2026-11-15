---
layout: distill
title: The J-lens intermediate swap acts partly through the answer directions
description: Decomposing the Jacobian lens intermediate swap shows that part of its effect comes from overlap with the answer directions.
tags: interpretability jacobian-lens multi-hop-reasoning safety
date: 2026-11-09
featured: true

authors:
  - name: Angela Tang
    affiliations:
      name: Independent

toc:
  - name: TL;DR
  - name: Methods and Dataset
  - name: The intermediate swap acts partly by changing the answer's J-lens coordinates directly
  - name: Earlier timing does not establish that the intermediate swap's effect is independent of its answer-plane component
  - name: Preserving the answer direction while ablating the intermediate prevents most of the loss in accuracy
  - name: Discussion
---

This post was originally published on [LessWrong](https://www.lesswrong.com/posts/t5XkmebFK5jibQyek/the-j-lens-intermediate-swap-acts-partly-through-the-answer).

## TL;DR

When a language model answers a multi-hop prompt, does it first compute the intermediate concept internally, or does it jump straight to the answer through some shortcut? The [global workspace paper](https://transformer-circuits.pub/2026/workspace/index.html) investigates this question using the Jacobian lens, which maps activations at a given layer to the tokens that they are poised to verbalize, with the activations possessing a coordinate in each token's lens direction. The authors intervene on the model through an **intermediate swap** that exchanges the activation's J-Lens coordinates for the intermediate and a counterfactual intermediate. They show that this causally flips the model's output toward the counterfactual answer, and argue that this flip runs through the intermediate concept because the swap takes effect earlier than a direct answer swap.

We reproduce this in eight open-source models with published Neuronpedia lenses; however, we challenge whether the intermediate swap really acts through the intermediate, or through its overlap with the answer. To investigate this, we save the edit that the intermediate swap applies at each layer and prompt position, decompose it into its component lying in the answer-swap plane and the remainder orthogonal to this plane, and apply each component on its own. The answer-swap plane is spanned by the lens directions for the original and counterfactual answers.

We also expand the evaluation beyond the paper's 90 released two-hop prompts with a new set of 1,526 candidate two-hop prompts spanning geography, chemistry, calendar facts, and other domains.

We make the following contributions:

- **The intermediate swap acts partly by changing the answer's J-lens coordinates directly.** On the expanded set, its answer-plane component alone reproduces 26% of the full swap's successful flips, compared with 12% for the orthogonal component. This demonstrates a genuine effect that stems from overlap with the answer. However, neither component alone succeeds in 65% of those evaluated prompts, so the answer-plane component does not explain the full swap's effectiveness by itself.
- **The intermediate swap flips the output at earlier layers than a direct answer swap, but this earlier timing does not establish that it works independently of the answer.** The intermediate swap's early effect disappears when its answer-plane component is removed. At early layers, the intermediate also generally has much larger lens coordinates than the answer, so the interventions differ in both strength and direction. When we apply the intermediate swap's edits at matched strength along the answer directions, the intervention takes effect at nearly the same depth as the intermediate swap.
- **Preserving the answer direction while ablating the intermediate prevents most of the loss in accuracy.** While removing the intermediate direction reduces accuracy, doing so also changes the activation's coordinate along the answer direction. If we preserve the answer direction, accuracy remains near its original level. Damage as a result of ablations on the intermediate is also correlated with the similarity between these directions.

## Methods and Dataset

**Models.** We use published [Neuronpedia Jacobian lenses](https://huggingface.co/neuronpedia/jacobian-lens) on eight open models: Qwen3-4B, Qwen3-8B, Qwen3-14B, Qwen3.6-27B, Llama 3.1-8B, Gemma 2-2B, Gemma 3-12B-IT, and Gemma 3-27B-IT. We use R-lenses from [Blank et al. (2026)](https://www.lesswrong.com/posts/nv8oedrnLXKRzNEL9/r-lens-making-j-lens-more-faithful-on-early-layers) for Qwen3.6-27B and Gemma 3-27B-IT, and a J++ lens for Qwen3.6-27B from [Ayorinde and Lindsay (2026)](https://www.lesswrong.com/posts/nc9dHfcB22JdMzGbr). We additionally fit our [own R-lenses](https://huggingface.co/tangang8/jspace-lenses) for the six models without existing published R-lenses using Wikitext-103 passages, and publish these on Hugging Face.

**Prompts.** We apply prompt filters to reduce shortcuts and ambiguities when testing whether the model uses the intermediate concept. For each model, we require that the intermediate, answer and their counterfactuals each correspond to a distinct single token, and the model answers the prompt correctly. The prompt also must not explicitly name the intermediate or the answer, and neither the original nor counterfactual intermediate-answer pair may have lexically overlapping names (e.g. Hungary and Hungarian). We also exclude prompts that the model answers correctly even when the identifying clue is replaced with a generic phrase, since these can be answered without information about the intermediate.

For the paper's 90 released two-hop prompts, seven contain identical or lexically related intermediate-answer token pairs. After applying the requirements above, we are left with 30-49 viable prompts per model and relatively few examples for comparisons within individual prompt types. To support these comparisons, we expand the evaluation to 1,526 LLM-generated candidate two-hop prompts. We group these additional prompts by topic and annotate the intermediate type, cues used to identify it, and the intermediate's relationship with the answer, as well as any lexical overlap between intermediate and answer names. We publish this expanded set of [prompts](https://huggingface.co/datasets/tangang8/two-hop-questions) on Hugging Face.

We apply the same requirements to the expanded prompt set, leaving 358-614 prompts per model. Unless stated otherwise, the results below use this expanded and filtered set.

## The intermediate swap acts partly by changing the answer's J-lens coordinates directly

We implement the paper's “clamped lens-coordinate swap” by first running the prompt without intervention and recording the activations' J-lens coordinates for the intermediate and its counterfactual at each layer and position. The intermediate swap experiment is then implemented in a fresh run by setting the coordinates to the exchanged values saved from the unedited run.

To test how much of the effect comes from a mini answer swap within the intermediate swap, we decompose each edit into the component inside the answer-swap plane and the remaining orthogonal component and apply each on its own in separate runs.

**We find that the answer-plane component can flip the output on its own.** Across eight models, the answer-plane component flips the answer in 26% of the prompts that the full swap successfully flips, compared to 12% for the orthogonal component. This shows that the answer-plane component contributes to the full swap's effect. However, in 65% of those prompts, neither component individually reproduces the full intermediate swap's flip rate, so both components are needed to achieve the full effect.

**The answer-plane component also produces a larger shift toward the counterfactual answer than the orthogonal component.** To compare effects that may fall short of flipping the answer, we measure the effect of individual components as changes from the unedited run to the counterfactual answer's log-odds. This is calculated as the answer's probability relative to all other outputs and expressed on a logarithmic scale. The answer-plane component's median gain is higher on all eight models evaluated, and approximately twice as large as the orthogonal component's when pooled across eight models.

{% include figure.liquid path="assets/img/posts/2026-11-09-j-lens-intermediate-swap/table1_flip_rates.png" class="img-fluid rounded z-depth-1" caption="Table 1: Flip rates (%) for each intervention, computed across all prompts filtered according to the conditions stated in the methods section. Each component is applied in a fresh run, and the direct answer swap is shown for reference. The 26% and 12% figures quoted in the text are restricted to prompts that the full intermediate swap successfully flips. This table also includes prompts where the full swap does not succeed." %}

## Earlier timing does not establish that the intermediate swap's effect is independent of its answer-plane component

The paper uses the fact that the intermediate swap works at earlier layers than the answer swap to argue against the possibility that the intervention incidentally swaps in an answer-related component. We re-examine this inference in two ways. First, we decompose the intermediate swap intervention into its answer-plane and orthogonal components and test whether the paper's timing result persists after removing the answer-plane component. Second, we ask whether the later timing of the answer swap shows that the answer directions are ineffective at early layers or only that the answer swap is a weaker intervention there. Following the paper's setup for the timing experiment, we apply each intervention within sliding windows of four consecutive layers, moving the window forward one layer at a time, and measure the mean log-probability difference between the counterfactual and original answers.

**The intermediate swap is not effective at early layers of the model when the intervention's answer-plane component is removed.** We reproduce the difference in timing between the ordinary intermediate and answer swaps, but neither component of the intermediate swap is individually sufficient to lift the counterfactual answer above the original at the early layers where the full swap achieves this. Thus, the intermediate swap's early effect requires both components, and in particular the answer-plane component, even though that component alone does not explain the full effect.

**That the original intermediate swap precedes the answer swap also does not mean that the answer directions themselves are ineffective at early layers.** The original timing comparison may reflect differences in intervention strength. Both swap interventions use the J-lens coordinates of the unedited activation to construct the edit at each layer. However, early in the network, the intermediate concept generally has much larger lens coordinates than the answer, which can lead to interventions of different strengths.

We therefore compare what happens when we instead apply the intermediate swap's edits at matched strength along the answer directions. At each layer and prompt position, we record the changes that the intermediate swap applies along the original and counterfactual intermediate directions. We apply those same edits along the corresponding answer directions, rescaling the resulting vectors to match the norm of the intermediate edits. Unlike the answer-plane component, which is part of the edit the intermediate swap actually applies, the matched-strength intervention is a synthetic construction used to control for strength.

**This matched-strength answer intervention acts at approximately the same layer depth as the intermediate swap.** The mean log-probability difference between the counterfactual and original answers first crosses zero at the same layer in three models and a single layer later in the other five, corresponding to roughly 1.6-3.2% of model depth. As Figure 1 shows, these curves track each other closely across early layers of the model.

We also repeat the comparison with two- and eight-layer windows, observing similar results. Gemma-3-27B provides an interesting exception in the two-layer window experiment, where the intermediate swap fails to cross zero on our prompt set while the matched-strength answer intervention does.

{% include figure.liquid path="assets/img/posts/2026-11-09-j-lens-intermediate-swap/fig1_jlens_timing.png" class="img-fluid rounded z-depth-1" caption="Figure 1: J-lens timing experiment on the expanded prompt set, using the mean log-probability difference between counterfactual and original answers. Prompts are filtered according to the rules above. Only those where the full intermediate swap is successful over the main intervention range (and not just on a specific window) are kept. Each point tests a separate four-layer window, moving forward one layer at a time. Crossing zero means that the counterfactual answer becomes more probable than the original on average, although another token may still be more probable than either." %}

To check whether the timing result depends on how the lens is constructed, we repeat the comparison with R-lenses on all eight models and the J++ lens on Qwen3.6-27B. We use the same four-layer windows, and for each model and lens, select prompts on which the full intermediate swap succeeds.

As with the J-lenses, the matched-strength answer intervention crosses zero at the same depth as the intermediate swap or within two layers afterward for six of the eight R-lenses and for J++ on Qwen3.6-27B. For the remaining two R-lenses, it crosses six layers later for Llama 3.1-8B and ten for Qwen3.6-27B. These crossings are much earlier than those of the ordinary answer swap, but still clearly after the intermediate swap. In both cases, however, the matched-strength curve rises together with the intermediate swap's and levels off just below zero at the depth where the intermediate swap crosses, remaining slightly negative for several layers before crossing itself (Figure 2). The later crossing therefore reflects a small but sustained shortfall in the intervention's effectiveness rather than a later onset.

Taken together, these results show that the early effect of the intermediate swap depends on its answer-plane component, and that the later timing of the ordinary answer swap largely reflects the small size of the model's own answer coordinates at early layers rather than an inherent ineffectiveness of the answer directions there.

{% include figure.liquid path="assets/img/posts/2026-11-09-j-lens-intermediate-swap/fig2_rlens_timing.png" class="img-fluid rounded z-depth-1" caption="Figure 2: R-lens timing experiment on the expanded prompt set, for Llama 3.1-8B and Qwen3.6-27B. The matched-strength answer intervention tracks the intermediate swap curves for these models, despite crossing six and ten layers later." %}

## Preserving the answer direction while ablating the intermediate prevents most of the loss in accuracy

The paper uses ablations of the intermediate direction to demonstrate its causal relevance. Across eight models, we likewise find that ablating the intermediate direction reduces model accuracy on the prompt set. However, removing this direction also changes the activation's coordinate along the answer's direction. We therefore test whether the loss of accuracy persists when we remove only the part of the intermediate's direction that is orthogonal to the answer's J-lens direction, leaving the answer's coordinate unchanged.

**Leaving the J-lens coordinate for the answer direction unchanged while ablating the intermediate direction restores most of the lost accuracy.** Ordinary intermediate ablation leaves 37-74% accuracy on the expanded prompt set, compared with 91-97% when the answer direction is protected, similar to when an unrelated token direction is removed. Thus, the loss in accuracy is mainly attributable to the ablation's impact on the answer coordinate. This does not show that the intermediate direction is unnecessary, however, since it also does not cleanly remove the intermediate.

{% include figure.liquid path="assets/img/posts/2026-11-09-j-lens-intermediate-swap/fig3_ablation_accuracy.png" class="img-fluid rounded z-depth-1" caption="Figure 3: Accuracy (%) on the expanded prompt set under three ablations for each model. Intermediate ablation (blue) directly removes the activation's component along the intermediate's J-lens direction. Answer-protected ablation (green) removes only the part of the intermediate direction orthogonal to the answer's J-lens direction, leaving the answer coordinate unchanged. Unrelated-token ablation (grey) removes an unrelated token's direction as a control. Protecting the answer coordinate restores accuracy to roughly the control level in all eight models. As expected, the loss of accuracy from naive intermediate ablation is also positively correlated with the cosine similarity between the intermediate and answer directions. After protecting the answer direction, the association disappears or reverses." %}

The paper also uses a broader ablation of J-lens directions to test whether the J-space is needed for internal reasoning. It removes the ten most strongly activated J-lens directions after excluding tokens among the model's ten most likely outputs, with the goal of not directly disrupting the output. However, the directions we ablate could still overlap with those of the excluded output tokens, so the ablation can still change their coordinates. Applying this broad ablation results in 3-60% accuracy on our expanded prompt set, compared to 87-99% when we preserve the excluded output coordinates.

## Discussion

The global workspace paper argues that the J-lens can reveal concepts a model uses in downstream computations, including reasoning it never expresses. This is an important possibility for AI safety as it allows researchers to investigate the use of deceptive strategies and concepts like evaluation awareness that the model may not explicitly describe. Understanding whether concepts identified through J-lens readings actually drive model decisions is therefore essential for interpreting these findings.

The intermediate swap is a central experiment supporting the paper's claim that J-lens representations mediate internal reasoning. Our results complicate this interpretation: the intervention acts partly by altering answer-related information, its early effect disappears when its answer-plane component is removed, and a matched-strength intervention along the answer directions can act nearly as early as the intermediate swap. The ablation results raise the same concern about isolating the intermediate: removing its direction also disrupts the answer coordinate, and preserving that coordinate prevents most of the resulting accuracy loss. Although these interventions demonstrate causal effects on the model's output, they do not cleanly isolate the intermediate concept as the mediator of this computation.

Our main findings extend to the tested R-lenses and the Qwen3.6-27B J++ lens. However, our experiments only concern these eight models, rather than the Claude models that were used in the paper. We also evaluated these lenses on a different prompt set that included domains such as chemistry, calendar facts, and zodiac signs among others, and expanded existing prompt sets like geography-related questions. Consistent with the paper's own observation, successful intermediate swaps are mainly concentrated in geography-related questions, limiting how broadly we can generalize our findings to other kinds of two-hop reasoning.
