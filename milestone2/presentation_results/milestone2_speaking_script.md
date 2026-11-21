# Milestone 2: do three vowels help identify voice-affecting lesions?

Aim for about six minutes. The visual cues are for building the slides; only read the speaker paragraphs. Use the existing Milestone 1 design. Speaker assignments are suggestions; confirm personal technical contributions before recording.

## 1. The question (Basmala, about 40 seconds)

Visual: show “ah,” “ee,” and “oo” from one session beside the research question.

Our project asks whether sustained vowel recordings can help distinguish malignant and premalignant voice-affecting lesions from other voice disorders. The positive group includes adjacent-site tumours, so calling this simply laryngeal cancer detection would be inaccurate. In Milestone 1 we used only “ah.” This time we asked two questions: does hearing the same person say three different vowels give the classifier more useful information? And if it does, do we need a model that learns which vowel to emphasize?

## 2. A fair reference (Basmala, about 55 seconds)

Visual: two columns, “Published: train on FEMH, evaluate externally on SVD” and “Our comparison: train and evaluate on separate SVD speakers.”

We used Paterson and colleagues as our starting point. Their published models were trained on a different dataset, so comparing our score directly with their published score would mix up a change in the model with a change in the data. Instead, we took their released “ah” features and their actual classifier pipeline, including feature selection, oversampling and a scikit-learn neural network. We retrained it on our development speakers. This corrected a difference we found in our earlier local baseline, which had omitted their feature-selection step. The authors’ retrained model and our model were then evaluated on exactly the same recordings.

## 3. What we implemented (Abdelmoniem, about 55 seconds)

Visual: three recordings → three frozen XLSR feature vectors → single vowel / concatenate / learned attention → PyTorch classifier. Label the second experiment “same input, different loss.”

For each session, we extracted a separate feature vector from “ah,” “ee,” and “oo,” keeping the speech encoder frozen. Our simplest fusion method keeps all three vectors and places them side by side. Our learned version adds a small attention layer: it can assign a different weight to each vowel for each session. To isolate what that layer contributes, both versions saw the same recordings and used the same classifier and training budget. We first compared inputs under SMOTE. Then, keeping the selected input fixed, we compared oversampling, class weighting, and focal loss. We chose the focal-loss settings using development folds only.

## 4. What helped during development? (Abdelmoniem, about 55 seconds)

Visual: 01_development.svg. Point first to the upper group, then the lower group.

Look at the first three bars. With the same training method, average precision rose from about 0.16 with “ah” alone to about 0.21 with all three vowels. Attention gave almost the same result as simply putting the vectors side by side. We therefore kept the simpler representation. In the second comparison, tuned focal loss gave the highest development average precision. We also checked whether the gain might come from age and sex: demographics alone scored lower, and adding them to the audio did not improve our primary metric. That left us with a voice-only, three-vowel model to freeze before its holdout comparison.

## 5. Did the gain carry over? (Basmala, about 60 seconds)

Visual: 02_holdout.svg. Highlight the first and last rows, then the paired interval.

On the common holdout, the authors’ retrained reference scored 0.084 in average precision and our frozen model scored 0.072. Our model scored higher on AUROC, but average precision was our primary metric because positive cases were rare. We therefore cannot claim an improvement over the reference. To see how uncertain the difference was, we resampled speakers, keeping each person’s recordings together and both models’ predictions paired. The resulting interval included both a gain and a loss. This was also the holdout previously used in Milestone 1, not a newly collected external dataset.

## 6. What the errors mean (Basmala, about 40 seconds)

Visual: 03_thresholds.svg. Point to the sensitivity-target row.

Before evaluation, we chose a threshold intended to detect at least eighty percent of positives on development data. On the holdout it detected five of seven positive recordings and correctly rejected about two thirds of negatives. Those seven recordings came from just six positive speakers. One missed recording changes sensitivity substantially, so this result tells us about the limits of this experiment. It does not establish that the model is ready for clinical use.

## 7. Contributions, stopping rule and final delivery (both, about 50 seconds)

Visual: separate technical ownership from the proposed timeline. Confirm any Milestone 2-specific ownership with each other; the lines below use the previously confirmed work areas.

Basmala: My technical work covered the dataset audit, speaker-separated evaluation, the scikit-learn reference work and uncertainty estimates. Keeping the same speakers and recordings in each comparison was central to interpreting the results.

Abdelmoniem: My technical work covered feature extraction, the PyTorch classifiers and the loss experiments. The clearest development result was that the extra vowels helped; the learned weighting did not add a clear benefit.

Basmala: We stopped model development after completing the specified reference search and frozen comparison. Our proposed final-delivery plan is to use the first week to document the protocol, runtime and errors, and the second week to check reproducibility and finish the report. Further validation would require new speakers rather than more tuning on this holdout.
