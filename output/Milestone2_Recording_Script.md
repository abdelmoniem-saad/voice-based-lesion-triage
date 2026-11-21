# Milestone 2 recording script

Read slides 1–11. Slides 12–13 are hidden references. Speaker labels and bracketed cues are not spoken. Aim for six minutes.

**1 · Voice-Based Lesion Triage — Basmala**

Our project asks whether a short voice recording can help flag a possible lesion in someone who already has a voice disorder. In Milestone 1, we listened only to “ah.” This time, we asked what we could learn by adding “ee” and “oo” from the same person.

**2 · The Classification Problem — Basmala**

We classify one recording session. A positive session has one of eight specified lesion diagnoses, including malignant and premalignant conditions. A negative session comes from someone with a different voice disorder. That distinction matters: a model that simply recognizes an abnormal voice hasn’t solved our problem.

Our objective is to rank positive sessions near the top. We use average precision because positives are rare. Ordinary accuracy could look high even if we missed every positive case.

**3 · The Reference Model — Basmala**

Paterson and colleagues used features from one “ah” recording and trained a small neural network. They used SMOTE to create extra minority-class examples during training.

Their published model trained on a different dataset. For a fair comparison, we took their released speech features and retrained their actual pipeline on our development speakers. That way, their model and ours faced exactly the same holdout recordings.

**4 · Speakers and Recordings — Basmala**

Some people recorded more than once. To prevent the model from recognizing a familiar speaker, we kept each person entirely in development or entirely in the holdout. Every positive session had all three vowels, so adding vowels didn’t cost us positive cases. But the holdout had only seven positive recordings from six speakers. Keep that in mind when we show the results.

**5 · Our Model: Input to Output — Abdelmoniem**

Take one session. Our inputs are “ah,” “ee,” and “oo.” A pretrained speech encoder called XLSR-53 converts each sound into a vector. We keep the encoder fixed, join the three vectors, and pass them to a small PyTorch neural network with two hidden layers. The output is one score. A threshold chosen during development turns it into a positive or negative prediction.

Our contribution was to test two ideas separately: does adding vowels help, and does learning how much to weight each vowel help beyond simply joining them?

**6 · What Did Three Vowels Add? — Abdelmoniem**

[Point to the bars.] Compare the first two bars: with the same training method, three vowels beat “ah” alone in four of five development folds. Now look at the third bar. Attention let the model learn a weight for each vowel, but it performed almost identically to simple concatenation.

We kept concatenation and compared training methods. Tuned focal loss gave the highest development average precision. Adding age and sex lowered it, so we froze the voice-only model.

**7 · Matched Holdout Results — Abdelmoniem**

[Point to the two SMOTE rows.] On the holdout, the gain from adding vowels almost disappeared. Now compare the top and bottom rows: our frozen model had higher AUROC but slightly lower average precision than the authors’ retrained reference. We chose average precision in advance, and the interval for the difference includes zero. We didn’t establish an improvement.

In actual cases, our selected sensitivity-target threshold caught five of the seven positive recordings. That’s too little evidence to claim clinical readiness.

**8 · Evaluation Desk — Basmala**

Those two metrics are why we built Evaluation Desk. Ask, “Did our model beat the reference?” A local language model chooses which evidence to inspect. Python retrieves the actual numbers and their source, so a reviewer can check the conclusion.

The revised system passed ten predefined checks: eight used the language model, and two used explicit scope rules. It reviews experiments; it never examines patient audio.

**9 · Next Steps to Final Delivery — Basmala**

We stopped tuning after the planned reference search and holdout comparison. Our proposed November schedule moves from reviewing missed cases to verifying results and runtimes, then testing two new assistant questions and finishing the report. Stronger validation would require new speakers.

**10 · Technical Contributions — Both**

Basmala: I audited the labels and recordings, built the speaker-separated splits, and implemented the reference comparison. I estimated uncertainty by resampling speakers so repeated recordings stayed together.

Abdelmoniem: I extracted and cached the speech features, implemented the PyTorch classifiers, and ran the fusion and loss experiments. I also used learning curves to investigate overfitting.

**11 · Thank You — Abdelmoniem**

Three vowels helped during development. That gain was small on the holdout, and our final model did not clearly beat the reference. Thank you.
