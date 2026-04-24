## Title

**Counterfactual Simulation of Extreme Mental Health Scenarios for Clinical Preparedness via Fine-Tuned LLMs and Explainable AI**

## Overview

A **Counterfactual scenario simulator** for mental health analysis that generates possible negative future scenarios using the current patient state and scenarios using fine-tuned LLMs, with an XAI layer built on top  used for future preparedness of patient health and psychological students studies.

## Core Idea

- Generate **extreme end scenarios** from the current patient state
- **Not** for exact predictions  but to be **prepared** for extreme ends
- Acts as a **second reader**  a clinical decision support tool
- Dual audience: **Clinicians** and **Psychology Students**

## Key Components

- **Fine-Tuned LLMs** generates extreme negative future scenarios from current patient state
- **XAI Layer** built on top of the LLM for causal explainability of generated scenarios
- **Counterfactual Reasoning** drives the scenario generation engine
- **Second Reader Paradigm** AI acts as a consulting second opinion, not a final decision maker

## Goals

- Prepare clinicians for worst-case mental health trajectories
- Support psychological students in simulation-based training
- Improve patient health outcomes through proactive preparedness
- Provide explainable, trustworthy AI for clinical mental health use

"Counterfactual Simulation of Extreme Mental Health Scenarios for Clinical Preparedness via Fine-Tuned LLMs and Explainable AI"

Abstract:
Clinical notes on patients with mental health conditions contain rich, unstructured information that remains critically underutilized in clinical decision-making. Existing AI systems for mental health prioritize risk score prediction, leaving clinicians and psychology students unprepared for extreme adverse patient trajectories. We propose a safety-aware fine-tuned large language model (LLM) framework that acts as a second reader of mental health documentation not to predict outcomes, but to prepare clinicians and psychology students for extreme patient trajectories. The framework operates in three stages: (1) Clinical Factor Extraction extracting structured risk and protective factors from unstructured clinical notes using a DSM-5 and ICD-11 grounded schema; (2) Extreme Scenario Generation generating counterfactual narratives of extreme adverse mental health trajectories reachable from the patient's current state, targeting boundary conditions rather than probable outcomes; and (3) Causal Explainability an Explainable AI (XAI) layer providing causal pathway justifications and uncertainty estimates for each generated scenario. The system is trained on de-identified clinical notes and validated against clinician-annotated gold standards, evaluated across scenario plausibility, causal fidelity, hallucination rate, and clinician usability. To the best of our knowledge, this is the first framework to employ counterfactual simulation of extreme mental health scenarios for clinical preparedness, augmenting rather than replacing expert clinical judgment.












Literature Survey:
Unstructured clinical notes in mental health settings remain critically underutilized despite their richness in capturing patient histories, symptom progressions, and clinician observations. NLP-based extraction from free-text clinical notes has been shown to significantly improve downstream decision-making quality [1], with text-based clinical features contributing more substantially to mental health risk detection than structured or audio-based markers [2]. However, existing systems treat clinical notes purely as a source for risk scoring rather than as a basis for generating forward-looking preparedness insights leaving clinicians and psychology students without systematic tools for anticipating extreme adverse trajectories. This work addresses this gap by reframing clinical note analysis as a preparedness task rather than a scoring task, using clinical notes as the input to an extreme scenario generation pipeline.
Large language models have demonstrated strong potential for mental health documentation analysis, with domain-specific fine-tuning established as a prerequisite for reliable clinical performance [3]. Comprehensive reviews have identified explainability and clinical alignment as the two most critical unresolved challenges in current deployments [4], while formal grounding in DSM-5 and ICD-11 taxonomies has been shown to improve diagnostic traceability in hybrid LLM frameworks [5]. Despite these advances, existing LLM-based systems remain focused on risk classification and diagnosis none have been proposed as a second reader that generates structured extreme scenario narratives from current patient states. This work directly fills this gap by proposing a fine-tuned LLM framework that acts explicitly as a second reader, generating extreme adverse trajectories rather than diagnostic labels.
Counterfactual reasoning has emerged as a foundational technique for preparedness-oriented clinical AI, with counterfactual modeling shown to simulate rare and extreme clinical scenarios beyond conventional prediction [6]. Fine-tuned LLMs have demonstrated counterfactual scenario generation with up to 99% plausibility, outperforming optimization-based baselines in clinical coherence and actionability [7]. However, general-purpose LLMs lack the depth required for reliable counterfactual reasoning in high-stakes settings [8], and no existing work has applied counterfactual simulation to extreme mental health trajectory generation for clinical preparedness. Equally, conventional XAI techniques such as SHAP and LIME have been found to lack clinical actionability [9], while generative XAI frameworks translating model reasoning into clinical narratives represent the missing architectural link for deployment [10] yet no existing framework combines counterfactual extreme scenario generation with a causal XAI layer for mental health preparedness. This work closes this gap by integrating both into a unified pipeline specifically designed for clinical and educational use.
Safety remains a central concern, with medical hallucination formally established as a direct patient safety risk across foundation models [11], and chain-of-thought reasoning alone demonstrated insufficient for safe clinical deployment [12]. Critically, existing safety frameworks focus on constraining outputs within probable boundaries the inverse of what is required for a system designed to safely generate extreme boundary scenarios. This work addresses this gap directly: a fine-tuned LLM framework acting as a second reader of mental health documentation, generating extreme counterfactual adverse trajectories with an integrated causal XAI layer designed not to predict, but to prepare.


References:
[1] E. Sezgin, S.-A. Hussain, S. Rust, and Y. Huang, “Extracting medical information from Free-Text and unstructured Patient-Generated health data using natural language processing methods: Feasibility study with real-world data,” JMIR Formative Research, vol. 7, p. e43014, Jan. 2023, doi: 10.2196/43014.
[2] Y. Hua et al., “Large Language Models in Mental Health Care: a Scoping Review,” arXiv.org, Jan. 01, 2024. https://arxiv.org/abs/2401.02984
[3] X. Xu et al., “Mental-LLM,” Proceedings of the ACM on Interactive Mobile Wearable and Ubiquitous Technologies, vol. 8, no. 1, pp. 1–32, Mar. 2024, doi: 10.1145/3643540.
[4] Y. Hua et al., “Large Language Models in Mental Health Care: a Scoping Review,” arXiv.org, Jan. 01, 2024. https://arxiv.org/abs/2401.02984
[5]
[6]
[7]
[8]
[9]
[10]
[11]
[12]
















#	Paper	Link
1	Extracting Medical Information From Free-Text Clinical Notes	https://formative.jmir.org/2023/1/e43014

2	NLP for Mental Health Interventions: A Systematic Review	https://www.nature.com/articles/s41398-023-02592-2

3	Mental-LLM: LLMs for Mental Health Prediction	https://arxiv.org/abs/2307.14385

4	LLMs in Mental Health Care: A Scoping Review	https://arxiv.org/abs/2401.02984

5	LLMs for Interpretable Mental Health Diagnosis	https://chaowang-vt.github.io/pubDOC/KimW25_LLM4Health.pdf

6	The Clinical Potential of Counterfactual AI Models	https://www.thelancet.com/journals/lancet/article/PIIS0140-6736(24)00313-1/fulltext

7	Counterfactual Modeling with Fine-Tuned LLMs	https://arxiv.org/abs/2601.14590

8	Do Models Explain Themselves? Counterfactual Simulatability	https://arxiv.org/abs/2307.08678

9	Explainable AI in Healthcare: Systematic Review	https://www.medrxiv.org/content/10.1101/2024.08.10.24311735v1

10	From Explainability to Action: Generative XAI Framework	https://arxiv.org/abs/2510.13828

11	Medical Hallucination in Foundation Models	https://arxiv.org/abs/2503.05777

12	Diagnosing Hallucination Risk in AI Decision-Support	https://arxiv.org/abs/2511.00588

















#	Sentence	Citations
1	"detect depression and suicide risk from narratives/notes and questionnaire text, and they achieve good AUROC"	[1]
2	"review papers show many works on screening and risk prediction from text, and some on note summarization"	[2][3]
3	"The systematic reviews call out the lack of safety focused, explainable LLM workflows in mental health not just better accuracy"	[3][4]
4	"LLMs and transformers detecting depression or suicide risk from narratives or notes"	[1][2][3]

References:
[1] S. K. Lho et al., “Large language models and text embeddings for detecting depression and suicide in patient narratives,” JAMA Network Open, vol. 8, no. 5, p. e2511922, May 2025, doi: 10.1001/jamanetworkopen.2025.11922.
https://jamanetwork.com/journals/jamanetworkopen/fullarticle/2834372
[2]  B. G. Teferra et al., “Screening for Depression Using Natural Language Processing: Literature review,” Interactive Journal of Medical Research, vol. 13, p. e55067, Nov. 2024, doi: 10.2196/55067.
https://www.i-jmr.org/2024/1/e55067
[3] Z. Guo, A. Lai, J. H. Thygesen, J. Farrington, T. Keen, and K. Li, “Large Language Models for Mental Health Applications: Systematic review,” JMIR Mental Health, vol. 11, p. e57400, Sep. 2024, doi: 10.2196/57400.
https://mental.jmir.org/2024/1/e57400
[4] A. M. Alkalbani et al., “A Systematic review of large language models in medical specialties: applications, challenges and future directions,” Information, vol. 16, no. 6, p. 489, Jun. 2025, doi: 10.3390/info16060489.
https://www.mdpi.com/2078-2489/16/6/489
[5] https://arxiv.org/pdf/2601.14590
[6] https://link.springer.com/article/10.1007/s41870-025-02590-1

What Already Exists
In short: Using AI to directly predict risk or summarize notes from clinical text is well-established.

What Is Innovative:
This project introduces three novel elements that don't exist together in current literature:
1. Structured Schema-Based Extraction (not just prediction)
Existing: Models output a single risk score.
Novel: Our LLM Should extract risk/protective factors using a clinically curated, fixed schema (e.g., suicidal ideation, substance misuse, family support) with per-factor uncertainty (present/absent/uncertain). This makes the system explainable and auditable.

2. Counterfactual "Second Reader" Pipeline
Existing: No prior work generates counterfactual clinical notes for risk assessment.
Novel: The LLM should rewrite the note as "what would this look like if the patient were high-risk?", then a classifier uses both the original and counterfactual embeddings together. This contrastive signal ("what's missing vs. what's present") is a fundamentally new way to augment clinical decision-making.


3. Explicit Safety & Hallucination Metrics for Mental Health LLMs
Existing: Most work reports only accuracy (AUROC, F1). Systematic reviews highlight the lack of safety-focused LLM evaluation in mental health.
Novel: We define and measure:
Hallucination rate — how often the LLM invents risk factors not in the note
Over/under-estimation — extreme misclassifications (high confidence but wrong)
Calibration (Expected Calibration Error) — are predicted probabilities trustworthy?

What exists = using AI to predict risk or summarize clinical text.
What's new = our system shouldn't just predict — it should read like a second clinician: extracting structured factors, imagining a worst-case version of the note, and being held accountable for hallucinations and safety. The combination of counterfactual generation + schema extraction + safety metrics is the core novelty and has not been done before in mental health NLP.

What makes it novel:
	"Second reader" framing — Using an LLM not to replace clinical judgment but to augment it as a structured second set of eyes is conceptually fresh. Most prior work does direct risk prediction; yours treats the LLM as an assistive reviewer.

	Counterfactual generation — Asking "what would this note look like if the patient were high-risk?" and using both original + counterfactual embeddings as classifier features is genuinely innovative. This is not standard in clinical NLP. Counterfactual reasoning exists in ML fairness literature but has barely been applied to mental health note analysis.

	Explicit safety-first evaluation — Most LLM-for-psychiatry papers report AUROC/F1 and stop. Your proposal to quantify hallucination rate, over/under-estimation of risk, and calibration is a meaningful methodological contribution. Reviewers increasingly demand this for clinical AI papers.

	Structured schema extraction + uncertainty — Combining clinically-curated factor extraction with per-factor uncertainty estimates goes beyond generic summarization.

	Publication potential: High. This fits journals/venues like Nature Medicine (AI), JMIR Mental Health, npj Digital Medicine, JAMIA, ACL Clinical NLP workshop, ML4H, or AMIA. The topic (LLMs + safety + mental health) is extremely timely.


Can it Be Achieved with These Datasets? — Yes, with solid foundations.
Here's what our data gives :
Resource	What You Have
Mental health patients	~107,956 unique patients
Admissions	~238,565 admissions
Self-harm / suicide ICD codes	1,515 distinct codes
Patients with self-harm/suicide diagnoses	~11,327 patients across ~21,070 admissions (strong positive class)
Discharge notes (Dataset2)	Full-text clinical notes with mental health filtering already done
Structured data	diagnoses, procedures, labs, medications, pharmacy, DRGs, transfers — all filtered for mental health

Key feasibility points:
	Labels are constructible: You have ~11.3K patients with self-harm/suicide codes (positive class) vs. ~96.6K depression patients without them (negative class). This is a healthy class split for binary classification with standard techniques for imbalance.

	Clinical notes exist: Dataset2 has discharge summaries linked by subject_id and hadm_id, which you can join to the ICD-coded labels from Dataset1. This is the exact pipeline the abstract describes.

	30-day readmission is also computable from the admissions.csv table (admittime/dischtime fields), giving you a second outcome label option.

	Cohort is large enough: Even after requiring both a discharge note and a self-harm label, you'll likely have thousands of positive-class notes — more than sufficient for training/validation/testing.

Honest Caveats
	MIMIC is ICU-heavy, not psychiatry-specialist: Most notes are from medical/surgical ICU stays, not dedicated psychiatric units. Depression and self-harm content will appear but may be secondary to the primary medical reason for admission. You should acknowledge this as a limitation.

	Notes are de-identified: PHI placeholders (___) will replace names, dates, and some contextual details, which can affect LLM extraction quality. Manageable, but worth noting.

	No real clinician validation in your current setup: The abstract mentions "clinician-annotated gold standards" and a "simulated chart review study." With MIMIC alone you can do the computational pipeline, but clinician annotation would need to be stated as planned/future work unless you can recruit annotators.

	LLM API costs: Running extraction + counterfactual generation on thousands of notes via GPT-4 / Claude will cost real money. An open model (LLaMA-3, Mistral) locally could substitute if compute is available.

Suggested roadmap:
	Data prep — You already have MIMIC-IV + clinical notes (Dataset2 discharge/radiology notes). Filter to depression-related notes and define your outcome labels (e.g., subsequent self-harm ICD codes, readmission)
	Extraction module — Prompt an LLM to extract risk/protective factors from notes using a fixed schema. Start simple with few-shot prompting
	Counterfactual generation — Prompt the LLM: "Rewrite this note as if the patient were at high risk of self-harm" — then compare original vs. counterfactual embeddings
	Classifier — Train a lightweight model (logistic regression or small MLP) on the concatenated embeddings
	Evaluation — Measure extraction accuracy, calibration, hallucination rate, and ideally get 1-2 clinicians to review a sample
 The counterfactual angle is your differentiator — that's what makes this publishable. Make sure your eventual paper clearly demonstrates why the counterfactual adds value over just extracting risk factors directly. A simple ablation (with vs. without counterfactual encoding) would be very convincing.

1. Clarify the core research questions
From your abstract, your project can focus on three main questions:
	Information extraction:
Can an LLM reliably extract risk and protective factors for depression/self‑harm from clinical notes?
	Counterfactual “second reading”:
Does giving a model access to both the original note and an LLM‑generated counterfactual representation improve prediction of bad outcomes (e.g., self‑harm, re‑admission) compared with using only the original text?m
	Safety / hallucination:
How often does the LLM invent (hallucinate) non‑existent risk factors or overstate risk, and how can we measure it quantitatively?

2. Choose data and labels
You need de identified clinical notes with some outcome label.
Public data you can realistically use
	MIMIC III or MIMIC IV (notes) – ICU notes, discharge summaries, some psychiatric content.
	MIMIC IV Note is best; you can treat:
	Depression / self harm labels from ICD 10/ICD 9 codes,
	or 30 day readmission / in hospital mortality as proxy risk outcomes.
You’ll need to complete the PhysioNet credentialing to download MIMIC.
How to construct labels
Example for a first version:
	Positive class (high risk):
	patients with depression diagnosis plus self harm / suicide attempt code, or
	patients readmitted within 30 days with a psychiatric diagnosis.
	Negative class (lower risk):
	patients with depression diagnosis but no self harm code and no early readmission.
Each clinical note (e.g., discharge summary, psychiatry consult note) becomes one example with a binary label.

3. Define the “schema” of risk/protective factors
Before you touch the model, define a simple, fixed schema your LLM will extract.
Example:
	Risk factors:
	prior suicide attempt
	current suicidal ideation
	substance misuse
	severe hopelessness
	social isolation
	recent major loss/stressor
	non adherence to medication
	Protective factors:
	strong family support
	religious/spiritual engagement
	stable employment/education
	engagement with therapy
	no access to lethal means
You can design this from clinical guidelines and literature (no need to invent from scratch).
For evaluation, you can manually annotate a small subset of notes (e.g., 150–200) yourself (not as a clinician, but following simple rules) to check whether the LLM is consistent with your rules. You just need a basic annotation guideline.
________________________________________
4. System design: simple, implementable architecture
4.1 Baseline classifier (no LLM second reading)
	Encode the raw text of the note:
	Either use a pre trained clinical transformer (e.g., ClinicalBERT) or a strong text model like roberta-base.
	Train a binary classifier on top of the [CLS] embedding to predict your outcome label.
	This gives you:
	AUC, F1, calibration plots.
	This is your baseline.
4.2 LLM based extraction and counterfactuals
Step 1 – Prompt the LLM for extraction
For each note, you send a prompt like:
“From the following clinical note, list whether each of these risk and protective factors is present (yes/no/uncertain) and briefly justify.”
The LLM output is converted to:
	a vector r of risk factor indicators (0/1/0.5 for uncertain),
	a vector p of protective factor indicators,
	optional confidence scores (you can approximate by mapping ‘uncertain’ to 0.5).
Step 2 – Generate a counterfactual representation
Rather than a long narrative, you can keep it simple and implementation friendly:
	Ask the LLM:
“Rewrite this note as if the patient were at high risk of self harm, being consistent with the clinical context. Do not change age or basic medical information.”
	Then encode this counterfactual note with the same encoder as the baseline (ClinicalBERT/Roberta) to get an embedding cf.
Step 3 – Build the “second reader” classifier
Now you have for each note:
	Original embedding orig
	Counterfactual embedding cf
	Risk/protective factor vectors r, p
Concatenate them:
X = [orig || cf || r || p]
Train a small neural or logistic regression classifier on X to predict the same outcome label.
Compare to baseline:
	Does AUC or recall for positive class improve?
	Does calibration improve?
________________________________________
5. Safety and hallucination analysis
You need simple, quantifiable metrics.
5.1 Hallucination of risk factors
On the manually annotated subset:
	For each factor, compute:
	Precision: of the factors LLM marks as present, how many are truly present in the text?
	Recall: of the factors truly present, how many does the LLM detect?
	A hallucination is essentially a false positive (LLM claims risk that isn’t in the note).
You can report hallucination rate as:
'#' can not be used here


\text{Hallucination rate} = \frac{\text{# factors marked present but not annotated}}{\text{# all factors marked present}}
5.2 Over /under estimation of risk
Compare:
	predicted probability from baseline classifier,
	predicted probability from second reader model.
On held out test data:
	Focus on cases where model is very confident but wrong (e.g., prob > 0.9 but label is negative).
	Count how many such over estimates and under estimates each model makes.
You can describe safety as:
	fewer extreme misclassifications,
	improved calibration (e.g., using Expected Calibration Error).
________________________________________
6. Implementation outline (step by step)
	Data access
	Register and download MIMIC IV Note (or similar).
	Extract a cohort: select notes with depression or related codes; define outcome labels.
	Preprocessing
	Clean text (remove PHI tokens, headers).
	Limit note length (truncate long notes to a max number of tokens).
	Split into train/val/test by patient.
	Schema + small annotation
	Finalize list of risk/protective factors.
	Annotate 150–200 notes yourself according to simple rules.
	Baseline model
	Train ClinicalBERT/Roberta classifier on original text.
	Evaluate AUC, F1, calibration.
	LLM prompts
	Choose an LLM you can access (e.g., OpenAI / local LLaMA variant).
	Implement scripts:
	extraction prompt → JSON risk/protective vector
	counterfactual prompt → generated text
	Run on train/val/test sets.
	Second reader classifier
	Encode original + counterfactual text.
	Build concatenated features with risk/protective indicators.
	Train simple classifier; compare metrics with baseline.
	Safety analysis
	For annotated subset, compute hallucination metrics.
	For test set, analyze calibration and extreme misclassifications.
	Writing
	Introduction: problem + why safety/second reading matters.
	Methods: data, schema, models, prompts, metrics.
	Results: tables for extraction quality, prediction performance, safety.
	Discussion: limitations (no real clinicians yet, outcome labels as proxy, etc.) and future work (real clinician studies, multi hospital data).
________________________________________
7. Where you may need to simplify
Depending on your time/resources:
	You can skip real clinician evaluation and clearly state it as future work.
	Counterfactuals can be text or a simple “risk amplified” embedding (easier: ask LLM to output a structured description of “what would make this high risk” instead of full rewritten note).
	If federated or privacy aspects are too heavy, treat it as a single center retrospective study.

1. Safety Aware LLM “Second Reader” for Depression Notes
What exists:
	LLMs and other NLP models have been used to:
	detect depression and suicide risk from narratives/notes and questionnaire text, and they achieve good AUROC[1].
	review papers show many works on screening and risk prediction from text, and some on note summarization[2][4].
	However, the focus is usually:
	direct prediction (risk score), or
	generic summarization.
What is not really there yet:
	An LLM explicitly designed as a “second reader” that:
	uses a clinically defined schema of risk & protective factors,
	generates counterfactual narratives (“if this patient were high risk…”) and
	feeds both original + counterfactual encodings into a downstream classifier,
	with explicit safety metrics (hallucination, over /under estimation).
The systematic reviews call out the lack of safety focused, explainable LLM workflows in mental health—not just better accuracy[4][6].
Novelty / weightage:
	Highly topical (LLMs + safety + mental health).
	Your twist—counterfactual second reading + safety evaluation—is not standard in existing work.
	Strong fit for AI in medicine and clinical NLP conferences (e.g., ACL clinical track, ML4H, AMIA, AI for health workshops).
Caveat: needs access to clinical notes and careful evaluation design, but technically you can do a scaled down version with public data.
Verdict: Very innovative, high potential impact. Arguably the strongest and most modern of the four ideas.

1. Refine the idea into clear research questions
You can frame the paper around three main questions:
	Extraction quality:
Can an LLM reliably extract risk and protective factors for depression/self harm from clinical notes using a fixed clinical schema?
	Second reader value:
Does combining:
	the original note representation and
	LLM generated counterfactual representation / factor vector
improve prediction of adverse outcomes (e.g., self harm, psychiatric readmission) compared with:
	a baseline text classifier using only the original note?
	Safety:
How often does the LLM hallucinate risk factors or seriously over /under estimate risk, compared to a baseline model?
Everything you implement should answer at least one of these.
________________________________________
2. Data: what you can use in practice
2.1 Likely choice: MIMIC IV Note
	Public, de identified ICU dataset with notes; has:
	discharge summaries, nursing notes, etc.
	diagnoses and procedures with ICD codes.
	You can:
	filter for notes where depression/mental health codes appear,
	define an outcome label such as:
	30 day unplanned readmission, or
	presence of self harm / suicide attempt ICD codes at or after the index stay.
You don’t need the dataset right now to outline the paper, but you should confirm you can get PhysioNet access.
2.2 Outcome labels examples
Pick one main outcome:
	Option A: Self harm related outcome
	Positive = patient has any ICD code for self harm / suicide attempt during or within 6 months after the index admission.
	Negative = patient with depression but no self harm codes in that window.
	Option B: Early psychiatric readmission
	Positive = readmission within 30 days with a psychiatric primary diagnosis (depression, bipolar, etc.).
	Negative = no such readmission.
Each index note (e.g., discharge summary) → one labelled sample.
________________________________________
3. Define the risk/protective factor schema
Before modeling, you must define a fixed schema your LLM will output.
Example (you can tweak):
Risk factors:
	Current suicidal ideation or plan
	History of suicide attempt
	Severe hopelessness / worthlessness
	Substance misuse (alcohol/drugs)
	Psychosis or severe agitation
	Social isolation / lack of support
	Recent major stressor or loss
	Non adherence to treatment
Protective factors:
	Strong family/social support
	Stable employment/education
	Religious/spiritual engagement
	Active engagement in therapy
	No access to lethal means
	Positive future plans / goals
For each factor, the LLM will output: present / absent / uncertain.
You can build this schema from clinical guidelines (e.g., suicide risk assessment tools) and cite them.
________________________________________
4. Overall system architecture
You’ll compare two systems:
	Baseline model (no second reader)
	Second reader model (your method)
4.1 Baseline
	Use a strong text model:
	e.g., ClinicalBERT, Bio_ClinicalBERT, or roberta-base fine tuned on notes.
	Input: raw note text.
	Output: probability of outcome (e.g., self harm, readmission).
	Train with standard cross entropy on training set.
This is your reference point for both performance and safety.
4.2 LLM second reader
For each note:
	Extraction step
Prompt an LLM (e.g., GPT 4o or a strong open model):
	Ask it to fill in the schema:
“Given the clinical note below, decide for each risk and protective factor if it is present, absent, or uncertain. Then briefly justify each ‘present’ decision using an exact phrase from the note.”
	Parse output into:
	risk vector r (values 1, 0, 0.5 for present/absent/uncertain)
	protective vector p (same)
	optional textual justifications (for analysis, not necessarily used as features).
	Counterfactual generation step
Ask the LLM to imagine a high risk version:
“Rewrite the note as if this patient were at high risk of self harm, keeping demographics and medical comorbidities consistent. Add only clinically plausible changes.”
	Get counterfactual text cf_text.
	Encode original + counterfactual text
Use the same encoder as baseline (e.g., ClinicalBERT):
	orig_emb = Encoder(note_text)
	cf_emb = Encoder(cf_text)
	Build feature representation
Concatenate:
X=[orig_e mb,|,cf_e mb,|,r,|,p]

	Train lightweight classifier
	A small MLP or logistic regression on X to predict outcome.
	Train only this classifier (encoder and LLM are frozen) to keep compute manageable.
This “second reader” thus uses both original content and “what a high risk note would look like”, plus structured factors.
________________________________________
5. Safety and hallucination evaluation
You need quantitative, not just anecdotal, safety analysis.
5.1 Manual evaluation subset
Create a small gold set:
	Randomly sample ~150–200 notes.
	Manually annotate:
	presence/absence of each factor according to clear rules,
	whether the note truly mentions self harm, etc.
You don’t have to be a clinician; just follow rule based definitions (e.g., any mention of “overdose”, “cut wrists”, etc.).
5.2 Hallucination metric
For each factor:
	Precision = TP / (TP + FP) where:
	TP = LLM says “present” and you also annotated “present”.
	FP = LLM says “present” but you annotated “absent”.
	Hallucination rate can be defined as:
HR=FP/(TP+FP)

You can report:
	overall HR across all factors,
	HR per factor (e.g., does it often hallucinate “family support”?).
5.3 Over /under estimation of risk
Compare baseline vs second reader:
	On test set, look at cases where model probability is >0.9 but label is negative (over estimation) and <0.1 but label is positive (under estimation).
	Count and compare:
	number of extreme errors,
	whether the second reader reduces these.
Also compute calibration (e.g., Expected Calibration Error, reliability diagrams) for both models.
________________________________________
6. Experimental design
6.1 Data splits
	Split by patient: train / validation / test (e.g., 70/10/20).
	Ensure no patient appears in multiple splits.
6.2 Baselines to report
	Logistic regression / XGBoost on bag of words or simple embeddings (to show traditional baseline).
	ClinicalBERT (fine tuned) – main strong baseline.
	Your Second Reader model.
Metrics:
	AUROC, AUPRC
	Sensitivity/specificity at clinically relevant thresholds
	Calibration (ECE)
	Safety metrics (hallucination, extreme errors)
________________________________________
7. How to position the novelty in your paper
In Introduction + Related Work, emphasise:
	Prior work:
	LLMs and transformers detecting depression or suicide risk from narratives or notes[1][2][4].
	Most do direct prediction and maybe simple explanation.
	Gaps:
	Lack of safety focused, structured workflows (risk/protective schema, uncertainty per factor).
	Almost no work on LLMs as second readers that provide:
	structured extractions,
	counterfactual narratives,
	explicit hallucination & calibration analysis.
	Your contribution:
	Define a clinically motivated schema and show LLM extraction reliability.
	Introduce a counterfactual second reader pipeline that improves predictions.
	Propose practical safety metrics for LLMs in mental health documentation (hallucination, extreme errors, calibration).
That trio gives your paper “weight” at conferences.
________________________________________
8. Immediate next steps for you
	Confirm data source:
	Most likely MIMIC IV Note; check you can get access.
	Choose the outcome:
	Decide: self harm–related vs readmission, based on what’s easier to code from ICDs.
	Lock in the factor schema:
	Draft your list of risk and protective factors; we can refine wording if you paste it here.
	Decide tools:
	Python + PyTorch + HuggingFace for encoders.
	Which LLM API or open model you can actually use.






DATASET INFO:
ICD-9 (International Classification of Diseases, 9th Revision) and ICD-10 (10th Revision) are two versions of the WHO's disease classification system. Key differences:
Aspect	ICD-9	ICD-10
Code format	3–5 digit numeric (e.g., 29600)	3–7 character alphanumeric (e.g., F321)
Total codes	~14,000 diagnosis codes	~70,000+ diagnosis codes
Specificity	Less granular	Much more detailed (laterality, severity, encounter type)
Structure	Numeric categories (001–999)	Alphanumeric chapters (A00–Z99)
Mental health range	Categories 290–319	Chapter F01–F99
US adoption	Used until Sept 30, 2015	Mandated from Oct 1, 2015 onward

Here's what the mental health ICD code ranges cover:
ICD-9 (Categories 290–319)
Range	Category
290–294	Organic psychotic conditions (dementias, delirium, amnestic disorders)
295	Schizophrenic disorders
296	Mood/affective disorders (bipolar, major depression)
297	Delusional/paranoid disorders
298	Other non-organic psychoses
299	Pervasive developmental disorders (autism spectrum)
300	Anxiety, phobic, dissociative, somatoform disorders
301	Personality disorders
302	Sexual and gender identity disorders
303–305	Substance use disorders (alcohol, drugs, tobacco)
306	Psychophysiological/psychosomatic disorders
307	Eating disorders, sleep disorders, tics, stuttering
308	Acute stress reactions
309	Adjustment disorders, PTSD
310	Non-psychotic conditions due to brain damage
311	Depressive disorder NEC
312	Conduct & impulse control disorders
313	Childhood emotional disorders
314	ADHD / hyperkinetic syndrome
315	Developmental learning/speech/coordination disorders
316	Psychic factors associated with physical diseases
317–319	Intellectual disabilities (mild, moderate, severe, profound)

ICD-10 (F01–F99)
Range	Category
F01–F09	Organic mental disorders (dementias, delirium)
F10–F19	Substance use disorders (alcohol, opioids, cannabis, cocaine, stimulants, hallucinogens, etc.)
F20–F29	Schizophrenia & psychotic disorders
F30–F39	Mood disorders (bipolar, depression, mania)
F40–F48	Anxiety, phobic, OCD, stress-related, dissociative, somatoform disorders
F50–F59	Eating disorders, sleep disorders, behavioral syndromes
F60–F69	Personality & behavioral disorders
F70–F79	Intellectual disabilities
F80–F89	Developmental disorders (speech, learning, autism spectrum)
F90–F98	Childhood-onset behavioral/emotional disorders (ADHD, conduct, tic disorders)
F99	Unspecified mental disorder

In short, these codes collectively cover the full spectrum of mental health conditions: dementias, psychoses, mood disorders, anxiety, substance use, personality disorders, developmental disorders, childhood behavioural issues, and intellectual disabilities.

Feature	v3.1 (latest)
Patients	364,627
Admissions	546,028
ICU stays	94,458
Years covered	2008–2022
Data fixes	Lab/diagnoses linkage, itemid fix, more

diagnoses_icd.csv - has icd_code, icd_version
procedures_icd.csv - has icd_code, icd_version


Files with subject_id and/or hadm_id (can be filtered by joining with diagnoses_icd):
admissions.csv - subject_id, hadm_id
drgcodes.csv - subject_id, hadm_id
emar.csv - subject_id, hadm_id
emar_detail.csv - subject_id (also emar_id)
hcpcsevents.csv - subject_id, hadm_id
labevents.csv - subject_id, hadm_id
microbiologyevents.csv - subject_id, hadm_id
omr.csv - subject_id
patients.csv - subject_id
pharmacy.csv - subject_id, hadm_id
poe.csv - subject_id, hadm_id












 













