# Datasheet for Sakhi

This datasheet follows the structure proposed by Gebru et al., *Datasheets for Datasets* (CACM, 2021). It accompanies the Sakhi benchmark, distributed alongside this directory.

The paper companion to this datasheet is the NeurIPS 2026 Evaluations and Datasets track submission "Sakhi: A Community-Validated Multilingual Benchmark for Maternal and Reproductive Health LLMs in Rural India."

---

## Motivation

**For what purpose was the dataset created?**
To evaluate large language models and conversational agents on maternal and reproductive health questions in English, Hindi, and Marathi, on a question set grounded in a non-WEIRD source population (rural India) and validated by practising doctors and Accredited Social Health Activist (ASHA) workers who serve that population.

**Who created the dataset and on behalf of which entity?**
The Sakhi authors, in collaboration with the operating partner of a deployed maternal-health bot. No commissioning entity funded the benchmark per response.

**Who funded the creation of the dataset?**
The data-collection phase was supported by the operating-partner healthcare nonprofit and a credit grant plus mentorship from a frontier AI lab. We are keeping both names anonymous for the duration of peer review and will name them in the final version.

**Any other comments?**
The dataset is released to support safer deployment of AI maternal-health systems in multilingual rural settings. It is not intended as a deployment-readiness certification.

---

## Composition

**What do the instances represent?**
The release ships in three files. Each row of the first two is a maternal- or reproductive-health question paired with a reference answer:
- **Expert arm** (`sakhi_benchmark_expert.csv`): 149 doctor-edited question-answer pairs.
- **Non-expert arm** (`sakhi_benchmark_non_expert.csv`): 231 community-sourced pairs with doctor-reviewed reference answers.

Each pair appears in three parallel languages (English, Hindi, Marathi), giving 1,140 query-language pairs in total.

The third file is the doctor calibration record:
- **Doctor ratings** (`sakhi_doctor_ratings.csv`): 2,103 criterion-level binary verdicts spanning 169 (reviewer, question) pairs across 148 unique expert-track questions, supplied by 11 practising Indian doctors. 21 of those questions received two independent verdicts; the remaining 127 received one. Each row carries an anonymised reviewer pseudonym (R1 to R11), reviewer role (OB/GYN or General Practitioner), self-reported clinical experience and prior AI exposure, the question and AI response shown to the reviewer, the rubric criterion and its axis, and the binary pass/fail verdict.

**How many instances are there in total?** 380 question-answer pairs across the two reference tracks; 1,140 query-language pairs; 2,103 criterion-level doctor verdicts spanning 169 (reviewer, question) pairs on 148 expert-track questions.

**Does the dataset contain all possible instances or is it a sample?**
A sample drawn from a generation pool of 845 candidate pairs, after multi-stage human review. The sampling is documented in Section 2 of the paper.

**What data does each instance consist of?**
Columns:
- `id` — unique identifier per pair.
- `theme` — one of 10 maternal-health themes.
- `question_en`, `question_hi`, `question_mr` — parallel question texts.
- `reference_en`, `reference_hi`, `reference_mr` — parallel reference answers.
- `track` — `expert` or `non_expert`.

**Is there a label or target associated with each instance?**
Yes: a thematic label (10 classes) and a track marker. The reference answer serves as the per-instance gold target for evaluation.

**Is any information missing from individual instances?**
A small number of instances may have missing entries in one of the three languages where translation review flagged ambiguity. These rows are explicitly marked. Coverage is otherwise complete.

**Are relationships between individual instances made explicit?** No.

**Are there recommended data splits?**
Sakhi is an evaluation-only benchmark. We do not recommend training-test splits because the benchmark should not be used as training data; doing so defeats its purpose as an independent evaluation set.

**Are there any errors, sources of noise, or redundancies in the dataset?**
The non-expert arm contains some pairs that an additional doctor pass could further sharpen. The translation review flagged a small number of pairs where the Hindi or Marathi rendering uses a regionally-specific idiom that is more colloquial than the doctor reference; we kept these as-is because the colloquial form is the one ASHA workers identified as most patient-faithful.

**Does the dataset contain confidential or sensitive data?**
No personally identifying information about any individual. Topics include sensitive but routine maternal-health subjects (contraception, miscarriage, postnatal mental health). The dataset is age-appropriate adult content but contains no explicit material beyond what a patient-style health question would naturally contain.

**Does the dataset relate to people?** Indirectly: the questions reflect concerns of women in rural and semi-urban India (married, ages 23–33, education ranging from no formal schooling to graduate). No specific individual is identifiable from any instance.

---

## Collection process

**How was the data associated with each instance acquired?**
Generated by a two-stage Aya Expanse + MedGemma generator-validator from a curated knowledge corpus of public maternal-health guidelines (WHO Recommendations, India National ANC Guideline, ANM Training Manual, NHM protocols). Surviving pairs were routed through a three-stakeholder review pipeline (doctors, ASHA workers, nonprofit staff) on a purpose-built Q&A review platform. Translations to Hindi and Marathi were performed by professional native-speaker translators and reviewed by ASHA workers for patient-voice fidelity.

**What was the time frame for the collection?**
The doctor-platform sessions, including both the reference-editing rounds and the rubric-rating rounds that produced `sakhi_doctor_ratings.csv`, ran from 2025-10-31 to 2025-12-01.

**Were the people whose data is included notified?**
The dataset does not contain individual user data. Doctors, ASHA workers, and nonprofit staff who reviewed the pairs were briefed on the purpose and intended use of the benchmark.

**Did the people whose data is included consent to its use?**
The benchmark contains no individual user data, so user-level consent does not apply. Reviewers (doctors, ASHA workers, nonprofit staff) consented to the review activity and to the release of the curated benchmark.

**Was an Institutional Review Board (IRB) review performed?**
Exempt. The source material was provided to the research team in fully anonymized form and contains no personally identifying information. We document this exemption explicitly in Section 3 of the paper rather than relying on it tacitly.

---

## Preprocessing, cleaning, and labelling

**Was preprocessing/cleaning/labelling done?**
- Theme labels are produced by a DSPy-based few-shot classifier initialised with 10–15 expert-labelled seed examples per theme. Low-confidence classifier predictions were reviewed by clinicians before final assignment.
- Deduplication uses embedding similarity against the source corpus and fuzzy string match against existing entries.
- Pairs that failed any of the three stakeholder reviews were dropped at the corresponding stage.

**Was the "raw" data saved in addition to the cleaned data?**
The 845-pair generation pool and the per-stage drop logs are retained internally for audit and reproducibility but are not part of the public release.

**Is the software used to preprocess/clean/label the data available?**
Yes; the `scoring/pipeline/` and `generation/` directories of the accompanying code repository contain the full pipeline.

---

## Uses

**Has the dataset been used for any tasks already?**
Yes. The companion paper evaluates 13 frontier LLMs on this benchmark using a five-axis fifteen-criterion clinical rubric.

**Is there a repository linking to papers/systems that use the dataset?**
https://huggingface.co/datasets/SimPPL/sakhi

**What other tasks could the dataset be used for?**
- Cross-lingual maternal-health LLM evaluation.
- Calibration of LLM-as-judge protocols on multilingual medical content. The doctor-rating record is designed to support exactly this: the companion paper anchors the LLM-judge agreement against the per-axis Cohen's kappa between LLM verdicts and the doctor verdicts on the same (question, response, rubric) triples.
- Audit of clinical-rubric performance across model families.
- Inter-rater agreement studies on a low-resource clinical NLP domain. The 21 doubly-rated questions support a doctor-doctor kappa baseline that any new rubric judge can be measured against.

**Are there tasks for which the dataset should NOT be used?**
- The dataset must not be used to make individual clinical decisions; it scores aggregate model behaviour on patient-style questions, not individual answers to individual patients.
- The dataset must not be used as training data; doing so defeats its purpose as an independent evaluation set.
- The dataset must not be used as a deployment-readiness certification on its own.

---

## Distribution

**How will the dataset be distributed?**
- Hugging Face dataset repo: https://huggingface.co/datasets/SimPPL/sakhi.
- GitHub mirror: https://github.com/SwapneelM/sakhi.
- Croissant metadata file (core fields plus Responsible-AI fields per the NeurIPS 2026 Evaluations and Datasets requirements) accompanies the release.

**When will the dataset be distributed?** On acceptance of the companion paper.

**License?** Creative Commons Attribution 4.0 International (CC BY 4.0). See `LICENSE`.

**Have any third parties imposed IP-based or other restrictions on the data?** No.

**Do any export controls or other regulatory restrictions apply?** No.

---

## Maintenance

**Who is supporting/hosting/maintaining the dataset?** The Sakhi authors.

**How can the owner/curator/manager be contacted?** swapneel@simppl.org

**Is there an erratum?** Maintained in the GitHub issue tracker.

**Will the dataset be updated?** Versioned releases tagged on the Hugging Face repo. Approximate cadence: annual. Two follow-up releases are planned: (1) an Indic-first native-authored extension; (2) a human-rater scoring subset that mirrors the LLM three-judge calibration.

**If others want to extend/augment/build on/contribute, is there a mechanism?**
Yes; pull requests against the GitHub mirror are welcome. Contributions that change reference answers must include doctor sign-off; contributions that change translations must include ASHA-worker review for patient-voice fidelity.
