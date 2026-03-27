# MOT Runtime Replay / Association Reranking Reading List

Date: 2026-03-17

This note is organized for the current paper direction:

- strong tracking-by-detection host
- association-time plug-in rather than full tracker replacement
- ambiguity-aware reranking / confusion handling
- runtime candidate groups rather than GT-clean proxy pairs

I split the literature into three buckets:

1. Most directly related to our current direction
2. Learnable association / proposal scoring / relation modeling
3. Supporting papers on host strength, memory, and appearance

For each paper, I list:

- venue and year
- official paper link
- the main technical idea
- the decision object it operates on
- why it matters for our work
- what to pay attention to when reading

Important reading advice:

- Do not only read for "what module they used".
- Read for "what inference object they are solving" and "where they intervene in the tracker".
- For our current project, the most important distinction is whether a paper learns:
  - pairwise score calibration,
  - candidate/proposal scoring,
  - ambiguity/confusion handling,
  - end-to-end association,
  - or stronger host features.

---

## 1. Most Directly Related to Our Current Direction

These are the papers closest to the current paper direction:

- confusion-aware association
- ambiguity-specific handling
- strong-host plug-in logic
- association refinement rather than full detector/tracker redesign

### 1.1 DeconfuseTrack: Dealing with Confusion for Multi-Object Tracking

- Venue: CVPR 2024
- Official paper: https://openaccess.thecvf.com/content/CVPR2024/papers/Huang_DeconfuseTrack_Dealing_with_Confusion_for_Multi-Object_Tracking_CVPR_2024_paper.pdf
- Why it matters:
  This is one of the closest papers to our current "ambiguity/confusion-focused" direction. It is not just another generic association paper. Its whole framing is that confusion itself should be modeled and reduced.
- Core idea:
  The paper decomposes confusion sources and reduces them through confusion-aware tracker components rather than relying on a single monolithic association score.
- Decision object:
  It is still operating at tracker association time, but the emphasis is on confusion decomposition and mitigation, not on runtime replay training.
- Overlap with our work:
  Very high on the problem motivation side. Both directions say that hard cases and confusion cases should not be treated the same as easy cases.
- Key difference from our work:
  Our strongest distinction is the training object:
  we want to learn from real runtime candidate groups, while DeconfuseTrack is not built around runtime replay supervision.
- What to focus on while reading:
  Read its problem definition and ablations carefully. The useful question is:
  how exactly do they define "confusion", and how do they prove that confusion-specific handling matters?
- Reviewer risk:
  If our paper says only "we handle ambiguity/confusion better", reviewers can say this paper already does that. Our claim must therefore emphasize runtime replay and safe reranking on real candidate groups.

### 1.2 Focusing on Tracks for Online Multi-Object Tracking (TrackTrack)

- Venue: CVPR 2025
- Official paper: https://openaccess.thecvf.com/content/CVPR2025/papers/Shim_Focusing_on_Tracks_for_Online_Multi-Object_Tracking_CVPR_2025_paper.pdf
- Why it matters:
  This is a strong recent paper that questions the default "frame-centric matching matrix plus Hungarian" habit and instead emphasizes track-focused online association logic.
- Core idea:
  The tracker prioritizes active tracks and association stability, especially during difficult online decisions such as track initialization and hard associations.
- Decision object:
  It is not a plug-in reranker in our exact sense. It rethinks the online association logic itself from a track-centered perspective.
- Overlap with our work:
  High on the insight that not all associations should be treated symmetrically and that hard online cases need special handling.
- Key difference from our work:
  TrackTrack is mainly about redesigning the online matching procedure. Our current paper is narrower: keep a strong host fixed and add a safe ambiguity-triggered plugin.
- What to focus on while reading:
  Pay attention to how they justify moving away from a purely detection-centered global matching view.
- Reviewer risk:
  If we only claim "we improve hard online associations", TrackTrack will be a very natural comparator.

### 1.3 Modelling Ambiguous Assignments for Multi-Person Tracking in Crowds

- Venue: WACV 2022 Workshop
- Official paper: https://openaccess.thecvf.com/content/WACV2022W/HADCV/papers/Stadler_Modelling_Ambiguous_Assignments_for_Multi-Person_Tracking_in_Crowds_WACVW_2022_paper.pdf
- Why it matters:
  This paper is very close in spirit to the "ambiguity-triggered" part of our current direction.
- Core idea:
  Separate ambiguous assignment cases from ordinary ones and handle them explicitly rather than running one uniform matching procedure.
- Decision object:
  Assignment ambiguity set.
- Overlap with our work:
  High on the idea that ambiguous cases deserve a different decision rule.
- Key difference from our work:
  It is not built around runtime-replay-trained reranking on real candidate groups.
- What to focus on while reading:
  Read how they define ambiguous assignments and how they partition the problem.
- Reviewer risk:
  If we present "ambiguity-aware tracking" as our novelty, this paper weakens that claim. We need the stronger and more specific claim:
  runtime-replay-trained safe reranking on real candidate groups.

### 1.4 Deep OC-SORT: Multi-Pedestrian Tracking by Adaptive Re-Identification

- Venue: ICIP 2023
- Official paper page: https://resourcecenter.ieee.org/conferences/icip-2023/spsicip23vid0663
- ArXiv: https://arxiv.org/abs/2302.11813
- Why it matters:
  This is not a CVPR/ICCV/ECCV main-track paper, but it is one of the most relevant "strong host + adaptive association refinement" references.
- Core idea:
  Build on a high-performing motion-based tracker and adaptively integrate appearance cues instead of using fixed heuristic appearance fusion.
- Decision object:
  Association-time adaptive cue integration inside a strong online host.
- Overlap with our work:
  Very high in engineering philosophy:
  do not rebuild the whole tracker; improve association decisions within a strong host.
- Key difference from our work:
  Deep OC-SORT is mainly adaptive appearance integration. Our current direction is runtime replay and ambiguity-aware candidate-group reranking.
- What to focus on while reading:
  Study how they justify a plug-in style contribution and how they isolate the value of adaptive association.
- Reviewer risk:
  Reviewers may ask whether our method is just another adaptive association add-on. We need to answer:
  no, the central novelty is the runtime replay object and safe ambiguity-triggered reranking.

### 1.5 Towards Generalizable Multi-Object Tracking (GeneralTrack)

- Venue: CVPR 2024
- Official page: https://openaccess.thecvf.com/content/CVPR2024/html/Qin_Towards_Generalizable_Multi-Object_Tracking_CVPR_2024_paper.html
- PDF: https://openaccess.thecvf.com/content/CVPR2024/papers/Qin_Towards_Generalizable_Multi-Object_Tracking_CVPR_2024_paper.pdf
- Why it matters:
  This is one of the most important papers for our current debugging situation because we are already seeing random-split gains but unstable cross-sequence gains.
- Core idea:
  Existing trackers overfit to scenario-specific balances of motion and appearance. The paper pushes toward more instance-wise relation modeling and better generalization.
- Decision object:
  Instance-wise relation rather than hand-balanced cue fusion.
- Overlap with our work:
  High on the generalization problem and on skepticism toward hand-tuned balancing.
- Key difference from our work:
  GeneralTrack is a broader framework-level proposal; our current work is a narrower association-time reranking plugin.
- What to focus on while reading:
  Read their diagnosis of why trackers generalize poorly.
- Reviewer risk:
  If our replay-based reranker works only on random splits and fails under leave-one-sequence-out, this paper becomes a direct weakness signal for us.

---

## 2. Learnable Association / Proposal Scoring / Relation Modeling

These papers matter because they define the broader learned-association landscape. They are the main novelty competitors if we say "we propose a learnable association module."

### 2.1 Multiple Object Tracking as ID Prediction (MOTIP)

- Venue: CVPR 2025
- Official paper: https://openaccess.thecvf.com/content/CVPR2025/papers/Gao_Multiple_Object_Tracking_as_ID_Prediction_CVPR_2025_paper.pdf
- Why it matters:
  This is one of the strongest recent learned-association papers. It reconceptualizes MOT as in-context ID prediction.
- Core idea:
  Treat object association as predicting identity labels for current detections conditioned on trajectory context, instead of relying on handcrafted online matching rules.
- Decision object:
  Identity prediction for current detections given trajectory context.
- Overlap with our work:
  Both directions reject the idea that heuristic matching is the final answer.
- Key difference from our work:
  MOTIP is much more end-to-end and ambitious. Our current direction is intentionally narrower and safer: frozen strong host plus plug-in reranker.
- What to focus on while reading:
  Focus on how they formulate association in a fully learnable way and how they distinguish themselves from ordinary tracking-by-detection heuristics.
- Reviewer risk:
  If we call our work "learnable association" without narrowing the claim, MOTIP will dominate that narrative.

### 2.2 LA-MOTR: End-to-End Multi-Object Tracking by Learnable Association

- Venue: ICCV 2025
- Official page: https://openaccess.thecvf.com/content/ICCV2025/html/Wang_LA-MOTR_End-to-End_Multi-Object_Tracking_by_Learnable_Association_ICCV_2025_paper.html
- PDF: https://openaccess.thecvf.com/content/ICCV2025/papers/Wang_LA-MOTR_End-to-End_Multi-Object_Tracking_by_Learnable_Association_ICCV_2025_paper.pdf
- Why it matters:
  The title alone tells you the novelty battlefield. This is a direct reminder that "learnable association" is already a crowded claim.
- Core idea:
  End-to-end MOT with learnable association replacing more classical hand-crafted association logic.
- Decision object:
  Learnable association inside an end-to-end transformer-style tracker.
- Overlap with our work:
  Shared high-level concern: association should be learned, not fully heuristic.
- Key difference from our work:
  Our current direction is not to out-compete end-to-end trackers on architecture scale. It is to provide a stronger training object and a safer plug-in for strong hosts.
- What to focus on while reading:
  Read how they frame the limits of heuristic association.
- Reviewer risk:
  This paper makes it dangerous for us to say our novelty is simply "learnable association."

### 2.3 Learning a Proposal Classifier for Multiple Object Tracking (LPC-MOT)

- Venue: CVPR 2021
- Official page: https://openaccess.thecvf.com/content/CVPR2021/html/Dai_Learning_a_Proposal_Classifier_for_Multiple_Object_Tracking_CVPR_2021_paper.html
- PDF: https://openaccess.thecvf.com/content/CVPR2021/papers/Dai_Learning_a_Proposal_Classifier_for_Multiple_Object_Tracking_CVPR_2021_paper.pdf
- Why it matters:
  This is one of the most relevant proposal-level MOT papers for our direction.
- Core idea:
  Instead of only scoring local pairwise links, model MOT as proposal generation, proposal scoring, and trajectory inference on an affinity graph.
- Decision object:
  Candidate trajectory proposal.
- Overlap with our work:
  Strong overlap in spirit. Both directions move beyond simple pairwise score calibration and care about higher-order candidate structure.
- Key difference from our work:
  LPC-MOT operates more at trajectory-proposal level, while our current direction is detection-centered runtime candidate-group reranking.
- What to focus on while reading:
  Look at why proposal scoring is more expressive than pairwise affinity calibration.
- Reviewer risk:
  If we only say "we use higher-order candidate context," reviewers may view our work as a narrower proposal-scoring variant.

### 2.4 Learning a Neural Solver for Multiple Object Tracking

- Venue: CVPR 2020
- Official paper: https://openaccess.thecvf.com/content_CVPR_2020/papers/Braso_Learning_a_Neural_Solver_for_Multiple_Object_Tracking_CVPR_2020_paper.pdf
- Why it matters:
  This is a foundational graph-based learned tracking paper. It is older than the recent transformer line, but still important for framing.
- Core idea:
  Learn graph edge decisions with message passing so that the model reasons globally over detections and edges instead of only local pairwise similarity.
- Decision object:
  Edge classification in a detection graph.
- Overlap with our work:
  Shared rejection of naive pairwise-only reasoning.
- Key difference from our work:
  The Neural Solver learns over a larger graph optimization structure; we are aiming at a safer, host-compatible reranking plug-in.
- What to focus on while reading:
  The useful lesson is not the exact network, but how the paper argues for graph-structured global reasoning.

### 2.5 Learnable Graph Matching: Incorporating Graph Partitioning With Deep Feature Learning for Multiple Object Tracking (GMTracker)

- Venue: CVPR 2021
- Official paper: https://openaccess.thecvf.com/content/CVPR2021/papers/He_Learnable_Graph_Matching_Incorporating_Graph_Partitioning_With_Deep_Feature_Learning_CVPR_2021_paper.pdf
- ArXiv: https://arxiv.org/abs/2103.16178
- Why it matters:
  This paper explicitly argues that context among tracklets and detections matters and that learning should be aligned with optimization.
- Core idea:
  Formulate tracklet-detection association as graph matching and make the optimization differentiable through relaxation.
- Decision object:
  General graph matching between tracklet graph and detection graph.
- Overlap with our work:
  Shared belief that pairwise decisions alone are too weak in difficult cases.
- Key difference from our work:
  GMTracker is graph-matching-centric and optimization-heavy; our direction is more practical and modular, centered on runtime replay candidate groups.
- What to focus on while reading:
  Read their diagnosis of why separate feature learning and graph optimization become inconsistent.

### 2.6 TrackFlow: Multi-Object Tracking with Normalizing Flows

- Venue: ICCV 2023
- Official page: https://openaccess.thecvf.com/content/ICCV2023/html/Mancusi_TrackFlow_Multi-Object_tracking_with_Normalizing_Flows_ICCV_2023_paper.html
- PDF: https://openaccess.thecvf.com/content/ICCV2023/papers/Mancusi_TrackFlow_Multi-Object_tracking_with_Normalizing_Flows_ICCV_2023_paper.pdf
- Why it matters:
  This paper is useful if we want a stronger probabilistic interpretation of association quality instead of ad hoc score fusion.
- Core idea:
  Use normalizing flows to model the distribution of correct associations more flexibly than fixed heuristic combinations.
- Decision object:
  Association likelihood / matching distribution.
- Overlap with our work:
  Both directions try to replace hand-engineered association confidence with learned modeling.
- Key difference from our work:
  TrackFlow is a learned probabilistic association model; our current direction is more constrained and host-compatible.
- What to focus on while reading:
  The key question is how they justify probabilistic association modeling and whether that helps generalization.

### 2.7 FAMNet: Joint Learning of Feature, Affinity and Multi-dimensional Assignment

- Venue: ICCV 2019
- Official paper: https://openaccess.thecvf.com/content_ICCV_2019/papers/Chu_FAMNet_Joint_Learning_of_Feature_Affinity_and_Multi-Dimensional_Assignment_for_ICCV_2019_paper.pdf
- Why it matters:
  This is an older but important reference showing that feature learning, affinity estimation, and assignment can be learned jointly.
- Core idea:
  Instead of treating feature, affinity, and assignment as isolated modules, optimize them together.
- Decision object:
  Joint feature-affinity-assignment pipeline.
- Overlap with our work:
  It is part of the historical lineage behind the idea that association should be learned rather than fully heuristic.
- Key difference from our work:
  Our current direction is much narrower and more modular, focused on strong-host plug-ins rather than end-to-end redesign.
- What to focus on while reading:
  Use it mainly as historical context, not as the closest baseline.

---

## 3. Supporting Papers on Host Strength, Memory, and Appearance

These are not the closest papers to runtime replay reranking, but they matter because they explain where MOT gains often actually come from: better appearance, better memory, better host design, or better generalization.

### 3.1 MeMOT: Multi-Object Tracking With Memory

- Venue: CVPR 2022
- Official page: https://openaccess.thecvf.com/content/CVPR2022/html/Cai_MeMOT_Multi-Object_Tracking_With_Memory_CVPR_2022_paper.html
- PDF: https://openaccess.thecvf.com/content/CVPR2022/papers/Cai_MeMOT_Multi-Object_Tracking_With_Memory_CVPR_2022_paper.pdf
- ArXiv: https://arxiv.org/abs/2203.16761
- Why it matters:
  This paper is important for the "history encoder" side of our project.
- Core idea:
  Use explicit memory to improve long-term association and maintain more stable identity information over time.
- Decision object:
  Transformer-style memory-enhanced tracking.
- Overlap with our work:
  If we later reintroduce a learnable history branch into our reranker, this is one of the best references.
- Key difference from our work:
  MeMOT is not a frozen-host plug-in and not centered on runtime replay candidate groups.
- What to focus on while reading:
  Read how they motivate memory and how they separate history representation from current-frame association.

### 3.2 Quasi-Dense Similarity Learning for Multiple Object Tracking (QDTrack)

- Venue: CVPR 2021
- Official page: https://openaccess.thecvf.com/content/CVPR2021/html/Pang_Quasi-Dense_Similarity_Learning_for_Multiple_Object_Tracking_CVPR_2021_paper.html
- PDF: https://openaccess.thecvf.com/content/CVPR2021/papers/Pang_Quasi-Dense_Similarity_Learning_for_Multiple_Object_Tracking_CVPR_2021_paper.pdf
- Why it matters:
  This is one of the strongest reminders that many MOT gains come from stronger appearance supervision, not necessarily from ever more complicated association heads.
- Core idea:
  Learn much richer similarity supervision by sampling many region proposals rather than sparse matched pairs only.
- Decision object:
  Dense appearance similarity learning for MOT.
- Overlap with our work:
  It strongly supports the argument that the quality of the host feature space matters a lot.
- Key difference from our work:
  QDTrack is about learning better matching features, not about runtime candidate-group reranking.
- What to focus on while reading:
  Read it as a warning and as a support:
  sometimes stronger ReID or similarity learning helps more than clever association heads.

### 3.3 Deep OC-SORT is also important here

- Reason to repeat:
  Even though it appears above in the "most direct" section, it also belongs here conceptually.
- Why:
  It is a clean example of a paper whose gains come from a better strong-host association refinement story rather than a brand-new end-to-end framework.

---

## 4. Recent Watchlist

These are worth watching, but should be cited carefully based on publication status.

### 4.1 UniTrack: Differentiable Graph Representation Learning for Multi-Object Tracking

- Status:
  recent 2026 work; OpenReview page and arXiv are available, but publication status should be checked carefully before final citation wording
- OpenReview: https://openreview.net/forum?id=XpddZpGck9
- ArXiv: https://arxiv.org/abs/2602.05037
- Why it matters:
  It proposes a graph-theoretic plug-and-play loss for MOT and claims improvements across multiple host trackers.
- Core idea:
  Improve MOT by directly optimizing graph-structured objectives related to detection, temporal consistency, and identity preservation.
- Relevance to us:
  Conceptually relevant because it is also modular and host-compatible, but its focus is a training objective rather than a runtime candidate-group reranker.
- Citation caution:
  Before using it in the paper, verify the exact venue status you want to claim in the bibliography text.

---

## 5. Recommended Reading Order

If you want to read in the most useful order for our project, do this:

1. DeconfuseTrack
2. TrackTrack
3. GeneralTrack
4. LPC-MOT
5. MOTIP
6. Deep OC-SORT
7. MeMOT
8. QDTrack
9. GMTracker
10. Neural Solver
11. LA-MOTR
12. TrackFlow
13. FAMNet
14. Ambiguous Assignments in Crowds
15. UniTrack

Why this order:

- The first three sharpen the problem statement:
  confusion, hard cases, and generalization.
- The next three show stronger learnable-association alternatives:
  proposal scoring, ID prediction, adaptive association.
- The next two explain where host-side gains often really come from:
  memory and appearance.
- The last set fills in broader graph / end-to-end / probabilistic context.

---

## 6. What These Papers Mean for Our Current Paper

After reading this literature, the safest and strongest positioning for our project is:

- not "we propose a better pairwise association score"
- not "we propose a Laplace/frequency module for MOT"
- not simply "we model ambiguity"

The positioning that still has room is:

- strong host
- real runtime candidate groups
- ambiguity-triggered activation
- bounded or safety-aware reranking
- plugin on/off gains under a frozen host

In one sentence:

> Our likely paper niche is not generic learnable association, but runtime-replay-trained safe reranking for ambiguous candidate groups on strong MOT hosts.

That is the point of this reading list: to make it clear where the literature is crowded and where our remaining room still exists.

