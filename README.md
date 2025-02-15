
## DOCUMENTATION & EVALUATION

Evaluation and Usage Documentation can be browsed in a cleaner format [here](https://transposedai.notion.site/MasterClass-Retrieval-Docs-6063484218a44940b6b3325b69d92597?pvs=4) on the dedicated Notion page.


### DOCUMENTATION

## Structure

---

---

> *This package prefers to be run in a Unix-like environment, preferably with CUDA GPU support during **evaluation tasks**. Windows has only been partially tested with AutoRAG or this package, and it is advisable to use WSL2 if you are on that OS.*
> 

![Untitled](https://transposedai.notion.site/image/https%3A%2F%2Fprod-files-secure.s3.us-west-2.amazonaws.com%2F25525b37-b834-4bb3-a77d-41ae99667998%2F12979360-07a3-44c3-897b-634e99af86b4%2FUntitled.png?table=block&id=e5ebfa36-7290-4309-85d7-813ae02d9fd7&spaceId=25525b37-b834-4bb3-a77d-41ae99667998&width=1420&userId=&cache=v2)

### Usage

<aside>
⚓ This package is **containerizable**, and can be built via the **Dockerfile** provided.

</aside>

**pip** is the preferred package manager for this project, and dependencies can be installed like so

```bash
	pip install -r requirements_m3.txt
```

To run the **AutoRAG evaluation dashboard** for a given run/trial, **for example for run3, trial 7**:

```bash
autorag dashboard --trial_dir run3/7
```

### Core Directory Structure

**eval.py**

- Contains all response generations across different RAG pipeline configuration points (V1 and V2)

**documentProcessing.py**

- Executes the HDP chunking strategies and vectorDB upsert for Pinecone

**generateResponses.py**

- Creates Ragas and AutoRAG trial responses for the provided dataset, handling formatting of the output dataframe and general

**Dockerfile** 

- Executes containerization with **docker**

**/util**

- Folder for all backing corpus’ and datasets generated/utilized

**/eval**

- Folder for all output responses generated for the provided input chatlog queries

**/config**

- Configurations for each AutoRAG run pipeline

**/run*** 

- Each of the project run folders generated by AutoRAG

### Core References

AutoRAG: https://docs.auto-rag.com/install.html

Ragas: https://docs.ragas.io/en/stable/getstarted/index.html

# Considerations

---

---

### **Chunking & Embedding**

Given that it seems these content descriptions per row will be insanely large, it makes sense to chunk prior to embedding.

Considerations:

- We need to maintain the metadata across the chunked sections
- We need to chunk in a manner that makes sense for each content type/text (HDP may work well here, but we should test methods)

**Choice:** We will chunk across each content entry separately, and generate embeddings per content entry segment

**Why: *We are prioritizing matching and clustering within the same topic**,* rather than the versatility and the ability to compare or link between topics.

**Choice: We are using HDP for text topic splitting.**

### Supplementary Information

Context Response Length Averages (for Example Output Provided)

**avg_context_size:** 1917 char, 352 words

*~**x7** and ~**x4** times the size of my avg. output context lengths for run3 and runv5 respectively*

General RAG Pipeline Flow (Example):

- **Chunking:** HDP + Dominant Topic Chunking (in parallel) (seg=5)
- **Embedding:** openai-small
- **VectorDB:** Chroma (local)
- **QA Generation: AutoRAG + Ragas**  (~50q, Adversarial Critique)
- **Retrieval:** hybrid-rrf (bm25+vectordb) (topk:3)
- **Passage_Reranker:** tart (topk:2)
- **Passage_Filter:** threshold_cutoff

# Evaluation: Overview

---

### **V1 Evaluation**

Description: 

**Variable** **question QA synth-dataset** built by **Ragas** from the **entire corpus**, but limited to **50** questions, and built on Pinecone *similarity_search* retrieval. **Evaluated** by Ragas. 

LLM: gpt-4o - generation, gpt-4-turbo - critique

Chunking:

- HDP + Dominant Topic Text Chunking - in series *(processing time: ~45min-1hr)*

### **V2 Evaluation (run 1)**

Description:  ****

**Variable question QA synth-dataset** built by **AutoRAG + Ragas** of **52** questions from **5** different document bundles tied to independent source material. **Corpus’** used for QA synth were segregated by document, meaning *individual bundle embedding.* 

LLM: gpt-4o (qa critique & generation)

Chunking:

- HDP + Dominant Topic Text Chunking - in parallel *(processing time: ~8min)*

### **V3 Evaluation (run 5 - Used for Response Generation)**

Description: 

**Multi-context** **only QA synth-dataset** built by **AutoRAG + Ragas** of **52** questions from **5** different document bundles tied to independent source material. Corpus’ used for QA synth were segregated by document, meaning *individual bundle embedding.*
**Moreover, prompt condensing was utilized prior to response generation on the input test set,**

LLM: gpt-4o (qa critique & generation)

Chunking:

- HDP + Dominant Topic Text Chunking - in parallel *(processing time: ~14min)*

<aside>
To reiterate the QA Generation approach for V2 onwards:

- Takes the master corpus (in autoRAG format) and randomly selects all documents across **5** different int_id’s to create **5** child df corpus’
- Run QA df generation on each of the **5** child corpus’ with **multi-context.**
</aside>

# Evaluation: Metrics

---

## V1: Ragas Evaluation (best scoring)

<aside>
*Chunking - Avg. Context Length**:***

- 311 char, 54 words

*QA generation:* 

- *Adversarial LLM evolution - Ragas*
- Using the following question generation distribution:
    - `{simple: 0.5, reasoning: 0.4, multi_context: 0.10}`
</aside>

## *Retrieval Scoring (V1 - Ragas)*

![Screenshot 2024-06-21 at 7.49.28 PM.png](https://transposedai.notion.site/image/https%3A%2F%2Fprod-files-secure.s3.us-west-2.amazonaws.com%2F25525b37-b834-4bb3-a77d-41ae99667998%2F71bf0c9b-9de3-44cd-81b1-47a5cc5b97eb%2FUntitled.png?table=block&id=5f61c709-896e-4863-b035-26a4c4dbe9dc&spaceId=25525b37-b834-4bb3-a77d-41ae99667998&width=2000&userId=&cache=v2)

## V2: run 1, trial 0 (best scoring)

<aside>

*Chunking - Avg. Context Length**:***

- 279 char, 49 words

*QA generation:* 

- *Adversarial LLM evolution, Ragas + AutoRAG*
- Using the following question generation distribution:
    - `{simple: 0.1, multi_context: 0.9}`
- Generated from **5** document int_id’s randomly targeted, with a backing corpus summed of each of those
</aside>

<aside>
 *This trial is special:* 
To test embedding flexibility, the backing corpus is a union of all items from each of the **5** document **int_id** collections, making it a *subset* of the larger dataset. This is to mimic what a more verbose QA synthesis across the dataset would appear to be.
</aside>

## *Retrieval Scoring (run1, trial 0)*

![Untitled](https://transposedai.notion.site/image/https%3A%2F%2Fprod-files-secure.s3.us-west-2.amazonaws.com%2F25525b37-b834-4bb3-a77d-41ae99667998%2F71bf0c9b-9de3-44cd-81b1-47a5cc5b97eb%2FUntitled.png?table=block&id=5f61c709-896e-4863-b035-26a4c4dbe9dc&spaceId=25525b37-b834-4bb3-a77d-41ae99667998&width=2000&userId=&cache=v2)

### Metric Values

| metric_name | metric_value |
| --- | --- |
| retrieval_f1 | 0.5750000000000001 |
| retrieval_recall | 0.9464285714285714 |
| retrieval_precision | 0.4285714285714285 |
| retrieval_ndcg | 0.7537812703304715 |
| retrieval_map | 0.8110119047619048 |
| retrieval_mrr | 0.8095238095238095 |

## **run 3, trial 7 (best scoring while inclusive)**

<aside>
*Chunking size:* 3 sentences minimum

*Chunking - Avg. Context Length**:***

- 289 char, 52 words

*QA generation:* 

- *Adversarial LLM evolution, Ragas + AutoRAG*
- Using the following question generation distribution *(100% multi-context)*:
    - `{multi_context: 1.0}`
- Generated from **5** document int_id’s randomly targeted. Corpus is provided in its entirety for eval.
</aside>

## *Retrieval Scoring (run3, trial7)*

![Untitled](https://transposedai.notion.site/image/https%3A%2F%2Fprod-files-secure.s3.us-west-2.amazonaws.com%2F25525b37-b834-4bb3-a77d-41ae99667998%2F717d4339-aab1-446c-9eb0-870c4af2cb33%2FUntitled.png?table=block&id=5634f323-e762-4a90-9251-50f8dfffb237&spaceId=25525b37-b834-4bb3-a77d-41ae99667998&width=2000&userId=&cache=v2)

### Metric Values

| metric_name | metric_value |
| --- | --- |
| retrieval_f1 | 0.3192273135669363 |
| retrieval_recall | 0.8113207547169812 |
| retrieval_precision | 0.2066037735849056 |
| retrieval_ndcg | 0.4722147173662014 |
| retrieval_map | 0.4820754716981131 |
| retrieval_mrr | 0.4987421383647798 |

## *Passage Reranker Scoring (run 3, trial 7)*

![X axis for passage_reranker_retrieval scoring, from left to right: **f1, recall, ndcg, map, mrr**](https://transposedai.notion.site/image/https%3A%2F%2Fprod-files-secure.s3.us-west-2.amazonaws.com%2F25525b37-b834-4bb3-a77d-41ae99667998%2Fba601da6-0c50-4fe5-984d-162afc91e971%2FUntitled.png?table=block&id=9690a0c8-dbc0-4c4d-b3b8-1bdc286f7be4&spaceId=25525b37-b834-4bb3-a77d-41ae99667998&width=2000&userId=&cache=v2)

X axis for passage_reranker_retrieval scoring, from left to right: **f1, recall, ndcg, map, mrr**

### Metric Values

| metric_name | metric_value |
| --- | --- |
| passage_reranker_retrieval_f1 | 0.5283018867924529 |
| passage_reranker_retrieval_recall | 0.7547169811320755 |
| passage_reranker_retrieval_precision | 0.4150943396226415 |
| passage_reranker_retrieval_ndcg | 0.5439271253783649 |
| passage_reranker_retrieval_map | 0.6981132075471698 |
| passage_reranker_retrieval_mrr | 0.6981132075471698 |

## *Passage Filter Scoring (run3, trial7)*

![X axis for passage_filter_retrieval scoring, from left to right: **f1, recall, ndcg, map, mrr**](https://transposedai.notion.site/image/https%3A%2F%2Fprod-files-secure.s3.us-west-2.amazonaws.com%2F25525b37-b834-4bb3-a77d-41ae99667998%2Fd37b6c71-a016-41bd-b2e7-5951f9503690%2FUntitled.png?table=block&id=cb73a1d4-5db8-4bab-b5f8-32dab3d40905&spaceId=25525b37-b834-4bb3-a77d-41ae99667998&width=2000&userId=&cache=v2)

X axis for passage_filter_retrieval scoring, from left to right: **f1, recall, ndcg, map, mrr**

### Passage Filter - Metric Values

| metric_name | metric_value |
| --- | --- |
| passage_filter_retrieval_f1 | 0.6415094339622641 |
| passage_filter_retrieval_recall | 0.6792452830188679 |
| passage_filter_retrieval_precision | 0.6226415094339622 |
| passage_filter_retrieval_ndcg | 0.6607128785057735 |
| passage_filter_retrieval_map | 0.660377358490566 |
| passage_filter_retrieval_mrr | 0.660377358490566 |

## **V3: runv5, trial 0 (Best Test-Set Output Relevance)**

<aside>
*Chunking size:* 5 sentences minimum

*Chunking - Avg. Context Length**:***

- 520 char, 90.5 words

*QA generation:* 

- *Adversarial LLM evolution, Ragas + AutoRAG*
- Across segregated index (segregated by 5 random int_id’s)
- Using the following question generation distribution:
    - `{simple: 0.25, reasoning: 0.45, multi_context: 0.30}`
</aside>

## *Retrieval Scoring (runv5, trial 0)*

![Untitled](https://transposedai.notion.site/image/https%3A%2F%2Fprod-files-secure.s3.us-west-2.amazonaws.com%2F25525b37-b834-4bb3-a77d-41ae99667998%2F4452b69b-3bca-435d-a923-b2df833de6b2%2FUntitled.png?table=block&id=cd58a406-58ce-409a-8eca-1d90a9a59556&spaceId=25525b37-b834-4bb3-a77d-41ae99667998&width=2000&userId=&cache=v2)

### Retrieval Scoring - Metric Values

| metric_name | metric_value |
| --- | --- |
| retrieval_f1 | 0.2092592592592592 |
| retrieval_recall | 0.4074074074074074 |
| retrieval_precision | 0.1419753086419752 |
| retrieval_ndcg | 0.2625719103072731 |
| retrieval_map | 0.3333333333333333 |
| retrieval_mrr | 0.3364197530864197 |

## *Passage Reranker (runv5, trial 0)*

![X axis for passage_reranker_retrieval scoring, from left to right: **f1, recall, ndcg, map, mrr**](https://transposedai.notion.site/image/https%3A%2F%2Fprod-files-secure.s3.us-west-2.amazonaws.com%2F25525b37-b834-4bb3-a77d-41ae99667998%2F9376471c-071e-4465-a5f8-057ce118a5fd%2FUntitled.png?table=block&id=f0df7d04-f200-4811-be0a-72dc5e2b64bf&spaceId=25525b37-b834-4bb3-a77d-41ae99667998&width=2000&userId=&cache=v2)

X axis for passage_reranker_retrieval scoring, from left to right: **f1, recall, ndcg, map, mrr**

### Passage Reranker - Metric Values

| metric_name | metric_value |
| --- | --- |
| passage_reranker_retrieval_f1 | 0.2530864197530864 |
| passage_reranker_retrieval_recall | 0.3703703703703703 |
| passage_reranker_retrieval_precision | 0.1944444444444444 |
| passage_reranker_retrieval_ndcg | 0.2578325944986975 |
| passage_reranker_retrieval_map | 0.3333333333333333 |
| passage_reranker_retrieval_mrr | 0.3333333333333333 |

## *Passage Filter Scoring (run3, trial7)*

![X axis for passage_filter_retrieval scoring, from left to right: **f1, recall, ndcg, map, mrr**](https://transposedai.notion.site/image/https%3A%2F%2Fprod-files-secure.s3.us-west-2.amazonaws.com%2F25525b37-b834-4bb3-a77d-41ae99667998%2F0efa50e8-da7f-4f50-9dc1-638cd0e182ad%2FUntitled.png?table=block&id=a7dc43bb-8d3c-4d26-a212-97b6d344d301&spaceId=25525b37-b834-4bb3-a77d-41ae99667998&width=2000&userId=&cache=v2)

X axis for passage_filter_retrieval scoring, from left to right: **f1, recall, ndcg, map, mrr**

### Passage Filter - Metric Values

| metric_name | metric_value |
| --- | --- |
| passage_filter_retrieval_f1 | 0.2901234567901234 |
| passage_filter_retrieval_recall | 0.3333333333333333 |
| passage_filter_retrieval_precision | 0.2685185185185185 |
| passage_filter_retrieval_ndcg | 0.2933229921906736 |
| passage_filter_retrieval_map | 0.324074074074074 |
| passage_filter_retrieval_mrr | 0.324074074074074 |

---

## Evaluation Showcase: 
Adjusting Precision for *run 3, trial 7 (best - most inclusive)*

Given the limitations in generation of synthetic QA datasets with more than 2 context ID’s (even when utilizing multi-context generation flags), we should adjust our precision ratio to reflect the maximum number of correct options. 

This *does* ignore “wrong” context values, however given:

- the Precision@K of 3,
- the solid contextual retrieval characteristics of the RAG pipeline (human percieved), and
- the limitations of extending multi-context QA dataset synthesis past 2 results

It is *more* correct to assume that the other entries retrieved would provide correct relevance to the dataset, and should not be immediately discarded as inaccurate. 

Therefore, we adjust the evaluation dataset with the following Precision ratio rebalance, and recalculation of f1 (harmonic mean), in the following manner:

```python
adjusted_precision= []
adjusted_f1=[]
for index, f in enumerate(df['retrieval_precision'].tolist()):
    length = len(df['retrieved_ids'][index])
    precision = f* (length/2)
    adjusted_precision.append(precision)
    recall= df['retrieval_recall'][index]
    epsilon = 1e-7
    f1 = 2*((float(precision)*recall)/(float(precision)+recall+ epsilon))
    adjusted_f1.append(f1)

df['adjusted_precision'] = adjusted_precision
df['adjusted_f1'] = adjusted_f1

avg_precision = df['adjusted_precision'].mean()
avg_f1 = df['adjusted_f1'].mean()
avg_recall = df['retrieval_recall'].mean()
print("precision: " + str(avg_precision), "recall: " + str(avg_recall), "f1: " + str(avg_f1))
df.head()
```

**Using our selection “best”** performing project-trial-retrieval set (**run3 trial7**), we end up with the following **core averages** to **run3, trial 7:**

```python
avg_precision: 0.5094339622641509 
avg_recall: 0.8113207547169812 
avg_f1: 0.6100628558700233
```

<aside>
For the purposes of time, this will not be done across all entries. This showcase is designed to highlight issues with relying on Precision measurements across varied corpus’.

</aside>

##

### LICENSING

Per the adoption of the Creative Commons Attribution Non-Commercial No Derivatives 4.0 International
License, no **derivation, reproduction, utilization, or distribution** of this work is allowed in commercial contexts without explicit permission from the code author.

### ACKNOWLEDGEMENTS
Daniel Campbell - Author & Maintainer