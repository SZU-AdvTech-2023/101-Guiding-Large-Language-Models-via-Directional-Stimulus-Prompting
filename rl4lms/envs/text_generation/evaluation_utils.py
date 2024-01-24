from typing import Any, Dict, List

from stable_baselines3.common.policies import BasePolicy
from tqdm import tqdm
from transformers import AutoTokenizer

from rl4lms.data_pools.custom_text_generation_pools import Sample
from rl4lms.envs.text_generation.logging_utils import Tracker
from rl4lms.envs.text_generation.metric import BaseMetric


def get_batch(samples: List[Sample], batch_size: int):
    current_ix = 0
    n_samples = len(samples)
    while current_ix < n_samples:
        current_batch = samples[current_ix : current_ix + batch_size]
        yield current_batch
        current_ix += batch_size


def evaluate_on_samples(
    policy: BasePolicy,
    tokenizer: AutoTokenizer,
    samples: List[Sample],
    batch_size: int,
    max_prompt_length: int,
    metrics: List[BaseMetric],
    epoch: int,
    split_name: str,
    tracker: Tracker = None,
    dt_control_token: str = "",
    gen_kwargs: Dict[str, Any] = None,
    llm_tokenizer=None,
    llm_pipeline=None,
):
    # generate text by batch
    all_generated_texts = []
    all_ref_texts = []
    all_prompt_texts = []
    all_meta_infos = []
    n_samples = len(samples)
    for batch in tqdm(list(get_batch(samples, batch_size)), desc="Evaluating"):
        batch_generated_texts = generate_text(
            policy, tokenizer, batch, max_prompt_length, dt_control_token, gen_kwargs
        )
        batch_ref_texts = [sample.references for sample in batch]
        batch_prompt_texts = [sample.prompt_or_input_text for sample in batch]
        batch_meta_infos = [sample.meta_data for sample in batch]
        all_generated_texts.extend(batch_generated_texts)
        all_ref_texts.extend(batch_ref_texts)
        all_prompt_texts.extend(batch_prompt_texts)
        all_meta_infos.extend(batch_meta_infos)

    # compute metrics
    corpus_level_metrics = {}
    sample_scores_by_metric = {}
    if metrics is not None:
        for metric in metrics:
            if hasattr(metric, "use_llm") and metric.use_llm == True:
                metric_dict = metric.compute(
                    all_prompt_texts,
                    all_generated_texts,
                    all_ref_texts,
                    all_meta_infos,
                    policy.get_language_model(),
                    split_name,
                    llm_tokenizer,
                    llm_pipeline,
                )
            else:
                metric_dict = metric.compute(
                    all_prompt_texts,
                    all_generated_texts,
                    all_ref_texts,
                    all_meta_infos,
                    policy.get_language_model(),
                    split_name,
                )
            print("compute结束了") # 以上都能正常运行

            i=1
            for metric_key, (sample_scores, corpus_score) in metric_dict.items():
                print("进入metric_key这个循环了，此时i=",i)
                i=i+1
                print("sample_scores的维度是：",len(sample_scores))  # 5
                if sample_scores is None:
                    sample_scores = ["n/a"] * n_samples
                corpus_level_metrics[metric_key] = corpus_score
                sample_scores_by_metric[metric_key] = sample_scores
            print("跳出metric_key这个循环了")
        print("跳出metric这个循环了！")

    # aggregate sample metric scores
    sample_predictions_dict = []
    for ix, (sample, prompt_text, generated_text, ref_texts) in enumerate(
        zip(samples, all_prompt_texts, all_generated_texts, all_ref_texts)
    ):
        print("进入到aggregate这个循环，此时ix=",ix)
        sample_prediction = {
            "split_name": split_name,
            "sample_id": sample.id,
            "prompt_text": prompt_text,
            "generated_text": generated_text,
            "ref_text": "".join(
                [
                    f"<START-{ref_ix+1}>" + ref_text + f"<END-{ref_ix+1}>"
                    for ref_ix, ref_text in enumerate(ref_texts)
                ]
            ),
        }
        #sample_score_by_metric按理来说是10
        for metric_key, sample_scores in sample_scores_by_metric.items():
            print("进到sample_score_by_metric循环，此时metric_key=",metric_key)
            print("sample_scores的长度为：",len(sample_scores)) #5
            # print("sample_scores=",sample_scores)
            sample_prediction[metric_key] = sample_scores[ix] # 报错位置，IndexError: list index out of range
            print("sample_prediction[metric_key]=",sample_prediction[metric_key])
        sample_predictions_dict.append(sample_prediction)

    if tracker is not None:
        # log the entire predictions
        tracker.log_predictions(epoch, split_name, sample_predictions_dict)
        # log the corpus level scores
        tracker.log_metrics(epoch, split_name, corpus_level_metrics)


def generate_text(
    policy: BasePolicy,
    tokenizer: AutoTokenizer,
    samples: List[Sample],
    max_prompt_length: int,
    dt_control_token: str,
    gen_kwargs: Dict[str, Any],
):
    prompt_texts = [
        dt_control_token + sample.prompt_or_input_text for sample in samples
    ]
    generated_texts = policy.generate(
        tokenizer, prompt_texts, max_prompt_length, gen_kwargs=gen_kwargs
    ).gen_texts
    return generated_texts
