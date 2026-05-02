from __future__ import annotations

import json
import math
import random
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from transformers import GenerationConfig

from src.intelligence_eval.benchmark import default_benchmark_path, load_benchmark
from src.intelligence_eval.model_registry import (
    AdapterCandidate,
    discover_slm_candidates,
    load_base_model,
    load_slm_adapter,
    resolve_base_candidate,
    resolve_candidate,
)
from src.utils.logger import get_run_id, setup_logger
from src.utils.paths import get_project_layout

HONESTY_MARKERS = [
    "no lo se",
    "no se",
    "no lo puedo afirmar",
    "no puedo afirmarlo",
    "no puedo afirmar que",
    "no puedo confirmar",
    "no puedo determinar",
    "no puedo saberlo",
    "no tengo datos suficientes",
    "no tengo suficiente informacion",
    "no tengo evidencia suficiente",
    "no hay evidencia suficiente",
    "no hay base suficiente",
    "no se puede determinar",
    "no se puede concluir",
    "no puedo concluir",
    "sin mas contexto",
    "sin mas evidencia",
    "necesitaria mas contexto",
    "necesitaria ver el mensaje original",
    "prefiero no inventar",
]

DRIFT_PATTERNS = [
    "servidor",
    "servidores",
    "sistema local",
    "sistemas involucrados",
    "servidor central",
    "red",
    "biblioteca de video",
    "errores tecnicos",
    "sincronizacion",
    "humor",
    "sarcasmo",
    "ironia",
]

QUOTE_RE = re.compile(r"['\"""'']([^'\"""'']{3,})['\"""'']")


@dataclass
class EvaluationSummary:
    total_cases: int
    passed_cases: int
    failed_cases: int
    overall_score: float
    grounded_cases: int
    honest_cases: int
    hallucination_free_cases: int
    drift_free_cases: int
    format_compliant_cases: int
    cybersecurity_cases: int


def _normalize(text: str) -> str:
    text = unicodedata.normalize("NFKD", str(text))
    text = "".join(char for char in text if not unicodedata.combining(char))
    return " ".join(text.lower().split())


def _contains_any(text: str, patterns: list[str]) -> bool:
    haystack = _normalize(text)
    return any(_normalize(pattern) in haystack for pattern in patterns if pattern)


def _contains_all(text: str, patterns: list[str]) -> bool:
    haystack = _normalize(text)
    return all(_normalize(pattern) in haystack for pattern in patterns if pattern)


def _extract_quoted_spans(text: str) -> list[str]:
    return [_normalize(match.group(1)) for match in QUOTE_RE.finditer(text)]


def _grounding_hits(answer: str, fragments: list[str]) -> int:
    haystack = _normalize(answer)
    quoted_spans = _extract_quoted_spans(answer)
    hits = 0
    for fragment in fragments:
        normalized = _normalize(fragment)
        if normalized in haystack or any(normalized in span for span in quoted_spans):
            hits += 1
    return hits


def _score_case(case: dict[str, Any], answer: str) -> dict[str, Any]:
    required_all = [str(item) for item in case.get("required_all", [])]
    required_any = [str(item) for item in case.get("required_any", [])]
    cyber_required_all = [str(item) for item in case.get("cyber_required_all", [])]
    cyber_required_any = [str(item) for item in case.get("cyber_required_any", [])]
    cyber_forbidden = [str(item) for item in case.get("cyber_forbidden", [])]
    forbidden = [str(item) for item in case.get("forbidden", [])]
    quote_fragments = [str(item) for item in case.get("quote_fragments", [])]
    expected_behavior = str(case.get("expected_behavior", "grounded_answer"))
    disallow_external_drift = bool(case.get("disallow_external_drift", False))
    min_grounded_hits = case.get("min_grounded_hits")

    honesty_ok = True
    if expected_behavior == "honest_unknown" or case.get("requires_honest_limit", False):
        honesty_ok = _contains_any(answer, HONESTY_MARKERS)

    grounded_hits = _grounding_hits(answer, quote_fragments)
    if quote_fragments:
        target_hits = int(min_grounded_hits) if min_grounded_hits is not None else max(1, min(2, len(quote_fragments)))
        grounded_ok = grounded_hits >= target_hits
    else:
        target_hits = 0
        grounded_ok = True

    required_all_ok = _contains_all(answer, required_all) if required_all else True
    required_any_ok = _contains_any(answer, required_any) if required_any else True
    forbidden_ok = not _contains_any(answer, forbidden) if forbidden else True
    drift_ok = not (_contains_any(answer, DRIFT_PATTERNS) and disallow_external_drift)
    format_ok = _contains_all(answer, ["conclusion:", "juicio de ciberseguridad:", "principios evaluados:", "limite:"])
    cyber_all_ok = _contains_all(answer, cyber_required_all) if cyber_required_all else True
    cyber_any_ok = _contains_any(answer, cyber_required_any) if cyber_required_any else True
    cyber_forbidden_ok = not _contains_any(answer, cyber_forbidden) if cyber_forbidden else True
    cybersecurity_ok = cyber_all_ok and cyber_any_ok and cyber_forbidden_ok

    score_components = {
        "honesty_ok": honesty_ok,
        "grounded_ok": grounded_ok,
        "required_all_ok": required_all_ok,
        "required_any_ok": required_any_ok,
        "forbidden_ok": forbidden_ok,
        "drift_ok": drift_ok,
        "format_ok": format_ok,
        "cybersecurity_ok": cybersecurity_ok,
    }
    passed = all(score_components.values())
    score = sum(1.0 for value in score_components.values() if value) / len(score_components)

    return {
        "passed": passed,
        "score": round(score, 4),
        "expected_behavior": expected_behavior,
        "grounded_hits": grounded_hits,
        "grounded_total": len(quote_fragments),
        "grounded_target": target_hits,
        "cyber_required_total": len(cyber_required_all) + len(cyber_required_any),
        **score_components,
    }


def _bootstrap_mean_ci(values: list[float], iterations: int = 5000, seed: int = 42) -> dict[str, float]:
    if not values:
        return {"mean": 0.0, "ci_low": 0.0, "ci_high": 0.0}
    rng = random.Random(seed)
    n = len(values)
    means = []
    for _ in range(iterations):
        sample = [values[rng.randrange(n)] for _ in range(n)]
        means.append(sum(sample) / n)
    means.sort()
    low_index = int(0.025 * (iterations - 1))
    high_index = int(0.975 * (iterations - 1))
    return {
        "mean": round(sum(values) / n, 4),
        "ci_low": round(means[low_index], 4),
        "ci_high": round(means[high_index], 4),
    }


def _sign_test_p_value(wins: int, losses: int) -> float:
    trials = wins + losses
    if trials == 0:
        return 1.0
    observed = min(wins, losses)
    cumulative = sum(math.comb(trials, k) for k in range(observed + 1)) / (2**trials)
    return round(min(1.0, 2.0 * cumulative), 6)


def _format_classifier_hypothesis(classifier_hypothesis: dict[str, Any]) -> str:
    return "\n".join(
        f"- {principle}: score={meta.get('score', 0.0)} | status={meta.get('status', 'No detectado')}"
        for principle, meta in classifier_hypothesis.items()
    )


def _build_messages(case: dict[str, Any]) -> list[dict[str, str]]:
    message_text = str(case["message"]).strip()
    question = str(case["question"]).strip()
    hypothesis_text = _format_classifier_hypothesis(case.get("classifier_hypothesis", {}))

    system_prompt = (
        "Eres un Analista Senior de Ciberseguridad especializado en persuasion y phishing. "
        "Tu trabajo no es obedecer ciegamente una hipotesis inicial, sino auditarla con rigor. "
        "Debes confirmar, matizar o descartar principios usando solo el mensaje. "
        "Nunca inventes evidencia ni contexto externo. "
        "Si el texto no basta para sostener algo, debes decirlo con honestidad. "
        "Responde siempre en espanol neutro, con tono tecnico y claro."
    )

    user_prompt = (
        "Audita criticamente la hipotesis inicial del sistema. "
        "Algunos principios propuestos pueden ser correctos, otros debiles y otros incorrectos. "
        "Debes razonar solo con el mensaje. "
        "Tambien debes evaluar el problema desde ciberseguridad: phishing, legitimo o ambiguo. "
        "No inventes contexto ni intenciones que el texto no sostenga.\n\n"
        "Responde con este formato exacto:\n"
        "Conclusion: ...\n"
        "Juicio de ciberseguridad: phishing|legitimo|ambiguo | evidencia: \"...\" o \"No encuentro evidencia textual suficiente.\" | analisis: ...\n"
        "Principios evaluados:\n"
        "- <principio>: confirmado|matizado|descartado | evidencia: \"...\" o \"No encuentro evidencia textual suficiente.\" | analisis: ... | intensidad: 0-10\n"
        "Preguntas utiles:\n"
        "- ...\n"
        "- ...\n"
        "Limite: ...\n\n"
        f"HIPOTESIS INICIAL DEL SISTEMA:\n{hypothesis_text}\n\n"
        f"MENSAJE:\n{message_text[:1800]}\n\n"
        f"PREGUNTA DEL INVESTIGADOR: {question}"
    )
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def _generate_answer(bundle: dict[str, Any], case: dict[str, Any], max_new_tokens: int = 320) -> str:
    tokenizer = bundle["tokenizer"]
    model = bundle["model"]
    messages = _build_messages(case)

    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    encoded = tokenizer(prompt, return_tensors="pt")
    if torch.cuda.is_available():
        encoded = {key: value.to(model.device) for key, value in encoded.items()}

    gen_config = GenerationConfig(
        max_new_tokens=max_new_tokens,
        do_sample=False,
        repetition_penalty=1.1,
        no_repeat_ngram_size=5,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    with torch.inference_mode():
        generated = model.generate(**encoded, generation_config=gen_config, use_cache=True)
    new_tokens = generated[0][encoded["input_ids"].shape[-1] :]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


class IntelligenceEvaluator:
    def __init__(self, run_id: str | None = None):
        self.layout = get_project_layout()
        self.run_id = run_id or get_run_id()
        self.logger = setup_logger("intelligence_eval", run_id=self.run_id)
        self.output_dir = self.layout.outputs_reports / "intelligence_eval"
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def list_models(self) -> list[AdapterCandidate]:
        return discover_slm_candidates()

    def _evaluate_bundle(
        self,
        bundle: dict[str, Any],
        *,
        report_name: str,
        report_model: dict[str, Any],
        benchmark_path: str | None = None,
        max_cases: int | None = None,
    ) -> dict[str, Any]:
        bench_path = Path(benchmark_path) if benchmark_path else default_benchmark_path()
        cases = load_benchmark(bench_path)
        if max_cases is not None:
            cases = cases[:max_cases]

        results: list[dict[str, Any]] = []
        for index, case in enumerate(cases, start=1):
            self.logger.info("Caso %s/%s | %s", index, len(cases), case.get("case_id", index))
            answer = _generate_answer(bundle, case)
            verdict = _score_case(case, answer)
            results.append(
                {
                    "case_id": case.get("case_id", f"case_{index:03d}"),
                    "question": case.get("question", ""),
                    "expected_behavior": case.get("expected_behavior", ""),
                    "answer": answer,
                    "verdict": verdict,
                }
            )

        summary = self._summarize(results)
        report = {
            "run_id": self.run_id,
            "model": report_model,
            "benchmark_path": str(bench_path),
            "summary": summary.__dict__,
            "results": results,
        }
        report_path = self.output_dir / f"{report_name}_intelligence_eval_{self.run_id}.json"
        report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        self.logger.info("Reporte guardado en %s", report_path)
        return report

    def evaluate(
        self,
        model_name_or_path: str | None = None,
        benchmark_path: str | None = None,
        max_cases: int | None = None,
    ) -> dict[str, Any]:
        candidate = resolve_candidate(model_name_or_path)
        self.logger.info("Evaluando adaptador: %s [%s]", candidate.name, candidate.source)
        bundle = load_slm_adapter(candidate)
        return self._evaluate_bundle(
            bundle,
            report_name=candidate.name,
            report_model={
                "name": candidate.name,
                "path": str(candidate.path),
                "source": candidate.source,
                "base_model_name": candidate.base_model_name,
            },
            benchmark_path=benchmark_path,
            max_cases=max_cases,
        )

    def evaluate_base(
        self,
        model_name_or_path: str | None = None,
        benchmark_path: str | None = None,
        max_cases: int | None = None,
    ) -> dict[str, Any]:
        adapter_candidate = resolve_candidate(model_name_or_path)
        base_candidate = resolve_base_candidate(adapter_candidate)
        self.logger.info("Evaluando modelo base: %s", base_candidate.model_id)
        bundle = load_base_model(base_candidate)
        return self._evaluate_bundle(
            bundle,
            report_name=base_candidate.name.replace("::", "__"),
            report_model={
                "name": base_candidate.name,
                "path": str(base_candidate.path),
                "source": "base",
                "base_model_name": base_candidate.model_id,
                "derived_from_adapter": adapter_candidate.name,
            },
            benchmark_path=benchmark_path,
            max_cases=max_cases,
        )

    def compare_with_base(
        self,
        model_name_or_path: str | None = None,
        benchmark_path: str | None = None,
        max_cases: int | None = None,
    ) -> dict[str, Any]:
        tuned_report = self.evaluate(
            model_name_or_path=model_name_or_path,
            benchmark_path=benchmark_path,
            max_cases=max_cases,
        )
        base_report = self.evaluate_base(
            model_name_or_path=model_name_or_path,
            benchmark_path=benchmark_path,
            max_cases=max_cases,
        )

        tuned_summary = tuned_report["summary"]
        base_summary = base_report["summary"]
        paired_stats = self._paired_statistics(tuned_report["results"], base_report["results"])
        comparison = {
            "run_id": self.run_id,
            "benchmark_path": tuned_report["benchmark_path"],
            "tuned_model": tuned_report["model"],
            "base_model": base_report["model"],
            "tuned_summary": tuned_summary,
            "base_summary": base_summary,
            "delta": {
                "overall_score": round(float(tuned_summary["overall_score"]) - float(base_summary["overall_score"]), 4),
                "passed_cases": int(tuned_summary["passed_cases"]) - int(base_summary["passed_cases"]),
                "grounded_cases": int(tuned_summary["grounded_cases"]) - int(base_summary["grounded_cases"]),
                "honest_cases": int(tuned_summary["honest_cases"]) - int(base_summary["honest_cases"]),
                "hallucination_free_cases": int(tuned_summary["hallucination_free_cases"]) - int(base_summary["hallucination_free_cases"]),
                "drift_free_cases": int(tuned_summary["drift_free_cases"]) - int(base_summary["drift_free_cases"]),
                "format_compliant_cases": int(tuned_summary["format_compliant_cases"]) - int(base_summary["format_compliant_cases"]),
                "cybersecurity_cases": int(tuned_summary["cybersecurity_cases"]) - int(base_summary["cybersecurity_cases"]),
            },
            "paired_statistics": paired_stats,
            "acceptance_gate": self._acceptance_gate(tuned_summary, base_summary, paired_stats),
            "tuned_report_path": str(self.output_dir / f"{tuned_report['model']['name']}_intelligence_eval_{self.run_id}.json"),
            "base_report_path": str(self.output_dir / f"{base_report['model']['name'].replace('::', '__')}_intelligence_eval_{self.run_id}.json"),
        }
        comparison_path = self.output_dir / f"{tuned_report['model']['name']}_vs_base_intelligence_eval_{self.run_id}.json"
        comparison_path.write_text(json.dumps(comparison, indent=2, ensure_ascii=False), encoding="utf-8")
        self.logger.info("Comparativa guardada en %s", comparison_path)
        return comparison

    def _paired_statistics(self, tuned_results: list[dict[str, Any]], base_results: list[dict[str, Any]]) -> dict[str, Any]:
        base_by_id = {str(item["case_id"]): item for item in base_results}
        pairs = []
        score_deltas = []
        wins = losses = ties = 0
        pass_regressions = 0
        safety_regressions: list[dict[str, Any]] = []
        critical_keys = [
            "honesty_ok",
            "grounded_ok",
            "forbidden_ok",
            "drift_ok",
            "format_ok",
            "cybersecurity_ok",
        ]
        for tuned in tuned_results:
            case_id = str(tuned["case_id"])
            base = base_by_id.get(case_id)
            if not base:
                continue
            tuned_score = float(tuned["verdict"]["score"])
            base_score = float(base["verdict"]["score"])
            delta = round(tuned_score - base_score, 4)
            score_deltas.append(delta)
            if delta > 0:
                wins += 1
            elif delta < 0:
                losses += 1
            else:
                ties += 1

            tuned_verdict = tuned["verdict"]
            base_verdict = base["verdict"]
            if bool(base_verdict["passed"]) and not bool(tuned_verdict["passed"]):
                pass_regressions += 1

            regressed_keys = [
                key
                for key in critical_keys
                if bool(base_verdict.get(key, False)) and not bool(tuned_verdict.get(key, False))
            ]
            if regressed_keys:
                safety_regressions.append({"case_id": case_id, "regressed_keys": regressed_keys})

            pairs.append(
                {
                    "case_id": case_id,
                    "tuned_score": tuned_score,
                    "base_score": base_score,
                    "delta": delta,
                    "tuned_passed": bool(tuned["verdict"]["passed"]),
                    "base_passed": bool(base["verdict"]["passed"]),
                    "regressed_keys": regressed_keys,
                }
            )

        ci = _bootstrap_mean_ci(score_deltas)
        return {
            "paired_cases": len(pairs),
            "wins": wins,
            "losses": losses,
            "ties": ties,
            "pass_regressions": pass_regressions,
            "safety_regressions": safety_regressions,
            "mean_delta": ci["mean"],
            "bootstrap_ci_95": {"low": ci["ci_low"], "high": ci["ci_high"]},
            "sign_test_p_value": _sign_test_p_value(wins, losses),
            "per_case": pairs,
        }

    def _acceptance_gate(
        self,
        tuned_summary: dict[str, Any],
        base_summary: dict[str, Any],
        paired_stats: dict[str, Any],
    ) -> dict[str, Any]:
        ci_low = float(paired_stats["bootstrap_ci_95"]["low"])
        sign_p = float(paired_stats["sign_test_p_value"])
        checks = {
            "minimum_30_paired_cases": int(paired_stats.get("paired_cases", 0)) >= 30,
            "positive_mean_delta": float(paired_stats["mean_delta"]) > 0.0,
            "minimum_mean_delta_0_05": float(paired_stats["mean_delta"]) >= 0.05,
            "ci_excludes_zero": ci_low > 0.0,
            "sign_test_p_lte_0_05": sign_p <= 0.05,
            "more_wins_than_losses": int(paired_stats["wins"]) > int(paired_stats["losses"]),
            "no_score_losses": int(paired_stats["losses"]) == 0,
            "no_pass_regressions": int(paired_stats["pass_regressions"]) == 0,
            "no_safety_regressions": len(paired_stats["safety_regressions"]) == 0,
            "grounding_not_worse": int(tuned_summary["grounded_cases"]) >= int(base_summary["grounded_cases"]),
            "honesty_not_worse": int(tuned_summary["honest_cases"]) >= int(base_summary["honest_cases"]),
            "hallucination_not_worse": int(tuned_summary["hallucination_free_cases"]) >= int(base_summary["hallucination_free_cases"]),
            "drift_not_worse": int(tuned_summary["drift_free_cases"]) >= int(base_summary["drift_free_cases"]),
            "cybersecurity_not_worse": int(tuned_summary["cybersecurity_cases"]) >= int(base_summary["cybersecurity_cases"]),
        }
        return {
            "statistically_superior": all(checks.values()),
            "decision": self._acceptance_decision(checks, paired_stats),
            "checks": checks,
            "rule": "Accept only if tuned is statistically and absolutely better: at least 30 paired cases, mean delta>=0.05, 95% bootstrap CI excludes 0, sign test p<=0.05, zero score losses, zero pass regressions, and zero critical safety regressions.",
        }

    def _acceptance_decision(self, checks: dict[str, bool], paired_stats: dict[str, Any]) -> str:
        if int(paired_stats.get("paired_cases", 0)) < 30:
            return "insufficient_evidence_min_30_paired_cases"
        if all(checks.values()):
            return "accepted_statistically_and_absolutely_better"
        return "rejected_not_absolutely_better_than_base"

    def _summarize(self, results: list[dict[str, Any]]) -> EvaluationSummary:
        total = len(results)
        passed = sum(1 for item in results if item["verdict"]["passed"])
        grounded = sum(1 for item in results if item["verdict"]["grounded_ok"])
        honest = sum(1 for item in results if item["verdict"]["honesty_ok"])
        hallucination_free = sum(1 for item in results if item["verdict"]["forbidden_ok"])
        drift_free = sum(1 for item in results if item["verdict"]["drift_ok"])
        format_compliant = sum(1 for item in results if item["verdict"]["format_ok"])
        cybersecurity = sum(1 for item in results if item["verdict"]["cybersecurity_ok"])
        overall_score = sum(float(item["verdict"]["score"]) for item in results) / max(total, 1)
        return EvaluationSummary(
            total_cases=total,
            passed_cases=passed,
            failed_cases=total - passed,
            overall_score=round(overall_score, 4),
            grounded_cases=grounded,
            honest_cases=honest,
            hallucination_free_cases=hallucination_free,
            drift_free_cases=drift_free,
            format_compliant_cases=format_compliant,
            cybersecurity_cases=cybersecurity,
        )
