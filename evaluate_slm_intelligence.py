from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from src.intelligence_eval.evaluator import IntelligenceEvaluator


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluador de inteligencia practica para adaptadores SLM.")
    parser.add_argument("--model", type=str, default=None, help="Nombre del adaptador o ruta directa al directorio del modelo.")
    parser.add_argument("--benchmark", type=str, default=None, help="Ruta a un benchmark JSONL.")
    parser.add_argument("--max-cases", type=int, default=None, help="Limita el numero de casos del benchmark.")
    parser.add_argument("--base-only", action="store_true", help="Evalua solo el modelo base asociado al adaptador indicado en --model.")
    parser.add_argument("--compare-base", action="store_true", help="Evalua el adaptador y su modelo base con el mismo benchmark.")
    parser.add_argument("--list-models", action="store_true", help="Lista adaptadores SLM detectados y sale.")
    parser.add_argument("--json", action="store_true", help="Emite salida JSON para automatizacion.")
    return parser


def _source_label(source: str) -> str:
    return {
        "active": "Activo",
        "tuned_models": "Tuned models",
        "manual": "Manual",
    }.get(source, source)


def _print_model_list(rows) -> None:
    print()
    print("Modelos SLM detectados")
    print("=" * 80)
    for index, row in enumerate(rows, start=1):
        print(f"{index}. {row.name}")
        print(f"   Origen: {_source_label(row.source)}")
        print(f"   Base:   {row.base_model_name or 'Desconocida'}")
        print(f"   Ruta:   {row.path}")
        print()


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    evaluator = IntelligenceEvaluator()

    if args.list_models:
        rows = evaluator.list_models()
        if not rows:
            print("No se detectaron adaptadores SLM.")
            return
        if args.json:
            for row in rows:
                print(json.dumps({
                    "name": row.name,
                    "source": row.source,
                    "path": str(row.path),
                    "base_model_name": row.base_model_name,
                }, ensure_ascii=False))
        else:
            _print_model_list(rows)
        return

    if args.compare_base:
        report = evaluator.compare_with_base(
            model_name_or_path=args.model,
            benchmark_path=args.benchmark,
            max_cases=args.max_cases,
        )
    elif args.base_only:
        report = evaluator.evaluate_base(
            model_name_or_path=args.model,
            benchmark_path=args.benchmark,
            max_cases=args.max_cases,
        )
    else:
        report = evaluator.evaluate(
            model_name_or_path=args.model,
            benchmark_path=args.benchmark,
            max_cases=args.max_cases,
        )

    if args.json:
        if args.compare_base:
            print(json.dumps(report, indent=2, ensure_ascii=False))
        else:
            print(json.dumps(report["summary"], indent=2, ensure_ascii=False))
    else:
        if args.compare_base:
            tuned = report["tuned_summary"]
            base = report["base_summary"]
            delta = report["delta"]
            print()
            print("Comparativa de evaluacion")
            print("=" * 80)
            print(f"Adaptador:           {report['tuned_model']['name']}")
            print(f"Modelo base:         {report['base_model']['base_model_name']}")
            print(f"Score tuned:         {tuned['overall_score']}")
            print(f"Score base:          {base['overall_score']}")
            print(f"Delta score:         {delta['overall_score']}")
            print(f"Grounding delta:     {delta['grounded_cases']}")
            print(f"Honestidad delta:    {delta['honest_cases']}")
            print(f"Alucinacion delta:   {delta['hallucination_free_cases']}")
            print(f"Deriva delta:        {delta['drift_free_cases']}")
            print(f"Ciberseguridad delta:{delta['cybersecurity_cases']}")
            stats = report["paired_statistics"]
            gate = report["acceptance_gate"]
            print(f"Casos pareados:      {stats['paired_cases']}")
            print(f"Wins/Losses/Ties:    {stats['wins']}/{stats['losses']}/{stats['ties']}")
            print(f"Pass regressions:    {stats['pass_regressions']}")
            print(f"Safety regressions:  {len(stats['safety_regressions'])}")
            print(f"Delta medio pareado: {stats['mean_delta']}")
            print(f"IC bootstrap 95%:    [{stats['bootstrap_ci_95']['low']}, {stats['bootstrap_ci_95']['high']}]")
            print(f"Sign test p-value:   {stats['sign_test_p_value']}")
            print(f"Superior estadistico:{gate['statistically_superior']}")
            print(f"Decision final:      {gate['decision']}")
            print()
            return

        summary = report["summary"]
        print()
        print("Resumen de evaluacion")
        print("=" * 80)
        print(f"Modelo:              {report['model']['name']}")
        print(f"Casos totales:       {summary['total_cases']}")
        print(f"Casos aprobados:     {summary['passed_cases']}")
        print(f"Casos fallidos:      {summary['failed_cases']}")
        print(f"Score global:        {summary['overall_score']}")
        print(f"Grounding:           {summary['grounded_cases']}/{summary['total_cases']}")
        print(f"Honestidad:          {summary['honest_cases']}/{summary['total_cases']}")
        print(f"Sin alucinacion:     {summary['hallucination_free_cases']}/{summary['total_cases']}")
        print(f"Sin deriva externa:  {summary['drift_free_cases']}/{summary['total_cases']}")
        print(f"Ciberseguridad:      {summary['cybersecurity_cases']}/{summary['total_cases']}")
        print()


if __name__ == "__main__":
    main()
