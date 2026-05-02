# Intelligence Eval

Este modulo sirve para evaluar la "inteligencia practica" de adaptadores SLM afinados en el proyecto.

Objetivo:

- medir grounding
- medir honestidad epistemica
- medir tendencia a alucinar
- medir utilidad conversacional sobre mensajes

No intenta medir una inteligencia abstracta universal.
Evalua si el modelo funciona como analista honesto y util para este proyecto.

## Descubrimiento de modelos

El evaluador busca adaptadores en:

- `outputs/results/models/` : modelos activos
- `outputs/tuned_models/` : carpeta de respaldo de modelos afinados y corridas largas

Un adaptador valido se reconoce por la presencia de `adapter_config.json`.

## Comando principal

```bash
python evaluate_slm_intelligence.py --list-models
python evaluate_slm_intelligence.py --model Qwen_Qwen2.5-1.5B-Instruct_paper_run_20260429
python evaluate_slm_intelligence.py --model Qwen_Qwen2.5-1.5B-Instruct_paper_run_20260429 --max-cases 5
python evaluate_slm_intelligence.py --model Qwen_Qwen2.5-1.5B-Instruct_paper_run_20260429 --base-only
python evaluate_slm_intelligence.py --model Qwen_Qwen2.5-1.5B-Instruct_paper_run_20260429 --compare-base
python evaluate_slm_intelligence.py --model outputs/tuned_models/mi_modelo_futuro
```

## Benchmark

Por defecto usa:

- `intelligence_eval/benchmarks/seed_cases.jsonl`

Cada linea del benchmark es un objeto JSON independiente.

Campos recomendados:

- `case_id`
- `message`
- `classifier_hypothesis`
- `question`
- `expected_behavior`
- `required_all`
- `required_any`
- `forbidden`
- `quote_fragments`
- `min_grounded_hits`
- `disallow_external_drift`

## Comportamientos esperados

Ejemplos de `expected_behavior`:

- `grounded_answer`
- `honest_unknown`
- `comparative_reasoning`

## Salidas

Los reportes se guardan en:

- `outputs/results/reports/intelligence_eval/`

El resumen incluye:

- total de casos
- casos aprobados
- score promedio
- grounding
- honestidad
- tasa libre de alucinacion segun la rubrica
- tasa libre de deriva externa segun la rubrica

## Nota metodologica

Este modulo no reemplaza revision humana.
Su funcion es darte una primera lectura reproducible para detectar:

- fine-tuning degradado
- respuestas inventadas
- obediencia ciega al clasificador
- incapacidad para admitir incertidumbre
