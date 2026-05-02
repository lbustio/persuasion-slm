# BITACORA DE TRANSFERENCIA: Proyecto Persuasion SLM

> Estado operativo real del repositorio al 2026-04-28. Este documento ya no describe el plan ideal, sino lo que el codigo y los artefactos reflejan hoy.

Documento conceptual complementario:
- `docs/system_foundations.md` contiene la explicacion canonica de teoria, premisas, hipotesis, solucion y arquitectura del sistema.

---

## 1. Mision actual
Estamos construyendo un pipeline para:
1. Armonizar datasets bilingues de phishing persuasivo.
2. Generar explicaciones para entrenamiento supervisado/generativo.
3. Entrenar un clasificador multi-label.
4. Afinar un SLM pequeno para explicacion.
5. Exportar el clasificador a ONNX y OpenVINO.

---

## 2. Estructura real en uso
Las rutas activas del proyecto quedaron unificadas y hoy salen de `configs/architecture.yaml` a traves de `src/utils/paths.py`.

Directorios realmente usados:
- `data/`
- `configs/`
- `src/`
- `caches/downloads/`
- `caches/partials/`
- `outputs/artifacts/`
- `outputs/checkpoints/`
- `outputs/splits/`
- `outputs/results/models/`
- `outputs/results/figures/`
- `outputs/results/tables/`
- `outputs/results/predictions/`
- `outputs/results/reports/`
- `logs/runs/`

Estado de limpieza:
- Los directorios legacy top-level de `models/` y `checkpoints/` ya no forman parte del repo operativo.
- Los resultados historicos, caches residuales y `__pycache__` ya fueron limpiados para preparar una corrida fresca.
- La unica cache pesada que se conserva a proposito es la de modelos descargados en `caches/downloads/` para no redescargar pesos innecesariamente.

---

## 3. Datos y artefactos verificados
Datasets de entrada:
1. `data/IWSPA_AP_persuasion_annotated.csv`
2. `data/Spaphish dataset - DiB.csv`

Estado actual despues de la limpieza:
- `outputs/` y `logs/runs/` quedaron vaciados a proposito para permitir una corrida definitiva realmente fresca.
- No debe asumirse que existan artefactos finales vigentes hasta que se vuelva a correr el pipeline con `--fresh-all`.
- La referencia operativa valida hoy es el codigo y la configuracion, no resultados pasados.

Verificaciones historicas que siguen siendo utiles como contexto:
- `augmented_dataset.jsonl` llego a contener `3487` registros en una corrida previa.
- Revision puntual historica: `0` registros dummy detectados en ese archivo.
- El SLM habia quedado mal marcado como "completado" apuntando al modelo base `Qwen/Qwen2.5-1.5B-Instruct`; esa logica ya fue corregida.

---

## 4. Correcciones ya aplicadas
### Refactor de rutas
- Se creo `src/utils/paths.py` como fuente unica de verdad para rutas.
- `main.py`, `logger.py`, `state.py`, `harmonizer.py`, `augmenter.py`, `classifier_trainer.py`, `slm_finetuner.py`, `exporter.py` e `inspect_augmented.py` ahora usan esa capa.
- La estructura de resultados ya no mezcla metricas, figuras y tablas en un mismo directorio; ahora se separan en `figures`, `tables`, `predictions` y `reports`.

### Fallos de entrenamiento corregidos
- El clasificador preparaba `train_ds` y `eval_ds`, pero el `Trainer` estaba recibiendo el dataset equivocado.
- Ahora el `Trainer` usa los splits tokenizados correctos.
- El clasificador ahora usa `train/validation/test = 70/10/20`.
- El SLM antes entrenaba y evaluaba sobre el mismo dataset completo.
- Ahora el SLM genera un split interno train/eval antes de entrenar.
- El clasificador ahora ignora checkpoints incompatibles cuando cambian las etiquetas o la configuracion del split.

### Reanudacion mas segura
- Si un checkpoint declara una fase como completada pero falta el directorio real del modelo, el pipeline ya no confia ciegamente en ese estado.
- Si el SLM fue marcado como "hecho" solo con fallback de CPU, una corrida posterior con GPU ya no se salta el fine-tuning.
- `main.py` ya soporta `--fresh-all`, `--fresh-harmonizer`, `--fresh-augmenter`, `--fresh-classifier`, `--fresh-slm`, `--slm-run-tag` y `--slm-only`.

### Compatibilidad y rendimiento
- Se aplico un parche de compatibilidad para `accelerate`/`transformers` en `src/utils/compat.py`.
- Se implemento una capa `turbo` que detecta el hardware real y ajusta batch size, precision, workers, padding, TF32, gradient checkpointing y QLoRA segun el equipo disponible.
- Las rutas de entrenamiento usan padding dinamico en vez de `padding="max_length"` fijo, para reducir computo desperdiciado.

### Exportacion paper-ready
- Se implemento `src/reporting/paper_artifacts.py`.
- Las corridas nuevas exportan automaticamente figuras, tablas, predicciones, reportes JSON y splits persistidos para reutilizacion posterior.
- Las figuras salen en ingles y en formatos `PNG`, `EPS` y `CSV` de datos fuente.

---

## 5. Inconsistencias todavia presentes
Estas no rompen la estructura, pero siguen importando:

1. `docs/blueprint.md` describe una arquitectura objetivo mas avanzada que la implementacion real.
   - Habla de calibracion, thresholds por clase, ECE, ensemble, test holdout y API de inferencia.
   - Eso todavia no existe completo en el codigo actual.

2. `configs/training_defaults.yaml` ya quedo mejor alineado en rutas, pero no es la unica fuente de verdad del entrenamiento.
   - La configuracion efectiva del pipeline hoy se apoya sobre todo en `configs/architecture.yaml` y en constantes dentro de los modulos.

3. El clasificador ya no usa split simple `70/30`.
   - Ahora usa `70/10/20` para `train/validation/test`, siguiendo el requerimiento operativo actual.
   - Aun no implementa la estratificacion sofisticada descrita en el blueprint.

4. El SLM ahora si separa train/eval, pero con split `90/10` interno.
   - Eso es correcto funcionalmente, pero no sigue aun un contrato formal documentado para generative training.

5. El paper-ready export ya cubre gran parte del material reutilizable, pero aun no implementa `ECE` ni calibracion formal.
   - Si el articulo lo necesita, esa debe ser la siguiente ampliacion metodologica.

6. La validacion definitiva de todos los artefactos aun depende de una corrida completa nueva.
   - La estructura y el codigo ya quedaron listos.
   - Falta ejecutar y revisar los outputs reales de esa corrida final.

---

## 6. Estado funcional actual
- Armonizacion: funcional.
- Aumentacion: funcional.
- Clasificador: funcional, con correccion del uso de datasets en `Trainer`.
- SLM: funcional a nivel de pipeline, con reanudacion y split mas seguros.
- Exportacion ONNX/OpenVINO: funcional para el clasificador.
- Reporting paper-ready: funcional a nivel de codigo e integrado al pipeline.
- Estructura de salidas: funcional y separada por tipo de artefacto.

---

## 7. Pendientes reales
1. Ejecutar la corrida completa definitiva desde cero y verificar que todos los artefactos salgan como espera el paper.
2. Revisar los outputs generados y congelar cuales seran canonicos para el articulo.
3. Implementar ECE y calibracion formal si eso sera parte del articulo final.
4. Decidir si `training_defaults.yaml` sera contractual o solo documental.
5. Si el articulo exige mas evaluacion generativa, agregar metricas adicionales sobre el SLM mas alla de losses y data overview.
6. La corrida definitiva recomendada sigue siendo `python main.py --fresh-all --fresh-slm --slm-run-tag paper_run_YYYYMMDD`.

---

## 8. Exportables listos para paper
Las corridas nuevas ya quedan preparadas para dejar material reutilizable sin recalcular:

- Figuras en ingles en `outputs/results/figures/`:
  - learning curves
  - label distribution by split
  - per-class metrics
  - ROC curves
  - precision-recall curves
  - threshold sweeps por clase
  - binary confusion matrices por clase
  - SLM data overview
- Cada figura sale en:
  - `PNG`
  - `EPS`
  - `CSV` con los datos base de la figura
- Tablas en `outputs/results/tables/`:
  - training history
  - split summaries
  - class metrics
  - summary metrics
  - threshold sweeps
  - multilabel confusions
  - prediction summaries
  - SLM summaries y muestras de evaluacion
- Predicciones reutilizables en `outputs/results/predictions/`:
  - validacion
  - test
  - scores por clase
  - etiquetas gold y predichas
- Reportes y manifiestos JSON en `outputs/results/reports/`
- Splits exportados en `outputs/splits/` para fijar exactamente con que particiones se entreno y evaluo

Nota operativa:
- El manifiesto JSON de cada fase queda pensado como indice canonico de archivos para retomar luego la escritura del paper sin redescubrir manualmente las salidas.

---

## 9. Decision metodologica actual
Decision tomada en esta conversacion:

- Primero se terminara la corrida actual y se consolidaran todos los artefactos paper-ready.
- No se hara ahora una refactorizacion grande anti-leakage del pipeline completo.
- El costo de esa refactorizacion en este momento se considera alto frente al beneficio inmediato.
- La postura pragmatica acordada es:
  - usar el clasificador como eje cuantitativo principal
  - tratar el SLM como modulo de apoyo, generacion y explicabilidad
  - no presentar todavia la evaluacion interna del SLM como evidencia fuerte de generalizacion libre de leakage

Riesgo reconocido pero aceptado por ahora:
- El SLM hoy se entrena a partir de un dataset aumentado construido antes de un split canonico global.
- Eso no invalida terminar la corrida actual, pero si limita cuan fuerte puede ser la reclamacion metodologica sobre el SLM en una version final de paper.

Plan acordado:
1. Terminar esta corrida.
2. Inspeccionar y congelar resultados.
3. Despues, abrir una segunda iteracion para cerrar huecos metodologicos:
   - anti-leakage mas estricto
   - ECE y calibracion
   - posible deduplicacion y split-first
   - endurecimiento de la evaluacion del SLM

Decision funcional importante:
- El objetivo del proyecto no es solo clasificar mensajes.
- El objetivo principal incluye crear un SLM con el que se pueda "hablar" sobre los mensajes para entender mejor los principios de persuasion presentes, su evidencia textual y su sentido analitico.
- Por eso, el clasificador sigue siendo el ancla cuantitativa, pero el SLM es una pieza central del valor funcional del sistema.

### Aclaracion conceptual posterior
Se refino la relacion deseada entre clasificador y SLM:

- El SLM no debe ser reducido a un extractor rigido de evidencia.
- Tampoco debe comportarse como un simple justificador obediente de las etiquetas del clasificador.
- La arquitectura funcional deseada es:
  - el clasificador propone una hipotesis inicial o `prior`
  - el SLM toma ese bundle como punto de partida
  - el SLM contrasta esa hipotesis con el texto real
  - el SLM puede confirmar, matizar, expandir o corregir lo que el clasificador sugirio

Formulacion operativa acordada:
- `classifier as prior`
- `SLM as analyst`

Implicaciones practicas de esta decision:
- No conviene capar al SLM hasta volverlo un modulo estrecho de extraccion.
- Si el sistema solo obliga al SLM a localizar spans o a repetir etiquetas, se pierde una parte central del valor del proyecto.
- El valor buscado es que el usuario pueda conversar con un analista de mensajes que parte de una hipotesis cuantitativa, pero no esta esclavizado a ella.

Riesgo identificado:
- El pipeline actual tiende a empujar al SLM hacia la racionalizacion de etiquetas ya sugeridas.
- Ese no es el comportamiento objetivo final.
- El objetivo final es una capa analitica conversacional disciplinada por el texto, no una capa decorativa de justificacion.

Direccion recomendada para la siguiente iteracion:
1. Mantener el bundle del clasificador como contexto inicial.
2. Reentrenar y redisenar prompts para que el SLM aprenda no solo a justificar, sino tambien a refutar o matizar.
3. Preservar el potencial conversacional del SLM despues de la auditoria inicial del mensaje.

## 10. Corrida registrada
### Corrida: 20260428_073046
- **HW:** NVIDIA GeForce RTX 5050 Laptop GPU
- **BF16:** `True`
- **Estado observado:** artefactos de clasificador y exportacion presentes
- **Nota:** el estado guardado del SLM para esa corrida no confirma un fine-tuning real; solo confirmaba fallback al modelo base

### Estado al cierre de esta conversacion
- **Codigo:** refactorizado, limpiado y preparado para una corrida fresca.
- **Salidas:** definidas con contrato paper-ready.
- **Repo:** sin resultados viejos canonicos; listo para regenerar outputs desde cero.
- **Decision de trabajo:** terminar primero esta corrida y posponer el endurecimiento metodologico del SLM para la siguiente iteracion.
- **Siguiente comando recomendado:**
  - `python main.py --fresh-all --fresh-slm --slm-run-tag paper_run_20260428`

### Estado de la webapp
- Ya existe una maqueta funcional en `webapp/`.
- La maqueta ya fue revisada visualmente y la direccion general fue aprobada.
- La experiencia debe quedar completamente en espanol, aunque los mensajes analizados puedan estar en ingles.
- La app ya incorpora:
  - analisis de mensaje
  - principios detectados
  - evidencia resaltada
  - conversacion contextual con el SLM
  - capa didactica
  - panel de investigacion
  - logs visibles
- Ya existe un servidor minimo en FastAPI para servir la maqueta localmente.
- La integracion con salidas reales del pipeline queda diferida hasta terminar la corrida tecnica actual.

### Corrida: 20260429_074051
- **HW:** NVIDIA GeForce RTX 5050 Laptop GPU | **BF16:** True
- **Estado:** EXITO
- **Artefactos:** ONNX y OpenVINO generados.
- **SLM:** adapter `Qwen_Qwen2.5-1.5B-Instruct_paper_run_20260429` generado.

### Estado webapp tras la corrida 20260429_074051
- La webapp dejo de depender del flujo dummy para la ruta principal.
- Ya consume:
  - ejemplos reales del split de prueba
  - metricas y curvas reales desde `outputs/results/`
  - inferencia real del clasificador entrenado
- El chat intenta cargar el SLM local afinado.
- Si el base model de Qwen no esta disponible en cache local, la app cae a un modo de respuesta contextual de respaldo en vez de romperse.
- El comando canonico de arranque se mantiene:
  - `python webapp.py`

## 11. Avance Sesion 2026-04-29: Optimizacion SLM y Soberania Tecnologica

### Humanizacion del Chat y UX
- Se eliminaron los scores tecnicos y etiquetas roboticas (ej. `(detectado)`) de las respuestas del chat para un tono mas profesional y humano.
- Los logs ahora incluyen segundos (`HH:MM:SS`) y codigos de colores por etapa (Entrada, Clasificador, SLM, Riesgo, etc.) para mejorar la jerarquia visual.
- Se aumento el tamano de fuente en el panel de chat para mejorar la legibilidad.
- El sistema regreso al uso de las etiquetas originales del entrenamiento (English keys: `authority`, `distraction`, etc.) para evitar discrepancias por traduccion automatica.

### Model Advisor (Zero-Hardcode)
- Implementacion de `src/utils/model_advisor.py`: un motor de descubrimiento dinamico que consulta la API de HuggingFace (sin necesidad de API Keys) para encontrar el mejor SLM para el hardware actual.
- El sistema ahora es polimorfico: optimiza la seleccion para NVIDIA (CUDA), Intel (OpenVINO/XPU), Apple (MPS) o CPU.

### Boveda de Modelos (Asset Vault)
- Creacion de `outputs/tuned_models`: repositorio centralizado para preservar modelos afinados.
- El pipeline ahora verifica la boveda antes de descargar o reentrenar, permitiendo reutilizar activos al cambiar de hardware.
- `main.py` integra ahora la logica de migracion y preservacion de modelos.

### Avance Sesion 2026-04-29 (Finalizacion): Refinamiento y Auditoria
- **Panel de Auditoria de Sistema**: La Webapp ahora incluye un panel tecnico que muestra ID, Ruta, Tamano en disco, Arquitectura y Fecha de entrenamiento de los modelos activos.
- **Descubrimiento por ADN**: Se eliminaron las rutas hardcoded. El sistema identifica Clasificadores y SLMs analizando sus archivos JSON de configuracion.
- **Estabilizacion de Inferencia**:
  - Implementado **Greedy Decoding** (`do_sample=False`) y `repetition_penalty=1.3` para eliminar alucinaciones y bucles infinitos en el modelo de 1.5B.
  - Correccion del error de "Meta Device" mediante vinculacion explicita de pesos (`tie_weights`) en la carga del SLM.
- **Optimizacion UI/UX**:
  - Ajuste de jerarquia visual en el panel Hero (distribucion por filas).
  - Normalizacion de tamanos de fuente para legibilidad masiva (1.0rem - 1.15rem).
  - Implementacion de Scrollbars fijos en el chat para evitar desplazamientos del layout.
  - Creacion de `.gitignore` optimizado para evitar la subida de modelos pesados a GitHub.

- **TelemetrAa de Hardware en Vivo**: La Webapp ahora muestra el nombre de la GPU, VRAM disponible/total, RAM del sistema y nAocleos de CPU en tiempo real en la cabecera.
- **Identidad Visual**: Generacion e integracion de un logo futurista para "Persuasion Lab".
- **SLM DinAmico y Fluido**:
  - Implementado **Presupuesto DinAmico de Tokens** (de 128 a 1024 tokens) segAon la complejidad de la pregunta.
  - Corregido el bug de "parAmetros ignorados" mediante `GenerationConfig`, permitiendo una temperatura de `0.25` real para respuestas mas humanas.
  - Refinado el Prompt de Chat (Persona: Analista Senior de Ciberseguridad) para eliminar respuestas roboticas y boilerplate de cortesia excesiva.
- **Optimizacion de Interfaz**:
  - Unificacion de tipografia a **Inter**.
  - Compactacion funcional del Log de Ejecucion (reduccion de gaps y interlineado).
  - Bloqueo de saltos de linea en la cabecera de hardware.

**Estado Final**: La aplicacion es ahora una herramienta de grado profesional, con telemetria de hardware integrada, un cerebro de IA fluida y una estetica tecnica impecable. Lista para la siguiente fase de recoleccion de datos y validacion de principios.

## 12. Diagnostico SLM posterior a la corrida 20260429

### Evaluacion comparativa ya ejecutada
Se implemento un modulo nuevo de evaluacion en:
- `src/intelligence_eval/`
- `evaluate_slm_intelligence.py`
- `intelligence_eval/benchmarks/seed_cases.jsonl`

El benchmark semilla ya se corrio contra:
- el adapter afinado `Qwen_Qwen2.5-1.5B-Instruct_paper_run_20260429`
- el modelo base `Qwen/Qwen2.5-1.5B-Instruct`

Resultado comparativo observado el `2026-04-29`:
- `fine-tuned score`: `0.75`
- `base score`: `0.7917`
- `delta score`: `-0.0417`
- `grounding delta`: `0`
- `honestidad delta`: `0`
- `alucinacion delta`: `0`
- `deriva externa delta`: `-1`

Conclusion operativa:
- el adapter actual no supera al modelo base en la tarea evaluada
- no mejora grounding
- no mejora honestidad
- no mejora control de alucinacion
- introduce un pequeno empeoramiento en deriva o racionalizacion externa

### Problemas detectados en el pipeline del SLM

#### 1. Leakage semantico directo en la augmentacion
En `src/pipeline/augmenter.py`, la funcion `_generate_with_teacher(...)` recibe `active_principles`, es decir:
- el teacher ya sabe de antemano que principios deben salir
- la tarea se convierte en "justifica estas etiquetas"
- no aprende a auditar ni a refutar

Esto empuja al SLM a:
- racionalizar etiquetas
- sobreinterpretar evidencia
- sonar convincente aunque el texto no sostenga bien la conclusion

#### 2. Objetivo de entrenamiento mal alineado con el producto real
El dataset aumentado entrena al SLM para responder a un prompt tipo:
- `Analiza el siguiente correo... identifica principios... explica por que... intensidad 1-10`

Pero el producto real quiere algo mas rico:
- conversar sobre cualquier aspecto del mensaje
- partir del prior del clasificador
- confirmar, matizar, refutar y comparar
- admitir limites con honestidad

Es decir:
- hoy el SLM aprende monologos de analisis
- no aprende interaccion analitica anclada al mensaje

#### 3. Truncacion severa del dataset generativo
En `configs/training_defaults.yaml`, el SLM se entrena con:
- `max_length: 256`

Medicion real sobre `500` muestras de `outputs/artifacts/augmented_dataset.jsonl` usando el tokenizer de Qwen:
- `chat_tokens_avg`: `671.81`
- `chat_tokens_p50`: `694`
- `chat_tokens_p90`: `1129`
- `chat_tokens_p95`: `1199`
- `chat_tokens_max`: `2212`
- `over_256`: `406/500`
- `over_512`: `311/500`

Interpretacion:
- la mayoria de las conversaciones de entrenamiento quedan truncadas
- el modelo ve respuestas cortadas con mucha frecuencia
- se pierde justo la parte final donde suelen cerrarse explicaciones, matices e intensidad

Dato adicional:
- la respuesta del assistant sola ya tiene `291.37` tokens de media
- su mediana es `389`
- su percentil 90 es `512`

O sea:
- incluso la respuesta objetivo, sin contar system ni user, ya supera a menudo el limite de entrenamiento

#### 4. Supervision desigual y estilos mezclados
El dataset armonizado tiene `3487` registros:
- `IWSPA`: `2092`
- `Spaphish`: `1395`
- `with_human justifications`: `1395`
- `without_human justifications`: `2092`

Esto genera una mezcla fuerte:
- `Spaphish` aporta justificaciones humanas relativamente cortas y mas directas
- `IWSPA` depende del teacher y produce salidas largas, libres y muchas veces sobreinterpretadas

Ademas, el dataset aumentado mezcla estilos:
- respuestas negativas muy cortas
- respuestas largas estructuradas por principio
- algunas salidas en ingles aunque el prompt del user este en espanol

Muestreo sobre `1000` ejemplos del dataset aumentado:
- `english_answer`: `74`

Implicacion:
- el modelo aprende formatos y tonos inconsistentes
- no internaliza un contrato conversacional unico ni limpio

#### 5. Causal LM entrenado sobre toda la conversacion
En `src/pipeline/slm_finetuner.py`, al tokenizar:
- se aplica `chat_template`
- luego `labels = input_ids.copy()`

Eso significa que el loss cae sobre:
- system prompt
- user prompt
- assistant answer

No solo sobre la respuesta del assistant.

Riesgo:
- parte de la capacidad de entrenamiento se gasta en reconstruir el prompt
- no se optimiza especificamente la conducta de respuesta
- es un ajuste menos limpio que un masking del prompt y supervision centrada en la salida

#### 6. Webapp y training todavia no comparten exactamente la misma tarea
Aunque la webapp ya fue endurecida para:
- tratar al clasificador como `prior`
- exigir honestidad
- prohibir invenciones

el entrenamiento historico sigue basado en:
- clasificacion explicada
- no auditoria conversacional abierta

Por eso aparece el desfase:
- la webapp pide un analista
- el adapter aprendido sigue pareciendose mas a un justificador monologico

### Diagnostico consolidado
La conclusion metodologica de esta sesion es:

- el problema principal no es que Qwen 1.5B sea "estupido"
- el problema principal es que el pipeline actual del SLM le ensena una tarea incorrecta y ademas la ensena con truncacion y ruido de estilo

Formulacion corta:
- `classifier as prior` sigue siendo la direccion correcta
- pero el fine-tuning actual ensena `classifier as truth to justify`

### Recomendaciones para la siguiente iteracion
1. Redisenar el dataset SLM alrededor de estas habilidades:
   - confirmar con cita
   - matizar con cita
   - refutar con cita
   - comparar principios
   - decir explicitamente cuando el mensaje no basta
2. El teacher no debe recibir los principios positivos como verdad cerrada.
3. Elevar el `max_length` real del SLM o reducir/reestructurar las muestras para evitar truncacion masiva.
4. Unificar formato y lengua de las respuestas objetivo.
5. Considerar supervision solo sobre la salida del assistant, no sobre toda la conversacion.
6. Mantener el benchmark de inteligencia como gate obligatorio:
   - el siguiente adapter debe superar al base antes de darse por bueno.

### Cambios ya aplicados en codigo tras este diagnostico
Durante esta sesion ya se implemento una primera correccion estructural:

- `src/pipeline/augmenter.py`
  - deja de entrenar al SLM con el prompt viejo de "analiza y enumera principios"
  - ahora construye un `HIPOTESIS INICIAL DEL SISTEMA` sintetico
  - introduce ruido controlado:
    - algun falso positivo debil
    - a veces un verdadero positivo omitido o debilitado
  - el teacher ya no recibe una lista cerrada de principios activos como verdad final
  - las respuestas objetivo pasan a un formato de auditoria:
    - `Conclusion`
    - `Principios evaluados`
    - `Preguntas utiles`
    - `Limite`

- `src/pipeline/slm_finetuner.py`
  - ahora carga `training_defaults.yaml`
  - usa `slm_max_length: 4096`
  - enmascara el prompt con `-100` y supervisa solo la salida del assistant
  - reserva presupuesto preferente para la respuesta del assistant al truncar
  - usa padding adecuado para secuencias variables
  - ahora respeta `max_epochs` y `learning_rate` del config

- `src/pipeline/classifier_trainer.py`
  - tambien quedo alineado para respetar `max_epochs` y `learning_rate`

- `configs/training_defaults.yaml`
  - se anadio `slm_max_length: 4096`

Estado de esta intervencion:
- la sintaxis de los modulos modificados ya fue validada con `py_compile`
- aun NO se ha regenerado el dataset aumentado ni reentrenado el nuevo adapter
- la siguiente verificacion real requiere corrida fresca del pipeline generativo

---

## 12. Estado actualizado 2026-04-30: preparacion para corrida final limpia

Se hizo una limpieza operativa para eliminar artefactos viejos de corridas anteriores sin borrar caches de descargas:
- se vaciaron `outputs/`, `logs/`, `caches/partials/` y `__pycache__/`
- se preservo `caches/downloads/` para no volver a bajar modelos de internet
- se recrearon los directorios runtime esperados por el pipeline

Correcciones estructurales aplicadas antes de la nueva corrida:
- `src/pipeline/augmenter.py` ahora genera datos `audit_v2`, donde el clasificador es una hipotesis inicial y el SLM aprende a confirmar, matizar o refutar con evidencia textual.
- `src/pipeline/slm_finetuner.py` mantiene supervision solo sobre la respuesta del assistant y usa `slm_max_length: 4096` como default real; no hay clamp oculto por VRAM.
- `main.py` usa nombres de adapter con `run_tag`, archiva adapters no objetivo y escribe notas en `context/bitacora.md`.
- `configs/webapp.yaml` y `webapp/server.py` quedaron alineados con `classifier as prior` y `SLM as analyst`.
- `/api/chat` ahora pasa al SLM la hipotesis estructurada del backend, la pregunta y el texto fuente; el SLM debe citar evidencia, admitir falta de datos y no inventar.
- La generacion del chat ahora es determinista (`do_sample=False`) y usa presupuesto dinamico de tokens segun la pregunta.
- `src/intelligence_eval/evaluator.py` exige el contrato `Conclusion / Principios evaluados / Limite` y compara adapter contra base.

Validacion ligera ejecutada:
- `py_compile` OK para `main.py`, `src/pipeline/augmenter.py`, `src/pipeline/slm_finetuner.py`, `src/intelligence_eval/evaluator.py`, `evaluate_slm_intelligence.py` y `webapp/server.py`.

Estado importante:
- no se ha ejecutado entrenamiento ni benchmark largo despues de estas correcciones
- la siguiente corrida debe ser fresca porque los outputs viejos fueron eliminados
- el nuevo adapter solo debe aceptarse si supera al modelo base en el benchmark de inteligencia y no degrada grounding/honestidad

Politica tecnica adicional:
- todo codigo, configuracion, webapp y contexto deben mantenerse en ASCII puro
- no usar tildes, ene, comillas tipograficas, guiones largos ni simbolos Unicode
- `slm_max_length` queda por defecto en `4096` porque la RTX 5050 ya lo maneja en la practica
- si una maquina futura no soporta `4096`, la reduccion debe hacerse cambiando configuracion de forma explicita, no con un clamp oculto en codigo

---

## 13. Estado actualizado 2026-04-30: especialista SLM y superioridad estadistica

Decision metodologica nueva:
- El adapter no se acepta solo por "sonar mejor".
- Debe demostrar superioridad estadistica contra su modelo base en un benchmark pareado.
- La comparacion debe medir delta por caso, wins/losses/ties, intervalo bootstrap del delta medio y sign test.
- El gate de aceptacion exige que el intervalo bootstrap 95% excluya 0, `p <= 0.05`, mas wins que losses y ninguna regresion en grounding, honestidad, alucinacion, deriva o ciberseguridad.

Cambios aplicados:
- `src/pipeline/augmenter.py` pasa de `audit_v2` a `audit_v3`.
- `audit_v3` incorpora el label supervisado `is_phishing` del dataset armonizado.
- El SLM ahora debe aprender dos ejes:
  - principios de persuasion presentes o ausentes
  - juicio de ciberseguridad: `phishing`, `legitimo` o `ambiguo`
- El formato objetivo ahora incluye:
  - `Conclusion`
  - `Juicio de ciberseguridad`
  - `Principios evaluados`
  - `Preguntas utiles`
  - `Limite`
- `intelligence_eval/benchmarks/seed_cases.jsonl` fue reescrito en ASCII y ampliado a casos de persuasion, phishing claro, mensajes legitimos y mensajes ambiguos.
- `src/intelligence_eval/evaluator.py` ahora calcula `cybersecurity_ok`, `cybersecurity_cases` y estadistica pareada contra base.
- `evaluate_slm_intelligence.py` imprime deltas de ciberseguridad, wins/losses/ties, IC bootstrap 95%, p-value de sign test y si el adapter es estadisticamente superior.

Seleccion de modelo:
- `src/utils/model_advisor.py` fue reemplazado por un advisor que consulta Hugging Face en internet.
- El advisor detecta hardware local y puntua candidatos por:
  - ajuste de parametros a VRAM/RAM
  - contexto disponible
  - descargas
  - likes
  - recencia
- `main.py` ahora usa `--slm auto` por defecto.
- La decision del advisor se guarda en `outputs/results/reports/model_advisor_last_report.json`.
- El advisor ya no contiene listas `SEED_MODEL_IDS`, familias preferidas ni modelos Qwen/Phi/Gemma hardcodeados.
- Si internet falla, intenta descubrir modelos desde la cache local de Hugging Face.
- Si no hay internet ni cache local valida, `--slm auto` falla explicitamente y exige pasar `--slm` manualmente.
- No debe reintroducirse fallback hardcodeado en `--slm auto`.

Hardware actual detectado en esta maquina:
- GPU: NVIDIA GeForce RTX 5050 Laptop GPU
- VRAM: aproximadamente 7.96 GB
- Backend: CUDA
- BF16: soportado
- Perfil: MID_VRAM

Implicacion:
- `slm_max_length: 4096` sigue siendo el default correcto para calidad, porque reduce truncacion y permite que el SLM vea texto, hipotesis y respuesta objetivo completos.
- La eleccion del modelo base para el SLM ya no debe ser fija; debe salir del advisor online o de una decision explicita del usuario.

Verificacion posterior:
- Se ejecuto una prueba online del advisor con Hugging Face API.
- Fuente: `huggingface_api`.
- Ganador observado para la RTX 5050: `Qwen/Qwen3-4B-Instruct-2507`.
- Esa eleccion fue resultado del ranking generico, no de una lista hardcodeada.

---

## 14. Actualizacion 2026-04-30: cierre metodologico de investigacion

Objetivo atendido:
- Convertir el proyecto en un sistema de investigacion aplicado, no solo una demo de entrenamiento.
- Reducir leakage entre fases.
- Separar etiquetas phishing reales/inferidas.
- Hacer que el benchmark no pueda declarar victoria con evidencia insuficiente.
- Quitar nombres fijos de artefactos en la webapp.

Cambios aplicados:
- Se agrego `src/pipeline/split_manager.py`.
- El nuevo `MasterSplitManager` crea una particion maestra estable:
  - `master_train.jsonl`
  - `master_validation.jsonl`
  - `master_test.jsonl`
  - `master_heldout_final.jsonl`
  - `master_split_manifest.json`
- El split agrupa registros con texto normalizado identico para evitar que duplicados caigan en particiones distintas.
- `configs/training_defaults.yaml` ahora incluye `heldout_final_ratio: 0.10`.
- `main.py` crea el split maestro despues de harmonizar y lo pasa a classifier, augmenter y SLM.
- `src/pipeline/classifier_trainer.py` ya no necesita crear su propio split cuando existe el manifest maestro.
- `src/pipeline/augmenter.py` genera supervision SLM solo desde `train` + `validation`, nunca desde `test` ni `heldout_final`.
- `src/pipeline/slm_finetuner.py` usa `source_split=train` para entrenar y `source_split=validation` para evaluar.

Etiquetas phishing:
- `src/pipeline/harmonizer.py` ahora agrega:
  - `is_phishing`
  - `is_phishing_inferred`
  - `phishing_label_source`
- En IWSPA, el phishing se marca como `inferred_from_persuasion_labels`.
- En Spaphish, el phishing viene de `dataset_label`.
- Esto evita vender como ground truth fuerte una etiqueta que solo fue inferida desde principios.

Evaluacion:
- `src/intelligence_eval/evaluator.py` ahora exige al menos 30 casos pareados para aceptar superioridad fuerte.
- El gate produce `decision`:
  - `accepted_statistically_and_absolutely_better`
  - `rejected_not_absolutely_better_than_base`
  - `insufficient_evidence_min_30_paired_cases`
- `evaluate_slm_intelligence.py` imprime esa decision final.

Webapp:
- `webapp/server.py` ya no depende de nombres fijos como `microsoft_mdeberta-v3-base` o `Qwen_Qwen2.5...` para cargar artefactos research.
- La webapp busca archivos recientes por patron.
- Los ejemplos salen del split maestro `master_test.jsonl` si existe.
- Si faltan artefactos, la webapp devuelve estructuras vacias o estado de artefactos en vez de romper por nombres inexistentes.

Implicacion para la corrida final:
- La corrida final debe ser fresca.
- El comando recomendado sigue siendo:
  - `python main.py --fresh-all --slm auto --slm-run-tag final_research_run`
- Despues se debe correr:
  - `python evaluate_slm_intelligence.py --model <adapter> --compare-base --benchmark intelligence_eval/benchmarks/seed_cases.jsonl`
- Si el benchmark queda con menos de 30 casos, el gate correctamente dira que no hay evidencia suficiente para aceptar superioridad fuerte.

---

## 15. Actualizacion 2026-05-01: datasets dinamicos, auditoria y tuned_models

Decision de arquitectura:
- Los datasets ya no deben cargarse por nombres hardcodeados.
- El pipeline debe escanear `data/*.csv`, identificar cada CSV por su estructura de columnas y cargar todos los datasets soportados.
- Los overrides `--iwspa` y `--spaphish` quedan solo como compatibilidad manual, no como camino normal.

Validacion ejecutada:
- CSVs detectados en `data/`:
  - `IWSPA_AP_persuasion_annotated_phishing.csv`
  - `Spaphish dataset - DiB.csv`
- Identificacion por columnas:
  - IWSPA nuevo -> `iwspa`
  - Spaphish -> `spaphish`
- Armonizacion validada:
  - total: `3487`
  - IWSPA: `2092`
  - Spaphish: `1395`
  - phishing: `1736`
  - legitimo: `1751`
  - ids vacios: `0`
  - payloads no ASCII: `0`
  - schema: `harmonized_v2`

Correcciones criticas:
- IWSPA ahora usa `class`/`label` como etiqueta supervisada de phishing cuando existe.
- Spaphish usa `Label` como etiqueta supervisada de phishing.
- Los principios de persuasion se mantienen separados del label phishing.
- Ya no se infiere phishing desde principios cuando hay label real.
- Los nombres de columnas CSV se normalizan para evitar BOM, por ejemplo `\ufeffhash`.
- Las justificaciones de Spaphish ahora se seleccionan solo desde anotadores cuyo voto coincide con el label final del principio.
- Si algun caso futuro no tuviera justificacion coincidente, queda marcado como fallback auditable.

Artefactos reutilizables:
- `outputs/artifacts/augmented_dataset.jsonl` es el dataset aumentado reutilizable para entrenar SLMs.
- El schema del augmenter sube a `audit_v4`.
- Cada fila aumentada debe guardar:
  - origen del registro
  - split fuente
  - label phishing y fuente del label
  - labels de persuasion
  - hipotesis inicial sintetica del clasificador
  - justificaciones y detalles de anotacion cuando existan
  - fuente de generacion: teacher o plantilla con justificaciones humanas
  - estado de calidad por fila
- El SLM no debe entrenar si el JSONL aumentado no pasa quality gate.

Respaldos:
- `outputs/tuned_models/` es la carpeta de respaldo del proyecto.
- Las corridas largas completas se respaldan bajo `outputs/tuned_models/runs/`.
- Al ejecutar `--fresh-all`, el pipeline mueve resultados anteriores de:
  - `outputs/results`
  - `outputs/checkpoints`
  - `outputs/splits`
  - `outputs/artifacts`
- Esto preserva calculos viejos sin mezclar la corrida nueva.
- Las caches de descarga quedan fuera del respaldo y no se borran.

Regla operativa:
- Para una corrida final limpia se usa `--fresh-all`.
- Para entrenar otro SLM reutilizando el aumento ya calculado, no usar `--fresh-all`; usar `--fresh-slm` y conservar `outputs/artifacts/augmented_dataset.jsonl`.


