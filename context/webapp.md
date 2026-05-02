# WEBAPP NOTES: Product, UX, and Demo Strategy

> Working document for the future web application. This file captures the product vision, UX direction, and design constraints discussed while the training pipeline is still running, so implementation can start later with a clear blueprint.

---

## 1. Why we want the webapp

The webapp is not just a frontend for the project.

It should serve several purposes at the same time:

- demonstrate the scientific and technical value of the system
- show the practical value of the classifier + SLM combination
- make the project understandable to non-expert users
- act as a convincing demo for colleagues, reviewers, talks, and paper presentations
- function as a useful analysis tool, not only as a visual showcase

The app should communicate that this project is about more than classification.

The real value is the ability to analyze a message, explain the persuasion principles present in it, and allow the user to interact with the system to understand the message more deeply.

---

## 2. Core product vision

The webapp should allow a user to:

- paste a suspicious email, SMS, or text message
- detect persuasion principles in that message
- inspect confidence scores and analysis summaries
- see textual evidence supporting the decision
- ask the SLM follow-up questions about the message
- learn what the persuasion principles mean
- view scientific outputs, metrics, and experiment artifacts

This means the app must combine:

- analysis
- explanation
- interaction
- didactics
- research visibility

---

## 3. Product identity

The app should feel:

- visually attractive but sober
- technically careful and serious
- educational without feeling childish
- research-oriented without becoming unusable
- practical as a real tool

It should not feel like:

- a generic chatbot
- a cold enterprise dashboard
- an AI toy
- a flashy demo with no rigor

The app should feel like a:

- persuasion analysis desk
- forensic workbench
- research-grade analyst assistant

---

## 4. Main audiences

The application should work for several user profiles:

### 4.1 Non-expert users

They need:

- clear language
- guided examples
- simple explanations
- visible interpretation support

### 4.2 Technical/research colleagues

They need:

- scores
- evidence
- principle breakdown
- logs
- methodological traceability

### 4.3 Reviewers / academic evaluators / presentation audiences

They need:

- visible scientific depth
- experiment artifacts
- clarity of system behavior
- trust that the app reflects real research rather than a superficial UI

---

## 5. Core functional pillars

The app should combine five major pillars:

### 5.1 Message analysis

- paste text
- analyze it
- return detected persuasion principles
- display confidence and overall interpretation

### 5.2 Explainability

- explain why principles were detected
- show evidence in the text
- connect reasoning to model outputs

### 5.3 Conversation with the SLM

- allow the user to ask follow-up questions about the current message
- let the SLM discuss the principles, evidence, ambiguity, and meaning
- keep the conversation anchored to the currently loaded message

### 5.4 Didactic support

- explain what each principle means
- show examples
- help non-experts interpret outputs

### 5.5 Research and experiment visibility

- expose figures, tables, metrics, and results from the experimental pipeline
- support scientific demos and later paper presentations

---

## 6. Central role of the SLM in the app

The app should not treat the SLM as a minor extra.

It should be a central feature because the broader project goal is to create a system that allows users to "talk" with messages and understand the persuasion strategies used in them.

This means:

- the classifier detects
- the SLM explains, elaborates, and helps interpret

The SLM conversation should be:

- contextual
- grounded in the current message
- connected to detected principles
- connected to textual evidence
- useful for non-experts

It should not be a free-floating general chatbot.

It should feel like:

- a message-specific reasoning assistant
- an explainability layer on top of the prediction system

Important product decision:

- this is not optional
- the SLM conversation is a central feature of the product vision
- one of the main goals of the overall project is precisely to enable message-centered interaction about persuasion principles

### 6.1 Refined relation between classifier and SLM

The app should not frame the SLM as:

- a narrow evidence extractor
- a decorative chatbot
- a passive narrator that simply restates classifier labels

The intended design is:

- the classifier produces the initial structured hypothesis
- the SLM receives that hypothesis as starting context
- the SLM audits, expands, and interprets it against the actual message

This means the conversational layer should preserve the SLM's analytic freedom while keeping it anchored to:

- the current message
- the classifier bundle
- the textual evidence

Desired behavior:

- confirm a principle when the text supports it
- question or weaken a principle when the evidence is thin
- explain ambiguity
- compare principles within the same message
- move naturally between simple, educational, and technical explanation modes

Undesired behavior:

- repeating labels without analysis
- acting like a rigid span-extractor only
- obeying the classifier as if it were ground truth

The webapp should therefore present the system as:

- classifier = quantitative prior
- SLM = analyst-facing reasoning layer

---

## 7. Suggested interaction style for the SLM

The conversation interface should support prompts such as:

- `Why was Authority detected?`
- `Show the textual evidence`
- `Explain this in simple language`
- `Which principle is strongest here?`
- `What makes this manipulative?`
- `Is there ambiguity in this case?`
- `How would a non-expert be persuaded by this message?`
- `What is the difference between Authority and Social Proof in this case?`

Useful conversation modes:

- `Simple`
- `Technical`
- `Educational`
- `Analyst`

The app should make it obvious that the SLM is talking about:

- this message
- this analysis
- these persuasion principles

Additional interaction principle:

- the SLM should begin from the classifier's reading of the case, but it must remain free to validate, challenge, nuance, and deepen that reading

So the conversation should feel less like:

- "the classifier said X and the SLM is repeating X"

And more like:

- "the classifier surfaced a plausible hypothesis and the SLM is now reasoning through that hypothesis with the user"

---

## 8. High-level information architecture

The app should feel like a single coherent workspace with several major areas:

- `Home`
- `Analyze`
- `Converse`
- `Learn`
- `Evidence`
- `Research`

These can be separate pages or sections within one main workspace.

---

## 9. Wireframe-level vision

### 9.1 Home / Landing

Purpose:

- explain what the system does in less than 20 seconds
- make the project feel serious, memorable, and research-based

Content:

- title / product identity
- short description
- clear call to action
- credibility strip:
  - multi-label classifier
  - SLM-based explanation
  - bilingual analysis
  - research artifacts
- a miniature preview of the analysis workflow

### 9.2 Main analysis workspace

The main screen should resemble an analysis desk.

Suggested layout:

- left panel: message input
- center panel: analysis results
- right panel: SLM conversation
- bottom or collapsible panel: logs and trace

### 9.3 Left panel: input

Should include:

- large message text area
- example loader
- language mode
- analysis mode
- main `Analyze Message` button

The input experience must be approachable for non-experts.

### 9.4 Center panel: results

Should include:

- analysis status header
- detected principles summary
- confidence profile
- short narrative explanation
- evidence highlights in the text
- short cards per persuasion principle

### 9.5 Right panel: SLM conversation

Should include:

- persistent chat history
- current-message context reminder
- suggested prompts
- input box for natural-language questions
- support for different explanation modes

### 9.6 Learn section

Should include:

- explanation of each persuasion principle
- examples
- basic interpretation guidance
- simple non-expert educational content

### 9.7 Example gallery

Should include:

- curated phishing / suspicious message examples
- bilingual examples
- ambiguous examples
- multi-principle examples

This will be useful for:

- demos
- presentations
- onboarding
- non-expert exploration

### 9.8 Evidence / trace section

Should expose:

- message metadata
- language detection
- class scores
- thresholded outputs
- links to generated artifacts
- timestamp / run metadata

This must be available without overwhelming casual users.

### 9.9 Research dashboard

Should include:

- learning curves
- split summaries
- per-class metrics
- ROC curves
- PR curves
- threshold sweeps
- confusion summaries
- downloadable figures and tables

This supports:

- scientific transparency
- paper demos
- review discussions
- internal project validation

---

## 10. Need for visible logs

Logs are considered important, not optional.

The app should clearly indicate:

- where the user is
- what the system is doing
- what step is currently running
- whether the process is completed

Example visible logs:

- `Message received`
- `Language detected: Spanish`
- `Running classifier inference`
- `Scoring persuasion principles`
- `Generating explanation with SLM`
- `Preparing evidence highlights`
- `Analysis complete`

The logs should feel like a clean, readable technical trace.

Why this matters:

- increases trust
- improves clarity
- supports demos
- reinforces the sense that the tool is serious and methodical

---

## 11. UX requirement: support non-expert users

This is one of the most important constraints.

The system must be useful to people who are not experts in:

- phishing
- machine learning
- persuasion theory
- explainability

This means:

- simpler explanations should always be available
- jargon should be minimized in the default view
- examples must be easy to access
- the app should guide the user, not assume prior knowledge

At the same time, advanced technical layers must remain available for expert use.

---

## 12. Scientific visibility requirement

The app should make the scientific work behind the project visible.

This includes:

- figures
- tables
- metrics
- artifacts
- traces
- evidence-based explanation

The point is not only to make the app useful, but also to make the seriousness of the research visible.

This is important for:

- demos
- paper support
- colleague communication
- reviewer persuasion

---

## 13. Visual direction

Desired visual tone:

- sober
- attractive
- technically polished
- not generic
- not dark-dashboard cliche unless it truly improves readability

Potential visual identity:

- forensic / analytical / editorial
- careful typography
- refined spacing
- subtle motion
- research-grade information layout

Visual principles:

- clarity first
- hierarchy matters
- use color intentionally for principle categories, confidence, and system states
- avoid visual noise

---

## 14. Information layers

The app should operate in at least two clearly distinguishable layers:

### 14.1 Practical layer

Focused on:

- analyze message
- see results
- ask questions
- understand the message

### 14.2 Scientific layer

Focused on:

- inspect metrics
- review evidence
- trace outputs
- connect the app to the experiment and research artifacts

This dual-layer design helps the app remain useful to both beginners and experts.

---

## 15. Suggested MVP

The initial implementation should probably focus on:

- Home
- Analyze workspace
- SLM contextual chat
- Example gallery
- Logs panel
- Basic research tab

Later expansions can include:

- Learn mode
- compare-messages workflow
- richer evidence tools
- annotation-assistant behavior

---

## 16. Why we are not implementing yet

Current strategy:

- use this time to define the app properly while training is still running
- wait for the real results and artifacts before full implementation

Why:

- final outputs may influence the best UX decisions
- real model behavior will shape what the app should emphasize
- it is more efficient to define now and implement after results are available

So the plan is:

1. define product and UX now
2. finish the training run
3. inspect outputs
4. implement the app with real artifacts and realistic model behavior

---

## 17. Main memory anchor for later

If we resume later, the key idea is:

The webapp should be a visually careful, sober, technically credible, didactic, and practically useful system for analyzing phishing persuasion, showing scientific depth, and allowing message-centered conversation with the SLM.

---

## 18. Webapp prototype status

A first static prototype now exists under:

- `webapp/index.html`
- `webapp/styles.css`
- `webapp/app.js`
- `webapp/server.py`

Purpose of this prototype:

- validate the product direction visually
- test layout, hierarchy, and interaction concepts
- rehearse how classifier outputs, SLM conversation, logs, and research panels should coexist

Current serving approach:

- the webapp prototype can now be served locally with FastAPI
- static assets are exposed through `/static`
- the root route `/` returns the webapp entry point
- a convenience launcher now exists at `webapp.py`, so the app can be started with `python webapp.py`

Current language decision:

- the application experience should be fully in Spanish
- analyzed messages may still be in English or Spanish
- interface text, guidance, explanations, logs, and educational framing should default to Spanish

Current review status:

- the first visual prototype was already reviewed and judged to be an excellent direction
- the current phase is still product validation, not real model integration
- the next future step after training is to connect the UI to real classifier, SLM, and research outputs

Important constraint for the prototype data:

- dummy content should stay aligned with outputs the pipeline can realistically produce
- avoid inventing unsupported features or impossible model behavior
- when real training artifacts are ready, the prototype should be upgraded by replacing dummy values with actual outputs rather than redesigning the app from scratch

## 19. Connected webapp status

After the successful end-to-end run `20260429_074051`, the webapp was upgraded from a static prototype to a connected local application.

Current integration status:

- `webapp/server.py` now exposes real API endpoints:
  - `/api/bootstrap`
  - `/api/examples`
  - `/api/research/summary`
  - `/api/analyze`
  - `/api/chat`
  - `/api/artifacts/{kind}/{filename}`
- `webapp/app.js` no longer depends on embedded dummy scenarios for the main flow
- the app now consumes:
  - real examples from the classifier test split
  - real research metrics and learning curves from `outputs/results/reports` and `outputs/results/tables`
  - real classifier inference from the trained `microsoft_mdeberta-v3-base` artifact

Current behavior of the conversational layer:

- the backend attempts to load the fine-tuned Qwen adapter locally
- if the required local base model cache is not available, the app falls back to a deterministic contextual assistant mode instead of crashing
- this means the webapp remains functional even when full local SLM chat cannot be restored instantly

Conceptual product requirement recorded after integration review:

- the long-term goal is not to replace the SLM with a deterministic extractor
- the long-term goal is to preserve a high-capability, message-centered analyst experience
- therefore future backend changes should improve grounding and discipline without collapsing the SLM into a narrow evidence-only component

Current launcher decision:

- the canonical local command remains:
  - `python webapp.py`
- the launcher was adjusted so the root script `webapp.py` no longer conflicts with the `webapp/` directory import path

Current validation status:

- backend bootstrap loading was validated locally
- backend analysis generation was validated locally against the trained classifier artifact
- full HTTP test automation was not executed because the environment lacked `httpx`
- the next practical validation step is interactive browser testing through the FastAPI server

## 20. Refinamiento de UX y Chat Humano (Sesion 2026-04-29)

### Humanizacion de la Respuesta
- Se elimino la exposicion de probabilidades numericas (0.91, 0.4, etc.) en las respuestas directas del chat para evitar que el sistema parezca un script tecnico.
- Se eliminaron las etiquetas de estado redundantes como "(detectado)" o "(senal debil)" en el texto narrativo, moviendo esa informacion a la interpretacion cualitativa del SLM.
- El tono del chat ahora es mas directo, profesional y humano, centrado en la evidencia y no en el reporte de metricas.

### Mejoras en el Panel de Actividad
- **Timestamps**: Ahora incluyen segundos para una trazabilidad tecnica precisa durante los demos.
- **Color-Coding**: Se implemento una paleta de colores por etapa de ejecucion:
  - **Entrada**: Gris (Recepcion de datos).
  - **Clasificador**: Purpura (Inferencia multi-label).
  - **SLM**: Rosa (Razonamiento narrativo).
  - **Idioma**: Verde (Deteccion linguistica).
  - **Riesgo/Phishing**: Rojos (Alertas criticas).
- **Legibilidad**: Se incremento el tamano de fuente del chat y se optimizo el espaciado para que el texto sea el protagonista.

### Soberania de Etiquetas
- Se restauraron las etiquetas originales del dataset (`authority`, `social_proof`, etc.) en toda la interfaz. Esto elimina el riesgo de "alucinaciones de traduccion" y mantiene la consistencia cientifica con el paper.

### 21. Auditoria de Sistema y Telemetria Forense (Finalizacion Sesion 2026-04-29)

#### Dashboard de Telemetria Real
- Se reemplazo el subtitulo estatico del Hero por un monitor de hardware en tiempo real.
- La cabecera ahora muestra: **Modelo de GPU activo**, **VRAM libre/total**, **RAM de sistema** y **Cores de CPU**.
- Se anadio un indicador visual de **Privacidad Local-Only** para reforzar la seguridad de los datos.

#### Estabilizacion del Cerebro (SLM)
- **Modo Experto**: El prompt del chat se redefinio para actuar como un "Analista Senior de Ciberseguridad", con autoridad para corregir al clasificador estadistico si la evidencia textual lo amerita.
- **Presupuesto Dinamico**: Implementacion de `_calculate_dynamic_max_tokens` para ajustar el limite de respuesta (128 - 1024 tokens) segun la complejidad de la pregunta.
- **Sampling Optimizado**: Ajuste de `temperature=0.25` y `repetition_penalty=1.1` mediante `GenerationConfig`, eliminando respuestas repetitivas y boilerplate robotico.

#### Decision conceptual posterior sobre el SLM
- El SLM debe ser tratado como un analista conversacional que parte del bundle del clasificador.
- El clasificador propone la lectura inicial del caso, pero el SLM debe tener margen para:
  - confirmar
  - matizar
  - refutar
  - comparar principios
  - explicar ambiguedad
- No se considera deseable convertir el chat en un modulo rigido de extraccion de citas.
- La disciplina deseada es "anclaje al texto", no "reduccion de capacidades".

#### Refinamiento de Interfaz (Look & Feel)
- **Identidad Visual**: Integracion de un logo minimalista y futurista junto al titulo principal.
- **Tipografia Profesional**: Migracion total a **Inter** (estandar UI/UX profesional).
- **Log Compacto**: Reduccion de gaps y ajuste de interlineado en el Log de Ejecucion para maxima densidad de informacion tecnica.
- **Sincronizacion Visual**: Igualacion de tamanos de fuente y alineacion entre el panel de analisis y el hilo de conversacion.

**Estado Final**: La Webapp ha pasado de ser un prototipo visual a un espacio de trabajo forense maduro, capaz de auditar modelos de forma transparente y con una experiencia de usuario de alto nivel.

## 22. Implicaciones nuevas para la webapp tras la auditoria del SLM

La auditoria interna realizada el `2026-04-29` cambia la lectura del estado actual del chat:

- la webapp ya esta mejor orquestada que antes
- pero el adapter afinado actual no esta a la altura del rol conversacional esperado
- el modelo base de Qwen rinde ligeramente mejor que el adapter actual en el benchmark semilla

Resultado clave:
- el problema no esta solo en la UX o en el prompt de la webapp
- el problema central esta en el dataset y el objetivo de fine-tuning del SLM

### 22.1 Lo que ya sabemos que NO es suficiente
No basta con:
- mejorar prompts del chat
- endurecer un poco el sistema de respuesta
- anadir mas reglas locales contra alucinacion

Eso ayuda, pero no corrige la raiz:
- el adapter fue entrenado para justificar etiquetas
- no para actuar como analista conversacional disciplinado por el texto

### 22.2 Comportamiento deseado de la webapp a futuro
La webapp debe seguir defendiendo esta arquitectura:
- clasificador = hipotesis cuantitativa inicial
- SLM = analista conversacional del mensaje

Pero la proxima iteracion debe exigir que el SLM pueda:
- citar o parafrasear de forma claramente anclada al mensaje
- cuestionar la hipotesis inicial si la evidencia es pobre
- comparar principios sin salir hacia abstracciones genericas
- admitir honestamente cuando el texto no alcanza
- responder preguntas abiertas sin convertirse en una "cotorra estocastica"

### 22.3 Restricciones de producto derivadas del diagnostico
Mientras no exista un adapter mejor:
- la webapp no debe sobrerrepresentar la capacidad del SLM actual
- el chat debe seguir teniendo frenos de honestidad y grounding
- el modo conversacional debe tratar al adapter como experimental, no como analista plenamente confiable

Esto NO significa:
- convertir el chat en un extractor rigido
- matar la libertad analitica del SLM

Si significa:
- mantener el ideal de `SLM as analyst`
- pero reconocer que el adapter actual aun no alcanza ese rol de forma robusta

### 22.4 Requisitos para la siguiente integracion fuerte
La siguiente version del chat deberia desplegar su "potencia" solo cuando el nuevo fine-tuning haya demostrado en benchmark que:
- supera al modelo base
- mejora grounding
- no empeora honestidad
- no introduce mas deriva conceptual

### 22.5 Conclusion de producto
La vision de la webapp no cambia:
- el chat sigue siendo central
- el objetivo sigue siendo hablar con el mensaje
- el clasificador sigue siendo el prior

Lo que cambia es la comprension del estado actual:
- la webapp ya apunta en la direccion correcta
- el cuello de botella real esta hoy en la calidad del adapter y del pipeline generativo

## 23. Actualizacion 2026-04-30: chat alineado con `SLM as analyst`

La webapp fue ajustada para no tratar la salida del clasificador como verdad final:
- `/api/analyze` mantiene el bundle del clasificador como hipotesis inicial visible.
- `/api/chat` entrega al SLM el texto fuente, la pregunta del usuario y una hipotesis estructurada con scores, estados y principios visibles.
- El prompt de chat exige citar fragmentos si se habla de evidencia.
- Si el texto no basta para responder, el SLM debe decirlo con honestidad.
- Queda prohibido inventar intenciones, hechos externos, entidades o evidencia.
- La generacion del chat ahora es determinista y usa presupuesto dinamico de tokens.

Esto preserva la idea central:
- el clasificador orienta
- el SLM conversa, audita y razona
- el texto manda

Estado operativo:
- no se ha probado con un adapter nuevo porque aun no se ha ejecutado la corrida final fresca
- la webapp queda preparada para desplegar el nuevo adapter cuando el benchmark confirme que supera al modelo base

## 24. Actualizacion 2026-04-30: SLM especialista, phishing y seleccion de modelo

La meta funcional queda formalizada asi:
- el SLM debe conversar sobre el texto como analista de ciberseguridad
- debe detectar y explicar principios de persuasion
- debe entender si el mensaje parece phishing, legitimo o ambiguo
- debe separar evidencia textual de inferencias
- debe admitir limites cuando falten datos

Cambios de evaluacion:
- El benchmark semilla ahora incluye phishing claro, legitimos, ambiguos y casos donde el clasificador esta equivocado.
- La evaluacion compara adapter contra base con estadistica pareada.
- La webapp no debe tratar un adapter como "especialista" hasta que supere el gate estadistico.

Cambios de seleccion de modelo:
- El pipeline ahora puede usar `--slm auto`.
- En modo auto, se encuesta el hardware y se consulta Hugging Face para escoger un candidato adecuado.
- La seleccion no debe depender de un nombre hardcodeado.
- El advisor no debe contener semillas de modelos, familias preferidas ni fallback hardcodeado.
- Si no hay internet, solo puede usar modelos descubiertos automaticamente en la cache local.
- Si no hay internet ni cache local valida, debe fallar y pedir `--slm` explicito.

Regla de producto:
- Si el adapter no supera estadisticamente al modelo base, la webapp puede usarlo como experimental, pero no debe presentarlo como especialista validado.

## 25. Actualizacion 2026-04-30: webapp desacoplada de corridas fijas

Problema corregido:
- La webapp tenia nombres fijos de modelos y artefactos de corrida.
- Eso podia hacer que una corrida final con otro SLM o con otro timestamp quedara mal reflejada en `/api/research/summary`.

Cambios:
- La webapp ahora descubre el split activo desde `master_split_manifest.json` cuando existe.
- Los ejemplos de UI salen de `master_test.jsonl`.
- Si no hay manifest maestro, usa el archivo `*_test.jsonl` mas reciente.
- Los artefactos research se resuelven por patron:
  - `*_paper_class_metrics.json`
  - `*_paper_summary_metrics.json`
  - `*_paper_split_summary.json`
  - `*_slm_paper_summary.json`
  - `*_paper_training_history.csv`
  - `*_slm_paper_training_history.csv`
- La respuesta research incluye `artifactStatus` para saber exactamente que archivos alimentan la UI.

Regla de producto:
- La webapp debe mostrar la corrida activa, no una corrida historica hardcodeada.
- Si falta un artefacto, debe degradar con informacion vacia/controlada, no explotar ni inventar metricas.

## 26. Actualizacion 2026-05-01: reglas de datos para webapp y SLM

La webapp y el SLM deben respetar esta separacion:
- `is_phishing` viene de un label supervisado cuando el dataset lo trae.
- Los principios de persuasion son un objetivo distinto.
- Un principio de persuasion puede aparecer en mensajes legitimos.
- Un mensaje phishing puede o no usar todos los principios.
- La UI no debe convertir persuasion en acusacion automatica de phishing.

El dataset aumentado para el chat/SLM es reutilizable:
- `outputs/artifacts/augmented_dataset.jsonl`
- schema requerido: `audit_v4`
- si el archivo existe y pasa quality gate, se puede reutilizar para entrenar otro adapter SLM sin regenerar teacher.

Regla de operacion:
- `--fresh-all` archiva corridas anteriores en `outputs/tuned_models/runs/` y reconstruye todo.
- `--fresh-slm` debe usarse cuando se quiere entrenar otro SLM sobre el mismo dataset aumentado ya calculado.
- Nunca borrar caches de descarga para limpiar una corrida.
