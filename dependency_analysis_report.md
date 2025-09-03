# Pipeline Dependency Analysis Report
Generated: 2025-09-03T13:37:22.923023

## Summary Statistics
- Total Components: 341
- Total Phases: 17
- Total Dependencies: 192246
- Cycles Detected: 300
- DAG Valid: False

## Phase Distribution
- **unknown**: 268 components
- **analysis_core**: 18 components
- **O_orchestration**: 28 components
- **G_reporting**: 8 components
- **S_synthesis**: 4 components
- **T_storage**: 12 components
- **R_retrieval**: 21 components
- **L_classification**: 16 components
- **evaluation_core**: 4 components
- **A_analysis**: 11 components
- **I_ingestion**: 11 components
- **K_knowledge**: 14 components
- **knowledge_core**: 5 components
- **calibration**: 10 components
- **X_context**: 6 components
- **ingestion_core**: 1 components
- **mathematical**: 14 components

## ⚠️  Detected Cycles
The following cycles were detected in the dependency graph:
1. validator → contracts_validation_utility → evidence_processor → validator
2. evidence_system → evidence_processor
3. visual_testing_framework → validator
4. run_l_stage_tests → validator
5. dependency_analysis_module → validator
6. module_distributed_processor → evidence_processor
7. airflow_orchestrator → evidence_processor
8. step_handlers → EXTRACTOR DE EVIDENCIAS CONTEXTUAL → evidence_processor
9. validate_contract_imports → question_analyzer → validator
10. validate_mathematical_compatibility_matrix_minimal → models → validator
11. smoke_tests → validator
12. example_constraint_usage → meso_aggregator → validator
13. run_contract_validation → evidence_processor
14. run_integrity_tests → report_compiler → validator
15. config_consolidated → ocr → evidence_processor
16. validate_k_workflow_tests → validator
17. validate_recovery_system → schema_registry → validator
18. implementacion_mapeo → validator
19. validate_thresholds → question_analyzer
20. run_mcc_tests → models
21. run_refusal_tests → validate_dependency_compatibility → validator
22. validate_l_orchestrator → validator
23. run_audit_validation → EXTRACTOR DE EVIDENCIAS CONTEXTUAL
24. total_ordering_base → evidence_processor
25. recovery_system → validator
26. project_health_check → EXTRACTOR DE EVIDENCIAS CONTEXTUAL
27. automated_dependency_resolver → EXTRACTOR DE EVIDENCIAS CONTEXTUAL
28. validate_audit_logger → validate_dependency_compatibility
29. run_demo_quick → data_integrity_checker → validator
30. pdf_reader → validator
31. monitoring_dashboard → validator
32. parallel_processor → validator
33. fault_injector → validator
34. validate_g_aggregation → validator
35. exception_monitoring → evidence_processor
36. rubric_validator → evidence_processor
37. metadata → evidence_processor
38. fix_syntax_errors → fix_remaining_imports → query_generator → evidence_processor
39. answer_formatter → evidence_processor
40. serializable_wrappers → evidence_processor
41. causal_graph → evidence_processor
42. analysis_nlp_orchestrator → validator
43. telemetry_collector → validator
44. dnp_alignment_engine → step_handlers
45. conformal_risk_demo → validate_l_orchestrator
46. dependency_manager → validator
47. lexical_index → evidence_processor
48. run_tests → validate_dependency_compatibility
49. canonical_output_auditor → validator
50. workflow_engine → validator
51. core_orchestrator → evidence_processor
52. json_canonicalizer → validator
53. workflow_definitions → visual_testing_framework
54. integration_example → EXTRACTOR DE EVIDENCIAS CONTEXTUAL
55. improved_mapeo → validator
56. validate_atroz_server → raw_data_generator → validator
57. deterministic_hybrid_retrieval → evidence_processor
58. alert_system → evidence_processor
59. hybrid_retrieval → dependency_analysis_module
60. pipeline_orchestrator → evidence_processor
61. validate_comprehensive_ci_system → evidence_processor
62. audit_last_execution → evidence_processor
63. pattern_matcher → evidence_processor
64. distributed_processor → evidence_processor
65. compensation_engine → evidence_processor
66. scoring → run_l_stage_tests
67. exception_telemetry → evidence_processor
68. validate_pipeline_dependencies → evidence_processor
69. document_processor → retrieval_engine → evidence_processor
70. mathematical_compatibility_matrix → validator
71. evaluation_driven_processor → validator
72. run_pipeline_analysis → validator
73. validate_meso_aggregator → validate_dependency_compatibility
74. packager → step_handlers
75. setup_visual_testing → json_canonicalizer
76. run_basic_tests → validate_dependency_compatibility
77. update_distributed_processor_demo → raw_data_generator
78. __init__ → ui_state → validator
79. pipeline_orchestrator_audit → EXTRACTOR DE EVIDENCIAS CONTEXTUAL
80. run_macro_alignment_demo → validator
81. validate_pipeline_analysis → dependency_analysis_module
82. constraint_validator → evidence_processor
83. score_calculator → evidence_processor
84. validate_deterministic_embedder → validator
85. hybrid_retrieval_bridge → evidence_processor
86. Comprehensive dependency analysis for PIROBOSTALES system Identifies root causes of import failures and dependency conflicts → evidence_processor
87. Comprehensive Deterministic Pipeline Validator with Micro-Level Characterization → validator
88. validate_decalogo_artifacts → run_l_stage_tests
89. analytics_enhancement → evidence_processor
90. decision_engine → evidence_processor
91. handoff_validation_system → validator
92. intelligent_recommendation_engine → evidence_processor
93. stream_processor → validator
94. optimization_engine → evidence_processor
95. cluster_execution_controller → evidence_processor
96. validate_memory_optimizations → pipeline_orchestrator
97. process_inventory → evidence_processor
98. pipeline_value_analysis_system → validator
99. validate_audit_system → EXTRACTOR DE EVIDENCIAS CONTEXTUAL
100. installation_troubleshooting → evidence_processor
101. validate_dependencies → evidence_processor
102. confidence_quality_metrics → validator
103. contract_validator → evidence_processor
104. validate_monitoring → evidence_processor
105. src:nlp_engine:semantic_inference_engine → evidence_processor
106. run_safety_demo → improved_mapeo
107. normative_validator → validator
108. causal_dnp_framework → evidence_processor
109. pdf_text_reader → evidence_processor
110. comprehensive_pipeline_orchestrator → validator
111. validate_refactor → visual_testing_framework
112. gw_alignment → visual_testing_framework
113. lineage_tracker → evidence_processor
114. snapshot_manager → validator
115. deterministic_pipeline_validator → validator
116. contract_system → validator
117. orjson → dependency_analysis_module
118. run_comprehensive_validation → Comprehensive Deterministic Pipeline Validator with Micro-Level Characterization
119. ComprehensiveStaticAudit → validator
120. validate_stage_middleware → contract_system
121. event_schemas → total_ordering_base
122. metrics_collector → evidence_processor
123. audit_trail → evidence_processor
124. audit_logger → validator
125. circuit_breaker → evidence_processor
126. validate_parallel_processor → question_analyzer
127. stage_validation_middleware → validator
128. dnp_alignment_adapter → validator
129. recovery_scripts → evidence_processor
130. validate_l_stage_tests → evidence_processor
131. enhanced_core_orchestrator → evidence_processor
132. validate_decalogo_registry → integration_example
133. validate_advanced_loader → module_distributed_processor
134. deterministic_shield → evidence_system
135. validate_import_safety → models
136. evidence_validation_model → validator
137. advanced_loader → validator
138. serialization_config → EXTRACTOR DE EVIDENCIAS CONTEXTUAL
139. PIPELINEORCHESTRATOR → visual_testing_framework
140. theory_validator → evidence_processor
141. validate_safety_controller → validate_dependency_compatibility
142. validate_installation → question_analyzer
143. data_models → advanced_knowledge_graph_builder → validator
144. adaptive_scoring_engine → evidence_processor
145. embedding_generator → evidence_processor
146. example_lineage_usage → run_l_stage_tests
147. run_canonical_stability → total_ordering_base
148. validate_operadic_implementation → question_analyzer
149. atroz_api_demo → evidence_processor
150. structure_parser → validator
151. run_canonical_audit_demo → evidence_processor
152. normalizer → Copia de pattern_matcher → validator
153. stage_justification_framework → validator
154. submodular_selector_demo → total_ordering_base
155. path_verification → validate_dependency_compatibility
156. traceability → retrieval_engine
157. validate_handoff_system → validate_dependency_compatibility
158. kubernetes_istio_integration → run_l_stage_tests
159. adaptive_controller → evidence_processor
160. deterministic_flow_risk_guard → validator
161. run_g_aggregation_tests → validator
162. evidence_router → evidence_processor
163. main → validator
164. comprehensive_import_validator → evidence_processor
165. answer_synthesizer → evidence_processor
166. EnvironmentVerification → evidence_processor
167. validate_feature_extractor → validate_dependency_compatibility
168. run_integration_tests → validator
169. validate_dashboard_implementation → validate_dependency_compatibility
170. conformal_risk_certification_demo → evidence_processor
171. validate_mathematical_foundations → validate_dependency_compatibility
172. public_transformer_adapter → evidence_processor
173. question_level_scoring_pipeline → validator
174. atroz_integration_example → validator
175. run_k_workflow_tests → validator
176. feedback_loop → evidence_processor
177. installation_diagnostics → visual_testing_framework
178. gcp_io → EXTRACTOR DE EVIDENCIAS CONTEXTUAL
179. retrieval_trace → validator
180. decalogo_question_registry → validator
181. validate_analysis_nlp_components → pipeline_orchestrator
182. pdf_processing_error_handler → validator
183. automated_dependency_resolution → validator
184. macro_alignment_calculator → validator
185. demo_import_safety → validator
186. knowledge_extraction_error_handler → visual_testing_framework
187. service_discovery → evidence_processor
188. embedding_builder → evidence_processor
189. validate_mathematical_compatibility_matrix → validate_dependency_compatibility
190. GOVERNANCE → validator
191. per_point_scoring_system_demo → validate_l_orchestrator
192. dependency_analysis_tool → validator
193. connection_pool → dependency_analysis_module
194. table_extractor → evidence_processor
195. patterns → question_analyzer
196. stable_gw_aligner → evidence_processor
197. api → evidence_processor
198. graph_ops → run_l_stage_tests
199. apply_academic_essay → evidence_processor
200. label_explain → dependency_analysis_module
201. refusal_matrix → visual_testing_framework
202. canonical_path_auditor → evidence_processor
203. dependency_audit → evidence_processor
204. sort_sanity → validator
205. plan_diff → validator
206. certificate_generator → evidence_processor
207. organize_canonical_structure → evidence_processor
208. ot_digest → validator
209. rc_check → evidence_processor
210. generate_flux_report → run_l_stage_tests
211. apply_adv_graphics_stack → visual_testing_framework
212. pic_probe → validator
213. strategies → validator
214. demo_integration → validator
215. schemas → validator
216. preflight_validator → validator
217. provenance_tracker → evidence_processor
218. demo_orchestration → validate_audit_logger
219. orchestrator → validator
220. auto_deactivation_monitor → evidence_processor
221. pipeline_schemas → evidence_processor
222. store → EXTRACTOR DE EVIDENCIAS CONTEXTUAL
223. state_manager → validator
224. persistence → validator
225. backend_state → evidence_processor
226. transitions → evidence_processor
227. vector_index → evidence_processor
228. hybrid_retriever → evidence_processor
229. early_error_detector_demo → validator
230. reranker → evidence_processor
231. calibration_dashboard → evidence_processor
232. pipeline_state_manager → validator
233. mathematical_foundations → validator
234. deterministic_embedder → validator
235. mock_faiss → validator
236. mock_numpy → validator
237. mock_torch → validator
238. mock_sklearn → validator
239. mock_utils → validator
240. conformal_risk_control → evidence_processor
241. m_c_c_engine_monotone_compliance_evaluator → evidence_processor
242. deterministic_router → evidence_processor
243. mathematical_safety_controller → evidence_processor
244. early_error_detector → validator
245. m_c_c_label_evaluation_engine → evidence_processor
246. project_analyzer → validator
247. immutable_context → evidence_processor
248. confluent_orchestrator → evidence_processor
249. context_adapter → evidence_processor
250. total_ordering → evidence_processor
251. submodular_task_selector → evidence_processor
252. permutation_invariant_processor → validator
253. safety_controller → evidence_processor
254. linear_type_enforcer → evidence_processor
255. auto_enhancement_orchestrator → validator
256. import_safety → validator
257. task_selector_demo → validator
258. troubleshoot → json_canonicalizer
259. dimension_aggregator → validator
260. audit_validation → validator
261. integration_demo → run_l_stage_tests
262. dnp_causal_correction_system → EXTRACTOR DE EVIDENCIAS CONTEXTUAL
263. per_point_scoring_system → validator
264. artifact_generator → validator
265. macro_alignment → validator
266. calibration_artifacts → theory_validator
267. adaptive_analyzer → validator
268. extractor_evidencias_contextual → validator
269. retrieval_enhancer → evidence_processor
270. aggregation_enhancer → validator
271. mathematical_pipeline_coordinator → validator
272. hyperbolic_tensor_networks → evidence_processor
273. integration_enhancer → validator
274. knowledge_enhancer → validator
275. ingestion_enhancer → validator
276. pre_flight_validator → evidence_processor
277. scoring_enhancer → evidence_processor
278. orchestration_enhancer → validator
279. analysis_enhancer → evidence_processor
280. context_enhancer → validator
281. backpressure_manager → evidence_processor
282. causal_graph_constructor → validator
283. gate_validator → evidence_processor
284. chunking_processor → validator
285. entity_concept_extractor → validator
286. dnp_alignment_analyzer → validator
287. demo_schema_validation → dependency_analysis_module
288. stage_orchestrator → validator
289. demo_stage_orchestrator → validator
290. question_registry → evidence_processor
291. evidence_adapter → validator
292. decalogo_scoring_system → evidence_processor
293. conformal_prediction → validator
294. knowledge_audit_system → validator
295. hybrid_retrieval_core → evidence_processor
296. lexical_index_base → evidence_processor
297. gate_validation_system → evidence_processor
298. feature_extractor → validator
299. preflight_validation → evidence_processor
300. ingestion_orchestrator → validator