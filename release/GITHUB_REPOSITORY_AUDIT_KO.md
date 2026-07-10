# heosanghun 공개 GitHub 저장소 감사

최종 새로고침: 2026-07-10  
공개 저장소: 55개  
계정: <https://github.com/heosanghun?tab=repositories>

## 감사 방식

- 공개 저장소 55개의 이름, URL, 주 언어, GitHub 표시 크기, 최종 갱신 시각을
  API로 재목록화했습니다.
- Cogni-OS와 직접 연결되는 11개 저장소는 `work/upstream`에 shallow clone하여
  구현·테스트·보고서를 소스 수준으로 비교했습니다.
- 나머지 저장소는 거래, 웹/회사, RAG/agent, 원격조작 등의 범주로 triage했습니다.
  런타임은 어느 upstream checkout도 import하지 않습니다.
- 공개 코드라는 이유만으로 안전성·정확성을 신뢰하지 않고, 수학 gate와 자체
  회귀 테스트를 통과한 설계만 독립 구현했습니다.

## 직접 소스 검토·비교한 11개

1. [AkasicDB](https://github.com/heosanghun/AkasicDB)
2. [BIO-HAMA_MAIN](https://github.com/heosanghun/BIO-HAMA_MAIN)
3. [Cognitive-Tree-Search](https://github.com/heosanghun/Cognitive-Tree-Search)
4. [Cognitive-Tree-Search-2-](https://github.com/heosanghun/Cognitive-Tree-Search-2-)
5. [System1.5](https://github.com/heosanghun/System1.5)
6. [System1.5_260515](https://github.com/heosanghun/System1.5_260515)
7. [System2.5](https://github.com/heosanghun/System2.5)
8. [System3](https://github.com/heosanghun/System3)
9. [System3.5](https://github.com/heosanghun/System3.5)
10. [System4](https://github.com/heosanghun/System4)
11. [System5](https://github.com/heosanghun/System5)

## 전체 55개 목록

1. `-251008-ESWA_Dynamic_Ensemble_Trading_GitHub`
2. `1_ESWA_Dynamic_Ensemble_Trading`
3. `1_Multimodal-Trading-System`
4. `2_ESWARegime`
5. `2_Hybrid_VLM_Trading_System`
6. `3_hybrid_GR2PO_Trading_System`
7. `3_HyperGraphTrading`
8. `4_Hybrid_GEPA_Trading_System`
9. `AERO-TWIN-PRO-v2.1`
10. `AGENTIC-RAG`
11. `AkasicDB`
12. `AutonomousCompanyMVP`
13. `AutonomousCompanyRPG`
14. `BIO-HAMA_MAIN`
15. `btslovearmy`
16. `Cognitive-Tree-Search`
17. `Cognitive-Tree-Search-2-`
18. `crewai-multi-agent-blog`
19. `dynamic_ensemble_rl_trading`
20. `dynamic-ensemble-rl-trading-v2`
21. `dynamic-ensemble-rl-tradingv2`
22. `ESWA`
23. `ESWA_Dynamic_Ensemble_Trading`
24. `ESWA_Dynamic_Ensemble_Trading_GitHub_Upload`
25. `FashionRAG`
26. `FinAgent`
27. `FinAgent_251205`
28. `gpt_oss`
29. `heosanghun-retire-cashflow-sim-kr`
30. `Hybrid_GEPA_Trading_System`
31. `Hybrid_VLM_Trading_System`
32. `HyperGraphTrading`
33. `HyperGraphTrading_251204`
34. `Level1_GRPO`
35. `Level2_HybridGRPO`
36. `Level3_H-MTR`
37. `mcts-financial-world-model`
38. `ModuirumCompany_v1.0`
39. `ModuirumHompage`
40. `Multi_Level_RL_Trading_Systems`
41. `NotebookLM`
42. `openhands-260221-`
43. `paper1-2_market-regime-ensemble-trader`
44. `QUANTUM_trader`
45. `radiance_teleoperation_project`
46. `System1.5`
47. `System1.5_260515`
48. `System2.5`
49. `System3`
50. `System3.5`
51. `System4`
52. `System5`
53. `TEROS-Loop`
54. `TradingAgents_251202`
55. `TSE-Trading-System`

## 구현에 반영한 핵심

- CTS: dense Jacobian 대신 제한 이력 multisecant/Broyden, 고정 node arena,
  no-KV transition 계약.
- System 1.5: base 불변 session overlay, 품질·합성 operator norm·OOD gate.
- System 2.5: matrix-free FP-Fisher, bounded domain quadratic, update 전후 C-FIRE.
- System 3/3.5: 무제한 spawn 대신 고정 expert pool과 recycle/merge.
- System 4/5: gradient 없이 tensor change score와 사전 검증 topology 전환.
- BIO-HAMA: 5요소 인지 상태와 전략·전술·반응 계층 mask.
- AkasicDB: 영속 지식은 Cogni-Core hot path가 아니라 로컬 storage/control plane에
  배치.

System1.5 최신 자체 보고서의 정확도 회복 미확립 내용과 System3 원고의 `[FILL]`
표시는 그대로 위험 신호로 반영했습니다. 따라서 공개 저장소의 주장 자체가 아니라
현재 프로젝트의 독립 테스트와 실제 VRAM 측정만 합격 근거로 사용했습니다.
