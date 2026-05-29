# Hybrid Traffic-Law Benchmark

## Purpose

This benchmark measures whether `hybrid` outperforms `naive` on synthesis-oriented traffic-law questions.

## Query Groups

1. Scope of application and regulated subjects
2. Conditions and exceptions
3. Responsibilities of subjects or authorities
4. Violations and related sanctions
5. One anchor provision with supporting provisions across nearby articles

## Example Cases

- Query: `Đối tượng nào được phép hoặc không được phép sử dụng xe cứu hộ giao thông đường bộ?`
  - Expected anchor: `Điều 79`
  - Expected focus: `scope`, `conditions`

- Query: `Ai chịu trách nhiệm quản lý nhà nước về hoạt động đường bộ và trách nhiệm đó được phân chia như thế nào?`
  - Expected anchor: `Điều 82`
  - Expected focus: `responsibilities`, `scope`

- Query: `Những hành vi nào bị cấm và có thể dẫn đến xử phạt trong hoạt động đường bộ?`
  - Expected anchor: article varies by corpus coverage
  - Expected focus: `violations`

## Re-Index Requirement

Changing `ENTITY_TYPES` requires rebuilding the graph.

Minimum validation checklist after deployment:

1. Start backend with the updated ontology
2. Use a fresh or cleared LightRAG database state
3. Re-upload `35-2024-qh15.pdf` and `36-2024-qh15.pdf`
4. Re-run the benchmark queries in comparison mode
5. Confirm `hybrid` shows one legal anchor and grounded supporting points
