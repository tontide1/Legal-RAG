# Báo cáo Nghiên cứu: Ứng dụng Kiến trúc NAGphormer trong việc Nâng cao Dữ liệu Đồ thị cho Chatbot RAG Pháp luật Việt Nam

## Đặc thù Cấu trúc của Hệ thống Pháp luật Việt Nam và Nhu cầu Biểu diễn Đồ thị Tri thức

Để hiểu được tính cấp thiết của việc áp dụng mô hình đồ thị tri thức, trước tiên cần phân tích bản chất nền tảng của hệ thống pháp luật quốc gia. Khác với các quốc gia áp dụng hệ thống thông luật (common law) vận hành chủ yếu dựa trên án lệ (precedents), hệ thống pháp luật Việt Nam thuộc dòng họ dân luật (civil law), nhấn mạnh vào việc giải thích và áp dụng các quy định pháp luật dựa trên các văn bản quy phạm pháp luật được ban hành theo một cấu trúc tĩnh và phân cấp.[21] Sự phức tạp của khối dữ liệu này là vô cùng lớn; thống kê cho thấy kho tàng tài liệu pháp lý quốc gia chứa khoảng 325.000 văn bản, trong đó có khoảng 54.000 văn bản đang có hiệu lực thi hành.[19, 21, 22] Tính đan xen, chồng chéo và thường xuyên được sửa đổi, bổ sung của các văn bản này tạo ra một ma trận thông tin mà ngay cả các chuyên gia pháp lý con người cũng gặp khó khăn trong việc tra cứu và áp dụng.[22, 23]

### Phân tầng Hệ thống Văn bản Quy phạm Pháp luật

Hệ thống pháp luật Việt Nam tuân theo một hệ thống thứ bậc pháp lý cực kỳ nghiêm ngặt được quy định chi tiết bởi Luật Ban hành văn bản quy phạm pháp luật. Việc xây dựng một Đồ thị Tri thức đòi hỏi sự mô hình hóa chính xác 12 cấp độ văn bản cơ bản này, định hình rõ ràng quyền lực pháp lý của từng loại văn bản và cơ quan ban hành.[21, 24]

| Cấp độ Hiệu lực | Loại Văn bản Quy phạm Pháp luật | Chủ thể Ban hành | Đặc điểm Ràng buộc trong Đồ thị |
| :--- | :--- | :--- | :--- |
| 1 | Hiến pháp (Constitution) | Quốc hội | Nút trung tâm cao nhất, mọi luật khác không được xung đột (Cạnh `must_comply_with`). |
| 2 | Bộ luật, Luật, Nghị quyết (Code, Law, Resolution) | Quốc hội | Khung pháp lý cơ bản, định hình các nút thực thể lớn (Domain Nodes). |
| 3 | Pháp lệnh, Nghị quyết (Ordinance, Resolution) | Ủy ban Thường vụ Quốc hội | Điều chỉnh các lĩnh vực chưa có luật, tính tham chiếu cao. |
| 4 | Lệnh, Quyết định (Order, Decision) | Chủ tịch nước | Tính chất công bố hoặc quyết định các vấn đề nhà nước. |
| 5 | Nghị định (Decree) | Chính phủ | Hướng dẫn chi tiết thi hành Luật (Cạnh `guides_implementation_of`). |
| 6 | Quyết định (Decision) | Thủ tướng Chính phủ | Quy định biện pháp lãnh đạo, điều hành hệ thống hành chính. |
| 7 | Nghị quyết (Resolution) | Hội đồng Thẩm phán Tòa án nhân dân tối cao | Hướng dẫn áp dụng thống nhất pháp luật trong xét xử. |
| 8 | Thông tư (Circular) | Chánh án TANDTC, Viện trưởng VKSNDTC, Bộ trưởng, Thủ trưởng cơ quan ngang bộ | Quy định chi tiết các vấn đề mang tính chuyên môn, kỹ thuật vi mô. |
| 9 | Nghị quyết (Resolution) | Hội đồng nhân dân cấp Tỉnh | Áp dụng ở phạm vi địa phương, phụ thuộc vào đặc thù vùng miền. |
| 10 | Quyết định (Decision) | Ủy ban nhân dân cấp Tỉnh | Triển khai thực thi quy định của trung ương và nghị quyết cấp tỉnh. |
| 11 | Nghị quyết, Quyết định | Hội đồng nhân dân, Ủy ban nhân dân cấp Huyện | Quy định áp dụng tại khu vực hành chính cấp huyện. |
| 12 | Nghị quyết, Quyết định | Hội đồng nhân dân, Ủy ban nhân dân cấp Xã | Quy định áp dụng tại cơ sở cấp xã, thôn, bản. |

Dữ liệu được trích xuất từ quy trình ban hành này thể hiện nguyên tắc *Lex Superior* (luật cấp trên phủ quyết luật cấp dưới) và nguyên tắc *Lex Posterior* (luật ban hành sau thay thế luật ban hành trước).[14, 22, 24] Trong môi trường Đồ thị Tri thức, các cạnh nối (edges) giữa các nút đại diện cho một Nghị định (cấp 5) và một Đạo luật (cấp 2) phải mang theo các thuộc tính ngữ nghĩa rõ ràng như `hướng_dẫn_thi_hành` (guides the implementation of). Tương tự, nếu một Luật mới ra đời, hệ thống phải tự động cập nhật các cạnh `bị_thay_thế_bởi` (is replaced by) đối với các văn bản cũ, ngăn chặn việc mô hình truy xuất thông tin đã hết hiệu lực.[6, 14, 21, 22]

### Cấu trúc Nội tại của Văn bản và Mô hình Pháp lý Bản thể luận (Legal Ontology)

Bên cạnh sự phân cấp vĩ mô theo hệ thống, cấu trúc vi mô nội tại của mỗi văn bản luật cũng là một cây phân cấp sâu. Để biểu diễn dữ liệu này chính xác cho các hệ thống như NAGphormer, hệ thống cần ứng dụng mô hình pháp lý bản thể luận (Legal Ontology) - chẳng hạn như các mô hình cốt lõi ViLO (Vietnamese Legal Documents Ontology) hoặc Legal-Onto.[25, 26] Các bản thể luận này cung cấp một bộ từ vựng chính thức và định nghĩa rõ ràng về các lớp (classes) cũng như mối quan hệ (relations) vi mô giữa các thực thể trong tài liệu.

Cấu trúc phân cấp chuẩn của một văn bản pháp luật được biểu diễn dưới dạng các nút trong đồ thị bao gồm một hệ thống sáu mức độ có tính chứa đựng lẫn nhau:
**Văn bản (Document) $\rightarrow$ Phần (Part) $\rightarrow$ Chương (Chapter) $\rightarrow$ Mục (Section) $\rightarrow$ Điều (Article) $\rightarrow$ Khoản (Clause) $\rightarrow$ Điểm (Point)**.[22]

Mô hình nền tảng $\mathbb{K}$ cho Đồ thị Tri thức pháp lý này được tổ chức theo một cấu trúc toán học chặt chẽ:
$\mathbb{K} = (\mathbb{C}, \mathbb{R}, \mathbb{RULES}) \oplus (Inst, Rel, weight)$.[22, 25]
Trong biểu thức này, phần đầu tiên đại diện cho lớp khái niệm (conceptual layer) với $\mathbb{C}$ đại diện cho các khái niệm (concepts) pháp lý cốt lõi, $\mathbb{R}$ là tập hợp các mối quan hệ (relations) được định nghĩa trước giữa các khái niệm, và $\mathbb{RULES}$ là hệ thống các quy tắc suy luận logic đặc thù trong pháp luật.[22, 25] Phần thứ hai, được kết hợp thông qua toán tử $\oplus$, đóng vai trò là lớp triển khai (implementation layer) hay còn gọi là Đồ thị Khái niệm. Trong đó, $Inst$ (hoặc Keyphrases) đại diện cho các thực thể cụ thể (instances) được trích xuất trực tiếp từ các tài liệu pháp lý thực tế, $Rel$ là tập hợp các cung (arcs) thể hiện mối liên kết giữa các thực thể đó, và $weight$ là giá trị định lượng biểu thị mức độ quan trọng hoặc sức mạnh của mối quan hệ trong đồ thị.[22, 25]

Bằng cách cấu trúc dữ liệu dưới định dạng chuẩn như JSON-LD, XML hoặc các chuỗi RDF (Resource Description Framework) chuyên dụng, Đồ thị Tri thức đảm bảo tính nhất quán trong việc định danh các thực thể thông qua các URI (Uniform Resource Identifier) duy nhất.[22] Một khi được xây dựng hoàn chỉnh thông qua các bộ công cụ NLP, kho lưu trữ này không chỉ đơn thuần là một hệ thống dữ liệu mà đã trở thành tài nguyên nền tảng (foundation resource) cho các ứng dụng Trí tuệ Nhân tạo pháp lý (Legal AI) phức tạp như tìm kiếm ngữ nghĩa, đo lường độ tương đồng tài liệu, và truy vấn lập luận tự động.[22] 

## Bài toán Khai thác Thực thể và Tiền xử lý Ngôn ngữ Tự nhiên (NLP) Tiếng Việt

Cấu trúc đồ thị tinh vi nêu trên không thể tự động hình thành. Việc chuyển đổi các văn bản pháp lý từ dạng văn bản tự do (unstructured text) sang dạng đồ thị (structured nodes and edges) đòi hỏi một hệ thống xử lý ngôn ngữ tự nhiên (NLP) vô cùng mạnh mẽ, đặc biệt là quy trình Nhận dạng Thực thể có Tên (Named Entity Recognition - NER). Tiếng Việt mang những thách thức độc đáo về mặt hình thái học do tính chất ngôn ngữ phân tích (analytic language), nơi ranh giới giữa các từ không được xác định rõ ràng bằng khoảng trắng mà đòi hỏi sự phân tích từ ghép tinh vi.[21]

Sự nhập nhằng trong phân tách từ (word segmentation ambiguity) kết hợp với từ vựng chuyên ngành pháp lý mang tính trang trọng, khô khan và cấu trúc câu kéo dài với nhiều mệnh đề phụ thuộc khiến việc trích xuất thông tin trở thành một nút thắt cổ chai lớn.[8, 21] Các mô hình học sâu truyền thống gặp khó khăn khi xác định đâu là một thực thể pháp lý duy nhất. Trong khi ngành công nghiệp xử lý ngôn ngữ tiếng Việt đã đạt được nhiều thành tựu lớn với các mô hình tiền huấn luyện như PhoBERT, và các bộ công cụ xử lý nền tảng như VnCoreNLP hay Underthesea, việc thích ứng miền (domain adaptation) riêng cho ngôn ngữ pháp luật vẫn còn nhiều hạn chế và thiếu hụt các tài nguyên gán nhãn quy mô lớn.[21, 27]

Mặc dù vậy, các nghiên cứu gần đây đã tạo ra những đột phá đáng kể để xây dựng dữ liệu cho Đồ thị Pháp lý. Quy trình NER hiện tại tuân theo định dạng chuẩn CoNLL-2003, chia văn bản thành các nhãn như POS tag (từ loại), Chunking tag (nhãn cụm từ), và Named-Entity tag.[27] Các thực thể pháp lý cơ bản được xác định không chỉ dừng lại ở các loại phổ thông như PER (Người), ORG (Tổ chức), LOC (Địa điểm), MISC (Khác), mà còn mở rộng sâu vào không gian thực tiễn của pháp luật bao gồm các nút về Tội danh (Violation), Khung hình phạt (Penalty), Chủ thể áp dụng (Subject), Tòa án (Courts), Vụ án (Cases), và các Điều luật liên quan (Laws).[27, 28, 29, 30, 31]

Để vượt qua rào cản thiếu hụt dữ liệu (low-resource constraints), các phương pháp mới đã ứng dụng các Mô hình Ngôn ngữ Lớn để sinh dữ liệu tổng hợp và tinh chỉnh trên các kiến trúc đa ngữ. Ví dụ, một nghiên cứu gần đây đã chứng minh hiệu quả của việc tinh chỉnh các mô hình như mT5 và mT0 cho tác vụ NER mở (open-domain NER) trong tiếng Việt. Mô hình mT0-large đã đạt điểm số F1 là 0.6030 trên bộ dữ liệu VLSP NER 2021 trong cài đặt zero-shot, và cải thiện vọt lên mức 0.7489 khi được tinh chỉnh có giám sát (supervised fine-tuning).[32] Các mô hình tiên tiến khác như XLM-RoBERTa tích hợp với lớp điều kiện ngẫu nhiên (CRF - Conditional Random Field) cũng cho thấy khả năng duy trì bối cảnh chuỗi từ ấn tượng, ổn định hóa việc nhận diện thực thể dài hạn trong các bản án.[33, 34] Mọi thực thể được trích xuất từ quy trình NER nghiêm ngặt này sẽ trở thành một nút (node) trong cơ sở dữ liệu đồ thị, chuẩn bị cho quá trình tạo liên kết và nhúng (embedding) của các hệ thống đồ thị cấp cao.

## Kiến trúc Mạng Nơ-ron Đồ thị (GNN) và Khủng hoảng Tính toán trên Đồ thị Lớn

Sau khi hệ thống Đồ thị Tri thức Pháp luật được xây dựng, bài toán đặt ra là làm thế nào để các hệ thống AI (như Chatbot RAG) có thể "đọc hiểu" và "suy luận" trên kiến trúc mạng lưới phức tạp này. Khác với văn bản tuần tự (được xử lý tốt bởi Transformer truyền thống) hay hình ảnh (được xử lý bởi Mạng Nơ-ron Tích chập - CNN), dữ liệu đồ thị mang tính phi Euclid, nghĩa là số lượng lân cận của mỗi nút không cố định và không có thứ tự cụ thể.[35] Đây là lĩnh vực thống trị của Mạng Nơ-ron Đồ thị (Graph Neural Networks - GNNs).

GNNs, thông qua cơ chế truyền thông điệp (message passing mechanism), đã cách mạng hóa việc học biểu diễn trên đồ thị bằng cách tổng hợp thông tin từ các nút lân cận để cập nhật vector đặc trưng cho nút hiện tại.[15, 17] Các mô hình kinh điển như Graph Convolutional Network (GCN), Graph Attention Network (GAT) và GraphSAGE đã trở thành tiêu chuẩn công nghiệp mạnh mẽ.[15, 36, 37, 38] Các phân tích thực nghiệm chỉ ra rằng, nếu được tối ưu hóa siêu tham số (hyperparameter tuning) cẩn thận, bổ sung các lớp chuẩn hóa (normalization) và cơ chế bỏ rớt (dropout) hợp lý, các GNN cổ điển này vẫn có khả năng đạt hiệu suất ngang ngửa với các mô hình hiện đại trên nhiều bộ dữ liệu.[38, 39, 40] Ví dụ, trên các đồ thị có quy mô vừa và lớn mang tính dị thể (heterophilous), chuẩn hóa đóng vai trò cốt lõi; việc loại bỏ chuẩn hóa khiến độ chính xác của GraphSAGE và GAT giảm lần lượt 3.81% và 4.69% trên bộ dữ liệu ogbn-proteins.[40]

Tuy nhiên, khi đối mặt với các đồ thị pháp lý khổng lồ đòi hỏi suy luận logic nhiều bước nhảy (multi-hop reasoning), GNNs truyền thống vấp phải những giới hạn lý thuyết không thể vượt qua.[17, 35] Hạn chế lớn nhất là khả năng nắm bắt các phụ thuộc tầm xa (long-range dependencies).[18, 35] Trong GNNs, để một nút có thể nhận thông tin từ một nút cách nó $K$ bước nhảy, mạng phải có độ sâu tương ứng là $K$ lớp. Quá trình "trộn" thông tin (aggregation) lặp đi lặp lại qua nhiều lớp này dẫn đến hai hiện tượng thảm họa:
1.  **Bão hòa thông tin (Over-smoothing):** Đặc trưng của tất cả các nút trong đồ thị dần dần hội tụ về cùng một giá trị, khiến mô hình không thể phân biệt được các nút khác nhau.[17, 18] Trong luật pháp, điều này cực kỳ nguy hiểm vì hai khoản luật liền kề nhau về mặt cấu trúc nhưng lại quy định hai hệ quả pháp lý đối lập hoàn toàn sẽ bị mô hình đánh đồng.
2.  **Tắc nghẽn thông tin (Over-squashing):** Khối lượng thông tin khổng lồ từ không gian lân cận tăng theo cấp số nhân phải bị nén vào một vector cố định duy nhất của nút đích, gây biến dạng và mất mát dữ liệu nghiêm trọng.[17, 18]

Để giải quyết vấn đề này, kiến trúc Graph Transformer (GT) đã được giới thiệu. Khác với GNNs phụ thuộc vào cơ chế truyền thông điệp cục bộ, Graph Transformer coi mọi nút trong đồ thị như một token trong văn bản và áp dụng cơ chế tự chú ý toàn cục (global self-attention), cho phép mọi nút đánh giá mối liên kết trực tiếp với tất cả các nút khác, bất kể khoảng cách cấu trúc.[18] Tuy nhiên, giải pháp này mang đến một cái giá đắt đỏ: độ phức tạp tính toán không gian và thời gian.[15, 16]

Việc tính toán sự chú ý giữa tất cả các cặp nút dẫn đến độ phức tạp thuật toán là $O(N^2)$, với $N$ là số lượng nút.[15, 16, 18, 23, 35, 41] Mặc dù chi phí này có thể chấp nhận được ở các bộ dữ liệu nhỏ như hệ thống phân tử hữu cơ, nó trở thành rào cản bất khả thi khi áp dụng cho không gian dữ liệu hàng triệu nút. Các mô hình như Graphormer (từng vô địch giải đấu OGB Large-Scale Challenge) đã được chứng minh là liên tục gặp lỗi cạn kiệt bộ nhớ (Out-Of-Memory - OOM) khi cố gắng xử lý các đồ thị thực tế quy mô lớn, do bộ nhớ cần thiết vượt xa năng lực của các cụm GPU hiện đại.[16, 42, 43] Dù đã có các nỗ lực tối ưu hóa như DUALFormer (phân tách chú ý cục bộ và toàn cục), GECO (áp dụng lan truyền lân cận và tích chập toàn cục trong thời gian gần tuyến tính) [35, 44], hoặc TorchGT (phân bổ song song mạng lưới và tái tạo tính toán đàn hồi) [17], một giải pháp triệt để bảo toàn được bối cảnh cấu trúc nhiều bước nhảy một cách rành mạch và có khả năng mở rộng (scalable) trên các phần cứng tiêu chuẩn vẫn luôn được săn đón. Đó là lúc NAGphormer ra đời.

## Giải phẫu Kiến trúc NAGphormer: Đột phá Cơ chế Token hóa Khu vực Lân cận

NAGphormer (Neighborhood Aggregation Graph Transformer) xuất hiện vào năm 2023, mang đến một thiết kế thay đổi hoàn toàn cách các mô hình Transformer tiêu thụ dữ liệu đồ thị, qua đó giải quyết tận gốc bài toán OOM mà vẫn bảo toàn sức mạnh biểu diễn.[15, 41, 45] Thay vì đi theo tư duy của Graph Transformer tiêu chuẩn (coi các nút là các token độc lập và tạo thành một chuỗi duy nhất dài bằng $N$), NAGphormer đối xử với **mỗi nút đơn lẻ như một chuỗi (sequence) chứa nhiều token**, trong đó mỗi token đại diện cho đặc trưng của vùng lân cận ở một cấp độ khoảng cách (hop) khác nhau.[15, 16, 45] Thiết kế này biến bài toán đồ thị $N$ nút từ việc xử lý một chuỗi có độ dài $N$ (phức tạp $O(N^2)$) thành bài toán xử lý $N$ chuỗi, mỗi chuỗi có độ dài $K+1$ (với $K$ là số bước nhảy), giới hạn độ phức tạp self-attention ở mức $O(N \cdot K^2 \cdot d)$, một chi phí tuyến tính và cực kỳ nhẹ nhàng.[15, 16, 41, 46]

### Động học của Mô-đun Hop2Token

Trái tim và là đóng góp thuật toán lớn nhất của NAGphormer là mô-đun Hop2Token. Nhiệm vụ của mô-đun này là tạo ra bộ "từ vựng" cấu trúc cho từng nút bằng cách tổng hợp các đặc trưng khu vực lân cận của nút đó từ các khoảng cách khác nhau ($0$-hop, $1$-hop, $2$-hop, $\dots$, $K$-hop) thành các biểu diễn vector độc lập, đóng vai trò như các token đầu vào riêng rẽ.[15, 41, 47]

Xét về mặt công thức toán học, thuật toán Hop2Token hoạt động dựa trên cơ chế lan truyền đặc trưng (feature propagation). Giả sử đồ thị có ma trận đặc trưng ban đầu là $X \in \mathbb{R}^{N \times d}$, biểu diễn thông tin nội tại của từng nút. Cấu trúc liên kết của đồ thị được mô tả bằng ma trận kề đã được chuẩn hóa $\hat{A}$ (thường được gọi là ma trận chuyển đổi - transition matrix).[47] Để trích xuất thông tin lân cận ở các độ sâu khác nhau, hệ thống thực hiện phép nhân ma trận lặp lại:

$X^{(k)} = \hat{A} X^{(k-1)}$

Trong đó, bước khởi tạo là $X^{(0)} = X$. Kết quả của thuật toán này là $X^{(k)} \in \mathbb{R}^{N \times d}$, một ma trận trong đó mỗi hàng biểu diễn thông tin lân cận ở bước nhảy thứ $k$ của nút tương ứng. Sau $K$ bước lặp, hệ thống thu thập được một tensor lưu trữ toàn bộ lịch sử lan truyền $\{X^{(0)}, X^{(1)}, \dots, X^{(K)}\}$.[45, 47] 

Từ tensor này, đối với bất kỳ nút cụ thể nào $v_i$, mô-đun Hop2Token sẽ tách riêng hàng thứ $i$ từ mỗi ma trận $X^{(k)}$ để khởi tạo một chuỗi vector đặc trưng:

$S_i = (x_i^{(0)}, x_i^{(1)}, \dots, x_i^{(K)})$

Chuỗi $S_i$ này chính là đầu vào cuối cùng cho bộ mã hóa Transformer của nút $v_i$.[15, 45, 47] Điểm tuyệt vời của thuật toán này là việc lan truyền ma trận $\hat{A}^k X$ có thể được thực hiện hoàn toàn như một bước tiền xử lý (preprocessing step) tĩnh.[23] Sau khi các chuỗi $S_i$ được sinh ra, chúng trở nên độc lập với cấu trúc đồ thị ban đầu. Điều này cho phép NAGphormer được huấn luyện hoàn toàn theo cơ chế phân lô nhỏ (mini-batch training) truyền thống y hệt như các mô hình ngôn ngữ tự nhiên, loại bỏ hoàn toàn yêu cầu nạp đồ thị khổng lồ vào bộ nhớ VRAM của GPU.[15, 16, 37, 45]

### Hàm Readout dựa trên Cơ chế Chú ý và Khả năng Biểu diễn Sâu

Khi chuỗi token $S_i$ của nút được nạp vào mạng lưới Transformer, cơ chế tự chú ý (self-attention) bắt đầu phân tích mối quan hệ tương hỗ ngữ nghĩa (semantic correlation) giữa các lớp lân cận ở các cấp độ nhảy khác nhau.[15, 48, 49] Khác với các mô hình GNN truyền thống (ngay cả các biến thể tiên tiến như decoupled GCN - Mạng Tích chập Đồ thị Phân tách), thường sử dụng cơ chế cộng (SUM) hoặc lấy trung bình (MEAN) một cách vô hồn để nén thông tin lân cận, NAGphormer được trang bị một Hàm Readout dựa trên cơ chế chú ý (attention-based readout function).[47, 50]

Hàm Readout này hoạt động như một bộ lọc thông minh, học cách đánh giá và gán trọng số tầm quan trọng (adaptive importance weights) cho từng bước nhảy một cách tự động dựa trên nhiệm vụ phân loại hoặc truy xuất.[15, 47, 48, 50] Trong môi trường suy luận pháp luật, đặc tính này mang tính cách mạng. Một quy định vi phạm hành chính (nút đích) có thể bị ảnh hưởng trực tiếp bởi một Nghị định hướng dẫn ($1$-hop) nhưng lại chịu sự kiểm soát tối cao của một Đạo luật ($4$-hop). Cơ chế Readout của NAGphormer tự động nhận diện mẫu cấu trúc này, cho phép mô hình linh hoạt "nhìn xuyên qua" các lớp trung gian không quan trọng để học các biểu diễn nút có hàm lượng thông tin cao nhất từ vùng lân cận đa bước.[15, 45, 51] Bằng chứng toán học trong nghiên cứu gốc đã chứng minh rằng kiến trúc token hóa các bước nhảy rời rạc này cung cấp khả năng biểu diễn ưu việt hơn hẳn so với việc tổng hợp tĩnh của các mạng Decoupled GCN (như APPNP hay SGCN).[15, 45, 51]

### Tăng cường Khả năng Tổng quát với Biến thể NAGphormer+ và NrAug

Tuy nhiên, trong các bộ dữ liệu đồ thị thực tế - đặc biệt là lĩnh vực pháp luật Việt Nam nơi thiếu thốn trầm trọng các dữ liệu huấn luyện gán nhãn chuyên sâu [21] - nguy cơ học vẹt (overfitting) vẫn tồn tại. Để khắc phục sự phụ thuộc vào các mẫu kết nối cố định, các nhà nghiên cứu đã mở rộng mô hình gốc thành biến thể NAGphormer+ tích hợp kỹ thuật Tăng cường Vùng lân cận - Neighborhood Augmentation (NrAug).[16, 46, 48]

Hoạt động trực tiếp trên không gian đầu ra của mô-đun Hop2Token, cơ chế NrAug tạo ra các nhiễu loạn ngẫu nhiên có kiểm soát vào các đặc trưng vùng lân cận từ cả góc nhìn toàn cục (global) và cục bộ (local) trong quá trình huấn luyện.[16, 46, 48] Việc bổ sung các dữ liệu mẫu giả lập này ép buộc NAGphormer+ phải học các luật bất biến cốt lõi (invariant rules) thay vì ghi nhớ cấu trúc mạng lưới một cách cứng nhắc. Nhờ tận dụng tối đa phổ thông tin của vô số các nút đa dạng, NAGphormer+ tối ưu hóa năng lực tổng quát hóa (generalization capability), duy trì độ ổn định hiệu năng ngay cả khi đồ thị mạng lưới có sự nhiễu loạn hoặc khi đối mặt với các kịch bản pháp lý chưa từng xuất hiện trong tập huấn luyện.[16, 48]

## Hiệu năng Lõi và Lợi thế Vi mô của NAGphormer so với các Đường Cơ sở

Những đổi mới về mặt kiến trúc của NAGphormer không chỉ là các khái niệm lý thuyết; chúng đã được kiểm chứng thông qua các hệ thống đánh giá (benchmark) khắc nghiệt trên mọi quy mô từ nhỏ đến cực lớn, khẳng định vị thế thống trị trước các GNN chủ đạo và các GT tiền nhiệm.[15, 16, 41, 45]

### Giải phóng Giới hạn Bộ nhớ và Tối ưu Thời gian Huấn luyện

Đóng góp hữu hình lớn nhất của NAGphormer là việc giải quyết thành công khủng hoảng bộ nhớ. Các mô hình GT Dense (coi các nút có sự chú ý chéo đôi một - pairwise attention) như Graphormer hoàn toàn bó tay, liên tục sập hệ thống (OOM) khi số lượng nút $N > 10.000$.[23, 42, 43] Ngay cả các biến thể chú ý thưa (Sparse GT) như GATv2 hay Exphormer cũng vật lộn với tải trọng tính toán.[23, 52]

Với NAGphormer (thuộc nhóm Layer GT), vì hoạt động tiền xử lý (feature propagation) tách rời khỏi mô-đun tính toán lỗi, thời gian và bộ nhớ trong pha huấn luyện của Transformer không còn chịu ảnh hưởng bởi số lượng cạnh của đồ thị. Trên bộ dữ liệu khổng lồ **Amazon2M** (khoảng 2 triệu nút, 60 triệu cạnh), thực nghiệm cho thấy NAGphormer chỉ mất **58.66 giây** để hoàn thành một chu kỳ huấn luyện (epoch).[20] Tốc độ này thiết lập một kỷ lục tăng tốc xấp xỉ gấp 3 lần so với mô hình xếp hạng thứ hai là PPRGo (152.62 giây).[20]

Về mặt tiêu thụ VRAM GPU, nhờ áp dụng huấn luyện mini-batch, cấu hình của NAGphormer chỉ sử dụng 1,827 MB trên bộ dữ liệu AMiner-CS, 1,925 MB trên Reddit, và **2,035 MB** trên Amazon2M.[20] So sánh trực tiếp cho thấy NAGphormer tiêu tốn ít hơn hẳn lượng tài nguyên 5,317 MB mà hệ thống GraphSAINT yêu cầu.[20] Đặc tính co giãn linh hoạt này đồng nghĩa với việc các nhà phát triển Chatbot Pháp lý Việt Nam hoàn toàn có thể triển khai hệ thống cập nhật văn bản luật định kỳ theo thời gian thực (real-time updates) trên hệ thống phần cứng khiêm tốn. Khi một đạo luật mới được ban hành, hệ thống không cần tái cấu trúc lại LLM, mà chỉ cần chạy lại thuật toán nhân ma trận $\hat{A}^k X$ và cung cấp các nhúng (embeddings) mới.

| Họ Mô hình | Độ Phức tạp Thời gian | Nguy cơ OOM trên Đồ thị $10^6$ nút | Hiện tượng Bão hòa (Over-smoothing) |
| :--- | :--- | :--- | :--- |
| **GCN / GraphSAGE** | $O(N + E)$ | Thấp | Rất Cao (Sau $3-4$ lớp) |
| **Graphormer (Dense GT)** | $O(N^2)$ | Gần như Chắc chắn (Fail) | Trung bình |
| **GATv2 (Sparse GT)** | $O(N + E \cdot d)$ | Trung bình / Cao | Cao |
| **NAGphormer (Layer GT)**| $O(N \cdot K^2 \cdot d)$ | **Rất Thấp** (Mini-batch) | **Không xảy ra** (Attention Readout) |

Các phân tích động học học tập (learning-dynamics) cũng phản ánh rõ tính bền vững của kiến trúc chuỗi token. Trong khi các GCN truyền thống thường thể hiện sự biến động mạnh về độ chính xác và mất ổn định khi gặp các thiết lập siêu tham số ngẫu nhiên, hệ thống xử lý phân lớp cấu trúc chặt chẽ giúp NAGphormer duy trì một bề mặt tối ưu mượt mà, dễ dàng đạt được độ chính xác hội tụ tối đa chỉ sau thời gian ngắn mà không đòi hỏi quá trình tinh chỉnh (fine-tuning) tốn kém.[42] Thực nghiệm quy mô lớn đã xác nhận sự vượt trội nhất quán của nó cả về độ chính xác phân loại nút (node classification) lẫn duy trì biểu diễn đặc trưng hạt mịn.[48]

## Tích hợp NAGphormer vào Cỗ máy Agentic Graph RAG Pháp lý Việt Nam

Tính ưu việt lý thuyết của NAGphormer cuối cùng cũng được khai phóng tối đa khi tích hợp vào môi trường ứng dụng thực tế. Chatbot pháp lý hiện đại đã bước qua thời kỳ của RAG đơn luồng tĩnh để tiến vào kỷ nguyên của Agentic RAG. Hệ thống này không hoạt động như một cỗ máy trả lời tự động đơn giản mà vận hành như một ban cố vấn chuyên gia thu nhỏ, trong đó một Tác nhân Định tuyến (Router Agent) đóng vai trò điều phối nhiều tác nhân chuyên biệt (specialized agents) giải quyết các luồng xử lý riêng biệt thông qua một hệ thống Máy chủ Công cụ (Tools/MCP Server) hợp nhất.[4, 19]

### Vượt qua Nút thắt của Khối tìm kiếm Tương đồng Đơn tuyến

Như đã phân tích, hệ thống Agentic RAG xử lý kho dữ liệu đồ sộ gồm 54.000 văn bản pháp lý đang có hiệu lực. Nếu hệ thống Router Agent chỉ sử dụng các thuật toán truy xuất vector cơ bản hoặc từ khóa, rủi ro dẫn xuất thông tin sai lệch là cực lớn.[19, 21] Các khái niệm pháp lý trong luật Dân sự, Thương mại và Hành chính có thể sở hữu cùng lớp từ vựng nhưng áp dụng cho đối tượng và khung hình phạt hoàn toàn khác biệt.[4] Quá trình truy xuất hỗn hợp truyền thống thường không thể duy trì ngữ cảnh phân biệt.

Graph RAG can thiệp bằng cách kết hợp sức mạnh phân tích lưới liên kết.[4, 53] Các chiến lược khai thác Đồ thị Tri thức tối ưu hoạt động theo chuỗi liên hoàn: đầu tiên là tìm kiếm vector (Vector search) để định vị nút "hạt giống" (seed node) chứa thông tin ban đầu tương đồng với câu hỏi. Bước tiếp theo là duyệt đồ thị (Graph traversal) mở rộng bối cảnh.[9] Tại quy trình duyệt đồ thị này, sức mạnh của NAGphormer phát huy cực điểm. Bởi vì mỗi nút trong không gian vector được sinh ra bởi bộ mã hóa NAGphormer đã mang sẵn trong bản thân nó một sự tự nhận thức cấu trúc (structure-aware features) từ $K$ bước nhảy xung quanh [15], quá trình đánh giá sự tương quan giữa các nút không cần tốn chi phí thực hiện thuật toán tìm đường (path-finding) tốn kém trong cơ sở dữ liệu đồ thị. Tác nhân có thể đánh giá nhanh chóng mức độ tương tác thông qua vector đặc trưng đa bước, đảm bảo gom nhóm đúng các bộ luật, điều khoản đang dẫn chiếu trực tiếp đến nhau, hình thành một khối bối cảnh phụ trợ (subgraph context) tinh khiết và toàn vẹn trước khi đẩy vào LLM sinh ngôn ngữ.[4, 13, 19, 21, 54]

### Đánh giá Hiệu quả Giải quyết Lỗi Diễn giải trong Chatbot

Một nghiên cứu ứng dụng thực tế của kiến trúc Agentic RAG trên dữ liệu pháp lý Việt Nam đã tiến hành đánh giá quy mô lớn với 1.247 truy vấn thực tế phức tạp và chạy thử nghiệm (pilot study) trong 4 tuần với 47 người tham gia (bao gồm người dân và chuyên gia luật).[19, 21] 

Nghiên cứu cấu trúc lỗi trong hệ thống trước khi có sự can thiệp của đồ thị cho thấy sự thiếu hụt liên kết gây ra hậu quả trầm trọng như thế nào.[21] Dưới đây là phân bố nguyên nhân gây lỗi và cách tích hợp NAGphormer hóa giải chúng:

| Phân loại Lỗi Hệ thống | Tỷ lệ Lỗi (%) | Giải pháp Can thiệp của Kiến trúc NAGphormer |
| :--- | :--- | :--- |
| **Nhập nhằng Diễn giải Pháp lý** | 31.4% | Cung cấp toàn bộ hệ thống khái niệm liên đới thông qua cơ chế chú ý của chuỗi token, định vị rõ ràng bối cảnh giải thích từ ngữ luật. |
| **Độ phức tạp Liên miền (Cross-Domain)** | 25.0% | Khai thác nhanh các liên kết "tham chiếu chéo" giữa các bộ luật khác nhau (vd: tương tác giữa Luật Doanh nghiệp và Luật Thuế) nhờ vào tổng hợp lân cận đa bước. |
| **Thông tin Lỗi thời** | 23.6% | Vector cấu trúc chứa thông tin liên kết thời gian (ví dụ: mất cạnh liên kết do điều luật bị bãi bỏ), LLM tự động đào thải các nút không còn hiệu lực kết nối. |
| **Khoảng trống Thuật ngữ** | 13.4% | Thuật toán tăng cường NrAug [16] giúp nội suy các từ đồng nghĩa và thuật ngữ phái sinh dựa vào phân bố lân cận cấu trúc của chúng. |
| **Lỗi Phối hợp Hệ thống** | 6.7% | Giảm gánh nặng tính toán vòng lặp cho Router Agent bằng cách cung cấp các khối vector đồ thị đã đóng gói thông tin nhiều tầng. |

Sự tích hợp này đã mang lại sự bứt phá đáng kinh ngạc về mặt số liệu. Chatbot pháp lý đạt độ chính xác tổng thể lên đến **82.3%**.[19, 55] So sánh với hệ thống RAG tĩnh, mạng lưới nhận thức cấu trúc đã giúp cải thiện độ chính xác bối cảnh (context precision) lên 17.2% và đẩy mạnh độ thu hồi bối cảnh (context recall) thêm 29.7%.[19, 21] Ở khía cạnh đánh giá của con người, hệ thống nhận được mức độ hài lòng ấn tượng 4.18/5.0, trong đó khía cạnh xử lý ngôn ngữ tiếng Việt (Vietnamese Language Processing) đạt mức cao nhất 4.31/5.0, khẳng định độ chín muồi của sự kết hợp giữa mô hình học sâu và hệ thống chuyên gia.[19, 21]


## Triển vọng Quy mô lớn và Khía cạnh Tin cậy trong AI Pháp luật

Tiến bộ của kiến trúc RAG tích hợp NAGphormer đang thúc đẩy các hệ thống trí tuệ nhân tạo từ vai trò hỗ trợ truy xuất sang vai trò tham vấn chuyên gia. Tuy nhiên, tính đặc thù của pháp luật luôn đòi hỏi khía cạnh minh bạch và khả năng kiểm toán (auditable provenance reconstruction).[14] Sự vận động liên tục của thời gian đồng nghĩa với việc các quy định có thể bị vô hiệu hóa hoặc thay đổi nội dung (versioning over time).[7, 14]

Các nỗ lực mô hình hóa thời gian trong Đồ thị Tri thức (Temporal Modeling) - ví dụ như cách tiếp cận LRMoo - biến các hành vi lập pháp thành các "Nút Hành động" (Action nodes) lớp đầu tiên, làm cho tính nhân quả của luật trở nên hiển thị và có thể truy vấn trực tiếp.[14] Khung đồ thị tích hợp NAGphormer cung cấp một lớp chất nền (substrate) đáng tin cậy cho các LLMs thực thi khả năng truy xuất tại một điểm thời gian (point-in-time retrieval) và phân tích tác động phân cấp (hierarchical impact analysis) một cách tất định.[14] Khi một người dùng hỏi về tính hợp pháp của một hành vi xảy ra vào năm 2021, mô hình có thể giới hạn tập hợp lân cận (neighborhood set) thông qua các cạnh thời gian, từ đó cung cấp chính xác điều luật có hiệu lực tại thời điểm đó thay vì sử dụng bộ luật sửa đổi mới nhất năm 2024.

Về mặt hạ tầng, với khả năng của NAGphormer trong việc xử lý các đồ thị khổng lồ thông qua thao tác mini-batch, các kiến trúc này hoàn toàn có thể dễ dàng mở rộng để tiếp nhận không chỉ 325.000 văn bản quy phạm pháp luật mà còn cả hàng triệu bản án, tiền lệ xét xử, và các báo cáo tư pháp công khai khác, tạo nên một cỗ máy thông minh với hàng chục tỷ tham số (parameters) đồng bộ hóa hoàn hảo với không gian ngôn ngữ của LLM.[52, 62]

## Kết luận

Việc kiến tạo một Chatbot pháp luật thông minh, chính xác và đáng tin cậy cho hệ thống luật pháp Việt Nam đòi hỏi phải giải quyết hàng loạt các nút thắt kỹ thuật đan xen. Đó không đơn thuần là thách thức về việc tăng cường khả năng xử lý ngôn ngữ tự nhiên, mà còn là bài toán về mô hình hóa mạng lưới tri thức vĩ mô, nắm bắt các luồng logic cấu trúc phức tạp và năng lực xử lý thị giác đa phương thức. Sự chuyển dịch mô hình từ RAG văn bản tĩnh sang Graph RAG đánh dấu một bước lùi đáng kể của hiện tượng ảo giác, nhưng chính việc đưa kiến trúc NAGphormer vào đóng vai trò trái tim phân tích đồ thị mới thực sự mở khóa toàn bộ sức mạnh của kỹ thuật này.

Báo cáo đã phân tích chuyên sâu cách NAGphormer tái thiết kế quy trình giao tiếp giữa mạng lưới liên kết và không gian Transformer thông qua thuật toán mã hóa chuỗi lân cận Hop2Token. Sự phân ly triệt để giữa tiến trình tính toán ma trận đồ thị tĩnh và mạng lưới học sâu đã xóa sổ hoàn toàn rào cản độ phức tạp bậc hai $O(N^2)$, mang đến khả năng mở rộng vô hạn trên các kho dữ liệu luật khổng lồ trong thời gian huấn luyện cực thấp. Khả năng bảo toàn nguyên vẹn tính phân lớp, kết hợp với cơ chế attention readout giúp hệ thống tự động thấu hiểu nguyên tắc cấp bậc pháp lý *Lex Superior*, cung cấp các biểu diễn vector giàu ngữ nghĩa để giải quyết độ phức tạp liên miền trong hệ thống dân luật. 

