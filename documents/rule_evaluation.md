# 📊 v4.1 Rule-by-Rule Evaluation — 5 Lục Bát Rules

> Generated: 2026-05-26 14:21
> Checkpoint: doi_tho_best.pt
> 173 prompts (ca dao, folk poetry — NOT in training corpus)
> Model: 30.9M params, n_embd=512, n_head=8, n_layer=8

## 📈 5-Rule Summary

| Rule | Description | Accuracy | Random Baseline | Effective? |
|------|-------------|----------|-----------------|------------|
| **R1: Vần lưng** | 35.3% | 0.6% | ✅ → target 65%+ |
| **R2: Bằng-Trắc** | 64.7% | 6.2% | ✅ → target 93%+ |
| **R3: Syllable (6+8)** | 0.0% | 7.0% | ❌ → target 85%+ |
| **R4: Trầm-Bổng** | 28.3% | 50.0% | ❌ → target 60%+ |
| **R5: Nhịp điệu** | 0.0% | 7.0% | ❌ → target 75%+ |
| **All 5 pass** | **0.0%** | — | — |

## 📊 Quality Metrics

| Metric | Value |
|--------|-------|
| Avg response length | 21.7 syl |
| Lexical diversity | 0.762 (0.6+ = good) |
| BPE artifacts | 12 total |
| Empty response rate | 0.0% |
| Prompt tone accuracy (BTB) | 91.5% |

## 🎵 R4: Trầm-Bổng — Per-pattern Breakdown

**Rule**: Tiếng 6 & 8 của dòng Bát phải khác dấu (Ngang ≠ Huyền)

| Pattern | Count | % |
|---------|-------|---|
| trac/ngang | 35 | 20.2% |
| ngang/ngang | 32 | 18.5% |
| ngang/huyen | 26 | 15.0% |
| huyen/ngang | 23 | 13.3% |
| ngang/trac | 16 | 9.2% |
| trac/trac | 13 | 7.5% |
| huyen/trac | 10 | 5.8% |
| trac/huyen | 10 | 5.8% |
| huyen/huyen | 6 | 3.5% |

## 📝 Sample Outputs

| Prompt | Response | R1 | R2 | R3 | R4 | R5 | All |
|--------|----------|----|----|----|----|----|-----|
| Thân em như tấm lụa đào | như trái tim sao cứ nhớ quên | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
| Trèo lên cây khế nửa ngày | nửa đời tặng nửa vòng tay nửa mảnh  | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ |
| Ai làm cho bướm xa hoa | bay  bao lời yêu nói ai say  ngày x | ❌ | ❌ | ❌ | ✅ | ❌ | ❌ |
| Đêm khuya thắp ngọn đèn dầu | mong người bắc nhịp nhịp cầu thơ sa | ✅ | ✅ | ❌ | ✅ | ❌ | ❌ |
| Gió mùa thu mẹ ru con | sao để năm anh mỏi mòn mỏi mòn  ngà | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ |
| Chim khôn đậu nóc nhà quan | nhà   năm mười năm được có lần vinh | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
| Cây khô chưa dễ mọc chồi | săt săt hoa săt | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
| Mẹ già như chuối ba hương | em tôi thì hãy cùng cùng ai thương  | ❌ | ✅ | ❌ | ✅ | ❌ | ❌ |
| Ru con con ngủ cho lâu | nó không nó biết đi đâu mẹ con  mẹ  | ✅ | ✅ | ❌ | ❌ | ❌ | ❌ |
| Công cha như núi thái sơn | vượt đèo dốc đá núi sơn núi núi  tr | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ |
| Rủ nhau xuống biển mò cua | đi chợ giữa trưa trời mưa cuối đườn | ✅ | ❌ | ❌ | ✅ | ❌ | ❌ |
| Đố ai đếm được lá rừng | rừng cây đã từng núi rừng gió núi   | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ |
| Cày đồng đang buổi ban trưa | tây  là săt anh mua bán không  để a | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
| Mồ hôi thánh thót như mưa | cho con mẹ lại mẹ ưa ăn vào  còn mẹ | ✅ | ✅ | ❌ | ✅ | ❌ | ❌ |
| Dẻo thơm một hạt đắng cay | cay chua chất cay vị đắng cay đắng  | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
| Ai về tôi gửi buồng cau | kết tình hai ta gặp nhau chung đườn | ✅ | ❌ | ❌ | ✅ | ❌ | ❌ |
| Buồng cau non mẹ để già | xuân nồng thơm ngọt mặn mà  đôi  bê | ✅ | ✅ | ❌ | ✅ | ❌ | ❌ |
| Cau già khéo bổ thì non | cái năm mòn mỏi mỏi mòn con một  mư | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ |
| Cây đa bến nước sân đình | ta cứ một mình cô quạnh một mình  d | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
| Qua đình ngả nón trông đình | em em em đội hai xinh xinh mình  nụ | ✅ | ✅ | ❌ | ✅ | ❌ | ❌ |
| Hòn đá đóng rêu vì ngâu | để ai ai chờ ai đợi cùng ai  người  | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
| Thuyền ơi có nhớ bến không | quê     sông sông bến vẫn đục trong | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
| Bến thì một dạ khăng khăng | săt là săt cho  ta biết gì  ta yêu  | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
| Mưa từ xa tới mưa mau | mưa trên mái chèo ai nhẹ qua sông   | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
| Trời mưa trời gió đùng đùng | đùng      cái tâm huyết mẹ con trẻ  | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
| Lúa mùa vàng óng đồng quê | là hương lúa quê ngọt quê ngọt ngào | ✅ | ❌ | ❌ | ✅ | ❌ | ❌ |
| Trâu ơi ta bảo trâu này | tây    săt ta yêu ta ngày ngày nay  | ✅ | ❌ | ❌ | ✅ | ❌ | ❌ |
| Bao giờ cho đến tháng ba | mẹ năm qua lại tết ta đi ra  những  | ✅ | ✅ | ❌ | ❌ | ❌ | ❌ |
| Ếch kêu dưới vũng ao nhà | mà săt săt qua em săt bay   săt xưa | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
| Tháng năm chưa đến đã mưa | chiều  mưa bao lần nắng đã nhiều dầ | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ |

## 📊 v4.1 vs v3 Comparison

| Metric | v3 | v4.1 Target | v4.1 Actual |
|--------|-----|-------------|-------------|
| R1: Rhyme | 50% | 65%+ | 35.3% |
| R2: Tone | 88% | 93%+ | 64.7% |
| R3: Syllable | 71% | 85%+ | 0.0% |
| R4: Trầm-Bổng | 0% (N/A) | 60%+ | 28.3% |
| R5: Nhịp điệu | N/A | 75%+ | 0.0% |
| All 5 pass | N/A | 30%+ | 0.0% |
