# 📊 v4.1 Rule-by-Rule Evaluation — 5 Lục Bát Rules

> Generated: 2026-05-26 11:30
> Checkpoint: doi_tho_best.pt
> 173 prompts (ca dao, folk poetry — NOT in training corpus)
> Model: 30.9M params, n_embd=512, n_head=8, n_layer=8

## 📈 5-Rule Summary

| Rule | Description | Accuracy | Random Baseline | Effective? |
|------|-------------|----------|-----------------|------------|
| **R1: Vần lưng** | 5.8% | 0.6% | ✅ → target 65%+ |
| **R2: Bằng-Trắc** | 55.8% | 6.2% | ✅ → target 93%+ |
| **R3: Syllable (6+8)** | 0.0% | 7.0% | ❌ → target 85%+ |
| **R4: Trầm-Bổng** | 10.4% | 50.0% | ❌ → target 60%+ |
| **R5: Nhịp điệu** | 0.0% | 7.0% | ❌ → target 75%+ |
| **All 5 pass** | **0.0%** | — | — |

## 📊 Quality Metrics

| Metric | Value |
|--------|-------|
| Avg response length | 23.1 syl |
| Lexical diversity | 0.749 (0.6+ = good) |
| BPE artifacts | 9 total |
| Empty response rate | 0.0% |
| Prompt tone accuracy (BTB) | 91.5% |

## 🎵 R4: Trầm-Bổng — Per-pattern Breakdown

**Rule**: Tiếng 6 & 8 của dòng Bát phải khác dấu (Ngang ≠ Huyền)

| Pattern | Count | % |
|---------|-------|---|
| trac/trac | 43 | 24.9% |
| trac/ngang | 40 | 23.1% |
| ngang/trac | 29 | 16.8% |
| ngang/ngang | 20 | 11.6% |
| huyen/trac | 13 | 7.5% |
| huyen/ngang | 9 | 5.2% |
| ngang/huyen | 9 | 5.2% |
| trac/huyen | 7 | 4.0% |
| huyen/huyen | 3 | 1.7% |

## 📝 Sample Outputs

| Prompt | Response | R1 | R2 | R3 | R4 | R5 | All |
|--------|----------|----|----|----|----|----|-----|
| Thân em như tấm lụa đào | tôi côại mộnguốnượng núi chắc đùa t | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
| Trèo lên cây khế nửa ngày | tôi sợi cất trầu thamòi bay trái tô | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ |
| Ai làm cho bướm xa hoa | tôi nuôi phúc rạo nhè rạo đổ môi mô | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
| Đêm khuya thắp ngọn đèn dầu | tôi giấc rửa xo nghề bạc ao bay bạc | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
| Gió mùa thu mẹ ru con | tôi phận lụt đềm xông mỗi nẻo quá t | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
| Chim khôn đậu nóc nhà quan | tôi chí hơn điềuuốn vấn héo nhè tôi | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
| Cây khô chưa dễ mọc chồi | tôi to thấp dốc xo thấpBTB nónướng  | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
| Mẹ già như chuối ba hương | tôiấn niệmấnếng chí chăm điềuấu tôi | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
| Ru con con ngủ cho lâu | thông tôi lụt sức niệm chăm đô thấp | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
| Công cha như núi thái sơn | tôi tươiạo chăm già hơi liệ mộng đủ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
| Rủ nhau xuống biển mò cua | tôi khách cảnh cảnh nên đành laìu t | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
| Đố ai đếm được lá rừng | sức tôi bay cảnh xơấu mộng già hẹn  | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
| Cày đồng đang buổi ban trưa | tôi khách quá sức điều bạn du la ba | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
| Mồ hôi thánh thót như mưa | tôi khói tri tri hy vết lụt trận mô | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
| Dẻo thơm một hạt đắng cay | tôi hơi chăm chăm dạ chăm nga dang  | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
| Ai về tôi gửi buồng cau | tôi tôiènộn tào lụt niệm chăm đang  | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
| Buồng cau non mẹ để già | BBTBBB tôi ngổn ngổn niệm đơn bay r | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
| Cau già khéo bổ thì non | tôi to niệm gầyấu nên già đô bản tô | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
| Cây đa bến nước sân đình | đình tôi bắc niệm thấp biế niệm già | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
| Qua đình ngả nón trông đình | ép tôi quá đô niệm chí máuép ngơ tô | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
| Hòn đá đóng rêu vì ngâu | tôi ngổn sương ngổn ngổn ngổnổn ngổ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
| Thuyền ơi có nhớ bến không | tôi nghề não lụt tham niệm niệm niệ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
| Bến thì một dạ khăng khăng | tôi thuyền nên bi cảnh bồi thuyền q | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
| Mưa từ xa tới mưa mau | tôi thuyền hay nên danh đúng hơn ch | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
| Trời mưa trời gió đùng đùng | tôi bay quá nênẹniềm tiền rưng tôi  | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
| Lúa mùa vàng óng đồng quê | ủi tôi chăm cách tàoếngủi gia bộ tô | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
| Trâu ơi ta bảo trâu này | tôi thuyền niệm vuốtTTTTTB ngổn giò | ❌ | ✅ | ❌ | ✅ | ❌ | ❌ |
| Bao giờ cho đến tháng ba | đắng tôiBTTTBT món niệm thấpBTB như | ❌ | ✅ | ❌ | ❌ | ❌ | ❌ |
| Ếch kêu dưới vũng ao nhà | tôi thủ bay niệm nế biế biế bộ tôi  | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
| Tháng năm chưa đến đã mưa | tôi nên bạc hyiềm dại dưỡng dại tôi | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |

## 📊 v4.1 vs v3 Comparison

| Metric | v3 | v4.1 Target | v4.1 Actual |
|--------|-----|-------------|-------------|
| R1: Rhyme | 50% | 65%+ | 5.8% |
| R2: Tone | 88% | 93%+ | 55.8% |
| R3: Syllable | 71% | 85%+ | 0.0% |
| R4: Trầm-Bổng | 0% (N/A) | 60%+ | 10.4% |
| R5: Nhịp điệu | N/A | 75%+ | 0.0% |
| All 5 pass | N/A | 30%+ | 0.0% |
