# 📊 Stage 1 vs Stage 2 — Poetry Evaluation Report

> Generated: 2026-05-20 20:40
> 20 prompts × 3 samples = 60 per model
> Model: 30.9M params, n_embd=512, n_head=8, n_layer=8

## 📈 Summary

| Metric | Stage 1 | Stage 2 | Winner |
|--------|---------|---------|--------|
| Syllable: exact 8 | 25.0% | 8.3% | Stage 1 |
| Syllable: 6-10 range | 78.3% | 76.7% | Stage 1 |
| Avg response length | 7.4 syl | 7.6 syl | Stage 2 (closer) |
| Prompt tone (B-T-B) | 96.7% | 96.7% | Tie |
| Response tone (B-T-B-B) | 74.0% | 76.8% | Stage 2 |
| Rhyme (vần lưng) | 15.0% | 30.0% | Stage 2 |

## 🎭 Per-Prompt Comparison

| Prompt | S1 Syll | S2 Syll | S1 Rhyme | S2 Rhyme | S1 Tone | S2 Tone | Best S1 | Best S2 |
|--------|---------|---------|----------|----------|---------|---------|---------|---------|
| Trăm năm trong cõi người ta | 0% | 0% | 0% | 33% | 89% | 70% | như chưa gặp đã mơ ngày về | có trăng tròn ai biết ta hay |
| Thân em như chẽn lúa đòng | 0% | 0% | 33% | 67% | 100% | 83% | anh ra hái lúa em ăn lúa em về | anh về em lúa trổ đòng vàng hong phơi |
| Gió đưa cành trúc la đà | 0% | 0% | 0% | 0% | 44% | 55% | gió thoảng qua loa kèn nở hoa | đào hồng tươi rói ý xuân |
| Anh đi anh nhớ quê nhà | 0% | 0% | 0% | 67% | 50% | 89% | nhớ sông trà nhớ tàu nhớ bến ba vua | nhớ về quê mẹ nhớ mẹ cha |
| Trèo lên cây bưởi hái hoa | 0% | 0% | 0% | 33% | 58% | 67% | nhài hoa nhài  thơm thoảng mùi hương chanh hương c | lộc vừng đong đưa trước gió đưa la đà |
| Đêm qua em những mơ màng | 0% | 0% | 0% | 33% | 100% | 78% | bao nhiêu ký ức ùa về ta | không để anh chờ đợi em sang |
| Chiều chiều ra đứng ngõ sau | 33% | 0% | 0% | 33% | 82% | 73% | ai không nhìn thấy cảnh vật xôn xao | ai đi mãi trông ngóng trông ngóng chờ ai |
| Ai về ai có nhớ không | 33% | 0% | 0% | 33% | 55% | 70% | nhớ về một thuở tình yêu thuở nào | có một người yêu đang nhớ ai |
| Núi cao chi lắm núi ơi | 0% | 0% | 0% | 33% | 100% | 90% | cùng hòa hợp tác với con người | đất trời mây núi đất sinh sôi |
| Con cò bay lả bay la | 33% | 0% | 33% | 33% | 100% | 100% | mẹ còn cặm cụi đi cày ruộng sâu | trời cao mây trắng vờn mây trắng như pha |
| Qua cầu ngả nón trông cầu | 33% | 0% | 33% | 33% | 70% | 50% | vồng  nắng lên nhuộm đỏ phía lưng trời | xe  chỉ thấy tình hình có thật cao siêu |
| Hỡi cô tát nước bên đàng | 0% | 0% | 0% | 0% | 0% | 0% |  |  |
| Người về em những trông theo | 33% | 0% | 0% | 0% | 70% | 80% | em trao lời hẹn ước gửi vào anh | nắng mưa đi qua phố phường đẹp phố phường |
| Bây giờ mận mới hỏi đào | 100% | 100% | 67% | 100% | 92% | 92% | nguyên đây mai mối mối cho xem nào | hồng nhé các cháu thế nào dám mơ |
| Trúc xinh trúc mọc đầu đình | 67% | 0% | 33% | 33% | 67% | 86% | rêu còn tươi mắt long lanh mắt em | ngả nghiêng trời xanh |
| Đất Quảng Nam chưa mưa đà thấm | 33% | 0% | 0% | 0% | 33% | 50% | nhuần  thuận lại lành mưa thuận nắng mưa | dân  mưa giông bão nắng vẫn là mưa chan hòa |
| Công anh bắt cá dưới ao | 33% | 0% | 0% | 33% | 60% | 78% | cá lớn bé con rồng cháu lạc hồng | anh bắt em xuống ao cá vào |
| Đêm nằm lưng chẳng tới giường | 33% | 0% | 67% | 0% | 80% | 100% | hai tay ôm mộng uyên ương suốt đời | cả nhà đến bữa cơm cùng nhau |
| Lên non mới biết non cao | 33% | 67% | 0% | 0% | 90% | 83% | nên con cháu phải quyết lòng không tha | trời đà sắp đổ mưa nắng lại về |
| Ngó lên nuộc lạt mái nhà | 33% | 0% | 33% | 33% | 67% | 71% | tôi thay gạo trắng những là thường dân | ta thăng long tuyền của việt nam |

## 🧪 Methodology

- 20 Lục Bát prompts × 3 samples = 60 generations per model
- temperature=0.75, top_k=50
- Syllable: response first 8 syllables checked for exact 8-syl count
- Tone: prompt pos 2,4,6 = B-T-B; response pos 2,4,6,8 = B-T-B-B
- Rhyme: response syllable 6 rhyme group matches prompt syllable 6
- Note: model often generates >8 syllables (multi-line). Analysis focuses on first 8.
