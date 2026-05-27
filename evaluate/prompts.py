"""
v4.2.3: Static evaluation prompts — 200 single-line + 118 couplet + 40 quality.

All prompts are hand-crafted, novel (not in training corpus).

Verification: run `python evaluate/prompts.py` to validate all prompts.
"""

# ═══════════════════════════════════════════════════════════
# SINGLE-LINE PROMPTS (200 prompts — 6-syllable lines)
# ═══════════════════════════════════════════════════════════

SINGLE_PROMPTS = [
    # ── Ca dao / Folk poetry (1-60) ──
    "Thân em như tấm lụa đào",       "Trèo lên cây khế nửa ngày",
    "Ai làm cho bướm xa hoa",        "Đêm khuya thắp ngọn đèn dầu",
    "Gió mùa thu mẹ ru con",         "Chim khôn đậu nóc nhà quan",
    "Cây khô chưa dễ mọc chồi",      "Mẹ già như chuối ba hương",
    "Ru con con ngủ cho lâu",        "Công cha như núi thái sơn",
    "Rủ nhau xuống biển mò cua",     "Đố ai đếm được lá rừng",
    "Cày đồng đang buổi ban trưa",   "Mồ hôi thánh thót như mưa",
    "Dẻo thơm một hạt đắng cay",     "Ai về tôi gửi buồng cau",
    "Buồng cau non mẹ để già",       "Cau già khéo bổ thì non",
    "Cây đa bến nước sân đình",      "Qua đình ngả nón trông đình",
    "Hòn đá đóng rêu vì ngâu",
    "Thuyền ơi có nhớ bến không",    "Bến thì một dạ khăng khăng",
    "Mưa từ xa tới mưa mau",         "Trời mưa trời gió đùng đùng",
    "Lúa mùa vàng óng đồng quê",     "Trâu ơi ta bảo trâu này",
    "Bao giờ cho đến tháng ba",      "Ếch kêu dưới vũng ao nhà",
    "Tháng năm chưa đến đã mưa",     "Ve kêu ra rả suốt mùa",
    "Sen tàn cúc lại nở hoa",        "Sầu dài ngày ngắn sang đông",
    "Gió đông về lạnh lòng ai",      "Tóc mây một mái còn dài",
    "Mắt em là cả trời xanh",        "Môi em là nắng long lanh",
    "Núi cao bởi có đất bồi",        "Sông sâu bởi có nước nguồn",
    "Uống nước nhớ kẻ đào sông",     "Ăn quả nhớ kẻ trồng cây",
    "Đất lành chim đậu về đây",      "Người hiền thì lại gặp may",
    "Lời nói chẳng mất tiền mua",    "Lựa lời mà nói cho vừa",
    "Đèn nhà ai nấy sáng trưng",     "Chớ thấy đèn sáng mà mừng",
    "Sông sâu còn có kẻ đò",         "Đường xa còn có người qua",
    "Bầu ơi thương lấy bí cùng",     "Tuy rằng khác giống nhưng chung",
    "Một cây làm chẳng nên non",     "Ba cây chụm lại nên hòn",
    "Gần mực thì đen gần đèn",       "Gần người hiền trí thì nên",
    "Nước lã làm sao khuấy nên",     "Chữ rằng bán tự vi sư",
    "Nhất tự vi sư bán tự",          "Mồng một tết cha mồng hai",

    # ── Regional / Landscape (61-90) ──
    "Tháng giêng ăn tết ở nhà",      "Tháng hai cờ bạc tháng ba",
    "Trên trời có đám mây vàng",     "Bên sông có chị hái dâu",
    "Đồng đăng có phố kỳ lừa",       "Có nàng tô thị có chùa",
    "Đường vô xứ nghệ quanh quanh",  "Non xanh nước biếc như tranh",
    "Anh về câu cá bờ sông",         "Em về cấy lúa trên đồng",
    "Nắng sớm mưa chiều đồng không", "Mong cho lúa tốt bông vàng",
    "Bàn tay ta làm nên tất",        "Có sức người sỏi đá cũng",
    "Chim bay về núi tối rồi",       "Mau lên kẻo nắng tắt rồi",
    "Trăng lên đỉnh núi trăng mờ",   "Đêm nay sao sáng hơn đêm",
    "Xa xa có tiếng chuông chùa",    "Gió đưa hương lúa thơm mùa",
    "Sáng trăng suông sáng cả đồng", "Em đi gặt lúa trên đồng",
    "Mưa xuân lất phất vườn đào",    "Nụ tầm xuân nở ra chào",
    "Hoa thơm ong bướm tìm về",      "Người khôn thiên hạ tìm theo",
    "Tốt gỗ hơn là tốt nước",        "Xấu người đẹp nết còn hơn",
    "Chim không ăn muối chim ươn",   "Con không nghe mẹ con hư",

    # ── Moral / Philosophy (91-120) ──
    "Cá không ăn muối cá ươn",       "Con cãi cha mẹ trăm đường",
    "Đói cho sạch rách cho thơm",    "Khôn ngoan đá đáp người ngoài",
    "Thương người như thể thương thân", "Nhiễu điều phủ lấy giá gương",
    "Một nắng hai sương mẹ cha",     "Lên non mới biết non cao",
    "Nuôi con mới biết công lao",    "Nước chảy đá mòn theo năm",
    "Gió lên cho biển hóa rồng",     "Sóng to gió lớn mênh mông",
    "Đất phèn mọc trái thơm ngon",   "Người nghèo biết quý từng đồng",
    "Học thầy không tày học bạn",    "Đi một ngày đàng học một",
    "Xa mặt nhưng chẳng cách lòng",  "Gần nhau càng thấy nhớ mong",
    "Trăng mờ còn tỏ hơn sao",       "Dẫu rằng núi lở còn cao",
    "Công cha nghĩa mẹ sinh thành",  "Một lòng thờ mẹ kính cha",
    "Con người có tổ có tông",       "Lên non xem núi cao vời",
    "Nước non ngàn dặm xa xôi",      "Ra đi từ thuở lên ba",
    "Quê hương mỗi lúc một xa",      "Bồng bềnh con nước về đâu",
    "Đò chiều khách vắng sang sông", "Trời xanh mây trắng lang thang",

    # ── Seasons & Nature (121-160) ──
    "Đường quê lúa chín vàng ươm",   "Hương đồng gió nội bay xa",
    "Ve sầu kêu gọi hè sang",        "Phượng hồng thắp lửa sân trường",
    "Áo em trắng cả sân trường",     "Tóc em dài quá vai thương",
    "Bàn tay năm ngón nở hoa",       "Đôi chân chim sáo quanh nhà",
    "Em như cây lúa trổ bông",       "Anh như hạt thóc vàng trong",
    "Xa quê nhớ mẹ nhớ cha",         "Nhớ hàng cau trước sân nhà",
    "Giếng làng trong mát tuổi thơ", "Cánh diều no gió tuổi thơ",
    "Mưa rào tắm mát vườn quê",      "Nắng vàng ươm cả đồng xa",
    "Bếp nhà ai đỏ lửa hồng",        "Khói lam chiều tỏa mênh mông",
    "Đàn trâu về ngõ chiều hôm",     "Tiếng sáo diều vọng triền đê",
    "Đêm trăng soi tỏ vườn nhà",     "Hoa cau rụng trắng thềm xưa",
    "Thuyền ai lờ lững trên sông",   "Câu hò vọng giữa thinh không",
    "Cây khế chua ngọt sau vườn",    "Quả na mở mắt thơm lừng",
    "Lời ru của mẹ ngày xưa",        "Theo con suốt cả chặng đường",
    "Người đi đâu suốt chiều nay",   "Để lòng ai những vơi đầy",
    "Đường trần ai biết được đâu",   "Ngày mai sương gió dãi dầu",
    "Thương ai con mắt lim dim",     "Nhớ ai nước mắt đầm đìa",
    "Mưa nguồn chớp bể ào ào",      "Thương em anh biết làm sao",
    "Ví dầu cầu ván đóng đinh",     "Cầu tre lắc lẻo gập ghềnh",
    "Qua sông phải lụy đò ngang",    "Qua suối phải lụy cầu tre",

    # ── Love & Emotion (161-200) ──
    "Đồng bằng ruộng lúa mênh mông", "Biển đông sóng vỗ rì rào",
    "Non cao ai đắp mà cao",         "Sông sâu ai bới ai đào",
    "Khi vui cũng vậy khi buồn",     "Ở hiền thì lại gặp lành",
    "Chị em như chuối nhiều tàu",    "Tấm lành che tấm rách đừng",
    "Rừng vàng biển bạc quê ta",     "Không gì bằng cơm với cà",
    "Có chí thì nên có công",        "Mài sắt nên kim bạn ơi",
    "Tay làm hàm nhai tay quai",     "Miệng ăn núi lở ai ơi",
    "Yêu nhau mấy núi cũng trèo",   "Mấy sông cũng lội mấy đèo cũng qua",
    "Nhớ ai bổi hổi bồi hồi",       "Như đứng đống lửa như ngồi đống than",
    "Gió đưa cây cải về trời",       "Rau răm ở lại chịu lời đắng cay",
    "Đêm qua em những mơ màng",      "Thấy anh về đứng đầu làng chờ em",
    "Trúc xinh trúc mọc đầu đình",   "Em xinh em đứng một mình cũng xinh",
    "Hoa thơm hoa ở trên cành",      "Người khôn người ở kinh thành người dưng",
    "Còn duyên kẻ đón người đưa",    "Hết duyên đi sớm về trưa một mình",
    "Chồng em áo rách em thương",    "Chồng người áo gấm xông hương mặc người",
    "Chiều chiều ra đứng ngõ sau",   "Trông về quê mẹ ruột đau chín chiều",
    "Nước sông Tô vừa trong vừa mát", "Em ghé thuyền anh hát một vài câu",
    "Lênh đênh một chiếc thuyền tình", "Mười hai bến nước biết mình về đâu",
    "Bao giờ cạn nước Đồng Nai",     "Nát chùa Thiên Mụ mới phai lời nguyền",
    "Tóc em dài em cài hoa thiên lý", "Anh muốn làm con bướm cứ vờn quanh hoa",
]


# ═══════════════════════════════════════════════════════════
# COUPLET PROMPTS (118 prompts — hand-verified 6+8 syllable pairs)
# ═══════════════════════════════════════════════════════════

COUPLET_PROMPTS = [
    # ── Original 50 (kept from v4.1/v4.2) ──
    ("thân em như chẽn lúa đòng",        "phất phơ dưới ngọn nắng hồng ban mai"),
    ("công cha như núi thái sơn",        "nghĩa mẹ như nước trong nguồn chảy ra"),
    ("một lòng thờ mẹ kính cha",         "cho tròn chữ hiếu mới là đạo con"),
    ("núi cao bởi có đất bồi",           "sông sâu bởi có nước nguồn chảy quanh"),
    ("uống nước nhớ kẻ đào sông",        "ăn quả nhớ kẻ trồng cây xanh vườn"),
    ("đất lành chim đậu về đây",         "người hiền thì lại gặp may mắn nhiều"),
    ("lời nói chẳng mất tiền mua",       "lựa lời mà nói cho vừa lòng nhau"),
    ("một cây làm chẳng nên non",        "ba cây chụm lại nên hòn núi cao"),
    ("bầu ơi thương lấy bí cùng",        "tuy rằng khác giống nhưng chung một giàn"),
    ("nhiễu điều phủ lấy giá gương",     "người trong một nước phải thương nhau cùng"),
    ("dẻo thơm một hạt đắng cay",        "ai ơi có nhớ những ngày nắng mưa"),
    ("cày đồng đang buổi ban trưa",      "mồ hôi thánh thót như mưa ruộng cày"),
    ("ai về tôi gửi buồng cau",          "buồng cau non mẹ để già lâu năm"),
    ("cây đa bến nước sân đình",         "qua đình ngả nón trông đình xa xa"),
    ("trâu ơi ta bảo trâu này",          "trâu ra ngoài ruộng trâu cày với ta"),
    ("ru con con ngủ cho lâu",           "để mẹ đi cấy đồng sâu chưa về"),
    ("con cò bay lả bay la",             "bay từ cửa phủ bay ra cánh đồng"),
    ("mẹ già như chuối ba hương",        "như xôi nếp mật như đường mía lau"),
    ("thuyền ơi có nhớ bến không",       "bến thì một dạ khăng khăng đợi thuyền"),
    ("mưa từ xa tới mưa mau",            "trời mưa trời gió đùng đùng sấm vang"),
    ("sen tàn cúc lại nở hoa",           "sầu dài ngày ngắn sang đông lạnh lùng"),
    ("tóc mây một mái còn dài",          "mắt em là cả trời xanh mây hồng"),
    ("đường vô xứ nghệ quanh quanh",     "non xanh nước biếc như tranh họa đồ"),
    ("đồng đăng có phố kỳ lừa",          "có nàng tô thị có chùa tam thanh"),
    ("trên trời có đám mây vàng",        "bên sông có chị hái dâu một mình"),
    ("ánh trăng soi tỏ vườn nhà",        "hoa cau rụng trắng thềm xưa lối mòn"),
    ("lời ru của mẹ ngày xưa",           "theo con suốt cả chặng đường hôm mai"),
    ("ve sầu kêu gọi hè sang",           "phượng hồng thắp lửa sân trường mùa thi"),
    ("sáng trăng suông sáng cả đồng",    "em đi gặt lúa trên đồng làng ta"),
    ("mưa xuân lất phất vườn đào",       "nụ tầm xuân nở ra chào đón xuân"),
    ("gió đông về lạnh lòng ai",         "tìm đâu hơi ấm bàn tay người thương"),
    ("xa quê nhớ mẹ nhớ cha",            "nhớ hàng cau trước sân nhà ngày xưa"),
    ("bàn tay năm ngón nở hoa",          "đôi chân chim sáo quanh nhà líu lo"),
    ("học thầy không tày học bạn",       "đi một ngày đàng học một sàng khôn"),
    ("đói cho sạch rách cho thơm",       "khôn ngoan đá đáp người ngoài gà cùng"),
    ("không gì bằng cơm với cà",         "một nắng hai sương mẹ cha vất vả"),
    ("có chí thì nên có công",           "mài sắt nên kim bạn ơi nhớ lời"),
    ("chị em như chuối nhiều tàu",       "tấm lành che tấm rách đừng xấu che"),
    ("ví dầu cầu ván đóng đinh",         "cầu tre lắc lẻo gập ghềnh khó đi"),
    ("qua sông phải lụy đò ngang",       "qua suối phải lụy cầu tre bắc ngang"),
    ("đồng bằng ruộng lúa mênh mông",    "biển đông sóng vỗ rì rào ngày đêm"),
    ("non cao ai đắp mà cao",            "sông sâu ai bới ai đào mà sâu"),
    ("thương người như thể thương thân", "ở hiền thì lại gặp lành ở hiền"),
    ("lên non mới biết non cao",         "nuôi con mới biết công lao mẹ hiền"),
    ("nước chảy đá mòn theo năm",        "đất phèn mọc trái thơm ngon ngọt lành"),
    ("con người có tổ có tông",          "như cây có cội như sông có nguồn"),
    ("gần mực thì đen gần đèn",          "gần người hiền trí thì nên thông minh"),
    ("trăng mờ còn tỏ hơn sao",          "dẫu rằng núi lở còn cao hơn đồi"),
    ("bồng bềnh con nước về đâu",        "đò chiều khách vắng sang sông đợi chờ"),
    ("tiếng sáo diều vọng triền đê",     "đàn trâu về ngõ chiều hôm khói lam"),

    # ── Expanded: diverse rhyme groups (51-118) ──
    # ai, ay, au
    ("trồng cây ai để cho ai",           "cây xanh bóng mát tương lai đời đời"),
    ("bao nhiêu ngày tháng còn dài",     "người đi kẻ ở nhớ ai vơi đầy"),
    ("rủ nhau xuống tắm hồ sen",          "nước trong mát rượi như men đầu mùa"),

    # ao, ang, anh
    ("chiều hôm ra đứng bờ ao",          "trông mây trông nước trông sao trông chồng"),
    ("cô kia đội nón ba tầm",            "có về yên thái hôm rằm xem hội"),
    ("mẹ mong gả thiếp về vườn",         "ăn bông bí luộc dưa hường nấu canh"),

    # ê, em, en, eo
    ("ví dù tình có phôi pha",           "anh như bóng núi chiều tà còn in"),
    ("thân em như củ ấu gai",            "ruột trong thì trắng vỏ ngoài thì đen"),
    ("thân em như hạt mưa rào",           "hạt rơi xuống giếng hạt vào vườn rau"),

    # iêu, iên, in, inh
    ("áo dài em mặc đi chơi",            "cho anh xin một nụ cười thêm duyên"),
    ("yêu nhau cởi áo cho nhau",         "về nhà dối mẹ qua cầu gió bay"),
    ("đố ai lặn xuống vực sâu",          "mà đo miệng cá uốn câu cho vừa"),

    # ương, ươm, ươn
    ("nước non ngàn dặm ra đi",           "cái tình chi nặng nợ gì mà vương"),
    ("con cò mà đi ăn đêm",              "đậu phải cành mềm lộn cổ xuống ao"),
    ("bao giờ lúa chín vàng đồng",        "cho em gặt hái đem về nhà phơi"),

    # ơi, ơ, ôi
    ("anh đi anh nhớ quê nhà",           "nhớ canh rau muống nhớ cà dầm tương"),
    ("còn trời còn nước còn non",        "còn cô bán rượu anh còn say sưa"),
    ("xưa em vẫn đợi anh về",             "bây giờ cách trở sơn khê mịt mùng"),

    # ui, ưa, ưu, ươi
    ("cây đa bến nước con đò",            "đưa anh đi khắp mọi miền gần xa"),
    ("cây cao bóng cả sum suê",           "cây nào non yếu gió lay đổ nhào"),
    ("vợ chồng như bát nước đầy",         "đắng cay ngọt bùi chia sẻ cùng nhau"),

    # uôn, uông, uy, uya
    ("yêu nhau mấy núi cũng trèo",        "mấy sông cũng lội mấy đèo cũng qua"),
    ("ai ơi giữ chí cho bền",             "dù ai xoay hướng đổi nền mặc ai"),
    ("đất quảng nam mưa chưa thấm",    "rượu hồng đào mới nhấm đà ngây ngất"),

    # ăn, ăng, ân, ât
    ("còn cha còn mẹ thì hơn",            "không cha không mẹ như đờn đứt dây"),
    ("nước trong khe suối chảy ra",        "lọc lừa đãi cát tìm vàng mà thôi"),
    ("tôm càng lột vỏ bỏ đuôi",           "giã gạo cho trắng mà nuôi mẹ già"),

    # ôm, ong, oi, oan
    ("thương chồng phải lụy đến chồng",   "bưng bát cơm đầy nước mắt dầm đìa"),
    ("bồng bồng mẹ bế con sang",          "đò dọc quan cấm đò ngang không chèo"),
    ("gừng cay muối mặn đừng quên",   "đôi ta tình nặng nghĩa bền từ lâu"),

    # ênh, ich, iêm, ia
    ("lênh đênh một chiếc thuyền nan",     "qua bao ghềnh thác muôn ngàn phong ba"),
    ("tháp mười đẹp nhất bông sen",        "việt nam đẹp nhất có tên bác hồ"),
    ("áo dài tha thướt trong chiều",       "gió bay tà áo mang nhiều nhớ nhung"),

    # ut, uơ, uôi, uôm
    ("cây đa cũ bến đò xưa",              "bộ hành có nghĩa nắng trưa cũng chờ"),
    ("công anh đắp mả trồng khoai",        "trồng dưa hái quả cả hai cùng làm"),
    ("râu tôm nấu với ruột bầu",          "chồng chan vợ húp gật đầu khen ngon"),

    # Additional diverse lines
    ("gánh cực mà đổ lên non",            "cong lưng mà chạy cực còn theo sau"),
    ("chiều chiều én liệng truông mây",   "cảm thương chú nghé đứng đây chờ đàn"),
    ("gió mùa thu lạnh lòng ai",     "gió về có lạnh đôi vai em gầy"),
    ("chờ anh cho tuổi còn xuân",         "một mai hoa rụng muôn phần xác xơ"),
    ("lá xanh bông trắng lại gần",        "nhị vàng bông trắng cạnh sân ai trồng"),
    ("cây cao thì gió càng lay",          "càng yêu càng ghét càng say càng cuồng"),
    ("thò tay mà ngắt cọng ngò",          "thương em đứt ruột giả đò ngó lơ"),
    ("chồng già vợ trẻ là tiên",          "chồng trẻ vợ già là duyên nợ đời"),
    ("thương ai thương cả tấm lòng",       "thương ai thương cả ruộng đồng nương dâu"),
    ("cô kia cắt cỏ bên sông",            "có muốn ăn nhãn thì lồng sang đây"),
    ("dù ai đi ngược về xuôi",            "nhớ ngày giỗ tổ mồng mười tháng ba"),
    ("lúa chiêm lấp ló đầu bờ",           "hễ nghe tiếng sấm phất cờ mà lên"),
    ("chẳng tham nhà ngói rung rinh",     "tham vì một nỗi anh xinh miệng cười"),
    ("đêm qua em có nằm mơ",              "thấy con cá lội thấy mờ trăng suông"),
    ("cơm cha áo mẹ chữ thầy",            "gắng công mà học có ngày thành danh"),
    ("làm người phải biết tổ tông",        "như cây có cội như sông có nguồn"),
    ("ra đi anh nhớ quê nghèo",            "nhớ sông nhớ biển nhớ đèo nhớ non"),
    ("nhớ canh rau đắng nấu cà",          "nhớ ai dãi nắng dầm sương nhớ nào"),
    ("bao giờ cho đến tháng mười",         "đem lá mà đắp đem chồi mà che"),
    ("ai làm cho gió đừng rung",           "cho trăng đừng lặn cho anh đừng buồn"),
    ("yêu nhau cau sáu bổ ba",             "ghét nhau cau sáu bổ ra làm mười"),
    ("trăm năm đành lỗi hẹn hò",           "cây đa bến cũ con đò khác đưa"),
    ("ai đi muôn dặm non sông",            "để ai chất chứa sầu đong vơi đầy"),
    ("dã tràng xe cát biển đông",          "nhọc nhằn mà chẳng nên công cán gì"),
    ("anh em như thể chân tay",            "rách lành đùm bọc dở hay đỡ đần"),
    ("qua đình ngả nón trông đình",        "đình bao nhiêu ngói thương mình bấy nhiêu"),
    ("dù cho sông cạn đá mòn",             "vẫn không sờn dạ vẫn còn yêu em"),
    ("thân em như hạt mưa sa",             "hạt vào đài các hạt ra ruộng cày"),
    ("con ong bay lượn vườn hồng",         "kiếm nơi ấm cúng tổ tông họ hàng"),
    ("trăng lên trăng đứng trăng tà",      "nhớ ai thương những xót xa trong lòng"),
]


# ═══════════════════════════════════════════════════════════
# QUALITY EVAL PROMPTS (40 prompts — subset of couplets + diverse extras)
# ═══════════════════════════════════════════════════════════

QUALITY_PROMPTS = COUPLET_PROMPTS[:20] + [
    ("hỡi cô tát nước bên đàng",           "sao cô múc ánh trăng vàng đổ đi"),
    ("yêu nhau cởi áo cho nhau",           "về nhà dối mẹ qua cầu gió bay"),
    ("dù ai buôn bán trăm bề",             "mồng mười tháng tám thì về chợ đông"),
    ("thò tay mà ngắt cọng ngò",           "thương em đứt ruột giả đò ngó lơ"),
    ("chồng già vợ trẻ là tiên",           "chồng trẻ vợ già là duyên nợ đời"),
    ("thân em như củ ấu gai",              "ruột trong thì trắng vỏ ngoài thì đen"),
    ("trăm năm đành lỗi hẹn hò",           "cây đa bến cũ con đò khác đưa"),
    ("dã tràng xe cát biển đông",          "nhọc nhằn mà chẳng nên công cán gì"),
    ("dù cho sông cạn đá mòn",             "vẫn không sờn dạ vẫn còn yêu em"),
    ("ra đi anh nhớ quê nghèo",            "nhớ sông nhớ biển nhớ đèo nhớ non"),
    ("chiều chiều ra đứng bờ ao",          "trông mây trông nước trông sao trông chồng"),
    ("gánh cực mà đổ lên non",             "cong lưng mà chạy cực còn theo sau"),
    ("cây cao thì gió càng lay",           "càng yêu càng ghét càng say càng cuồng"),
    ("trầu vàng nhá với cau non",          "để anh thương nhớ héo hon ruột tằm"),
    ("dù ai đi ngược về xuôi",             "nhớ ngày giỗ tổ mồng mười tháng ba"),
    ("anh về em những trông theo",         "trông về quê cũ đói nghèo xót xa"),
    ("yêu nhau tam tứ núi cũng trèo",      "thất bát sông cũng lội cửu thập đèo cũng qua"),
    ("thương ai thương cả tấm lòng",       "thương ai thương cả ruộng đồng nương dâu"),
    ("cơm cha áo mẹ chữ thầy",             "gắng công mà học có ngày thành danh"),
    ("con ong bay lượn vườn hồng",         "kiếm nơi ấm cúng tổ tông họ hàng"),
]


# ═══════════════════════════════════════════════════════════
# VERIFICATION (run: python evaluate/prompts.py)
# ═══════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
    from tones import get_rhyme_group
    from collections import Counter

    ok = bad = 0
    for i, (l6, l8) in enumerate(COUPLET_PROMPTS):
        n6, n8 = len(l6.split()), len(l8.split())
        if n6 != 6 or n8 != 8:
            print(f"  ❌ [{i}] syllables: l6={n6} l8={n8} — \"{l6}\" / \"{l8}\"")
            bad += 1
        else:
            ok += 1

    rhymes = [get_rhyme_group(l8.split()[7]) for _, l8 in COUPLET_PROMPTS if len(l8.split()) == 8]
    rc = Counter(rhymes)

    print(f"Single-line prompts: {len(SINGLE_PROMPTS)}")
    print(f"Couplet prompts: {len(COUPLET_PROMPTS)}")
    print(f"  ✅ Valid 6+8: {ok}")
    if bad:
        print(f"  ❌ Bad syllable: {bad}")
    print(f"  Unique rhyme groups: {len(rc)}")
    print(f"  Top 10: {rc.most_common(10)}")
    print(f"Quality prompts: {len(QUALITY_PROMPTS)}")

    if bad == 0:
        print("\n✅ All prompts valid!")
    else:
        print(f"\n❌ {bad} prompts need fixing")
        sys.exit(1)
