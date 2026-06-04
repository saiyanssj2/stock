import streamlit as st


def show_wyckoff_edu():
    st.subheader("📚 Phương pháp Wyckoff — Kiến thức nền tảng")

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🧠 Tổng quan",
        "📈 Accumulation",
        "📉 Distribution",
        "🔬 VSA",
        "🎯 Ứng dụng thực tế"
    ])

    # ===================== TAB 1: TONG QUAN =====================
    with tab1:
        st.markdown("### Richard Wyckoff là ai?")
        st.markdown("""
Richard Wyckoff (1873–1934) là nhà giao dịch và phân tích thị trường người Mỹ.
Ông phát triển phương pháp phân tích dựa trên **hành vi giá và khối lượng** để xác định
dấu vết của **"Composite Man"** — tức các tổ chức lớn (smart money) đang làm gì trên thị trường.
        """)

        st.markdown("### 3 Quy luật nền tảng")
        c1, c2, c3 = st.columns(3)
        with c1:
            st.info("""
**⚖️ Quy luật Cung — Cầu**

Giá tăng khi cầu > cung.
Giá giảm khi cung > cầu.

Wyckoff đo lường cung/cầu qua
**spread nến + volume**.
            """)
        with c2:
            st.info("""
**🔗 Quy luật Nhân — Quả**

Tích lũy (Accumulation) tạo ra
nền tảng cho đà tăng (Markup).

Phân phối (Distribution) tạo ra
nền tảng cho đà giảm (Markdown).

Tích lũy càng lâu → tăng càng mạnh.
            """)
        with c3:
            st.info("""
**💪 Quy luật Nỗ lực — Kết quả**

Volume = Nỗ lực.
Price move = Kết quả.

Nỗ lực lớn mà kết quả nhỏ
→ lực cản mạnh → cảnh báo đảo chiều.
            """)

        st.markdown("### Composite Man là gì?")
        st.markdown("""
Wyckoff hình dung thị trường như một **"Composite Man"** — một thực thể duy nhất đại diện
cho toàn bộ smart money (quỹ lớn, tổ chức, market maker).

Composite Man hoạt động theo chu kỳ:
        """)
        st.markdown("""
```
Accumulation → Markup → Distribution → Markdown → Accumulation → ...
(Gom hàng)    (Đẩy giá)  (Xả hàng)    (Giảm giá)   (Gom lại)
```
        """)

        st.markdown("### 4 Pha thị trường")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.success("""
**🔵 Accumulation**
Tích lũy

Smart money gom hàng
từ retail investors
đang hoảng loạn bán.

Volume: Cao ở đáy,
giảm dần khi sideway.
            """)
        with col2:
            st.success("""
**🟢 Markup**
Tăng giá

Smart money đẩy giá lên.
Retail bắt đầu mua theo.

Volume: Tăng theo giá.
Pullback nhẹ, volume thấp.
            """)
        with col3:
            st.warning("""
**🟠 Distribution**
Phân phối

Smart money xả hàng
cho retail đang FOMO.

Volume: Cao ở đỉnh,
giá không tăng thêm.
            """)
        with col4:
            st.error("""
**🔴 Markdown**
Giảm giá

Smart money đã thoát.
Retail bán tháo.

Volume: Cao khi giảm.
Rally yếu, volume thấp.
            """)

    # ===================== TAB 2: ACCUMULATION =====================
    with tab2:
        st.markdown("### Sơ đồ Accumulation — Wyckoff")
        st.markdown("""
```
Giá
 │
 │    PS    SC         ST    SOS
 │     ↓    ↓    AR    ↓     ↑
 │─────●────●────●─────●─────●──────→ Markup
 │         │    │     │
 │         └────┘     └── Spring (Phase C)
 │          (Phase A)
 │
 └──────────────────────────────────→ Thời gian
```
        """)

        st.markdown("### Các sự kiện trong Accumulation")

        events = [
            ("Phase A — Dừng xu hướng giảm", [
                ("PS (Preliminary Support)", "Lực cầu xuất hiện lần đầu sau đà giảm dài. Volume tăng, giá bật nhẹ nhưng chưa đủ mạnh để dừng xu hướng."),
                ("SC (Selling Climax)", "Đáy thực sự. Volume cực lớn, spread rộng, giá giảm mạnh rồi đóng cửa phục hồi. Retail hoảng loạn bán, smart money hấp thụ."),
                ("AR (Automatic Rally)", "Giá bật mạnh sau SC do lực cung cạn kiệt. Xác định vùng kháng cự trên của trading range."),
                ("ST (Secondary Test)", "Giá quay lại test vùng SC với volume thấp hơn → xác nhận lực cung đã yếu đi."),
            ]),
            ("Phase B — Xây dựng nền tảng", [
                ("UT (Upthrust)", "Giá vượt AR rồi quay đầu — bẫy tăng, smart money test lực cung."),
                ("SOW (Sign of Weakness)", "Giá giảm về gần đáy SC với volume tăng — test lực cầu."),
                ("Sideway", "Giá dao động trong range, volume giảm dần → smart money đang gom hàng âm thầm."),
            ]),
            ("Phase C — Test cuối cùng", [
                ("Spring", "⭐ Tín hiệu quan trọng nhất! Giá xuyên đáy SC rồi phục hồi nhanh với volume thấp → lực cung đã cạn kiệt hoàn toàn. Đây là điểm mua tốt nhất."),
                ("LPS (Last Point of Support)", "Điểm hỗ trợ cuối trước khi breakout. Volume thấp, spread hẹp."),
            ]),
            ("Phase D — Breakout", [
                ("SOS (Sign of Strength)", "Giá breakout khỏi trading range với volume lớn, spread rộng → xác nhận Markup bắt đầu."),
                ("BU (Back Up)", "Giá pullback về test lại vùng breakout với volume thấp → cơ hội mua thêm."),
            ]),
        ]

        for phase_name, phase_events in events:
            with st.expander(f"**{phase_name}**", expanded=False):
                for event_name, event_desc in phase_events:
                    st.markdown(f"**{event_name}**")
                    st.caption(event_desc)
                    st.markdown("---")

        st.markdown("### Checklist xác nhận Accumulation")
        st.markdown("""
| Tiêu chí | Mô tả |
|---|---|
| ✅ Giá vùng đáy 60 phiên | Close ≤ Low60 + Range60 × 25% |
| ✅ OBV tăng dần | Smart money đang gom |
| ✅ Volume giảm khi sideway | No Supply — cung cạn |
| ✅ Spring hoặc LPS | Test đáy thất bại → đảo chiều |
| ✅ MACD histogram chuyển dương | Momentum bắt đầu tích cực |
| ✅ RSI thoát vùng oversold (>40) | Momentum phục hồi |
        """)

    # ===================== TAB 3: DISTRIBUTION =====================
    with tab3:
        st.markdown("### Sơ đồ Distribution — Wyckoff")
        st.markdown("""
```
Giá
 │         UT    UTAD
 │    PSY   ↑     ↑
 │─────●────●─────●────●──────→ Markdown
 │         │     │    │
 │         └─────┘    └── LPSY (Phase D)
 │          (Phase B)
 │
 └──────────────────────────────────→ Thời gian
```
        """)

        st.markdown("### Các sự kiện trong Distribution")

        dist_events = [
            ("Phase A — Dừng xu hướng tăng", [
                ("PSY (Preliminary Supply)", "Lực cung xuất hiện lần đầu. Volume tăng đột biến, giá vẫn tăng nhưng chậm lại."),
                ("BC (Buying Climax)", "Đỉnh thực sự. Volume cực lớn, spread rộng, giá tăng mạnh rồi đóng cửa giảm. Retail FOMO mua, smart money xả."),
                ("AR (Automatic Reaction)", "Giá giảm mạnh sau BC do lực cầu cạn. Xác định vùng hỗ trợ dưới của trading range."),
                ("ST (Secondary Test)", "Giá quay lại test vùng BC với volume thấp hơn → xác nhận lực cầu đã yếu."),
            ]),
            ("Phase B — Xây dựng phân phối", [
                ("UT (Upthrust)", "Giá vượt BC rồi quay đầu — bẫy tăng, smart money xả hàng cho retail FOMO."),
                ("SOW (Sign of Weakness)", "Giá giảm mạnh về gần AR với volume lớn → cảnh báo xu hướng giảm."),
            ]),
            ("Phase C — Test cuối cùng", [
                ("UTAD (Upthrust After Distribution)", "⭐ Tín hiệu quan trọng! Giá vượt đỉnh BC rồi quay đầu nhanh → bẫy tăng cuối cùng, smart money xả nốt. Đây là điểm bán tốt nhất."),
            ]),
            ("Phase D — Breakdown", [
                ("LPSY (Last Point of Supply)", "Rally yếu, volume thấp, spread hẹp → cơ hội bán/short cuối."),
                ("SOW (Sign of Weakness)", "Giá breakdown khỏi trading range với volume lớn → xác nhận Markdown bắt đầu."),
            ]),
        ]

        for phase_name, phase_events in dist_events:
            with st.expander(f"**{phase_name}**", expanded=False):
                for event_name, event_desc in phase_events:
                    st.markdown(f"**{event_name}**")
                    st.caption(event_desc)
                    st.markdown("---")

        st.markdown("### Checklist xác nhận Distribution")
        st.markdown("""
| Tiêu chí | Mô tả |
|---|---|
| ⚠️ Giá vùng đỉnh 60 phiên | Close ≥ High60 - Range60 × 25% |
| ⚠️ OBV giảm dần | Smart money đang xả |
| ⚠️ Volume cao nhưng giá không tăng | No Demand — cầu yếu |
| ⚠️ Upthrust hoặc UTAD | Test đỉnh thất bại → đảo chiều |
| ⚠️ MACD histogram âm nới rộng | Momentum tiêu cực |
| ⚠️ RSI phân kỳ âm | Giá tăng nhưng RSI giảm |
        """)

    # ===================== TAB 4: VSA =====================
    with tab4:
        st.markdown("### Volume Spread Analysis (VSA)")
        st.markdown("""
VSA là phần mở rộng của Wyckoff, phân tích **mối quan hệ giữa Volume, Spread và Closing Position**
để xác định lực cung/cầu thực sự.
        """)

        st.markdown("### 4 tình huống VSA cơ bản")

        col1, col2 = st.columns(2)
        with col1:
            st.success("""
**🟢 Effort Up — Result Up**
*(Nỗ lực tăng — Kết quả tăng)*

- Volume cao
- Spread rộng
- Đóng cửa phần trên nến

→ **Demand mạnh**, xu hướng tăng được xác nhận.
Nên mua theo.
            """)
            st.error("""
**🔴 Effort Up — Result Down**
*(Nỗ lực tăng — Kết quả giảm)*

- Volume cao
- Spread rộng
- Đóng cửa phần dưới nến

→ **Supply mạnh**, smart money đang xả.
Cảnh báo đảo chiều giảm.
            """)
        with col2:
            st.warning("""
**🟡 No Demand**
*(Không có cầu)*

- Volume thấp
- Spread hẹp
- Giá tăng nhẹ hoặc sideway

→ Rally thiếu lực, không có smart money
tham gia. Không nên mua.
            """)
            st.warning("""
**🟡 No Supply**
*(Không có cung)*

- Volume thấp
- Spread hẹp
- Giá giảm nhẹ hoặc sideway

→ Pullback thiếu lực cung.
**Tín hiệu tích cực** — cơ hội mua tốt
nếu đang trong Accumulation.
            """)

        st.markdown("### Các nến VSA quan trọng")
        vsa_candles = {
            "🕯️ Stopping Volume": "Volume đột biến cực lớn, spread rộng, đóng cửa giữa nến → lực cung/cầu đang được hấp thụ. Thường xuất hiện ở SC hoặc BC.",
            "🕯️ Inverse Upthrust": "Nến giảm mạnh với volume cao nhưng đóng cửa phục hồi về giữa/trên → lực cầu hấp thụ cung. Tín hiệu tích cực.",
            "🕯️ Pseudo Upthrust": "Nến tăng với volume thấp, đóng cửa thấp → không có lực cầu thực sự. Cảnh báo.",
            "🕯️ Test": "Nến có volume rất thấp, spread hẹp, giá không giảm được → cung đã cạn. Tín hiệu mua nếu trong Accumulation.",
            "🕯️ Shakeout": "Giá giảm đột ngột xuyên hỗ trợ rồi phục hồi ngay trong phiên → bẫy giảm, smart money gom hàng từ retail hoảng loạn.",
        }
        for name, desc in vsa_candles.items():
            with st.expander(name):
                st.markdown(desc)

        st.markdown("### Công thức đọc VSA")
        st.code("""
# Spread (biên độ nến)
spread = high - low
spread_avg = average(spread, 20 phiên)

# Closing position (giá đóng cửa ở đâu trong nến)
close_pos = (close - low) / (high - low)
# > 0.7 → đóng cửa cao (strength)
# < 0.3 → đóng cửa thấp (weakness)

# Phân loại
if volume > avg_vol * 1.3 and spread > avg_spread * 1.3:
    if close_pos > 0.6:  → Demand mạnh (Effort↑ Result↑)
    if close_pos < 0.4:  → Supply mạnh (Effort↑ Result↓)

if volume < avg_vol * 0.7 and spread < avg_spread * 0.7:
    if price_up:  → No Demand (cảnh báo)
    if price_down: → No Supply (tích cực)
        """, language="python")

    # ===================== TAB 5: UNG DUNG =====================
    with tab5:
        st.markdown("### Quy trình phân tích Wyckoff thực tế")

        st.markdown("""
**Bước 1: Xác định xu hướng lớn (Top-down)**
- Nhìn chart tuần/tháng → đang ở pha nào?
- VNINDEX đang Accumulation hay Distribution?
- Chỉ mua khi thị trường chung đang Markup hoặc cuối Accumulation
        """)

        st.markdown("""
**Bước 2: Tìm cổ phiếu trong Accumulation**
- Giá sideway sau đà giảm dài
- OBV tăng dần trong khi giá đi ngang
- Volume giảm khi pullback (No Supply)
- Xuất hiện Spring hoặc LPS
        """)

        st.markdown("""
**Bước 3: Xác nhận bằng VSA**
- Nến cuối phiên: spread hẹp, volume thấp → No Supply
- Không có nến Supply mạnh (volume cao, đóng cửa thấp)
- Test thành công: giá về hỗ trợ với volume thấp
        """)

        st.markdown("""
**Bước 4: Điểm vào lệnh**
- **Tốt nhất**: Mua tại Spring (xuyên đáy rồi phục hồi)
- **Tốt**: Mua tại LPS (pullback về hỗ trợ sau SOS)
- **Chấp nhận được**: Mua tại BU (pullback sau breakout)
        """)

        st.markdown("""
**Bước 5: Quản lý rủi ro**
- Stop Loss: Dưới đáy Spring hoặc dưới LPS
- Target: Đỉnh của trading range (kháng cự)
- R:R tối thiểu 1:2
        """)

        st.divider()
        st.markdown("### Những lỗi phổ biến khi dùng Wyckoff")
        st.error("""
❌ **Mua sớm trong Phase A/B** — chưa có xác nhận tích lũy xong

❌ **Bỏ qua volume** — phân tích giá mà không xem volume là thiếu sót lớn

❌ **Nhầm Spring với breakdown thật** — Spring phải phục hồi nhanh trong cùng phiên hoặc phiên kế

❌ **Không xem thị trường chung** — cổ phiếu tốt cũng khó tăng khi VNINDEX đang Markdown

❌ **Kỳ vọng sơ đồ hoàn hảo** — thực tế thị trường không bao giờ đẹp như sách giáo khoa
        """)

        st.divider()
        st.markdown("### Tóm tắt nhanh — Bảng tra cứu")
        st.markdown("""
| Tín hiệu | Ý nghĩa | Hành động |
|---|---|---|
| Spring + volume thấp | Cung cạn, sắp tăng | 🟢 Mua |
| SOS + volume cao | Breakout xác nhận | 🟢 Mua thêm |
| BU về hỗ trợ | Pullback sau breakout | 🟢 Mua thêm |
| No Supply liên tiếp | Tích lũy đang diễn ra | 🟡 Theo dõi |
| Upthrust + volume cao | Bẫy tăng, sắp giảm | 🔴 Thoát/Bán |
| UTAD + volume cao | Phân phối xong | 🔴 Bán ngay |
| SOW + volume cao | Breakdown xác nhận | 🔴 Thoát hết |
| No Demand khi rally | Rally yếu, không bền | 🟡 Không mua |
        """)
