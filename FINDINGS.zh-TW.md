# DroneDetect — 階段性發現與驗證閉環

> English: [FINDINGS.md](FINDINGS.md) · 專案總覽：[README.zh-TW.md](README.zh-TW.md)

本文件收束目前的驗證閉環：**為什麼選擇 PSD + 線性模型**、**各項驗證的狀態**（含圖連結）、以及**結論**。剩餘工作皆為加分項（見§待辦）。

## 摘要（TL;DR）

在此資料集上，**正規化的 1024-bin Welch PSD 餵給線性分類器（LDA）是平衡最佳的無人機機型分類器**——7 機型 segment 準確率 0.97、跨干擾最魯棒、訓練成本近乎零。Spectrogram CNN 學到*互補*線索，但在準確率與魯棒性上都未勝出。部署上，**單一約 25 ms 窗即達 ~0.95**（約 12.5 ms 達 ~0.92）。唯一在此結構下無法解決的限制是 **機型 ≡ 錄製 session 的 confound**。

## 為什麼選擇 PSD（與線性模型）

這個選擇是各階段證據**逼出來的**，不是預設的：

1. **資料品質排除了絕對振幅特徵。** 約 15 dB 的 gain confound 把資料分成強訊號群（AIR/DIS/PHA）與弱訊號群（INS/MIN/MP1/MP2），且強訊號群約 50–60% 錄製削波。因此任何原始 power 特徵編碼的是採集設定而非無人機。正規化 PSD（總功率→1）在*特徵層*直接消除 gain confound。
   → [EDA/results/overview_by_drone.png](EDA/results/overview_by_drone.png)、
   [EDA/results/box_clip_ratio.png](EDA/results/box_clip_ratio.png)

2. **頻譜形狀近乎線性可分。** LDA 對正規化 PSD 達 0.972 segment / 1.000 recording 準確率；XGBoost 沒有更好，代表結構是線性的、不需重模型。唯一有意義的混淆是 MP1↔MP2（同家族 OcuSync）。
   → [embedding/results/baseline_confusion.png](embedding/results/baseline_confusion.png)

3. **CNN 沒有勝過它。** Spectrogram CNN 準確率較低（0.946）、*惡化* MP1/MP2，且跨干擾遷移*更差*（掉分 0.19 vs 0.13）。它確實學到不同線索（McNemar 顯著、CKA 0.18、三模型 ensemble 0.98）——但是互補，不是更優。
   → [CNN/results/cnn_confusion.png](CNN/results/cnn_confusion.png)

**平衡評分：** 準確率（LDA 最佳）、跨干擾魯棒性（LDA 最佳）、訓練成本（LDA 近乎零 vs CNN 在 CPU 上數小時）、可解釋性（LDA 透明）。在此案的每個關鍵面向，PSD + LDA 都勝出。

## 各項驗證狀態

| # | 驗證 | 結果 | 圖 |
|---|---|---|---|
| 1 | 資料品質稽核（gain 分群、削波） | 絕對特徵不可用 → 必須正規化 | [overview](EDA/results/overview_by_drone.png)、[clip](EDA/results/box_clip_ratio.png) |
| 2 | Baseline 可分性（LDA vs XGBoost, LORO） | 0.972 / 1.000；線性足夠；僅 MP1↔MP2 混淆 | [confusion](embedding/results/baseline_confusion.png) |
| 3 | 干擾遷移（LDA） | 跨條件掉 ~12–15 pts，不崩盤（≥0.75） | [transfer](verify/results/interference_transfer.png) |
| 4 | 模型比較（McNemar、ensemble） | CNN 錯誤顯著不同；ensemble 0.98 > 任一單模型 | `verify/results/model_comparison.json` |
| 5 | Session leakage probing + CKA | 無 run-level 指紋（PSD run_index 0.05）；CKA 0.18 | [probe](verify/results/session_leakage_probe.png) |
| 6 | CNN vs LDA 干擾遷移 | **假設被推翻**——CNN 遷移更差（掉分 0.19 vs 0.13） | [cnn-vs-lda](verify/results/cnn_vs_lda_interference_transfer.png) |
| 7 | 最短窗長度 | ~12.5 ms → 0.92、~25 ms → 0.95，之後報酬遞減 | [sweep](verify/results/segment_length_sweep.png) |
| 8 | 多窗投票 vs 單一長窗 | soft > hard；相同時間下單一長窗 ≥ 投票 | [voting](verify/results/multiwindow_voting.png) |

*（LORO = leave-one-run-out；除另註明外，所有測試皆混合 4 種干擾條件。）*

## 結論

- **PSD + 線性模型是平衡最佳解**——由 baseline、模型比較、probing、干擾遷移四方獨立證實。
- **部署窗長：** 連續且乾淨觀測下，優先用單一長窗——**~25 ms ≈ 0.95**、**~12.5 ms ≈ 0.92**。多窗投票（要用 *soft* voting）在相同觀測時間下不勝單一長窗；只有在觀測斷續或單一窗可能被污染時才有價值。
- **CNN 的定位** 是互補的第二意見（ensemble 有增益），不是主分類器。
- **最難的殘留問題：** 同家族的 MP1↔MP2。

## 已知限制

- **機型 ≡ session confound 在此結構上無解。** 每個機型很可能單一 session 錄製；leave-one-run-out 與 probe 只能排除*session 內重複*的指紋，無法排除 session 身分本身。跨 SDR / 跨日期泛化未經驗證。
- **沒有「無人機不在場」的錄製**——支援機型*分類*，非在場*偵測*（後者需外部負樣本）。
- **256-bin 低估：** 窗長與投票研究重用 256-bin spectrogram；原生 1024-bin PSD 約高 1 個百分點。趨勢不受影響。

## 待辦（加分項）

- **Grad-CAM** 對 spectrogram 做歸因——CNN 看的是哪些時頻區域，是訊號還是接收端 artifact。
- **增益擾動壓力測試**——確認正規化 PSD 在測試時 ±dB 縮放下仍增益不變。
- **更高頻率解析度 CNN**（512/1024 bins）——再一次嘗試縮小 MP1↔MP2 差距。
