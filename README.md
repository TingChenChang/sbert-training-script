# sbert-training-script
SBERT 成對句子分類任務

-----

## Bi-Encoder

給定句子生成句子的embedding。

訓練方式：將A,B句子獨立傳遞給BERT Model，產生句子的embedding分別為u, v，使用cosine-similarity衡量句子間相似度。

優點：embedding方式運算效率較高，適合作為索引搜尋

缺點：準確度較Cross-Encoder差

---

## Cross-Encoder

相當於BERT的SequenceClassification，給定句對輸出句子間相似度。

訓練方式：將A,B句子傳遞給同一個BERT Model，產出0-1間輸出值表示句子間相似度。

優點：準確度較Bi-Encoder好

缺點：輸入句對方式運算效率低

---

## Retrieve(Bi-Encoder) & Re-Rank(Cross-Encoder)

當一個新的問題要找出最佳回覆

先使用Bi-Encoder將問題embedding與現有所有問題/答案的embedding計算cosine-similarity，找出相似度前100問題/答案

在使用Cross-Encoder計算問題與Bi-Encoder相似度前100問題/答案各組合的分數，決定最佳回覆


-----

