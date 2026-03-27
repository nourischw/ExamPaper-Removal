# 試卷橡皮擦

自動去除試卷上的手寫答案，生成乾淨的空白卷。

## 部署

1. 去 [HuggingFace](https://huggingface.co/settings/tokens) 申請免費 API Token
2. 在 Vercel 專案設定中加入環境變數：
   - `HUGGINGFACE_TOKEN` = 你的 token
3. Deploy

## 原理

使用 BiRefNet 語意分割模型 + 墨跡偵測，自動識別並去除手寫筆跡，保留印刷內容。

## 限制

- 需要 HuggingFace API Token（免費）
- 模型加載需要幾秒鐘（冷啟動）
- 效果取決於試卷品質（掃描質量越好效果越好）
