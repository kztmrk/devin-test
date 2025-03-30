# Devin Test Repository

# Streamlit カスタムLLM Agent ストリーミングチャットボット

このアプリケーションは、StreamlitとカスタムLLM Agentを介してAzure OpenAI APIを使用するチャットボットで、ストリーミングレスポンスをサポートしています。独自のロジックを持つLLM Agentを通じて高度なチャット機能を提供します。ローカルPCで実行できます。

## 機能

- カスタムLLM Agentを介したAzure OpenAI API連携
- 独自のロジックによるメッセージ処理
- リアルタイムのストリーミングレスポンス表示
- 会話履歴の管理
- デプロイメント名の指定
- 会話リセット機能
- 拡張可能なエージェントフレームワーク

## エージェントタイプ

このアプリケーションは、以下のLLMエージェントタイプをサポートしています：

1. **Azure OpenAIエージェント** - Azure OpenAI APIを直接使用する基本エージェント
2. **コンテキスト対応エージェント** - ドキュメントコンテキストを活用した情報提供エージェント
3. **ツール活用エージェント** - 外部ツール/APIを利用できる拡張エージェント
4. **DuckDuckGo検索エージェント** - DuckDuckGoを使用して外部情報を検索するエージェント

エージェントタイプは開発者オプションから選択できます。

### DuckDuckGo検索エージェント

DuckDuckGo検索エージェントは、ユーザーの質問に応じて自動的に外部情報を検索し、最新かつ正確な情報を提供します。

**主な機能:**
- ユーザーの質問から検索の必要性を自動判断
- 質問から最適な検索クエリを生成
- DuckDuckGoを使用したリアルタイム検索
- 検索結果をAIの回答に統合
- 明示的な検索コマンド（「検索:」または「search:」で始まるメッセージ）のサポート

**設定オプション:**
- 検索機能の有効/無効切り替え
- 最大検索結果数の調整
- 検索リージョンの選択（日本語、英語など）
- ニュース検索の有効/無効切り替え
- 検索クエリ最適化回数の調整

**追加機能:**
- 検索クエリの自動最適化（検索結果が少ない場合に再検索）
- ニュース検索を活用した最新情報の取得
- 検索結果の公開日時情報の抽出と表示
- 情報ソースの種類（一次情報/二次情報）の判別と表示

## セットアップ手順

### 前提条件

- Python 3.8以上
- Azure OpenAI リソースへのアクセス権
  - Azure OpenAI APIキー
  - Azure OpenAIエンドポイント
  - デプロイメント名

### インストール

1. リポジトリをクローンまたはダウンロードします

2. 依存関係をインストールします
```bash
pip install -r requirements.txt
```

3. `.env.example`ファイルを`.env`にコピーし、Azure OpenAI設定を入力します
```bash
cp .env.example .env
# .envファイルを編集して以下の設定を行います
# AZURE_OPENAI_API_KEY=your_api_key_here
# AZURE_OPENAI_ENDPOINT=your_endpoint_here
# AZURE_OPENAI_API_VERSION=2023-05-15
# AZURE_OPENAI_DEPLOYMENT_NAME=your_deployment_name_here
```

### 実行方法

以下のコマンドでアプリケーションを起動します：
```bash
streamlit run src/streamlit_app/app.py
```

ブラウザでStreamlitが提供するURLにアクセスしてアプリケーションを使用できます。

## 使用方法

1. サイドバーで以下の情報を入力します（環境変数で設定済みの場合は不要）
   - Azure OpenAI APIキー
   - Azure OpenAIエンドポイント
   - デプロイメント名
   - APIバージョン
2. チャット入力欄にメッセージを入力し、Enterキーを押すか送信ボタンをクリックします
3. AIからのレスポンスがリアルタイムでストリーミング表示されます
4. 会話をリセットするには、サイドバーの「会話をリセット」ボタンをクリックします

## 注意事項

- Azure OpenAI APIの使用には料金が発生する場合があります
- APIキーは安全に管理してください
- このアプリケーションはローカルでの使用を想定しています
