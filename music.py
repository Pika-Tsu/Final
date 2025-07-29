import streamlit as st
import numpy as np
import pandas as pd

from gensim.models import KeyedVectors
import numpy as np

# ── 読み込み ───────────────────────────
vectors = np.load("data/my_musicvectors.npy")                    # shape = (V, D)
vocab   = np.load("data/my_musicvocab.npy", allow_pickle=True)   # shape = (V,)
df      = pd.read_csv("data/song_meta.csv").set_index("name")    # 音楽の公開年とbpmのデータ

# ── KeyedVectors を組み立てる ─────────
kv = KeyedVectors(vector_size=vectors.shape[1])
kv.add_vectors(vocab.tolist(), vectors)      # gensim 4.x 以降の公式 API :contentReference[oaicite:0]{index=0}
kv.fill_norms()                              # 類似度計算を高速化（省略可）


st.title("音楽レコメンドアプリ")
st.write("アーティスト Eveが制作した曲を選択することで、選択した曲と似ているEveが作った曲を調べることができます")
st.write("曲のタイトルはYoutubeへのリンクになっています")
st.write(" ")

music_titles = vocab.tolist()

# スコアの計算をするよ
def combined_score(original_score, year_diff, bpm_diff, alpha=0.01, beta=0.05):
        ##year_diffは公開年の差　bpm_diffはBPMの差　alphaは年の重み　betaはbpmの重み　→bpmが近ければ年が遠くてもok
        return original_score - alpha * abs(year_diff) -beta *abs(bpm_diff)


# ── 1曲を選ぶ ───────────────────────────
st.markdown("### １曲の音楽に対して似ている音楽を表示する")
selected_music = st.selectbox("音楽を選んでください", music_titles)

# 似ている音楽を表示
if selected_music: 
    try:
        target_url  = df.loc[selected_music, "url"] 
        target_year = df.loc[selected_music, "year"]    #dfから選んだ曲の公開年を取得
        target_bpm  = df.loc[selected_music, "bpm"]     #dfから選んだ曲のbpmを取得
    except KeyError:                                    #dfに曲が見つからない時
        st.warning("情報が見つかりませんでした")
        st.stop()
    candidates = kv.most_similar(selected_music, topn=100)

    st.markdown(f"##### 「 [{selected_music}]({target_url}) 」  に似ている曲")

    results = []
    for recommend_music, original_score in candidates:
        if recommend_music not in df.index:
            continue
        try:
            year_diff = df.loc[recommend_music, "year"] - target_year                   #公開年の差を計算
            bpm_diff = df.loc[recommend_music, "bpm"] - target_bpm                      #bpmの差を計算
            score = combined_score(original_score, year_diff,bpm_diff)                  #combined_scoreでスコアを計算
            url   = df.loc[recommend_music, "url"]                                      #dfからおすすめ曲のurlを取得
            results.append({"title": recommend_music, "score": score, "URL":url})       #結果表示のためのappend
        except KeyError:
            st.warning("情報が見つかりませんでした")
    results = pd.DataFrame(results).sort_values("score", ascending=False).head(30)  #結果をスコアの上から30こまで絞る
    
    for i, row in enumerate(results.itertuples(), 1):
        title = row.title
        url   = row.URL
        if i<11:
            st.markdown(f"{i} 位  「 [{title}]({url}) 」")
        if i==1:
            st.video(url)

    with st.expander("さらに詳しい結果"):
        st.write ("こちらでは30位までの結果がスコア付きで見られます")

        #結果画面にYoutubeのURLをリンクとして埋め込み
        st.dataframe(
            results,
            column_config={
                "URL":st.column_config.LinkColumn("URL")
            },
            hide_index=True
        )


# ── 複数曲を選ぶ ───────────────────────────
st.markdown("### 複数の音楽を選んでおすすめの音楽を表示する")

selected_musics = st.multiselect("音楽を複数選んでください", music_titles) ##映画のタイトルを配列に入れる
if len(selected_musics) > 0:
    try:
        user_vector = np.mean(vectors, axis=0)
        ave_year    = df.loc[selected_musics, "year"].mean()
        ave_bpm     = df.loc[selected_musics, "bpm"].mean()
    except KeyError:
        st.warning("選択した曲の中に情報が見つからないものがありました")
        st.stop()

    # 各曲についてリンク付きの文字列を作る
    linked_names = []
    for name in selected_musics:
        url = df.loc[name,"url"]
        linked_names.append(f"[{name}]({url})")

    # 結合して表示
    base_message = f"「 {'」「'.join(linked_names)} 」 に似ている曲"
    st.markdown(f"##### {base_message}")


    # base_message = f"『{'』『'.join(selected_musics)}』 に似ている曲"
    # st.markdown(f"### {base_message}")


    # st.markdown(f"##### 選択した曲に似ている曲")
    # for music in selected_musics:
    #     try:
    #         url = df.loc[music, "url"]
    #         st.markdown(f"-[{music}]({url})")
    #     except KeyError:
    #         st.markdown(f"-{music}（URLが見つかりません）")
    
    candidates = kv.similar_by_vector(user_vector, topn=100)

    results = []
    for recommend_music, original_score in candidates:
        if recommend_music in selected_musics:
            continue
        year_diff = df.loc[recommend_music, "year"] - ave_year
        bpm_diff  = df.loc[recommend_music, "bpm"] - ave_bpm
        score     = combined_score(original_score, year_diff, bpm_diff)
        url       = df.loc[recommend_music, "url"]
        results.append({"title": recommend_music, "score": score, "URL": url})
    results = pd.DataFrame(results).sort_values("score", ascending=False).head(30)

    for i, row in enumerate(results.itertuples(), 1):
        title = row.title
        url   = row.URL
        if i<11:
            st.markdown(f"{i} 位  「 [{title}]({url}) 」")
        if i==1:
            st.video(url)

    with st.expander("さらに詳しい結果"):
        st.write ("こちらでは30位までの結果がスコア付きで見られます")

        #結果画面にYoutubeのURLをリンクとして埋め込み
        st.dataframe(
            results,
            column_config={
                "URL":st.column_config.LinkColumn("URL")
            },
            hide_index=True
        )
    
# 本来なら下記のような簡単な読み込みで対抗可能。ただし、gensimバージョンが異なるとエラー出る
# model = gensim.models.word2vec.Word2Vec.load("data/manga_item2vec.model")