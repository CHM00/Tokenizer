# import jieba
# from gensim import corpora
#
# # Step 1: 准备分词后的语料 (新闻标题)
# raw_headlines = [
#     "央行降息，刺激股市反弹",
#     "球队赢得总决赛冠军，球员表现出色"
# ]
# tokenized_headlines = [jieba.lcut(doc) for doc in raw_headlines]
# print(f"分词后语料: {tokenized_headlines}")
#
# # Step 2: 创建词典
# dictionary = corpora.Dictionary(tokenized_headlines)
# print(f"词典: {dictionary.token2id}")
#
# # Step 3: 转换为BoW向量语料库（Bag of Word 向量语料库）
# corpus_bow = [dictionary.doc2bow(doc) for doc in tokenized_headlines]
# print(f"BoW语料库: {corpus_bow}")
#
#
# print("------------------TF-IDF----------------------")
# import jieba
# from gensim import corpora, models
#
# # 1. 准备语料 (新闻标题，包含财经和体育两个明显主题)
# headlines = [
#     "央行降息，刺激股市反弹",
#     "球队赢得总决赛冠军，球员表现出色",
#     "国家队公布最新一期足球集训名单",
#     "A股市场持续震荡，投资者需谨慎",
#     "篮球巨星刷新历史得分记录",
#     "理财产品收益率创下新高"
# ]
# tokenized_headlines = [jieba.lcut(title) for title in headlines]
#
# # 2. 创建词典和BoW语料库
# dictionary = corpora.Dictionary(tokenized_headlines)
# print(f"词典: {dictionary.token2id}")
#
# corpus_bow = [dictionary.doc2bow(doc) for doc in tokenized_headlines]
# print(f"BoW语料库: {corpus_bow}")
#
# # 3. 训练TF-IDF模型
# tfidf_model = models.TfidfModel(corpus_bow)
#
# # 4. 将BoW语料库转换为TF-IDF向量表示
# corpus_tfidf = tfidf_model[corpus_bow]
#
# # 辅助函数：把 (token_id, weight) 转成 (token, weight)，并按权重降序展示
# def tfidf_with_words(tfidf_vec, id2word):
#     pairs = [(id2word[token_id], weight) for token_id, weight in tfidf_vec]
#     return sorted(pairs, key=lambda x: x[1], reverse=True)
#
# # 打印第一篇标题的TF-IDF向量
# first_tfidf = list(corpus_tfidf)[0]
# print("第一篇标题的TF-IDF向量:")
# print(first_tfidf)
# print("第一篇标题的TF-IDF向量(带词语):")
# print(tfidf_with_words(first_tfidf, dictionary))
#
# # 5. 对新标题应用模型
# new_headline = "股市大涨，牛市来了"
# new_headline_bow = dictionary.doc2bow(list(jieba.cut(new_headline)))
# new_headline_tfidf = tfidf_model[new_headline_bow]
# print("\n新标题的TF-IDF向量:")
# print(new_headline_tfidf)



print("------------------------LDA------------------------")
from gensim import corpora, models
import jieba
# 1. 准备语料
headlines = [
    "央行降息，刺激股市反弹",
    "球队赢得总决赛冠军，球员表现出色",
    "国家队公布最新一期足球集训名单",
    "A股市场持续震荡，投资者需谨慎",
    "篮球巨星刷新历史得分记录",
    "理财产品收益率创下新高"
]
tokenized_headlines = [jieba.lcut(title) for title in headlines]


# 2. 创建词典和BoW语料库
dictionary = corpora.Dictionary(tokenized_headlines)
print(f"词典: {dictionary.token2id}")

corpus_bow = [dictionary.doc2bow(doc) for doc in tokenized_headlines]
print(corpus_bow)

# 3. 训练LDA模型 (假设需要发现2个主题)
lda_model = models.LdaModel(corpus=corpus_bow, id2word=dictionary, num_topics=2, random_state=100)

# 4. 查看模型发现的主题
print("模型发现的2个主题及其关键词:")
for topic in lda_model.print_topics():
    print(topic)

# 5. 推断新文档的主题分布
# new_headline = "詹姆斯获得常规赛MVP"
new_headline = "巨星詹姆斯获得常规赛MVP"
new_headline_bow = dictionary.doc2bow(jieba.lcut(new_headline))
topic_distribution = lda_model[new_headline_bow]
print(f"\n新标题 '{new_headline}' 的主题分布:")
print(topic_distribution)
'''
    如果新文本在词典中几乎没有重叠词，没在词典中找到，推断出的主题分布可能接近均匀（例如 2 个主题时约为 0.5/0.5）。
'''



from gensim.models import Word2Vec

# 1. 准备语料
headlines = [
    # 财经
    "央行降息，刺激股市反弹",
    "A股市场持续震荡，投资者需谨慎",
    "理财产品收益率创下新高",
    "证监会发布新规，规范市场交易",
    "创业板指数上涨，科技股领涨大盘",
    "房价调控政策出台，房地产市场降温",
    "全球股市动荡，影响资本市场信心",
    "分析师认为，当前股市风险与机遇并存，市场情绪复杂",

    # 体育
    "球队赢得总决赛冠军，球员表现出色",
    "国家队公布最新一期足球集训名单",
    "篮球巨星刷新历史得分记录",
    "奥运会开幕，中国代表团旗手确定",
    "马拉松比赛圆满结束，选手创造佳绩",
    "电子竞技联赛吸引大量年轻观众",
    "这支球队的每位球员都表现出色",
    "球员转会市场活跃，多支球队积极引援"
]
tokenized_headlines = [jieba.lcut(title) for title in headlines]
print("分词token: ", tokenized_headlines)

from collections import Counter
word_count = Counter()
for title in tokenized_headlines:
    word_count.update(title)
print("word_count: ", len(word_count), word_count)

# 2. 训练Word2Vec模型
model = Word2Vec(tokenized_headlines, vector_size=50, window=3, min_count=1, sg=1)

# 训练完成后，所有词向量都存储在 model.wv 对象中
# model.wv 是一个 KeyedVectors 实例
# 1. 寻找最相似的词
# 在小语料上，结果可能不完美，但能体现出模型学习到了主题内的关联
similar_to_market = model.wv.most_similar('股市')
print(f"与 '股市' 最相似的词: {similar_to_market}")

# similar_to_market = model.wv.most_similar('北京')
# print(f"与 '北京' 最相似的词: {similar_to_market}")

# 2. 计算两个词的余弦相似度
similarity = model.wv.similarity('球队', '球员')
print(f"\n'球队' 和 '球员' 的相似度: {similarity:.4f}")

# 3. 获取一个词的向量
market_vector = model.wv['市场']
print(f"\n'市场' 的向量维度: {market_vector.shape}")


from gensim.models import KeyedVectors

# 保存词向量到文件
model.wv.save("news_vectors.kv")

# 从文件加载词向量
loaded_wv = KeyedVectors.load("news_vectors.kv")

# 加载后可以执行同样的操作
print(f"\n加载后，'球队' 和 '球员' 的相似度: {loaded_wv.similarity('球队', '球员'):.4f}")


