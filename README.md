# CL-Bert

计算语言学作业

本项目baseline及数据集划分来自项目[ChID_Baseline](https://github.com/Zce1112zslx/ChID_baseline)

## 数据集
项目原始数据集[ChID](https://github.com/chujiezheng/ChID-Dataset),并引入外部成语词典数据[chinese-xinhua](https://github.com/pwxcoo/chinese-xinhua/blob/master/data/idiom.json)

项目数据[下载链接](https://disk.pku.edu.cn:443/link/EA423797D6BC5E8CBC322F17B7DC3471)(有效期限：2024-12-31 23:59)
，需下载并解压至项目文件夹下

### 数据处理
- [x] 将数据集中待填成语数量大于1的数据都替换为多条数据
- [ ] 同义词等增广方式

### 数据使用
数据使用分为两种方式
- [ ] 成语分类：
  - 将数据集中3848条成语编号为成语词表
  - 每条数据包含：
    - `senetence_token`
    - `sentence_mask`
    - `idiom_mask`
    - `idiom_candidiate_index`
    - `label`

- [x] 成语释义： 
  - 将所有的成语与对应释义进行拼接,成为`idiom + '：' + explanation`的形式
  - 每条数据包含：
    - `senetence_token`
    - `sentence_mask`
    - `idiom_mask`
    - `idiom_candidiate_patterns_token`
    - `idiom_candidiate_pattern_mask`
    - `label`
  
## 模型
本项目基于在中文语料上预训练的[RoBERTa模型](https://github.com/ymcui/Chinese-BERT-wwm)