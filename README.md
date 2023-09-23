# 🔍📚 TextDoe: Thai Document Domain Classification Model Based on Bow, LSTM, Pre-trained Roberta-Base
This project is supported by the [AI Builder](https://ai-builders.github.io/) program. The main objective is to classify Thai documents into eight different domains, including:

- 🔮 Imaginative
- 🌱 Natural & Pure Science
- 🔬 Applied Science
- 📚 Social Science
- 🔎 History
- 💵 Commerce & Finance
- 🖌️ Arts
- 🙏 Belief & Thought


---

## 🗂️ Data Information

- Source: [TNC:Thai National Corpus](https://www.arts.chula.ac.th/ling/tnc/)
- Organization: Department of Linguistics, Faculty of Arts, Chulalongkorn University
- After data cleaning, the dataset originally consisting of 45,000 articles has been refined to 36,000 articles
  
- ### 📚 Article Sources

| Sources                                         | Proportion (%) |
| ---------------------------------------------   | :------------: |
| Physical Book                                   |      60%       |
| Journal                                         |      25%       |
| Newspaper                                       |    5-10%       |
| Other publications (e.g. advertising brochures) |    5-10%       |
| Online content                                  |      <5%       |
  
- ### 📊 Data Splitting
  
| Data Split | Proportion (%) | Volume |
| ---------- | :------------: | ------ |
| Training   |      70%       | 25,200 |
| Validation |      15%       | 5,400  |
| Testing    |      15%       | 5,400  |

