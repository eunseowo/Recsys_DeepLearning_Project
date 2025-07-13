## 🎬 딥러닝 기반 영화 추천 시스템 with AutoInt 
이 프로젝트는 MovieLens 1M 데이터를 기반으로, 딥러닝 추천 시스템 모델인 AutoInt (Automatic Feature Interaction Learning via Self-Attentive Neural Networks) 를 활용하여 영화 추천 시스템을 구현한다.

## 🛠️ 설치 방법 및 실행 환경

📂 **폴더 구조 (아래의 구조와 동일하게 폴더를 구성)**

```
Recsys_DeepLearning_Project
│
├── autointmlp.py
├── show_st.py
├── model/
│   └── autoInt_model_weights.h5
│
├── data/
│   ├── field_dims.npy
│   ├── label_encoders.pkl
│   └── ml-1m/
│       ├── movies_prepro.csv
│       ├── ratings_prepro.csv
│       └── users_prepro.csv
│
└── __pycache__/         # 자동 생성됨

```

1️⃣ **가상환경 생성 (권장: Python 3.11)**
```
conda create -n autoint python=3.11
conda activate autoint
```

2️⃣ **필수 라이브러리 설치**
```
pip install tensorflow==2.15.0
pip install streamlit numpy pandas joblib scikit-learn tqdm
```

3️⃣ **Streamlit 앱 실행**
```
streamlit run show_st.py
```

## 🤖 Streamlit을 사용한 시각화
Streamlit 앱을 통해 특정 사용자 ID를 입력하면 해당 사용자에게 맞춤화된 영화 추천 결과를 시각적으로 확인할 수 있다.

<p align="center">
  <img src="https://github.com/user-attachments/assets/07151cd4-9e86-42ad-ab37-fa3872e971f1" width="400"/>
  <br/>
  <img src="https://github.com/user-attachments/assets/dd72b1e8-98b2-4233-b5ee-514e64e62d5c" width="400"/>
</p>

## 📂 디렉토리 구조 

```
Recsys_DeepLearning_Project
│
├── autointmlp.py
├── show_st.py
├── model/
│   └── autoInt_model_weights.h5
│
├── data/
│   ├── field_dims.npy
│   ├── label_encoders.pkl
│   ├── 데이터_전처리.ipynb
│   └── ml-1m/
│       ├── movies_prepro.csv
│       ├── ratings_prepro.csv
│       └── users_prepro.csv
│
├── __pycache__/           # 자동 생성됨

```
